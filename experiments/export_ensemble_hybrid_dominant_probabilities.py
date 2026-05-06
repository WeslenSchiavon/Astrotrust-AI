from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_ENSEMBLE_DIR = RESULTS_DIR / "hybrid_tabular_ensemble_250k"
DEFAULT_ENSEMBLE_PREDICTIONS = DEFAULT_ENSEMBLE_DIR / "ensemble_hybrid_dominant_predictions.csv"
DEFAULT_OUTPUT_PROBABILITIES = DEFAULT_ENSEMBLE_DIR / "ensemble_hybrid_dominant_test_probabilities.npy"
DEFAULT_OUTPUT_METRICS = DEFAULT_ENSEMBLE_DIR / "ensemble_hybrid_dominant_probability_metrics.csv"
DEFAULT_OUTPUT_REPORT = DEFAULT_ENSEMBLE_DIR / "ensemble_hybrid_dominant_probability_export_report.md"

TRUE_CANDIDATES = [
    "true_label",
    "y_true",
    "label",
    "target",
    "true_class",
    "true_class_id",
    "class_id",
]
PRED_CANDIDATES = [
    "predicted_label",
    "y_pred",
    "prediction",
    "pred_label",
    "predicted_class_id",
    "ensemble_predicted_label",
    "yhat",
    "y_hat",
]

PROBABILITY_PREFIXES = [
    "prob_",
    "proba_",
    "p_",
    "class_prob_",
    "prob_class_",
]

DEFAULT_BASE_PROBABILITY_CANDIDATES = {
    "hybrid_temporal_tabular_cnn": RESULTS_DIR
    / "hybrid_temporal_tabular_cnn_250k"
    / "hybrid_temporal_tabular_cnn_test_probabilities.npy",
    "temporal_cnn": RESULTS_DIR
    / "temporal_cnn_250k"
    / "temporal_cnn_test_probabilities.npy",
    "temporal_cnn_v2": RESULTS_DIR
    / "temporal_cnn_v2_250k"
    / "temporal_cnn_v2_test_probabilities.npy",
}

DEFAULT_BASE_PREDICTION_CANDIDATES = {
    "hybrid_temporal_tabular_cnn": RESULTS_DIR
    / "hybrid_temporal_tabular_cnn_250k"
    / "hybrid_temporal_tabular_cnn_test_predictions.csv",
    "temporal_cnn": RESULTS_DIR
    / "temporal_cnn_250k"
    / "temporal_cnn_test_predictions.csv",
    "temporal_cnn_v2": RESULTS_DIR
    / "temporal_cnn_v2_250k"
    / "temporal_cnn_v2_test_predictions.csv",
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def sanitize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    topk = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def metric_bundle(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = sanitize_probabilities(probabilities)
    y_pred = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "top2_accuracy": topk_accuracy(y_true, probabilities, 2),
        "top3_accuracy": topk_accuracy(y_true, probabilities, 3),
        "top5_accuracy": topk_accuracy(y_true, probabilities, 5),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def detect_probability_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        low = str(col).lower()
        if any(low.startswith(prefix) for prefix in PROBABILITY_PREFIXES):
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)

    def class_index(col: str):
        numbers = re.findall(r"\d+", str(col))
        return int(numbers[-1]) if numbers else 10**9

    cols = sorted(cols, key=class_index)
    return cols


def export_from_probability_columns(predictions_path: Path) -> tuple[np.ndarray, str] | None:
    df = pd.read_csv(predictions_path)
    prob_cols = detect_probability_columns(df)
    if len(prob_cols) < 2:
        return None
    probabilities = sanitize_probabilities(df[prob_cols].to_numpy(dtype=float))
    return probabilities, f"extracted from probability columns in {predictions_path.name}: {prob_cols[:5]}..."


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_weight_metadata(ensemble_dir: Path) -> dict | None:
    candidates = []
    for pattern in ["*weight*.json", "*ensemble*.json", "*metadata*.json", "*summary*.json"]:
        candidates.extend(sorted(ensemble_dir.glob(pattern)))

    for path in candidates:
        meta = load_json_if_exists(path)
        if not meta:
            continue

        # Common structures.
        for key in ["weights", "model_weights", "ensemble_weights", "best_weights"]:
            if key in meta and isinstance(meta[key], dict):
                return {"path": str(path), "weights": meta[key]}

        # Sometimes best config is nested.
        for key in ["best", "best_config", "best_model", "ensemble_hybrid_dominant"]:
            if key in meta and isinstance(meta[key], dict):
                nested = meta[key]
                for wkey in ["weights", "model_weights", "ensemble_weights", "best_weights"]:
                    if wkey in nested and isinstance(nested[wkey], dict):
                        return {"path": str(path), "weights": nested[wkey]}

    # Try CSV files with model/weight columns.
    for path in sorted(ensemble_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        model_col = cols.get("model") or cols.get("model_name") or cols.get("base_model")
        weight_col = cols.get("weight") or cols.get("ensemble_weight") or cols.get("best_weight")
        if model_col and weight_col:
            weights = dict(zip(df[model_col].astype(str), df[weight_col].astype(float)))
            if weights:
                return {"path": str(path), "weights": weights}

    return None


def canonicalize_weight_name(name: str) -> str | None:
    low = str(name).lower()
    if "hybrid" in low and "temporal" in low:
        return "hybrid_temporal_tabular_cnn"
    if "temporal_cnn_v2" in low or "cnn_v2" in low:
        return "temporal_cnn_v2"
    if "temporal_cnn" in low or low == "cnn":
        return "temporal_cnn"
    return None


def reconstruct_from_base_probabilities(weights: dict[str, float]) -> tuple[np.ndarray, str]:
    mapped_weights: dict[str, float] = {}
    for name, weight in weights.items():
        canonical = canonicalize_weight_name(name)
        if canonical is None:
            # Ignore models without probability files. This means reconstruction may not be exact.
            continue
        mapped_weights[canonical] = mapped_weights.get(canonical, 0.0) + float(weight)

    if not mapped_weights:
        raise RuntimeError(
            "No usable model weights matched available probability files. "
            "Available probability models: " + ", ".join(DEFAULT_BASE_PROBABILITY_CANDIDATES.keys())
        )

    arrays = []
    used = []
    for model_name, weight in mapped_weights.items():
        path = DEFAULT_BASE_PROBABILITY_CANDIDATES[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Missing base probability file for {model_name}: {path}")
        p = sanitize_probabilities(np.load(path))
        arrays.append((float(weight), p))
        used.append(f"{model_name}={float(weight):.6f}")

    n = len(arrays[0][1])
    k = arrays[0][1].shape[1]
    for _, p in arrays:
        if p.shape != (n, k):
            raise ValueError("Base probability arrays are not aligned by shape.")

    total_weight = sum(w for w, _ in arrays)
    probs = np.zeros_like(arrays[0][1], dtype=float)
    for w, p in arrays:
        probs += (w / total_weight) * p

    return sanitize_probabilities(probs), "reconstructed from base probabilities using weights: " + ", ".join(used)


def compare_with_ensemble_predictions(predictions_path: Path, probabilities: np.ndarray) -> dict:
    df = pd.read_csv(predictions_path)
    true_col = find_column(df, TRUE_CANDIDATES)
    pred_col = find_column(df, PRED_CANDIDATES)

    y_true = df[true_col].astype(int).to_numpy() if true_col else None
    stored_pred = df[pred_col].astype(int).to_numpy() if pred_col else None
    prob_pred = probabilities.argmax(axis=1)

    out = {}
    if y_true is not None:
        out.update(metric_bundle(y_true, probabilities))
    if stored_pred is not None:
        out["agreement_with_stored_ensemble_predictions"] = float(np.mean(stored_pred == prob_pred))
        out["n_disagreements_with_stored_predictions"] = int(np.sum(stored_pred != prob_pred))
    return out


def main():
    parser = argparse.ArgumentParser(description="Export probabilities for ensemble_hybrid_dominant if they can be recovered or reconstructed.")
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--ensemble-predictions", type=Path, default=DEFAULT_ENSEMBLE_PREDICTIONS)
    parser.add_argument("--output-probabilities", type=Path, default=DEFAULT_OUTPUT_PROBABILITIES)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_OUTPUT_METRICS)
    parser.add_argument(
        "--weights-json",
        type=str,
        default="",
        help="Optional explicit JSON weights, e.g. '{\"hybrid_temporal_tabular_cnn\":0.8,\"temporal_cnn\":0.2}'.",
    )
    args = parser.parse_args()

    if not args.ensemble_predictions.exists():
        raise FileNotFoundError(args.ensemble_predictions)

    args.output_probabilities.parent.mkdir(parents=True, exist_ok=True)

    method = None
    probabilities = None

    # 1. Best case: probabilities already exist as columns in the ensemble predictions CSV.
    extracted = export_from_probability_columns(args.ensemble_predictions)
    if extracted is not None:
        probabilities, method = extracted
    else:
        # 2. Reconstruct from known base probabilities and ensemble weights.
        if args.weights_json.strip():
            weights = json.loads(args.weights_json)
            weight_source = "explicit --weights-json"
        else:
            metadata = find_weight_metadata(args.ensemble_dir)
            if metadata is None:
                raise RuntimeError(
                    "Could not find probability columns in the ensemble predictions file and could not find ensemble weight metadata.\n"
                    "To reconstruct probabilities, rerun this script with --weights-json, or modify the original ensemble script to save the probability matrix."
                )
            weights = metadata["weights"]
            weight_source = metadata["path"]

        probabilities, method = reconstruct_from_base_probabilities(weights)
        method += f"; weight source: {weight_source}"

    np.save(args.output_probabilities, probabilities)

    metrics = compare_with_ensemble_predictions(args.ensemble_predictions, probabilities)
    metrics["method"] = method
    metrics["output_probabilities"] = str(args.output_probabilities)
    pd.DataFrame([metrics]).to_csv(args.output_metrics, index=False)

    md = "# Ensemble probability export report\n\n"
    md += f"Method: `{method}`\n\n"
    md += f"Saved probabilities: `{args.output_probabilities}`\n\n"
    md += "## Metrics / agreement check\n\n"
    md += pd.DataFrame([metrics]).to_markdown(index=False)
    md += "\n\n"
    if metrics.get("agreement_with_stored_ensemble_predictions", 1.0) < 0.999:
        md += (
            "**Warning:** reconstructed probabilities do not exactly reproduce the stored ensemble predictions. "
            "Use them only as an approximation unless the original ensemble weight/probability generation is confirmed.\n"
        )
    args.output_report.write_text(md, encoding="utf-8")

    print("[OK] Ensemble probabilities exported.")
    print(f"Method: {method}")
    print(f"Saved: {args.output_probabilities}")
    print("\nMetrics / agreement:")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
