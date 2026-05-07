from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_TEMPERATURE_JSON = RESULTS_DIR / "hybrid_inference_artifacts_250k" / "temperature_scaling.json"
DEFAULT_PREDICTIONS = RESULTS_DIR / "hybrid_temporal_tabular_cnn_250k" / "hybrid_temporal_tabular_cnn_test_predictions.csv"
DEFAULT_PROBABILITIES = RESULTS_DIR / "hybrid_temporal_tabular_cnn_250k" / "hybrid_temporal_tabular_cnn_test_probabilities.npy"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "final_publication" / "temperature_scaled_hybrid_250k"

TRUE_CANDIDATES = ["true_label", "y_true", "label", "target", "true_class", "true_class_id", "class_id"]
OBJECT_ID_CANDIDATES = ["object_id", "diaobjectid", "diaObjectId", "oid", "snid"]


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
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.maximum(exp_z.sum(axis=1, keepdims=True), 1e-12)


def apply_temperature_to_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    p = sanitize_probabilities(probabilities)
    logits = np.log(np.clip(p, 1e-12, 1.0))
    return sanitize_probabilities(softmax(logits / temperature))


def find_temperature_in_object(obj: Any) -> float | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in ["temperature", "temp", "t", "best_temperature", "optimal_temperature"]:
                try:
                    value_f = float(value)
                    if value_f > 0:
                        return value_f
                except Exception:
                    pass
        for value in obj.values():
            found = find_temperature_in_object(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_temperature_in_object(value)
            if found is not None:
                return found
    return None


def load_temperature(path: Path, explicit_temperature: float | None = None) -> tuple[float, str]:
    if explicit_temperature is not None:
        return float(explicit_temperature), "explicit --temperature"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    temperature = find_temperature_in_object(data)
    if temperature is None:
        raise ValueError(
            f"Could not find a positive temperature value in {path}. "
            "Use --temperature to pass it explicitly."
        )
    return float(temperature), str(path)


def one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_true), n_classes), dtype=np.float64)
    valid = (y_true >= 0) & (y_true < n_classes)
    y[np.arange(len(y_true))[valid], y_true[valid]] = 1.0
    return y


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = one_hot(y_true.astype(int), probs.shape[1])
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    p_true = probs[np.arange(len(y_true)), y_true.astype(int)]
    return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "top2_accuracy": topk_accuracy(y_true, probs, 2),
        "top3_accuracy": topk_accuracy(y_true, probs, 3),
        "top5_accuracy": topk_accuracy(y_true, probs, 5),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "ece": ece(y_true, probs, n_bins=n_bins),
        "brier_score": brier_score_multiclass(y_true, probs),
        "negative_log_likelihood": nll(y_true, probs),
    }


def main():
    parser = argparse.ArgumentParser(description="Apply validation-fitted temperature scaling to final hybrid 250k test probabilities.")
    parser.add_argument("--temperature-json", type=Path, default=DEFAULT_TEMPERATURE_JSON)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    temperature, temperature_source = load_temperature(args.temperature_json, args.temperature)

    df = pd.read_csv(args.predictions)
    true_col = find_column(df, TRUE_CANDIDATES)
    obj_col = find_column(df, OBJECT_ID_CANDIDATES)
    if true_col is None:
        raise ValueError(f"No true-label column found in {args.predictions}")

    y_true = df[true_col].astype(int).to_numpy()
    object_ids = df[obj_col].to_numpy() if obj_col else np.arange(len(df))

    raw_probs = sanitize_probabilities(np.load(args.probabilities))
    if len(raw_probs) != len(df):
        raise ValueError(f"Probability row mismatch: {len(raw_probs)} vs {len(df)}")

    temp_probs = apply_temperature_to_probabilities(raw_probs, temperature)

    raw_metrics = {"model": "hybrid_temporal_tabular_cnn_raw", **metrics(y_true, raw_probs, args.n_bins)}
    temp_metrics = {"model": "hybrid_temporal_tabular_cnn_temperature_scaled", **metrics(y_true, temp_probs, args.n_bins)}

    metrics_df = pd.DataFrame([raw_metrics, temp_metrics])
    metrics_df["temperature"] = temperature
    metrics_df["temperature_source"] = temperature_source

    out_prob = args.output_dir / "hybrid_temporal_tabular_cnn_temperature_scaled_test_probabilities.npy"
    out_pred = args.output_dir / "hybrid_temporal_tabular_cnn_temperature_scaled_test_predictions.csv"
    out_metrics = args.output_dir / "temperature_scaled_hybrid_metrics.csv"
    out_report = args.output_dir / "temperature_scaled_hybrid_report.md"

    np.save(out_prob, temp_probs.astype(np.float32))

    pred = temp_probs.argmax(axis=1)
    pd.DataFrame({
        "object_id": object_ids,
        "true_label": y_true,
        "predicted_label": pred,
        "correct": pred == y_true,
        "confidence": temp_probs.max(axis=1),
    }).to_csv(out_pred, index=False)

    metrics_df.to_csv(out_metrics, index=False)

    md = "# Temperature-scaled hybrid temporal-tabular CNN\n\n"
    md += f"Temperature: `{temperature}`\n\n"
    md += f"Temperature source: `{temperature_source}`\n\n"
    md += "## Raw versus temperature-scaled test metrics\n\n"
    md += metrics_df.to_markdown(index=False)
    md += "\n\n"
    md += "The temperature was applied to the test probabilities without refitting on the test set. The source temperature should have been fitted on the validation set.\n"
    out_report.write_text(md, encoding="utf-8")

    print("[OK] Temperature-scaled probabilities saved.")
    print(f"Temperature: {temperature}")
    print(f"Source: {temperature_source}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nSaved:")
    print(f"- {out_prob}")
    print(f"- {out_pred}")
    print(f"- {out_metrics}")
    print(f"- {out_report}")


if __name__ == "__main__":
    main()
