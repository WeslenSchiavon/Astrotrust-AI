from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "final_publication" / "bootstrap_final_250k"

TRUE_CANDIDATES = [
    "true_label", "y_true", "label", "target", "true_class", "true_class_id", "class_id"
]
PRED_CANDIDATES = [
    "predicted_label", "y_pred", "prediction", "pred_label", "predicted_class_id",
    "ensemble_predicted_label", "predicted_label_id", "yhat", "y_hat"
]
OBJECT_ID_CANDIDATES = ["object_id", "diaobjectid", "diaObjectId", "oid", "snid"]

EXCLUDE_PATH_PATTERNS = [
    "25k", "100k", "25000", "100000", "25000obj", "100000obj",
    "family", "hierarchical", "oracle", "interface_case", "case_studies",
    "error_analysis_hierarchy_topk", "test_predictions_with_families",
    "final_probability_ensemble_search",  # test-tuned exploratory ensemble, not final model
    "dataset_size_scaling", "bootstrap_statistical_analysis", "bootstrap_final_250k",
]

DEFAULT_INCLUDE_MODELS = [
    "ensemble_hybrid_dominant",
    "hybrid_temporal_tabular_cnn",
    "lightgbm_v4_baseline",
    "temporal_cnn",
    "temporal_cnn_v2",
]

# Preferred names in the final manuscript table.
PREFERRED_NAME_PATTERNS = [
    ("ensemble_hybrid_dominant", ["ensemble_hybrid_dominant", "hybrid_dominant"]),
    ("hybrid_temporal_tabular_cnn", ["hybrid_temporal_tabular_cnn_250k", "hybrid_temporal_tabular_cnn"]),
    ("lightgbm_v4_baseline", ["lightgbm_v4_baseline", "lightgbm"]),
    ("temporal_cnn", ["temporal_cnn_250k", "temporal_cnn_test"]),
    ("temporal_cnn_v2", ["temporal_cnn_v2_250k", "temporal_cnn_v2"]),
]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def sanitize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    denom = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p / denom


def should_exclude(path: Path) -> bool:
    low = str(path).lower()
    return any(pattern.lower() in low for pattern in EXCLUDE_PATH_PATTERNS)


def infer_model_name(path: Path) -> str:
    low = str(path).lower().replace("\\", "/")
    for canonical, patterns in PREFERRED_NAME_PATTERNS:
        if any(p.lower() in low for p in patterns):
            return canonical
    return path.parent.name


def find_probability_file(pred_path: Path) -> Path | None:
    folder = pred_path.parent
    candidates = sorted(folder.glob("*probabilities*.npy")) + sorted(folder.glob("*probs*.npy"))
    if not candidates:
        return None

    pred_tokens = set(re.split(r"[_\-]+", pred_path.stem.lower().replace("predictions", "")))
    scored = []
    for path in candidates:
        prob_tokens = set(re.split(r"[_\-]+", path.stem.lower().replace("probabilities", "")))
        score = len(pred_tokens & prob_tokens)
        scored.append((score, len(str(path)), path))
    scored.sort(reverse=True)
    return scored[0][2]


def load_prediction_candidate(path: Path, min_test_rows: int, max_test_rows: int) -> dict[str, Any] | None:
    if should_exclude(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    n = len(df)
    if n < min_test_rows or n > max_test_rows:
        return None

    true_col = find_column(df, TRUE_CANDIDATES)
    if true_col is None:
        return None

    pred_col = find_column(df, PRED_CANDIDATES)
    obj_col = find_column(df, OBJECT_ID_CANDIDATES)

    probs = None
    prob_path = find_probability_file(path)
    if prob_path is not None:
        try:
            p = sanitize_probabilities(np.load(prob_path))
            if len(p) == n:
                probs = p
            else:
                prob_path = None
        except Exception:
            prob_path = None

    if pred_col is None and probs is None:
        return None

    y_true = df[true_col].astype(int).to_numpy()
    if pred_col is not None:
        y_pred = df[pred_col].astype(int).to_numpy()
    else:
        y_pred = np.argmax(probs, axis=1).astype(int)
        pred_col = "argmax(probabilities)"

    model_name = infer_model_name(path)
    object_ids = df[obj_col].astype(str).to_numpy() if obj_col else np.arange(n).astype(str)

    return {
        "model": model_name,
        "prediction_path": str(path),
        "probability_path": str(prob_path) if prob_path else "",
        "n_test": n,
        "true_col": true_col,
        "pred_col": pred_col,
        "object_ids": object_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


def discover_final_250k_candidates(min_test_rows: int, max_test_rows: int) -> list[dict[str, Any]]:
    paths = sorted(RESULTS_DIR.glob("**/*predictions*.csv")) + sorted(RESULTS_DIR.glob("**/*prediction*.csv"))
    candidates = []
    seen_paths = set()

    for path in paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        cand = load_prediction_candidate(path, min_test_rows, max_test_rows)
        if cand is not None:
            candidates.append(cand)

    # Keep best candidate per canonical model. Prefer those with probabilities and larger n.
    best: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        key = cand["model"]
        score = (1 if cand["probs"] is not None else 0, cand["n_test"], -len(cand["prediction_path"]))
        if key not in best:
            best[key] = cand
        else:
            old = best[key]
            old_score = (1 if old["probs"] is not None else 0, old["n_test"], -len(old["prediction_path"]))
            if score > old_score:
                best[key] = cand

    out = list(best.values())
    out.sort(key=lambda c: c["model"])
    return out


def align_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cand in candidates:
        sig = (cand["n_test"], hash(cand["y_true"].tobytes()))
        groups.setdefault(sig, []).append(cand)

    # Choose the largest aligned group, but require final-sized test set by construction.
    best_sig = max(groups.keys(), key=lambda sig: (len(groups[sig]), sig[0]))
    aligned = groups[best_sig]

    dropped = [c for c in candidates if c not in aligned]
    if dropped:
        print("\n[INFO] Dropped candidates not aligned with the selected 250k test split:")
        for c in dropped:
            print(f"  - {c['model']} | n={c['n_test']} | {c['prediction_path']}")

    return aligned


def load_manual_candidate(name: str, pred_path: Path, prob_path: Path | None = None) -> dict[str, Any]:
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)
    true_col = find_column(df, TRUE_CANDIDATES)
    pred_col = find_column(df, PRED_CANDIDATES)
    obj_col = find_column(df, OBJECT_ID_CANDIDATES)

    if true_col is None:
        raise ValueError(f"No true-label column found in {pred_path}")

    probs = None
    if prob_path is not None and str(prob_path) != "" and prob_path.exists():
        probs = sanitize_probabilities(np.load(prob_path))
        if len(probs) != len(df):
            raise ValueError(f"Probability row mismatch for {prob_path}")

    if pred_col is None:
        if probs is None:
            raise ValueError(f"No prediction column and no probability file for {pred_path}")
        pred_col = "argmax(probabilities)"
        y_pred = np.argmax(probs, axis=1).astype(int)
    else:
        y_pred = df[pred_col].astype(int).to_numpy()

    return {
        "model": name,
        "prediction_path": str(pred_path),
        "probability_path": str(prob_path) if prob_path else "",
        "n_test": len(df),
        "true_col": true_col,
        "pred_col": pred_col,
        "object_ids": df[obj_col].astype(str).to_numpy() if obj_col else np.arange(len(df)).astype(str),
        "y_true": df[true_col].astype(int).to_numpy(),
        "y_pred": y_pred,
        "probs": probs,
    }


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray | None, k: int) -> float:
    if probs is None:
        return np.nan
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probs is not None:
        metrics["top2_accuracy"] = topk_accuracy(y_true, probs, 2)
        metrics["top3_accuracy"] = topk_accuracy(y_true, probs, 3)
        metrics["top5_accuracy"] = topk_accuracy(y_true, probs, 5)
        metrics["mean_confidence"] = float(np.max(probs, axis=1).mean())
    return metrics


def bootstrap_ci(cand: dict[str, Any], n_bootstrap: int, seed: int, ci_level: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = cand["y_true"]
    y_pred = cand["y_pred"]
    probs = cand["probs"]
    n = len(y_true)

    estimate = metric_bundle(y_true, y_pred, probs)
    values = {metric: [] for metric in estimate}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled_probs = probs[idx] if probs is not None else None
        vals = metric_bundle(y_true[idx], y_pred[idx], sampled_probs)
        for metric, value in vals.items():
            values[metric].append(value)

        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"  {cand['model']}: bootstrap {b + 1}/{n_bootstrap}", flush=True)

    alpha = (1.0 - ci_level) / 2.0
    rows = []
    for metric, estimate_value in estimate.items():
        arr = np.asarray(values[metric], dtype=float)
        rows.append({
            "model": cand["model"],
            "metric": metric,
            "estimate": estimate_value,
            "ci_lower": float(np.quantile(arr, alpha)),
            "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
            "bootstrap_std": float(np.std(arr, ddof=1)),
            "ci_level": ci_level,
            "n_test": n,
            "n_bootstrap": n_bootstrap,
            "prediction_path": cand["prediction_path"],
            "probability_path": cand["probability_path"],
        })
    return pd.DataFrame(rows)


def paired_bootstrap_difference(a: dict[str, Any], b: dict[str, Any], n_bootstrap: int, seed: int, ci_level: float) -> pd.DataFrame:
    if len(a["y_true"]) != len(b["y_true"]) or not np.array_equal(a["y_true"], b["y_true"]):
        raise ValueError(f"Candidates are not aligned: {a['model']} vs {b['model']}")

    rng = np.random.default_rng(seed)
    y_true = a["y_true"]
    n = len(y_true)

    est_a = metric_bundle(y_true, a["y_pred"], a["probs"])
    est_b = metric_bundle(y_true, b["y_pred"], b["probs"])
    common_metrics = [m for m in est_a if m in est_b]
    values = {m: [] for m in common_metrics}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals_a = metric_bundle(y_true[idx], a["y_pred"][idx], a["probs"][idx] if a["probs"] is not None else None)
        vals_b = metric_bundle(y_true[idx], b["y_pred"][idx], b["probs"][idx] if b["probs"] is not None else None)
        for m in common_metrics:
            values[m].append(vals_a[m] - vals_b[m])

    alpha = (1.0 - ci_level) / 2.0
    rows = []
    for m in common_metrics:
        arr = np.asarray(values[m], dtype=float)
        diff = est_a[m] - est_b[m]
        p_value = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
        rows.append({
            "model_a": a["model"],
            "model_b": b["model"],
            "comparison": f"{a['model']} - {b['model']}",
            "metric": m,
            "difference": float(diff),
            "ci_lower": float(np.quantile(arr, alpha)),
            "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
            "bootstrap_std": float(np.std(arr, ddof=1)),
            "approx_p_value_two_sided": float(min(max(p_value, 0.0), 1.0)),
            "ci_level": ci_level,
            "n_test": n,
            "n_bootstrap": n_bootstrap,
        })
    return pd.DataFrame(rows)


def mcnemar_test(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    y_true = a["y_true"]
    correct_a = a["y_pred"] == y_true
    correct_b = b["y_pred"] == y_true

    b_count = int(np.sum(correct_a & ~correct_b))
    c_count = int(np.sum(~correct_a & correct_b))

    if b_count + c_count == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b_count - c_count) - 1.0) ** 2 / (b_count + c_count)
        p_value = math.erfc(math.sqrt(chi2 / 2.0))

    return {
        "model_a": a["model"],
        "model_b": b["model"],
        "n_a_correct_b_wrong": b_count,
        "n_a_wrong_b_correct": c_count,
        "mcnemar_chi2_continuity_corrected": float(chi2),
        "mcnemar_p_value_approx": float(p_value),
    }


def per_class_sensitivity(cand: dict[str, Any]) -> pd.DataFrame:
    y_true = cand["y_true"]
    y_pred = cand["y_pred"]
    probs = cand["probs"]
    rows = []

    for label in sorted(np.unique(y_true).astype(int).tolist()):
        mask = y_true == label
        n = int(mask.sum())
        row = {
            "model": cand["model"],
            "class_label": label,
            "n_test": n,
            "recall_top1": float(np.mean(y_pred[mask] == y_true[mask])),
            "error_rate_top1": float(1.0 - np.mean(y_pred[mask] == y_true[mask])),
        }
        if probs is not None:
            row["recall_top3"] = topk_accuracy(y_true[mask], probs[mask], 3)
            row["recall_top5"] = topk_accuracy(y_true[mask], probs[mask], 5)
            row["mean_true_class_probability"] = float(probs[mask, label].mean()) if label < probs.shape[1] else np.nan
            row["mean_confidence"] = float(np.max(probs[mask], axis=1).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def format_ci_table(ci_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "top5_accuracy"]
    rows = []
    for model, group in ci_df.groupby("model"):
        row = {"model": model}
        for metric in metrics:
            r = group[group["metric"] == metric]
            if r.empty:
                row[metric] = "--"
            else:
                rr = r.iloc[0]
                row[metric] = f"{rr['estimate']:.4f} [{rr['ci_lower']:.4f}, {rr['ci_upper']:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(output_dir: Path, point_df: pd.DataFrame, ci_df: pd.DataFrame, diff_df: pd.DataFrame, mcnemar_df: pd.DataFrame, per_class_df: pd.DataFrame):
    ci_table = format_ci_table(ci_df)
    md = "# Final 250k bootstrap statistical analysis\n\n"
    md += "This report includes only final-sized 250k test-split models. Small 25k/100k exploratory runs, oracle models, hierarchical analyses, family-level models, and test-tuned exploratory ensembles are excluded.\n\n"
    md += "## Point estimates\n\n"
    md += point_df.to_markdown(index=False)
    md += "\n\n## Manuscript-ready 95% bootstrap confidence interval table\n\n"
    md += ci_table.to_markdown(index=False)
    md += "\n\n## Paired bootstrap differences\n\n"
    md += diff_df.to_markdown(index=False) if not diff_df.empty else "_No paired differences computed._"
    md += "\n\n## McNemar tests\n\n"
    md += mcnemar_df.to_markdown(index=False) if not mcnemar_df.empty else "_No McNemar tests computed._"
    md += "\n\n## Per-class sensitivity for the best available final model\n\n"
    md += per_class_df.to_markdown(index=False) if not per_class_df.empty else "_No per-class table generated._"
    md += "\n\n## Suggested interpretation\n\n"
    md += "Use the paired bootstrap table to decide whether the difference between the final ensemble and the calibrated hybrid model is statistically robust. If the 95% CI of the difference includes zero, describe the ensemble gain as marginal rather than decisive. If it excludes zero, the gain can be reported as statistically supported under bootstrap resampling.\n"
    (output_dir / "bootstrap_final_250k_summary.md").write_text(md, encoding="utf-8")


def parse_manual_model_arg(value: str) -> tuple[str, Path, Path | None]:
    # Format: name|prediction_csv|probability_npy(optional)
    parts = value.split("|")
    if len(parts) < 2:
        raise ValueError("Manual model must be formatted as: name|prediction_csv|probability_npy(optional)")
    name = parts[0]
    pred = Path(parts[1])
    prob = Path(parts[2]) if len(parts) >= 3 and parts[2].strip() else None
    return name, pred, prob


def main():
    parser = argparse.ArgumentParser(description="Clean final 250k bootstrap confidence intervals and paired model tests for AstroTrust-AI.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-test-rows", type=int, default=50000)
    parser.add_argument("--max-test-rows", type=int, default=80000)
    parser.add_argument(
        "--include-models",
        type=str,
        default=",".join(DEFAULT_INCLUDE_MODELS),
        help="Comma-separated canonical model names to keep in the final 250k bootstrap analysis.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only list final-sized candidates and stop.")
    parser.add_argument(
        "--manual-model",
        action="append",
        default=[],
        help="Add/override a model as name|prediction_csv|probability_npy(optional). Can be used multiple times.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_final_250k_candidates(args.min_test_rows, args.max_test_rows)
    include_models = {m.strip() for m in args.include_models.split(",") if m.strip()}

    if include_models:
        candidates = [c for c in candidates if c["model"] in include_models]

    # Add manual candidates, overriding same model name.
    manual = []
    for item in args.manual_model:
        name, pred, prob = parse_manual_model_arg(item)
        manual.append(load_manual_candidate(name, pred, prob))

    by_model = {c["model"]: c for c in candidates}
    for c in manual:
        by_model[c["model"]] = c
    candidates = list(by_model.values())

    print("\nDiscovered final-sized candidates before alignment:")
    if not candidates:
        print("  None found. Use --manual-model to provide explicit prediction/probability files.")
    for c in sorted(candidates, key=lambda x: x["model"]):
        print(f"  - {c['model']} | n={c['n_test']} | probs={'yes' if c['probs'] is not None else 'no'}")
        print(f"    pred: {c['prediction_path']}")
        if c["probability_path"]:
            print(f"    prob: {c['probability_path']}")

    if args.list_only:
        return

    candidates = align_candidates(candidates)
    if not candidates:
        raise RuntimeError("No aligned final 250k candidates available.")

    print("\nAligned final 250k candidates:")
    for c in sorted(candidates, key=lambda x: x["model"]):
        print(f"  - {c['model']} | n={c['n_test']} | probs={'yes' if c['probs'] is not None else 'no'}")

    # Point estimates.
    point_rows = []
    for c in candidates:
        point_rows.append({"model": c["model"], "n_test": c["n_test"], **metric_bundle(c["y_true"], c["y_pred"], c["probs"])})
    point_df = pd.DataFrame(point_rows).sort_values("accuracy", ascending=False)
    point_df.to_csv(args.output_dir / "model_point_estimates_final_250k.csv", index=False)

    # Bootstrap CIs.
    ci_frames = []
    print("\nRunning bootstrap CIs...")
    for i, c in enumerate(candidates):
        ci_frames.append(bootstrap_ci(c, args.n_bootstrap, args.seed + i * 101, args.ci))
    ci_df = pd.concat(ci_frames, ignore_index=True)
    ci_df.to_csv(args.output_dir / "bootstrap_metric_confidence_intervals_final_250k.csv", index=False)
    format_ci_table(ci_df).to_csv(args.output_dir / "table_bootstrap_ci_final_250k_manuscript.csv", index=False)

    # Paired comparisons.
    by_model = {c["model"]: c for c in candidates}
    desired_pairs = []
    if "ensemble_hybrid_dominant" in by_model and "hybrid_temporal_tabular_cnn" in by_model:
        desired_pairs.append((by_model["ensemble_hybrid_dominant"], by_model["hybrid_temporal_tabular_cnn"]))
    if "hybrid_temporal_tabular_cnn" in by_model and "lightgbm_v4_baseline" in by_model:
        desired_pairs.append((by_model["hybrid_temporal_tabular_cnn"], by_model["lightgbm_v4_baseline"]))
    if "ensemble_hybrid_dominant" in by_model and "lightgbm_v4_baseline" in by_model:
        desired_pairs.append((by_model["ensemble_hybrid_dominant"], by_model["lightgbm_v4_baseline"]))

    # If preferred pairs are unavailable, compare best model against all others.
    if not desired_pairs and len(candidates) >= 2:
        best_name = point_df.iloc[0]["model"]
        best = by_model[best_name]
        desired_pairs = [(best, c) for c in candidates if c["model"] != best_name]

    diff_frames = []
    mcnemar_rows = []
    for i, (a, b) in enumerate(desired_pairs):
        diff_frames.append(paired_bootstrap_difference(a, b, args.n_bootstrap, args.seed + 999 + i * 17, args.ci))
        mcnemar_rows.append(mcnemar_test(a, b))
    diff_df = pd.concat(diff_frames, ignore_index=True) if diff_frames else pd.DataFrame()
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    diff_df.to_csv(args.output_dir / "paired_bootstrap_model_differences_final_250k.csv", index=False)
    mcnemar_df.to_csv(args.output_dir / "mcnemar_tests_final_250k.csv", index=False)

    # Per-class sensitivity for best available final model.
    best_model = point_df.iloc[0]["model"]
    per_class_df = per_class_sensitivity(by_model[best_model])
    per_class_df.to_csv(args.output_dir / "per_class_sensitivity_best_final_250k.csv", index=False)

    write_markdown(args.output_dir, point_df, ci_df, diff_df, mcnemar_df, per_class_df)

    print("\nPoint estimates:")
    print(point_df.to_string(index=False))

    print("\nManuscript-ready CI table:")
    print(format_ci_table(ci_df).to_string(index=False))

    print("\nPaired bootstrap differences:")
    print(diff_df.to_string(index=False) if not diff_df.empty else "No paired differences computed.")

    print("\nMcNemar tests:")
    print(mcnemar_df.to_string(index=False) if not mcnemar_df.empty else "No McNemar tests computed.")

    print("\nSaved:")
    print(f"- {args.output_dir / 'model_point_estimates_final_250k.csv'}")
    print(f"- {args.output_dir / 'bootstrap_metric_confidence_intervals_final_250k.csv'}")
    print(f"- {args.output_dir / 'table_bootstrap_ci_final_250k_manuscript.csv'}")
    print(f"- {args.output_dir / 'paired_bootstrap_model_differences_final_250k.csv'}")
    print(f"- {args.output_dir / 'mcnemar_tests_final_250k.csv'}")
    print(f"- {args.output_dir / 'per_class_sensitivity_best_final_250k.csv'}")
    print(f"- {args.output_dir / 'bootstrap_final_250k_summary.md'}")


if __name__ == "__main__":
    main()
