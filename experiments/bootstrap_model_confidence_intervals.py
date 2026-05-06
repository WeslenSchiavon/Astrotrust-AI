from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "bootstrap_statistical_analysis"

TRUE_CANDIDATES = [
    "true_label", "y_true", "label", "target", "true_class", "true_class_id", "class_id",
]
PRED_CANDIDATES = [
    "predicted_label", "y_pred", "prediction", "pred_label", "predicted_class",
    "predicted_class_id", "ensemble_predicted_label", "single_pred_label", "hier_pred_label",
]
OBJECT_ID_CANDIDATES = ["object_id", "diaobjectid", "oid", "snid"]

# Files that should not enter the 32-class bootstrap comparison.
EXCLUDE_PATTERNS = [
    "family_test_predictions",
    "family_test_probabilities",
    "hierarchical_lightgbm",
    "hierarchical_upper_bound",
    "hierarchical_probability_reranking",
    "final_probability_ensemble_search_250k",  # exploratory/test-tuned ensemble, not final unbiased result
    "interface_case_studies",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def sanitize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    denom = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p / denom


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    if probs is None:
        return np.nan
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probs is not None:
        out["top2_accuracy"] = topk_accuracy(y_true, probs, 2)
        out["top3_accuracy"] = topk_accuracy(y_true, probs, 3)
        out["top5_accuracy"] = topk_accuracy(y_true, probs, 5)
        out["mean_confidence"] = float(np.max(probs, axis=1).mean())
    return out


def infer_model_name(path: Path) -> str:
    folder = path.parent.name
    stem = path.stem

    name = folder
    low = str(path).lower()

    if "ensemble_hybrid_dominant" in low:
        name = "ensemble_hybrid_dominant"
    elif "hybrid_temporal_tabular_cnn_250k" in low:
        name = "hybrid_temporal_tabular_cnn_250k"
    elif "hybrid_temporal_tabular_cnn_100k" in low:
        name = "hybrid_temporal_tabular_cnn_100k"
    elif "hybrid_temporal_tabular_cnn_25k" in low:
        name = "hybrid_temporal_tabular_cnn_25k"
    elif "temporal_cnn_v2" in low:
        name = "temporal_cnn_v2_250k"
    elif "temporal_cnn_250k" in low:
        name = "temporal_cnn_250k"
    elif "lightgbm" in low:
        name = folder

    return name


def should_exclude(path: Path) -> bool:
    low = str(path).lower()
    return any(pattern.lower() in low for pattern in EXCLUDE_PATTERNS)


def find_probability_file(pred_path: Path) -> Path | None:
    folder = pred_path.parent
    candidates = sorted(folder.glob("*probabilities*.npy"))
    if not candidates:
        return None

    pred_low = pred_path.stem.lower().replace("predictions", "").replace("prediction", "")
    scored = []
    for p in candidates:
        prob_low = p.stem.lower().replace("probabilities", "").replace("probability", "")
        common = len(set(pred_low.split("_")) & set(prob_low.split("_")))
        scored.append((common, p))
    scored.sort(reverse=True)
    return scored[0][1]


def load_candidate(pred_path: Path, name_override: str | None = None) -> dict[str, Any] | None:
    if should_exclude(pred_path):
        return None

    try:
        df = pd.read_csv(pred_path)
    except Exception as exc:
        print(f"[SKIP] Could not read {pred_path}: {exc}")
        return None

    if df.empty:
        return None

    true_col = find_column(df, TRUE_CANDIDATES)
    pred_col = find_column(df, PRED_CANDIDATES)
    obj_col = find_column(df, OBJECT_ID_CANDIDATES)

    if true_col is None:
        return None

    probs = None
    prob_path = find_probability_file(pred_path)
    if prob_path is not None:
        try:
            probs = sanitize_probabilities(np.load(prob_path))
            if len(probs) != len(df):
                print(f"[WARN] Probability row mismatch for {pred_path}. Ignoring probabilities.")
                probs = None
                prob_path = None
        except Exception as exc:
            print(f"[WARN] Could not load probabilities for {pred_path}: {exc}")
            probs = None
            prob_path = None

    if pred_col is not None:
        y_pred = df[pred_col].astype(int).to_numpy()
    elif probs is not None:
        y_pred = np.argmax(probs, axis=1).astype(int)
        pred_col = "argmax(probabilities)"
    else:
        print(f"[SKIP] No prediction column or probabilities for {pred_path}")
        return None

    y_true = df[true_col].astype(int).to_numpy()
    object_ids = df[obj_col].astype(str).to_numpy() if obj_col is not None else np.arange(len(df)).astype(str)

    model_name = name_override or infer_model_name(pred_path)

    return {
        "model_name": model_name,
        "prediction_path": str(pred_path),
        "probability_path": str(prob_path) if prob_path is not None else None,
        "true_col": true_col,
        "pred_col": pred_col,
        "object_ids": object_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


# -----------------------------------------------------------------------------
# Discovery and alignment
# -----------------------------------------------------------------------------

def discover_candidates() -> list[dict[str, Any]]:
    paths = sorted(RESULTS_DIR.glob("**/*predictions*.csv"))
    candidates = []

    for path in paths:
        cand = load_candidate(path)
        if cand is not None:
            candidates.append(cand)

    # De-duplicate by model name: keep the largest test-like file for each name.
    best: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        name = cand["model_name"]
        if name not in best or len(cand["y_true"]) > len(best[name]["y_true"]):
            best[name] = cand

    return list(best.values())


def align_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    # Most important: keep only candidates with the same y_true vector and same length.
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cand in candidates:
        sig = (len(cand["y_true"]), hash(cand["y_true"].tobytes()))
        groups.setdefault(sig, []).append(cand)

    best_sig = max(groups, key=lambda k: len(groups[k]))
    aligned = groups[best_sig]

    dropped = [c for c in candidates if c not in aligned]
    if dropped:
        print("\n[INFO] Dropped non-aligned candidates:")
        for c in dropped:
            print(f"  - {c['model_name']} | n={len(c['y_true'])} | {c['prediction_path']}")

    return aligned


# -----------------------------------------------------------------------------
# Bootstrap and paired tests
# -----------------------------------------------------------------------------

def bootstrap_metric_ci(
    cand: dict[str, Any],
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = cand["y_true"]
    y_pred = cand["y_pred"]
    probs = cand["probs"]
    n = len(y_true)

    full = metric_bundle(y_true, y_pred, probs)
    metric_names = list(full.keys())
    boot_values = {m: [] for m in metric_names}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled_probs = probs[idx] if probs is not None else None
        vals = metric_bundle(y_true[idx], y_pred[idx], sampled_probs)
        for m in metric_names:
            boot_values[m].append(vals[m])

        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"  {cand['model_name']}: bootstrap {b + 1}/{n_bootstrap}", flush=True)

    alpha = (1.0 - ci) / 2.0
    rows = []
    for m in metric_names:
        arr = np.asarray(boot_values[m], dtype=float)
        rows.append(
            {
                "model": cand["model_name"],
                "metric": m,
                "estimate": full[m],
                "ci_level": ci,
                "ci_lower": float(np.quantile(arr, alpha)),
                "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
                "bootstrap_std": float(np.std(arr, ddof=1)),
                "n_test": n,
                "n_bootstrap": n_bootstrap,
                "prediction_path": cand["prediction_path"],
                "probability_path": cand["probability_path"],
            }
        )

    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    cand_a: dict[str, Any],
    cand_b: dict[str, Any],
    n_bootstrap: int,
    seed: int,
    ci: float,
) -> pd.DataFrame:
    if len(cand_a["y_true"]) != len(cand_b["y_true"]) or not np.array_equal(cand_a["y_true"], cand_b["y_true"]):
        raise ValueError("Candidates are not aligned for paired bootstrap.")

    rng = np.random.default_rng(seed)
    y_true = cand_a["y_true"]
    n = len(y_true)

    probs_a = cand_a["probs"]
    probs_b = cand_b["probs"]

    full_a = metric_bundle(y_true, cand_a["y_pred"], probs_a)
    full_b = metric_bundle(y_true, cand_b["y_pred"], probs_b)
    common_metrics = sorted(set(full_a.keys()) & set(full_b.keys()))

    diffs = {m: [] for m in common_metrics}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals_a = metric_bundle(y_true[idx], cand_a["y_pred"][idx], probs_a[idx] if probs_a is not None else None)
        vals_b = metric_bundle(y_true[idx], cand_b["y_pred"][idx], probs_b[idx] if probs_b is not None else None)
        for m in common_metrics:
            diffs[m].append(vals_a[m] - vals_b[m])

    alpha = (1.0 - ci) / 2.0
    rows = []
    for m in common_metrics:
        arr = np.asarray(diffs[m], dtype=float)
        estimate = full_a[m] - full_b[m]

        # Two-sided approximate bootstrap p-value for H0: difference = 0.
        p_lower = np.mean(arr <= 0.0)
        p_upper = np.mean(arr >= 0.0)
        p_value = float(2 * min(p_lower, p_upper))
        p_value = min(max(p_value, 0.0), 1.0)

        rows.append(
            {
                "model_a": cand_a["model_name"],
                "model_b": cand_b["model_name"],
                "metric": m,
                "difference_a_minus_b": float(estimate),
                "ci_level": ci,
                "ci_lower": float(np.quantile(arr, alpha)),
                "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
                "bootstrap_std": float(np.std(arr, ddof=1)),
                "approx_p_value_two_sided": p_value,
                "n_test": n,
                "n_bootstrap": n_bootstrap,
            }
        )

    return pd.DataFrame(rows)


def mcnemar_test(cand_a: dict[str, Any], cand_b: dict[str, Any]) -> dict[str, Any]:
    """McNemar test for paired top-1 correctness.

    Uses continuity-corrected chi-square approximation.
    Avoids scipy dependency.
    """
    y_true = cand_a["y_true"]
    correct_a = cand_a["y_pred"] == y_true
    correct_b = cand_b["y_pred"] == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        chi2 = 0.0
        p_approx = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # For df=1 chi-square, survival function = erfc(sqrt(x/2)).
        import math
        p_approx = math.erfc(math.sqrt(chi2 / 2.0))

    return {
        "model_a": cand_a["model_name"],
        "model_b": cand_b["model_name"],
        "n_a_correct_b_wrong": b,
        "n_a_wrong_b_correct": c,
        "mcnemar_chi2_continuity_corrected": float(chi2),
        "mcnemar_p_value_approx": float(p_approx),
    }


# -----------------------------------------------------------------------------
# Per-class sensitivity
# -----------------------------------------------------------------------------

def per_class_sensitivity(cand: dict[str, Any]) -> pd.DataFrame:
    y_true = cand["y_true"]
    y_pred = cand["y_pred"]
    probs = cand["probs"]
    labels = sorted(np.unique(y_true).astype(int).tolist())
    rows = []

    for label in labels:
        mask = y_true == label
        n = int(mask.sum())
        if n == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        recall = float(np.mean(yp == yt))

        row = {
            "model": cand["model_name"],
            "class_label": label,
            "n_test": n,
            "recall": recall,
            "error_rate": 1.0 - recall,
        }

        if probs is not None:
            row["top3_recall"] = topk_accuracy(yt, probs[mask], 3)
            row["top5_recall"] = topk_accuracy(yt, probs[mask], 5)
            row["mean_true_class_probability"] = float(probs[mask, label].mean())
            row["mean_confidence"] = float(probs[mask].max(axis=1).mean())

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Optional final ensemble reconstruction
# -----------------------------------------------------------------------------

def build_simple_weighted_ensemble(candidates: list[dict[str, Any]], weights_json: str, name: str) -> dict[str, Any] | None:
    """Build an ensemble from already aligned candidates.

    weights_json example:
    '{"hybrid_temporal_tabular_cnn_250k": 0.8, "temporal_cnn_250k": 0.2}'

    This is useful if the final ensemble probabilities were not saved as a standalone file.
    """
    if not weights_json:
        return None

    weights = json.loads(weights_json)
    by_name = {c["model_name"]: c for c in candidates}

    selected = []
    selected_weights = []
    for model_name, w in weights.items():
        if model_name not in by_name:
            print(f"[WARN] Ensemble requested model not found: {model_name}")
            return None
        if by_name[model_name]["probs"] is None:
            print(f"[WARN] Ensemble requested model has no probabilities: {model_name}")
            return None
        selected.append(by_name[model_name])
        selected_weights.append(float(w))

    if not selected:
        return None

    selected_weights = np.asarray(selected_weights, dtype=float)
    selected_weights = selected_weights / np.maximum(selected_weights.sum(), 1e-12)

    y_true = selected[0]["y_true"]
    object_ids = selected[0]["object_ids"]
    probs = np.zeros_like(selected[0]["probs"], dtype=float)
    for c, w in zip(selected, selected_weights):
        if not np.array_equal(c["y_true"], y_true):
            raise ValueError("Ensemble candidates are not aligned.")
        probs += w * c["probs"]
    probs = sanitize_probabilities(probs)

    return {
        "model_name": name,
        "prediction_path": "constructed_in_memory",
        "probability_path": "constructed_in_memory",
        "true_col": "true_label",
        "pred_col": "argmax(ensemble_probabilities)",
        "object_ids": object_ids,
        "y_true": y_true,
        "y_pred": np.argmax(probs, axis=1).astype(int),
        "probs": probs,
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def write_markdown(
    output_dir: Path,
    candidate_metrics: pd.DataFrame,
    ci_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
):
    md = "# Bootstrap statistical analysis for AstroTrust-AI\n\n"
    md += (
        "This report estimates uncertainty in test-set metrics using non-parametric bootstrap resampling. "
        "It also reports paired bootstrap differences and McNemar tests for top-1 correctness.\n\n"
    )

    md += "## Point estimates\n\n"
    md += candidate_metrics.to_markdown(index=False)
    md += "\n\n## 95% bootstrap confidence intervals\n\n"
    md += ci_df.to_markdown(index=False)
    md += "\n\n## Paired bootstrap model differences\n\n"
    md += paired_df.to_markdown(index=False) if not paired_df.empty else "_No paired comparisons._"
    md += "\n\n## McNemar tests\n\n"
    md += mcnemar_df.to_markdown(index=False) if not mcnemar_df.empty else "_No McNemar comparisons._"
    md += "\n\n## Per-class sensitivity\n\n"
    md += per_class_df.head(80).to_markdown(index=False) if not per_class_df.empty else "_No per-class data._"
    md += "\n\n## Manuscript interpretation notes\n\n"
    md += (
        "- If confidence intervals for two models overlap, this does not by itself prove no difference; use the paired bootstrap difference table.\n"
        "- For the ensemble versus hybrid comparison, focus on the paired difference in accuracy and Macro-F1.\n"
        "- A confidence interval for the difference that excludes zero supports a robust difference.\n"
        "- If the interval includes zero, describe the ensemble gain as small or marginal rather than statistically robust.\n"
        "- Per-class sensitivity should be used to discuss which classes remain difficult and whether top-k retrieval mitigates those errors.\n"
    )

    (output_dir / "bootstrap_statistical_analysis_summary.md").write_text(md, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals and paired statistical tests for AstroTrust-AI models.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble-weights-json",
        type=str,
        default="",
        help="Optional JSON mapping model_name to probability weight to reconstruct a final ensemble.",
    )
    parser.add_argument(
        "--ensemble-name",
        type=str,
        default="constructed_weighted_ensemble",
        help="Name for optional reconstructed ensemble.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering 32-class prediction files...")
    candidates = discover_candidates()
    candidates = align_candidates(candidates)

    if not candidates:
        raise RuntimeError("No aligned 32-class candidates found.")

    reconstructed = build_simple_weighted_ensemble(candidates, args.ensemble_weights_json, args.ensemble_name)
    if reconstructed is not None:
        candidates.append(reconstructed)

    print("\nAligned candidates:")
    for c in candidates:
        print(
            f"- {c['model_name']} | n={len(c['y_true'])} | "
            f"probs={'yes' if c['probs'] is not None else 'no'} | {c['prediction_path']}"
        )

    # Point estimates.
    metric_rows = []
    for c in candidates:
        row = {"model": c["model_name"], "n_test": len(c["y_true"]), **metric_bundle(c["y_true"], c["y_pred"], c["probs"])}
        metric_rows.append(row)
    candidate_metrics = pd.DataFrame(metric_rows).sort_values("accuracy", ascending=False)
    candidate_metrics.to_csv(args.output_dir / "model_point_estimates.csv", index=False)

    print("\nPoint estimates:")
    print(candidate_metrics.to_string(index=False))

    # Bootstrap CIs.
    print("\nRunning bootstrap confidence intervals...")
    ci_frames = []
    for i, c in enumerate(candidates):
        ci_frames.append(bootstrap_metric_ci(c, args.n_bootstrap, args.seed + 100 * i, args.ci))
    ci_df = pd.concat(ci_frames, ignore_index=True)
    ci_df.to_csv(args.output_dir / "bootstrap_metric_confidence_intervals.csv", index=False)

    # Paired differences against the best model and against hybrid, when available.
    paired_frames = []
    mcnemar_rows = []
    best_model_name = candidate_metrics.iloc[0]["model"]
    best_cand = next(c for c in candidates if c["model_name"] == best_model_name)

    # Compare best to all others.
    for c in candidates:
        if c["model_name"] == best_model_name:
            continue
        paired_frames.append(paired_bootstrap_difference(best_cand, c, args.n_bootstrap, args.seed + 999, args.ci))
        mcnemar_rows.append(mcnemar_test(best_cand, c))

    # Also compare ensemble_hybrid_dominant/constructed ensemble against hybrid if both exist.
    by_name = {c["model_name"]: c for c in candidates}
    hybrid_names = [n for n in by_name if "hybrid_temporal_tabular_cnn_250k" == n or n.endswith("hybrid_temporal_tabular_cnn_250k")]
    ensemble_names = [n for n in by_name if "ensemble" in n.lower()]
    if hybrid_names and ensemble_names:
        h = by_name[hybrid_names[0]]
        e = by_name[ensemble_names[0]]
        if e["model_name"] != best_cand["model_name"] or h["model_name"] != best_cand["model_name"]:
            paired_frames.append(paired_bootstrap_difference(e, h, args.n_bootstrap, args.seed + 1999, args.ci))
            mcnemar_rows.append(mcnemar_test(e, h))

    paired_df = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    paired_df.to_csv(args.output_dir / "paired_bootstrap_model_differences.csv", index=False)

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv(args.output_dir / "mcnemar_tests_top1_correctness.csv", index=False)

    # Per-class sensitivity for all candidates.
    per_class_frames = []
    for c in candidates:
        per_class_frames.append(per_class_sensitivity(c))
    per_class_df = pd.concat(per_class_frames, ignore_index=True) if per_class_frames else pd.DataFrame()
    per_class_df.to_csv(args.output_dir / "per_class_sensitivity.csv", index=False)

    write_markdown(args.output_dir, candidate_metrics, ci_df, paired_df, mcnemar_df, per_class_df)

    print("\nBootstrap CIs:")
    print(ci_df[["model", "metric", "estimate", "ci_lower", "ci_upper", "bootstrap_std"]].to_string(index=False))

    print("\nPaired differences:")
    print(paired_df.to_string(index=False) if not paired_df.empty else "No paired comparisons.")

    print("\nMcNemar tests:")
    print(mcnemar_df.to_string(index=False) if not mcnemar_df.empty else "No McNemar tests.")

    print("\nSaved:")
    print(f"- {args.output_dir / 'model_point_estimates.csv'}")
    print(f"- {args.output_dir / 'bootstrap_metric_confidence_intervals.csv'}")
    print(f"- {args.output_dir / 'paired_bootstrap_model_differences.csv'}")
    print(f"- {args.output_dir / 'mcnemar_tests_top1_correctness.csv'}")
    print(f"- {args.output_dir / 'per_class_sensitivity.csv'}")
    print(f"- {args.output_dir / 'bootstrap_statistical_analysis_summary.md'}")


if __name__ == "__main__":
    main()
