from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "final_publication" / "bootstrap_calibration_followup"

FINAL_MODEL_FILES = {
    "ensemble_hybrid_dominant": {
        "predictions": RESULTS_DIR / "hybrid_tabular_ensemble_250k" / "ensemble_hybrid_dominant_predictions.csv",
        "probabilities": RESULTS_DIR / "hybrid_tabular_ensemble_250k" / "ensemble_hybrid_dominant_test_probabilities.npy",
    },
    "hybrid_temporal_tabular_cnn": {
        "predictions": RESULTS_DIR / "hybrid_temporal_tabular_cnn_250k" / "hybrid_temporal_tabular_cnn_test_predictions.csv",
        "probabilities": RESULTS_DIR / "hybrid_temporal_tabular_cnn_250k" / "hybrid_temporal_tabular_cnn_test_probabilities.npy",
    },
    "lightgbm_v4_baseline": {
        "predictions": RESULTS_DIR / "hybrid_tabular_ensemble_250k" / "lightgbm_baseline_predictions.csv",
        "probabilities": RESULTS_DIR / "hybrid_tabular_ensemble_250k" / "lightgbm_baseline_test_probabilities.npy",
    },
    "temporal_cnn": {
        "predictions": RESULTS_DIR / "temporal_cnn_250k" / "temporal_cnn_test_predictions.csv",
        "probabilities": RESULTS_DIR / "temporal_cnn_250k" / "temporal_cnn_test_probabilities.npy",
    },
    "temporal_cnn_v2": {
        "predictions": RESULTS_DIR / "temporal_cnn_v2_250k" / "temporal_cnn_v2_test_predictions.csv",
        "probabilities": RESULTS_DIR / "temporal_cnn_v2_250k" / "temporal_cnn_v2_test_probabilities.npy",
    },
    "hybrid_temporal_tabular_cnn_temperature_scaled": {
        "predictions": RESULTS_DIR
        / "final_publication"
        / "temperature_scaled_hybrid_250k"
        / "hybrid_temporal_tabular_cnn_temperature_scaled_test_predictions.csv",
        "probabilities": RESULTS_DIR
        / "final_publication"
        / "temperature_scaled_hybrid_250k"
        / "hybrid_temporal_tabular_cnn_temperature_scaled_test_probabilities.npy",
    },
}

TRUE_CANDIDATES = [
    "true_label", "y_true", "label", "target", "true_class", "true_class_id", "class_id",
]
PRED_CANDIDATES = [
    "predicted_label", "y_pred", "prediction", "pred_label", "predicted_class_id",
    "ensemble_predicted_label", "yhat", "y_hat",
]
OBJECT_ID_CANDIDATES = ["object_id", "diaobjectid", "diaObjectId", "oid", "snid"]

PRIORITY_CANDIDATES = [
    "novelty_rarity", "novelty_rarity_score", "followup_priority", "priority_score",
    "candidate_priority", "score", "priority", "mean_priority",
]
RARE_CANDIDATES = [
    "rare_true", "is_rare", "rare", "true_is_rare", "rare_class", "is_rare_class",
    "target_is_rare", "rarity_true", "true_rare",
]
UNCERTAINTY_CANDIDATES = ["uncertainty", "uncertainty_score", "entropy", "predictive_entropy"]
NOVELTY_CANDIDATES = ["novelty", "novelty_score", "ood_score", "outlier_score"]
RARITY_SCORE_CANDIDATES = ["rarity", "rarity_score", "class_rarity", "inverse_frequency"]


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


def one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_true), n_classes), dtype=np.float64)
    valid = (y_true >= 0) & (y_true < n_classes)
    y[np.arange(len(y_true))[valid], y_true[valid]] = 1.0
    return y


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    p = sanitize_probabilities(probabilities)
    y = one_hot(y_true.astype(int), p.shape[1])
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def negative_log_likelihood(y_true: np.ndarray, probabilities: np.ndarray, eps: float = 1e-12) -> float:
    p = sanitize_probabilities(probabilities)
    chosen = p[np.arange(len(y_true)), y_true.astype(int)]
    return float(-np.mean(np.log(np.clip(chosen, eps, 1.0))))


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    p = sanitize_probabilities(probabilities)
    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def maximum_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    p = sanitize_probabilities(probabilities)
    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    errors = []
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            errors.append(abs(correct[mask].mean() - confidences[mask].mean()))
    return float(max(errors) if errors else 0.0)


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int) -> dict[str, float]:
    p = sanitize_probabilities(probabilities)
    return {
        "ece": expected_calibration_error(y_true, p, n_bins=n_bins),
        "mce": maximum_calibration_error(y_true, p, n_bins=n_bins),
        "brier_score": multiclass_brier_score(y_true, p),
        "negative_log_likelihood": negative_log_likelihood(y_true, p),
        "mean_confidence": float(p.max(axis=1).mean()),
    }


def load_final_model(model_name: str) -> dict[str, Any]:
    files = FINAL_MODEL_FILES[model_name]
    pred_path = files["predictions"]
    prob_path = files["probabilities"]
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not prob_path.exists():
        raise FileNotFoundError(prob_path)

    df = pd.read_csv(pred_path)
    true_col = find_column(df, TRUE_CANDIDATES)
    pred_col = find_column(df, PRED_CANDIDATES)
    if true_col is None:
        raise ValueError(f"No true-label column found in {pred_path}")

    probabilities = sanitize_probabilities(np.load(prob_path))
    if len(probabilities) != len(df):
        raise ValueError(f"Probability row mismatch for {model_name}: {len(probabilities)} vs {len(df)}")

    y_true = df[true_col].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy() if pred_col else probabilities.argmax(axis=1)

    return {
        "model": model_name,
        "prediction_path": str(pred_path),
        "probability_path": str(prob_path),
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "n_test": len(df),
    }


def bootstrap_calibration_ci(
    model: dict[str, Any],
    n_bootstrap: int,
    seed: int,
    ci_level: float,
    n_bins: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = model["y_true"]
    probabilities = model["probabilities"]
    n = len(y_true)

    estimates = calibration_metrics(y_true, probabilities, n_bins=n_bins)
    boot = {metric: [] for metric in estimates}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals = calibration_metrics(y_true[idx], probabilities[idx], n_bins=n_bins)
        for metric, value in vals.items():
            boot[metric].append(value)
        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"  {model['model']}: calibration bootstrap {b + 1}/{n_bootstrap}", flush=True)

    alpha = (1.0 - ci_level) / 2.0
    rows = []
    for metric, estimate in estimates.items():
        arr = np.asarray(boot[metric], dtype=float)
        rows.append({
            "model": model["model"],
            "metric": metric,
            "estimate": float(estimate),
            "ci_lower": float(np.quantile(arr, alpha)),
            "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
            "bootstrap_std": float(np.std(arr, ddof=1)),
            "ci_level": ci_level,
            "n_test": n,
            "n_bootstrap": n_bootstrap,
            "n_bins": n_bins,
            "prediction_path": model["prediction_path"],
            "probability_path": model["probability_path"],
        })
    return pd.DataFrame(rows)


def paired_bootstrap_calibration_difference(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    n_bootstrap: int,
    seed: int,
    ci_level: float,
    n_bins: int,
) -> pd.DataFrame:
    if not np.array_equal(model_a["y_true"], model_b["y_true"]):
        raise ValueError("Models are not aligned by y_true.")

    rng = np.random.default_rng(seed)
    y_true = model_a["y_true"]
    p_a = model_a["probabilities"]
    p_b = model_b["probabilities"]
    n = len(y_true)

    est_a = calibration_metrics(y_true, p_a, n_bins=n_bins)
    est_b = calibration_metrics(y_true, p_b, n_bins=n_bins)
    metrics = list(est_a.keys())
    boot = {metric: [] for metric in metrics}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals_a = calibration_metrics(y_true[idx], p_a[idx], n_bins=n_bins)
        vals_b = calibration_metrics(y_true[idx], p_b[idx], n_bins=n_bins)
        for metric in metrics:
            boot[metric].append(vals_a[metric] - vals_b[metric])

    alpha = (1.0 - ci_level) / 2.0
    rows = []
    for metric in metrics:
        arr = np.asarray(boot[metric], dtype=float)
        diff = est_a[metric] - est_b[metric]
        p_value = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
        rows.append({
            "model_a": model_a["model"],
            "model_b": model_b["model"],
            "comparison": f"{model_a['model']} - {model_b['model']}",
            "metric": metric,
            "difference": float(diff),
            "ci_lower": float(np.quantile(arr, alpha)),
            "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
            "bootstrap_std": float(np.std(arr, ddof=1)),
            "approx_p_value_two_sided": float(min(max(p_value, 0.0), 1.0)),
            "ci_level": ci_level,
            "n_test": n,
            "n_bootstrap": n_bootstrap,
            "n_bins": n_bins,
        })
    return pd.DataFrame(rows)


def discover_followup_candidates() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("**/*.csv")):
        low = str(path).lower()
        if any(skip in low for skip in ["bootstrap", "final_publication", "classification_report", "confusion", "metrics_summary"]):
            continue
        try:
            df = pd.read_csv(path, nrows=100)
        except Exception:
            continue
        true_col = find_column(df, TRUE_CANDIDATES)
        rare_col = find_column(df, RARE_CANDIDATES)
        priority_col = find_column(df, PRIORITY_CANDIDATES)
        novelty_col = find_column(df, NOVELTY_CANDIDATES)
        rarity_score_col = find_column(df, RARITY_SCORE_CANDIDATES)
        uncertainty_col = find_column(df, UNCERTAINTY_CANDIDATES)
        if true_col or rare_col or priority_col or novelty_col or rarity_score_col or uncertainty_col:
            rows.append({
                "path": str(path),
                "n_preview_rows": len(df),
                "columns": ",".join(map(str, df.columns)),
                "true_col": true_col or "",
                "rare_col": rare_col or "",
                "priority_col": priority_col or "",
                "novelty_col": novelty_col or "",
                "rarity_score_col": rarity_score_col or "",
                "uncertainty_col": uncertainty_col or "",
                "usable_for_followup_bootstrap": bool(rare_col and priority_col),
            })
    return pd.DataFrame(rows)


def parse_budget_fractions(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("%"):
            values.append(float(part[:-1]) / 100.0)
        else:
            values.append(float(part))
    return values


def normalize_boolean_series(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float) > 0
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "rare", "sim"]).to_numpy(dtype=bool)


def followup_metrics(
    is_rare: np.ndarray,
    priority: np.ndarray,
    budget_fractions: list[float],
) -> dict[str, float]:
    is_rare = np.asarray(is_rare, dtype=bool)
    priority = np.asarray(priority, dtype=float)
    n = len(is_rare)
    baseline_rate = float(np.mean(is_rare))
    order = np.argsort(-priority)

    out = {
        "baseline_rare_rate": baseline_rate,
        "n_candidates": float(n),
        "n_rare": float(np.sum(is_rare)),
    }

    for frac in budget_fractions:
        k = max(1, int(round(frac * n)))
        selected = order[:k]
        rare_rate = float(np.mean(is_rare[selected]))
        enrichment = rare_rate / baseline_rate if baseline_rate > 0 else np.nan
        key = f"top{int(round(frac * 100))}pct"
        out[f"{key}_budget_size"] = float(k)
        out[f"{key}_rare_rate"] = rare_rate
        out[f"{key}_rare_enrichment"] = enrichment
        out[f"{key}_n_rare"] = float(np.sum(is_rare[selected]))

    return out


def bootstrap_followup_ci(
    df: pd.DataFrame,
    rare_col: str,
    priority_col: str,
    budget_fractions: list[float],
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    is_rare = normalize_boolean_series(df[rare_col])
    priority = pd.to_numeric(df[priority_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = len(df)

    estimates = followup_metrics(is_rare, priority, budget_fractions)
    boot = {metric: [] for metric in estimates}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals = followup_metrics(is_rare[idx], priority[idx], budget_fractions)
        for metric, value in vals.items():
            boot[metric].append(value)
        if (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"  follow-up bootstrap {b + 1}/{n_bootstrap}", flush=True)

    alpha = (1.0 - ci_level) / 2.0
    rows = []
    for metric, estimate in estimates.items():
        arr = np.asarray(boot[metric], dtype=float)
        rows.append({
            "metric": metric,
            "estimate": float(estimate),
            "ci_lower": float(np.nanquantile(arr, alpha)),
            "ci_upper": float(np.nanquantile(arr, 1.0 - alpha)),
            "bootstrap_std": float(np.nanstd(arr, ddof=1)),
            "ci_level": ci_level,
            "n_candidates": n,
            "n_bootstrap": n_bootstrap,
            "rare_col": rare_col,
            "priority_col": priority_col,
        })

    selected_rows = []
    order = np.argsort(-priority)
    for frac in budget_fractions:
        k = max(1, int(round(frac * n)))
        selected = order[:k]
        selected_rows.append({
            "budget_fraction": frac,
            "budget_size": k,
            "rare_rate": float(np.mean(is_rare[selected])),
            "n_rare": int(np.sum(is_rare[selected])),
            "baseline_rare_rate": float(np.mean(is_rare)),
        })

    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def format_ci_table(ci_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
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


def write_markdown(
    output_dir: Path,
    calibration_ci: pd.DataFrame,
    calibration_diff: pd.DataFrame,
    followup_ci: pd.DataFrame | None,
    followup_file: str | None,
):
    md = "# Calibration and follow-up uncertainty bootstrap\n\n"
    md += "This report complements the final 250k bootstrap analysis by estimating uncertainty for calibration metrics and, when a suitable per-object follow-up file is available, rare-class enrichment.\n\n"

    md += "## Calibration metrics with 95% bootstrap confidence intervals\n\n"
    md += format_ci_table(
        calibration_ci,
        ["ece", "mce", "brier_score", "negative_log_likelihood", "mean_confidence"],
    ).to_markdown(index=False)
    md += "\n\n## Paired calibration differences\n\n"
    md += calibration_diff.to_markdown(index=False) if not calibration_diff.empty else "_No paired calibration differences._"
    md += "\n\n"

    if followup_ci is not None:
        md += f"## Follow-up rare-enrichment bootstrap\n\nFile: `{followup_file}`\n\n"
        md += followup_ci.to_markdown(index=False)
        md += "\n\n"
    else:
        md += "## Follow-up rare-enrichment bootstrap\n\nNo follow-up bootstrap table was generated. Use `--discover-followup-files` to locate candidate files or pass `--followup-file`, `--rare-col`, and `--priority-col`.\n\n"

    md += "## Suggested manuscript interpretation\n\n"
    md += (
        "- Calibration intervals quantify uncertainty in ECE, Brier score, and negative log-likelihood on the final test split.\n"
        "- Lower ECE/Brier/NLL values indicate better-calibrated probability estimates.\n"
        "- Follow-up enrichment intervals quantify whether the selected priority policy consistently enriches rare candidates under finite test-sample variation.\n"
        "- These intervals complement, but do not replace, repeated-seed training experiments.\n"
    )
    (output_dir / "bootstrap_calibration_followup_summary.md").write_text(md, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap uncertainty for calibration metrics and follow-up rare enrichment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--models", type=str, default=",".join(FINAL_MODEL_FILES.keys()))
    parser.add_argument("--discover-followup-files", action="store_true")
    parser.add_argument("--followup-file", type=Path, default=None)
    parser.add_argument("--rare-col", type=str, default="")
    parser.add_argument("--priority-col", type=str, default="")
    parser.add_argument("--budget-fractions", type=str, default="0.01,0.05,0.10,0.20")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.discover_followup_files:
        candidates = discover_followup_candidates()
        out = args.output_dir / "followup_candidate_files.csv"
        candidates.to_csv(out, index=False)
        print("[OK] Follow-up candidate files saved:")
        print(out)
        if not candidates.empty:
            print(candidates[["path", "rare_col", "priority_col", "usable_for_followup_bootstrap"]].to_string(index=False))
        return

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    models = [load_final_model(m) for m in selected_models]

    print("Loaded final models for calibration bootstrap:")
    for model in models:
        print(f"- {model['model']} | n={model['n_test']} | probs={model['probabilities'].shape}")

    calibration_frames = []
    for i, model in enumerate(models):
        calibration_frames.append(
            bootstrap_calibration_ci(
                model=model,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + i * 101,
                ci_level=args.ci,
                n_bins=args.n_bins,
            )
        )
    calibration_ci = pd.concat(calibration_frames, ignore_index=True)
    calibration_ci.to_csv(args.output_dir / "bootstrap_calibration_metric_confidence_intervals.csv", index=False)
    format_ci_table(
        calibration_ci,
        ["ece", "mce", "brier_score", "negative_log_likelihood", "mean_confidence"],
    ).to_csv(args.output_dir / "table_calibration_ci_manuscript.csv", index=False)

    by_model = {m["model"]: m for m in models}
    diff_frames = []
    if "ensemble_hybrid_dominant" in by_model and "hybrid_temporal_tabular_cnn" in by_model:
        diff_frames.append(
            paired_bootstrap_calibration_difference(
                by_model["ensemble_hybrid_dominant"],
                by_model["hybrid_temporal_tabular_cnn"],
                args.n_bootstrap,
                args.seed + 999,
                args.ci,
                args.n_bins,
            )
        )
    if "hybrid_temporal_tabular_cnn" in by_model and "lightgbm_v4_baseline" in by_model:
        diff_frames.append(
            paired_bootstrap_calibration_difference(
                by_model["hybrid_temporal_tabular_cnn"],
                by_model["lightgbm_v4_baseline"],
                args.n_bootstrap,
                args.seed + 1999,
                args.ci,
                args.n_bins,
            )
        )
    if "hybrid_temporal_tabular_cnn_temperature_scaled" in by_model and "hybrid_temporal_tabular_cnn" in by_model:
        diff_frames.append(
            paired_bootstrap_calibration_difference(
                by_model["hybrid_temporal_tabular_cnn_temperature_scaled"],
                by_model["hybrid_temporal_tabular_cnn"],
                args.n_bootstrap,
                args.seed + 2999,
                args.ci,
                args.n_bins,
            )
        )

    if "hybrid_temporal_tabular_cnn_temperature_scaled" in by_model and "ensemble_hybrid_dominant" in by_model:
        diff_frames.append(
            paired_bootstrap_calibration_difference(
                by_model["hybrid_temporal_tabular_cnn_temperature_scaled"],
                by_model["ensemble_hybrid_dominant"],
                args.n_bootstrap,
                args.seed + 3999,
                args.ci,
                args.n_bins,
            )
        )
    calibration_diff = pd.concat(diff_frames, ignore_index=True) if diff_frames else pd.DataFrame()
    calibration_diff.to_csv(args.output_dir / "paired_bootstrap_calibration_differences.csv", index=False)

    followup_ci = None
    followup_selected = None
    if args.followup_file is not None:
        if not args.followup_file.exists():
            raise FileNotFoundError(args.followup_file)
        df = pd.read_csv(args.followup_file)
        rare_col = args.rare_col or find_column(df, RARE_CANDIDATES)
        priority_col = args.priority_col or find_column(df, PRIORITY_CANDIDATES)
        if rare_col is None or priority_col is None:
            raise ValueError(
                "Could not infer rare/priority columns. Pass --rare-col and --priority-col explicitly."
            )
        budget_fractions = parse_budget_fractions(args.budget_fractions)
        followup_ci, followup_selected = bootstrap_followup_ci(
            df=df,
            rare_col=rare_col,
            priority_col=priority_col,
            budget_fractions=budget_fractions,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 2999,
            ci_level=args.ci,
        )
        followup_ci.to_csv(args.output_dir / "bootstrap_followup_enrichment_confidence_intervals.csv", index=False)
        followup_selected.to_csv(args.output_dir / "followup_selected_budget_summary.csv", index=False)

    write_markdown(
        output_dir=args.output_dir,
        calibration_ci=calibration_ci,
        calibration_diff=calibration_diff,
        followup_ci=followup_ci,
        followup_file=str(args.followup_file) if args.followup_file else None,
    )

    print("\nCalibration manuscript table:")
    print(format_ci_table(
        calibration_ci,
        ["ece", "mce", "brier_score", "negative_log_likelihood", "mean_confidence"],
    ).to_string(index=False))

    print("\nPaired calibration differences:")
    print(calibration_diff.to_string(index=False) if not calibration_diff.empty else "No paired calibration differences.")

    if followup_ci is not None:
        print("\nFollow-up enrichment bootstrap:")
        print(followup_ci.to_string(index=False))

    print("\nSaved:")
    print(f"- {args.output_dir / 'bootstrap_calibration_metric_confidence_intervals.csv'}")
    print(f"- {args.output_dir / 'table_calibration_ci_manuscript.csv'}")
    print(f"- {args.output_dir / 'paired_bootstrap_calibration_differences.csv'}")
    if followup_ci is not None:
        print(f"- {args.output_dir / 'bootstrap_followup_enrichment_confidence_intervals.csv'}")
        print(f"- {args.output_dir / 'followup_selected_budget_summary.csv'}")
    print(f"- {args.output_dir / 'bootstrap_calibration_followup_summary.md'}")


if __name__ == "__main__":
    main()
