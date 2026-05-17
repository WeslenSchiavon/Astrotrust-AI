#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_publication_safe_ensemble_temperature_calibration.py

Publication-safe temperature scaling for the final ensemble.

Supports two modes:

MODE A: ensemble probabilities already exist for validation and test.
python .\experiments\run_publication_safe_ensemble_temperature_calibration.py `
  --ensemble-val-probs PATH_TO_VALIDATION_ENSEMBLE_PROBS.npy `
  --val-labels-csv PATH_TO_VALIDATION_LABELS.csv `
  --ensemble-test-probs results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_test_probabilities.npy `
  --test-predictions results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_predictions.csv `
  --output-dir results\ensemble_temperature_calibration_250k_final `
  --n-bootstrap 1000 `
  --n-bins 15

MODE B: reconstruct ensemble probabilities from base-model probabilities and weights.
python .\experiments\run_publication_safe_ensemble_temperature_calibration.py `
  --base-val-probs val_model1.npy,val_model2.npy `
  --base-test-probs test_model1.npy,test_model2.npy `
  --weights 0.7,0.3 `
  --val-labels-csv PATH_TO_VALIDATION_LABELS.csv `
  --test-predictions results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_predictions.csv `
  --test-reference-probs results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_test_probabilities.npy `
  --output-dir results\ensemble_temperature_calibration_250k_final `
  --n-bootstrap 1000 `
  --n-bins 15

Important:
- Temperature is fitted ONLY on validation probabilities.
- Test-set fitting is not performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-12


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def temp_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    probs = normalize_probs(probs)
    logp = np.log(np.clip(probs, EPS, 1.0)) / float(temperature)
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    n, k = probs.shape
    one = np.zeros_like(probs)
    one[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((probs - one) ** 2, axis=1)))


def ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        out += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(out)


def reliability_bins(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        rows.append({
            "bin": i,
            "bin_low": lo,
            "bin_high": hi,
            "n": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if np.any(mask) else np.nan,
            "accuracy": float(correct[mask].mean()) if np.any(mask) else np.nan,
            "fraction": float(mask.mean()),
        })
    return pd.DataFrame(rows)


def metrics(y: np.ndarray, probs: np.ndarray, n_bins: int) -> dict:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return {
        "accuracy": float(np.mean(pred == y)),
        "mean_confidence": float(conf.mean()),
        "calibration_gap_abs": float(abs(np.mean(pred == y) - conf.mean())),
        "ece": ece(y, probs, n_bins=n_bins),
        "brier_score": brier(y, probs),
        "negative_log_likelihood": nll(y, probs),
        "top3_accuracy": float(np.mean([y[i] in np.argsort(probs[i])[-3:] for i in range(len(y))])),
        "top5_accuracy": float(np.mean([y[i] in np.argsort(probs[i])[-5:] for i in range(len(y))])),
    }


def fit_temperature_grid_then_refine(y_val: np.ndarray, probs_val: np.ndarray) -> tuple[float, float]:
    # Robust and dependency-light. Uses scipy if available, otherwise dense grid.
    def obj(t: float) -> float:
        return nll(y_val, temp_scale_probs(probs_val, t))

    grid = np.exp(np.linspace(np.log(0.2), np.log(5.0), 250))
    vals = np.array([obj(float(t)) for t in grid])
    best_idx = int(np.argmin(vals))
    best_t = float(grid[best_idx])
    best_nll = float(vals[best_idx])

    try:
        from scipy.optimize import minimize_scalar
        lo = float(grid[max(0, best_idx - 3)])
        hi = float(grid[min(len(grid) - 1, best_idx + 3)])
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
        if res.success and math.isfinite(res.fun):
            best_t = float(res.x)
            best_nll = float(res.fun)
    except Exception:
        pass

    return best_t, best_nll


def read_labels(csv_path: Path, label_col: str, split_col: str | None, split_value: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if split_col and split_col in df.columns and split_value is not None:
        df = df[df[split_col].astype(str).str.lower().eq(split_value.lower())].copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column `{label_col}` not found in {csv_path}. Columns: {df.columns.tolist()}")
    return df.reset_index(drop=True)


def parse_paths(s: str) -> list[Path]:
    return [Path(x.strip()) for x in s.split(",") if x.strip()]


def parse_weights(s: str) -> np.ndarray:
    w = np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    if w.sum() <= 0:
        raise ValueError("At least one weight must be positive.")
    return w / w.sum()


def weighted_average_probs(paths: list[Path], weights: np.ndarray) -> np.ndarray:
    if len(paths) != len(weights):
        raise ValueError(f"Number of paths ({len(paths)}) != number of weights ({len(weights)})")
    arrs = [normalize_probs(np.load(p)) for p in paths]
    shape = arrs[0].shape
    if any(a.shape != shape for a in arrs):
        raise ValueError(f"Probability arrays have different shapes: {[a.shape for a in arrs]}")
    out = np.zeros(shape, dtype=np.float64)
    for a, w in zip(arrs, weights):
        out += float(w) * a
    return normalize_probs(out)


def bootstrap_deltas(y: np.ndarray, raw: np.ndarray, temp: np.ndarray, n_bootstrap: int, n_bins: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx_all = np.arange(n)

    point_raw = metrics(y, raw, n_bins)
    point_temp = metrics(y, temp, n_bins)

    rows = []
    directions = {
        "accuracy": "higher_is_better",
        "top3_accuracy": "higher_is_better",
        "top5_accuracy": "higher_is_better",
        "mean_confidence": "descriptive",
        "calibration_gap_abs": "lower_is_better",
        "ece": "lower_is_better",
        "brier_score": "lower_is_better",
        "negative_log_likelihood": "lower_is_better",
    }

    for metric in directions:
        samples = []
        for _ in range(n_bootstrap):
            idx = rng.choice(idx_all, size=n, replace=True)
            samples.append(metrics(y[idx], raw[idx], n_bins)[metric])
        raw_samples = np.array(samples)

        samples = []
        for _ in range(n_bootstrap):
            idx = rng.choice(idx_all, size=n, replace=True)
            samples.append(metrics(y[idx], temp[idx], n_bins)[metric])
        temp_samples = np.array(samples)

        # Paired enough in object distribution? Better use same index for paired deltas:
        paired = []
        for _ in range(n_bootstrap):
            idx = rng.choice(idx_all, size=n, replace=True)
            paired.append(metrics(y[idx], temp[idx], n_bins)[metric] - metrics(y[idx], raw[idx], n_bins)[metric])
        paired = np.array(paired)

        if directions[metric] == "lower_is_better":
            p_no_improve = float(np.mean(paired >= 0))
        elif directions[metric] == "higher_is_better":
            p_no_improve = float(np.mean(paired <= 0))
        else:
            p_no_improve = np.nan

        rows.append({
            "metric": metric,
            "raw": point_raw[metric],
            "temperature_scaled": point_temp[metric],
            "delta_temperature_minus_raw": point_temp[metric] - point_raw[metric],
            "bootstrap_ci_low_95": float(np.quantile(paired, 0.025)),
            "bootstrap_ci_high_95": float(np.quantile(paired, 0.975)),
            "direction": directions[metric],
            "one_sided_bootstrap_pvalue_no_improvement": p_no_improve,
        })

    return pd.DataFrame(rows)


def make_figures(y: np.ndarray, raw: np.ndarray, temp: np.ndarray, n_bins: int, out: Path) -> None:
    rb_raw = reliability_bins(y, raw, n_bins)
    rb_temp = reliability_bins(y, temp, n_bins)
    rb_raw["variant"] = "raw"
    rb_temp["variant"] = "temperature_scaled"
    bins = pd.concat([rb_raw, rb_temp], ignore_index=True)
    bins.to_csv(out / "ensemble_temperature_reliability_bins.csv", index=False)

    plt.figure(figsize=(5.5, 5.0))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    for variant, df in bins.groupby("variant"):
        df2 = df.dropna(subset=["mean_confidence", "accuracy"])
        plt.plot(df2["mean_confidence"], df2["accuracy"], marker="o", label=variant)
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Ensemble reliability: raw vs temperature-scaled")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "fig_ensemble_temperature_reliability.png", dpi=220)
    plt.close()

    mraw = metrics(y, raw, n_bins)
    mtemp = metrics(y, temp, n_bins)
    names = ["ece", "brier_score", "negative_log_likelihood", "calibration_gap_abs"]
    vals_raw = [mraw[n] for n in names]
    vals_temp = [mtemp[n] for n in names]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(x - width / 2, vals_raw, width, label="raw")
    plt.bar(x + width / 2, vals_temp, width, label="temperature_scaled")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.title("Ensemble calibration metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "fig_ensemble_temperature_metric_comparison.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ensemble-val-probs")
    parser.add_argument("--ensemble-test-probs")

    parser.add_argument("--base-val-probs")
    parser.add_argument("--base-test-probs")
    parser.add_argument("--weights")
    parser.add_argument("--test-reference-probs")

    parser.add_argument("--val-labels-csv", required=True)
    parser.add_argument("--val-label-col", default="label")
    parser.add_argument("--val-split-col", default="split")
    parser.add_argument("--val-split-value", default="validation")

    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--test-label-col", default="true_label")

    parser.add_argument("--output-dir", default="results/ensemble_temperature_calibration_250k_final")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reconstruction-tolerance", type=float, default=1e-7)

    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.ensemble_val_probs and args.ensemble_test_probs:
        val_probs = normalize_probs(np.load(args.ensemble_val_probs))
        test_probs = normalize_probs(np.load(args.ensemble_test_probs))
        mode = "direct_ensemble_probabilities"
        reconstruction_report = {}
    elif args.base_val_probs and args.base_test_probs and args.weights:
        weights = parse_weights(args.weights)
        val_paths = parse_paths(args.base_val_probs)
        test_paths = parse_paths(args.base_test_probs)
        val_probs = weighted_average_probs(val_paths, weights)
        test_probs = weighted_average_probs(test_paths, weights)
        mode = "reconstructed_weighted_ensemble_probabilities"
        reconstruction_report = {
            "base_val_probs": [str(p) for p in val_paths],
            "base_test_probs": [str(p) for p in test_paths],
            "weights_normalized": weights.tolist(),
        }
        if args.test_reference_probs:
            ref = normalize_probs(np.load(args.test_reference_probs))
            max_abs = float(np.max(np.abs(ref - test_probs)))
            mean_abs = float(np.mean(np.abs(ref - test_probs)))
            reconstruction_report["test_reference_probs"] = args.test_reference_probs
            reconstruction_report["max_abs_difference_vs_reference_test_probs"] = max_abs
            reconstruction_report["mean_abs_difference_vs_reference_test_probs"] = mean_abs
            reconstruction_report["matches_reference_within_tolerance"] = bool(max_abs <= args.reconstruction_tolerance)
            if max_abs > args.reconstruction_tolerance:
                print(f"WARNING: reconstructed test probabilities differ from reference. max_abs={max_abs:.3e}")
    else:
        raise ValueError("Use either direct ensemble mode or reconstruction mode. See docstring examples.")

    val_df = read_labels(
        Path(args.val_labels_csv),
        label_col=args.val_label_col,
        split_col=args.val_split_col,
        split_value=args.val_split_value,
    )
    y_val = val_df[args.val_label_col].to_numpy(dtype=int)

    test_df = pd.read_csv(args.test_predictions)
    if args.test_label_col not in test_df.columns:
        raise ValueError(f"Test label column `{args.test_label_col}` not found in {args.test_predictions}")
    y_test = test_df[args.test_label_col].to_numpy(dtype=int)

    if len(y_val) != val_probs.shape[0]:
        raise ValueError(f"Validation labels/probabilities length mismatch: {len(y_val)} vs {val_probs.shape[0]}")
    if len(y_test) != test_probs.shape[0]:
        raise ValueError(f"Test labels/probabilities length mismatch: {len(y_test)} vs {test_probs.shape[0]}")
    if val_probs.shape[1] != test_probs.shape[1]:
        raise ValueError(f"Validation/test class count mismatch: {val_probs.shape[1]} vs {test_probs.shape[1]}")

    T, val_nll = fit_temperature_grid_then_refine(y_val, val_probs)
    val_temp = temp_scale_probs(val_probs, T)
    test_temp = temp_scale_probs(test_probs, T)

    np.save(out / "ensemble_validation_temperature_scaled_probabilities.npy", val_temp)
    np.save(out / "ensemble_test_temperature_scaled_probabilities.npy", test_temp)

    val_point = pd.DataFrame([
        {"split": "validation", "variant": "raw", **metrics(y_val, val_probs, args.n_bins)},
        {"split": "validation", "variant": "temperature_scaled", **metrics(y_val, val_temp, args.n_bins)},
    ])
    test_point = pd.DataFrame([
        {"split": "test", "variant": "raw", **metrics(y_test, test_probs, args.n_bins)},
        {"split": "test", "variant": "temperature_scaled", **metrics(y_test, test_temp, args.n_bins)},
    ])
    point = pd.concat([val_point, test_point], ignore_index=True)
    point.to_csv(out / "ensemble_temperature_point_metrics.csv", index=False)

    boot = bootstrap_deltas(
        y_test,
        test_probs,
        test_temp,
        n_bootstrap=args.n_bootstrap,
        n_bins=args.n_bins,
        seed=args.seed,
    )
    boot.to_csv(out / "ensemble_temperature_bootstrap_deltas.csv", index=False)

    make_figures(y_test, test_probs, test_temp, args.n_bins, out)

    pred_raw = test_probs.argmax(axis=1)
    pred_temp = test_temp.argmax(axis=1)
    top1_unchanged = float(np.mean(pred_raw == pred_temp))

    schema = {
        "analysis": "publication-safe final ensemble temperature calibration",
        "mode": mode,
        "temperature_fit_split": "validation",
        "temperature_applied_split": "test",
        "temperature": float(T),
        "validation_nll_at_temperature": float(val_nll),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_classes": int(test_probs.shape[1]),
        "n_bins": int(args.n_bins),
        "n_bootstrap": int(args.n_bootstrap),
        "top1_unchanged_fraction_on_test": top1_unchanged,
        "reconstruction_report": reconstruction_report,
        "no_test_set_temperature_fitting": True,
    }
    (out / "ensemble_temperature_calibration_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    raw_test = test_point[test_point["variant"] == "raw"].iloc[0].to_dict()
    tmp_test = test_point[test_point["variant"] == "temperature_scaled"].iloc[0].to_dict()

    def row(metric: str) -> pd.Series:
        return boot[boot["metric"] == metric].iloc[0]

    e = row("ece")
    br = row("brier_score")
    nl = row("negative_log_likelihood")
    cg = row("calibration_gap_abs")

    md = []
    md.append("# Publication-safe final ensemble temperature calibration\n")
    md.append("This report calibrates the final `ensemble_hybrid_dominant` probabilities using temperature scaling fitted on a validation split and applied to the held-out test split.\n")
    md.append("## Protocol\n")
    md.append(f"- Mode: `{mode}`")
    md.append(f"- Temperature fitted on: `validation`")
    md.append(f"- Temperature applied to: `test`")
    md.append(f"- Temperature: `{T:.6f}`")
    md.append(f"- Validation objects: `{len(y_val)}`")
    md.append(f"- Test objects: `{len(y_test)}`")
    md.append(f"- Classes: `{test_probs.shape[1]}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`")
    md.append(f"- Top-1 unchanged fraction on test: `{top1_unchanged:.6f}`")
    md.append("- No temperature parameter was fitted on the test set.\n")

    if reconstruction_report:
        md.append("## Reconstruction check\n")
        for k, v in reconstruction_report.items():
            md.append(f"- {k}: `{v}`")
        md.append("")

    md.append("## Point metrics\n")
    md.append(point.to_markdown(index=False))
    md.append("\n## Test-set paired bootstrap deltas\n")
    md.append(boot.to_markdown(index=False))

    md.append("\n## Key interpretation\n")
    md.append(f"- Test raw ECE: `{raw_test['ece']:.6f}`; temperature-scaled ECE: `{tmp_test['ece']:.6f}`; delta: `{float(e['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(e['bootstrap_ci_low_95']):.6f}`, `{float(e['bootstrap_ci_high_95']):.6f}`].")
    md.append(f"- Test raw Brier score: `{raw_test['brier_score']:.6f}`; temperature-scaled Brier score: `{tmp_test['brier_score']:.6f}`; delta: `{float(br['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(br['bootstrap_ci_low_95']):.6f}`, `{float(br['bootstrap_ci_high_95']):.6f}`].")
    md.append(f"- Test raw NLL: `{raw_test['negative_log_likelihood']:.6f}`; temperature-scaled NLL: `{tmp_test['negative_log_likelihood']:.6f}`; delta: `{float(nl['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(nl['bootstrap_ci_low_95']):.6f}`, `{float(nl['bootstrap_ci_high_95']):.6f}`].")
    md.append(f"- Test calibration gap delta: `{float(cg['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(cg['bootstrap_ci_low_95']):.6f}`, `{float(cg['bootstrap_ci_high_95']):.6f}`].")

    md.append("\n## Suggested manuscript wording\n")
    md.append("```latex")
    md.append(
        f"We calibrated the final \\texttt{{ensemble\\_hybrid\\_dominant}} probabilities using temperature scaling fitted exclusively on the validation split "
        f"and then applied the learned temperature to the held-out test split. The fitted temperature was \\textbf{{{T:.4f}}}. "
        f"On the test set, ECE changed from \\textbf{{{raw_test['ece']:.4f}}} to \\textbf{{{tmp_test['ece']:.4f}}} "
        f"(bootstrap delta = \\textbf{{{float(e['delta_temperature_minus_raw']):+.4f}}}; 95\\% CI [{float(e['bootstrap_ci_low_95']):.4f}, {float(e['bootstrap_ci_high_95']):.4f}]). "
        f"The Brier score changed from \\textbf{{{raw_test['brier_score']:.4f}}} to \\textbf{{{tmp_test['brier_score']:.4f}}} "
        f"(delta = \\textbf{{{float(br['delta_temperature_minus_raw']):+.4f}}}; 95\\% CI [{float(br['bootstrap_ci_low_95']):.4f}, {float(br['bootstrap_ci_high_95']):.4f}]), "
        f"and the negative log-likelihood changed from \\textbf{{{raw_test['negative_log_likelihood']:.4f}}} to \\textbf{{{tmp_test['negative_log_likelihood']:.4f}}} "
        f"(delta = \\textbf{{{float(nl['delta_temperature_minus_raw']):+.4f}}}; 95\\% CI [{float(nl['bootstrap_ci_low_95']):.4f}, {float(nl['bootstrap_ci_high_95']):.4f}]). "
        f"No temperature parameter was fitted on the test set."
    )
    md.append("```\n")

    md.append("## Output files\n")
    for name in [
        "ensemble_temperature_point_metrics.csv",
        "ensemble_temperature_bootstrap_deltas.csv",
        "ensemble_temperature_reliability_bins.csv",
        "ensemble_temperature_calibration_schema.json",
        "ensemble_validation_temperature_scaled_probabilities.npy",
        "ensemble_test_temperature_scaled_probabilities.npy",
        "fig_ensemble_temperature_reliability.png",
        "fig_ensemble_temperature_metric_comparison.png",
    ]:
        md.append(f"- `{out / name}`")

    (out / "ensemble_temperature_calibration_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results written to: {out}")
    print(f"Temperature: {T:.6f}")
    print(f"Summary: {out / 'ensemble_temperature_calibration_summary.md'}")


if __name__ == "__main__":
    main()
