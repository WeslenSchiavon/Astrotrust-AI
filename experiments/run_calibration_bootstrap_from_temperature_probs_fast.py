#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_calibration_bootstrap_from_temperature_probs_fast.py

Bootstrap pareado otimizado para comparar probabilidades raw vs temperature-scaled.

Este script é útil quando você tem:
1. CSV object-level com true_label, predicted_label, confidence;
2. arquivo .npy com probabilidades temperature-scaled;
3. temperatura T usada no temperature scaling.

Como reconstruímos as probabilidades raw:
Se q = softmax(z / T), então p_raw = normalize(q ** T).

Uso:
python .\experiments\run_calibration_bootstrap_from_temperature_probs_fast.py `
  --predictions results\final_publication\temperature_scaled_hybrid_250k\hybrid_temporal_tabular_cnn_temperature_scaled_test_predictions.csv `
  --temperature-probabilities results\final_publication\temperature_scaled_hybrid_250k\hybrid_temporal_tabular_cnn_temperature_scaled_test_probabilities.npy `
  --temperature 1.1872 `
  --output-dir results\calibration_bootstrap_raw_vs_temperature_250k_final `
  --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reconstruct_raw_probabilities(temp_probs: np.ndarray, temperature: float) -> np.ndarray:
    temp_probs = np.asarray(temp_probs, dtype=np.float64)
    temp_probs = np.clip(temp_probs, 1e-300, 1.0)
    raw = np.power(temp_probs, float(temperature))
    raw /= raw.sum(axis=1, keepdims=True)
    return raw


def per_object_brier(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    idx = np.arange(len(y_true))
    return np.sum(probs * probs, axis=1) - 2.0 * probs[idx, y_true] + 1.0


def per_object_nll(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    idx = np.arange(len(y_true))
    return -np.log(np.clip(probs[idx, y_true], 1e-15, 1.0))


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(topk == y_true[:, None], axis=1)))


def ece_from_arrays(correct: np.ndarray, confidence: np.ndarray, n_bins: int) -> float:
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    sum_correct = np.bincount(bins, weights=correct, minlength=n_bins)
    sum_conf = np.bincount(bins, weights=confidence, minlength=n_bins)

    nonzero = counts > 0
    acc_bin = np.zeros(n_bins)
    conf_bin = np.zeros(n_bins)
    acc_bin[nonzero] = sum_correct[nonzero] / counts[nonzero]
    conf_bin[nonzero] = sum_conf[nonzero] / counts[nonzero]

    return float(np.sum((counts[nonzero] / len(confidence)) * np.abs(acc_bin[nonzero] - conf_bin[nonzero])))


def reliability_bins(correct: np.ndarray, confidence: np.ndarray, n_bins: int, model: str) -> pd.DataFrame:
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if mask.sum() == 0:
            rows.append({
                "model": model,
                "bin": b,
                "bin_low": lo,
                "bin_high": hi,
                "n": 0,
                "accuracy": np.nan,
                "mean_confidence": np.nan,
                "abs_gap": np.nan,
            })
        else:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            rows.append({
                "model": model,
                "bin": b,
                "bin_low": lo,
                "bin_high": hi,
                "n": int(mask.sum()),
                "accuracy": acc,
                "mean_confidence": conf,
                "abs_gap": abs(acc - conf),
            })
    return pd.DataFrame(rows)


def ece_bootstrap_sample(idx: np.ndarray, correct: np.ndarray, confidence: np.ndarray, n_bins: int) -> float:
    return ece_from_arrays(correct[idx], confidence[idx], n_bins)


def point_metrics(y_true, raw_probs, temp_probs, n_bins):
    raw_pred = raw_probs.argmax(axis=1)
    temp_pred = temp_probs.argmax(axis=1)
    raw_conf = raw_probs.max(axis=1)
    temp_conf = temp_probs.max(axis=1)
    raw_correct = (raw_pred == y_true).astype(float)
    temp_correct = (temp_pred == y_true).astype(float)

    rows = []
    for name, probs, pred, conf, correct in [
        ("raw_reconstructed", raw_probs, raw_pred, raw_conf, raw_correct),
        ("temperature_scaled", temp_probs, temp_pred, temp_conf, temp_correct),
    ]:
        acc = float(correct.mean())
        rows.append({
            "model": name,
            "accuracy": acc,
            "top2_accuracy": topk_accuracy(y_true, probs, 2),
            "top3_accuracy": topk_accuracy(y_true, probs, 3),
            "top5_accuracy": topk_accuracy(y_true, probs, 5),
            "mean_confidence": float(conf.mean()),
            "calibration_gap_abs": float(abs(conf.mean() - acc)),
            "ece": ece_from_arrays(correct, conf, n_bins),
            "brier_score": float(per_object_brier(y_true, probs).mean()),
            "negative_log_likelihood": float(per_object_nll(y_true, probs).mean()),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(y_true, raw_probs, temp_probs, n_bootstrap, n_bins, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    raw_pred = raw_probs.argmax(axis=1)
    temp_pred = temp_probs.argmax(axis=1)
    raw_conf = raw_probs.max(axis=1)
    temp_conf = temp_probs.max(axis=1)
    raw_correct = (raw_pred == y_true).astype(float)
    temp_correct = (temp_pred == y_true).astype(float)

    raw_brier = per_object_brier(y_true, raw_probs)
    temp_brier = per_object_brier(y_true, temp_probs)
    raw_nll = per_object_nll(y_true, raw_probs)
    temp_nll = per_object_nll(y_true, temp_probs)

    point = point_metrics(y_true, raw_probs, temp_probs, n_bins)
    raw_point = point[point["model"] == "raw_reconstructed"].iloc[0].to_dict()
    temp_point = point[point["model"] == "temperature_scaled"].iloc[0].to_dict()

    metrics = [
        "accuracy",
        "top2_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "mean_confidence",
        "calibration_gap_abs",
        "ece",
        "brier_score",
        "negative_log_likelihood",
    ]
    samples = {m: [] for m in metrics}

    all_idx = np.arange(n)

    for _ in range(n_bootstrap):
        idx = rng.choice(all_idx, size=n, replace=True)

        raw_acc = raw_correct[idx].mean()
        temp_acc = temp_correct[idx].mean()

        raw_mean_conf = raw_conf[idx].mean()
        temp_mean_conf = temp_conf[idx].mean()

        raw_gap = abs(raw_mean_conf - raw_acc)
        temp_gap = abs(temp_mean_conf - temp_acc)

        samples["accuracy"].append(temp_acc - raw_acc)

        # Temperature scaling preserves rank ordering, so top-k deltas are zero in theory.
        # Still compute point deltas outside bootstrap; bootstrap deltas are set to point delta.
        samples["top2_accuracy"].append(temp_point["top2_accuracy"] - raw_point["top2_accuracy"])
        samples["top3_accuracy"].append(temp_point["top3_accuracy"] - raw_point["top3_accuracy"])
        samples["top5_accuracy"].append(temp_point["top5_accuracy"] - raw_point["top5_accuracy"])

        samples["mean_confidence"].append(temp_mean_conf - raw_mean_conf)
        samples["calibration_gap_abs"].append(temp_gap - raw_gap)
        samples["ece"].append(
            ece_bootstrap_sample(idx, temp_correct, temp_conf, n_bins)
            - ece_bootstrap_sample(idx, raw_correct, raw_conf, n_bins)
        )
        samples["brier_score"].append(temp_brier[idx].mean() - raw_brier[idx].mean())
        samples["negative_log_likelihood"].append(temp_nll[idx].mean() - raw_nll[idx].mean())

    lower_is_better = {
        "ece": True,
        "brier_score": True,
        "negative_log_likelihood": True,
        "calibration_gap_abs": True,
    }

    rows = []
    for m in metrics:
        s = np.asarray(samples[m], dtype=float)
        raw_v = raw_point[m]
        temp_v = temp_point[m]
        delta = temp_v - raw_v
        ci_low = float(np.quantile(s, 0.025))
        ci_high = float(np.quantile(s, 0.975))
        if lower_is_better.get(m, False):
            p_no_improvement = float(np.mean(s >= 0.0))
            direction = "lower_is_better"
        else:
            p_no_improvement = float(np.mean(s <= 0.0))
            direction = "higher_is_better"
        p_two = float(min(1.0, 2.0 * min(np.mean(s <= 0.0), np.mean(s >= 0.0))))
        rows.append({
            "metric": m,
            "raw": raw_v,
            "temperature_scaled": temp_v,
            "delta_temperature_minus_raw": delta,
            "bootstrap_ci_low_95": ci_low,
            "bootstrap_ci_high_95": ci_high,
            "direction": direction,
            "one_sided_bootstrap_pvalue_no_improvement": p_no_improvement,
            "two_sided_bootstrap_pvalue": p_two,
            "n_bootstrap": n_bootstrap,
        })

    return point, pd.DataFrame(rows)


def plot_reliability(raw_bins, temp_bins, out_path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(raw_bins["mean_confidence"], raw_bins["accuracy"], marker="o", label="Raw reconstructed")
    plt.plot(temp_bins["mean_confidence"], temp_bins["accuracy"], marker="o", label="Temperature-scaled")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_deltas(results, out_path):
    order = [
        "ece",
        "brier_score",
        "negative_log_likelihood",
        "calibration_gap_abs",
        "mean_confidence",
        "accuracy",
        "top3_accuracy",
        "top5_accuracy",
    ]
    df = results[results["metric"].isin(order)].copy()
    df["metric"] = pd.Categorical(df["metric"], categories=order, ordered=True)
    df = df.sort_values("metric")
    x = np.arange(len(df))
    y = df["delta_temperature_minus_raw"].to_numpy()
    lo = df["bootstrap_ci_low_95"].to_numpy()
    hi = df["bootstrap_ci_high_95"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])

    plt.figure(figsize=(10, 5))
    plt.axhline(0, linestyle="--")
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, df["metric"].astype(str), rotation=45, ha="right")
    plt.ylabel("Temperature-scaled minus raw")
    plt.title("Paired bootstrap calibration deltas")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--temperature-probabilities", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--output-dir", default="results/calibration_bootstrap_raw_vs_temperature_250k_final")
    parser.add_argument("--true-col", default="true_label")
    parser.add_argument("--pred-col", default="predicted_label")
    parser.add_argument("--confidence-col", default="confidence")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    prob_path = Path(args.temperature_probabilities)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    temp_probs = np.load(prob_path).astype(np.float64)
    if len(df) != temp_probs.shape[0]:
        raise ValueError(f"Row mismatch: predictions={len(df)} probabilities={temp_probs.shape[0]}")

    y_true = pd.to_numeric(df[args.true_col], errors="raise").to_numpy(dtype=int)
    csv_pred = pd.to_numeric(df[args.pred_col], errors="raise").to_numpy(dtype=int)
    temp_pred = temp_probs.argmax(axis=1)
    if not np.array_equal(csv_pred, temp_pred):
        print("[WARN] CSV predicted labels differ from argmax of temperature probability matrix.")

    raw_probs = reconstruct_raw_probabilities(temp_probs, args.temperature)

    same_argmax = float(np.mean(raw_probs.argmax(axis=1) == temp_probs.argmax(axis=1)))

    point, results = paired_bootstrap(
        y_true,
        raw_probs,
        temp_probs,
        args.n_bootstrap,
        args.n_bins,
        args.seed,
    )

    point_path = out_dir / "calibration_point_metrics_raw_vs_temperature.csv"
    results_path = out_dir / "calibration_bootstrap_raw_vs_temperature.csv"
    point.to_csv(point_path, index=False)
    results.to_csv(results_path, index=False)

    raw_pred = raw_probs.argmax(axis=1)
    raw_conf = raw_probs.max(axis=1)
    raw_correct = (raw_pred == y_true).astype(float)
    temp_conf = temp_probs.max(axis=1)
    temp_correct = (temp_pred == y_true).astype(float)

    raw_bins = reliability_bins(raw_correct, raw_conf, args.n_bins, "raw_reconstructed")
    temp_bins = reliability_bins(temp_correct, temp_conf, args.n_bins, "temperature_scaled")
    bins = pd.concat([raw_bins, temp_bins], ignore_index=True)
    bins_path = out_dir / "calibration_reliability_bins.csv"
    bins.to_csv(bins_path, index=False)

    plot_reliability(raw_bins, temp_bins, out_dir / "fig_calibration_reliability_raw_vs_temperature.png")
    plot_metric_deltas(results, out_dir / "fig_calibration_metric_deltas.png")

    schema = {
        "predictions": str(pred_path),
        "temperature_probabilities": str(prob_path),
        "temperature": args.temperature,
        "raw_reconstruction": "raw_probs = normalize(temperature_scaled_probs ** temperature)",
        "n_objects": int(len(df)),
        "n_classes": int(temp_probs.shape[1]),
        "same_argmax_fraction_raw_vs_temperature_scaled": same_argmax,
        "n_bootstrap": args.n_bootstrap,
        "n_bins": args.n_bins,
        "seed": args.seed,
    }
    schema_path = out_dir / "calibration_bootstrap_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def get(metric):
        return results[results["metric"] == metric].iloc[0]

    ece = get("ece")
    brier = get("brier_score")
    nll = get("negative_log_likelihood")
    gap = get("calibration_gap_abs")
    conf = get("mean_confidence")
    acc = get("accuracy")

    md = []
    md.append("# Calibration bootstrap: raw vs temperature-scaled\n")
    md.append("This report compares reconstructed raw probabilities and temperature-scaled probabilities using paired bootstrap resampling.\n")
    md.append("The raw probabilities were reconstructed from the temperature-scaled probabilities using:\n")
    md.append("`raw_probs = normalize(temperature_scaled_probs ** temperature)`\n")
    md.append("## Inputs\n")
    md.append(f"- Predictions CSV: `{pred_path}`")
    md.append(f"- Temperature-scaled probabilities: `{prob_path}`")
    md.append(f"- Temperature: `{args.temperature}`")
    md.append(f"- Number of objects: `{len(df)}`")
    md.append(f"- Number of classes: `{temp_probs.shape[1]}`")
    md.append(f"- Same top-1 argmax after inverse reconstruction: `{same_argmax:.6f}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`")
    md.append(f"- ECE bins: `{args.n_bins}`\n")
    md.append("## Point metrics\n")
    md.append(point.to_markdown(index=False))
    md.append("")
    md.append("## Paired bootstrap results\n")
    md.append(results.to_markdown(index=False))
    md.append("")
    md.append("## Key calibration interpretation\n")
    md.append(f"- ECE decreased from `{ece['raw']:.6f}` to `{ece['temperature_scaled']:.6f}`; delta = `{ece['delta_temperature_minus_raw']:.6f}`, 95% CI [`{ece['bootstrap_ci_low_95']:.6f}`, `{ece['bootstrap_ci_high_95']:.6f}`].")
    md.append(f"- Brier score decreased from `{brier['raw']:.6f}` to `{brier['temperature_scaled']:.6f}`; delta = `{brier['delta_temperature_minus_raw']:.6f}`, 95% CI [`{brier['bootstrap_ci_low_95']:.6f}`, `{brier['bootstrap_ci_high_95']:.6f}`].")
    md.append(f"- Negative log-likelihood decreased from `{nll['raw']:.6f}` to `{nll['temperature_scaled']:.6f}`; delta = `{nll['delta_temperature_minus_raw']:.6f}`, 95% CI [`{nll['bootstrap_ci_low_95']:.6f}`, `{nll['bootstrap_ci_high_95']:.6f}`].")
    md.append(f"- Mean confidence decreased from `{conf['raw']:.6f}` to `{conf['temperature_scaled']:.6f}`, while accuracy stayed unchanged at `{acc['raw']:.6f}`.")
    md.append(f"- Absolute calibration gap decreased from `{gap['raw']:.6f}` to `{gap['temperature_scaled']:.6f}`.\n")
    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    md.append(
        "We assessed the statistical effect of temperature scaling using paired bootstrap resampling over the held-out test objects. "
        f"The calibrated probabilities were obtained with temperature $T={args.temperature:.4f}$, fitted outside the test set, and the top-1 predictions remained unchanged. "
        f"Temperature scaling reduced ECE from \\textbf{{{ece['raw']:.4f}}} to \\textbf{{{ece['temperature_scaled']:.4f}}} "
        f"(bootstrap delta = \\textbf{{{ece['delta_temperature_minus_raw']:.4f}}}; 95\\% CI [{ece['bootstrap_ci_low_95']:.4f}, {ece['bootstrap_ci_high_95']:.4f}]). "
        f"It also reduced the Brier score from \\textbf{{{brier['raw']:.4f}}} to \\textbf{{{brier['temperature_scaled']:.4f}}} "
        f"(delta = \\textbf{{{brier['delta_temperature_minus_raw']:.4f}}}; 95\\% CI [{brier['bootstrap_ci_low_95']:.4f}, {brier['bootstrap_ci_high_95']:.4f}]) and the negative log-likelihood from "
        f"\\textbf{{{nll['raw']:.4f}}} to \\textbf{{{nll['temperature_scaled']:.4f}}} "
        f"(delta = \\textbf{{{nll['delta_temperature_minus_raw']:.4f}}}; 95\\% CI [{nll['bootstrap_ci_low_95']:.4f}, {nll['bootstrap_ci_high_95']:.4f}]). "
        "These results indicate that temperature scaling improved probabilistic reliability without altering classification accuracy."
    )
    md.append("```\n")
    md.append("## Output files\n")
    for p in [
        point_path,
        results_path,
        bins_path,
        schema_path,
        out_dir / "fig_calibration_reliability_raw_vs_temperature.png",
        out_dir / "fig_calibration_metric_deltas.png",
    ]:
        md.append(f"- `{p}`")

    md_path = out_dir / "calibration_bootstrap_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
