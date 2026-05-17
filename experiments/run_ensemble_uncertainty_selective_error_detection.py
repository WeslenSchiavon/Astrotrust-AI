#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_ensemble_uncertainty_selective_error_detection.py

Avalia, para o modelo final ensemble_hybrid_dominant:

1. Selective prediction / risk-coverage:
   - ordena objetos por confiança decrescente
   - calcula risco e acurácia seletiva em diferentes coberturas
   - calcula AURC

2. Error detection using uncertainty:
   - score = 1 - max_probability
   - positivo = prediction_error
   - calcula AUROC e AUPRC com bootstrap CIs

3. Exporta:
   - CSVs
   - resumo .md
   - figuras publication-ready simples

Uso:
python .\experiments\run_ensemble_uncertainty_selective_error_detection.py `
  --predictions results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_predictions.csv `
  --probabilities results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_test_probabilities.npy `
  --output-dir results\ensemble_uncertainty_selective_error_250k_final `
  --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

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


def binary_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """
    AUROC via rank statistic. y_true must be 0/1, positive=1.
    Handles ties by average ranks.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(score)
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)

    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """
    Average precision / AUPRC with positives ranked by descending score.
    Equivalent to step-wise precision-recall AP.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return np.nan

    order = np.argsort(-score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)

    ap = (precision * y_sorted).sum() / n_pos
    return float(ap)


def roc_curve_points(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    s = score[order]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return pd.DataFrame(columns=["threshold", "fpr", "tpr"])

    distinct = np.r_[True, s[1:] != s[:-1]]
    idx = np.where(distinct)[0]

    rows = [{"threshold": np.inf, "fpr": 0.0, "tpr": 0.0}]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    for start in idx:
        # include all items with score >= current threshold
        end = np.searchsorted(-s, -s[start], side="right") - 1
        rows.append({
            "threshold": float(s[start]),
            "fpr": float(fp[end] / n_neg),
            "tpr": float(tp[end] / n_pos),
        })
    rows.append({"threshold": -np.inf, "fpr": 1.0, "tpr": 1.0})
    return pd.DataFrame(rows).drop_duplicates(subset=["fpr", "tpr"])


def pr_curve_points(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    s = score[order]

    n_pos = int(y.sum())
    if n_pos == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall"])

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # Downsample for file/plot readability if huge.
    max_points = 5000
    if len(y) > max_points:
        idx = np.linspace(0, len(y) - 1, max_points).astype(int)
    else:
        idx = np.arange(len(y))

    return pd.DataFrame({
        "threshold": s[idx],
        "precision": precision[idx],
        "recall": recall[idx],
    })


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray, coverages: np.ndarray) -> pd.DataFrame:
    n = len(confidence)
    order = np.argsort(-confidence, kind="mergesort")
    corr_sorted = correct[order].astype(float)
    cum_correct = np.cumsum(corr_sorted)

    rows = []
    for cov in coverages:
        k = int(np.ceil(cov * n))
        k = min(max(k, 1), n)
        acc = float(cum_correct[k - 1] / k)
        risk = float(1.0 - acc)
        threshold = float(confidence[order[k - 1]])
        rows.append({
            "coverage": float(k / n),
            "n_selected": int(k),
            "confidence_threshold": threshold,
            "selective_accuracy": acc,
            "risk": risk,
        })
    return pd.DataFrame(rows)


def interpolate_at_coverage(curve: pd.DataFrame, target: float) -> pd.Series:
    idx = (curve["coverage"] - target).abs().idxmin()
    return curve.loc[idx]


def bootstrap_error_detection(
    y_error: np.ndarray,
    uncertainty: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_error)
    idx_all = np.arange(n)

    aurocs = np.empty(n_bootstrap, dtype=float)
    auprcs = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.choice(idx_all, size=n, replace=True)
        yb = y_error[idx]
        sb = uncertainty[idx]
        aurocs[b] = binary_roc_auc(yb, sb)
        auprcs[b] = binary_average_precision(yb, sb)

    auroc = binary_roc_auc(y_error, uncertainty)
    auprc = binary_average_precision(y_error, uncertainty)

    return pd.DataFrame([
        {
            "task": "detect_incorrect_predictions",
            "positive_label": "prediction_error",
            "score": "uncertainty_1_minus_confidence",
            "metric": "AUROC",
            "point_estimate": auroc,
            "ci_low_95": float(np.nanquantile(aurocs, 0.025)),
            "ci_high_95": float(np.nanquantile(aurocs, 0.975)),
            "bootstrap_mean": float(np.nanmean(aurocs)),
            "bootstrap_std": float(np.nanstd(aurocs, ddof=1)),
            "n_objects": n,
            "n_bootstrap": n_bootstrap,
        },
        {
            "task": "detect_incorrect_predictions",
            "positive_label": "prediction_error",
            "score": "uncertainty_1_minus_confidence",
            "metric": "AUPRC",
            "point_estimate": auprc,
            "ci_low_95": float(np.nanquantile(auprcs, 0.025)),
            "ci_high_95": float(np.nanquantile(auprcs, 0.975)),
            "bootstrap_mean": float(np.nanmean(auprcs)),
            "bootstrap_std": float(np.nanstd(auprcs, ddof=1)),
            "n_objects": n,
            "n_bootstrap": n_bootstrap,
        },
    ])


def bootstrap_risk_at_coverages(
    confidence: np.ndarray,
    correct: np.ndarray,
    targets: Tuple[float, ...],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(confidence)
    idx_all = np.arange(n)

    point_curve = risk_coverage_curve(confidence, correct, np.array(targets))
    rows = []

    for target in targets:
        boot_risk = np.empty(n_bootstrap, dtype=float)
        boot_acc = np.empty(n_bootstrap, dtype=float)

        for b in range(n_bootstrap):
            idx = rng.choice(idx_all, size=n, replace=True)
            c = confidence[idx]
            corr = correct[idx]
            cur = risk_coverage_curve(c, corr, np.array([target]))
            boot_risk[b] = cur.iloc[0]["risk"]
            boot_acc[b] = cur.iloc[0]["selective_accuracy"]

        p = point_curve.iloc[list(targets).index(target)]
        rows.append({
            "target_coverage": target,
            "actual_coverage": float(p["coverage"]),
            "n_selected": int(p["n_selected"]),
            "confidence_threshold": float(p["confidence_threshold"]),
            "risk": float(p["risk"]),
            "risk_ci_low_95": float(np.quantile(boot_risk, 0.025)),
            "risk_ci_high_95": float(np.quantile(boot_risk, 0.975)),
            "selective_accuracy": float(p["selective_accuracy"]),
            "selective_accuracy_ci_low_95": float(np.quantile(boot_acc, 0.025)),
            "selective_accuracy_ci_high_95": float(np.quantile(boot_acc, 0.975)),
            "n_bootstrap": n_bootstrap,
        })

    return pd.DataFrame(rows)


def plot_risk_coverage(curve: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(curve["coverage"], curve["risk"], linewidth=1.8)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title("Risk-coverage curve")
    plt.xlim(0, 1)
    plt.ylim(0, max(0.01, float(curve["risk"].max()) * 1.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_selective_accuracy(curve: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(curve["coverage"], curve["selective_accuracy"], linewidth=1.8)
    plt.xlabel("Coverage")
    plt.ylabel("Selective accuracy")
    plt.title("Selective prediction")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc(roc: pd.DataFrame, auroc: float, out_path: Path) -> None:
    plt.figure(figsize=(5.2, 5.0))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(roc["fpr"], roc["tpr"], linewidth=1.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Error detection ROC (AUROC={auroc:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pr(pr: pd.DataFrame, auprc: float, baseline: float, out_path: Path) -> None:
    plt.figure(figsize=(5.2, 5.0))
    plt.axhline(baseline, linestyle="--", linewidth=1)
    plt.plot(pr["recall"], pr["precision"], linewidth=1.8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Error detection PR (AUPRC={auprc:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--output-dir", default="results/ensemble_uncertainty_selective_error_250k_final")
    parser.add_argument("--true-col", default="true_label")
    parser.add_argument("--pred-col", default="predicted_label")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage-grid-size", type=int, default=101)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    prob_path = Path(args.probabilities)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    probs = normalize_probs(np.load(prob_path))

    if len(df) != probs.shape[0]:
        raise ValueError(f"Row mismatch: predictions={len(df)} probabilities={probs.shape[0]}")
    for col in [args.true_col, args.pred_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column `{col}` in predictions CSV.")

    y_true = df[args.true_col].to_numpy().astype(int)
    pred_from_probs = probs.argmax(axis=1).astype(int)
    pred_csv = df[args.pred_col].to_numpy().astype(int)

    if not np.array_equal(pred_from_probs, pred_csv):
        mismatch = int((pred_from_probs != pred_csv).sum())
        print(f"WARNING: predicted_label differs from argmax(probabilities) for {mismatch} objects. Using argmax(probabilities).")

    confidence = probs.max(axis=1)
    uncertainty = 1.0 - confidence
    correct = (pred_from_probs == y_true).astype(int)
    error = 1 - correct

    n = len(y_true)
    full_accuracy = float(correct.mean())
    full_risk = float(1.0 - full_accuracy)
    error_rate = float(error.mean())
    mean_confidence = float(confidence.mean())
    mean_uncertainty = float(uncertainty.mean())

    coverages = np.linspace(0.01, 1.0, args.coverage_grid_size)
    curve = risk_coverage_curve(confidence, correct, coverages)
    aurc = float(np.trapz(curve["risk"], curve["coverage"]))

    risk_key = bootstrap_risk_at_coverages(
        confidence=confidence,
        correct=correct,
        targets=(0.10, 0.20, 0.50, 0.80, 1.00),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
    )

    error_boot = bootstrap_error_detection(
        y_error=error,
        uncertainty=uncertainty,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    auroc = float(error_boot[error_boot["metric"] == "AUROC"]["point_estimate"].iloc[0])
    auprc = float(error_boot[error_boot["metric"] == "AUPRC"]["point_estimate"].iloc[0])

    roc = roc_curve_points(error, uncertainty)
    pr = pr_curve_points(error, uncertainty)

    object_level = pd.DataFrame({
        "object_id": df["object_id"] if "object_id" in df.columns else np.arange(n),
        "true_label": y_true,
        "predicted_label": pred_from_probs,
        "correct": correct.astype(bool),
        "error": error.astype(bool),
        "confidence": confidence,
        "uncertainty": uncertainty,
    })

    summary_metrics = pd.DataFrame([
        {"metric": "n_objects", "value": n},
        {"metric": "accuracy", "value": full_accuracy},
        {"metric": "risk", "value": full_risk},
        {"metric": "error_rate", "value": error_rate},
        {"metric": "mean_confidence", "value": mean_confidence},
        {"metric": "mean_uncertainty", "value": mean_uncertainty},
        {"metric": "AURC", "value": aurc},
        {"metric": "error_detection_AUROC", "value": auroc},
        {"metric": "error_detection_AUPRC", "value": auprc},
    ])

    summary_path_csv = out_dir / "ensemble_uncertainty_selective_summary_metrics.csv"
    curve_path = out_dir / "ensemble_risk_coverage_curve.csv"
    risk_key_path = out_dir / "ensemble_selective_key_coverages_bootstrap.csv"
    error_boot_path = out_dir / "ensemble_error_detection_bootstrap.csv"
    roc_path = out_dir / "ensemble_error_detection_roc_curve.csv"
    pr_path = out_dir / "ensemble_error_detection_pr_curve.csv"
    object_path = out_dir / "ensemble_uncertainty_object_level.csv"

    summary_metrics.to_csv(summary_path_csv, index=False)
    curve.to_csv(curve_path, index=False)
    risk_key.to_csv(risk_key_path, index=False)
    error_boot.to_csv(error_boot_path, index=False)
    roc.to_csv(roc_path, index=False)
    pr.to_csv(pr_path, index=False)
    object_level.to_csv(object_path, index=False)

    plot_risk_coverage(curve, out_dir / "fig_ensemble_risk_coverage_curve.png")
    plot_selective_accuracy(curve, out_dir / "fig_ensemble_selective_accuracy_curve.png")
    plot_roc(roc, auroc, out_dir / "fig_ensemble_error_detection_roc.png")
    plot_pr(pr, auprc, error_rate, out_dir / "fig_ensemble_error_detection_pr.png")

    schema = {
        "predictions": str(pred_path),
        "probabilities": str(prob_path),
        "n_objects": n,
        "n_classes": int(probs.shape[1]),
        "score": "uncertainty = 1 - max_probability",
        "positive_label_for_error_detection": "prediction_error",
        "selective_prediction_sort": "descending confidence",
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
    }
    schema_path = out_dir / "ensemble_uncertainty_selective_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def row_metric(metric: str) -> pd.Series:
        return error_boot[error_boot["metric"] == metric].iloc[0]

    auroc_row = row_metric("AUROC")
    auprc_row = row_metric("AUPRC")

    def coverage_row(target: float) -> pd.Series:
        return risk_key.loc[(risk_key["target_coverage"] - target).abs().idxmin()]

    c50 = coverage_row(0.50)
    c80 = coverage_row(0.80)

    md = []
    md.append("# Ensemble uncertainty, error detection, and selective prediction\n")
    md.append("This report evaluates uncertainty-derived error detection and selective prediction for the final `ensemble_hybrid_dominant` model.\n")
    md.append("## Inputs\n")
    md.append(f"- Predictions CSV: `{pred_path}`")
    md.append(f"- Probability matrix: `{prob_path}`")
    md.append(f"- Objects: `{n}`")
    md.append(f"- Classes: `{probs.shape[1]}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`\n")

    md.append("## Summary metrics\n")
    md.append(summary_metrics.to_markdown(index=False))
    md.append("")

    md.append("## Error detection bootstrap\n")
    md.append(error_boot.to_markdown(index=False))
    md.append("")

    md.append("## Selective prediction key coverages\n")
    md.append(risk_key.to_markdown(index=False))
    md.append("")

    md.append("## Key interpretation\n")
    md.append(f"- Full-coverage accuracy is `{full_accuracy:.6f}`, corresponding to risk `{full_risk:.6f}`.")
    md.append(f"- At approximately 50% coverage, selective accuracy is `{c50['selective_accuracy']:.6f}` (95% CI [`{c50['selective_accuracy_ci_low_95']:.6f}`, `{c50['selective_accuracy_ci_high_95']:.6f}`]) and risk is `{c50['risk']:.6f}`.")
    md.append(f"- At approximately 80% coverage, selective accuracy is `{c80['selective_accuracy']:.6f}` (95% CI [`{c80['selective_accuracy_ci_low_95']:.6f}`, `{c80['selective_accuracy_ci_high_95']:.6f}`]) and risk is `{c80['risk']:.6f}`.")
    md.append(f"- Uncertainty detects incorrect predictions with AUROC `{auroc_row['point_estimate']:.6f}` (95% CI [`{auroc_row['ci_low_95']:.6f}`, `{auroc_row['ci_high_95']:.6f}`]).")
    md.append(f"- Uncertainty detects incorrect predictions with AUPRC `{auprc_row['point_estimate']:.6f}` (95% CI [`{auprc_row['ci_low_95']:.6f}`, `{auprc_row['ci_high_95']:.6f}`]).\n")

    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    md.append(
        f"We evaluated whether the final ensemble's confidence scores support selective prediction and error detection. "
        f"At full coverage, the ensemble achieved an accuracy of \\textbf{{{full_accuracy:.4f}}}, corresponding to a risk of \\textbf{{{full_risk:.4f}}}. "
        f"When retaining approximately 50\\% of the highest-confidence predictions, selective accuracy increased to "
        f"\\textbf{{{c50['selective_accuracy']:.4f}}} (95\\% CI [{c50['selective_accuracy_ci_low_95']:.4f}, {c50['selective_accuracy_ci_high_95']:.4f}]), "
        f"with risk decreasing to \\textbf{{{c50['risk']:.4f}}}. "
        f"At approximately 80\\% coverage, selective accuracy remained \\textbf{{{c80['selective_accuracy']:.4f}}} "
        f"(95\\% CI [{c80['selective_accuracy_ci_low_95']:.4f}, {c80['selective_accuracy_ci_high_95']:.4f}]). "
        f"Using uncertainty defined as one minus the maximum class probability, incorrect predictions were identified with an AUROC of "
        f"\\textbf{{{auroc_row['point_estimate']:.4f}}} (95\\% CI [{auroc_row['ci_low_95']:.4f}, {auroc_row['ci_high_95']:.4f}]) "
        f"and an AUPRC of \\textbf{{{auprc_row['point_estimate']:.4f}}} "
        f"(95\\% CI [{auprc_row['ci_low_95']:.4f}, {auprc_row['ci_high_95']:.4f}]). "
        f"These results indicate that ensemble confidence is operationally informative for broker-like triage, allowing high-confidence predictions to be prioritized while low-confidence cases are flagged for review."
    )
    md.append("```\n")

    md.append("## Output files\n")
    for p in [
        summary_path_csv,
        curve_path,
        risk_key_path,
        error_boot_path,
        roc_path,
        pr_path,
        object_path,
        schema_path,
        out_dir / "fig_ensemble_risk_coverage_curve.png",
        out_dir / "fig_ensemble_selective_accuracy_curve.png",
        out_dir / "fig_ensemble_error_detection_roc.png",
        out_dir / "fig_ensemble_error_detection_pr.png",
    ]:
        md.append(f"- `{p}`")

    summary_md = out_dir / "ensemble_uncertainty_selective_error_summary.md"
    summary_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")
    print(f"Summary: {summary_md}")


if __name__ == "__main__":
    main()
