#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_ensemble_uncertainty.py

Consolida o teste:
- Ensemble uncertainty/error detection
- Selective prediction / risk-coverage

no arquivo:
results\final_publication\final_publication_summary.md

Uso:
python .\experiments\update_final_publication_with_ensemble_uncertainty.py `
  --uncertainty-dir results\ensemble_uncertainty_selective_error_250k_final `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


SECTION_TITLE = "## Ensemble uncertainty, error detection, and selective prediction"


def fmt(x: float, ndigits: int = 6) -> str:
    return f"{float(x):.{ndigits}f}"


def read_metric(summary_df: pd.DataFrame, metric: str) -> float:
    row = summary_df.loc[summary_df["metric"] == metric]
    if row.empty:
        raise ValueError(f"Metric not found in summary metrics: {metric}")
    return float(row["value"].iloc[0])


def read_error_metric(error_df: pd.DataFrame, metric: str) -> pd.Series:
    row = error_df.loc[error_df["metric"] == metric]
    if row.empty:
        raise ValueError(f"Error-detection metric not found: {metric}")
    return row.iloc[0]


def read_coverage_row(risk_df: pd.DataFrame, target: float) -> pd.Series:
    idx = (risk_df["target_coverage"] - target).abs().idxmin()
    return risk_df.loc[idx]


def copy_outputs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    patterns = ["*.csv", "*.json", "*.md", "*.png"]
    for pattern in patterns:
        for src in src_dir.glob(pattern):
            shutil.copy2(src, dst_dir / src.name)


def build_section(src_dir: Path, final_subdir: Path) -> str:
    summary_csv = src_dir / "ensemble_uncertainty_selective_summary_metrics.csv"
    risk_csv = src_dir / "ensemble_selective_key_coverages_bootstrap.csv"
    error_csv = src_dir / "ensemble_error_detection_bootstrap.csv"
    schema_json = src_dir / "ensemble_uncertainty_selective_schema.json"

    for p in [summary_csv, risk_csv, error_csv, schema_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    summary_df = pd.read_csv(summary_csv)
    risk_df = pd.read_csv(risk_csv)
    error_df = pd.read_csv(error_csv)

    n_objects = int(read_metric(summary_df, "n_objects"))
    accuracy = read_metric(summary_df, "accuracy")
    risk = read_metric(summary_df, "risk")
    mean_confidence = read_metric(summary_df, "mean_confidence")
    mean_uncertainty = read_metric(summary_df, "mean_uncertainty")
    aurc = read_metric(summary_df, "AURC")

    auroc = read_error_metric(error_df, "AUROC")
    auprc = read_error_metric(error_df, "AUPRC")

    c10 = read_coverage_row(risk_df, 0.10)
    c20 = read_coverage_row(risk_df, 0.20)
    c50 = read_coverage_row(risk_df, 0.50)
    c80 = read_coverage_row(risk_df, 0.80)
    c100 = read_coverage_row(risk_df, 1.00)

    compact_summary = pd.DataFrame([
        {"metric": "full_coverage_accuracy", "value": accuracy},
        {"metric": "full_coverage_risk", "value": risk},
        {"metric": "mean_confidence", "value": mean_confidence},
        {"metric": "mean_uncertainty", "value": mean_uncertainty},
        {"metric": "AURC", "value": aurc},
        {
            "metric": "error_detection_AUROC",
            "value": float(auroc["point_estimate"]),
            "ci_low_95": float(auroc["ci_low_95"]),
            "ci_high_95": float(auroc["ci_high_95"]),
        },
        {
            "metric": "error_detection_AUPRC",
            "value": float(auprc["point_estimate"]),
            "ci_low_95": float(auprc["ci_low_95"]),
            "ci_high_95": float(auprc["ci_high_95"]),
        },
    ])

    key_coverages = risk_df.loc[
        risk_df["target_coverage"].isin([0.1, 0.2, 0.5, 0.8, 1.0]),
        [
            "target_coverage",
            "actual_coverage",
            "n_selected",
            "confidence_threshold",
            "risk",
            "risk_ci_low_95",
            "risk_ci_high_95",
            "selective_accuracy",
            "selective_accuracy_ci_low_95",
            "selective_accuracy_ci_high_95",
        ],
    ].copy()

    section = []
    section.append(SECTION_TITLE)
    section.append("")
    section.append("This section consolidates the uncertainty-derived selective prediction and error-detection analysis for the final `ensemble_hybrid_dominant` model.")
    section.append("")
    section.append("### Inputs and protocol")
    section.append("")
    section.append(f"- Model: `ensemble_hybrid_dominant`")
    section.append(f"- Predictions: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_predictions.csv`")
    section.append(f"- Probabilities: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_test_probabilities.npy`")
    section.append(f"- Objects: `{n_objects}`")
    section.append(f"- Error-detection score: `uncertainty = 1 - max_probability`")
    section.append(f"- Selective-prediction policy: retain objects by descending confidence")
    section.append(f"- Bootstrap iterations: `{int(error_df['n_bootstrap'].iloc[0])}`")
    section.append("")
    section.append("### Summary metrics")
    section.append("")
    section.append(compact_summary.to_markdown(index=False))
    section.append("")
    section.append("### Selective prediction at key coverages")
    section.append("")
    section.append(key_coverages.to_markdown(index=False))
    section.append("")
    section.append("### Error detection bootstrap")
    section.append("")
    section.append(error_df.to_markdown(index=False))
    section.append("")
    section.append("### Key interpretation")
    section.append("")
    section.append(f"- Full-coverage accuracy is `{fmt(accuracy)}`, corresponding to risk `{fmt(risk)}`.")
    section.append(f"- At approximately 50% coverage, selective accuracy is `{fmt(c50['selective_accuracy'])}` with 95% CI [`{fmt(c50['selective_accuracy_ci_low_95'])}`, `{fmt(c50['selective_accuracy_ci_high_95'])}`], while risk decreases to `{fmt(c50['risk'])}`.")
    section.append(f"- At approximately 80% coverage, selective accuracy remains `{fmt(c80['selective_accuracy'])}` with 95% CI [`{fmt(c80['selective_accuracy_ci_low_95'])}`, `{fmt(c80['selective_accuracy_ci_high_95'])}`].")
    section.append(f"- Uncertainty detects incorrect predictions with AUROC `{fmt(auroc['point_estimate'])}` and 95% CI [`{fmt(auroc['ci_low_95'])}`, `{fmt(auroc['ci_high_95'])}`].")
    section.append(f"- Uncertainty detects incorrect predictions with AUPRC `{fmt(auprc['point_estimate'])}` and 95% CI [`{fmt(auprc['ci_low_95'])}`, `{fmt(auprc['ci_high_95'])}`].")
    section.append("")
    section.append("### Suggested manuscript wording")
    section.append("")
    section.append("```latex")
    section.append(
        f"We evaluated whether the final ensemble's confidence scores support selective prediction and error detection. "
        f"At full coverage, the ensemble achieved an accuracy of \\textbf{{{accuracy:.4f}}}, corresponding to a risk of \\textbf{{{risk:.4f}}}. "
        f"When retaining approximately 50\\% of the highest-confidence predictions, selective accuracy increased to "
        f"\\textbf{{{float(c50['selective_accuracy']):.4f}}} (95\\% CI [{float(c50['selective_accuracy_ci_low_95']):.4f}, {float(c50['selective_accuracy_ci_high_95']):.4f}]), "
        f"with risk decreasing to \\textbf{{{float(c50['risk']):.4f}}}. "
        f"At approximately 80\\% coverage, selective accuracy remained \\textbf{{{float(c80['selective_accuracy']):.4f}}} "
        f"(95\\% CI [{float(c80['selective_accuracy_ci_low_95']):.4f}, {float(c80['selective_accuracy_ci_high_95']):.4f}]). "
        f"Using uncertainty defined as one minus the maximum class probability, incorrect predictions were identified with an AUROC of "
        f"\\textbf{{{float(auroc['point_estimate']):.4f}}} (95\\% CI [{float(auroc['ci_low_95']):.4f}, {float(auroc['ci_high_95']):.4f}]) "
        f"and an AUPRC of \\textbf{{{float(auprc['point_estimate']):.4f}}} "
        f"(95\\% CI [{float(auprc['ci_low_95']):.4f}, {float(auprc['ci_high_95']):.4f}]). "
        f"These results indicate that ensemble confidence is operationally informative for broker-like triage, allowing high-confidence predictions to be prioritized while low-confidence cases are flagged for review."
    )
    section.append("```")
    section.append("")
    section.append("### Consolidated output files")
    section.append("")
    for name in [
        "ensemble_uncertainty_selective_error_summary.md",
        "ensemble_uncertainty_selective_summary_metrics.csv",
        "ensemble_risk_coverage_curve.csv",
        "ensemble_selective_key_coverages_bootstrap.csv",
        "ensemble_error_detection_bootstrap.csv",
        "ensemble_error_detection_roc_curve.csv",
        "ensemble_error_detection_pr_curve.csv",
        "ensemble_uncertainty_object_level.csv",
        "ensemble_uncertainty_selective_schema.json",
        "fig_ensemble_risk_coverage_curve.png",
        "fig_ensemble_selective_accuracy_curve.png",
        "fig_ensemble_error_detection_roc.png",
        "fig_ensemble_error_detection_pr.png",
    ]:
        if (final_subdir / name).exists() or (src_dir / name).exists():
            section.append(f"- `{final_subdir / name}`")

    section.append("")
    return "\n".join(section)


def insert_or_replace_section(text: str, new_section: str) -> str:
    # Replace existing section if present, stopping before the next level-2 section.
    pattern = rf"{re.escape(SECTION_TITLE)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _m: new_section, text, flags=re.S)

    # Prefer inserting after publication-strength validation if present.
    anchor_titles = [
        "## Publication-strength statistical validation",
        "## Top-k and family-level bootstrap confidence intervals",
        "## Calibration bootstrap: raw vs temperature-scaled",
    ]

    # Insert after the last existing consolidated analysis among anchors.
    insert_pos = None
    for anchor in anchor_titles:
        pattern_anchor = rf"{re.escape(anchor)}\n.*?(?=\n## |\Z)"
        matches = list(re.finditer(pattern_anchor, text, flags=re.S))
        if matches:
            insert_pos = matches[-1].end()

    if insert_pos is not None:
        return text[:insert_pos].rstrip() + "\n\n" + new_section + "\n" + text[insert_pos:].lstrip()

    # Fallback: append at end.
    return text.rstrip() + "\n\n" + new_section + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncertainty-dir", required=True)
    parser.add_argument("--final-publication-dir", required=True)
    args = parser.parse_args()

    src_dir = Path(args.uncertainty_dir)
    final_dir = Path(args.final_publication_dir)
    final_summary = final_dir / "final_publication_summary.md"

    if not src_dir.exists():
        raise FileNotFoundError(f"Uncertainty directory not found: {src_dir}")
    if not final_summary.exists():
        raise FileNotFoundError(f"Final publication summary not found: {final_summary}")

    final_subdir = final_dir / "ensemble_uncertainty_selective_error"
    copy_outputs(src_dir, final_subdir)

    new_section = build_section(src_dir=src_dir, final_subdir=final_subdir)

    text = final_summary.read_text(encoding="utf-8")
    updated = insert_or_replace_section(text, new_section)
    final_summary.write_text(updated, encoding="utf-8")

    print(f"Updated: {final_summary}")
    print(f"Copied outputs to: {final_subdir}")
    print("Inserted/replaced section:")
    print(SECTION_TITLE)


if __name__ == "__main__":
    main()
