#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_calibration_safe_ensemble.py

Consolida a calibração calibration-safe do ensemble no arquivo:
results\final_publication\final_publication_summary.md

Uso:
python .\experiments\update_final_publication_with_calibration_safe_ensemble.py `
  --calibration-dir results\ensemble_temperature_calibration_safe_rebuild_250k_final `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd

SECTION_TITLE = "## Calibration-safe ensemble temperature scaling"


def fmt(x, ndigits: int = 6) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "NA"


def copy_outputs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.csv", "*.json", "*.md", "*.png", "*.npy"]:
        for src in src_dir.glob(pattern):
            shutil.copy2(src, dst_dir / src.name)


def metric_row(df: pd.DataFrame, metric: str) -> pd.Series:
    row = df[df["metric"].astype(str).eq(metric)]
    if row.empty:
        raise ValueError(f"Metric not found: {metric}")
    return row.iloc[0]


def point_row(df: pd.DataFrame, split: str, variant: str) -> pd.Series:
    row = df[(df["split"].astype(str).eq(split)) & (df["variant"].astype(str).eq(variant))]
    if row.empty:
        raise ValueError(f"Point metric row not found: split={split}, variant={variant}")
    return row.iloc[0]


def build_section(src_dir: Path, dst_dir: Path) -> str:
    point_csv = src_dir / "calibration_safe_ensemble_temperature_point_metrics.csv"
    boot_csv = src_dir / "calibration_safe_ensemble_temperature_bootstrap_deltas.csv"
    model_csv = src_dir / "calibration_safe_rebuilt_model_test_metrics.csv"
    schema_json = src_dir / "calibration_safe_ensemble_temperature_schema.json"

    for p in [point_csv, boot_csv, model_csv, schema_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    point = pd.read_csv(point_csv)
    boot = pd.read_csv(boot_csv)
    models = pd.read_csv(model_csv)
    schema = json.loads(schema_json.read_text(encoding="utf-8"))

    val_raw = point_row(point, "validation", "raw")
    val_temp = point_row(point, "validation", "temperature_scaled")
    test_raw = point_row(point, "test", "raw")
    test_temp = point_row(point, "test", "temperature_scaled")

    d_acc = metric_row(boot, "accuracy")
    d_ece = metric_row(boot, "ece")
    d_brier = metric_row(boot, "brier_score")
    d_nll = metric_row(boot, "negative_log_likelihood")
    d_gap = metric_row(boot, "calibration_gap_abs")

    official = schema.get("official_comparison", {})
    split = schema.get("split", {})
    weights = schema.get("weights", {})
    temperature = float(schema.get("temperature"))

    compact_cols = [
        "model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "top3_accuracy", "top5_accuracy", "mean_confidence", "ece",
        "brier_score", "negative_log_likelihood",
    ]
    compact_cols = [c for c in compact_cols if c in models.columns]
    model_compact = models[compact_cols].copy()

    lines = []
    lines.append(SECTION_TITLE)
    lines.append("")
    lines.append("This section consolidates the calibration-safe temperature-scaling analysis for an ensemble variant using the same dominant weights as the final `ensemble_hybrid_dominant` configuration.")
    lines.append("")
    lines.append("### Protocol")
    lines.append("")
    lines.append("- The tabular members were trained only on the training split.")
    lines.append("- The validation split was used exclusively to fit the temperature parameter.")
    lines.append("- The held-out test split was used only for final evaluation.")
    lines.append("- No temperature parameter was fitted on the test set.")
    lines.append(f"- Train objects: `{int(split.get('n_train', 0))}`")
    lines.append(f"- Validation objects: `{int(split.get('n_validation', 0))}`")
    lines.append(f"- Test objects: `{int(split.get('n_test', 0))}`")
    lines.append(f"- Dominant ensemble weights: `{weights}`")
    lines.append(f"- Fitted temperature: `{temperature:.6f}`")
    lines.append(f"- Bootstrap iterations: `{int(schema.get('n_bootstrap', d_ece.get('n_bootstrap', 0)))}`")
    lines.append("")
    lines.append("### Split metadata verification")
    lines.append("")
    lines.append(f"- Train split matches existing split metadata: `{split.get('train_object_id_set_matches_existing_split_metadata')}`")
    lines.append(f"- Validation split matches existing split metadata: `{split.get('validation_object_id_set_matches_existing_split_metadata')}`")
    lines.append(f"- Test split matches existing split metadata: `{split.get('test_object_id_set_matches_existing_split_metadata')}`")
    lines.append("")
    lines.append("### Comparison against the previous official ensemble probabilities")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|:--|--:|")
    lines.append(f"| Official accuracy from probabilities | `{fmt(official.get('official_accuracy_from_probs'))}` |")
    lines.append(f"| Calibration-safe rebuilt accuracy from probabilities | `{fmt(official.get('rebuilt_accuracy_from_probs'))}` |")
    lines.append(f"| Top-1 agreement with previous official probabilities | `{fmt(official.get('top1_agreement_fraction'))}` |")
    lines.append(f"| Mean absolute probability difference | `{fmt(official.get('mean_abs_probability_difference'))}` |")
    lines.append(f"| Max absolute probability difference | `{fmt(official.get('max_abs_probability_difference'))}` |")
    lines.append("")
    lines.append("> The calibration-safe rebuild is not used to replace the headline classifier metrics. It is used to evaluate ensemble calibration without validation/test leakage. Differences from the previous official ensemble are expected because the rebuilt tabular members are trained only on the training split.")
    lines.append("")
    lines.append("### Rebuilt model test metrics")
    lines.append("")
    lines.append(model_compact.to_markdown(index=False))
    lines.append("")
    lines.append("### Ensemble raw versus temperature-scaled metrics")
    lines.append("")
    lines.append(point.to_markdown(index=False))
    lines.append("")
    lines.append("### Test-set bootstrap deltas")
    lines.append("")
    lines.append(boot.to_markdown(index=False))
    lines.append("")
    lines.append("### Key interpretation")
    lines.append("")
    lines.append(f"- Rebuilt raw ensemble test accuracy: `{fmt(test_raw['accuracy'])}`.")
    lines.append(f"- Temperature-scaled ensemble test accuracy: `{fmt(test_temp['accuracy'])}`.")
    lines.append(f"- Accuracy delta: `{fmt(d_acc['delta_temperature_minus_raw'])}` with 95% CI [`{fmt(d_acc['bootstrap_ci_low_95'])}`, `{fmt(d_acc['bootstrap_ci_high_95'])}`].")
    lines.append(f"- ECE changed from `{fmt(test_raw['ece'])}` to `{fmt(test_temp['ece'])}`; delta `{fmt(d_ece['delta_temperature_minus_raw'])}` with 95% CI [`{fmt(d_ece['bootstrap_ci_low_95'])}`, `{fmt(d_ece['bootstrap_ci_high_95'])}`].")
    lines.append(f"- Calibration gap changed from `{fmt(test_raw['calibration_gap_abs'])}` to `{fmt(test_temp['calibration_gap_abs'])}`; delta `{fmt(d_gap['delta_temperature_minus_raw'])}` with 95% CI [`{fmt(d_gap['bootstrap_ci_low_95'])}`, `{fmt(d_gap['bootstrap_ci_high_95'])}`].")
    lines.append(f"- Brier score changed from `{fmt(test_raw['brier_score'])}` to `{fmt(test_temp['brier_score'])}`; delta `{fmt(d_brier['delta_temperature_minus_raw'])}` with 95% CI [`{fmt(d_brier['bootstrap_ci_low_95'])}`, `{fmt(d_brier['bootstrap_ci_high_95'])}`].")
    lines.append(f"- NLL changed from `{fmt(test_raw['negative_log_likelihood'])}` to `{fmt(test_temp['negative_log_likelihood'])}`; delta `{fmt(d_nll['delta_temperature_minus_raw'])}` with 95% CI [`{fmt(d_nll['bootstrap_ci_low_95'])}`, `{fmt(d_nll['bootstrap_ci_high_95'])}`].")
    lines.append("")
    lines.append("### Suggested manuscript wording")
    lines.append("")
    lines.append("```latex")
    lines.append(
        f"To evaluate calibration for the ensemble without test-set leakage, we rebuilt a calibration-safe "
        f"\\texttt{{ensemble\\_hybrid\\_dominant}} variant using the same dominant weights as the final system "
        f"(0.70 hybrid, 0.15 LightGBM baseline, 0.10 regularized LightGBM, and 0.05 XGBoost). "
        f"In this analysis, tabular members were trained only on the training split, the validation split was used exclusively to fit the temperature parameter, "
        f"and the held-out test split was used only for evaluation. The fitted temperature was \\textbf{{{temperature:.4f}}}. "
        f"On the test set, ECE changed from \\textbf{{{float(test_raw['ece']):.4f}}} to \\textbf{{{float(test_temp['ece']):.4f}}} "
        f"(bootstrap delta = \\textbf{{{float(d_ece['delta_temperature_minus_raw']):+.4f}}}; "
        f"95\\% CI [{float(d_ece['bootstrap_ci_low_95']):.4f}, {float(d_ece['bootstrap_ci_high_95']):.4f}]). "
        f"The Brier score changed from \\textbf{{{float(test_raw['brier_score']):.4f}}} to \\textbf{{{float(test_temp['brier_score']):.4f}}} "
        f"(delta = \\textbf{{{float(d_brier['delta_temperature_minus_raw']):+.4f}}}; "
        f"95\\% CI [{float(d_brier['bootstrap_ci_low_95']):.4f}, {float(d_brier['bootstrap_ci_high_95']):.4f}]), "
        f"and the negative log-likelihood changed from \\textbf{{{float(test_raw['negative_log_likelihood']):.4f}}} to "
        f"\\textbf{{{float(test_temp['negative_log_likelihood']):.4f}}} "
        f"(delta = \\textbf{{{float(d_nll['delta_temperature_minus_raw']):+.4f}}}; "
        f"95\\% CI [{float(d_nll['bootstrap_ci_low_95']):.4f}, {float(d_nll['bootstrap_ci_high_95']):.4f}]). "
        f"The top-1 predictions were unchanged by temperature scaling, preserving test accuracy at \\textbf{{{float(test_raw['accuracy']):.4f}}}."
    )
    lines.append("```")
    lines.append("")
    lines.append("### Consolidated output files")
    lines.append("")
    for name in [
        "calibration_safe_ensemble_temperature_summary.md",
        "calibration_safe_rebuilt_model_test_metrics.csv",
        "calibration_safe_ensemble_temperature_point_metrics.csv",
        "calibration_safe_ensemble_temperature_bootstrap_deltas.csv",
        "calibration_safe_ensemble_test_predictions_raw_vs_temperature.csv",
        "calibration_safe_ensemble_temperature_schema.json",
        "fig_calibration_safe_ensemble_reliability.png",
        "fig_calibration_safe_ensemble_metric_comparison.png",
        "ensemble_hybrid_dominant_validation_probabilities.npy",
        "ensemble_hybrid_dominant_test_probabilities.npy",
        "ensemble_hybrid_dominant_validation_temperature_scaled_probabilities.npy",
        "ensemble_hybrid_dominant_test_temperature_scaled_probabilities.npy",
    ]:
        if (src_dir / name).exists() or (dst_dir / name).exists():
            lines.append(f"- `{dst_dir / name}`")
    lines.append("")
    return "\n".join(lines)


def update_authoritative_index(text: str) -> str:
    old = "| Raw ensemble calibration | `ensemble_hybrid_dominant` | not yet consolidated in this package | pending/diagnostic |"
    new = "| Calibration-safe ensemble temperature scaling | calibration-safe `ensemble_hybrid_dominant` rebuild | T = `0.950256`; test ECE `0.009935` -> `0.006706`; Brier `0.404509` -> `0.404339`; NLL `0.898975` -> `0.898082`; accuracy unchanged at `0.683690` | auxiliary/main calibration evidence |"
    if old in text:
        text = text.replace(old, new)

    old2 = "| Temperature scaling | standalone `hybrid_temporal_tabular_cnn` | ECE reduced from `0.027016` to `0.014727`; Brier from `0.416519` to `0.415015`; NLL from `0.929091` to `0.919960` | auxiliary calibration result |"
    new2 = "| Standalone hybrid temperature scaling | standalone `hybrid_temporal_tabular_cnn` | ECE reduced from `0.027016` to `0.014727`; Brier from `0.416519` to `0.415015`; NLL from `0.929091` to `0.919960` | auxiliary supporting calibration result |"
    if old2 in text:
        text = text.replace(old2, new2)
    return text


def insert_or_replace(text: str, section: str) -> str:
    pattern = rf"{re.escape(SECTION_TITLE)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _m: section, text, flags=re.S)

    anchors = [
        "## Model-usage audit and methodological consistency",
        "## Ensemble hierarchical upper-bound analysis",
        "## Calibration bootstrap: raw vs temperature-scaled",
    ]
    for anchor in anchors:
        pat = rf"{re.escape(anchor)}\n.*?(?=\n## |\Z)"
        matches = list(re.finditer(pat, text, flags=re.S))
        if matches:
            pos = matches[-1].end()
            return text[:pos].rstrip() + "\n\n" + section + "\n" + text[pos:].lstrip()
    return text.rstrip() + "\n\n" + section + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--final-publication-dir", required=True)
    args = parser.parse_args()

    src_dir = Path(args.calibration_dir)
    final_dir = Path(args.final_publication_dir)
    final_summary = final_dir / "final_publication_summary.md"
    dst_dir = final_dir / "calibration_safe_ensemble_temperature"

    if not src_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {src_dir}")
    if not final_summary.exists():
        raise FileNotFoundError(f"Final publication summary not found: {final_summary}")

    copy_outputs(src_dir, dst_dir)
    section = build_section(src_dir, dst_dir)
    text = final_summary.read_text(encoding="utf-8")
    text = update_authoritative_index(text)
    updated = insert_or_replace(text, section)
    final_summary.write_text(updated, encoding="utf-8")

    print(f"Updated: {final_summary}")
    print(f"Copied outputs to: {dst_dir}")
    print("Inserted/replaced section:")
    print(SECTION_TITLE)


if __name__ == "__main__":
    main()
