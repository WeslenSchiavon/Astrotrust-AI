#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_ensemble_hierarchical_upper_bound.py

Consolida a análise hierarchical upper-bound do ensemble final no:
results\final_publication\final_publication_summary.md

Uso:
python .\experiments\update_final_publication_with_ensemble_hierarchical_upper_bound.py `
  --hierarchical-dir results\ensemble_hierarchical_upper_bound_250k_final `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


SECTION_TITLE = "## Ensemble hierarchical upper-bound analysis"


def fmt(x: float, ndigits: int = 6) -> str:
    return f"{float(x):.{ndigits}f}"


def copy_outputs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.csv", "*.json", "*.md", "*.png"]:
        for src in src_dir.glob(pattern):
            shutil.copy2(src, dst_dir / src.name)


def get_metric(metrics: pd.DataFrame, strategy: str, metric: str) -> pd.Series:
    row = metrics[(metrics["strategy"] == strategy) & (metrics["metric"] == metric)]
    if row.empty:
        raise ValueError(f"Missing metric row: strategy={strategy}, metric={metric}")
    return row.iloc[0]


def get_delta(deltas: pd.DataFrame, comparison: str, metric: str) -> pd.Series:
    row = deltas[
        (deltas["comparison_strategy"] == comparison)
        & (deltas["metric"] == metric)
    ]
    if row.empty:
        raise ValueError(f"Missing delta row: comparison={comparison}, metric={metric}")
    return row.iloc[0]


def build_section(src_dir: Path, dst_dir: Path) -> str:
    metrics_csv = src_dir / "hierarchical_bootstrap_accuracy.csv"
    deltas_csv = src_dir / "hierarchical_paired_bootstrap_deltas_vs_original.csv"
    point_csv = src_dir / "hierarchical_strategy_point_metrics.csv"
    family_csv = src_dir / "hierarchical_per_family_metrics.csv"
    schema_json = src_dir / "hierarchical_upper_bound_schema.json"

    for p in [metrics_csv, deltas_csv, point_csv, family_csv, schema_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    metrics = pd.read_csv(metrics_csv)
    deltas = pd.read_csv(deltas_csv)
    point = pd.read_csv(point_csv)
    family = pd.read_csv(family_csv)
    schema = json.loads(schema_json.read_text(encoding="utf-8"))

    orig_fine = get_metric(metrics, "original_top1", "fine_accuracy")
    fm_fine = get_metric(metrics, "family_mass_then_subclass", "fine_accuracy")
    oracle_fine = get_metric(metrics, "oracle_true_family_then_subclass", "fine_accuracy")

    orig_family = get_metric(metrics, "original_top1", "family_accuracy")
    fm_family = get_metric(metrics, "family_mass_then_subclass", "family_accuracy")
    oracle_family = get_metric(metrics, "oracle_true_family_then_subclass", "family_accuracy")

    fm_delta_fine = get_delta(deltas, "family_mass_then_subclass", "fine_accuracy")
    fm_delta_family = get_delta(deltas, "family_mass_then_subclass", "family_accuracy")
    oracle_delta_fine = get_delta(deltas, "oracle_true_family_then_subclass", "fine_accuracy")
    oracle_delta_family = get_delta(deltas, "oracle_true_family_then_subclass", "family_accuracy")

    compact_metrics = metrics.copy()
    compact_deltas = deltas.copy()

    family_compact = family[
        [
            "family", "support",
            "original_top1_family_recall",
            "original_top1_fine_accuracy_within_true_family",
            "family_mass_then_subclass_family_recall",
            "family_mass_then_subclass_fine_accuracy_within_true_family",
            "oracle_true_family_then_subclass_fine_accuracy_within_true_family",
        ]
    ].copy()

    lines = []
    lines.append(SECTION_TITLE)
    lines.append("")
    lines.append("This section consolidates the hierarchical headroom analysis for the final `ensemble_hybrid_dominant` model.")
    lines.append("")
    lines.append("### Inputs and protocol")
    lines.append("")
    lines.append("- Model: `ensemble_hybrid_dominant`")
    lines.append("- Predictions: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_predictions.csv`")
    lines.append("- Probabilities: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_test_probabilities.npy`")
    lines.append("- Family mapping source: `results\\ensemble_topk_family_bootstrap_ci_250k_final\\topk_family_object_level_indicators.csv`")
    lines.append(f"- Objects: `{int(schema.get('n_objects', 57058))}`")
    lines.append(f"- Classes: `{int(schema.get('n_classes', 32))}`")
    lines.append(f"- Families: `{int(schema.get('n_families', 11))}`")
    lines.append(f"- Bootstrap iterations: `{int(schema.get('bootstrap', {}).get('n_bootstrap', orig_fine['n_bootstrap']))}`")
    lines.append("- `original_top1`: standard ensemble fine-grained top-1 prediction.")
    lines.append("- `family_mass_then_subclass`: select the family with the largest summed probability mass, then the highest-probability subclass inside that family.")
    lines.append("- `oracle_true_family_then_subclass`: restrict prediction to the true family; this is an upper bound, not a deployable model.")
    lines.append("")
    lines.append("### Bootstrap accuracy summary")
    lines.append("")
    lines.append(compact_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("### Paired bootstrap deltas versus original top-1")
    lines.append("")
    lines.append(compact_deltas.to_markdown(index=False))
    lines.append("")
    lines.append("### Point metrics")
    lines.append("")
    lines.append(point.to_markdown(index=False))
    lines.append("")
    lines.append("### Per-family metrics")
    lines.append("")
    lines.append(family_compact.to_markdown(index=False))
    lines.append("")
    lines.append("### Key interpretation")
    lines.append("")
    lines.append(f"- Original fine-grained accuracy: `{fmt(orig_fine['estimate'])}` with 95% CI [`{fmt(orig_fine['ci_low_95'])}`, `{fmt(orig_fine['ci_high_95'])}`].")
    lines.append(f"- Family-mass fine-grained accuracy: `{fmt(fm_fine['estimate'])}` with 95% CI [`{fmt(fm_fine['ci_low_95'])}`, `{fmt(fm_fine['ci_high_95'])}`].")
    lines.append(f"- Oracle true-family fine-grained accuracy: `{fmt(oracle_fine['estimate'])}` with 95% CI [`{fmt(oracle_fine['ci_low_95'])}`, `{fmt(oracle_fine['ci_high_95'])}`].")
    lines.append(f"- Family-mass reranking changes fine accuracy by `{fmt(fm_delta_fine['comparison_minus_reference'])}` with 95% CI [`{fmt(fm_delta_fine['ci_low_95'])}`, `{fmt(fm_delta_fine['ci_high_95'])}`].")
    lines.append(f"- Family-mass reranking improves family accuracy by `{fmt(fm_delta_family['comparison_minus_reference'])}` with 95% CI [`{fmt(fm_delta_family['ci_low_95'])}`, `{fmt(fm_delta_family['ci_high_95'])}`].")
    lines.append(f"- Oracle true-family headroom for fine accuracy is `{fmt(oracle_delta_fine['comparison_minus_reference'])}` with 95% CI [`{fmt(oracle_delta_fine['ci_low_95'])}`, `{fmt(oracle_delta_fine['ci_high_95'])}`].")
    lines.append(f"- Oracle family-level headroom is `{fmt(oracle_delta_family['comparison_minus_reference'])}`, reflecting the gap between current family assignment and perfect family assignment.")
    lines.append("- Interpretation: simple family-mass reranking improves family-level accuracy but slightly hurts fine-grained accuracy. The oracle result shows that better family-level modeling could recover a substantial fraction of remaining fine-class errors.")
    lines.append("")
    lines.append("### Suggested manuscript wording")
    lines.append("")
    lines.append("```latex")
    lines.append(
        f"We further quantified the hierarchical headroom of the final ensemble by comparing the original fine-grained top-1 prediction with two family-aware variants. "
        f"The standard ensemble achieved a fine-grained accuracy of \\textbf{{{float(orig_fine['estimate']):.4f}}} "
        f"(95\\% CI [{float(orig_fine['ci_low_95']):.4f}, {float(orig_fine['ci_high_95']):.4f}]). "
        f"A simple family-mass reranking strategy achieved \\textbf{{{float(fm_fine['estimate']):.4f}}} "
        f"(95\\% CI [{float(fm_fine['ci_low_95']):.4f}, {float(fm_fine['ci_high_95']):.4f}]), corresponding to a delta of "
        f"\\textbf{{{float(fm_delta_fine['comparison_minus_reference']):+.4f}}}. "
        f"Although this reranking increased family-level accuracy from \\textbf{{{float(orig_family['estimate']):.4f}}} to "
        f"\\textbf{{{float(fm_family['estimate']):.4f}}}, it did not improve fine-grained classification. "
        f"When the prediction was restricted to the true astronomical family, the fine-grained upper bound increased to "
        f"\\textbf{{{float(oracle_fine['estimate']):.4f}}} "
        f"(95\\% CI [{float(oracle_fine['ci_low_95']):.4f}, {float(oracle_fine['ci_high_95']):.4f}]), yielding an oracle headroom of "
        f"\\textbf{{{float(oracle_delta_fine['comparison_minus_reference']):+.4f}}}. "
        f"These results indicate that naive hierarchical reranking alone is insufficient, but that improved family-level modeling could recover a substantial fraction of the remaining fine-grained errors."
    )
    lines.append("```")
    lines.append("")
    lines.append("### Consolidated output files")
    lines.append("")
    for name in [
        "hierarchical_upper_bound_summary.md",
        "hierarchical_bootstrap_accuracy.csv",
        "hierarchical_paired_bootstrap_deltas_vs_original.csv",
        "hierarchical_strategy_point_metrics.csv",
        "hierarchical_object_level_predictions.csv",
        "hierarchical_per_family_metrics.csv",
        "hierarchical_upper_bound_schema.json",
        "fig_hierarchical_upper_bound_two_panel.png",
        "fig_hierarchical_fine_accuracy.png",
        "fig_hierarchical_family_accuracy.png",
    ]:
        if (dst_dir / name).exists() or (src_dir / name).exists():
            lines.append(f"- `{dst_dir / name}`")
    lines.append("")
    return "\n".join(lines)


def insert_or_replace(text: str, section: str) -> str:
    pattern = rf"{re.escape(SECTION_TITLE)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _m: section, text, flags=re.S)

    anchors = [
        "## Clean ensemble follow-up policy ablation",
        "## Ensemble follow-up policy ablation bootstrap",
        "## Top-k and family-level bootstrap confidence intervals",
        "## Ensemble uncertainty, error detection, and selective prediction",
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
    parser.add_argument("--hierarchical-dir", required=True)
    parser.add_argument("--final-publication-dir", required=True)
    args = parser.parse_args()

    src_dir = Path(args.hierarchical_dir)
    final_dir = Path(args.final_publication_dir)
    final_summary = final_dir / "final_publication_summary.md"
    dst_dir = final_dir / "ensemble_hierarchical_upper_bound"

    if not src_dir.exists():
        raise FileNotFoundError(f"Hierarchical directory not found: {src_dir}")
    if not final_summary.exists():
        raise FileNotFoundError(f"Final publication summary not found: {final_summary}")

    copy_outputs(src_dir, dst_dir)
    section = build_section(src_dir, dst_dir)

    text = final_summary.read_text(encoding="utf-8")
    updated = insert_or_replace(text, section)
    final_summary.write_text(updated, encoding="utf-8")

    print(f"Updated: {final_summary}")
    print(f"Copied outputs to: {dst_dir}")
    print("Inserted/replaced section:")
    print(SECTION_TITLE)


if __name__ == "__main__":
    main()
