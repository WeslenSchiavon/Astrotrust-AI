#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_ensemble_followup_policy_ablation.py

Consolida a ablação de políticas de follow-up reconstruída para o ensemble final
no arquivo:
results\final_publication\final_publication_summary.md

Uso:
python .\experiments\update_final_publication_with_ensemble_followup_policy_ablation.py `
  --followup-dir results\ensemble_followup_policy_ablation_250k_final `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


SECTION_TITLE = "## Ensemble follow-up policy ablation bootstrap"


def fmt(x: float, ndigits: int = 6) -> str:
    return f"{float(x):.{ndigits}f}"


def copy_outputs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.csv", "*.json", "*.md", "*.png"]:
        for src in src_dir.glob(pattern):
            shutil.copy2(src, dst_dir / src.name)


def pick_summary(summary_df: pd.DataFrame, policy: str, budget: float) -> pd.Series:
    row = summary_df[
        (summary_df["policy"] == policy)
        & ((summary_df["budget_fraction"] - budget).abs() < 1e-12)
    ]
    if row.empty:
        raise ValueError(f"Missing summary row for policy={policy}, budget={budget}")
    return row.iloc[0]


def pick_pairwise(pairwise_df: pd.DataFrame, comparison_policy: str, metric: str, budget: float) -> pd.Series:
    row = pairwise_df[
        (pairwise_df["comparison_policy"] == comparison_policy)
        & (pairwise_df["metric"] == metric)
        & ((pairwise_df["budget_fraction"] - budget).abs() < 1e-12)
    ]
    if row.empty:
        raise ValueError(f"Missing pairwise row for comparison={comparison_policy}, metric={metric}, budget={budget}")
    return row.iloc[0]


def build_section(src_dir: Path, final_subdir: Path) -> str:
    summary_csv = src_dir / "followup_policy_ablation_summary_by_budget.csv"
    pairwise_csv = src_dir / "followup_policy_ablation_pairwise_vs_reference.csv"
    overlap_csv = src_dir / "followup_policy_ablation_overlap_vs_reference.csv"
    schema_json = src_dir / "followup_policy_ablation_schema.json"
    generation_schema_json = src_dir / "ensemble_followup_ranking_generation_schema.json"

    for p in [summary_csv, pairwise_csv, overlap_csv, schema_json, generation_schema_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    summary_df = pd.read_csv(summary_csv)
    pairwise_df = pd.read_csv(pairwise_csv)
    overlap_df = pd.read_csv(overlap_csv)

    key_budget = 0.05
    nr5 = pick_summary(summary_df, "novelty_rarity", key_budget)
    ro5 = pick_summary(summary_df, "rarity_only", key_budget)
    prev5 = pick_summary(summary_df, "previous_discovery", key_budget)
    fixed5 = pick_summary(summary_df, "fixed_discovery", key_budget)

    rare_diff = pick_pairwise(pairwise_df, "rarity_only", "rare_rate", key_budget)
    novelty_diff = pick_pairwise(pairwise_df, "rarity_only", "novelty", key_budget)
    correct_diff = pick_pairwise(pairwise_df, "rarity_only", "correct", key_budget)

    compact = summary_df[
        [
            "budget_fraction",
            "policy",
            "n_selected",
            "rare_rate",
            "rare_rate_ci_low_95",
            "rare_rate_ci_high_95",
            "rare_enrichment",
            "rare_enrichment_ci_low_95",
            "rare_enrichment_ci_high_95",
            "mean_novelty",
            "mean_uncertainty",
            "mean_confidence",
            "mean_correct",
        ]
    ].copy()

    pairwise_key = pairwise_df[
        (pairwise_df["budget_fraction"].isin([0.01, 0.02, 0.05, 0.10, 0.20]))
        & (pairwise_df["comparison_policy"] == "rarity_only")
        & (pairwise_df["metric"].isin(["rare_rate", "novelty", "correct"]))
    ].copy()

    overlap_key = overlap_df[
        (overlap_df["policy"].isin(["rarity_only", "previous_discovery", "fixed_discovery"]))
    ][
        [
            "budget_fraction",
            "reference_policy",
            "policy",
            "intersection_n",
            "jaccard_overlap",
            "overlap_fraction_of_reference",
        ]
    ].copy()

    n_common = int(nr5["n_objects_common"])
    baseline = float(nr5["baseline_rare_rate"])
    n_boot = int(nr5["n_bootstrap"])

    section = []
    section.append(SECTION_TITLE)
    section.append("")
    section.append("This section consolidates the follow-up policy ablation reconstructed for the final `ensemble_hybrid_dominant` model.")
    section.append("")
    section.append("### Inputs and protocol")
    section.append("")
    section.append("- Model: `ensemble_hybrid_dominant`")
    section.append("- Ensemble predictions: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_predictions.csv`")
    section.append("- Ensemble probabilities: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_test_probabilities.npy`")
    section.append("- Policy rankings were reconstructed from the final ensemble probabilities.")
    section.append("- `rarity_score` is the sum of final ensemble probabilities over rare labels inferred from the template `true_is_rare` column.")
    section.append("- `novelty_score` is reused from the original object-level ranking template because novelty is object metadata, not a model prediction.")
    section.append(f"- Common objects across policies: `{n_common}`")
    section.append(f"- Baseline rare-object rate: `{baseline:.6f}`")
    section.append(f"- Bootstrap iterations: `{n_boot}`")
    section.append("- Reference policy: `novelty_rarity`")
    section.append("")
    section.append("### Summary by budget")
    section.append("")
    section.append(compact.to_markdown(index=False))
    section.append("")
    section.append("### Key pairwise comparisons versus `rarity_only`")
    section.append("")
    section.append(pairwise_key.to_markdown(index=False))
    section.append("")
    section.append("### Ranking overlap with reference policy")
    section.append("")
    section.append(overlap_key.to_markdown(index=False))
    section.append("")
    section.append("### Key interpretation")
    section.append("")
    section.append(f"- At the 5% follow-up budget, `novelty_rarity` selects `{int(nr5['n_selected'])}` objects.")
    section.append(f"- `novelty_rarity` reaches rare-object rate `{fmt(nr5['rare_rate'])}` with rare enrichment `{fmt(nr5['rare_enrichment'])}`.")
    section.append(f"- `rarity_only` reaches rare-object rate `{fmt(ro5['rare_rate'])}` with rare enrichment `{fmt(ro5['rare_enrichment'])}`.")
    section.append(f"- The rare-rate difference `novelty_rarity - rarity_only` is `{fmt(rare_diff['reference_minus_comparison'])}` with 95% CI [`{fmt(rare_diff['bootstrap_ci_low_95'])}`, `{fmt(rare_diff['bootstrap_ci_high_95'])}`].")
    section.append(f"- The novelty-score difference `novelty_rarity - rarity_only` is `{fmt(novelty_diff['reference_minus_comparison'])}` with 95% CI [`{fmt(novelty_diff['bootstrap_ci_low_95'])}`, `{fmt(novelty_diff['bootstrap_ci_high_95'])}`].")
    section.append(f"- The correctness difference `novelty_rarity - rarity_only` is `{fmt(correct_diff['reference_minus_comparison'])}` with 95% CI [`{fmt(correct_diff['bootstrap_ci_low_95'])}`, `{fmt(correct_diff['bootstrap_ci_high_95'])}`].")
    section.append("- The final policy therefore preserves nearly identical rare-object enrichment relative to a purely rarity-driven policy while incorporating higher novelty.")
    section.append("")
    section.append("### Suggested manuscript wording")
    section.append("")
    section.append("```latex")
    section.append(
        f"We repeated the follow-up policy ablation using rankings reconstructed from the final ensemble probabilities. "
        f"At a 5\\% follow-up budget, the final \\texttt{{novelty\\_rarity}} policy selected \\textbf{{{int(nr5['n_selected'])}}} objects and achieved a rare-object rate of "
        f"\\textbf{{{float(nr5['rare_rate']):.4f}}}, corresponding to a rare-class enrichment of \\textbf{{{float(nr5['rare_enrichment']):.2f}$\\times$}} over the test-set baseline. "
        f"The purely rarity-driven baseline reached a rare-object rate of \\textbf{{{float(ro5['rare_rate']):.4f}}} and an enrichment of \\textbf{{{float(ro5['rare_enrichment']):.2f}$\\times$}}. "
        f"The paired bootstrap difference in rare-object rate between \\texttt{{novelty\\_rarity}} and \\texttt{{rarity\\_only}} was "
        f"\\textbf{{{float(rare_diff['reference_minus_comparison']):.4f}}} "
        f"(95\\% CI [{float(rare_diff['bootstrap_ci_low_95']):.4f}, {float(rare_diff['bootstrap_ci_high_95']):.4f}]), "
        f"whereas the novelty score was higher for \\texttt{{novelty\\_rarity}} by "
        f"\\textbf{{{float(novelty_diff['reference_minus_comparison']):.4f}}} "
        f"(95\\% CI [{float(novelty_diff['bootstrap_ci_low_95']):.4f}, {float(novelty_diff['bootstrap_ci_high_95']):.4f}]). "
        f"This result supports the final policy as a broker-like trade-off: it preserves near-maximal rare-object enrichment while explicitly incorporating novelty into the prioritization rule."
    )
    section.append("```")
    section.append("")
    section.append("### Consolidated output files")
    section.append("")
    for name in [
        "followup_policy_ablation_summary.md",
        "followup_policy_ablation_summary_by_budget.csv",
        "followup_policy_ablation_pairwise_vs_reference.csv",
        "followup_policy_ablation_overlap_vs_reference.csv",
        "followup_policy_ablation_schema.json",
        "ensemble_followup_ranking_generation_schema.json",
        "fig_policy_rare_enrichment_by_budget.png",
        "fig_policy_novelty_by_budget.png",
        "fig_policy_tradeoff_rare_vs_novelty.png",
        "ensemble_test_ranking_novelty_rarity.csv",
        "ensemble_test_ranking_rarity_only.csv",
        "ensemble_test_ranking_previous_discovery.csv",
        "ensemble_test_ranking_fixed_discovery.csv",
    ]:
        if (final_subdir / name).exists() or (src_dir / name).exists():
            section.append(f"- `{final_subdir / name}`")
    section.append("")
    return "\n".join(section)


def insert_or_replace_section(text: str, new_section: str) -> str:
    pattern = rf"{re.escape(SECTION_TITLE)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _m: new_section, text, flags=re.S)

    # Insert after the old follow-up ablation if present, otherwise after uncertainty, otherwise append.
    anchors = [
        "## Follow-up policy ablation bootstrap",
        "## Ensemble uncertainty, error detection, and selective prediction",
        "## Publication-strength statistical validation",
    ]

    insert_pos = None
    for anchor in anchors:
        pattern_anchor = rf"{re.escape(anchor)}\n.*?(?=\n## |\Z)"
        matches = list(re.finditer(pattern_anchor, text, flags=re.S))
        if matches:
            insert_pos = matches[-1].end()
            break

    if insert_pos is not None:
        return text[:insert_pos].rstrip() + "\n\n" + new_section + "\n" + text[insert_pos:].lstrip()

    return text.rstrip() + "\n\n" + new_section + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--followup-dir", required=True)
    parser.add_argument("--final-publication-dir", required=True)
    args = parser.parse_args()

    src_dir = Path(args.followup_dir)
    final_dir = Path(args.final_publication_dir)
    final_summary = final_dir / "final_publication_summary.md"

    if not src_dir.exists():
        raise FileNotFoundError(f"Follow-up directory not found: {src_dir}")
    if not final_summary.exists():
        raise FileNotFoundError(f"Final publication summary not found: {final_summary}")

    final_subdir = final_dir / "ensemble_followup_policy_ablation"
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
