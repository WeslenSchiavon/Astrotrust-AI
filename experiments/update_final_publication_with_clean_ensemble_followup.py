#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_clean_ensemble_followup.py

Consolida a versão cientificamente limpa da ablação de follow-up no:
results\final_publication\final_publication_summary.md

Uso:
python .\experiments\update_final_publication_with_clean_ensemble_followup.py `
  --clean-followup-dir results\ensemble_followup_policy_ablation_clean_250k_final `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


SECTION_TITLE = "## Clean ensemble follow-up policy ablation"


def fmt(x: float, ndigits: int = 6) -> str:
    return f"{float(x):.{ndigits}f}"


def copy_outputs(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.csv", "*.json", "*.md", "*.png"]:
        for src in src_dir.glob(pattern):
            shutil.copy2(src, dst_dir / src.name)


def pick(summary: pd.DataFrame, policy: str, budget: float) -> pd.Series:
    row = summary[
        (summary["policy"] == policy)
        & ((summary["budget_fraction"] - budget).abs() < 1e-12)
    ]
    if row.empty:
        raise ValueError(f"Missing row for policy={policy}, budget={budget}")
    return row.iloc[0]


def pick_pair(pairwise: pd.DataFrame, comparison: str, metric: str, budget: float) -> pd.Series:
    row = pairwise[
        (pairwise["comparison_policy"] == comparison)
        & (pairwise["metric"] == metric)
        & ((pairwise["budget_fraction"] - budget).abs() < 1e-12)
    ]
    if row.empty:
        raise ValueError(f"Missing pairwise row comparison={comparison}, metric={metric}, budget={budget}")
    return row.iloc[0]


def build_section(src_dir: Path, dst_dir: Path) -> str:
    summary_csv = src_dir / "clean_followup_policy_summary_by_budget.csv"
    pairwise_csv = src_dir / "clean_followup_policy_pairwise_vs_reference.csv"
    overlap_csv = src_dir / "clean_followup_policy_overlap_vs_reference.csv"
    schema_json = src_dir / "clean_followup_policy_ablation_schema.json"

    for p in [summary_csv, pairwise_csv, overlap_csv, schema_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    summary = pd.read_csv(summary_csv)
    pairwise = pd.read_csv(pairwise_csv)
    overlap = pd.read_csv(overlap_csv)
    schema = json.loads(schema_json.read_text(encoding="utf-8"))

    key_budget = 0.05
    nr = pick(summary, "novelty_rarity", key_budget)
    ro = pick(summary, "rarity_only", key_budget)
    rand = pick(summary, "random", key_budget)
    unr = pick(summary, "uncertainty_novelty_rarity", key_budget)

    diff_nr_ro_rare = pick_pair(pairwise, "rarity_only", "rare_rate", key_budget)
    diff_nr_ro_nov = pick_pair(pairwise, "rarity_only", "novelty", key_budget)
    diff_nr_rand_rare = pick_pair(pairwise, "random", "rare_rate", key_budget)
    diff_nr_unr_rare = pick_pair(pairwise, "uncertainty_novelty_rarity", "rare_rate", key_budget)

    compact = summary[
        [
            "budget_fraction", "policy", "n_selected",
            "rare_rate", "rare_rate_ci_low_95", "rare_rate_ci_high_95",
            "rare_enrichment", "rare_enrichment_ci_low_95", "rare_enrichment_ci_high_95",
            "mean_novelty", "mean_uncertainty", "mean_confidence",
            "mean_correct", "mean_rarity_score",
        ]
    ].copy()

    pairwise_key = pairwise[
        (pairwise["budget_fraction"].isin([0.01, 0.02, 0.05, 0.10, 0.20]))
        & (pairwise["comparison_policy"].isin(["random", "rarity_only", "uncertainty_novelty_rarity"]))
        & (pairwise["metric"].isin(["rare_rate", "novelty", "correct"]))
    ].copy()

    overlap_key = overlap[
        overlap["policy"].isin(["random", "rarity_only", "novelty_only", "uncertainty_novelty_rarity"])
    ][
        [
            "budget_fraction", "reference_policy", "policy",
            "intersection_n", "jaccard_overlap",
            "overlap_fraction_of_reference",
        ]
    ].copy()

    novelty = schema.get("novelty", {})
    lines = []
    lines.append(SECTION_TITLE)
    lines.append("")
    lines.append("This section consolidates the publication-safe follow-up policy ablation using final ensemble probabilities, an explicit rare-class set, and a model-independent novelty score.")
    lines.append("")
    lines.append("### Inputs and protocol")
    lines.append("")
    lines.append("- Model: `ensemble_hybrid_dominant`")
    lines.append("- Predictions: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_predictions.csv`")
    lines.append("- Probabilities: `results\\hybrid_tabular_ensemble_250k\\ensemble_hybrid_dominant_test_probabilities.npy`")
    lines.append(f"- Objects: `{int(schema.get('n_objects', nr['n_objects']))}`")
    lines.append(f"- Classes: `{int(schema.get('n_classes', 32))}`")
    lines.append(f"- Explicit rare labels: `{schema.get('rare_labels_explicit', [])}`")
    lines.append(f"- Baseline rare-object rate: `{float(nr['baseline_rare_rate']):.6f}`")
    lines.append(f"- Bootstrap iterations: `{int(nr['n_bootstrap'])}`")
    lines.append("- Bootstrap protocol: resample objects, rerank each policy within the bootstrap sample, and recompute selected-set metrics.")
    lines.append(f"- Novelty method: `{novelty.get('novelty_method', 'unknown')}`")
    lines.append(f"- Feature file: `{novelty.get('feature_file', 'unknown')}`")
    lines.append(f"- Reference feature file: `{novelty.get('reference_feature_file', 'unknown')}`")
    lines.append(f"- Number of features used after IQR filtering: `{novelty.get('n_features_used_after_iqr_filter', 'unknown')}`")
    lines.append("")
    lines.append("### Summary by budget")
    lines.append("")
    lines.append(compact.to_markdown(index=False))
    lines.append("")
    lines.append("### Key pairwise comparisons")
    lines.append("")
    lines.append(pairwise_key.to_markdown(index=False))
    lines.append("")
    lines.append("### Ranking overlap")
    lines.append("")
    lines.append(overlap_key.to_markdown(index=False))
    lines.append("")
    lines.append("### Key interpretation")
    lines.append("")
    lines.append(f"- At the 5% follow-up budget, `novelty_rarity` selects `{int(nr['n_selected'])}` objects.")
    lines.append(f"- `novelty_rarity` reaches rare-object rate `{fmt(nr['rare_rate'])}` with 95% CI [`{fmt(nr['rare_rate_ci_low_95'])}`, `{fmt(nr['rare_rate_ci_high_95'])}`], corresponding to rare-class enrichment `{fmt(nr['rare_enrichment'])}`.")
    lines.append(f"- `rarity_only` reaches rare-object rate `{fmt(ro['rare_rate'])}` and enrichment `{fmt(ro['rare_enrichment'])}`, but with much lower mean novelty `{fmt(ro['mean_novelty'])}`.")
    lines.append(f"- `random` reaches rare-object rate `{fmt(rand['rare_rate'])}` and enrichment `{fmt(rand['rare_enrichment'])}`.")
    lines.append(f"- `uncertainty_novelty_rarity` reaches rare-object rate `{fmt(unr['rare_rate'])}` and enrichment `{fmt(unr['rare_enrichment'])}`.")
    lines.append(f"- Compared with `random`, `novelty_rarity` improves rare rate by `{fmt(diff_nr_rand_rare['reference_minus_comparison'])}` at 5% budget.")
    lines.append(f"- Compared with `rarity_only`, `novelty_rarity` reduces rare rate by `{fmt(diff_nr_ro_rare['reference_minus_comparison'])}` with 95% CI [`{fmt(diff_nr_ro_rare['bootstrap_ci_low_95'])}`, `{fmt(diff_nr_ro_rare['bootstrap_ci_high_95'])}`], while increasing novelty by `{fmt(diff_nr_ro_nov['reference_minus_comparison'])}` with 95% CI [`{fmt(diff_nr_ro_nov['bootstrap_ci_low_95'])}`, `{fmt(diff_nr_ro_nov['bootstrap_ci_high_95'])}`].")
    lines.append(f"- Compared with `uncertainty_novelty_rarity`, `novelty_rarity` has rare-rate difference `{fmt(diff_nr_unr_rare['reference_minus_comparison'])}` at 5% budget.")
    lines.append("- This establishes a genuine broker-like trade-off: rarity-only maximizes rare-class retrieval, while novelty-rarity sacrifices a modest amount of rare-object rate to prioritize substantially more novel candidates.")
    lines.append("")
    lines.append("### Suggested manuscript wording")
    lines.append("")
    lines.append("```latex")
    lines.append(
        f"We performed a clean follow-up policy ablation using the final ensemble probabilities, an explicit predefined rare-class set, and a model-independent novelty score computed from robust distances in the feature space. "
        f"At a 5\\% follow-up budget, \\texttt{{novelty\\_rarity}} selected \\textbf{{{int(nr['n_selected'])}}} objects and achieved a rare-object rate of "
        f"\\textbf{{{float(nr['rare_rate']):.4f}}} (95\\% CI [{float(nr['rare_rate_ci_low_95']):.4f}, {float(nr['rare_rate_ci_high_95']):.4f}]), "
        f"corresponding to a rare-class enrichment of \\textbf{{{float(nr['rare_enrichment']):.2f}$\\times$}} over the test-set baseline. "
        f"The \\texttt{{rarity\\_only}} policy reached the maximum rare-object rate of \\textbf{{{float(ro['rare_rate']):.4f}}}, whereas a random policy reached only \\textbf{{{float(rand['rare_rate']):.4f}}}. "
        f"Compared with \\texttt{{rarity\\_only}}, \\texttt{{novelty\\_rarity}} reduced the rare-object rate by "
        f"\\textbf{{{float(diff_nr_ro_rare['reference_minus_comparison']):.4f}}} "
        f"(95\\% CI [{float(diff_nr_ro_rare['bootstrap_ci_low_95']):.4f}, {float(diff_nr_ro_rare['bootstrap_ci_high_95']):.4f}]), "
        f"but increased the novelty score by \\textbf{{{float(diff_nr_ro_nov['reference_minus_comparison']):.4f}}} "
        f"(95\\% CI [{float(diff_nr_ro_nov['bootstrap_ci_low_95']):.4f}, {float(diff_nr_ro_nov['bootstrap_ci_high_95']):.4f}]). "
        f"Thus, the follow-up module should be interpreted as a rare-class prioritization mechanism over a predefined target set, with novelty acting as an explicit secondary criterion for broker-like triage rather than as an independent rare-object discovery claim."
    )
    lines.append("```")
    lines.append("")
    lines.append("### Consolidated output files")
    lines.append("")
    for name in [
        "clean_followup_policy_ablation_summary.md",
        "clean_followup_policy_summary_by_budget.csv",
        "clean_followup_policy_pairwise_vs_reference.csv",
        "clean_followup_policy_overlap_vs_reference.csv",
        "clean_ensemble_followup_object_level.csv",
        "clean_followup_policy_ablation_schema.json",
        "fig_clean_policy_rare_enrichment_by_budget.png",
        "fig_clean_policy_novelty_by_budget.png",
        "fig_clean_policy_correctness_by_budget.png",
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
        "## Ensemble follow-up policy ablation bootstrap",
        "## Follow-up policy ablation bootstrap",
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
    parser.add_argument("--clean-followup-dir", required=True)
    parser.add_argument("--final-publication-dir", required=True)
    args = parser.parse_args()

    src_dir = Path(args.clean_followup_dir)
    final_dir = Path(args.final_publication_dir)
    final_summary = final_dir / "final_publication_summary.md"
    dst_dir = final_dir / "clean_ensemble_followup_policy_ablation"

    if not src_dir.exists():
        raise FileNotFoundError(f"Clean follow-up directory not found: {src_dir}")
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
