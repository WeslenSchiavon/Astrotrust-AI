#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_followup_policy_ablation.py

Consolida a ablação estatística das políticas de follow-up em:
  results/final_publication/followup_policy_ablation/
e atualiza:
  results/final_publication/final_publication_summary.md

Este script NÃO recalcula métricas. Ele lê os CSVs/figuras já gerados por
run_followup_policy_ablation_bootstrap.py e insere/substitui uma seção marcada
no resumo final do artigo.

Uso:
python ./experiments/update_final_publication_with_followup_policy_ablation.py --ablation-dir results/followup_policy_ablation_bootstrap_250k_final --final-publication-dir results/final_publication
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


START = "<!-- BEGIN_FOLLOWUP_POLICY_ABLATION_BOOTSTRAP -->"
END = "<!-- END_FOLLOWUP_POLICY_ABLATION_BOOTSTRAP -->"


FILES_TO_COPY = [
    "followup_policy_ablation_summary.md",
    "followup_policy_ablation_summary_by_budget.csv",
    "followup_policy_ablation_pairwise_vs_reference.csv",
    "followup_policy_ablation_overlap_vs_reference.csv",
    "followup_policy_ablation_schema.json",
    "fig_policy_rare_enrichment_by_budget.png",
    "fig_policy_novelty_by_budget.png",
    "fig_policy_tradeoff_rare_vs_novelty.png",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def fmt(x, digits=6):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


def pick_row(df: pd.DataFrame, budget: float, policy: str) -> pd.Series:
    sub = df[(df["budget_fraction"].round(6) == round(budget, 6)) & (df["policy"] == policy)]
    if sub.empty:
        raise ValueError(f"Missing row for budget={budget}, policy={policy}")
    return sub.iloc[0]


def pick_pairwise(pairwise: pd.DataFrame, budget: float, comparison_policy: str, metric: str) -> pd.Series:
    sub = pairwise[
        (pairwise["budget_fraction"].round(6) == round(budget, 6))
        & (pairwise["comparison_policy"] == comparison_policy)
        & (pairwise["metric"] == metric)
    ]
    if sub.empty:
        raise ValueError(f"Missing pairwise row: budget={budget}, comparison={comparison_policy}, metric={metric}")
    return sub.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", default="results/followup_policy_ablation_bootstrap_250k_final")
    parser.add_argument("--final-publication-dir", default="results/final_publication")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    final_dir = Path(args.final_publication_dir)
    target_dir = final_dir / "followup_policy_ablation"
    summary_path = final_dir / "final_publication_summary.md"

    summary_csv = ablation_dir / "followup_policy_ablation_summary_by_budget.csv"
    pairwise_csv = ablation_dir / "followup_policy_ablation_pairwise_vs_reference.csv"
    overlap_csv = ablation_dir / "followup_policy_ablation_overlap_vs_reference.csv"
    report_md = ablation_dir / "followup_policy_ablation_summary.md"

    for p in [summary_csv, pairwise_csv, overlap_csv, report_md, summary_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for name in FILES_TO_COPY:
        src = ablation_dir / name
        dst = target_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append((src, dst))
        else:
            missing.append(name)

    summary = pd.read_csv(summary_csv)
    pairwise = pd.read_csv(pairwise_csv)
    overlap = pd.read_csv(overlap_csv)

    budgets = sorted(summary["budget_fraction"].unique())
    policies = list(summary["policy"].drop_duplicates())

    # Key comparisons at 5% and broad summary.
    key_budget = 0.05
    nr5 = pick_row(summary, key_budget, "novelty_rarity")
    ro5 = pick_row(summary, key_budget, "rarity_only")
    pd5 = pick_row(summary, key_budget, "previous_discovery")
    fd5 = pick_row(summary, key_budget, "fixed_discovery")
    nr_vs_ro_rare5 = pick_pairwise(pairwise, key_budget, "rarity_only", "rare_rate")
    nr_vs_ro_nov5 = pick_pairwise(pairwise, key_budget, "rarity_only", "novelty")

    # Build compact tables.
    compact_rows = []
    for b in budgets:
        for pol in policies:
            r = pick_row(summary, float(b), pol)
            compact_rows.append({
                "budget_fraction": b,
                "policy": pol,
                "n_selected": int(r["n_selected"]),
                "rare_rate": fmt(r["rare_rate"]),
                "rare_enrichment": fmt(r["rare_enrichment"]),
                "rare_enrichment_95CI": f"[{fmt(r['rare_enrichment_ci_low_95'])}, {fmt(r['rare_enrichment_ci_high_95'])}]",
                "mean_novelty": fmt(r.get("mean_novelty", "NA")),
                "mean_uncertainty": fmt(r.get("mean_uncertainty", "NA")),
                "mean_confidence": fmt(r.get("mean_confidence", "NA")),
                "mean_correct": fmt(r.get("mean_correct", "NA")),
            })
    compact_df = pd.DataFrame(compact_rows)

    pair_compact = pairwise[
        (pairwise["comparison_policy"] == "rarity_only")
        & (pairwise["metric"].isin(["rare_rate", "novelty", "confidence", "correct"]))
    ].copy()
    pair_compact = pair_compact[[
        "budget_fraction", "reference_policy", "comparison_policy", "metric",
        "reference_minus_comparison", "bootstrap_ci_low_95", "bootstrap_ci_high_95",
        "two_sided_bootstrap_pvalue",
        "one_sided_pvalue_reference_more_than_margin_worse"
    ]]

    overlap_compact = overlap[
        overlap["policy"].isin(["rarity_only", "previous_discovery", "fixed_discovery"])
    ][[
        "budget_fraction", "policy", "intersection_n", "jaccard_overlap",
        "overlap_fraction_of_reference"
    ]].copy()

    final_md = read_text(summary_path)

    copied_rows = "\n".join(f"| `{src}` | `{dst}` |" for src, dst in copied)
    missing_block = ""
    if missing:
        missing_block = "\n### Missing optional files\n\n" + "\n".join(f"- `{m}`" for m in missing) + "\n"

    new_section = f"""
{START}

## Follow-up policy ablation bootstrap

Updated from: `{report_md}`

This section consolidates the statistical ablation of the follow-up policies used by AstroTrust-AI. The outputs were copied to:

`{target_dir}`

The analysis compares object-level rankings for the final 250k follow-up policy evaluation and uses `novelty_rarity` as the reference policy.

### Key ablation conclusion

The final `novelty_rarity` policy preserves nearly the same rare-class enrichment as `rarity_only`, while explicitly adding a novelty component. At the 5% follow-up budget, `novelty_rarity` reaches rare enrichment = {fmt(nr5['rare_enrichment'])} and rare rate = {fmt(nr5['rare_rate'])}, compared with rare enrichment = {fmt(ro5['rare_enrichment'])} and rare rate = {fmt(ro5['rare_rate'])} for `rarity_only`. The paired bootstrap difference in rare rate between `novelty_rarity` and `rarity_only` is {fmt(nr_vs_ro_rare5['reference_minus_comparison'])}, with 95% CI [{fmt(nr_vs_ro_rare5['bootstrap_ci_low_95'])}, {fmt(nr_vs_ro_rare5['bootstrap_ci_high_95'])}]. The novelty score is higher for `novelty_rarity` at the same budget, with paired difference {fmt(nr_vs_ro_nov5['reference_minus_comparison'])}, 95% CI [{fmt(nr_vs_ro_nov5['bootstrap_ci_low_95'])}, {fmt(nr_vs_ro_nov5['bootstrap_ci_high_95'])}].

This supports the methodological choice of `novelty_rarity`: it is effectively competitive with the purely rarity-driven policy in rare-candidate enrichment, but is more aligned with a broker-like triage objective because it combines rarity and novelty.

### Key numbers at 5% follow-up budget

| policy | rare_rate | rare_enrichment | mean_novelty | mean_uncertainty | mean_confidence | mean_correct |
|:--|--:|--:|--:|--:|--:|--:|
| novelty_rarity | {fmt(nr5['rare_rate'])} | {fmt(nr5['rare_enrichment'])} | {fmt(nr5.get('mean_novelty', 'NA'))} | {fmt(nr5.get('mean_uncertainty', 'NA'))} | {fmt(nr5.get('mean_confidence', 'NA'))} | {fmt(nr5.get('mean_correct', 'NA'))} |
| rarity_only | {fmt(ro5['rare_rate'])} | {fmt(ro5['rare_enrichment'])} | {fmt(ro5.get('mean_novelty', 'NA'))} | {fmt(ro5.get('mean_uncertainty', 'NA'))} | {fmt(ro5.get('mean_confidence', 'NA'))} | {fmt(ro5.get('mean_correct', 'NA'))} |
| previous_discovery | {fmt(pd5['rare_rate'])} | {fmt(pd5['rare_enrichment'])} | {fmt(pd5.get('mean_novelty', 'NA'))} | {fmt(pd5.get('mean_uncertainty', 'NA'))} | {fmt(pd5.get('mean_confidence', 'NA'))} | {fmt(pd5.get('mean_correct', 'NA'))} |
| fixed_discovery | {fmt(fd5['rare_rate'])} | {fmt(fd5['rare_enrichment'])} | {fmt(fd5.get('mean_novelty', 'NA'))} | {fmt(fd5.get('mean_uncertainty', 'NA'))} | {fmt(fd5.get('mean_confidence', 'NA'))} | {fmt(fd5.get('mean_correct', 'NA'))} |

### Summary by budget

{compact_df.to_markdown(index=False)}

### Pairwise bootstrap: novelty_rarity versus rarity_only

{pair_compact.to_markdown(index=False)}

### Ranking overlap with novelty_rarity

{overlap_compact.to_markdown(index=False)}

### Recommended manuscript wording

```latex
We further performed a follow-up policy ablation to justify the final rarity--novelty objective. At a 5\\% follow-up budget, the selected \\texttt{{novelty\\_rarity}} policy achieved a rare-class enrichment of \\textbf{{{fmt(nr5['rare_enrichment'])}$\\times$}}, compared with \\textbf{{{fmt(ro5['rare_enrichment'])}$\\times$}} for the purely rarity-driven ranking. The paired bootstrap difference in rare rate between the two policies was small ({fmt(nr_vs_ro_rare5['reference_minus_comparison'])}; 95\\% CI [{fmt(nr_vs_ro_rare5['bootstrap_ci_low_95'])}, {fmt(nr_vs_ro_rare5['bootstrap_ci_high_95'])}]), whereas \\texttt{{novelty\\_rarity}} selected candidates with a higher novelty score ({fmt(nr_vs_ro_nov5['reference_minus_comparison'])}; 95\\% CI [{fmt(nr_vs_ro_nov5['bootstrap_ci_low_95'])}, {fmt(nr_vs_ro_nov5['bootstrap_ci_high_95'])}]). Thus, the final policy remains competitive with \\texttt{{rarity\\_only}} in rare-candidate enrichment while better matching the intended broker-like triage objective of prioritizing both rare and potentially novel events.
```

### Copied follow-up policy ablation files

| source | destination |
|:--|:--|
{copied_rows}
{missing_block}
{END}
""".strip() + "\n"

    pattern = rf"{re.escape(START)}.*?{re.escape(END)}"
    if re.search(pattern, final_md, flags=re.S):
        updated = re.sub(pattern, new_section, final_md, flags=re.S)
    else:
        updated = final_md.rstrip() + "\n\n" + new_section

    write_text(summary_path, updated)

    print(f"Copied {len(copied)} files to: {target_dir}")
    if missing:
        print("Missing optional files:")
        for m in missing:
            print(f"  - {m}")
    print(f"Updated: {summary_path}")


if __name__ == "__main__":
    main()
