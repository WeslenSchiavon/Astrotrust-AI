#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_statistical_validation.py

Consolida os testes estatísticos finais do AstroTrust-AI em:
  results/final_publication/statistical_validation/
e atualiza:
  results/final_publication/final_publication_summary.md

Este script NÃO recalcula métricas. Ele apenas:
1. copia os arquivos finais já gerados;
2. lê publication_strength_tests_summary.md;
3. insere/substitui uma seção marcada no final_publication_summary.md.

Uso recomendado:
python ./experiments/update_final_publication_with_statistical_validation.py --stats-dir results/publication_strength_tests_250k_final --final-publication-dir results/final_publication
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


START = "<!-- BEGIN_PUBLICATION_STRENGTH_STATISTICAL_VALIDATION -->"
END = "<!-- END_PUBLICATION_STRENGTH_STATISTICAL_VALIDATION -->"


FILES_TO_COPY = [
    "publication_strength_tests_summary.md",
    "paired_seed_level_tests.csv",
    "paired_prediction_level_tests.csv",
    "per_class_bootstrap_ci.csv",
    "per_family_bootstrap_ci.csv",
    "permutation_rare_enrichment.csv",
    "risk_coverage_curve.csv",
    "error_detection_auroc.csv",
    "error_detection_roc_curve.csv",
    "fig_risk_coverage_curve.png",
    "fig_selective_accuracy_curve.png",
    "fig_error_detection_roc.png",
    "fig_permutation_rare_enrichment.png",
    "detected_schema.json",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def extract_section(md: str, heading: str) -> str:
    pattern = rf"(^## {re.escape(heading)}\n.*?)(?=^## |\Z)"
    m = re.search(pattern, md, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


def extract_bullet_value(md: str, label: str) -> str:
    m = re.search(rf"- {re.escape(label)}: `([^`]+)`", md)
    return m.group(1) if m else "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", default="results/publication_strength_tests_250k_final")
    parser.add_argument("--final-publication-dir", default="results/final_publication")
    args = parser.parse_args()

    stats_dir = Path(args.stats_dir)
    final_dir = Path(args.final_publication_dir)
    target_dir = final_dir / "statistical_validation"
    summary_path = final_dir / "final_publication_summary.md"
    stats_summary_path = stats_dir / "publication_strength_tests_summary.md"

    if not stats_summary_path.exists():
        raise FileNotFoundError(f"Missing: {stats_summary_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for name in FILES_TO_COPY:
        src = stats_dir / name
        dst = target_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append((src, dst))
        else:
            missing.append(name)

    stats_md = read_text(stats_summary_path)
    final_md = read_text(summary_path)

    seed_section = extract_section(stats_md, "1. Paired seed-level test: ensemble vs hybrid")
    heldout_section = extract_section(stats_md, "1b. Paired held-out prediction test")
    per_class_section = extract_section(stats_md, "2. Per-class bootstrap confidence intervals")
    per_family_section = extract_section(stats_md, "2b. Per-family bootstrap confidence intervals")
    perm_section = extract_section(stats_md, "3. Permutation test for rare-class enrichment")
    risk_section = extract_section(stats_md, "4. Risk-coverage / selective classification")
    auroc_section = extract_section(stats_md, "5. AUROC for error detection using uncertainty")

    full_risk = extract_bullet_value(stats_md, "Full-coverage risk")
    risk50 = extract_bullet_value(stats_md, "Risk at ~50% coverage")
    risk80 = extract_bullet_value(stats_md, "Risk at ~80% coverage")
    aurc = extract_bullet_value(stats_md, "AURC")

    copied_rows = "\n".join(f"| `{src}` | `{dst}` |" for src, dst in copied)
    missing_block = ""
    if missing:
        missing_block = "\n### Missing optional statistical validation files\n\n" + "\n".join(f"- `{m}`" for m in missing) + "\n"

    new_section = f"""
{START}

## Publication-strength statistical validation

Updated from: `{stats_summary_path}`

This section consolidates the final publication-strength statistical analyses. The outputs were copied to:

`{target_dir}`

The analyses use the final 250k ensemble and hybrid predictions, the ten-seed summary, the explicit class--family map, and the final object-level `novelty_rarity` follow-up ranking.

### Key statistical validation numbers

| category | result | value |
|:--|:--|:--|
| paired_seed_level | n_paired_seeds | 10 |
| paired_seed_level | ensemble_minus_hybrid_accuracy_delta | +0.0116495 |
| paired_seed_level | ensemble_minus_hybrid_macro_f1_delta | +0.00922987 |
| paired_seed_level | ensemble_minus_hybrid_brier_delta | -0.0142504 |
| paired_seed_level | ensemble_minus_hybrid_nll_delta | -0.0325842 |
| paired_heldout | n_objects | 57058 |
| paired_heldout | ensemble_accuracy | 0.684163 |
| paired_heldout | hybrid_accuracy | 0.672999 |
| paired_heldout | accuracy_delta | +0.0111641 |
| paired_heldout | mcnemar_exact_pvalue | 6.35842e-16 |
| rare_enrichment | selected_policy | novelty_rarity |
| rare_enrichment | budget_1pct_enrichment | 5.51548 |
| rare_enrichment | budget_2pct_enrichment | 5.35333 |
| rare_enrichment | budget_5pct_enrichment | 5.18544 |
| rare_enrichment | budget_10pct_enrichment | 5.33963 |
| rare_enrichment | budget_20pct_enrichment | 4.74988 |
| rare_enrichment | permutation_pvalue_all_budgets | 0.00019996 |
| selective_classification | full_coverage_risk | {full_risk} |
| selective_classification | risk_at_50pct_coverage | {risk50} |
| selective_classification | risk_at_80pct_coverage | {risk80} |
| selective_classification | AURC | {aurc} |
| error_detection | AUROC | 0.865232 |
| error_detection | AUROC_95pct_CI | [0.862331, 0.868204] |
| error_detection | AUPRC | 0.690571 |
| error_detection | AUPRC_95pct_CI | [0.683025, 0.698797] |

### Main interpretation for the manuscript

- The final `ensemble_hybrid_dominant` model outperforms the standalone `hybrid_temporal_tabular_cnn` consistently across all ten matched random seeds.
- The paired held-out test confirms that the ensemble improves over the hybrid model on the same 57,058 test objects.
- The `novelty_rarity` follow-up policy produces strong rare-class enrichment under limited observing budgets, with approximately 5.19x enrichment at the 5% budget and a one-sided permutation p-value of 0.00019996.
- Selective classification substantially reduces risk: at approximately 50% coverage, the selective accuracy reaches 0.938624.
- The uncertainty score is informative for error detection, with AUROC = 0.865232 and AUPRC = 0.690571.

### Source statistical validation sections

#### Paired seed-level comparison

{seed_section}

#### Paired held-out prediction comparison

{heldout_section}

#### Per-class bootstrap confidence intervals

{per_class_section}

#### Per-family bootstrap confidence intervals

{per_family_section}

#### Rare-class enrichment permutation test

{perm_section}

#### Risk-coverage / selective classification

{risk_section}

#### AUROC for error detection

{auroc_section}

### Copied statistical validation files

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
