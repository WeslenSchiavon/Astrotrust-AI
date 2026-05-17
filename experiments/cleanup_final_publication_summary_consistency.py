#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
cleanup_final_publication_summary_consistency.py

Faz uma limpeza metodológica no final_publication_summary.md:
1. cria um backup;
2. insere um índice oficial de resultados atuais;
3. marca tabelas/seções antigas como históricas/superseded;
4. preserva todos os dados para rastreabilidade.

Uso:
python .\experiments\cleanup_final_publication_summary_consistency.py `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


AUTHORITATIVE_TITLE = "## Authoritative current results index"


def build_authoritative_section() -> str:
    return r"""
## Authoritative current results index

This section supersedes the older preliminary summary tables when numerical conflicts occur. The current official result blocks are the ensemble-based analyses consolidated later in this file.

| result block | official model/artifact | official key numbers | status |
|:--|:--|:--|:--|
| Final classifier | `ensemble_hybrid_dominant` | Accuracy = `0.684163`; Macro-F1 = `0.676580`; Weighted-F1 = `0.676995` | main result |
| Top-k fine classification | `ensemble_hybrid_dominant` | Top-1 = `0.684163`; Top-3 = `0.890147`; Top-5 = `0.948053` | main result |
| Family-level classification | `ensemble_hybrid_dominant` | Family top-1 = `0.835904`; family top-3 = `0.958709`; family top-5 = `0.987907` | main result |
| Selective prediction | `ensemble_hybrid_dominant` | Accuracy at ~50% coverage = `0.938589`; accuracy at ~80% coverage = `0.783732` | main result |
| Error detection from uncertainty | `ensemble_hybrid_dominant` | AUROC = `0.865232`; AUPRC = `0.690571` | main result |
| Clean follow-up prioritization | `ensemble_hybrid_dominant` + robust feature-space novelty | At 5% budget: rare rate = `0.941465`; enrichment = `5.275792x`; random rare rate = `0.180512` | main result |
| Follow-up trade-off versus rarity-only | clean ensemble follow-up ablation | `novelty_rarity - rarity_only` rare-rate delta = `-0.058535`; novelty delta = `+0.118639` | main result with trade-off |
| Hierarchical upper-bound | `ensemble_hybrid_dominant` | original fine accuracy = `0.684163`; family-mass fine accuracy = `0.679677`; oracle true-family fine accuracy = `0.788864`; oracle headroom = `+0.104700` | main result / future-work evidence |
| Temperature scaling | standalone `hybrid_temporal_tabular_cnn` | ECE reduced from `0.027016` to `0.014727`; Brier from `0.416519` to `0.415015`; NLL from `0.929091` to `0.919960` | auxiliary calibration result |
| Raw ensemble calibration | `ensemble_hybrid_dominant` | not yet consolidated in this package | pending/diagnostic |
| ZTF/IRSA demonstrations | AstroTrust-AI interface/inference pipeline | real-data ingestion and OOD behavior only | operational demonstration, not primary benchmark |

### Consistency rule for manuscript writing

- Use `ensemble_hybrid_dominant` as the final operational AstroTrust-AI classifier.
- Use the clean ensemble follow-up ablation as the official follow-up result.
- Use the ensemble top-k/family, uncertainty/selective prediction, and hierarchical upper-bound sections as the official downstream analyses.
- Treat temperature scaling as an auxiliary calibration analysis of the standalone hybrid model unless a separate validation-set calibration artifact is produced for the final ensemble.
- Treat older follow-up, top-k, and hierarchical tables as historical provenance if they conflict with the official ensemble sections below.
""".strip()


def insert_after_core_positioning(text: str, section: str) -> str:
    # Replace existing authoritative section if present.
    pattern = rf"{re.escape(AUTHORITATIVE_TITLE)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _m: section, text, flags=re.S)

    anchor = "## Core positioning"
    pat = rf"{re.escape(anchor)}\n.*?(?=\n## |\Z)"
    m = re.search(pat, text, flags=re.S)
    if m:
        pos = m.end()
        return text[:pos].rstrip() + "\n\n" + section + "\n\n" + text[pos:].lstrip()

    return section + "\n\n" + text


def replace_once(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1) if old in text else text


def add_warning_after_heading(text: str, heading: str, warning: str) -> str:
    if heading not in text:
        return text
    idx = text.find(heading)
    end = idx + len(heading)
    next_chunk = text[end:end + 500]
    if warning.strip() in next_chunk:
        return text
    return text[:end] + "\n\n" + warning.strip() + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-publication-dir", required=True)
    args = parser.parse_args()

    final_dir = Path(args.final_publication_dir)
    summary = final_dir / "final_publication_summary.md"
    if not summary.exists():
        raise FileNotFoundError(f"Missing file: {summary}")

    original = summary.read_text(encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = summary.with_name(f"final_publication_summary.before_consistency_cleanup_{timestamp}.md")
    backup.write_text(original, encoding="utf-8")

    text = original
    text = insert_after_core_positioning(text, build_authoritative_section())

    # Rename legacy headings only once. These sections are preserved for provenance.
    text = replace_once(
        text,
        "## Key numbers",
        "## Historical key numbers partly superseded by the authoritative current results index",
    )
    text = replace_once(
        text,
        "## Table: top-k and family-level performance",
        "## Legacy table: top-k and family-level performance superseded by ensemble bootstrap section",
    )
    text = replace_once(
        text,
        "## Table: follow-up policy",
        "## Legacy table: follow-up policy superseded by clean ensemble follow-up ablation",
    )
    text = replace_once(
        text,
        "## Table: hierarchical upper-bound analysis",
        "## Legacy table: hierarchical upper-bound analysis superseded by ensemble hierarchical analysis",
    )
    text = replace_once(
        text,
        "## Follow-up policy ablation bootstrap",
        "## Deprecated legacy follow-up policy ablation bootstrap",
    )

    # Add clear notes.
    text = add_warning_after_heading(
        text,
        "## Historical key numbers partly superseded by the authoritative current results index",
        """
> **Consistency note.** This table contains historical values from earlier consolidation passes. When values conflict with the authoritative current results index or the later ensemble-specific sections, use the newer ensemble-specific sections.
""",
    )
    text = add_warning_after_heading(
        text,
        "## Legacy table: top-k and family-level performance superseded by ensemble bootstrap section",
        """
> **Superseded.** Use the later `Top-k and family-level bootstrap confidence intervals` section for official ensemble top-k/family metrics.
""",
    )
    text = add_warning_after_heading(
        text,
        "## Legacy table: follow-up policy superseded by clean ensemble follow-up ablation",
        """
> **Superseded.** Use the later `Clean ensemble follow-up policy ablation` section as the official follow-up analysis.
""",
    )
    text = add_warning_after_heading(
        text,
        "## Legacy table: hierarchical upper-bound analysis superseded by ensemble hierarchical analysis",
        """
> **Superseded.** Use the later `Ensemble hierarchical upper-bound analysis` section for official ensemble hierarchical results.
""",
    )
    text = add_warning_after_heading(
        text,
        "## Deprecated legacy follow-up policy ablation bootstrap",
        """
> **Deprecated for main-text claims.** This section is retained for provenance only. The official follow-up result is the clean ensemble follow-up ablation based on final ensemble probabilities and robust feature-space novelty.
""",
    )
    text = add_warning_after_heading(
        text,
        "## Calibration bootstrap: raw vs temperature-scaled",
        """
> **Auxiliary calibration result.** This section refers to the standalone `hybrid_temporal_tabular_cnn`, not to a temperature-scaled final ensemble.
""",
    )

    summary.write_text(text, encoding="utf-8")

    print(f"Updated: {summary}")
    print(f"Backup created: {backup}")
    print("Inserted/replaced section:")
    print(AUTHORITATIVE_TITLE)
    print("Marked legacy/superseded sections for methodological consistency.")


if __name__ == "__main__":
    main()
