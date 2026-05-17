#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
refresh_model_usage_audit_after_ensemble_calibration.py

Atualiza frases antigas no final_publication_summary.md depois que a calibração
calibration-safe do ensemble foi gerada e consolidada.

Uso:
python .\experiments\refresh_model_usage_audit_after_ensemble_calibration.py `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


OLD_BULLET = (
    "- Temperature scaling should be described as an auxiliary calibration analysis of the standalone "
    "`hybrid_temporal_tabular_cnn`, unless a separate validation-set calibration artifact for the final ensemble is generated."
)

NEW_BULLET = (
    "- Temperature scaling should now be described in two levels: the calibration-safe ensemble rebuild provides the official ensemble-level calibration evidence, while the standalone `hybrid_temporal_tabular_cnn` temperature-scaling experiment remains auxiliary supporting evidence."
)

OLD_LATEX = (
    "To avoid ambiguity between model variants, we explicitly distinguish the final operational classifier from auxiliary analyses. "
    "The primary AstroTrust-AI classification, top-$k$, family-level, uncertainty, selective-prediction, follow-up-prioritization, and hierarchical analyses are based on the final \\texttt{ensemble\\_hybrid\\_dominant} configuration. "
    "Temperature scaling is reported as an auxiliary calibration experiment for the standalone \\texttt{hybrid\\_temporal\\_tabular\\_cnn}, because no separate validation-set probability artifact was available to fit a publication-safe temperature parameter for the final ensemble. "
    "Accordingly, calibrated-probability claims are restricted to the standalone hybrid calibration experiment, whereas final broker-like triage claims rely on the raw final ensemble probabilities and the clean feature-space novelty analysis."
)

NEW_LATEX = (
    "To avoid ambiguity between model variants, we explicitly distinguish the final operational classifier from auxiliary analyses. "
    "The primary AstroTrust-AI classification, top-$k$, family-level, uncertainty, selective-prediction, follow-up-prioritization, and hierarchical analyses are based on the final \\texttt{ensemble\\_hybrid\\_dominant} configuration. "
    "For probabilistic calibration, we report a calibration-safe ensemble rebuild using the same dominant ensemble weights, with tabular members trained only on the training split, the temperature parameter fitted exclusively on the validation split, and final calibration metrics evaluated on the held-out test split. "
    "The standalone \\texttt{hybrid\\_temporal\\_tabular\\_cnn} temperature-scaling experiment is retained as auxiliary supporting evidence, whereas final broker-like triage claims rely on the ensemble-based analyses and the clean feature-space novelty analysis."
)


def replace_model_usage_rows(text: str) -> str:
    # Replace compact/full audit table rows when they still state pending/raw ensemble calibration.
    lines = text.splitlines()
    out = []
    for line in lines:
        if line.startswith("| Calibration via temperature scaling") and "hybrid_temporal_tabular_cnn" in line:
            out.append(
                "| Calibration-safe ensemble temperature scaling | Calibration evidence for ensemble probabilities | calibration-safe `ensemble_hybrid_dominant` rebuild | standalone `hybrid_temporal_tabular_cnn` calibration retained as auxiliary evidence | ensemble-level calibration is valid for the calibration-safe rebuild; do not claim the original train+validation tabular ensemble was directly temperature-fitted | main_calibration_evidence |"
            )
            continue

        if line.startswith("| Raw ensemble calibration audit") and "pending" in line:
            out.append(
                "| Calibration-safe ensemble rebuild audit | Completed calibration evidence | calibration-safe `ensemble_hybrid_dominant` rebuild | previous raw official ensemble probabilities used only for comparison | validation probabilities were generated through a calibration-safe rebuild; temperature was fitted only on validation and evaluated on test | completed |"
            )
            continue

        if line.startswith("| Calibration via temperature scaling") and "temperature_scaled_hybrid_250k" in line:
            out.append(
                "| Calibration-safe ensemble temperature scaling | Calibration evidence for ensemble probabilities | calibration-safe `ensemble_hybrid_dominant` rebuild | standalone `hybrid_temporal_tabular_cnn` calibration retained as auxiliary evidence | calibration_safe_ensemble_temperature/calibration_safe_ensemble_temperature_summary.md [yes]; calibration_safe_ensemble_temperature/calibration_safe_ensemble_temperature_point_metrics.csv [yes]; calibration_safe_ensemble_temperature/calibration_safe_ensemble_temperature_bootstrap_deltas.csv [yes] | Use the calibration-safe ensemble section as the official ensemble-level calibration evidence. The hybrid-only temperature scaling remains auxiliary. | main_calibration_evidence |"
            )
            continue

        if line.startswith("| Raw ensemble calibration audit") and "ensemble_calibration_audit_250k_final" in line:
            out.append(
                "| Calibration-safe ensemble rebuild audit | Completed calibration evidence | calibration-safe `ensemble_hybrid_dominant` rebuild | previous raw official ensemble probabilities used only for comparison | calibration_safe_ensemble_temperature/calibration_safe_ensemble_temperature_schema.json [yes]; calibration_safe_ensemble_temperature/fig_calibration_safe_ensemble_reliability.png [yes] | This replaces the earlier pending raw-audit status. Temperature was fitted on validation and evaluated on test. | completed |"
            )
            continue

        out.append(line)
    return "\n".join(out)


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
    backup = summary.with_name(f"final_publication_summary.before_calibration_audit_refresh_{timestamp}.md")
    backup.write_text(original, encoding="utf-8")

    text = original
    text = text.replace(OLD_BULLET, NEW_BULLET)
    text = text.replace(OLD_LATEX, NEW_LATEX)
    text = replace_model_usage_rows(text)

    # Add a short note after the model-usage audit title if not present.
    note = (
        "> **Updated after ensemble calibration.** The earlier pending ensemble-calibration caveat is superseded by the "
        "`Calibration-safe ensemble temperature scaling` section, which generated validation probabilities through a "
        "calibration-safe rebuild and fitted temperature only on the validation split."
    )
    title = "## Model-usage audit and methodological consistency"
    if title in text and note not in text:
        text = text.replace(title, title + "\n\n" + note, 1)

    summary.write_text(text, encoding="utf-8")

    print(f"Updated: {summary}")
    print(f"Backup created: {backup}")
    print("Refreshed model-usage audit after calibration-safe ensemble temperature scaling.")


if __name__ == "__main__":
    main()
