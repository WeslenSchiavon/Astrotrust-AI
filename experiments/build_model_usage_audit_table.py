#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
build_model_usage_audit_table.py

Cria e consolida uma tabela de auditoria metodológica para o artigo AstroTrust-AI.

Objetivo:
- Explicitar qual modelo sustenta cada bloco de resultado.
- Separar resultados principais, auxiliares, diagnósticos e operacionais.
- Evitar ambiguidade entre ensemble final e hybrid standalone calibrado.

Uso:
python .\experiments\build_model_usage_audit_table.py `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SECTION_TITLE = "## Model-usage audit and methodological consistency"


def exists(path: Path) -> str:
    return "yes" if path.exists() else "no"


def make_row(
    analysis_block: str,
    manuscript_role: str,
    primary_model: str,
    secondary_or_auxiliary_model: str,
    evidence_files: list[str],
    interpretation_constraint: str,
    publication_status: str,
    final_dir: Path,
) -> dict:
    evidence_status = []
    for e in evidence_files:
        p = final_dir / e
        evidence_status.append(f"{e} [{exists(p)}]")
    return {
        "analysis_block": analysis_block,
        "manuscript_role": manuscript_role,
        "primary_model_or_artifact": primary_model,
        "secondary_or_auxiliary_model": secondary_or_auxiliary_model,
        "evidence_files_in_final_publication": "; ".join(evidence_status),
        "interpretation_constraint": interpretation_constraint,
        "publication_status": publication_status,
    }


def build_audit(final_dir: Path) -> pd.DataFrame:
    rows = []

    rows.append(make_row(
        analysis_block="Final model selection and headline classification performance",
        manuscript_role="Primary result",
        primary_model="ensemble_hybrid_dominant",
        secondary_or_auxiliary_model="hybrid_temporal_tabular_cnn, LightGBM, XGBoost, other baselines for comparison only",
        evidence_files=[
            "statistical_validation/paired_prediction_level_tests.csv",
            "statistical_validation/paired_seed_level_tests.csv",
            "bootstrap_final_250k/mcnemar_tests_final_250k.csv",
        ],
        interpretation_constraint="The final AstroTrust-AI classifier should be described as the ensemble_hybrid_dominant configuration, not the standalone hybrid model.",
        publication_status="main_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Top-k and family-level performance",
        manuscript_role="Primary result",
        primary_model="ensemble_hybrid_dominant",
        secondary_or_auxiliary_model="none",
        evidence_files=[
            "topk_family_bootstrap/topk_family_bootstrap_summary.md",
            "topk_family_bootstrap/topk_family_bootstrap_metrics.csv",
            "topk_family_bootstrap/topk_family_object_level_indicators.csv",
        ],
        interpretation_constraint="Fine top-k and family-level metrics must be reported from the final ensemble analysis, not from the standalone hybrid analysis.",
        publication_status="main_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Uncertainty, error detection, and selective prediction",
        manuscript_role="Primary result",
        primary_model="ensemble_hybrid_dominant",
        secondary_or_auxiliary_model="none",
        evidence_files=[
            "ensemble_uncertainty_selective_error/ensemble_uncertainty_selective_error_summary.md",
            "ensemble_uncertainty_selective_error/ensemble_error_detection_bootstrap.csv",
            "ensemble_uncertainty_selective_error/ensemble_selective_key_coverages_bootstrap.csv",
            "ensemble_uncertainty_selective_error/fig_ensemble_risk_coverage_curve.png",
        ],
        interpretation_constraint="Confidence and uncertainty claims should refer to the final ensemble probabilities.",
        publication_status="main_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Follow-up prioritization and policy ablation",
        manuscript_role="Primary result",
        primary_model="ensemble_hybrid_dominant + robust feature-space novelty",
        secondary_or_auxiliary_model="none",
        evidence_files=[
            "clean_ensemble_followup_policy_ablation/clean_followup_policy_ablation_summary.md",
            "clean_ensemble_followup_policy_ablation/clean_followup_policy_summary_by_budget.csv",
            "clean_ensemble_followup_policy_ablation/clean_followup_policy_pairwise_vs_reference.csv",
            "clean_ensemble_followup_policy_ablation/fig_clean_policy_rare_enrichment_by_budget.png",
            "clean_ensemble_followup_policy_ablation/fig_clean_policy_novelty_by_budget.png",
        ],
        interpretation_constraint="Use the clean follow-up analysis as official. Older hybrid-template follow-up rankings should be treated as deprecated or exploratory.",
        publication_status="main_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Hierarchical upper-bound analysis",
        manuscript_role="Primary result / future-work evidence",
        primary_model="ensemble_hybrid_dominant",
        secondary_or_auxiliary_model="oracle_true_family_then_subclass is an upper bound, not a deployable model",
        evidence_files=[
            "ensemble_hierarchical_upper_bound/hierarchical_upper_bound_summary.md",
            "ensemble_hierarchical_upper_bound/hierarchical_bootstrap_accuracy.csv",
            "ensemble_hierarchical_upper_bound/hierarchical_paired_bootstrap_deltas_vs_original.csv",
            "ensemble_hierarchical_upper_bound/fig_hierarchical_upper_bound_two_panel.png",
        ],
        interpretation_constraint="Naive family-mass reranking should not be claimed as improving fine classification; the key claim is oracle headroom for future hierarchical modeling.",
        publication_status="main_claim_with_caveat",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Calibration via temperature scaling",
        manuscript_role="Auxiliary result",
        primary_model="hybrid_temporal_tabular_cnn",
        secondary_or_auxiliary_model="ensemble_hybrid_dominant raw calibration may be reported separately only if audited without test-set fitting",
        evidence_files=[
            "temperature_scaled_hybrid_250k/temperature_scaled_hybrid_report.md",
            "temperature_scaled_hybrid_250k/temperature_scaled_hybrid_metrics.csv",
            "temperature_scaled_hybrid_250k/hybrid_temporal_tabular_cnn_temperature_scaled_test_predictions.csv",
        ],
        interpretation_constraint="Do not state that the final ensemble was temperature-calibrated unless a separate validation-set ensemble calibration artifact is produced. This section should be explicitly labeled as standalone hybrid calibration.",
        publication_status="auxiliary_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Raw ensemble calibration audit",
        manuscript_role="Optional diagnostic / pending",
        primary_model="ensemble_hybrid_dominant",
        secondary_or_auxiliary_model="none",
        evidence_files=[
            "ensemble_calibration_audit_250k_final/ensemble_calibration_audit_summary.md",
            "ensemble_calibration_audit_250k_final/ensemble_calibration_bootstrap.csv",
            "ensemble_calibration_audit_250k_final/fig_ensemble_reliability_diagram.png",
        ],
        interpretation_constraint="If this directory is absent, the paper should not make calibrated-ensemble temperature-scaling claims. Raw calibration can still be described if the audit is later generated.",
        publication_status="pending_or_diagnostic",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Multi-seed stability and paired statistical validation",
        manuscript_role="Primary robustness evidence",
        primary_model="ensemble_hybrid_dominant compared against hybrid_temporal_tabular_cnn",
        secondary_or_auxiliary_model="multiple seeds; baselines for comparison",
        evidence_files=[
            "statistical_validation/paired_seed_level_tests.csv",
            "statistical_validation/paired_prediction_level_tests.csv",
            "statistical_validation/rare_enrichment_permutation_tests.csv",
        ],
        interpretation_constraint="Use as robustness evidence for the ensemble advantage and selected downstream analyses; keep metrics aligned with the final model where applicable.",
        publication_status="main_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Broker-like interface and software artifacts",
        manuscript_role="Operational validation",
        primary_model="AstroTrust-AI pipeline/interface",
        secondary_or_auxiliary_model="model outputs used for case studies",
        evidence_files=[
            "interface_case_studies/interface_case_studies_summary.md",
            "software_artifacts/software_artifacts_summary.md",
            "broker_like_validation/broker_like_validation_summary.md",
        ],
        interpretation_constraint="Interface case studies validate ingestion, reporting, and broker-like workflow; they should not be framed as independent accuracy benchmarks.",
        publication_status="operational_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="ZTF/IRSA real-object demonstrations",
        manuscript_role="Out-of-domain operational demonstration",
        primary_model="AstroTrust-AI inference pipeline",
        secondary_or_auxiliary_model="ZTF/IRSA ingestion and reliability/OOD diagnostics",
        evidence_files=[
            "ztf_reliability/ztf_reliability_summary.md",
            "interface_case_studies/case3_ztf_multiband_context.json",
            "figures/fig_ztf_reliability.png",
        ],
        interpretation_constraint="ZTF examples should be described as real-data ingestion/OOD demonstrations, not as the main survey-domain validation of classifier accuracy.",
        publication_status="demonstration_claim",
        final_dir=final_dir,
    ))

    rows.append(make_row(
        analysis_block="Deprecated or superseded follow-up analyses",
        manuscript_role="Do not use as main result",
        primary_model="hybrid_temporal_tabular_cnn or ensemble reconstructed from hybrid ranking template",
        secondary_or_auxiliary_model="older follow-up rankings",
        evidence_files=[
            "followup_policy_ablation/followup_policy_ablation_summary.md",
            "ensemble_followup_policy_ablation/followup_policy_ablation_summary.md",
        ],
        interpretation_constraint="These may remain in the repository for provenance, but the manuscript should cite the clean ensemble follow-up analysis instead.",
        publication_status="deprecated_for_main_text",
        final_dir=final_dir,
    ))

    return pd.DataFrame(rows)


def build_section(df: pd.DataFrame, out_dir: Path) -> str:
    main_df = df.copy()

    # Compact view for the summary file.
    compact_cols = [
        "analysis_block",
        "manuscript_role",
        "primary_model_or_artifact",
        "secondary_or_auxiliary_model",
        "publication_status",
        "interpretation_constraint",
    ]
    compact = main_df[compact_cols].copy()

    section = []
    section.append(SECTION_TITLE)
    section.append("")
    section.append("This audit table resolves model-use consistency across the AstroTrust-AI manuscript. It separates primary ensemble-based claims from auxiliary, diagnostic, operational, and deprecated analyses.")
    section.append("")
    section.append("### Compact audit table")
    section.append("")
    section.append(compact.to_markdown(index=False))
    section.append("")
    section.append("### Full evidence table")
    section.append("")
    section.append(main_df.to_markdown(index=False))
    section.append("")
    section.append("### Required manuscript interpretation")
    section.append("")
    section.append("- The final AstroTrust-AI classifier should be consistently described as `ensemble_hybrid_dominant`.")
    section.append("- Top-k, family-level, uncertainty, selective prediction, clean follow-up prioritization, and hierarchical headroom results should use the final ensemble analyses.")
    section.append("- Temperature scaling should be described as an auxiliary calibration analysis of the standalone `hybrid_temporal_tabular_cnn`, unless a separate validation-set calibration artifact for the final ensemble is generated.")
    section.append("- Older follow-up analyses based on hybrid rankings or hybrid-derived templates should not be used as the official manuscript result after the clean ensemble follow-up ablation.")
    section.append("- Interface and ZTF/IRSA sections should be framed as operational and out-of-domain demonstrations, not as primary survey-domain accuracy benchmarks.")
    section.append("")
    section.append("### Suggested manuscript wording")
    section.append("")
    section.append("```latex")
    section.append(
        "To avoid ambiguity between model variants, we explicitly distinguish the final operational classifier from auxiliary analyses. "
        "The primary AstroTrust-AI classification, top-$k$, family-level, uncertainty, selective-prediction, follow-up-prioritization, and hierarchical analyses are based on the final \\texttt{ensemble\\_hybrid\\_dominant} configuration. "
        "Temperature scaling is reported as an auxiliary calibration experiment for the standalone \\texttt{hybrid\\_temporal\\_tabular\\_cnn}, because no separate validation-set probability artifact was available to fit a publication-safe temperature parameter for the final ensemble. "
        "Accordingly, calibrated-probability claims are restricted to the standalone hybrid calibration experiment, whereas final broker-like triage claims rely on the raw final ensemble probabilities and the clean feature-space novelty analysis."
    )
    section.append("```")
    section.append("")
    section.append("### Output files")
    section.append("")
    for name in [
        "model_usage_audit_table.csv",
        "model_usage_audit_table.md",
        "model_usage_audit_schema.json",
    ]:
        section.append(f"- `{out_dir / name}`")
    section.append("")
    return "\n".join(section)


def insert_or_replace(text: str, section: str) -> str:
    pattern = rf"{re.escape(SECTION_TITLE)}\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _m: section, text, flags=re.S)

    anchors = [
        "## Ensemble hierarchical upper-bound analysis",
        "## Clean ensemble follow-up policy ablation",
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
    parser.add_argument("--final-publication-dir", required=True)
    args = parser.parse_args()

    final_dir = Path(args.final_publication_dir)
    final_summary = final_dir / "final_publication_summary.md"
    out_dir = final_dir / "model_usage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not final_dir.exists():
        raise FileNotFoundError(f"Final publication directory not found: {final_dir}")
    if not final_summary.exists():
        raise FileNotFoundError(f"Final publication summary not found: {final_summary}")

    audit = build_audit(final_dir)
    audit_csv = out_dir / "model_usage_audit_table.csv"
    audit_md = out_dir / "model_usage_audit_table.md"
    schema_json = out_dir / "model_usage_audit_schema.json"

    audit.to_csv(audit_csv, index=False)

    md = []
    md.append("# Model-usage audit table\n")
    md.append("This table records which model/artifact supports each manuscript result block.\n")
    md.append(audit.to_markdown(index=False))
    audit_md.write_text("\n".join(md), encoding="utf-8")

    schema = {
        "analysis": "model-usage audit",
        "purpose": "Resolve methodological consistency across AstroTrust-AI manuscript sections.",
        "final_model": "ensemble_hybrid_dominant",
        "auxiliary_calibrated_model": "hybrid_temporal_tabular_cnn",
        "created_files": [str(audit_csv), str(audit_md), str(schema_json)],
        "notes": [
            "Use ensemble_hybrid_dominant for primary classifier claims.",
            "Use clean ensemble follow-up ablation as official follow-up result.",
            "Treat hybrid temperature scaling as auxiliary unless ensemble validation calibration is generated.",
        ],
    }
    schema_json.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    section = build_section(audit, out_dir)
    text = final_summary.read_text(encoding="utf-8")
    updated = insert_or_replace(text, section)
    final_summary.write_text(updated, encoding="utf-8")

    print(f"Updated: {final_summary}")
    print(f"Created: {audit_csv}")
    print(f"Created: {audit_md}")
    print(f"Created: {schema_json}")
    print("Inserted/replaced section:")
    print(SECTION_TITLE)


if __name__ == "__main__":
    main()
