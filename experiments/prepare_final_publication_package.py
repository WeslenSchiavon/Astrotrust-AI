from __future__ import annotations

from pathlib import Path
import shutil
import json
import argparse
from datetime import datetime

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "final_publication"


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def copy_if_exists(src: Path, dst_dir: Path) -> Path | None:
    if not src.exists():
        print(f"[WARN] Missing: {src}")
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    print(f"[OK] Copied: {src} -> {dst}")
    return dst


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def first_row(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def find_row(df: pd.DataFrame | None, column: str, value: str) -> dict:
    if df is None or df.empty or column not in df.columns:
        return {}
    mask = df[column].astype(str) == str(value)
    if not mask.any():
        return {}
    return df.loc[mask].iloc[0].to_dict()


def format_pct(value):
    if value is None:
        return "—"
    return f"{100 * float(value):.2f}%"


def format_float(value, digits=4):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def make_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def prepare_directories(output_dir: Path):
    subdirs = {
        "model_performance": output_dir / "model_performance",
        "calibration": output_dir / "calibration",
        "followup_policy": output_dir / "followup_policy",
        "topk_family_analysis": output_dir / "topk_family_analysis",
        "hierarchical_analysis": output_dir / "hierarchical_analysis",
        "interface_case_studies": output_dir / "interface_case_studies",
        "broker_live_validation": output_dir / "broker_live_validation",
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "reports": output_dir / "reports",
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return subdirs


def collect_files(subdirs: dict[str, Path]):
    files_to_copy = [
        # Final AI summaries
        (RESULTS_DIR / "final_ai_summary_hybrid" / "final_model_performance_summary.csv", subdirs["model_performance"]),
        (RESULTS_DIR / "final_ai_summary_hybrid" / "final_hybrid_calibration_summary.csv", subdirs["calibration"]),
        (RESULTS_DIR / "final_ai_summary_hybrid" / "final_followup_policy_summary.csv", subdirs["followup_policy"]),
        (RESULTS_DIR / "final_ai_summary_hybrid" / "final_selected_followup_policies.csv", subdirs["followup_policy"]),
        (RESULTS_DIR / "final_ai_summary_hybrid" / "final_ai_summary.md", subdirs["reports"]),

        # Error / top-k / family analysis
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "error_analysis_metrics_summary.csv", subdirs["topk_family_analysis"]),
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "error_decomposition_summary.csv", subdirs["topk_family_analysis"]),
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "top_confused_class_pairs.csv", subdirs["topk_family_analysis"]),
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "per_class_classification_report.csv", subdirs["topk_family_analysis"]),
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "per_family_classification_report.csv", subdirs["topk_family_analysis"]),
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "error_analysis_summary.md", subdirs["reports"]),
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "confusion_matrix_32_classes.png", subdirs["figures"]),
        (RESULTS_DIR / "error_analysis_hierarchy_topk" / "confusion_matrix_families.png", subdirs["figures"]),

        # Hierarchical analysis
        (RESULTS_DIR / "hierarchical_upper_bound" / "hierarchical_upper_bound_summary.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hierarchical_upper_bound" / "hierarchical_upper_bound_summary.md", subdirs["reports"]),
        (RESULTS_DIR / "hierarchical_probability_reranking" / "hierarchical_reranking_grid.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hierarchical_probability_reranking" / "hierarchical_reranking_best_summary.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hierarchical_probability_reranking" / "hierarchical_reranking_summary.md", subdirs["reports"]),
        (RESULTS_DIR / "hierarchical_lightgbm_250k" / "hierarchical_lightgbm_summary.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hierarchical_lightgbm_250k" / "family_classifier_metrics.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hierarchical_lightgbm_250k" / "per_family_hierarchical_performance.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hierarchical_lightgbm_250k" / "hierarchical_lightgbm_summary.md", subdirs["reports"]),
        (RESULTS_DIR / "hybrid_family_classifier_250k" / "family_test_metrics.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hybrid_family_classifier_250k" / "family_classification_report.csv", subdirs["hierarchical_analysis"]),
        (RESULTS_DIR / "hybrid_family_classifier_250k" / "family_model_metadata.json", subdirs["hierarchical_analysis"]),

        # Final ensemble attempt
        (RESULTS_DIR / "final_probability_ensemble_search_250k" / "candidate_probability_files_metrics.csv", subdirs["model_performance"]),
        (RESULTS_DIR / "final_probability_ensemble_search_250k" / "best_ensemble_summary.csv", subdirs["model_performance"]),
        (RESULTS_DIR / "final_probability_ensemble_search_250k" / "best_ensemble_summary.json", subdirs["model_performance"]),
        (RESULTS_DIR / "final_probability_ensemble_search_250k" / "final_probability_ensemble_summary.md", subdirs["reports"]),

        # Interface case studies
        (RESULTS_DIR / "interface_case_studies_summary" / "interface_case_studies_summary.csv", subdirs["interface_case_studies"]),
        (RESULTS_DIR / "interface_case_studies_summary" / "interface_case_studies_summary.md", subdirs["reports"]),
    ]

    copied = []
    for src, dst in files_to_copy:
        out = copy_if_exists(src, dst)
        if out is not None:
            copied.append({"source": str(src), "destination": str(out)})

    return copied


def build_tables(subdirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    tables = {}

    model_perf = read_csv_if_exists(RESULTS_DIR / "final_ai_summary_hybrid" / "final_model_performance_summary.csv")
    if model_perf is not None:
        cols = [c for c in ["model", "family", "dataset", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"] if c in model_perf.columns]
        table = model_perf[cols].copy()
        tables["table_model_performance"] = table
        table.to_csv(subdirs["tables"] / "table_model_performance.csv", index=False)

    calibration = read_csv_if_exists(RESULTS_DIR / "final_ai_summary_hybrid" / "final_hybrid_calibration_summary.csv")
    if calibration is not None:
        cols = [c for c in ["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mean_confidence", "brier_score", "ece"] if c in calibration.columns]
        table = calibration[cols].copy()
        tables["table_calibration"] = table
        table.to_csv(subdirs["tables"] / "table_calibration.csv", index=False)

    followup = read_csv_if_exists(RESULTS_DIR / "final_ai_summary_hybrid" / "final_followup_policy_summary.csv")
    if followup is not None:
        cols = [
            c for c in [
                "configuration", "w_uncertainty", "w_novelty", "w_rarity",
                "top5_rare_enrichment", "top5_rare_rate", "top10_rare_enrichment",
                "top10_rare_rate", "weighted_rare_enrichment", "weighted_uncertainty", "weighted_novelty"
            ]
            if c in followup.columns
        ]
        table = followup[cols].copy()
        tables["table_followup_policy"] = table
        table.to_csv(subdirs["tables"] / "table_followup_policy.csv", index=False)

    topk_family = read_csv_if_exists(RESULTS_DIR / "error_analysis_hierarchy_topk" / "error_analysis_metrics_summary.csv")
    if topk_family is not None:
        table = topk_family.copy()
        tables["table_topk_family"] = table
        table.to_csv(subdirs["tables"] / "table_topk_family.csv", index=False)

    error_decomp = read_csv_if_exists(RESULTS_DIR / "error_analysis_hierarchy_topk" / "error_decomposition_summary.csv")
    if error_decomp is not None:
        table = error_decomp.copy()
        tables["table_error_decomposition"] = table
        table.to_csv(subdirs["tables"] / "table_error_decomposition.csv", index=False)

    hierarchical_upper = read_csv_if_exists(RESULTS_DIR / "hierarchical_upper_bound" / "hierarchical_upper_bound_summary.csv")
    if hierarchical_upper is not None:
        table = hierarchical_upper.copy()
        tables["table_hierarchical_upper_bound"] = table
        table.to_csv(subdirs["tables"] / "table_hierarchical_upper_bound.csv", index=False)

    hierarchical_lgbm = read_csv_if_exists(RESULTS_DIR / "hierarchical_lightgbm_250k" / "hierarchical_lightgbm_summary.csv")
    if hierarchical_lgbm is not None:
        table = hierarchical_lgbm.copy()
        tables["table_hierarchical_lightgbm"] = table
        table.to_csv(subdirs["tables"] / "table_hierarchical_lightgbm.csv", index=False)

    family_hybrid = read_csv_if_exists(RESULTS_DIR / "hybrid_family_classifier_250k" / "family_test_metrics.csv")
    if family_hybrid is not None:
        table = family_hybrid.copy()
        tables["table_hybrid_family_classifier"] = table
        table.to_csv(subdirs["tables"] / "table_hybrid_family_classifier.csv", index=False)

    final_ensemble = read_csv_if_exists(RESULTS_DIR / "final_probability_ensemble_search_250k" / "best_ensemble_summary.csv")
    if final_ensemble is not None:
        table = final_ensemble.copy()
        tables["table_final_ensemble_attempt"] = table
        table.to_csv(subdirs["tables"] / "table_final_ensemble_attempt.csv", index=False)

    interface_cases = read_csv_if_exists(RESULTS_DIR / "interface_case_studies_summary" / "interface_case_studies_summary.csv")
    if interface_cases is not None:
        table = interface_cases.copy()
        tables["table_interface_case_studies"] = table
        table.to_csv(subdirs["tables"] / "table_interface_case_studies.csv", index=False)

    return tables


def build_key_numbers(tables: dict[str, pd.DataFrame], subdirs: dict[str, Path]) -> pd.DataFrame:
    rows = []

    model_perf = tables.get("table_model_performance")
    if model_perf is not None and not model_perf.empty:
        best_raw = model_perf.iloc[0].to_dict()
        rows.append({
            "category": "model_performance",
            "metric": "best_raw_model",
            "value": best_raw.get("model"),
            "detail": f"accuracy={format_float(best_raw.get('accuracy'))}, macro_f1={format_float(best_raw.get('macro_f1'))}",
        })

        calibrated_row = None
        if "model" in model_perf.columns:
            mask = model_perf["model"].astype(str).str.contains("hybrid_temporal_tabular_cnn", case=False, na=False)
            if mask.any():
                calibrated_row = model_perf.loc[mask].iloc[0].to_dict()
        if calibrated_row:
            rows.append({
                "category": "model_performance",
                "metric": "hybrid_neural_model",
                "value": calibrated_row.get("model"),
                "detail": f"accuracy={format_float(calibrated_row.get('accuracy'))}, macro_f1={format_float(calibrated_row.get('macro_f1'))}",
            })

    calibration = tables.get("table_calibration")
    if calibration is not None and not calibration.empty:
        temp_test = None
        if "model" in calibration.columns:
            mask = calibration["model"].astype(str).str.contains("temp_test", case=False, na=False)
            if mask.any():
                temp_test = calibration.loc[mask].iloc[0].to_dict()
        if temp_test is None:
            temp_test = calibration.iloc[-1].to_dict()
        rows.append({
            "category": "calibration",
            "metric": "temperature_scaled_test_ece",
            "value": safe_float(temp_test.get("ece")),
            "detail": f"brier={format_float(temp_test.get('brier_score'))}, mean_confidence={format_float(temp_test.get('mean_confidence'))}",
        })

    followup = tables.get("table_followup_policy")
    if followup is not None and not followup.empty:
        novelty_rarity = None
        if "configuration" in followup.columns:
            mask = followup["configuration"].astype(str) == "novelty_rarity"
            if mask.any():
                novelty_rarity = followup.loc[mask].iloc[0].to_dict()
        if novelty_rarity is None:
            novelty_rarity = followup.iloc[0].to_dict()
        rows.append({
            "category": "followup_policy",
            "metric": "selected_policy_weighted_rare_enrichment",
            "value": safe_float(novelty_rarity.get("weighted_rare_enrichment")),
            "detail": f"configuration={novelty_rarity.get('configuration')}, top5={format_float(novelty_rarity.get('top5_rare_enrichment'))}",
        })

    topk_family = tables.get("table_topk_family")
    if topk_family is not None and not topk_family.empty:
        fine = find_row(topk_family, "task", "32-class fine labels")
        family = find_row(topk_family, "task", "coarse astronomical families")
        if fine:
            for metric in ["accuracy", "macro_f1", "top3_accuracy", "top5_accuracy"]:
                if metric in fine:
                    rows.append({
                        "category": "topk_family_analysis",
                        "metric": f"fine_{metric}",
                        "value": safe_float(fine.get(metric)),
                        "detail": "32-class fine labels",
                    })
        if family:
            for metric in ["accuracy", "macro_f1", "top3_family_accuracy", "top5_family_accuracy"]:
                if metric in family:
                    rows.append({
                        "category": "topk_family_analysis",
                        "metric": f"family_{metric}",
                        "value": safe_float(family.get(metric)),
                        "detail": "coarse astronomical families",
                    })

    error_decomp = tables.get("table_error_decomposition")
    if error_decomp is not None and not error_decomp.empty:
        row = error_decomp.iloc[0].to_dict()
        for metric in ["fraction_errors_same_family", "fraction_errors_cross_family", "family_accuracy", "fine_accuracy"]:
            if metric in row:
                rows.append({
                    "category": "error_decomposition",
                    "metric": metric,
                    "value": safe_float(row.get(metric)),
                    "detail": f"n_fine_errors={row.get('n_fine_errors')}",
                })

    hierarchical_upper = tables.get("table_hierarchical_upper_bound")
    if hierarchical_upper is not None and not hierarchical_upper.empty:
        oracle = find_row(hierarchical_upper, "method", "oracle_true_family_then_subclass")
        if oracle:
            rows.append({
                "category": "hierarchical_analysis",
                "metric": "oracle_family_fine_accuracy",
                "value": safe_float(oracle.get("fine_accuracy")),
                "detail": f"macro_f1={format_float(oracle.get('fine_macro_f1'))}",
            })

    family_hybrid = tables.get("table_hybrid_family_classifier")
    if family_hybrid is not None and not family_hybrid.empty:
        row = family_hybrid.iloc[0].to_dict()
        rows.append({
            "category": "hierarchical_analysis",
            "metric": "hybrid_family_classifier_accuracy",
            "value": safe_float(row.get("accuracy")),
            "detail": f"macro_f1={format_float(row.get('macro_f1'))}",
        })

    final_ensemble = tables.get("table_final_ensemble_attempt")
    if final_ensemble is not None and not final_ensemble.empty:
        row = final_ensemble.iloc[0].to_dict()
        rows.append({
            "category": "final_ensemble_attempt",
            "metric": "exploratory_probability_ensemble_accuracy",
            "value": safe_float(row.get("eval_accuracy")),
            "detail": f"macro_f1={format_float(row.get('eval_macro_f1'))}, exploratory={row.get('is_test_tuned_exploratory')}",
        })

    key_df = pd.DataFrame(rows)
    key_df.to_csv(subdirs["tables"] / "final_publication_key_numbers.csv", index=False)
    return key_df


def build_markdown_summary(output_dir: Path, tables: dict[str, pd.DataFrame], key_numbers: pd.DataFrame, copied_files: list[dict]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = "# AstroTrust-AI Final Publication Package\n\n"
    md += f"Generated at: `{now}`\n\n"

    md += "## Core positioning\n\n"
    md += (
        "AstroTrust-AI should be positioned as a calibrated, novelty-aware, broker-like framework "
        "for LSST-like astronomical alert triage, not as a universal classifier for arbitrary real-survey light curves. "
        "The main contribution combines hybrid temporal-tabular classification, probabilistic calibration, uncertainty, "
        "novelty, rarity-aware follow-up prioritization, explicit reliability assessment, and a broker-like interface.\n\n"
    )

    md += "## Key numbers\n\n"
    md += make_markdown_table(key_numbers)
    md += "\n\n"

    if "table_model_performance" in tables:
        md += "## Table: model performance\n\n"
        md += make_markdown_table(tables["table_model_performance"])
        md += "\n\n"

    if "table_calibration" in tables:
        md += "## Table: calibration\n\n"
        md += make_markdown_table(tables["table_calibration"])
        md += "\n\n"

    if "table_topk_family" in tables:
        md += "## Table: top-k and family-level performance\n\n"
        md += make_markdown_table(tables["table_topk_family"])
        md += "\n\n"

    if "table_followup_policy" in tables:
        md += "## Table: follow-up policy\n\n"
        md += make_markdown_table(tables["table_followup_policy"])
        md += "\n\n"

    if "table_hierarchical_upper_bound" in tables:
        md += "## Table: hierarchical upper-bound analysis\n\n"
        md += make_markdown_table(tables["table_hierarchical_upper_bound"])
        md += "\n\n"

    if "table_interface_case_studies" in tables:
        md += "## Table: interface case studies\n\n"
        md += make_markdown_table(tables["table_interface_case_studies"])
        md += "\n\n"

    md += "## Suggested manuscript claims\n\n"
    md += (
        "1. The best raw-performing model is an ensemble hybrid model, while the calibrated hybrid temporal-tabular CNN "
        "provides the most scientifically interpretable probability estimates.\n"
        "2. Fine-grained 32-class top-1 accuracy is moderate, but top-k and family-level performance are strong for broker-like triage.\n"
        "3. The rarity-aware follow-up policy substantially enriches rare-class candidates under limited observing budgets.\n"
        "4. Real ZTF/IRSA examples should be presented as ingestion and out-of-domain reliability demonstrations, not as final real-survey validation.\n"
        "5. The interface provides operational broker-like ranking with separate queues for validated follow-up, OOD/anomaly review, and low-priority candidates.\n\n"
    )

    md += "## Limitations to state explicitly\n\n"
    md += (
        "- The primary validation domain is ELAsTiCC/LSST-like data.\n"
        "- Direct transfer to real ZTF or other surveys requires additional validation or domain adaptation.\n"
        "- High model confidence does not imply scientific reliability when novelty/domain mismatch is high.\n"
        "- Missing astrophysical context features are not invented; they are reported as missing/filled placeholders for model compatibility.\n"
        "- The model should support follow-up triage rather than replace expert/observational confirmation.\n\n"
    )

    md += "## Copied evidence files\n\n"
    copied_df = pd.DataFrame(copied_files)
    if copied_df.empty:
        md += "_No files copied._\n"
    else:
        md += make_markdown_table(copied_df)
        md += "\n"

    summary_path = output_dir / "final_publication_summary.md"
    summary_path.write_text(md, encoding="utf-8")
    print(f"[OK] Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare final publication package for AstroTrust-AI.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean", action="store_true", help="Remove existing final_publication directory before regenerating.")
    args = parser.parse_args()

    output_dir = args.output_dir

    if args.clean and output_dir.exists():
        print(f"[INFO] Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing final publication package in: {output_dir}")
    subdirs = prepare_directories(output_dir)
    copied_files = collect_files(subdirs)
    tables = build_tables(subdirs)
    key_numbers = build_key_numbers(tables, subdirs)
    build_markdown_summary(output_dir, tables, key_numbers, copied_files)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "root_dir": str(ROOT_DIR),
        "output_dir": str(output_dir),
        "copied_files": copied_files,
        "tables": list(tables.keys()),
    }
    manifest_path = output_dir / "final_publication_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Saved manifest: {manifest_path}")

    print("\nKey numbers:")
    print(key_numbers.to_string(index=False) if not key_numbers.empty else "No key numbers generated.")

    print("\nFinal package ready.")
    print(f"Open: {output_dir / 'final_publication_summary.md'}")


if __name__ == "__main__":
    main()
