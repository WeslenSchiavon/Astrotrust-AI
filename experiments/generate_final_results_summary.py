from pathlib import Path
import pandas as pd
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT_DIR / Path("results/final_summary")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


PATHS = {
    "model_comparison": ROOT_DIR / Path(
        "results/model_comparison_v2_features/model_comparison_v2_features_metrics.csv"
    ),
    "calibration": ROOT_DIR / Path(
        "results/calibrated_hgb_v3/calibration_comparison_summary.csv"
    ),
    "followup_final": ROOT_DIR / Path(
        "results/followup_prioritization_v3_calibrated_hgb/followup_enrichment_vs_random.csv"
    ),
    "followup_robust": ROOT_DIR / Path(
        "results/followup_prioritization_v3_calibrated_hgb_robust_novelty/robust_novelty_priority_comparison.csv"
    ),
    "novelty_mahalanobis": ROOT_DIR / Path(
        "results/novelty_detection_mahalanobis/novelty_detection_mahalanobis_summary.csv"
    ),
    "novelty_isolation": ROOT_DIR / Path(
        "results/novelty_detection_leave_one_class_out/novelty_detection_summary.csv"
    ),
}


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    return pd.read_csv(path)


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(digits)
    return df


def df_to_markdown(df: pd.DataFrame) -> str:
    """
    Simple markdown table writer without requiring tabulate.
    """
    if df is None or len(df) == 0:
        return "_No data available._\n"

    df = df.copy()
    df = df.astype(str)

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(row.values.tolist()) + " |")

    return "\n".join(lines) + "\n"


def build_classification_summary() -> pd.DataFrame | None:
    df = read_csv_if_exists(PATHS["model_comparison"])
    if df is None:
        return None

    keep_cols = [
        "feature_set",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "n_features",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    summary = df[keep_cols].copy()
    summary = summary.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)

    return round_numeric(summary)


def build_best_classification_per_feature_set() -> pd.DataFrame | None:
    df = read_csv_if_exists(PATHS["model_comparison"])
    if df is None:
        return None

    best = (
        df.sort_values(["feature_set", "macro_f1", "balanced_accuracy"], ascending=[True, False, False])
        .groupby("feature_set", as_index=False)
        .head(1)
        .copy()
    )

    keep_cols = [
        "feature_set",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "n_features",
    ]

    keep_cols = [c for c in keep_cols if c in best.columns]

    best = best[keep_cols].sort_values("macro_f1", ascending=False)

    return round_numeric(best)


def build_calibration_summary() -> pd.DataFrame | None:
    df = read_csv_if_exists(PATHS["calibration"])
    if df is None:
        return None

    keep_cols = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "brier_score",
        "ece",
        "mean_confidence",
        "mean_uncertainty",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    summary = df[keep_cols].copy()
    summary = summary.sort_values(["ece", "macro_f1"], ascending=[True, False])

    return round_numeric(summary)


def build_followup_summary() -> pd.DataFrame | None:
    df = read_csv_if_exists(PATHS["followup_final"])
    if df is None:
        return None

    keep_cols = [
        "budget_fraction",
        "n_selected",
        "top_accuracy",
        "random_accuracy_mean",
        "top_error_rate",
        "random_error_rate_mean",
        "top_rare_true_rate",
        "random_rare_true_rate_mean",
        "rare_enrichment",
        "top_mean_uncertainty",
        "random_mean_uncertainty",
        "top_mean_novelty",
        "random_mean_novelty",
        "top_mean_priority",
        "random_mean_priority",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    summary = df[keep_cols].copy()
    summary = summary.sort_values("budget_fraction")

    return round_numeric(summary)


def build_robust_followup_summary() -> pd.DataFrame | None:
    df = read_csv_if_exists(PATHS["followup_robust"])
    if df is None:
        return None

    keep_cols = [
        "config",
        "budget_fraction",
        "n_selected",
        "top_rare_true_rate",
        "random_rare_true_rate_mean",
        "rare_enrichment",
        "top_accuracy",
        "random_accuracy_mean",
        "top_mean_uncertainty",
        "random_mean_uncertainty",
        "top_mean_novelty",
        "random_mean_novelty",
        "top_mean_priority",
        "random_mean_priority",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    summary = df[keep_cols].copy()
    summary = summary.sort_values(["budget_fraction", "rare_enrichment"], ascending=[True, False])

    return round_numeric(summary)


def build_novelty_summary() -> pd.DataFrame | None:
    mah = read_csv_if_exists(PATHS["novelty_mahalanobis"])
    iso = read_csv_if_exists(PATHS["novelty_isolation"])

    frames = []

    if mah is not None:
        mah = mah.copy()
        mah["method"] = "mahalanobis"
        frames.append(mah)

    if iso is not None:
        iso = iso.copy()
        iso["method"] = "isolation_forest"
        frames.append(iso)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    keep_cols = [
        "method",
        "held_out_class",
        "n_ood_test",
        "roc_auc",
        "average_precision",
        "recall_at_top_10_percent",
        "mean_novelty_known",
        "mean_novelty_ood",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    summary = df[keep_cols].copy()
    summary = summary.sort_values(["roc_auc", "average_precision"], ascending=False)

    return round_numeric(summary)


def build_selected_final_table(
    best_classification: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    followup: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Creates a compact table with the currently selected final configuration.
    Values are manually extracted from the generated summaries.
    """
    rows = []

    if best_classification is not None:
        final_cls = best_classification[
            best_classification["feature_set"].astype(str).str.contains(
                "features_v3_lightcurve_plus_object_context",
                regex=False,
            )
        ]

        if len(final_cls) > 0:
            r = final_cls.iloc[0]
            rows.append({
                "component": "classification",
                "selected_configuration": f"{r['feature_set']} + {r['model']}",
                "main_metric": "macro_f1",
                "value": r["macro_f1"],
                "secondary_metric": "accuracy",
                "secondary_value": r["accuracy"],
            })

    if calibration is not None:
        sig = calibration[calibration["model"] == "hgb_sigmoid_calibrated"]
        if len(sig) > 0:
            r = sig.iloc[0]
            rows.append({
                "component": "calibration",
                "selected_configuration": "HistGradientBoosting + sigmoid calibration",
                "main_metric": "ECE",
                "value": r["ece"],
                "secondary_metric": "Brier score",
                "secondary_value": r["brier_score"],
            })

    if followup is not None:
        top10 = followup[followup["budget_fraction"] == 0.10]
        if len(top10) > 0:
            r = top10.iloc[0]
            rows.append({
                "component": "follow-up prioritization",
                "selected_configuration": "uncertainty + Mahalanobis novelty + rarity",
                "main_metric": "rare enrichment @ top 10%",
                "value": r["rare_enrichment"],
                "secondary_metric": "top rare true rate",
                "secondary_value": r["top_rare_true_rate"],
            })

    return round_numeric(pd.DataFrame(rows))


def main():
    classification = build_classification_summary()
    best_classification = build_best_classification_per_feature_set()
    calibration = build_calibration_summary()
    followup = build_followup_summary()
    robust_followup = build_robust_followup_summary()
    novelty = build_novelty_summary()
    selected_final = build_selected_final_table(best_classification, calibration, followup)

    outputs = {
        "classification_model_comparison.csv": classification,
        "best_classification_per_feature_set.csv": best_classification,
        "calibration_comparison.csv": calibration,
        "followup_prioritization_final.csv": followup,
        "followup_robust_novelty_ablation.csv": robust_followup,
        "novelty_detection_comparison.csv": novelty,
        "selected_final_configuration.csv": selected_final,
    }

    for filename, df in outputs.items():
        if df is not None:
            path = RESULTS_DIR / filename
            df.to_csv(path, index=False)
            print(f"[OK] Saved {path}")

    md = []
    md.append("# AstroTrust-AI — Final Preliminary Results Summary\n")
    md.append("This summary consolidates the current preliminary results generated by the AstroTrust-AI MVP pipeline.\n")

    md.append("## Selected final configuration\n")
    md.append(df_to_markdown(selected_final))

    md.append("## Best classification result per feature set\n")
    md.append(df_to_markdown(best_classification))

    md.append("## Full model comparison\n")
    md.append(df_to_markdown(classification))

    md.append("## Calibration comparison\n")
    md.append(df_to_markdown(calibration))

    md.append("## Follow-up prioritization: selected final ranking\n")
    md.append(df_to_markdown(followup))

    md.append("## Follow-up prioritization: robust novelty ablation\n")
    md.append(df_to_markdown(robust_followup))

    md.append("## Novelty detection comparison\n")
    md.append(df_to_markdown(novelty))

    md.append("## Current interpretation\n")
    md.append(
        "- The most scientifically defensible feature set is `features_v3_lightcurve_plus_object_context`, "
        "because it combines light-curve descriptors with object/host-galaxy contextual metadata while avoiding direct use of truth-table features as model inputs.\n"
    )
    md.append(
        "- The best classifier on the v3 feature set is HistGradientBoosting, but its raw probabilities are overconfident; sigmoid calibration provides a better trade-off between predictive performance and probability calibration.\n"
    )
    md.append(
        "- The final follow-up ranking enriches rare classes compared with random selection, especially in the top 5% and top 10% candidate budgets.\n"
    )
    md.append(
        "- Robust novelty scaling was tested as an ablation, but it did not consistently improve rare-class enrichment beyond the original Mahalanobis-based ranking.\n"
    )

    md_path = RESULTS_DIR / "final_results_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Saved {md_path}")

    print("\nSelected final configuration:")
    print(selected_final)

    print("\nBest classification per feature set:")
    print(best_classification)

    print("\nFollow-up final summary:")
    print(followup)


if __name__ == "__main__":
    main()