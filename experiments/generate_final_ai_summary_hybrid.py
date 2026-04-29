from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "final_ai_summary_hybrid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path):
    if path.exists():
        print(f"[OK] Reading {path}")
        return pd.read_csv(path)
    print(f"[WARN] Missing {path}")
    return None


def add_row(rows, model, family, dataset, source, accuracy=None, balanced_accuracy=None,
            macro_f1=None, weighted_f1=None, notes=""):
    rows.append({
        "model": model,
        "family": family,
        "dataset": dataset,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "source": source,
        "notes": notes,
    })


def main():
    rows = []

    # 25k v4
    p25 = RESULTS_DIR / "v4_temporal_shape_25k" / "v4_model_comparison_25k.csv"
    df25 = read_csv_if_exists(p25)
    if df25 is not None:
        best = df25.sort_values("macro_f1", ascending=False).iloc[0]
        add_row(
            rows,
            model=str(best["model"]),
            family="tabular",
            dataset="25k",
            source=str(p25),
            accuracy=best.get("accuracy"),
            balanced_accuracy=best.get("balanced_accuracy"),
            macro_f1=best.get("macro_f1"),
            weighted_f1=best.get("weighted_f1"),
            notes="Best v4 tabular model on 25k sample.",
        )

    # 100k v4
    p100 = RESULTS_DIR / "v4_temporal_shape_100000obj" / "v4_model_comparison_100000obj.csv"
    df100 = read_csv_if_exists(p100)
    if df100 is not None:
        best = df100.sort_values("macro_f1", ascending=False).iloc[0]
        add_row(
            rows,
            model=str(best["model"]),
            family="tabular",
            dataset="100k",
            source=str(p100),
            accuracy=best.get("accuracy"),
            balanced_accuracy=best.get("balanced_accuracy"),
            macro_f1=best.get("macro_f1"),
            weighted_f1=best.get("weighted_f1"),
            notes="Best v4 tabular model on approximately 100k sample.",
        )

    # 250k v4
    p250 = RESULTS_DIR / "v4_temporal_shape_250000obj" / "v4_model_comparison_250000obj.csv"
    df250 = read_csv_if_exists(p250)
    if df250 is not None:
        best = df250.sort_values("macro_f1", ascending=False).iloc[0]
        add_row(
            rows,
            model=str(best["model"]),
            family="tabular",
            dataset="250k",
            source=str(p250),
            accuracy=best.get("accuracy"),
            balanced_accuracy=best.get("balanced_accuracy"),
            macro_f1=best.get("macro_f1"),
            weighted_f1=best.get("weighted_f1"),
            notes="Best v4 tabular model on 250k requested sample.",
        )

    # Tabular ensemble 250k
    p_tab_ens = RESULTS_DIR / "v4_ensemble_250k" / "v4_ensemble_comparison_250k.csv"
    df_tab_ens = read_csv_if_exists(p_tab_ens)
    if df_tab_ens is not None:
        best = df_tab_ens.sort_values("macro_f1", ascending=False).iloc[0]
        add_row(
            rows,
            model=str(best["model"]),
            family="tabular ensemble",
            dataset="250k",
            source=str(p_tab_ens),
            accuracy=best.get("accuracy"),
            balanced_accuracy=best.get("balanced_accuracy"),
            macro_f1=best.get("macro_f1"),
            weighted_f1=best.get("weighted_f1"),
            notes="Best ensemble among LightGBM and XGBoost tabular models.",
        )

    # Temporal CNN v1
    p_cnn = RESULTS_DIR / "temporal_cnn_250k" / "temporal_cnn_test_metrics.csv"
    df_cnn = read_csv_if_exists(p_cnn)
    if df_cnn is not None:
        r = df_cnn.iloc[0]
        add_row(
            rows,
            model=str(r.get("model", "temporal_cnn_lc_only")),
            family="temporal neural",
            dataset="250k",
            source=str(p_cnn),
            accuracy=r.get("accuracy"),
            balanced_accuracy=r.get("balanced_accuracy"),
            macro_f1=r.get("macro_f1"),
            weighted_f1=r.get("weighted_f1"),
            notes="CNN using raw temporal light-curve tensor only.",
        )

    # Temporal CNN v2
    p_cnn_v2 = RESULTS_DIR / "temporal_cnn_v2_250k" / "temporal_cnn_v2_test_metrics.csv"
    df_cnn_v2 = read_csv_if_exists(p_cnn_v2)
    if df_cnn_v2 is not None:
        r = df_cnn_v2.iloc[0]
        add_row(
            rows,
            model=str(r.get("model", "temporal_cnn_v2")),
            family="temporal neural",
            dataset="250k",
            source=str(p_cnn_v2),
            accuracy=r.get("accuracy"),
            balanced_accuracy=r.get("balanced_accuracy"),
            macro_f1=r.get("macro_f1"),
            weighted_f1=r.get("weighted_f1"),
            notes="Residual/dilated temporal CNN.",
        )

    # Hybrid CNN + tabular
    p_hybrid = (
        RESULTS_DIR
        / "hybrid_temporal_tabular_cnn_250k"
        / "hybrid_temporal_tabular_cnn_test_metrics.csv"
    )
    df_hybrid = read_csv_if_exists(p_hybrid)
    if df_hybrid is not None:
        r = df_hybrid.iloc[0]
        add_row(
            rows,
            model=str(r.get("model", "hybrid_temporal_tabular_cnn")),
            family="hybrid neural",
            dataset="250k",
            source=str(p_hybrid),
            accuracy=r.get("accuracy"),
            balanced_accuracy=r.get("balanced_accuracy"),
            macro_f1=r.get("macro_f1"),
            weighted_f1=r.get("weighted_f1"),
            notes="Hybrid model combining temporal light-curve tensor and v4 tabular/context features.",
        )

    # Hybrid + tabular ensemble
    p_hybrid_ens = (
        RESULTS_DIR
        / "hybrid_tabular_ensemble_250k"
        / "hybrid_tabular_ensemble_comparison_250k.csv"
    )
    df_hybrid_ens = read_csv_if_exists(p_hybrid_ens)
    if df_hybrid_ens is not None:
        best = df_hybrid_ens.sort_values("macro_f1", ascending=False).iloc[0]
        add_row(
            rows,
            model=str(best["model"]),
            family="hybrid ensemble",
            dataset="250k",
            source=str(p_hybrid_ens),
            accuracy=best.get("accuracy"),
            balanced_accuracy=best.get("balanced_accuracy"),
            macro_f1=best.get("macro_f1"),
            weighted_f1=best.get("weighted_f1"),
            notes="Best performance-oriented ensemble combining hybrid CNN and tabular models.",
        )

    performance = pd.DataFrame(rows)
    performance = performance.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    perf_path = OUTPUT_DIR / "final_model_performance_summary.csv"
    performance.to_csv(perf_path, index=False)

    # Calibration and follow-up
    cal_path = (
        RESULTS_DIR
        / "hybrid_followup_policy_eval_250k"
        / "hybrid_calibration_summary.csv"
    )
    cal = read_csv_if_exists(cal_path)

    policy_path = (
        RESULTS_DIR
        / "hybrid_followup_policy_eval_250k"
        / "hybrid_test_policy_summary.csv"
    )
    policy = read_csv_if_exists(policy_path)

    selected_policy_path = (
        RESULTS_DIR
        / "hybrid_followup_policy_eval_250k"
        / "hybrid_selected_policies_from_validation.csv"
    )
    selected_policy = read_csv_if_exists(selected_policy_path)

    if cal is not None:
        cal.to_csv(OUTPUT_DIR / "final_hybrid_calibration_summary.csv", index=False)

    if policy is not None:
        policy_sorted = policy.sort_values("weighted_rare_enrichment", ascending=False)
        policy_sorted.to_csv(OUTPUT_DIR / "final_followup_policy_summary.csv", index=False)

    if selected_policy is not None:
        selected_policy.to_csv(OUTPUT_DIR / "final_selected_followup_policies.csv", index=False)

    # Markdown summary
    md = []
    md.append("# Final AstroTrust-AI Hybrid Summary\n")

    md.append("## Model performance\n")
    md.append(performance[[
        "model",
        "family",
        "dataset",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "notes",
    ]].to_markdown(index=False))
    md.append("\n")

    if cal is not None:
        md.append("## Hybrid calibration\n")
        md.append(cal.to_markdown(index=False))
        md.append("\n")

    if selected_policy is not None:
        md.append("## Selected follow-up policies\n")
        md.append(selected_policy.to_markdown(index=False))
        md.append("\n")

    if policy is not None:
        md.append("## Follow-up policy summary\n")
        display_cols = [
            "configuration",
            "w_uncertainty",
            "w_novelty",
            "w_rarity",
            "weighted_rare_enrichment",
            "weighted_uncertainty",
            "weighted_novelty",
            "top5_rare_enrichment",
            "top10_rare_enrichment",
            "top20_rare_enrichment",
        ]
        existing_cols = [c for c in display_cols if c in policy.columns]
        md.append(policy.sort_values("weighted_rare_enrichment", ascending=False)[existing_cols].to_markdown(index=False))
        md.append("\n")

    md.append("## Recommended final positioning\n")
    md.append(
        "- Main scientific contribution: hybrid temporal-tabular learning, combining raw multiband light-curve morphology with v4 temporal-shape and contextual features.\n"
        "- Best standalone scientific model: Hybrid temporal-tabular CNN.\n"
        "- Best performance-oriented model: Hybrid + tabular ensemble.\n"
        "- Final calibrated follow-up model: Hybrid CNN with temperature scaling.\n"
        "- Final follow-up policy: novelty + rarity, because it preserves rare-class enrichment while incorporating a novelty signal.\n"
    )

    md_path = OUTPUT_DIR / "final_ai_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\nFinal model performance summary:")
    print(performance[[
        "model",
        "family",
        "dataset",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]])

    if cal is not None:
        print("\nCalibration summary:")
        print(cal)

    if selected_policy is not None:
        print("\nSelected follow-up policies:")
        print(selected_policy)

    if policy is not None:
        print("\nFollow-up policy summary:")
        print(policy.sort_values("weighted_rare_enrichment", ascending=False))

    print("\nSaved:")
    print(f"- {perf_path}")
    print(f"- {OUTPUT_DIR / 'final_hybrid_calibration_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'final_followup_policy_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'final_selected_followup_policies.csv'}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()