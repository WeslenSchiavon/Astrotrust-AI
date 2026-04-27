from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "final_large_scale_summary"

N_OBJECTS_LIST = [5000, 10000, 25000]
MAIN_CALIBRATION = "sigmoid"


def load_classification_results():
    rows = []

    for n_objects in N_OBJECTS_LIST:
        path = (
            RESULTS_DIR
            / "large_model_comparison"
            / f"model_comparison_{n_objects}obj.csv"
        )

        if not path.exists():
            print(f"[WARN] Missing classification file: {path}")
            continue

        df = pd.read_csv(path)
        df["n_objects"] = n_objects

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    cols = [
        "n_objects",
        "feature_set",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "n_train",
        "n_test",
        "n_classes",
        "n_features",
    ]

    out = out[[c for c in cols if c in out.columns]]
    out = out.sort_values(["n_objects", "macro_f1"], ascending=[True, False])

    return out


def load_best_classification_per_size(classification_df):
    if classification_df.empty:
        return pd.DataFrame()

    idx = classification_df.groupby("n_objects")["macro_f1"].idxmax()
    return classification_df.loc[idx].sort_values("n_objects").reset_index(drop=True)


def load_calibration_results():
    rows = []

    for n_objects in N_OBJECTS_LIST:
        path = (
            RESULTS_DIR
            / "large_calibrated_hgb"
            / f"{n_objects}obj"
            / "calibration_comparison_summary.csv"
        )

        if not path.exists():
            print(f"[WARN] Missing calibration file: {path}")
            continue

        df = pd.read_csv(path)
        df["n_objects"] = n_objects

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    cols = [
        "n_objects",
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

    out = out[[c for c in cols if c in out.columns]]
    out = out.sort_values(["n_objects", "ece"], ascending=[True, True])

    return out


def load_sigmoid_calibration(calibration_df):
    if calibration_df.empty:
        return pd.DataFrame()

    out = calibration_df[
        calibration_df["model"].astype(str).str.contains("sigmoid", case=False)
    ].copy()

    return out.sort_values("n_objects").reset_index(drop=True)


def load_followup_results():
    rows = []

    for n_objects in N_OBJECTS_LIST:
        path = (
            RESULTS_DIR
            / "large_followup_prioritization_global_rarity"
            / f"{n_objects}obj"
            / MAIN_CALIBRATION
            / "followup_enrichment_vs_random.csv"
        )

        if not path.exists():
            print(f"[WARN] Missing follow-up file: {path}")
            continue

        df = pd.read_csv(path)
        df["n_objects"] = n_objects
        df["calibration"] = MAIN_CALIBRATION

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    preferred_cols = [
        "n_objects",
        "calibration",
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
        "top_mean_rarity",
        "random_mean_rarity",
        "top_mean_priority",
        "random_mean_priority",
    ]

    out = out[[c for c in preferred_cols if c in out.columns]]
    out = out.sort_values(["n_objects", "budget_fraction"]).reset_index(drop=True)

    return out


def load_followup_top10(followup_df):
    if followup_df.empty:
        return pd.DataFrame()

    out = followup_df[followup_df["budget_fraction"] == 0.10].copy()
    return out.sort_values("n_objects").reset_index(drop=True)


def build_selected_configuration(best_classification, sigmoid_calibration, followup_top10):
    rows = []

    for n_objects in N_OBJECTS_LIST:
        cls_row = best_classification[best_classification["n_objects"] == n_objects]
        cal_row = sigmoid_calibration[sigmoid_calibration["n_objects"] == n_objects]
        fol_row = followup_top10[followup_top10["n_objects"] == n_objects]

        row = {
            "n_objects": n_objects,
            "selected_feature_set": "features_v3_contextual",
            "selected_classifier": "hist_gradient_boosting",
            "selected_calibration": MAIN_CALIBRATION,
            "selected_novelty": "mahalanobis",
            "selected_rarity": "global_bottom_quartile",
            "priority_score": "0.40 uncertainty + 0.40 novelty + 0.20 global rarity",
        }

        if not cls_row.empty:
            row["best_model_macro_f1"] = float(cls_row.iloc[0]["macro_f1"])
            row["best_model_accuracy"] = float(cls_row.iloc[0]["accuracy"])
            row["best_model"] = cls_row.iloc[0]["model"]

        if not cal_row.empty:
            row["sigmoid_macro_f1"] = float(cal_row.iloc[0]["macro_f1"])
            row["sigmoid_accuracy"] = float(cal_row.iloc[0]["accuracy"])
            row["sigmoid_brier_score"] = float(cal_row.iloc[0]["brier_score"])
            row["sigmoid_ece"] = float(cal_row.iloc[0]["ece"])

        if not fol_row.empty:
            row["rare_enrichment_top10"] = float(fol_row.iloc[0]["rare_enrichment"])
            row["top10_rare_true_rate"] = float(fol_row.iloc[0]["top_rare_true_rate"])
            row["random_rare_true_rate_top10"] = float(
                fol_row.iloc[0]["random_rare_true_rate_mean"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def write_markdown_summary(
    classification_df,
    best_classification,
    calibration_df,
    sigmoid_calibration,
    followup_df,
    followup_top10,
    selected,
):
    path = OUTPUT_DIR / "large_scale_results_summary.md"

    lines = []

    lines.append("# AstroTrust-AI Large-scale Learning Curve Summary\n")
    lines.append("This file summarizes the current large-scale experiments using ELAsTiCC2_TRAIN_02.\n")
    lines.append("The current large-scale pipeline uses:\n")
    lines.append("- features_v3_contextual\n")
    lines.append("- HistGradientBoosting\n")
    lines.append("- sigmoid calibration for the main follow-up pipeline\n")
    lines.append("- Mahalanobis novelty score\n")
    lines.append("- global rarity score based on full ELAsTiCC2_TRAIN_02 class frequencies\n")
    lines.append("- priority score = 0.40 uncertainty + 0.40 novelty + 0.20 rarity\n\n")

    lines.append("## Best classification model per dataset size\n\n")
    if not best_classification.empty:
        lines.append(best_classification.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("## Sigmoid calibration learning curve\n\n")
    if not sigmoid_calibration.empty:
        lines.append(sigmoid_calibration.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("## Follow-up enrichment at Top 10%\n\n")
    if not followup_top10.empty:
        lines.append(followup_top10.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("## Selected large-scale configuration\n\n")
    if not selected.empty:
        lines.append(selected.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("## Interpretation\n\n")
    lines.append(
        "The learning curve shows that classification performance improves as the training sample increases. "
        "HistGradientBoosting remains the strongest model among the evaluated tabular baselines. "
        "Sigmoid calibration provides a stable compromise between classification performance, Brier score, and calibration error. "
        "The global-rarity follow-up priority score consistently enriches globally rare classes in the top-ranked candidates.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classification_df = load_classification_results()
    best_classification = load_best_classification_per_size(classification_df)

    calibration_df = load_calibration_results()
    sigmoid_calibration = load_sigmoid_calibration(calibration_df)

    followup_df = load_followup_results()
    followup_top10 = load_followup_top10(followup_df)

    selected = build_selected_configuration(
        best_classification=best_classification,
        sigmoid_calibration=sigmoid_calibration,
        followup_top10=followup_top10,
    )

    files = {
        "classification": OUTPUT_DIR / "learning_curve_classification.csv",
        "best_classification": OUTPUT_DIR / "learning_curve_best_classification.csv",
        "calibration": OUTPUT_DIR / "learning_curve_calibration.csv",
        "sigmoid_calibration": OUTPUT_DIR / "learning_curve_sigmoid_calibration.csv",
        "followup": OUTPUT_DIR / "learning_curve_followup.csv",
        "followup_top10": OUTPUT_DIR / "learning_curve_followup_top10.csv",
        "selected": OUTPUT_DIR / "selected_large_scale_configuration.csv",
    }

    classification_df.to_csv(files["classification"], index=False)
    best_classification.to_csv(files["best_classification"], index=False)
    calibration_df.to_csv(files["calibration"], index=False)
    sigmoid_calibration.to_csv(files["sigmoid_calibration"], index=False)
    followup_df.to_csv(files["followup"], index=False)
    followup_top10.to_csv(files["followup_top10"], index=False)
    selected.to_csv(files["selected"], index=False)

    summary_path = write_markdown_summary(
        classification_df=classification_df,
        best_classification=best_classification,
        calibration_df=calibration_df,
        sigmoid_calibration=sigmoid_calibration,
        followup_df=followup_df,
        followup_top10=followup_top10,
        selected=selected,
    )

    print("[OK] Saved summary files:")
    for name, path in files.items():
        print(f"- {name}: {path}")

    print(f"- markdown_summary: {summary_path}")

    print("\nBest classification per dataset size:")
    print(best_classification)

    print("\nSigmoid calibration learning curve:")
    print(sigmoid_calibration)

    print("\nFollow-up enrichment at Top 10%:")
    print(followup_top10)

    print("\nSelected large-scale configuration:")
    print(selected)


if __name__ == "__main__":
    main()