from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


ROOT_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT_DIR
    / "results"
    / "large_model_comparison"
    / "features_v3_contextual_25000obj"
    / "hist_gradient_boosting"
    / "predictions.csv"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "large_error_analysis_25k"


def load_class_names():
    counts = pd.read_csv(CLASS_COUNTS_PATH)

    label_to_name = dict(
        zip(
            counts["label"].astype(int),
            counts["class_name"].astype(str),
        )
    )

    label_to_global_count = dict(
        zip(
            counts["label"].astype(int),
            counts["n_objects"].astype(int),
        )
    )

    return label_to_name, label_to_global_count


def class_family(class_name: str) -> str:
    name = class_name.lower()

    if name.startswith("snia"):
        return "SNIa"
    if name.startswith("snii"):
        return "SNII"
    if name.startswith("snib"):
        return "SNIb"
    if name.startswith("snic"):
        return "SNIc"
    if name.startswith("slsn"):
        return "SLSN"
    if name.startswith("kn"):
        return "KN"
    if "ulens" in name:
        return "uLens"
    if name in ["rrl", "cepheid", "d-sct", "eb"]:
        return "Variable star"
    if "dwarf" in name or "flare" in name:
        return "Stellar transient"
    if name in ["clagn", "tde"]:
        return "AGN/TDE"
    return "Other"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH}")

    predictions = pd.read_csv(PREDICTIONS_PATH)

    if "true_label" not in predictions.columns or "predicted_label" not in predictions.columns:
        raise ValueError("predictions.csv must contain true_label and predicted_label columns.")

    y_true = predictions["true_label"].astype(int).to_numpy()
    y_pred = predictions["predicted_label"].astype(int).to_numpy()

    labels = sorted(set(y_true) | set(y_pred))

    label_to_name, label_to_global_count = load_class_names()

    target_names = [label_to_name.get(label, str(label)) for label in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for label, name in zip(labels, target_names):
        metrics = report[name]

        rows.append({
            "label": label,
            "class_name": name,
            "family": class_family(name),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "test_support": int(metrics["support"]),
            "global_count": label_to_global_count.get(label, np.nan),
        })

    per_class = pd.DataFrame(rows)
    per_class = per_class.sort_values("f1_score", ascending=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    cm_norm_df = pd.DataFrame(cm_norm, index=target_names, columns=target_names)

    confusion_rows = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue

            count = cm[i, j]

            if count <= 0:
                continue

            true_name = label_to_name.get(true_label, str(true_label))
            pred_name = label_to_name.get(pred_label, str(pred_label))

            confusion_rows.append({
                "true_label": true_label,
                "true_class": true_name,
                "true_family": class_family(true_name),
                "predicted_label": pred_label,
                "predicted_class": pred_name,
                "predicted_family": class_family(pred_name),
                "count": int(count),
                "fraction_of_true_class": count / cm[i].sum() if cm[i].sum() > 0 else 0.0,
            })

    top_confusions = pd.DataFrame(confusion_rows)
    top_confusions = top_confusions.sort_values(
        ["count", "fraction_of_true_class"],
        ascending=False,
    )

    family_true = pd.Series([class_family(label_to_name.get(v, str(v))) for v in y_true])
    family_pred = pd.Series([class_family(label_to_name.get(v, str(v))) for v in y_pred])

    family_error = pd.DataFrame({
        "true_family": family_true,
        "predicted_family": family_pred,
        "correct_family": family_true == family_pred,
        "correct_class": y_true == y_pred,
    })

    family_summary = (
        family_error
        .groupby("true_family")
        .agg(
            n=("true_family", "size"),
            family_accuracy=("correct_family", "mean"),
            class_accuracy=("correct_class", "mean"),
        )
        .reset_index()
        .sort_values("class_accuracy", ascending=True)
    )

    per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix_counts.csv")
    cm_norm_df.to_csv(OUTPUT_DIR / "confusion_matrix_normalized.csv")
    top_confusions.to_csv(OUTPUT_DIR / "top_confusions.csv", index=False)
    family_summary.to_csv(OUTPUT_DIR / "family_error_summary.csv", index=False)

    summary_path = OUTPUT_DIR / "error_analysis_summary.md"

    lines = []
    lines.append("# Error Analysis - 25k HistGradientBoosting\n")

    lines.append("## Worst classes by F1-score\n")
    lines.append(per_class.head(12).to_markdown(index=False))
    lines.append("\n")

    lines.append("## Best classes by F1-score\n")
    lines.append(per_class.sort_values("f1_score", ascending=False).head(12).to_markdown(index=False))
    lines.append("\n")

    lines.append("## Top class confusions\n")
    lines.append(top_confusions.head(20).to_markdown(index=False))
    lines.append("\n")

    lines.append("## Family-level summary\n")
    lines.append(family_summary.to_markdown(index=False))
    lines.append("\n")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Error analysis saved:")
    print(f"- {OUTPUT_DIR / 'per_class_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'top_confusions.csv'}")
    print(f"- {OUTPUT_DIR / 'family_error_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix_counts.csv'}")
    print(f"- {OUTPUT_DIR / 'confusion_matrix_normalized.csv'}")
    print(f"- {summary_path}")

    print("\nWorst classes by F1-score:")
    print(per_class.head(12))

    print("\nTop class confusions:")
    print(top_confusions.head(20))

    print("\nFamily-level summary:")
    print(family_summary)


if __name__ == "__main__":
    main()