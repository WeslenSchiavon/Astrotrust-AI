from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "features_v3_contextual_25000obj.parquet"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "soft_hierarchical_classifier_25k"

RANDOM_STATE = 42
TEST_SIZE = 0.25

# alpha controls how strongly the family probability influences the final class score.
# alpha = 1.0 means P(class) = P(family) * P(class | family)
ALPHAS = [0.25, 0.50, 1.00, 1.50, 2.00]


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


def load_label_maps():
    counts = pd.read_csv(CLASS_COUNTS_PATH)

    label_to_name = dict(
        zip(
            counts["label"].astype(int),
            counts["class_name"].astype(str),
        )
    )

    label_to_family = {
        label: class_family(name)
        for label, name in label_to_name.items()
    }

    return label_to_name, label_to_family


def prepare_dataset():
    df = pd.read_parquet(FEATURES_PATH)

    label_to_name, label_to_family = load_label_maps()

    y_class = df["label"].astype(int)
    y_family = y_class.map(label_to_family)

    X = df.drop(columns=["object_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y_class, y_family, label_to_name, label_to_family


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.04,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def compute_soft_scores(
    X_test,
    family_model,
    subtype_models,
    family_constants,
    all_labels,
    alpha,
):
    n_samples = len(X_test)
    n_classes = len(all_labels)

    label_to_col = {label: idx for idx, label in enumerate(all_labels)}

    class_scores = np.zeros((n_samples, n_classes), dtype=float)

    family_probs = family_model.predict_proba(X_test)
    family_classes = list(family_model.classes_)
    family_to_col = {family: idx for idx, family in enumerate(family_classes)}

    for family in family_classes:
        family_prob = family_probs[:, family_to_col[family]]

        # Controls the strength of the family-stage probability.
        family_weight = np.power(family_prob, alpha)

        if family in family_constants:
            label = family_constants[family]
            class_scores[:, label_to_col[label]] = family_weight
            continue

        subtype_model = subtype_models[family]
        subtype_probs = subtype_model.predict_proba(X_test)

        for local_col, label in enumerate(subtype_model.classes_):
            label = int(label)
            global_col = label_to_col[label]
            class_scores[:, global_col] = family_weight * subtype_probs[:, local_col]

    return class_scores


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")

    X, y_class, y_family, label_to_name, label_to_family = prepare_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(X)}")
    print(f"Original classes: {y_class.nunique()}")
    print(f"Families: {y_family.nunique()}")
    print(f"Features: {X.shape[1]}")

    X_train, X_test, y_train_class, y_test_class, y_train_family, y_test_family = train_test_split(
        X,
        y_class,
        y_family,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    print("\nTraining family-level classifier...")

    family_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_family)

    family_model = make_model()
    family_model.fit(X_train, y_train_family, sample_weight=family_sample_weight)

    family_pred = family_model.predict(X_test)
    family_accuracy = accuracy_score(y_test_family, family_pred)
    family_macro_f1 = f1_score(y_test_family, family_pred, average="macro")

    print("\nFamily-stage performance:")
    print(f"Family accuracy: {family_accuracy:.4f}")
    print(f"Family Macro-F1: {family_macro_f1:.4f}")

    print("\nTraining one subtype classifier per family...")

    subtype_models = {}
    family_constants = {}

    for family in sorted(y_train_family.unique()):
        mask = y_train_family == family

        X_fam = X_train.loc[mask]
        y_fam = y_train_class.loc[mask]

        classes_in_family = sorted(y_fam.unique())

        print(f"\nFamily: {family}")
        print(f"Train objects: {len(X_fam)}")
        print(f"Classes: {[label_to_name[int(c)] for c in classes_in_family]}")

        if len(classes_in_family) == 1:
            family_constants[family] = int(classes_in_family[0])
            print(f"Single-class family. Using constant predictor.")
            continue

        subtype_sample_weight = compute_sample_weight(class_weight="balanced", y=y_fam)

        subtype_model = make_model()
        subtype_model.fit(X_fam, y_fam, sample_weight=subtype_sample_weight)

        subtype_models[family] = subtype_model

    all_labels = sorted(y_class.unique())
    y_true = y_test_class.to_numpy()

    summary_rows = []
    predictions_by_alpha = {}

    print("\nEvaluating soft hierarchical combinations...")

    for alpha in ALPHAS:
        scores = compute_soft_scores(
            X_test=X_test,
            family_model=family_model,
            subtype_models=subtype_models,
            family_constants=family_constants,
            all_labels=all_labels,
            alpha=alpha,
        )

        pred_cols = np.argmax(scores, axis=1)
        y_pred = np.asarray([all_labels[col] for col in pred_cols], dtype=int)

        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")

        print("\n" + "=" * 80)
        print(f"Soft hierarchical results | alpha={alpha}")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_acc:.4f}")
        print(f"Macro-F1:          {macro_f1:.4f}")
        print(f"Weighted-F1:       {weighted_f1:.4f}")

        summary_rows.append({
            "alpha": alpha,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "family_accuracy": family_accuracy,
            "family_macro_f1": family_macro_f1,
        })

        predictions_by_alpha[alpha] = y_pred

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("macro_f1", ascending=False)

    best_alpha = float(summary.iloc[0]["alpha"])
    best_pred = predictions_by_alpha[best_alpha]

    print("\n" + "=" * 80)
    print("Soft hierarchical summary:")
    print(summary)

    print(f"\nBest alpha: {best_alpha}")

    labels = sorted(y_class.unique())
    target_names = [label_to_name[int(label)] for label in labels]

    report = classification_report(
        y_true,
        best_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\nClassification report for best alpha:")
    print(report)

    predictions = pd.DataFrame({
        "true_label": y_true,
        "true_class_name": [label_to_name[int(v)] for v in y_true],
        "true_family": [label_to_family[int(v)] for v in y_true],
        "predicted_label": best_pred,
        "predicted_class_name": [label_to_name[int(v)] for v in best_pred],
        "predicted_family": [label_to_family[int(v)] for v in best_pred],
        "correct": y_true == best_pred,
        "best_alpha": best_alpha,
    })

    summary.to_csv(OUTPUT_DIR / "soft_hierarchical_alpha_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "soft_hierarchical_predictions.csv", index=False)

    with open(OUTPUT_DIR / "soft_hierarchical_best_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'soft_hierarchical_alpha_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'soft_hierarchical_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'soft_hierarchical_best_report.txt'}")


if __name__ == "__main__":
    main()