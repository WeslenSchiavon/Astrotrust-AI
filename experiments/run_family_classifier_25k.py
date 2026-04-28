from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
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

OUTPUT_DIR = ROOT_DIR / "results" / "family_classifier_25k"

RANDOM_STATE = 42
TEST_SIZE = 0.25


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


def load_label_to_family():
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

    label_to_name, label_to_family = load_label_to_family()

    y_class = df["label"].astype(int)
    y_family = y_class.map(label_to_family)

    X = df.drop(columns=["object_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y_family, y_class, label_to_name


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.04,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")

    X, y_family, y_class, label_to_name = prepare_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(X)}")
    print(f"Original classes: {y_class.nunique()}")
    print(f"Families: {y_family.nunique()}")
    print(f"Features: {X.shape[1]}")

    print("\nFamily distribution:")
    print(y_family.value_counts().sort_index())

    X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
        X,
        y_family,
        y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_family,
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print("\nTraining family-level HistGradientBoosting...")
    model = make_model()
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nFamily-level classification results:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification report:")
    print(report)

    predictions = pd.DataFrame({
        "true_family": y_test.to_numpy(),
        "predicted_family": y_pred,
        "correct_family": y_test.to_numpy() == y_pred,
        "true_class_label": y_class_test.to_numpy(),
        "true_class_name": [
            label_to_name.get(int(v), str(v))
            for v in y_class_test.to_numpy()
        ],
    })

    metrics = pd.DataFrame([{
        "model": "hist_gradient_boosting",
        "task": "family_classification",
        "n_objects": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_families": y_family.nunique(),
        "n_original_classes": y_class.nunique(),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }])

    families = sorted(y_family.unique())
    cm = confusion_matrix(y_test, y_pred, labels=families)
    cm_df = pd.DataFrame(cm, index=families, columns=families)

    predictions.to_csv(OUTPUT_DIR / "family_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "family_metrics.csv", index=False)
    cm_df.to_csv(OUTPUT_DIR / "family_confusion_matrix.csv")

    with open(OUTPUT_DIR / "family_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'family_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'family_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'family_confusion_matrix.csv'}")
    print(f"- {OUTPUT_DIR / 'family_classification_report.txt'}")


if __name__ == "__main__":
    main()