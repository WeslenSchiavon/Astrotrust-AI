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

OUTPUT_DIR = ROOT_DIR / "results" / "oracle_hierarchical_classifier_25k"

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

    family_models = {}
    family_constants = {}

    print("\nTraining one subtype classifier per true family...")

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
            print(f"Single-class family. Using constant predictor: {family_constants[family]}")
            continue

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_fam)

        model = make_model()
        model.fit(X_fam, y_fam, sample_weight=sample_weight)

        family_models[family] = model

    print("\nPredicting with oracle true family...")

    y_pred = []

    for idx in X_test.index:
        family = y_test_family.loc[idx]

        if family in family_constants:
            pred = family_constants[family]
        else:
            model = family_models[family]
            x_row = X_test.loc[[idx]]
            pred = int(model.predict(x_row)[0])

        y_pred.append(pred)

    y_pred = np.asarray(y_pred)
    y_true = y_test_class.to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nOracle hierarchical classification results:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    labels = sorted(y_class.unique())
    target_names = [label_to_name[int(label)] for label in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    predictions = pd.DataFrame({
        "true_label": y_true,
        "true_class_name": [label_to_name[int(v)] for v in y_true],
        "true_family": [label_to_family[int(v)] for v in y_true],
        "predicted_label": y_pred,
        "predicted_class_name": [label_to_name[int(v)] for v in y_pred],
        "predicted_family": [label_to_family[int(v)] for v in y_pred],
        "correct": y_true == y_pred,
    })

    metrics = pd.DataFrame([{
        "model": "oracle_hierarchical_hist_gradient_boosting",
        "n_objects": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": y_class.nunique(),
        "n_families": y_family.nunique(),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }])

    predictions.to_csv(OUTPUT_DIR / "oracle_hierarchical_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "oracle_hierarchical_metrics.csv", index=False)

    with open(OUTPUT_DIR / "oracle_hierarchical_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'oracle_hierarchical_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'oracle_hierarchical_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'oracle_hierarchical_report.txt'}")


if __name__ == "__main__":
    main()