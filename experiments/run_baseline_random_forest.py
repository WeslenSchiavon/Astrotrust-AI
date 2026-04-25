from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path("C:/Users/wesle/Desktop/Astrolara/MeuProjeto/astrotrust-ai")

FEATURES_PATH = CURRENT_DIR / Path("data/processed/elasticc2/features_1000obj.parquet")
RESULTS_DIR = CURRENT_DIR / Path("results/baseline_random_forest")
MIN_OBJECTS_PER_CLASS = 5
RANDOM_STATE = 42


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    print(f"Original objects: {len(df)}")
    print(f"Original classes: {df['label'].nunique()}")

    class_counts = df["label"].value_counts().sort_index()
    valid_classes = class_counts[class_counts >= MIN_OBJECTS_PER_CLASS].index

    df = df[df["label"].isin(valid_classes)].copy()

    print("\nAfter filtering rare classes:")
    print(f"Objects: {len(df)}")
    print(f"Classes: {df['label'].nunique()}")
    print(df["label"].value_counts().sort_index())

    y = df["label"]

    X = df.drop(columns=["object_id", "label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nResults:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    print("\nClassification report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=sorted(valid_classes))

    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_classes": df["label"].nunique(),
                "min_objects_per_class": MIN_OBJECTS_PER_CLASS,
            }
        ]
    )

    predictions = pd.DataFrame(
        {
            "true_label": y_test.values,
            "predicted_label": y_pred,
        }
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in sorted(valid_classes)],
        columns=[f"pred_{c}" for c in sorted(valid_classes)],
    )

    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    print("\nSaved results:")
    print(f"- {RESULTS_DIR / 'metrics.csv'}")
    print(f"- {RESULTS_DIR / 'predictions.csv'}")
    print(f"- {RESULTS_DIR / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()