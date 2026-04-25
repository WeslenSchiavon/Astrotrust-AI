from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
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

FEATURES_PATH = ROOT_DIR / "data" / "processed" / "elasticc2" / "features_1000obj.parquet"
RESULTS_DIR = ROOT_DIR / "results" / "model_comparison"

MIN_OBJECTS_PER_CLASS = 20
TEST_SIZE = 0.25
RANDOM_STATE = 42


def prepare_dataset():
    df = pd.read_parquet(FEATURES_PATH)

    class_counts = df["label"].value_counts().sort_index()
    valid_classes = class_counts[class_counts >= MIN_OBJECTS_PER_CLASS].index.tolist()

    df = df[df["label"].isin(valid_classes)].copy()

    y = df["label"]
    X = df.drop(columns=["object_id", "label"])

    # Garante que todas as features sejam numéricas.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y, df, valid_classes


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, use_sample_weight=False):
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")

    if use_sample_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification report:")
    print(report)

    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "true_label": y_test.to_numpy(),
        "predicted_label": y_pred,
    })

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    predictions.to_csv(model_dir / "predictions.csv", index=False)
    cm_df.to_csv(model_dir / "confusion_matrix.csv")

    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": y_train.nunique(),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")

    X, y, df, valid_classes = prepare_dataset()

    print("\nClasses used:")
    print(y.value_counts().sort_index())

    print(f"\nObjects: {len(df)}")
    print(f"Classes: {len(valid_classes)}")
    print(f"Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            False,
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=800,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            False,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.04,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    ]

    results = []

    for model_name, model, use_sample_weight in models:
        result = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            use_sample_weight=use_sample_weight,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("macro_f1", ascending=False)

    output_path = RESULTS_DIR / "model_comparison_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Model comparison summary:")
    print(results_df)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()