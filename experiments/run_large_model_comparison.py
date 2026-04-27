from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT_DIR / "results" / "large_model_comparison"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def prepare_dataset(features_path: Path):
    df = pd.read_parquet(features_path)

    y = df["label"]
    X = df.drop(columns=["object_id", "label"])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y, df


def evaluate_model(
    feature_set_name,
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    use_sample_weight=False,
):
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

    model_dir = RESULTS_ROOT / feature_set_name / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    predictions = pd.DataFrame({
        "true_label": y_test.to_numpy(),
        "predicted_label": y_pred,
    })

    predictions.to_csv(model_dir / "predictions.csv", index=False)

    return {
        "feature_set": feature_set_name,
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": y_train.nunique(),
        "n_features": X_train.shape[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, required=True)
    args = parser.parse_args()

    feature_set_name = f"features_v3_contextual_{args.n_objects}obj"

    features_path = (
        ROOT_DIR
        / "data"
        / "processed"
        / "elasticc2_large"
        / f"features_v3_contextual_{args.n_objects}obj.parquet"
    )

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {features_path}")

    X, y, df = prepare_dataset(features_path)

    print("\nDataset summary:")
    print(f"Objects: {len(df)}")
    print(f"Classes: {y.nunique()}")
    print(f"Features: {X.shape[1]}")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

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
            feature_set_name=feature_set_name,
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
    results_df = results_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)

    output_path = RESULTS_ROOT / f"model_comparison_{args.n_objects}obj.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Large model comparison summary:")
    print(results_df)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()