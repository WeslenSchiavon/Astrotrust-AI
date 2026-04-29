from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


ROOT_DIR = Path(__file__).resolve().parents[1]
CLASS_COUNTS_PATH = (
    ROOT_DIR / "data" / "processed" / "elasticc2_large" / "full_class_counts.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_label_names():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))


def prepare_dataset(features_path: Path):
    df = pd.read_parquet(features_path)

    y = df["label"].astype(int)

    X = df.drop(columns=["object_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y, df


def evaluate_model(name, model, X_train, X_test, y_train, y_test, output_dir):
    print("\n" + "=" * 80)
    print(f"Training {name}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    label_to_name = load_label_names()
    labels = sorted(y_test.unique())
    target_names = [label_to_name.get(int(label), str(label)) for label in labels]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    pd.DataFrame({
        "true_label": y_test.to_numpy(),
        "predicted_label": y_pred,
        "correct": y_test.to_numpy() == y_pred,
    }).to_csv(output_dir / f"{name}_predictions.csv", index=False)

    with open(output_dir / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": y_train.nunique(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, required=True)
    args = parser.parse_args()

    features_path = (
        ROOT_DIR
        / "data"
        / "processed"
        / "elasticc2_large"
        / f"features_v4_temporal_shape_{args.n_objects}obj.parquet"
    )

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    output_dir = (
        ROOT_DIR
        / "results"
        / f"v4_temporal_shape_{args.n_objects}obj"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    models = [
        (
            "hgb_v4",
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.04,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        )
    ]

    if HAS_LIGHTGBM:
        models.extend([
            (
                "lightgbm_v4_baseline",
                LGBMClassifier(
                    objective="multiclass",
                    n_estimators=800,
                    learning_rate=0.03,
                    num_leaves=63,
                    max_depth=-1,
                    min_child_samples=30,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=0.1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
            (
                "lightgbm_v4_regularized",
                LGBMClassifier(
                    objective="multiclass",
                    n_estimators=1200,
                    learning_rate=0.02,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=50,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    reg_alpha=0.1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ])

    results = []

    for name, model in models:
        result = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            output_dir=output_dir,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("macro_f1", ascending=False)

    output_path = output_dir / f"v4_model_comparison_{args.n_objects}obj.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("V4 model comparison summary:")
    print(results_df)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()