from pathlib import Path

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

FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "features_v4_temporal_shape_25000obj.parquet"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "v4_temporal_shape_25k"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_label_names():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))


def prepare_dataset():
    df = pd.read_parquet(FEATURES_PATH)

    y = df["label"].astype(int)

    X = df.drop(columns=["object_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 80)
    print(f"Training {name}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    if name.startswith("lightgbm"):
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
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
    }).to_csv(OUTPUT_DIR / f"{name}_predictions.csv", index=False)

    with open(OUTPUT_DIR / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_features": X_train.shape[1],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")

    X, y = prepare_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(X)}")
    print(f"Classes: {y.nunique()}")
    print(f"Features: {X.shape[1]}")

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
                "lightgbm_v4_regularized",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=y.nunique(),
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
            (
                "lightgbm_v4_baseline",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=y.nunique(),
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
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("macro_f1", ascending=False)

    output_path = OUTPUT_DIR / "v4_model_comparison_25k.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("V4 model comparison summary:")
    print(results_df)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()