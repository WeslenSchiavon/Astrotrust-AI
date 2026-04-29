from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "features_v4_temporal_shape_250000obj.parquet"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "v4_xgboost_gpu_250k"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_label_names():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(
        zip(
            counts["label"].astype(int),
            counts["class_name"].astype(str),
        )
    )


def prepare_dataset():
    df = pd.read_parquet(FEATURES_PATH)

    y = df["label"].astype(int)

    X = df.drop(columns=["object_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(np.float32)

    return X, y, df


def make_xgb_model(name, use_gpu=True):
    common = {
        "objective": "multi:softprob",
        "num_class": 32,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    if name == "xgboost_gpu_fast":
        params = {
            "n_estimators": 700,
            "learning_rate": 0.04,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "max_bin": 256,
        }

    elif name == "xgboost_gpu_regularized":
        params = {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.10,
            "reg_lambda": 2.0,
            "max_bin": 256,
        }

    elif name == "xgboost_gpu_deeper":
        params = {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "max_depth": 7,
            "min_child_weight": 2,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "max_bin": 256,
        }

    else:
        raise ValueError(f"Unknown model name: {name}")

    params.update(common)

    if use_gpu:
        # Current XGBoost GPU style.
        params["tree_method"] = "hist"
        params["device"] = "cuda"
    else:
        params["tree_method"] = "hist"
        params["device"] = "cpu"

    return XGBClassifier(**params)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 80)
    print(f"Training {name}")
    print(model)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    start = time.time()

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        verbose=False,
    )

    elapsed = time.time() - start

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nTraining time:      {elapsed / 60:.2f} min")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced accuracy:  {balanced_acc:.4f}")
    print(f"Macro-F1:           {macro_f1:.4f}")
    print(f"Weighted-F1:        {weighted_f1:.4f}")

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
        "training_time_minutes": elapsed / 60,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": y_train.nunique(),
    }


def train_with_gpu_fallback(model_name, X_train, X_test, y_train, y_test):
    try:
        model = make_xgb_model(model_name, use_gpu=True)
        return evaluate_model(model_name, model, X_train, X_test, y_train, y_test)

    except Exception as gpu_error:
        print("\n" + "!" * 80)
        print(f"[WARN] GPU training failed for {model_name}.")
        print("Error:")
        print(gpu_error)
        print("\nTrying CPU fallback...")
        print("!" * 80)

        cpu_name = model_name.replace("gpu", "cpu")
        model = make_xgb_model(model_name, use_gpu=False)
        return evaluate_model(cpu_name, model, X_train, X_test, y_train, y_test)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")

    X, y, df = prepare_dataset()

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

    model_names = [
        "xgboost_gpu_fast",
        "xgboost_gpu_regularized",
        "xgboost_gpu_deeper",
    ]

    results = []

    for model_name in model_names:
        result = train_with_gpu_fallback(
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("macro_f1", ascending=False)

    output_path = OUTPUT_DIR / "xgboost_gpu_comparison_250k.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("XGBoost comparison summary:")
    print(results_df)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()