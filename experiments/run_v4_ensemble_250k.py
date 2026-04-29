from pathlib import Path
import time

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
    / "features_v4_temporal_shape_250000obj.parquet"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "v4_ensemble_250k"

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


def make_lightgbm_baseline(n_classes):
    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
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
    )


def make_lightgbm_regularized(n_classes):
    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
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
    )


def make_xgboost_fast(n_classes):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=700,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=1.0,
        max_bin=256,
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgboost_deeper(n_classes):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=900,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=2,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=1.0,
        max_bin=256,
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def align_probabilities(probabilities, model_classes, all_classes):
    """
    Ensures probability columns follow the same class order for all models.
    """
    aligned = np.zeros((probabilities.shape[0], len(all_classes)), dtype=np.float32)

    class_to_col = {
        int(cls): idx
        for idx, cls in enumerate(all_classes)
    }

    for local_idx, cls in enumerate(model_classes):
        global_idx = class_to_col[int(cls)]
        aligned[:, global_idx] = probabilities[:, local_idx]

    return aligned


def evaluate_predictions(name, y_true, y_pred, probabilities=None):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    confidence = None
    if probabilities is not None:
        confidence = float(np.max(probabilities, axis=1).mean())

    print("\n" + "=" * 80)
    print(f"Results: {name}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    if confidence is not None:
        print(f"Mean confidence:   {confidence:.4f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_confidence": confidence,
    }


def train_predict_proba(name, model, X_train, X_test, y_train, all_classes):
    print("\n" + "=" * 80)
    print(f"Training {name}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    start = time.time()

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )

    elapsed = time.time() - start

    print(f"Training time: {elapsed / 60:.2f} min")

    probabilities = model.predict_proba(X_test)
    probabilities = align_probabilities(
        probabilities=probabilities,
        model_classes=model.classes_,
        all_classes=all_classes,
    )

    return probabilities, elapsed / 60


def save_classification_report(name, y_test, y_pred):
    label_to_name = load_label_names()
    labels = sorted(np.unique(y_test))
    target_names = [label_to_name.get(int(label), str(label)) for label in labels]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    with open(OUTPUT_DIR / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nClassification report:")
    print(report)


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

    all_classes = np.asarray(sorted(y.unique()), dtype=int)
    n_classes = len(all_classes)

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Classes: {n_classes}")

    models = [
        ("lightgbm_baseline", make_lightgbm_baseline(n_classes)),
        ("lightgbm_regularized", make_lightgbm_regularized(n_classes)),
        ("xgboost_gpu_fast", make_xgboost_fast(n_classes)),
        ("xgboost_gpu_deeper", make_xgboost_deeper(n_classes)),
    ]

    probabilities_by_model = {}
    individual_results = []

    for name, model in models:
        probs, train_time = train_predict_proba(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            all_classes=all_classes,
        )

        probabilities_by_model[name] = probs

        y_pred = all_classes[np.argmax(probs, axis=1)]

        metrics = evaluate_predictions(
            name=name,
            y_true=y_test,
            y_pred=y_pred,
            probabilities=probs,
        )
        metrics["training_time_minutes"] = train_time
        individual_results.append(metrics)

        pd.DataFrame({
            "true_label": y_test.to_numpy(),
            "predicted_label": y_pred,
            "correct": y_test.to_numpy() == y_pred,
            "confidence": np.max(probs, axis=1),
        }).to_csv(OUTPUT_DIR / f"{name}_predictions.csv", index=False)

    # Ensemble configurations.
    ensemble_configs = {
        "ensemble_lgbm_two": [
            "lightgbm_baseline",
            "lightgbm_regularized",
        ],
        "ensemble_lgbm_xgb_best2": [
            "lightgbm_baseline",
            "xgboost_gpu_deeper",
        ],
        "ensemble_all_equal": [
            "lightgbm_baseline",
            "lightgbm_regularized",
            "xgboost_gpu_fast",
            "xgboost_gpu_deeper",
        ],
        "ensemble_weighted_lgbm": [
            "lightgbm_baseline",
            "lightgbm_baseline",
            "lightgbm_regularized",
            "xgboost_gpu_deeper",
        ],
        "ensemble_weighted_mixed": [
            "lightgbm_baseline",
            "lightgbm_baseline",
            "xgboost_gpu_deeper",
            "xgboost_gpu_fast",
        ],
    }

    ensemble_results = []

    for ensemble_name, members in ensemble_configs.items():
        print("\n" + "=" * 80)
        print(f"Evaluating {ensemble_name}")
        print(f"Members: {members}")

        stacked = np.stack(
            [probabilities_by_model[m] for m in members],
            axis=0,
        )

        avg_probs = np.mean(stacked, axis=0)
        y_pred = all_classes[np.argmax(avg_probs, axis=1)]

        metrics = evaluate_predictions(
            name=ensemble_name,
            y_true=y_test,
            y_pred=y_pred,
            probabilities=avg_probs,
        )

        metrics["members"] = ",".join(members)
        ensemble_results.append(metrics)

        pd.DataFrame({
            "true_label": y_test.to_numpy(),
            "predicted_label": y_pred,
            "correct": y_test.to_numpy() == y_pred,
            "confidence": np.max(avg_probs, axis=1),
        }).to_csv(OUTPUT_DIR / f"{ensemble_name}_predictions.csv", index=False)

        save_classification_report(
            name=ensemble_name,
            y_test=y_test,
            y_pred=y_pred,
        )

    individual_df = pd.DataFrame(individual_results)
    ensemble_df = pd.DataFrame(ensemble_results)

    all_results = pd.concat([individual_df, ensemble_df], ignore_index=True)
    all_results = all_results.sort_values("macro_f1", ascending=False)

    individual_df.to_csv(OUTPUT_DIR / "individual_model_results.csv", index=False)
    ensemble_df.to_csv(OUTPUT_DIR / "ensemble_results.csv", index=False)
    all_results.to_csv(OUTPUT_DIR / "v4_ensemble_comparison_250k.csv", index=False)

    print("\n" + "=" * 80)
    print("Final ensemble comparison:")
    print(all_results)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'individual_model_results.csv'}")
    print(f"- {OUTPUT_DIR / 'ensemble_results.csv'}")
    print(f"- {OUTPUT_DIR / 'v4_ensemble_comparison_250k.csv'}")


if __name__ == "__main__":
    main()