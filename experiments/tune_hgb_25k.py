from pathlib import Path
import itertools

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

OUTPUT_DIR = ROOT_DIR / "results" / "hgb_tuning_25k"

RANDOM_STATE = 42


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
    X = X.astype(float)

    return X, y


def make_model(params):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        random_state=RANDOM_STATE,
    )


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")
    X, y = prepare_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(X)}")
    print(f"Classes: {y.nunique()}")
    print(f"Features: {X.shape[1]}")

    # Keep the same final test fraction used in previous experiments.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Validation is used only for tuning.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # Moderate grid: enough to test meaningful changes, not too huge.
    param_grid = {
        "max_iter": [300, 500, 800],
        "learning_rate": [0.03, 0.04, 0.06],
        "max_leaf_nodes": [31, 63],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [0.01, 0.1, 1.0],
    }

    all_configs = [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]

    print(f"\nTotal candidate configurations: {len(all_configs)}")

    results = []

    train_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    for i, params in enumerate(all_configs, start=1):
        print("\n" + "=" * 80)
        print(f"Candidate {i}/{len(all_configs)}")
        print(params)

        model = make_model(params)
        model.fit(X_train, y_train, sample_weight=train_sample_weight)

        y_val_pred = model.predict(X_val)
        metrics = evaluate(y_val, y_val_pred)

        row = {
            "candidate": i,
            **params,
            **{f"val_{k}": v for k, v in metrics.items()},
        }

        results.append(row)

        print(
            f"Validation | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"BalAcc={metrics['balanced_accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | "
            f"Weighted-F1={metrics['weighted_f1']:.4f}"
        )

        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "validation_tuning_results_partial.csv",
            index=False,
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["val_macro_f1", "val_balanced_accuracy", "val_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    best_params = {
        "max_iter": int(results_df.iloc[0]["max_iter"]),
        "learning_rate": float(results_df.iloc[0]["learning_rate"]),
        "max_leaf_nodes": int(results_df.iloc[0]["max_leaf_nodes"]),
        "min_samples_leaf": int(results_df.iloc[0]["min_samples_leaf"]),
        "l2_regularization": float(results_df.iloc[0]["l2_regularization"]),
    }

    print("\n" + "=" * 80)
    print("Best validation configuration:")
    print(best_params)
    print(results_df.head(10))

    results_df.to_csv(OUTPUT_DIR / "validation_tuning_results.csv", index=False)

    print("\nTraining final tuned HGB on train+validation...")

    final_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_val)

    final_model = make_model(best_params)
    final_model.fit(X_train_val, y_train_val, sample_weight=final_sample_weight)

    y_test_pred = final_model.predict(X_test)

    test_metrics = evaluate(y_test, y_test_pred)

    print("\nFinal tuned HGB test results:")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:       {test_metrics['weighted_f1']:.4f}")

    label_to_name = load_label_names()
    labels = sorted(y.unique())
    target_names = [label_to_name.get(int(label), str(label)) for label in labels]

    report = classification_report(
        y_test,
        y_test_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    metrics_df = pd.DataFrame([{
        "model": "tuned_hist_gradient_boosting",
        "n_objects": len(X),
        "n_train_val": len(X_train_val),
        "n_test": len(X_test),
        "n_classes": y.nunique(),
        "n_features": X.shape[1],
        **best_params,
        **test_metrics,
    }])

    predictions = pd.DataFrame({
        "true_label": y_test.to_numpy(),
        "predicted_label": y_test_pred,
        "correct": y_test.to_numpy() == y_test_pred,
    })

    metrics_df.to_csv(OUTPUT_DIR / "tuned_hgb_test_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "tuned_hgb_test_predictions.csv", index=False)

    with open(OUTPUT_DIR / "tuned_hgb_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'validation_tuning_results.csv'}")
    print(f"- {OUTPUT_DIR / 'tuned_hgb_test_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'tuned_hgb_test_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'tuned_hgb_classification_report.txt'}")


if __name__ == "__main__":
    main()