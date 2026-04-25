from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT_DIR / Path("data/processed/elasticc2/features_1000obj.parquet")
RESULTS_DIR = ROOT_DIR / Path("results/uncertainty_extra_trees")

MIN_OBJECTS_PER_CLASS = 20
RANDOM_STATE = 42
N_BINS = 10


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n_classes = probabilities.shape[1]

    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    max_entropy = np.log(n_classes)

    return entropy / max_entropy


def multiclass_brier_score(y_true_encoded: np.ndarray, probabilities: np.ndarray) -> float:
    n_samples = len(y_true_encoded)
    n_classes = probabilities.shape[1]

    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true_encoded] = 1.0

    return np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        n_bin = np.sum(mask)

        if n_bin == 0:
            rows.append({
                "bin": i,
                "lower": lower,
                "upper": upper,
                "n": 0,
                "accuracy": np.nan,
                "confidence": np.nan,
                "gap": np.nan,
            })
            continue

        bin_accuracy = np.mean(y_true[mask] == y_pred[mask])
        bin_confidence = np.mean(confidences[mask])
        gap = abs(bin_accuracy - bin_confidence)

        ece += (n_bin / len(y_true)) * gap

        rows.append({
            "bin": i,
            "lower": lower,
            "upper": upper,
            "n": int(n_bin),
            "accuracy": bin_accuracy,
            "confidence": bin_confidence,
            "gap": gap,
        })

    return ece, pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    class_counts = df["label"].value_counts().sort_index()
    valid_classes = class_counts[class_counts >= MIN_OBJECTS_PER_CLASS].index

    df = df[df["label"].isin(valid_classes)].copy()

    print("\nClasses used:")
    print(df["label"].value_counts().sort_index())

    y = df["label"]
    X = df.drop(columns=["object_id", "label"])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print("\nTraining Extra Trees...")

    model = ExtraTreesClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    y_pred = np.argmax(probabilities, axis=1)

    confidences = np.max(probabilities, axis=1)
    uncertainty = normalized_entropy(probabilities)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    brier = multiclass_brier_score(y_test, probabilities)
    ece, calibration_bins = expected_calibration_error(
        y_true=y_test,
        y_pred=y_pred,
        confidences=confidences,
        n_bins=N_BINS,
    )

    print("\nClassification results:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    print("\nUncertainty/calibration results:")
    print(f"Brier score:       {brier:.4f}")
    print(f"ECE:               {ece:.4f}")
    print(f"Mean confidence:   {confidences.mean():.4f}")
    print(f"Mean uncertainty:  {uncertainty.mean():.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0,
        )
    )

    true_labels = label_encoder.inverse_transform(y_test)
    pred_labels = label_encoder.inverse_transform(y_pred)

    predictions = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "confidence": confidences,
        "uncertainty": uncertainty,
        "correct": true_labels == pred_labels,
    })

    metrics = pd.DataFrame([{
        "model": "extra_trees",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier,
        "ece": ece,
        "mean_confidence": confidences.mean(),
        "mean_uncertainty": uncertainty.mean(),
        "n_test": len(y_test),
        "n_classes": len(label_encoder.classes_),
    }])

    metrics.to_csv(RESULTS_DIR / "metrics_uncertainty.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "predictions_uncertainty.csv", index=False)
    calibration_bins.to_csv(RESULTS_DIR / "calibration_bins.csv", index=False)

    print("\nSaved results:")
    print(f"- {RESULTS_DIR / 'metrics_uncertainty.csv'}")
    print(f"- {RESULTS_DIR / 'predictions_uncertainty.csv'}")
    print(f"- {RESULTS_DIR / 'calibration_bins.csv'}")


if __name__ == "__main__":
    main()