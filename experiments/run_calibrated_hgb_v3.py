from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT_DIR / Path("data/processed/elasticc2/features_v3_contextual_1000obj.parquet")
RESULTS_DIR = ROOT_DIR / Path("results/calibrated_hgb_v3")

MIN_OBJECTS_PER_CLASS = 20
RANDOM_STATE = 42
N_BINS = 10


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n_classes = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropy / np.log(n_classes)


def multiclass_brier_score(y_true_encoded: np.ndarray, probabilities: np.ndarray) -> float:
    n_samples = len(y_true_encoded)
    n_classes = probabilities.shape[1]

    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true_encoded] = 1.0

    return np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))


def expected_calibration_error(y_true, y_pred, confidences, n_bins=10):
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


def evaluate_probabilities(name, probabilities, y_test, label_encoder):
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    uncertainty = normalized_entropy(probabilities)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    brier = multiclass_brier_score(y_test, probabilities)
    ece, bins = expected_calibration_error(y_test, y_pred, confidences, n_bins=N_BINS)

    print("\n" + "=" * 80)
    print(f"Model: {name}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")
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

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier,
        "ece": ece,
        "mean_confidence": confidences.mean(),
        "mean_uncertainty": uncertainty.mean(),
        "predictions": predictions,
        "calibration_bins": bins,
    }


def make_base_model():
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


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

    # Split into train/calibration/test.
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_cal,
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Calibration size: {len(X_cal)}")
    print(f"Test size: {len(X_test)}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print("\nTraining base HistGradientBoosting...")

    base_model = make_base_model()
    base_model.fit(X_train, y_train, sample_weight=sample_weight)

    results = []

    # Uncalibrated
    probs_uncalibrated = base_model.predict_proba(X_test)
    results.append(
        evaluate_probabilities(
            "hgb_uncalibrated",
            probs_uncalibrated,
            y_test,
            label_encoder,
        )
    )

    # Sigmoid calibration
    print("\nFitting sigmoid calibration...")
    sigmoid_calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv="prefit",
    )
    sigmoid_calibrator.fit(X_cal, y_cal)

    probs_sigmoid = sigmoid_calibrator.predict_proba(X_test)
    results.append(
        evaluate_probabilities(
            "hgb_sigmoid_calibrated",
            probs_sigmoid,
            y_test,
            label_encoder,
        )
    )

    # Isotonic calibration
    print("\nFitting isotonic calibration...")
    isotonic_calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv="prefit",
    )
    isotonic_calibrator.fit(X_cal, y_cal)

    probs_isotonic = isotonic_calibrator.predict_proba(X_test)
    results.append(
        evaluate_probabilities(
            "hgb_isotonic_calibrated",
            probs_isotonic,
            y_test,
            label_encoder,
        )
    )

    summary_rows = []

    for result in results:
        name = result["model"]

        model_dir = RESULTS_DIR / name
        model_dir.mkdir(parents=True, exist_ok=True)

        result["predictions"].to_csv(model_dir / "predictions.csv", index=False)
        result["calibration_bins"].to_csv(model_dir / "calibration_bins.csv", index=False)

        summary_rows.append({
            "model": name,
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "brier_score": result["brier_score"],
            "ece": result["ece"],
            "mean_confidence": result["mean_confidence"],
            "mean_uncertainty": result["mean_uncertainty"],
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["ece", "macro_f1"], ascending=[True, False])

    print("\n" + "=" * 80)
    print("Calibration comparison summary:")
    print(summary)

    summary.to_csv(RESULTS_DIR / "calibration_comparison_summary.csv", index=False)

    print(f"\nSaved summary: {RESULTS_DIR / 'calibration_comparison_summary.csv'}")


if __name__ == "__main__":
    main()