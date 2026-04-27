from pathlib import Path
import argparse

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
from sklearn.utils.class_weight import compute_sample_weight


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT_DIR / "results" / "large_calibrated_hgb"

RANDOM_STATE = 42
N_BINS = 10


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n_classes = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropy / np.log(n_classes)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    n_samples = len(y_true)
    n_classes = probabilities.shape[1]

    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true] = 1.0

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


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def prepare_dataset(features_path: Path):
    df = pd.read_parquet(features_path)

    y = df["label"].to_numpy()
    X = df.drop(columns=["object_id", "label"])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y, df


def evaluate(name, probabilities, y_test, results_dir):
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    uncertainty = normalized_entropy(probabilities)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    brier = multiclass_brier_score(y_test, probabilities)
    ece, bins = expected_calibration_error(
        y_true=y_test,
        y_pred=y_pred,
        confidences=confidences,
        n_bins=N_BINS,
    )

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

    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification report:")
    print(report)

    model_dir = results_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_pred,
        "confidence": confidences,
        "uncertainty": uncertainty,
        "correct": y_test == y_pred,
    }).to_csv(model_dir / "predictions.csv", index=False)

    bins.to_csv(model_dir / "calibration_bins.csv", index=False)

    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

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
        / f"features_v3_contextual_{args.n_objects}obj.parquet"
    )

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    results_dir = RESULTS_ROOT / f"{args.n_objects}obj"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {features_path}")

    X, y, df = prepare_dataset(features_path)

    print("\nDataset summary:")
    print(f"Objects: {len(df)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Features: {X.shape[1]}")

    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
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
    base_model = make_model()
    base_model.fit(X_train, y_train, sample_weight=sample_weight)

    results = []

    probs_uncalibrated = base_model.predict_proba(X_test)
    results.append(
        evaluate(
            name="hgb_uncalibrated",
            probabilities=probs_uncalibrated,
            y_test=y_test,
            results_dir=results_dir,
        )
    )

    print("\nFitting sigmoid calibration...")
    sigmoid = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv="prefit",
    )
    sigmoid.fit(X_cal, y_cal)

    probs_sigmoid = sigmoid.predict_proba(X_test)
    results.append(
        evaluate(
            name="hgb_sigmoid_calibrated",
            probabilities=probs_sigmoid,
            y_test=y_test,
            results_dir=results_dir,
        )
    )

    print("\nFitting isotonic calibration...")
    isotonic = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv="prefit",
    )
    isotonic.fit(X_cal, y_cal)

    probs_isotonic = isotonic.predict_proba(X_test)
    results.append(
        evaluate(
            name="hgb_isotonic_calibrated",
            probabilities=probs_isotonic,
            y_test=y_test,
            results_dir=results_dir,
        )
    )

    summary = pd.DataFrame(results)
    summary = summary.sort_values(["ece", "macro_f1"], ascending=[True, False])

    output_path = results_dir / "calibration_comparison_summary.csv"
    summary.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Calibration comparison summary:")
    print(summary)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()