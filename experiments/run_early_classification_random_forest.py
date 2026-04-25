from pathlib import Path
import sys
import math

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.features.feature_extraction import build_feature_table


LIGHTCURVES_PATH = ROOT_DIR / "data" / "processed" / "elasticc2" / "lightcurves_1000obj.parquet"
RESULTS_DIR = ROOT_DIR / "results" / "early_classification_random_forest"

FRACTIONS = [0.25, 0.50, 0.75, 1.00]
MIN_OBJECTS_PER_CLASS = 20
RANDOM_STATE = 42


def make_partial_lightcurves(lightcurves: pd.DataFrame, fraction: float) -> pd.DataFrame:
    partial_groups = []

    for _, group in lightcurves.groupby("object_id"):
        group = group.sort_values("mjd")
        n_points = len(group)
        n_keep = max(1, math.ceil(n_points * fraction))
        partial_groups.append(group.iloc[:n_keep])

    return pd.concat(partial_groups, ignore_index=True)


def evaluate_fraction(lightcurves: pd.DataFrame, fraction: float, valid_classes) -> dict:
    print("\n" + "=" * 80)
    print(f"Evaluating fraction: {fraction:.2f}")

    partial_lightcurves = make_partial_lightcurves(lightcurves, fraction)

    print(f"Photometric points used: {len(partial_lightcurves)}")

    features = build_feature_table(partial_lightcurves)

    features = features[features["label"].isin(valid_classes)].copy()

    y = features["label"]
    X = features.drop(columns=["object_id", "label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

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

    return {
        "fraction": fraction,
        "n_objects": len(features),
        "n_points": len(partial_lightcurves),
        "n_classes": len(valid_classes),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading light curves from: {LIGHTCURVES_PATH}")
    lightcurves = pd.read_parquet(LIGHTCURVES_PATH)

    object_labels = lightcurves[["object_id", "label"]].drop_duplicates()

    class_counts = object_labels["label"].value_counts().sort_index()
    valid_classes = class_counts[class_counts >= MIN_OBJECTS_PER_CLASS].index.tolist()

    print("\nClasses selected for this experiment:")
    print(class_counts[class_counts >= MIN_OBJECTS_PER_CLASS])

    print(f"\nNumber of valid classes: {len(valid_classes)}")

    results = []

    for fraction in FRACTIONS:
        result = evaluate_fraction(lightcurves, fraction, valid_classes)
        results.append(result)

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "early_classification_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Summary:")
    print(results_df)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()