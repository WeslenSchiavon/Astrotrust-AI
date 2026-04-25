from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT_DIR / Path("results/uncertainty_extra_trees/predictions_uncertainty.csv")
CALIBRATION_PATH = ROOT_DIR / Path("results/uncertainty_extra_trees/calibration_bins.csv")


def main():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    calibration = pd.read_csv(CALIBRATION_PATH)

    print("\nGeneral uncertainty behavior:")
    print(predictions.groupby("correct")[["confidence", "uncertainty"]].mean())

    print("\nMedian values:")
    print(predictions.groupby("correct")[["confidence", "uncertainty"]].median())

    print("\nAccuracy by confidence quartile:")
    predictions["confidence_bin"] = pd.qcut(
        predictions["confidence"],
        q=4,
        duplicates="drop"
    )

    confidence_summary = (
        predictions
        .groupby("confidence_bin", observed=True)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_uncertainty=("uncertainty", "mean"),
        )
        .reset_index()
    )

    print(confidence_summary)

    print("\nCalibration bins:")
    print(calibration)


if __name__ == "__main__":
    main()