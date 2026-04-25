from pathlib import Path
import pandas as pd

CURRENT_DIR = Path("C:/Users/wesle/Desktop/Astrolara/MeuProjeto/astrotrust-ai")

PREDICTIONS_PATH = CURRENT_DIR / Path("results/uncertainty_random_forest/predictions_uncertainty.csv")
CALIBRATION_PATH = CURRENT_DIR / Path("results/uncertainty_random_forest/calibration_bins.csv")


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