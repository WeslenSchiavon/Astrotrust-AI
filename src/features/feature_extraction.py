from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ["u", "g", "r", "i", "z", "Y"]


def safe_std(x: pd.Series) -> float:
    if len(x) <= 1:
        return 0.0
    return float(x.std())


def extract_object_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("mjd")

    object_id = group["object_id"].iloc[0]
    label = group["label"].iloc[0]

    flux = group["flux"]
    flux_err = group["flux_err"]
    snr = group["snr"]
    mjd = group["mjd"]

    features = {
        "object_id": object_id,
        "label": label,

        "n_points": len(group),
        "n_bands": group["band"].nunique(),
        "time_span": float(mjd.max() - mjd.min()),

        "flux_mean": float(flux.mean()),
        "flux_std": safe_std(flux),
        "flux_min": float(flux.min()),
        "flux_max": float(flux.max()),
        "flux_median": float(flux.median()),
        "flux_amplitude": float(flux.max() - flux.min()),

        "flux_err_mean": float(flux_err.mean()),
        "flux_err_std": safe_std(flux_err),

        "snr_mean": float(snr.mean()),
        "snr_std": safe_std(snr),
        "snr_max": float(snr.max()),

        "zcmb": float(group["zcmb"].iloc[0]),
        "zhelio": float(group["zhelio"].iloc[0]),
        "mwebv": float(group["mwebv"].iloc[0]),
        "peakmjd": float(group["peakmjd"].iloc[0]),
        "mjd_detect_first": float(group["mjd_detect_first"].iloc[0]),
        "mjd_detect_last": float(group["mjd_detect_last"].iloc[0]),
        "nobs_truth": float(group["nobs"].iloc[0]),
    }

    for band in BANDS:
        band_group = group[group["band"] == band]

        features[f"n_{band}"] = len(band_group)

        if len(band_group) > 0:
            bflux = band_group["flux"]
            bsnr = band_group["snr"]

            features[f"flux_mean_{band}"] = float(bflux.mean())
            features[f"flux_std_{band}"] = safe_std(bflux)
            features[f"flux_min_{band}"] = float(bflux.min())
            features[f"flux_max_{band}"] = float(bflux.max())
            features[f"flux_amp_{band}"] = float(bflux.max() - bflux.min())
            features[f"snr_mean_{band}"] = float(bsnr.mean())
            features[f"snr_max_{band}"] = float(bsnr.max())
        else:
            features[f"flux_mean_{band}"] = 0.0
            features[f"flux_std_{band}"] = 0.0
            features[f"flux_min_{band}"] = 0.0
            features[f"flux_max_{band}"] = 0.0
            features[f"flux_amp_{band}"] = 0.0
            features[f"snr_mean_{band}"] = 0.0
            features[f"snr_max_{band}"] = 0.0

    return features


def build_feature_table(lightcurves: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, group in lightcurves.groupby("object_id"):
        rows.append(extract_object_features(group))

    features = pd.DataFrame(rows)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)

    return features


CURRENT_DIR = Path("C:/Users/wesle/Desktop/Astrolara/MeuProjeto/astrotrust-ai")
def main():
    input_path = CURRENT_DIR / Path("data/processed/elasticc2/lightcurves_1000obj.parquet")
    output_path = CURRENT_DIR / Path("data/processed/elasticc2/features_1000obj.parquet")

    print(f"Loading: {input_path}")
    lightcurves = pd.read_parquet(input_path)

    print("Extracting features...")
    features = build_feature_table(lightcurves)

    print(f"Objects: {len(features)}")
    print(f"Features: {len(features.columns)}")
    print(features.head())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()