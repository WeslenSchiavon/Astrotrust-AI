from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ["u", "g", "r", "i", "z", "Y"]


def safe_std(x: pd.Series) -> float:
    if len(x) <= 1:
        return 0.0
    return float(x.std())


def safe_skew(x: pd.Series) -> float:
    if len(x) <= 2:
        return 0.0
    value = x.skew()
    return 0.0 if pd.isna(value) else float(value)


def safe_kurtosis(x: pd.Series) -> float:
    if len(x) <= 3:
        return 0.0
    value = x.kurtosis()
    return 0.0 if pd.isna(value) else float(value)


def safe_slope(x_time: pd.Series, y_flux: pd.Series) -> float:
    if len(x_time) <= 1:
        return 0.0

    dt = float(x_time.max() - x_time.min())
    if np.isclose(dt, 0.0):
        return 0.0

    return float((y_flux.iloc[-1] - y_flux.iloc[0]) / dt)


def weighted_mean(values: pd.Series, errors: pd.Series) -> float:
    errors = errors.replace(0, np.nan)
    weights = 1.0 / (errors ** 2)
    weights = weights.replace([np.inf, -np.inf], np.nan)

    valid = values.notna() & weights.notna()

    if valid.sum() == 0:
        return float(values.mean())

    return float(np.average(values[valid], weights=weights[valid]))


def compute_peak_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("mjd")

    idx_max = group["flux"].idxmax()
    peak_row = group.loc[idx_max]

    first_mjd = float(group["mjd"].min())
    last_mjd = float(group["mjd"].max())
    peak_mjd_observed = float(peak_row["mjd"])

    time_span = last_mjd - first_mjd

    time_to_observed_peak = peak_mjd_observed - first_mjd
    time_after_observed_peak = last_mjd - peak_mjd_observed

    before_peak = group[group["mjd"] <= peak_mjd_observed].sort_values("mjd")
    after_peak = group[group["mjd"] >= peak_mjd_observed].sort_values("mjd")

    rise_slope = safe_slope(before_peak["mjd"], before_peak["flux"])
    decline_slope = safe_slope(after_peak["mjd"], after_peak["flux"])

    return {
        "observed_peak_mjd_relative": time_to_observed_peak,
        "time_after_observed_peak": time_after_observed_peak,
        "observed_peak_flux": float(peak_row["flux"]),
        "observed_peak_snr": float(peak_row["snr"]),
        "observed_peak_band_code": BANDS.index(peak_row["band"]) if peak_row["band"] in BANDS else -1,
        "rise_slope_observed": rise_slope,
        "decline_slope_observed": decline_slope,
        "relative_peak_position": float(time_to_observed_peak / time_span) if time_span > 0 else 0.0,
    }


def extract_band_features(group: pd.DataFrame, band: str) -> dict:
    band_group = group[group["band"] == band].sort_values("mjd")

    prefix = f"{band}"

    if len(band_group) == 0:
        return {
            f"n_{prefix}": 0,
            f"flux_mean_{prefix}": 0.0,
            f"flux_std_{prefix}": 0.0,
            f"flux_min_{prefix}": 0.0,
            f"flux_max_{prefix}": 0.0,
            f"flux_median_{prefix}": 0.0,
            f"flux_amp_{prefix}": 0.0,
            f"flux_weighted_mean_{prefix}": 0.0,
            f"flux_skew_{prefix}": 0.0,
            f"flux_kurtosis_{prefix}": 0.0,
            f"snr_mean_{prefix}": 0.0,
            f"snr_max_{prefix}": 0.0,
            f"time_span_{prefix}": 0.0,
            f"slope_{prefix}": 0.0,
            f"time_to_peak_{prefix}": 0.0,
            f"peak_flux_{prefix}": 0.0,
        }

    flux = band_group["flux"]
    flux_err = band_group["flux_err"]
    snr = band_group["snr"]
    mjd = band_group["mjd"]

    idx_max = flux.idxmax()
    peak_mjd = float(band_group.loc[idx_max, "mjd"])
    first_mjd = float(mjd.min())

    return {
        f"n_{prefix}": len(band_group),
        f"flux_mean_{prefix}": float(flux.mean()),
        f"flux_std_{prefix}": safe_std(flux),
        f"flux_min_{prefix}": float(flux.min()),
        f"flux_max_{prefix}": float(flux.max()),
        f"flux_median_{prefix}": float(flux.median()),
        f"flux_amp_{prefix}": float(flux.max() - flux.min()),
        f"flux_weighted_mean_{prefix}": weighted_mean(flux, flux_err),
        f"flux_skew_{prefix}": safe_skew(flux),
        f"flux_kurtosis_{prefix}": safe_kurtosis(flux),
        f"snr_mean_{prefix}": float(snr.mean()),
        f"snr_max_{prefix}": float(snr.max()),
        f"time_span_{prefix}": float(mjd.max() - mjd.min()),
        f"slope_{prefix}": safe_slope(mjd, flux),
        f"time_to_peak_{prefix}": float(peak_mjd - first_mjd),
        f"peak_flux_{prefix}": float(flux.max()),
    }


def add_color_like_features(features: dict) -> dict:
    """
    Simple color-like descriptors based on differences between band mean fluxes.
    These are not physical calibrated colors, but useful ML descriptors.
    """
    band_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("z", "Y"),
        ("g", "i"),
        ("r", "z"),
    ]

    for b1, b2 in band_pairs:
        f1 = features.get(f"flux_mean_{b1}", 0.0)
        f2 = features.get(f"flux_mean_{b2}", 0.0)

        features[f"flux_diff_{b1}_{b2}"] = float(f1 - f2)
        features[f"flux_ratio_{b1}_{b2}"] = float(f1 / f2) if not np.isclose(f2, 0.0) else 0.0

    return features


def extract_object_features(group: pd.DataFrame) -> dict:
    group = group.sort_values("mjd")

    object_id = group["object_id"].iloc[0]
    label = group["label"].iloc[0]

    flux = group["flux"]
    flux_err = group["flux_err"]
    snr = group["snr"]
    mjd = group["mjd"]

    first_mjd = float(mjd.min())
    last_mjd = float(mjd.max())
    time_span = last_mjd - first_mjd

    features = {
        "object_id": object_id,
        "label": label,

        # Source-position descriptors from detections.
        "ra_mean": float(group["ra"].mean()),
        "dec_mean": float(group["dec"].mean()),
        "ra_std": safe_std(group["ra"]),
        "dec_std": safe_std(group["dec"]),

        # Sampling descriptors.
        "n_points": len(group),
        "n_bands": group["band"].nunique(),
        "time_span": time_span,
        "mean_cadence": float(time_span / (len(group) - 1)) if len(group) > 1 else 0.0,

        # Global flux descriptors.
        "flux_mean": float(flux.mean()),
        "flux_weighted_mean": weighted_mean(flux, flux_err),
        "flux_std": safe_std(flux),
        "flux_min": float(flux.min()),
        "flux_max": float(flux.max()),
        "flux_median": float(flux.median()),
        "flux_amplitude": float(flux.max() - flux.min()),
        "flux_skew": safe_skew(flux),
        "flux_kurtosis": safe_kurtosis(flux),

        # Error and SNR descriptors.
        "flux_err_mean": float(flux_err.mean()),
        "flux_err_std": safe_std(flux_err),
        "flux_err_median": float(flux_err.median()),
        "snr_mean": float(snr.mean()),
        "snr_std": safe_std(snr),
        "snr_min": float(snr.min()),
        "snr_max": float(snr.max()),
        "snr_median": float(snr.median()),

        # Temporal trend descriptor.
        "global_slope": safe_slope(mjd, flux),
    }

    features.update(compute_peak_features(group))

    for band in BANDS:
        features.update(extract_band_features(group, band))

    features = add_color_like_features(features)

    return features


def build_feature_table(lightcurves: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, group in lightcurves.groupby("object_id"):
        rows.append(extract_object_features(group))

    features = pd.DataFrame(rows)

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)

    return features


def main():
    ROOT_DIR = Path(__file__).resolve().parents[2]
    
    input_path = ROOT_DIR /  Path("data/processed/elasticc2/lightcurves_1000obj.parquet")
    output_path = ROOT_DIR /  Path("data/processed/elasticc2/features_v2_1000obj.parquet")

    print(f"Loading: {input_path}")
    lightcurves = pd.read_parquet(input_path)

    print("Extracting leakage-safe temporal features...")
    features = build_feature_table(lightcurves)

    print(f"Objects: {len(features)}")
    print(f"Features: {len(features.columns)}")
    print(features.head())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()