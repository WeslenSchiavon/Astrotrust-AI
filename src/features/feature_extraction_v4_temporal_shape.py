from pathlib import Path
import argparse

import sys
import numpy as np
import pandas as pd

# Adiciona a pasta src ao path automaticamente
ROOT_DIR = Path(__file__).resolve().parents[2]  # vai até astrotrust-ai
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.features.feature_extraction_v2 import build_feature_table



INPUT_ROOT = ROOT_DIR / Path("data/processed/elasticc2_large")
OUTPUT_ROOT = ROOT_DIR / Path("data/processed/elasticc2_large")

BANDS = ["u", "g", "r", "i", "z", "Y"]

EPS = 1e-8


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) < EPS:
        return 0.0
    if a is None or not np.isfinite(a):
        return 0.0
    return float(a / b)


def safe_slope(y2, y1, t2, t1):
    dt = t2 - t1
    if not np.isfinite(dt) or abs(dt) < EPS:
        return 0.0
    return float((y2 - y1) / dt)


def normalize_band(value):
    value = str(value).strip()
    if value.lower() == "y":
        return "Y"
    return value.lower()


def positive_auc(mjd, flux):
    if len(mjd) < 2:
        return 0.0

    order = np.argsort(mjd)
    t = np.asarray(mjd)[order]
    f = np.asarray(flux)[order]

    f_pos = np.maximum(f, 0.0)

    try:
        return float(np.trapz(f_pos, t))
    except Exception:
        return 0.0


def width_above_fraction(mjd, flux, fraction):
    if len(mjd) == 0:
        return 0.0

    mjd = np.asarray(mjd, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if len(mjd) < 2:
        return 0.0

    peak = np.nanmax(flux)

    if not np.isfinite(peak) or peak <= 0:
        return 0.0

    threshold = fraction * peak
    mask = flux >= threshold

    if mask.sum() < 2:
        return 0.0

    return float(np.nanmax(mjd[mask]) - np.nanmin(mjd[mask]))


def extract_shape_for_series(mjd, flux, snr=None, prefix=""):
    mjd = np.asarray(mjd, dtype=float)
    flux = np.asarray(flux, dtype=float)

    valid = np.isfinite(mjd) & np.isfinite(flux)
    mjd = mjd[valid]
    flux = flux[valid]

    if snr is not None:
        snr = np.asarray(snr, dtype=float)
        snr = snr[valid]
    else:
        snr = np.zeros_like(flux)

    out = {}

    if len(mjd) == 0:
        keys = [
            "peak_flux",
            "peak_mjd",
            "time_to_peak",
            "time_after_peak",
            "width_25",
            "width_50",
            "width_75",
            "auc_positive",
            "rise_slope",
            "decline_slope",
            "pre_peak_n",
            "post_peak_n",
            "pre_post_n_ratio",
            "peak_snr",
            "positive_fraction",
            "asymmetry_time",
        ]

        for key in keys:
            out[f"{prefix}{key}"] = 0.0

        return out

    order = np.argsort(mjd)
    mjd = mjd[order]
    flux = flux[order]
    snr = snr[order]

    first_t = float(mjd[0])
    last_t = float(mjd[-1])
    first_flux = float(flux[0])
    last_flux = float(flux[-1])

    peak_idx = int(np.nanargmax(flux))
    peak_flux = float(flux[peak_idx])
    peak_mjd = float(mjd[peak_idx])
    peak_snr = float(snr[peak_idx]) if len(snr) > peak_idx else 0.0

    time_to_peak = max(0.0, peak_mjd - first_t)
    time_after_peak = max(0.0, last_t - peak_mjd)

    pre_peak_n = int(np.sum(mjd < peak_mjd))
    post_peak_n = int(np.sum(mjd > peak_mjd))

    rise_slope = safe_slope(peak_flux, first_flux, peak_mjd, first_t)
    decline_slope = safe_slope(last_flux, peak_flux, last_t, peak_mjd)

    total_time = time_to_peak + time_after_peak
    asymmetry_time = safe_div(time_after_peak - time_to_peak, total_time)

    positive_fraction = float(np.mean(flux > 0.0)) if len(flux) > 0 else 0.0

    out[f"{prefix}peak_flux"] = peak_flux
    out[f"{prefix}peak_mjd"] = peak_mjd
    out[f"{prefix}time_to_peak"] = time_to_peak
    out[f"{prefix}time_after_peak"] = time_after_peak
    out[f"{prefix}width_25"] = width_above_fraction(mjd, flux, 0.25)
    out[f"{prefix}width_50"] = width_above_fraction(mjd, flux, 0.50)
    out[f"{prefix}width_75"] = width_above_fraction(mjd, flux, 0.75)
    out[f"{prefix}auc_positive"] = positive_auc(mjd, flux)
    out[f"{prefix}rise_slope"] = rise_slope
    out[f"{prefix}decline_slope"] = decline_slope
    out[f"{prefix}pre_peak_n"] = float(pre_peak_n)
    out[f"{prefix}post_peak_n"] = float(post_peak_n)
    out[f"{prefix}pre_post_n_ratio"] = safe_div(pre_peak_n, post_peak_n + 1.0)
    out[f"{prefix}peak_snr"] = peak_snr
    out[f"{prefix}positive_fraction"] = positive_fraction
    out[f"{prefix}asymmetry_time"] = asymmetry_time

    return out


def extract_temporal_shape_features(lightcurves: pd.DataFrame) -> pd.DataFrame:
    lc = lightcurves.copy()

    lc["band"] = lc["band"].map(normalize_band)

    rows = []

    grouped = lc.groupby("object_id", sort=False)

    for object_id, group in grouped:
        group = group.sort_values("mjd")

        label = int(group["label"].iloc[0])

        row = {
            "object_id": int(object_id),
            "label": label,
        }

        global_shape = extract_shape_for_series(
            mjd=group["mjd"].to_numpy(),
            flux=group["flux"].to_numpy(),
            snr=group["snr"].to_numpy() if "snr" in group.columns else None,
            prefix="shape_global_",
        )
        row.update(global_shape)

        peak_flux_by_band = {}
        peak_mjd_by_band = {}

        for band in BANDS:
            band_group = group[group["band"] == band]

            band_shape = extract_shape_for_series(
                mjd=band_group["mjd"].to_numpy(),
                flux=band_group["flux"].to_numpy(),
                snr=band_group["snr"].to_numpy() if "snr" in band_group.columns else None,
                prefix=f"shape_{band}_",
            )

            row.update(band_shape)

            peak_flux_by_band[band] = band_shape.get(f"shape_{band}_peak_flux", 0.0)
            peak_mjd_by_band[band] = band_shape.get(f"shape_{band}_peak_mjd", 0.0)

        # Cross-band peak flux ratios and peak-time differences.
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
            f1 = peak_flux_by_band.get(b1, 0.0)
            f2 = peak_flux_by_band.get(b2, 0.0)

            t1 = peak_mjd_by_band.get(b1, 0.0)
            t2 = peak_mjd_by_band.get(b2, 0.0)

            row[f"shape_peak_flux_ratio_{b1}_{b2}"] = safe_div(f1, f2)

            if t1 > 0 and t2 > 0:
                row[f"shape_peak_mjd_diff_{b1}_{b2}"] = float(t1 - t2)
            else:
                row[f"shape_peak_mjd_diff_{b1}_{b2}"] = 0.0

        # Peak band one-hot.
        peak_band = normalize_band(group.loc[group["flux"].idxmax(), "band"])

        for band in BANDS:
            row[f"shape_peak_band_is_{band}"] = 1.0 if peak_band == band else 0.0

        rows.append(row)

    features = pd.DataFrame(rows)

    return features


def add_context_engineered_features(context: pd.DataFrame) -> pd.DataFrame:
    context = context.copy()

    color_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("z", "y"),
        ("g", "i"),
        ("r", "z"),
    ]

    for b1, b2 in color_pairs:
        c1 = f"hostgal_mag_{b1}"
        c2 = f"hostgal_mag_{b2}"

        if c1 in context.columns and c2 in context.columns:
            context[f"hostgal_color_{b1}_{b2}"] = context[c1] - context[c2]

    if "hostgal_zphot" in context.columns and "hostgal_zphot_err" in context.columns:
        denom = context["hostgal_zphot"].abs().replace(0, np.nan)
        context["hostgal_zphot_relative_err"] = context["hostgal_zphot_err"] / denom

    if "hostgal_zphot_q100" in context.columns and "hostgal_zphot_q000" in context.columns:
        context["hostgal_zphot_q100_q000_width"] = (
            context["hostgal_zphot_q100"] - context["hostgal_zphot_q000"]
        )

    if "hostgal_zphot_q090" in context.columns and "hostgal_zphot_q010" in context.columns:
        context["hostgal_zphot_q090_q010_width"] = (
            context["hostgal_zphot_q090"] - context["hostgal_zphot_q010"]
        )

    if "hostgal_zphot_q080" in context.columns and "hostgal_zphot_q020" in context.columns:
        context["hostgal_zphot_q080_q020_width"] = (
            context["hostgal_zphot_q080"] - context["hostgal_zphot_q020"]
        )

    return context


def add_position_to_lightcurves(lightcurves: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    position_cols = ["object_id", "ra", "decl"]
    missing = [c for c in position_cols if c not in context.columns]

    if missing:
        raise ValueError(f"Context missing required position columns: {missing}")

    position = context[position_cols].drop_duplicates("object_id")

    lightcurves = lightcurves.merge(position, on="object_id", how="left")
    lightcurves = lightcurves.rename(columns={"decl": "dec"})

    lightcurves["ra"] = pd.to_numeric(lightcurves["ra"], errors="coerce").fillna(0.0)
    lightcurves["dec"] = pd.to_numeric(lightcurves["dec"], errors="coerce").fillna(0.0)

    return lightcurves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)

    args = parser.parse_args()

    suffix = f"{args.n_objects}obj"

    lightcurves_path = args.input_root / f"forced_lightcurves_{suffix}.parquet"
    context_path = args.input_root / f"object_context_{suffix}.parquet"
    output_path = args.output_root / f"features_v4_temporal_shape_{suffix}.parquet"

    print(f"Loading light curves: {lightcurves_path}")
    lightcurves = pd.read_parquet(lightcurves_path)

    print(f"Loading context: {context_path}")
    context = pd.read_parquet(context_path)

    print("Adding ra/dec to light curves...")
    lightcurves = add_position_to_lightcurves(lightcurves, context)

    print("Extracting v2 temporal statistical features...")
    temporal_features = build_feature_table(lightcurves)
    print(f"Temporal statistical features: {temporal_features.shape}")

    print("Extracting v4 temporal shape features...")
    shape_features = extract_temporal_shape_features(lightcurves)
    print(f"Temporal shape features: {shape_features.shape}")

    print("Engineering contextual features...")
    context = add_context_engineered_features(context)
    context_features = context.drop(columns=["label"], errors="ignore")

    print("Merging v2 + v4 + context features...")

    features = temporal_features.merge(
        shape_features.drop(columns=["label"], errors="ignore"),
        on="object_id",
        how="left",
    )

    features = features.merge(
        context_features,
        on="object_id",
        how="left",
    )

    for col in features.columns:
        if col in ["object_id", "label"]:
            continue
        features[col] = pd.to_numeric(features[col], errors="coerce")

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)

    args.output_root.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Objects: {len(features)}")
    print(f"Features: {features.shape[1]}")

    print("\nClass distribution:")
    print(features["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()