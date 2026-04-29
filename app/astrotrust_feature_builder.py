from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "results" / "hybrid_inference_artifacts_250k"
INFERENCE_METADATA_PATH = ARTIFACT_DIR / "inference_metadata.json"

BAND_ALIASES = {
    "0": "u", "1": "g", "2": "r", "3": "i", "4": "z", "5": "Y",
    0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "Y",
    "u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "Y", "Y": "Y",
    "b'u'": "u", "b'g'": "g", "b'r'": "r", "b'i'": "i", "b'z'": "z", "b'Y'": "Y",
}

BANDS = ["u", "g", "r", "i", "z", "Y"]
EPS = 1e-8


def load_expected_tabular_columns() -> list[str]:
    if not INFERENCE_METADATA_PATH.exists():
        raise FileNotFoundError(INFERENCE_METADATA_PATH)

    meta = json.loads(INFERENCE_METADATA_PATH.read_text(encoding="utf-8"))
    return [str(c) for c in meta["tabular_columns"]]


def normalize_band(value):
    if isinstance(value, bytes):
        value = value.decode(errors="ignore").strip()

    value_str = str(value).strip()

    if value_str in BAND_ALIASES:
        return BAND_ALIASES[value_str]

    if value_str.lower() in BAND_ALIASES:
        return BAND_ALIASES[value_str.lower()]

    return value_str


def _safe_numeric(series, default=np.nan):
    out = pd.to_numeric(series, errors="coerce")
    if default is not np.nan:
        out = out.fillna(default)
    return out


def _stat_values(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q05": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "q95": 0.0,
            "amplitude": 0.0,
            "mad": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    median = float(np.median(values))
    q05, q25, q75, q95 = np.percentile(values, [5, 25, 75, 95])

    if std > EPS:
        centered = (values - mean) / std
        skew = float(np.mean(centered ** 3))
        kurtosis = float(np.mean(centered ** 4) - 3.0)
    else:
        skew = 0.0
        kurtosis = 0.0

    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": median,
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
        "amplitude": float(np.max(values) - np.min(values)),
        "mad": float(np.median(np.abs(values - median))),
        "skew": skew,
        "kurtosis": kurtosis,
    }


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2 or np.nanstd(x) < EPS:
        return 0.0

    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def _compute_lightcurve_feature_pool(lc_obj: pd.DataFrame, head_row: pd.Series | dict | None = None) -> dict:
    lc = lc_obj.copy()

    if "mjd" not in lc.columns or "flux" not in lc.columns or "band" not in lc.columns:
        raise ValueError("Expected normalized columns: mjd, band, flux")

    lc["mjd"] = _safe_numeric(lc["mjd"])
    lc["flux"] = _safe_numeric(lc["flux"])
    lc["band"] = lc["band"].map(normalize_band)

    if "flux_err" in lc.columns:
        lc["flux_err"] = _safe_numeric(lc["flux_err"])
    else:
        lc["flux_err"] = np.nan

    lc = lc.dropna(subset=["mjd", "flux"]).copy()

    if lc.empty:
        raise ValueError("No valid light-curve rows after cleaning.")

    flux = lc["flux"].to_numpy(dtype=float)
    mjd = lc["mjd"].to_numpy(dtype=float)
    flux_err = lc["flux_err"].to_numpy(dtype=float)

    features = {}

    object_id = None
    if "object_id" in lc.columns and len(lc) > 0:
        object_id = str(lc["object_id"].iloc[0])

    features["object_id"] = object_id
    features["n_obs"] = float(len(lc))
    features["n_detections"] = float(len(lc))
    features["n_bands"] = float(lc["band"].nunique())
    features["mjd_min"] = float(np.nanmin(mjd))
    features["mjd_max"] = float(np.nanmax(mjd))
    features["mjd_span"] = float(np.nanmax(mjd) - np.nanmin(mjd))
    features["time_span"] = features["mjd_span"]
    features["baseline"] = features["mjd_span"]

    flux_stats = _stat_values(flux)
    for key, value in flux_stats.items():
        features[f"flux_{key}"] = value
        features[key] = value

    if np.isfinite(flux_err).any():
        err_stats = _stat_values(flux_err)
        for key, value in err_stats.items():
            features[f"flux_err_{key}"] = value
            features[f"err_{key}"] = value

        valid_err = np.isfinite(flux_err) & (np.abs(flux_err) > EPS)
        snr = np.zeros_like(flux, dtype=float)
        snr[valid_err] = flux[valid_err] / flux_err[valid_err]
        snr_stats = _stat_values(snr)
        for key, value in snr_stats.items():
            features[f"snr_{key}"] = value
    else:
        snr = np.zeros_like(flux, dtype=float)

    # Temporal shape approximations.
    idx_max = int(np.nanargmax(flux))
    idx_min = int(np.nanargmin(flux))
    t_peak = float(mjd[idx_max])
    t_min_flux = float(mjd[idx_min])

    features["t_peak"] = t_peak
    features["time_peak"] = t_peak
    features["mjd_peak"] = t_peak
    features["t_min_flux"] = t_min_flux
    features["time_min_flux"] = t_min_flux
    features["flux_peak"] = float(flux[idx_max])
    features["peak_flux"] = float(flux[idx_max])
    features["flux_at_peak"] = float(flux[idx_max])

    features["slope_global"] = _linear_slope(mjd, flux)

    before = lc[lc["mjd"] <= t_peak]
    after = lc[lc["mjd"] >= t_peak]
    features["slope_rise"] = _linear_slope(before["mjd"].to_numpy(), before["flux"].to_numpy())
    features["slope_decline"] = _linear_slope(after["mjd"].to_numpy(), after["flux"].to_numpy())
    features["rise_time"] = float(t_peak - np.nanmin(mjd)) if len(mjd) else 0.0
    features["decline_time"] = float(np.nanmax(mjd) - t_peak) if len(mjd) else 0.0

    # Weighted time moments.
    weights = np.abs(flux)
    if np.sum(weights) > EPS:
        t_weighted = float(np.sum(mjd * weights) / np.sum(weights))
        features["time_weighted_mean"] = t_weighted
        features["weighted_mjd_mean"] = t_weighted
        features["time_weighted_std"] = float(np.sqrt(np.sum(weights * (mjd - t_weighted) ** 2) / np.sum(weights)))
    else:
        features["time_weighted_mean"] = 0.0
        features["weighted_mjd_mean"] = 0.0
        features["time_weighted_std"] = 0.0

    # Per-band features and colors.
    band_means = {}
    band_peaks = {}

    for band in BANDS:
        sub = lc[lc["band"] == band]
        prefix_variants = [f"{band}", f"band_{band}"]

        stats = _stat_values(sub["flux"].to_numpy(dtype=float) if not sub.empty else np.array([]))
        band_means[band] = stats["mean"]
        band_peaks[band] = stats["max"]

        for prefix in prefix_variants:
            features[f"{prefix}_n_obs"] = float(len(sub))
            features[f"{prefix}_count"] = float(len(sub))
            features[f"{prefix}_frac_obs"] = float(len(sub) / max(len(lc), 1))

            for key, value in stats.items():
                features[f"{prefix}_flux_{key}"] = value
                features[f"{prefix}_{key}"] = value

            if not sub.empty:
                features[f"{prefix}_mjd_min"] = float(sub["mjd"].min())
                features[f"{prefix}_mjd_max"] = float(sub["mjd"].max())
                features[f"{prefix}_mjd_span"] = float(sub["mjd"].max() - sub["mjd"].min())
                features[f"{prefix}_slope"] = _linear_slope(sub["mjd"].to_numpy(), sub["flux"].to_numpy())
            else:
                features[f"{prefix}_mjd_min"] = 0.0
                features[f"{prefix}_mjd_max"] = 0.0
                features[f"{prefix}_mjd_span"] = 0.0
                features[f"{prefix}_slope"] = 0.0

    for b1 in BANDS:
        for b2 in BANDS:
            if b1 == b2:
                continue
            features[f"color_{b1}_{b2}_mean"] = band_means[b1] - band_means[b2]
            features[f"color_{b1}_{b2}_peak"] = band_peaks[b1] - band_peaks[b2]
            features[f"{b1}_{b2}_mean_diff"] = band_means[b1] - band_means[b2]
            features[f"{b1}_{b2}_peak_diff"] = band_peaks[b1] - band_peaks[b2]

    # HEAD/context metadata when available.
    if head_row is not None:
        if isinstance(head_row, pd.Series):
            head_dict = head_row.to_dict()
        elif isinstance(head_row, dict):
            head_dict = head_row
        else:
            head_dict = {}

        for key, value in head_dict.items():
            key_str = str(key)
            try:
                value_num = float(value)
                if np.isfinite(value_num):
                    features[key_str] = value_num
                    features[key_str.lower()] = value_num
            except Exception:
                continue

    return features


def _candidate_keys_for_column(column: str) -> list[str]:
    c = str(column)
    cl = c.lower()

    keys = [c, cl]

    # Common aliases.
    replacements = {
        "psflux": "flux",
        "ps_flux": "flux",
        "fluxcal": "flux",
        "fluxcalerr": "flux_err",
        "mwebv": "mwebv",
        "redshift": "redshift",
        "z_final": "redshift",
        "hostgal_photoz": "hostgal_photoz",
        "hostgal_specz": "hostgal_specz",
    }

    for src, dst in replacements.items():
        if src in cl:
            keys.append(cl.replace(src, dst))

    # Remove common feature prefixes.
    for prefix in ["feature_", "v4_", "lc_", "shape_", "temporal_", "context_", "ctx_"]:
        if cl.startswith(prefix):
            keys.append(cl[len(prefix):])

    # Normalize separators.
    keys.append(cl.replace("__", "_"))
    keys.append(cl.replace("-", "_"))

    # If column contains band and statistic, generate possible names.
    for band in BANDS:
        bl = band.lower()
        if re.search(rf"(^|_){bl}($|_)", cl.lower()) or f"band_{bl}" in cl.lower():
            for stat in ["mean", "std", "min", "max", "median", "amplitude", "slope", "n_obs", "count"]:
                if stat in cl:
                    keys.extend([
                        f"{band}_{stat}",
                        f"{band}_flux_{stat}",
                        f"band_{band}_{stat}",
                        f"band_{band}_flux_{stat}",
                    ])

    # Global stats.
    for stat in ["mean", "std", "min", "max", "median", "amplitude", "skew", "kurtosis", "mad"]:
        if stat in cl and "flux" in cl:
            keys.append(f"flux_{stat}")
            keys.append(stat)

    # De-duplicate preserving order.
    out = []
    seen = set()
    for key in keys:
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def classify_feature_origin(feature_name: str) -> str:
        name = str(feature_name).lower()

        if any(k in name for k in ["host", "gal", "redshift", "photoz", "specz", "z_final"]):
            return "context_host_redshift"

        if any(k in name for k in ["ra", "dec", "coord", "galactic", "lat", "lon", "mwebv"]):
            return "context_position_extinction"

        if any(k in name for k in ["flux", "snr", "mag", "amplitude", "peak", "median", "mean", "std", "skew", "kurt"]):
            return "lightcurve_statistical"

        if any(k in name for k in ["rise", "decline", "slope", "duration", "span", "time", "mjd"]):
            return "lightcurve_temporal_shape"

        if any(k in name for k in ["band", "color", "_u_", "_g_", "_r_", "_i_", "_z_", "_y_"]):
            return "lightcurve_multiband"

        if any(k in name for k in ["period", "fft", "lomb", "frequency"]):
            return "periodicity"

        return "unknown"



def build_v4_feature_row_auto(
    lc_obj: pd.DataFrame,
    head_row: pd.Series | dict | None = None,
    expected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a one-row tabular feature DataFrame in the model's expected column order.

    This is an automatic interface-oriented feature builder. It fills every expected
    column, using computed light-curve statistics when names can be matched and 0.0
    otherwise.
    """
    if expected_columns is None:
        expected_columns = load_expected_tabular_columns()

    pool = _compute_lightcurve_feature_pool(lc_obj, head_row=head_row)

    values = {}
    matched = []
    missing = []

    for col in expected_columns:
        value = None
        for key in _candidate_keys_for_column(col):
            if key in pool:
                value = pool[key]
                break

        if value is None:
            value = 0.0
            missing.append(col)
        else:
            matched.append(col)

        try:
            values[col] = float(value)
        except Exception:
            values[col] = 0.0

    df = pd.DataFrame([values], columns=expected_columns)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_coverage_rows = []

    for col in expected_columns:
        status = "matched" if col in matched else "missing_filled_zero"

        feature_coverage_rows.append(
            {
                "feature_name": col,
                "status": status,
                "category": classify_feature_origin(col),
            }
        )

    feature_coverage = pd.DataFrame(feature_coverage_rows)

    
    report = {
        "n_expected_features": len(expected_columns),
        "n_matched_features": len(matched),
        "n_missing_filled_zero": len(missing),
        "matched_features": matched,
        "missing_features": missing,
        "feature_coverage": feature_coverage,
        "missing_by_category": (
            feature_coverage[feature_coverage["status"] == "missing_filled_zero"]
            ["category"]
            .value_counts()
            .to_dict()
        ),
        "matched_by_category": (
            feature_coverage[feature_coverage["status"] == "matched"]
            ["category"]
            .value_counts()
            .to_dict()
        ),
        "mode": "auto_lightcurve_v4_approximation",
        "warning": (
            "Automatically generated from available light-curve/HEAD information. "
            "Features requiring external host/context metadata are filled with zero if unavailable."
        ),
    }
    return df, report
