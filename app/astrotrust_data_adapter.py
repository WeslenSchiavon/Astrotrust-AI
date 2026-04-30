from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except Exception:
    fits = None
    HAS_ASTROPY = False


BAND_ALIASES = {
    "0": "u", "1": "g", "2": "r", "3": "i", "4": "z", "5": "Y",
    0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "Y",
    "u": "u", "g": "g", "r": "r", "i": "i", "z": "z", "y": "Y", "Y": "Y",
    "U": "u", "G": "g", "R": "r", "I": "i", "Z": "z",
    "zg": "g", "zr": "r", "zi": "i",
    "ztfg": "g", "ztfr": "r", "ztfi": "i",
    "ZTF_g": "g", "ZTF_r": "r", "ZTF_i": "i",
    "g_ZTF": "g", "r_ZTF": "r", "i_ZTF": "i",
    "sdssu": "u", "sdssg": "g", "sdssr": "r", "sdssi": "i", "sdssz": "z",
    "B": "B", "V": "V", "Rc": "r", "Ic": "i",
}

FITS_SUFFIXES = {".fits", ".fit", ".fts"}
TABLE_SUFFIXES = {".csv", ".parquet", ".fits", ".fit", ".fts"}


@dataclass
class LightCurveAdapterReport:
    source_type: str
    source_domain: str
    n_raw_rows: int
    n_raw_columns: int
    n_output_rows: int
    n_objects: int
    column_mapping: dict[str, str | None]
    issues: list[str]
    warnings: list[str]
    used_magnitude_conversion: bool
    selected_table_hdu: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_band(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="ignore").strip()

    if value in BAND_ALIASES:
        return BAND_ALIASES[value]

    value_str = str(value).strip()

    if value_str.startswith("b'") and value_str.endswith("'"):
        value_str = value_str[2:-1]
    if value_str.startswith('b"') and value_str.endswith('"'):
        value_str = value_str[2:-1]

    if value_str in BAND_ALIASES:
        return BAND_ALIASES[value_str]

    if value_str.lower() in BAND_ALIASES:
        return BAND_ALIASES[value_str.lower()]

    if value_str.lower() == "y":
        return "Y"

    return value_str


def safe_fits_table_to_dataframe(data, max_rows: int | None = None) -> pd.DataFrame:
    """Convert FITS table data to pandas with native byte order.

    Compatible with NumPy 2.x, where ndarray.newbyteorder() was removed.
    """
    arr = np.array(data if max_rows is None else data[:max_rows])

    try:
        if not arr.dtype.isnative:
            arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    except Exception:
        pass

    return pd.DataFrame(arr)


def _select_fits_table_hdu(hdul, preferred_names: list[str] | None = None):
    preferred = {str(v).upper() for v in (preferred_names or [])}
    table_hdus = []

    for idx, hdu in enumerate(hdul):
        if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "columns"):
            table_hdus.append((idx, hdu))

    if not table_hdus:
        raise ValueError(
            "No FITS table HDU was found. Image/datacube FITS files can be inspected, "
            "but they are not direct generic light-curve inputs."
        )

    selected_idx, selected_hdu = table_hdus[0]

    for idx, hdu in table_hdus:
        if str(hdu.name).upper() in preferred:
            selected_idx, selected_hdu = idx, hdu
            break

    return selected_idx, selected_hdu


def read_table_from_path(path: str | Path) -> tuple[pd.DataFrame, str, str | None]:
    path = Path(str(path).strip().strip('"')).expanduser()

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path), "csv_local", None

    if suffix == ".parquet":
        return pd.read_parquet(path), "parquet_local", None

    if suffix in FITS_SUFFIXES:
        if not HAS_ASTROPY:
            raise RuntimeError("Astropy is required to read FITS files. Install with: pip install astropy")

        with fits.open(path, memmap=True) as hdul:
            idx, hdu = _select_fits_table_hdu(
                hdul,
                preferred_names=["PHOT", "PHOTOMETRY", "LIGHTCURVE", "LC", "TABLE"],
            )
            df = safe_fits_table_to_dataframe(hdu.data)
            return df, f"fits_table_hdu_{idx}_{hdu.name}_local", f"hdu_{idx}_{hdu.name}"

    raise ValueError(f"Unsupported file type: {suffix}")


def read_table_from_uploaded(uploaded_file) -> tuple[pd.DataFrame, str, str | None]:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(uploaded_file), "csv_upload", None

    if suffix == ".parquet":
        return pd.read_parquet(uploaded_file), "parquet_upload", None

    if suffix in FITS_SUFFIXES:
        if not HAS_ASTROPY:
            raise RuntimeError("Astropy is required to read FITS files. Install with: pip install astropy")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        try:
            return read_table_from_path(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    raise ValueError(f"Unsupported uploaded file type: {suffix}")


def read_table_from_url(url: str) -> tuple[pd.DataFrame, str, str | None]:
    url = str(url).strip()

    if not url:
        raise ValueError("Empty URL.")

    lower_url = url.lower()

    if lower_url.endswith(".parquet"):
        return pd.read_parquet(url), "parquet_url", None

    # Most public astronomy light-curve APIs return CSV/VOTable-like text.
    # Our current generic adapter supports CSV directly; FORMAT=CSV is recommended.
    try:
        return pd.read_csv(url), "csv_url", None
    except Exception as exc:
        raise ValueError(
            "Could not read URL as CSV. Make sure the URL points to a public CSV table "
            "or an API response with FORMAT=CSV."
        ) from exc


def infer_lightcurve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    lower_map = {str(c).lower(): c for c in df.columns}

    candidates = {
        "object_id": [
            "object_id", "objectid", "objid", "diaobjectid", "diaobjectId",
            "snid", "id", "source_id", "oid", "name", "target_name",
        ],
        "mjd": [
            "mjd", "mjdobs", "time", "t", "jd", "hjd", "bjd",
            "date", "obsjd", "obsmjd",
        ],
        "band": [
            "band", "filter", "passband", "fid", "flt", "filtercode",
            "filter_code", "filtername", "bandname",
        ],
        "flux": [
            "flux", "fluxcal", "flx", "psflux", "forcediffimflux",
            "flux_jy", "fnu", "flux_density",
        ],
        "flux_err": [
            "flux_err", "fluxerr", "flux_error", "fluxcalerr", "psfluxerr",
            "forcediffimfluxunc", "flux_unc", "flux_uncertainty", "fluxerr_jy",
        ],
        "mag": [
            "mag", "magnitude", "magpsf", "psfmag", "mag_auto", "magap",
            "mag_calibrated", "magnitude_calibrated",
        ],
        "mag_err": [
            "magerr", "mag_err", "mag_error", "sigmapsf", "e_mag",
            "mag_unc", "magnitude_error", "uncertainty",
        ],
    }

    inferred = {}

    for target, names in candidates.items():
        inferred[target] = None

        for name in names:
            if str(name).lower() in lower_map:
                inferred[target] = lower_map[str(name).lower()]
                break

    return inferred


def convert_magnitude_to_relative_flux(mag, mag_err=None):
    """Convert magnitude to relative flux.

    The zero point is the median magnitude of the uploaded table, which keeps the
    scale stable when no physical zero point is available.
    """
    mag = pd.to_numeric(mag, errors="coerce")
    finite_mag = mag[np.isfinite(mag)]

    if len(finite_mag) == 0:
        flux = pd.Series(np.nan, index=mag.index)
        flux_err = pd.Series(np.nan, index=mag.index)
        return flux, flux_err

    mag0 = float(np.nanmedian(finite_mag))
    flux = 10.0 ** (-0.4 * (mag - mag0))

    if mag_err is None:
        flux_err = pd.Series(np.nan, index=mag.index)
    else:
        mag_err = pd.to_numeric(mag_err, errors="coerce")
        flux_err = flux * 0.4 * np.log(10.0) * mag_err

    return flux, flux_err


def detect_source_domain(df: pd.DataFrame, source_type: str, inferred: dict[str, str | None]) -> str:
    cols = {str(c).lower() for c in df.columns}

    if {"oid", "filtercode", "mag", "magerr"}.issubset(cols):
        return "ztf_irsa"

    if inferred.get("mag") is not None and inferred.get("band") is not None:
        return "generic_magnitude"

    if "fluxcal" in cols or "fluxcalerr" in cols:
        return "snana_like_table"

    if source_type.startswith("fits"):
        return "fits_table"

    return "generic"


def preserve_numeric_context_columns(
    raw_df: pd.DataFrame,
    out_df: pd.DataFrame,
    inferred: dict[str, str | None],
) -> pd.DataFrame:
    used_columns = {
        inferred.get("object_id"),
        inferred.get("mjd"),
        inferred.get("band"),
        inferred.get("flux"),
        inferred.get("flux_err"),
        inferred.get("mag"),
        inferred.get("mag_err"),
    }
    used_columns = {c for c in used_columns if c is not None}

    keep_exact = {
        "ra", "dec", "decl", "catflags", "clrcoeff", "field", "ccdid", "qid",
        "airmass", "seeing", "fwhm", "chi", "sharp", "mwebv", "mwebv_err",
    }

    for col in raw_df.columns:
        if col in used_columns:
            continue

        col_lower = str(col).lower()
        keep = (
            col_lower in keep_exact
            or "flag" in col_lower
            or "quality" in col_lower
            or col_lower.startswith("ra")
            or col_lower.startswith("dec")
        )

        if not keep:
            continue

        numeric_col = pd.to_numeric(raw_df[col], errors="coerce")

        if numeric_col.notna().any():
            out_df[col_lower] = numeric_col

    return out_df


def normalize_lightcurve_table(
    raw_df: pd.DataFrame,
    source_type: str = "unknown",
    selected_table_hdu: str | None = None,
) -> tuple[pd.DataFrame, LightCurveAdapterReport]:
    inferred = infer_lightcurve_columns(raw_df)
    issues: list[str] = []
    warnings: list[str] = []

    has_flux = inferred.get("flux") is not None
    has_mag = inferred.get("mag") is not None
    used_magnitude_conversion = False

    missing = []

    if inferred.get("mjd") is None:
        missing.append("mjd/time")
    if inferred.get("band") is None:
        missing.append("band/filter")
    if not has_flux and not has_mag:
        missing.append("flux or mag")

    if missing:
        issues.append(f"Missing required light-curve columns: {', '.join(missing)}")
        report = LightCurveAdapterReport(
            source_type=source_type,
            source_domain=detect_source_domain(raw_df, source_type, inferred),
            n_raw_rows=len(raw_df),
            n_raw_columns=len(raw_df.columns),
            n_output_rows=0,
            n_objects=0,
            column_mapping=inferred,
            issues=issues,
            warnings=warnings,
            used_magnitude_conversion=False,
            selected_table_hdu=selected_table_hdu,
        )
        return pd.DataFrame(), report

    out = pd.DataFrame(index=raw_df.index)

    if inferred.get("object_id") is not None:
        obj = raw_df[inferred["object_id"]]

        if obj.notna().any():
            out["object_id"] = obj.astype(str)
        else:
            out["object_id"] = "single_object"
            warnings.append("object_id column exists but is empty. Treating the file as a single object.")
    else:
        out["object_id"] = "single_object"
        warnings.append("No object_id column found; treating the file as a single object.")

    out["mjd"] = pd.to_numeric(raw_df[inferred["mjd"]], errors="coerce")
    out["band"] = raw_df[inferred["band"]].map(normalize_band)

    if has_flux:
        out["flux"] = pd.to_numeric(raw_df[inferred["flux"]], errors="coerce")

        if inferred.get("flux_err") is not None:
            out["flux_err"] = pd.to_numeric(raw_df[inferred["flux_err"]], errors="coerce")
        else:
            out["flux_err"] = np.nan
            warnings.append(
                "No flux uncertainty column found; plotting will work, but SNR/tensor construction will be incomplete."
            )

    else:
        mag = pd.to_numeric(raw_df[inferred["mag"]], errors="coerce")
        mag_err = (
            pd.to_numeric(raw_df[inferred["mag_err"]], errors="coerce")
            if inferred.get("mag_err") is not None
            else None
        )
        flux, flux_err = convert_magnitude_to_relative_flux(mag, mag_err)
        out["flux"] = flux
        out["flux_err"] = flux_err
        out["mag"] = mag

        # Common real-survey placeholder for invalid/non-detection magnitudes.
        invalid_mag = (
            (~np.isfinite(out["mag"]))
            | (out["mag"] >= 90)
        )

        if "mag_err" in out.columns:
            invalid_mag = invalid_mag | (~np.isfinite(out["mag_err"])) | (out["mag_err"] >= 90)

        if invalid_mag.any():
            warnings.append(
                f"Removed {int(invalid_mag.sum())} rows with invalid magnitude placeholders, e.g. mag/mag_err >= 90."
            )
            out = out[~invalid_mag].copy()

        if mag_err is not None:
            out["mag_err"] = mag_err

        used_magnitude_conversion = True
        warnings.append("Input used magnitude columns. Converted mag/magerr to relative flux/flux_err automatically.")

    out = preserve_numeric_context_columns(raw_df, out, inferred)

    if "flux_err" in out.columns:
        out["flux_err"] = pd.to_numeric(out["flux_err"], errors="coerce")
        out.loc[out["flux_err"] <= 0, "flux_err"] = np.nan

    before = len(out)
    out = out.dropna(subset=["mjd", "flux"]).reset_index(drop=True)
    dropped = before - len(out)

    if dropped > 0:
        warnings.append(f"Dropped {dropped} rows with invalid mjd/flux values.")

    source_domain = detect_source_domain(raw_df, source_type, inferred)

    report = LightCurveAdapterReport(
        source_type=source_type,
        source_domain=source_domain,
        n_raw_rows=len(raw_df),
        n_raw_columns=len(raw_df.columns),
        n_output_rows=len(out),
        n_objects=int(out["object_id"].nunique()) if not out.empty else 0,
        column_mapping=inferred,
        issues=issues,
        warnings=warnings,
        used_magnitude_conversion=used_magnitude_conversion,
        selected_table_hdu=selected_table_hdu,
    )

    return out, report


def summarize_objects(lc: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if lc.empty or "object_id" not in lc.columns:
        return pd.DataFrame(columns=["object_id", "n_obs", "n_bands", "time_span"])

    for object_id, group in lc.groupby("object_id"):
        n_obs = len(group)
        n_bands = group["band"].nunique() if "band" in group.columns else 0

        if "mjd" in group.columns and len(group) > 1:
            time_span = float(group["mjd"].max() - group["mjd"].min())
        else:
            time_span = 0.0

        rows.append(
            {
                "object_id": str(object_id),
                "n_obs": int(n_obs),
                "n_bands": int(n_bands),
                "time_span": time_span,
            }
        )

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    return summary.sort_values(
        ["n_bands", "n_obs", "time_span"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def read_and_normalize_from_path(path: str | Path):
    raw_df, source_type, selected_hdu = read_table_from_path(path)
    lc, report = normalize_lightcurve_table(raw_df, source_type=source_type, selected_table_hdu=selected_hdu)
    return raw_df, lc, report


def read_and_normalize_from_uploaded(uploaded_file):
    raw_df, source_type, selected_hdu = read_table_from_uploaded(uploaded_file)
    lc, report = normalize_lightcurve_table(raw_df, source_type=source_type, selected_table_hdu=selected_hdu)
    return raw_df, lc, report


def read_and_normalize_from_url(url: str):
    raw_df, source_type, selected_hdu = read_table_from_url(url)
    lc, report = normalize_lightcurve_table(raw_df, source_type=source_type, selected_table_hdu=selected_hdu)
    return raw_df, lc, report
