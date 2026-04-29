from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


HEAD_POINTER_CANDIDATES = [
    ("PTROBS_MIN", "PTROBS_MAX"),
    ("PTRPHOT_MIN", "PTRPHOT_MAX"),
    ("PHOT_MIN", "PHOT_MAX"),
]

OBJECT_ID_CANDIDATES = ["SNID", "object_id", "OBJECT_ID", "DIAOBJECTID", "diaObjectId"]

PHOT_COLUMN_MAP = {
    "mjd": ["MJD", "mjd", "TIME", "time"],
    "band": ["BAND", "band", "FLT", "flt", "FILTER", "filter", "PASSBAND", "passband"],
    "flux": ["FLUXCAL", "fluxcal", "FLUX", "flux", "PSFLUX", "psFlux", "FORCEDIFFIMFLUX", "forcediffimflux"],
    "flux_err": ["FLUXCALERR", "fluxcalerr", "FLUXERR", "fluxerr", "FLUX_ERR", "flux_err", "PSFLUXERR", "psFluxErr", "FORCEDIFFIMFLUXUNC", "forcediffimfluxunc"],
}


def _safe_to_native_table(data) -> pd.DataFrame:
    """Convert FITS table data to a pandas DataFrame with native byte order.

    Compatible with NumPy 2.x.
    """
    arr = np.array(data)

    try:
        if not arr.dtype.isnative:
            arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
    except Exception:
        pass

    return pd.DataFrame(arr)

def _find_column(df: pd.DataFrame, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if str(name).lower() in lower_map:
            return lower_map[str(name).lower()]
    return None


def _clean_object_id(value):
    if isinstance(value, bytes):
        return value.decode(errors="ignore").strip()
    return str(value).strip()


def read_fits_table_from_path(path, preferred_names=None) -> tuple[pd.DataFrame, str]:
    """Read the most appropriate table HDU from a FITS file."""
    if not HAS_ASTROPY:
        raise RuntimeError("Astropy is required to read FITS files. Install with: pip install astropy")

    preferred_names = [str(v).upper() for v in (preferred_names or [])]
    path = Path(str(path).strip().strip('"'))

    if not path.exists():
        raise FileNotFoundError(path)

    with fits.open(path, memmap=True) as hdul:
        table_hdus = []
        for idx, hdu in enumerate(hdul):
            if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "columns"):
                table_hdus.append((idx, hdu))

        if not table_hdus:
            raise ValueError(f"No FITS table HDU found in {path}")

        selected_idx, selected_hdu = table_hdus[0]
        for idx, hdu in table_hdus:
            if str(hdu.name).upper() in preferred_names:
                selected_idx, selected_hdu = idx, hdu
                break

        df = _safe_to_native_table(selected_hdu.data)
        return df, f"hdu_{selected_idx}_{selected_hdu.name}"


def read_fits_table_from_uploaded(uploaded_file, preferred_names=None) -> tuple[pd.DataFrame, str]:
    suffix = Path(uploaded_file.name).suffix.lower() or ".fits"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        return read_fits_table_from_path(tmp_path, preferred_names=preferred_names)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def detect_head_columns(head_df: pd.DataFrame) -> dict:
    object_col = _find_column(head_df, OBJECT_ID_CANDIDATES)

    ptr_min = None
    ptr_max = None
    for min_name, max_name in HEAD_POINTER_CANDIDATES:
        cmin = _find_column(head_df, [min_name])
        cmax = _find_column(head_df, [max_name])
        if cmin is not None and cmax is not None:
            ptr_min = cmin
            ptr_max = cmax
            break

    return {
        "object_id": object_col,
        "ptr_min": ptr_min,
        "ptr_max": ptr_max,
    }


def detect_phot_columns(phot_df: pd.DataFrame) -> dict:
    return {
        role: _find_column(phot_df, candidates)
        for role, candidates in PHOT_COLUMN_MAP.items()
    }


def build_snana_lightcurve_table(head_df: pd.DataFrame, phot_df: pd.DataFrame, max_objects=None) -> tuple[pd.DataFrame, dict]:
    """Build a normalized light-curve table from SNANA/ELAsTiCC HEAD+PHOT tables.

    Expected HEAD columns usually include SNID, PTROBS_MIN, PTROBS_MAX.
    Expected PHOT columns usually include MJD, BAND/FLT, FLUXCAL, FLUXCALERR.
    SNANA pointer columns are conventionally 1-based and inclusive.
    """
    head_cols = detect_head_columns(head_df)
    phot_cols = detect_phot_columns(phot_df)

    missing_head = [k for k in ["object_id", "ptr_min", "ptr_max"] if head_cols.get(k) is None]
    missing_phot = [k for k in ["mjd", "band", "flux"] if phot_cols.get(k) is None]

    if missing_head or missing_phot:
        raise ValueError(
            "Could not build SNANA/ELAsTiCC light curves. "
            f"Missing HEAD roles: {missing_head}; missing PHOT roles: {missing_phot}. "
            f"Detected HEAD: {head_cols}; detected PHOT: {phot_cols}."
        )

    rows = []
    n_head = len(head_df) if max_objects is None else min(len(head_df), int(max_objects))

    for _, obj in head_df.head(n_head).iterrows():
        object_id = _clean_object_id(obj[head_cols["object_id"]])

        try:
            start = int(obj[head_cols["ptr_min"]]) - 1
            end = int(obj[head_cols["ptr_max"]])
        except Exception:
            continue

        if start < 0:
            start = 0
        if end < start:
            continue

        obj_phot = phot_df.iloc[start:end].copy()
        if obj_phot.empty:
            continue

        norm = pd.DataFrame(index=obj_phot.index)
        norm["object_id"] = object_id
        norm["mjd"] = pd.to_numeric(obj_phot[phot_cols["mjd"]], errors="coerce")
        norm["band"] = obj_phot[phot_cols["band"]]
        norm["flux"] = pd.to_numeric(obj_phot[phot_cols["flux"]], errors="coerce")

        if phot_cols.get("flux_err") is not None:
            norm["flux_err"] = pd.to_numeric(obj_phot[phot_cols["flux_err"]], errors="coerce")
        else:
            norm["flux_err"] = np.nan

        norm = norm.dropna(subset=["mjd", "flux"])
        norm = norm.reset_index(drop=True)
        rows.append(norm)

    if not rows:
        raise ValueError("No valid light-curve rows could be reconstructed from HEAD+PHOT.")

    lc = pd.concat(rows, ignore_index=True)

    meta = {
        "head_columns": head_cols,
        "phot_columns": phot_cols,
        "n_head_objects": int(len(head_df)),
        "n_phot_rows": int(len(phot_df)),
        "n_output_objects": int(lc["object_id"].nunique()),
        "n_output_rows": int(len(lc)),
    }

    return lc, meta


def load_snana_pair_from_paths(head_path, phot_path, max_objects=None):
    head_df, head_source = read_fits_table_from_path(head_path, preferred_names=["HEAD", "Header"])
    phot_df, phot_source = read_fits_table_from_path(phot_path, preferred_names=["PHOT", "Photometry", "OBS"])
    lc, meta = build_snana_lightcurve_table(head_df, phot_df, max_objects=max_objects)
    meta["head_source"] = head_source
    meta["phot_source"] = phot_source
    meta["head_path"] = str(head_path)
    meta["phot_path"] = str(phot_path)
    return lc, meta, head_df, phot_df


def load_snana_pair_from_uploads(head_upload, phot_upload, max_objects=None):
    head_df, head_source = read_fits_table_from_uploaded(head_upload, preferred_names=["HEAD", "Header"])
    phot_df, phot_source = read_fits_table_from_uploaded(phot_upload, preferred_names=["PHOT", "Photometry", "OBS"])
    lc, meta = build_snana_lightcurve_table(head_df, phot_df, max_objects=max_objects)
    meta["head_source"] = head_source
    meta["phot_source"] = phot_source
    return lc, meta, head_df, phot_df
