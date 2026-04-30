from pathlib import Path
import sys
import json
import os
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "data" / "external" / "ztfcosmo_dr2_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Public example target used in the official ztfcosmo README.
# You can replace this with another ZTF SN Ia DR2 target later.
TARGET_NAME = "ZTF18aaqfziz"


# -----------------------------------------------------------------------------
# Compatibility fix for ztfcosmo remote access on Windows
# -----------------------------------------------------------------------------

def patch_ztfcosmo_remote_url_join():
    """
    Patch ztfcosmo URL construction when using remote data access on Windows.

    ztfcosmo internally builds remote paths with os.path.join(). On Windows,
    os.path.join() may insert backslashes into URLs, e.g.:

        https://ztfcosmo.in2p3.fr/download\\lightcurves__ZTF18aaqfziz_lc.csv

    The remote server expects POSIX-style URLs:

        https://ztfcosmo.in2p3.fr/download/lightcurves__ZTF18aaqfziz_lc.csv

    This patch keeps local access unchanged and only fixes the remote URL case.
    """
    try:
        import ztfcosmo.io as ztfcosmo_io
    except Exception:
        return False

    source_url = getattr(ztfcosmo_io, "SOURCE_URL", "https://ztfcosmo.in2p3.fr/download")
    old_func = getattr(ztfcosmo_io, "_ztfdr2name_to_fullpath_", None)

    if old_func is None:
        return False

    # Avoid patching more than once.
    if getattr(ztfcosmo_io, "_astrotrust_url_patch_applied", False):
        return True

    def fixed_ztfdr2name_to_fullpath(ztfdr2name, force_online=False, directory=None):
        dirname = ztfcosmo_io.get_ztfcosmodir(
            directory=directory,
            force_online=force_online,
        )

        dirname_str = str(dirname).rstrip("/\\")
        source_str = str(source_url).rstrip("/\\")

        # Remote mode: build a valid HTTP URL with forward slashes.
        if dirname_str == source_str:
            safe_name = str(ztfdr2name).replace("\\", "/").replace("/", "__")
            return f"{source_str}/{safe_name}"

        # Local mode: preserve the original behavior.
        return old_func(ztfdr2name, force_online=force_online, directory=directory)

    ztfcosmo_io._ztfdr2name_to_fullpath_ = fixed_ztfdr2name_to_fullpath
    ztfcosmo_io._astrotrust_url_patch_applied = True
    return True


def to_dataframe(obj):
    """Best-effort conversion of ztfcosmo return objects to pandas DataFrame."""
    if obj is None:
        return pd.DataFrame()

    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, pd.Series):
        return obj.to_frame().T

    # Astropy Table-like
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas()
        except Exception:
            pass

    # Some ztfcosmo objects expose a .data attribute.
    if hasattr(obj, "data"):
        try:
            return to_dataframe(obj.data)
        except Exception:
            pass

    # Some light-curve objects may expose photometry/table aliases.
    for attr in ["lightcurve", "photometry", "table", "lc", "df"]:
        if hasattr(obj, attr):
            try:
                return to_dataframe(getattr(obj, attr))
            except Exception:
                pass

    # Dict-like.
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame([obj])

    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()


def find_col(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower_map:
            return lower_map[str(c).lower()]
    return None


def normalize_band(value):
    value = str(value).strip()
    aliases = {
        "ztfg": "g", "ztfr": "r", "ztfi": "i",
        "zg": "g", "zr": "r", "zi": "i",
        "g": "g", "r": "r", "i": "i",
        "ZTF_g": "g", "ZTF_r": "r", "ZTF_i": "i",
    }
    return aliases.get(value, aliases.get(value.lower(), value))


def read_ztfcosmo_lightcurve_direct_remote(target_name):
    """Direct remote fallback using the URL pattern used by ztfcosmo."""
    url = f"https://ztfcosmo.in2p3.fr/download/lightcurves__{target_name}_lc.csv"
    print(f"Trying direct fixed remote URL: {url}")
    return pd.read_csv(url, sep=r"\s+", comment="#")


def read_ztfcosmo_lightcurve_local(target_name):
    """
    Try to read the light curve from local ZTFCOSMODIR, if configured.
    """
    ztfcosmo_dir = os.getenv("ZTFCOSMODIR")
    if not ztfcosmo_dir:
        return pd.DataFrame()

    base = Path(ztfcosmo_dir)
    candidates = [
        base / "lightcurves" / f"{target_name}_lc.csv",
        base / "lightcurves" / f"{target_name}.csv",
        base / f"lightcurves__{target_name}_lc.csv",
        base / f"{target_name}_lc.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"Loading local ZTFCosmo light curve: {path}")
            return pd.read_csv(path, sep=r"\s+", comment="#")

    print(f"ZTFCOSMODIR is set, but no local light curve was found for {target_name}: {base}")
    return pd.DataFrame()


def load_ztfcosmo_lightcurve_safe(ztfcosmo, target_name):
    """
    Robust loader for ZTF SN Ia DR2 light curves.

    Priority:
    1. Local ZTFCOSMODIR, when configured.
    2. ztfcosmo package call after applying Windows URL patch.
    3. Direct fixed remote URL.
    4. ztfcosmo object mode as final attempt.
    """
    errors = []

    # 1) Local data first, if available.
    try:
        local_df = read_ztfcosmo_lightcurve_local(target_name)
        if not local_df.empty:
            return local_df
    except Exception as exc:
        errors.append(f"local read failed: {repr(exc)}")

    # 2) Patch remote URL construction and use the official ztfcosmo function.
    try:
        patched = patch_ztfcosmo_remote_url_join()
        if patched:
            print("Applied ztfcosmo remote URL patch for Windows-safe access.")

        print("Loading light curve with ztfcosmo.get_target_lightcurve(..., as_data=True)...")
        lc_obj = ztfcosmo.get_target_lightcurve(target_name, as_data=True)
        lc_df = to_dataframe(lc_obj)
        if not lc_df.empty:
            return lc_df
        errors.append("ztfcosmo as_data=True returned an empty DataFrame")

    except (HTTPError, URLError, FileNotFoundError, OSError, Exception) as exc:
        errors.append(f"ztfcosmo as_data=True failed: {repr(exc)}")

    # 3) Direct remote fallback with the corrected URL.
    try:
        lc_df = read_ztfcosmo_lightcurve_direct_remote(target_name)
        if not lc_df.empty:
            return lc_df
        errors.append("direct remote URL returned an empty DataFrame")
    except Exception as exc:
        errors.append(f"direct remote URL failed: {repr(exc)}")

    # 4) Final attempt: object mode.
    try:
        print("Trying ztfcosmo.get_target_lightcurve(..., as_data=False)...")
        lc_obj = ztfcosmo.get_target_lightcurve(target_name, as_data=False)
        lc_df = to_dataframe(lc_obj)
        if not lc_df.empty:
            return lc_df
        errors.append("ztfcosmo as_data=False returned an empty DataFrame")
    except Exception as exc:
        errors.append(f"ztfcosmo as_data=False failed: {repr(exc)}")

    raise RuntimeError(
        "Could not load the ZTF SN Ia DR2 light curve.\n\n"
        f"Target: {target_name}\n"
        "Attempts made:\n- " + "\n- ".join(errors) + "\n\n"
        "Recommended fix: download the ZTF SN Ia DR2 data locally, extract it, "
        "and set ZTFCOSMODIR to the extracted 'ztfsniadr2' directory."
    )


def standardize_lightcurve(lc_df, target_name):
    """Convert ztfcosmo light-curve columns to AstroTrust generic schema."""
    if lc_df.empty:
        raise ValueError("The ztfcosmo light-curve table is empty or could not be converted to a DataFrame.")

    mjd_col = find_col(lc_df, ["mjd", "time", "jd", "hjd", "bjd"])
    band_col = find_col(lc_df, ["band", "filter", "filtercode", "passband", "fid"])
    mag_col = find_col(lc_df, ["mag", "magnitude", "magpsf", "psfmag"])
    magerr_col = find_col(lc_df, ["magerr", "mag_err", "mag_error", "sigmapsf", "e_mag"])
    flux_col = find_col(lc_df, ["flux", "fluxcal", "flx"])
    fluxerr_col = find_col(lc_df, ["flux_err", "fluxerr", "flux_error", "fluxcalerr"])

    missing = []
    if mjd_col is None:
        missing.append("mjd/time")
    if band_col is None:
        missing.append("band/filter")
    if mag_col is None and flux_col is None:
        missing.append("mag or flux")

    if missing:
        raise ValueError(
            "Could not standardize light curve. Missing: "
            + ", ".join(missing)
            + f". Available columns: {list(lc_df.columns)}"
        )

    out = pd.DataFrame()
    out["object_id"] = target_name
    out["mjd"] = pd.to_numeric(lc_df[mjd_col], errors="coerce")
    out["band"] = lc_df[band_col].map(normalize_band)

    if mag_col is not None:
        out["mag"] = pd.to_numeric(lc_df[mag_col], errors="coerce")
        if magerr_col is not None:
            out["mag_err"] = pd.to_numeric(lc_df[magerr_col], errors="coerce")
        else:
            out["mag_err"] = np.nan
    else:
        out["flux"] = pd.to_numeric(lc_df[flux_col], errors="coerce")
        if fluxerr_col is not None:
            out["flux_err"] = pd.to_numeric(lc_df[fluxerr_col], errors="coerce")
        else:
            out["flux_err"] = np.nan

    # Preserve useful optional context-like columns if present.
    for col in ["ra", "dec", "redshift", "z", "zhel", "z_cmb", "mwebv"]:
        real_col = find_col(lc_df, [col])
        if real_col is not None and real_col not in out.columns:
            out[col] = pd.to_numeric(lc_df[real_col], errors="coerce")

    out = out.dropna(subset=["mjd"]).reset_index(drop=True)
    return out


def extract_context_from_global_table(data_df, target_name):
    """Try to extract one host/context row from ztfcosmo.get_data()."""
    if data_df.empty:
        return pd.DataFrame()

    # ztfcosmo.get_data() usually uses ztfname as index, so first try the index.
    index_matches = data_df.index.astype(str).str.strip() == str(target_name).strip()
    if index_matches.any():
        row = data_df.loc[index_matches].iloc[[0]].copy()
        row.insert(0, "object_id", target_name)
        return normalize_context_columns(row)

    name_col = find_col(data_df, ["name", "ztfname", "ztf_name", "target", "target_name", "object_id", "sn_name"])

    if name_col is None:
        return pd.DataFrame()

    matches = data_df[data_df[name_col].astype(str).str.strip() == str(target_name).strip()].copy()
    if matches.empty:
        return pd.DataFrame()

    row = matches.iloc[[0]].copy()
    row = row.rename(columns={name_col: "object_id"})
    return normalize_context_columns(row)


def normalize_context_columns(row):
    """Rename common redshift/context columns toward AstroTrust-style names."""
    rename = {}
    for col in row.columns:
        cl = str(col).lower()
        if cl in ["z", "redshift", "zhel", "zcmb", "z_cmb"]:
            rename[col] = "hostgal_zphot"
        elif cl in ["z_err", "redshift_err", "zhel_err", "zcmb_err", "z_cmb_err"]:
            rename[col] = "hostgal_zphot_err"
        elif cl in ["mwebv", "mw_ebv", "ebv"]:
            rename[col] = "mwebv"
        elif cl == "ra":
            rename[col] = "ra"
        elif cl in ["dec", "decl"]:
            rename[col] = "dec"

    return row.rename(columns=rename)


def main():
    try:
        import ztfcosmo
    except Exception as exc:
        raise SystemExit(
            "Could not import ztfcosmo. Install it with:\n"
            "python -m pip install ztfcosmo\n\n"
            f"Original error: {exc}"
        )

    print(f"Preparing ZTF SN Ia DR2 test target: {TARGET_NAME}")

    lc_raw = load_ztfcosmo_lightcurve_safe(ztfcosmo, TARGET_NAME)

    print("Raw light-curve columns:")
    print(list(lc_raw.columns))
    print(lc_raw.head())

    lc_std = standardize_lightcurve(lc_raw, TARGET_NAME)

    lc_out = OUT_DIR / f"{TARGET_NAME}_lightcurve.csv"
    lc_raw_out = OUT_DIR / f"{TARGET_NAME}_lightcurve_raw.csv"
    lc_std.to_csv(lc_out, index=False)
    lc_raw.to_csv(lc_raw_out, index=False)

    print(f"[OK] Saved standardized light curve: {lc_out}")
    print(f"[OK] Saved raw light curve:          {lc_raw_out}")
    print(lc_std.head())

    print("\nTrying to load global DR2 table for context metadata...")
    context_out = OUT_DIR / f"{TARGET_NAME}_host_context.csv"
    global_data_out = OUT_DIR / "ztfcosmo_global_data_preview.csv"

    try:
        # Keep the URL patch active for get_data() too.
        patch_ztfcosmo_remote_url_join()

        data = ztfcosmo.get_data()
        data_df = to_dataframe(data)
        print("Global data columns:")
        print(list(data_df.columns))
        data_df.head(50).to_csv(global_data_out, index=False)
        print(f"[OK] Saved global data preview: {global_data_out}")

        context_df = extract_context_from_global_table(data_df, TARGET_NAME)
        if not context_df.empty:
            context_df.to_csv(context_out, index=False)
            print(f"[OK] Saved context metadata: {context_out}")
            print(context_df.T.head(80))
        else:
            print("[WARN] Could not match a context row in ztfcosmo.get_data().")
            pd.DataFrame([{"object_id": TARGET_NAME}]).to_csv(context_out, index=False)
            print(f"[OK] Saved minimal context placeholder: {context_out}")

    except Exception as exc:
        print(f"[WARN] Could not retrieve global context table: {exc}")
        pd.DataFrame([{"object_id": TARGET_NAME}]).to_csv(context_out, index=False)
        print(f"[OK] Saved minimal context placeholder: {context_out}")

    summary = {
        "target_name": TARGET_NAME,
        "lightcurve_csv": str(lc_out),
        "raw_lightcurve_csv": str(lc_raw_out),
        "context_csv": str(context_out),
        "n_rows": int(len(lc_std)),
        "n_bands": int(lc_std["band"].nunique()) if "band" in lc_std.columns else None,
        "bands": sorted(lc_std["band"].astype(str).unique().tolist()) if "band" in lc_std.columns else [],
    }

    summary_out = OUT_DIR / f"{TARGET_NAME}_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved summary: {summary_out}")

    print("\nNext test in AstroTrust-AI interface:")
    print("1. Upload Alert / Light Curve -> Generic light-curve table -> Browser upload")
    print(f"2. Upload: {lc_out}")
    print("3. Open Optional host/context metadata")
    print(f"4. Upload: {context_out}")
    print("5. Predict with AstroTrust-AI")


if __name__ == "__main__":
    main()
