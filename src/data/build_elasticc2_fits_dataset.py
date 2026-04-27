from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
from astropy.table import Table

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_ROOT = ROOT_DIR / Path("data/raw/elasticc2_train_02")
OUTPUT_ROOT = ROOT_DIR / Path("data/processed/elasticc2_large")


SENTINELS = [-9999, -999, -99, -9, 99, 999]


CONTEXT_COLUMN_MAP = {
    "RA": "ra",
    "DEC": "decl",
    "MWEBV": "mwebv",
    "MWEBV_ERR": "mwebv_err",

    "HOSTGAL_ELLIPTICITY": "hostgal_ellipticity",
    "HOSTGAL_SQRADIUS": "hostgal_sqradius",
    "HOSTGAL_PHOTOZ": "hostgal_zphot",
    "HOSTGAL_PHOTOZ_ERR": "hostgal_zphot_err",
    "HOSTGAL_SNSEP": "hostgal_snsep",

    "HOSTGAL_ZPHOT_Q000": "hostgal_zphot_q000",
    "HOSTGAL_ZPHOT_Q010": "hostgal_zphot_q010",
    "HOSTGAL_ZPHOT_Q020": "hostgal_zphot_q020",
    "HOSTGAL_ZPHOT_Q030": "hostgal_zphot_q030",
    "HOSTGAL_ZPHOT_Q040": "hostgal_zphot_q040",
    "HOSTGAL_ZPHOT_Q050": "hostgal_zphot_q050",
    "HOSTGAL_ZPHOT_Q060": "hostgal_zphot_q060",
    "HOSTGAL_ZPHOT_Q070": "hostgal_zphot_q070",
    "HOSTGAL_ZPHOT_Q080": "hostgal_zphot_q080",
    "HOSTGAL_ZPHOT_Q090": "hostgal_zphot_q090",
    "HOSTGAL_ZPHOT_Q100": "hostgal_zphot_q100",

    "HOSTGAL_MAG_u": "hostgal_mag_u",
    "HOSTGAL_MAG_g": "hostgal_mag_g",
    "HOSTGAL_MAG_r": "hostgal_mag_r",
    "HOSTGAL_MAG_i": "hostgal_mag_i",
    "HOSTGAL_MAG_z": "hostgal_mag_z",
    "HOSTGAL_MAG_Y": "hostgal_mag_y",

    "HOSTGAL_MAGERR_u": "hostgal_magerr_u",
    "HOSTGAL_MAGERR_g": "hostgal_magerr_g",
    "HOSTGAL_MAGERR_r": "hostgal_magerr_r",
    "HOSTGAL_MAGERR_i": "hostgal_magerr_i",
    "HOSTGAL_MAGERR_z": "hostgal_magerr_z",
    "HOSTGAL_MAGERR_Y": "hostgal_magerr_y",
}


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return value


def decode_dataframe_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(decode_value)

    return df


def class_name_from_dir(path: Path) -> str:
    name = path.name

    prefix = "ELASTICC2_TRAIN_02_"
    if name.startswith(prefix):
        return name.replace(prefix, "", 1)

    return name


def read_head_file(head_path: Path, class_name: str, label: int) -> pd.DataFrame:
    table = Table.read(head_path)
    df = table.to_pandas()
    df = decode_dataframe_strings(df)

    required = ["SNID", "NOBS", "PTROBS_MIN", "PTROBS_MAX"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{head_path} is missing columns: {missing}")

    df = df[df["NOBS"] > 0].copy()

    df["class_name"] = class_name
    df["label"] = label
    df["head_path"] = str(head_path)
    df["phot_path"] = str(head_path).replace("_HEAD.FITS.gz", "_PHOT.FITS.gz")

    return df


def collect_sample(raw_root: Path, n_objects: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    class_dirs = sorted(
        [p for p in raw_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {raw_root}")

    class_names = [class_name_from_dir(p) for p in class_dirs]
    label_map = {name: i for i, name in enumerate(sorted(class_names))}

    quota_base = n_objects // len(class_dirs)
    remainder = n_objects % len(class_dirs)

    selected_parts = []

    print(f"Classes found: {len(class_dirs)}")
    print(f"Target objects: {n_objects}")

    for idx, class_dir in enumerate(class_dirs):
        class_name = class_name_from_dir(class_dir)
        label = label_map[class_name]

        quota = quota_base + (1 if idx < remainder else 0)
        head_files = sorted(class_dir.glob("*_HEAD.FITS.gz"))

        if not head_files:
            print(f"[WARN] No HEAD files found for {class_name}")
            continue

        class_candidates = []

        print(f"\nCollecting class: {class_name} | quota={quota} | head files={len(head_files)}")

        for head_path in head_files:
            head_df = read_head_file(head_path, class_name, label)
            class_candidates.append(head_df)

            current = sum(len(x) for x in class_candidates)
            if current >= quota:
                break

        if not class_candidates:
            continue

        class_df = pd.concat(class_candidates, ignore_index=True)

        if len(class_df) > quota:
            sampled_idx = rng.choice(class_df.index.to_numpy(), size=quota, replace=False)
            class_df = class_df.loc[sampled_idx].copy()

        selected_parts.append(class_df)

        print(f"Selected {len(class_df)} objects from {class_name}")

    selected = pd.concat(selected_parts, ignore_index=True)

    if len(selected) > n_objects:
        sampled_idx = rng.choice(selected.index.to_numpy(), size=n_objects, replace=False)
        selected = selected.loc[sampled_idx].copy()

    selected = selected.reset_index(drop=True)
    selected["object_id"] = np.arange(1, len(selected) + 1, dtype=np.int64)

    print(f"\nFinal selected objects: {len(selected)}")
    print("Selected class distribution:")
    print(selected["class_name"].value_counts().sort_index())

    return selected


def clean_numeric_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if col in ["object_id", "label"]:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(SENTINELS, np.nan)

    return df


def build_context(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in selected.iterrows():
        item = {
            "object_id": int(row["object_id"]),
            "label": int(row["label"]),
        }

        for source_col, target_col in CONTEXT_COLUMN_MAP.items():
            item[target_col] = row[source_col] if source_col in row.index else np.nan

        rows.append(item)

    context = pd.DataFrame(rows)
    context = clean_numeric_context(context)

    return context


def build_metadata(selected: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "object_id",
        "SNID",
        "class_name",
        "label",
        "SNTYPE",
        "SIM_TYPE_INDEX",
        "SIM_TYPE_NAME",
        "SIM_MODEL_NAME",
        "NOBS",
        "head_path",
        "phot_path",
    ]

    cols = [c for c in cols if c in selected.columns]

    metadata = selected[cols].copy()
    metadata = decode_dataframe_strings(metadata)

    return metadata


def read_phot_file(phot_path: Path) -> pd.DataFrame:
    table = Table.read(phot_path)
    df = table.to_pandas()
    df = decode_dataframe_strings(df)

    required = ["MJD", "BAND", "FLUXCAL", "FLUXCALERR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{phot_path} is missing columns: {missing}")

    return df


def slice_photometry_for_object(phot: pd.DataFrame, pt_min: int, pt_max: int) -> pd.DataFrame:
    """
    SNANA FITS PTROBS_MIN/PTROBS_MAX are commonly 1-based inclusive pointers.
    For Python slicing, start = PTROBS_MIN - 1 and end = PTROBS_MAX.
    """
    start = max(int(pt_min) - 1, 0)
    end = min(int(pt_max), len(phot))

    obj_phot = phot.iloc[start:end].copy()

    if len(obj_phot) == 0:
        # Fallback in case a file uses zero-based indexing.
        start = max(int(pt_min), 0)
        end = min(int(pt_max) + 1, len(phot))
        obj_phot = phot.iloc[start:end].copy()

    return obj_phot


def build_lightcurves(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = selected.groupby("phot_path")

    print(f"\nReading PHOT files needed: {len(grouped)}")

    for phot_path_str, group in grouped:
        phot_path = Path(phot_path_str)

        if not phot_path.exists():
            print(f"[WARN] Missing PHOT file: {phot_path}")
            continue

        print(f"Reading PHOT: {phot_path.name} | objects={len(group)}")

        phot = read_phot_file(phot_path)

        for _, obj in group.iterrows():
            object_id = int(obj["object_id"])
            label = int(obj["label"])

            obj_phot = slice_photometry_for_object(
                phot=phot,
                pt_min=int(obj["PTROBS_MIN"]),
                pt_max=int(obj["PTROBS_MAX"]),
            )

            if len(obj_phot) == 0:
                continue

            for j, p in obj_phot.reset_index(drop=True).iterrows():
                flux = float(p["FLUXCAL"])
                flux_err = float(p["FLUXCALERR"])

                if np.isfinite(flux_err) and not np.isclose(flux_err, 0.0):
                    snr = flux / flux_err
                else:
                    snr = 0.0

                band = decode_value(p["BAND"])

                rows.append({
                    "object_id": object_id,
                    "source_id": int(object_id * 1_000_000 + j),
                    "mjd": float(p["MJD"]),
                    "band": str(band).strip(),
                    "flux": flux,
                    "flux_err": flux_err,
                    "snr": float(snr),
                    "photflag": int(p["PHOTFLAG"]) if "PHOTFLAG" in p.index else 0,
                    "label": label,
                })

    lightcurves = pd.DataFrame(rows)

    if len(lightcurves) == 0:
        raise RuntimeError("No light-curve rows were built.")

    lightcurves = lightcurves.replace([np.inf, -np.inf], np.nan)
    lightcurves = lightcurves.dropna(subset=["mjd", "band", "flux", "flux_err"])

    return lightcurves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-objects", type=int, required=True)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.n_objects}obj"

    selected = collect_sample(
        raw_root=args.raw_root,
        n_objects=args.n_objects,
        random_state=args.random_state,
    )

    context = build_context(selected)
    metadata = build_metadata(selected)
    lightcurves = build_lightcurves(selected)

    class_counts = (
        metadata
        .groupby(["label", "class_name"])
        .size()
        .reset_index(name="n_objects")
        .sort_values(["label", "class_name"])
    )

    output_lightcurves = args.output_root / f"forced_lightcurves_{suffix}.parquet"
    output_context = args.output_root / f"object_context_{suffix}.parquet"
    output_metadata = args.output_root / f"object_metadata_{suffix}.csv"
    output_class_counts = args.output_root / f"class_counts_{suffix}.csv"

    lightcurves.to_parquet(output_lightcurves, index=False)
    context.to_parquet(output_context, index=False)
    metadata.to_csv(output_metadata, index=False)
    class_counts.to_csv(output_class_counts, index=False)

    print("\nSaved:")
    print(f"- {output_lightcurves}")
    print(f"- {output_context}")
    print(f"- {output_metadata}")
    print(f"- {output_class_counts}")

    print("\nLight curves:")
    print(f"Rows: {len(lightcurves)}")
    print(f"Objects: {lightcurves['object_id'].nunique()}")
    print(f"Classes: {metadata['class_name'].nunique()}")

    print("\nClass counts:")
    print(class_counts)


if __name__ == "__main__":
    main()