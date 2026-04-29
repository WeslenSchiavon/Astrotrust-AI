from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_ROOT = ROOT_DIR / "data" / "processed" / "elasticc2_large"
OUTPUT_ROOT = ROOT_DIR / "data" / "processed" / "elasticc2_large"

BANDS = ["u", "g", "r", "i", "z", "Y"]
BAND_TO_IDX = {b: i for i, b in enumerate(BANDS)}

NUMERIC_BAND_MAP = {
    "0": "u",
    "1": "g",
    "2": "r",
    "3": "i",
    "4": "z",
    "5": "Y",
    0: "u",
    1: "g",
    2: "r",
    3: "i",
    4: "z",
    5: "Y",
}

EPS = 1e-8


def normalize_band(value):
    if value in NUMERIC_BAND_MAP:
        return NUMERIC_BAND_MAP[value]

    value_str = str(value).strip()

    if value_str in NUMERIC_BAND_MAP:
        return NUMERIC_BAND_MAP[value_str]

    if value_str.lower() == "y":
        return "Y"

    return value_str.lower()


def compute_snr(df: pd.DataFrame) -> np.ndarray:
    if "snr" in df.columns:
        return pd.to_numeric(df["snr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    flux = pd.to_numeric(df["flux"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    for err_col in ["flux_err", "fluxerr", "flux_error"]:
        if err_col in df.columns:
            err = pd.to_numeric(df[err_col], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
            err = np.where(np.isfinite(err) & (np.abs(err) > EPS), err, np.nan)
            snr = flux / err
            snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
            return snr.astype(np.float32)

    return np.zeros(len(df), dtype=np.float32)


def robust_flux_scale(flux: np.ndarray) -> float:
    flux = flux[np.isfinite(flux)]

    if len(flux) == 0:
        return 1.0

    scale = np.nanpercentile(np.abs(flux), 95)

    if not np.isfinite(scale) or scale < EPS:
        scale = np.nanstd(flux)

    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0

    return float(scale)


def object_to_tensor(group: pd.DataFrame, n_bins: int) -> np.ndarray:
    """
    Output shape:
    [18, n_bins]

    channels:
    0:6    normalized flux per band
    6:12   clipped normalized SNR per band
    12:18  observation mask per band
    """
    tensor = np.zeros((18, n_bins), dtype=np.float32)
    counts = np.zeros((6, n_bins), dtype=np.float32)

    group = group.copy()
    group["band_norm"] = group["band"].map(normalize_band)

    mjd = pd.to_numeric(group["mjd"], errors="coerce").to_numpy(dtype=np.float32)
    flux = pd.to_numeric(group["flux"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    snr = compute_snr(group)

    valid = np.isfinite(mjd) & np.isfinite(flux)

    if valid.sum() == 0:
        return tensor

    group = group.loc[valid].copy()
    mjd = mjd[valid]
    flux = flux[valid]
    snr = snr[valid]

    t_min = np.nanmin(mjd)
    t_max = np.nanmax(mjd)

    if not np.isfinite(t_min) or not np.isfinite(t_max) or abs(t_max - t_min) < EPS:
        bin_idx = np.zeros(len(mjd), dtype=int)
    else:
        t_norm = (mjd - t_min) / (t_max - t_min)
        bin_idx = np.floor(t_norm * (n_bins - 1)).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    scale = robust_flux_scale(flux)
    flux_norm = flux / scale
    flux_norm = np.clip(flux_norm, -10.0, 10.0) / 10.0

    snr_norm = np.clip(snr, -20.0, 20.0) / 20.0

    bands = group["band_norm"].to_numpy()

    for i in range(len(group)):
        band = bands[i]

        if band not in BAND_TO_IDX:
            continue

        b = BAND_TO_IDX[band]
        t = bin_idx[i]

        tensor[b, t] += flux_norm[i]
        tensor[6 + b, t] += snr_norm[i]
        tensor[12 + b, t] = 1.0
        counts[b, t] += 1.0

    nonzero = counts > 0

    for b in range(6):
        for t in range(n_bins):
            if nonzero[b, t]:
                tensor[b, t] /= counts[b, t]
                tensor[6 + b, t] /= counts[b, t]

    return tensor


def load_tabular_features(n_objects: int, object_ids: np.ndarray):
    features_path = INPUT_ROOT / f"features_v4_temporal_shape_{n_objects}obj.parquet"

    if not features_path.exists():
        print(f"[WARN] Tabular features not found: {features_path}")
        return None, []

    print(f"Loading tabular v4 features: {features_path}")
    features = pd.read_parquet(features_path)

    features = features.set_index("object_id")
    features = features.loc[object_ids]

    X_tab = features.drop(columns=["label"], errors="ignore")
    tabular_columns = X_tab.columns.tolist()

    X_tab = X_tab.apply(pd.to_numeric, errors="coerce")
    X_tab = X_tab.replace([np.inf, -np.inf], np.nan)
    X_tab = X_tab.fillna(0.0)
    X_tab = X_tab.to_numpy(dtype=np.float32)

    return X_tab, tabular_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, required=True)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--include-tabular", action="store_true")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)

    args = parser.parse_args()

    suffix = f"{args.n_objects}obj"

    lightcurves_path = args.input_root / f"forced_lightcurves_{suffix}.parquet"

    if not lightcurves_path.exists():
        raise FileNotFoundError(lightcurves_path)

    print(f"Loading light curves: {lightcurves_path}")
    lc = pd.read_parquet(lightcurves_path)

    required_cols = ["object_id", "label", "mjd", "band", "flux"]
    missing = [c for c in required_cols if c not in lc.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    lc["object_id"] = lc["object_id"].astype(int)
    lc["label"] = lc["label"].astype(int)

    object_info = (
        lc[["object_id", "label"]]
        .drop_duplicates("object_id")
        .sort_values("object_id")
        .reset_index(drop=True)
    )

    if args.max_objects is not None:
        object_info = object_info.head(args.max_objects)

    object_ids = object_info["object_id"].to_numpy(dtype=np.int64)
    y = object_info["label"].to_numpy(dtype=np.int64)

    n_objects_actual = len(object_ids)

    print("\nTensor dataset summary:")
    print(f"Objects: {n_objects_actual}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Bins: {args.n_bins}")
    print(f"Channels: 18")

    lc = lc[lc["object_id"].isin(object_ids)].copy()

    grouped = lc.groupby("object_id", sort=False)

    object_to_group = {
        int(object_id): group
        for object_id, group in grouped
    }

    X_lc = np.zeros((n_objects_actual, 18, args.n_bins), dtype=np.float32)

    print("\nBuilding temporal tensors...")

    for i, object_id in enumerate(object_ids):
        group = object_to_group.get(int(object_id))

        if group is not None:
            X_lc[i] = object_to_tensor(group, n_bins=args.n_bins)

        if (i + 1) % 5000 == 0 or (i + 1) == n_objects_actual:
            print(f"  processed {i + 1}/{n_objects_actual} objects")

    X_tab = None
    tabular_columns = []

    if args.include_tabular:
        X_tab, tabular_columns = load_tabular_features(
            n_objects=args.n_objects,
            object_ids=object_ids,
        )

    channel_names = (
        [f"flux_{b}" for b in BANDS]
        + [f"snr_{b}" for b in BANDS]
        + [f"mask_{b}" for b in BANDS]
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    debug_suffix = "" if args.max_objects is None else f"_debug{args.max_objects}"
    output_path = args.output_root / f"temporal_tensor_v1_{suffix}_{args.n_bins}bins{debug_suffix}.npz"
    metadata_path = args.output_root / f"temporal_tensor_v1_{suffix}_{args.n_bins}bins{debug_suffix}_metadata.json"

    print(f"\nSaving tensor dataset: {output_path}")

    save_dict = {
        "X_lc": X_lc,
        "y": y,
        "object_id": object_ids,
        "channel_names": np.asarray(channel_names),
        "bands": np.asarray(BANDS),
    }

    if X_tab is not None:
        save_dict["X_tab"] = X_tab
        save_dict["tabular_columns"] = np.asarray(tabular_columns)

    np.savez_compressed(output_path, **save_dict)

    metadata = {
        "n_objects_requested": args.n_objects,
        "n_objects_actual": int(n_objects_actual),
        "n_bins": int(args.n_bins),
        "n_channels": 18,
        "channels": channel_names,
        "classes": sorted([int(v) for v in np.unique(y)]),
        "include_tabular": bool(args.include_tabular),
        "X_lc_shape": list(X_lc.shape),
        "X_tab_shape": None if X_tab is None else list(X_tab.shape),
        "output_path": str(output_path),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"- {output_path}")
    print(f"- {metadata_path}")

    print("\nShapes:")
    print(f"X_lc: {X_lc.shape}")
    print(f"y:    {y.shape}")

    if X_tab is not None:
        print(f"X_tab: {X_tab.shape}")

    print("\nClass distribution:")
    print(pd.Series(y).value_counts().sort_index())


if __name__ == "__main__":
    main()