#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build temporal CNN tensors for broker-like partial light-curve scenarios.

This is the missing bridge for training the same AstroTrust-AI hybrid
Temporal+Tabular CNN under early-aware/partial-light-curve conditions.

Inputs expected from previous steps:
  data/processed/broker_like_partial_lightcurves/{scenario}.parquet
  data/processed/broker_like_partial_features/features_v4_temporal_shape_{scenario}.parquet

Outputs:
  data/processed/broker_like_partial_tensors/temporal_tensor_v1_{scenario}_64bins.npz
  data/processed/broker_like_partial_tensors/temporal_tensor_v1_{scenario}_64bins_metadata.json
  results/broker_like_hybrid_early_aware/partial_temporal_tensor_manifest.csv
  results/broker_like_hybrid_early_aware/partial_temporal_tensor_summary.md

Run:
  python experiments/build_broker_like_partial_temporal_tensors.py

Smoke test:
  python experiments/build_broker_like_partial_temporal_tensors.py --self-test
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_LIGHTCURVE_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_lightcurves"
DEFAULT_FEATURE_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_features"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_tensors"
DEFAULT_REFERENCE_TENSOR = ROOT_DIR / "data" / "processed" / "elasticc2_large" / "temporal_tensor_v1_250000obj_64bins.npz"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "broker_like_hybrid_early_aware"
DEFAULT_FINAL_DIR = ROOT_DIR / "results" / "final_publication" / "broker_like_realism"

DEFAULT_SCENARIOS = [
    "full_curve_reference",
    "first_3_points",
    "first_5_points",
    "first_10_points",
    "first_20_points",
    "window_2_days",
    "window_7_days",
    "window_14_days",
    "window_30_days",
]

BANDS = ["u", "g", "r", "i", "z", "Y"]
BAND_TO_IDX = {b: i for i, b in enumerate(BANDS)}
NUMERIC_BAND_MAP = {
    "0": "u", "1": "g", "2": "r", "3": "i", "4": "z", "5": "Y",
    0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "Y",
}
EPS = 1e-8

OBJECT_COL_CANDIDATES = ["object_id", "diaobjectid", "diaObjectId", "oid", "snid"]
LABEL_COL_CANDIDATES = ["label", "true_label", "target", "class_id", "true_class"]
TIME_COL_CANDIDATES = ["mjd", "midpointtai", "time", "jd"]
BAND_COL_CANDIDATES = ["band", "filter", "filtername", "passband"]
FLUX_COL_CANDIDATES = ["flux", "psflux", "forced_flux"]


def parse_list(value: str, default: list[str]) -> list[str]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def detect_column(df: pd.DataFrame, candidates: list[str], explicit: str | None, label: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit {label} column `{explicit}` not found. Available: {list(df.columns)}")
        return explicit
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"Could not detect {label} column. Available columns: {list(df.columns)}")


def normalize_band(value):
    if value in NUMERIC_BAND_MAP:
        return NUMERIC_BAND_MAP[value]
    value_str = str(value).strip()
    if value_str in NUMERIC_BAND_MAP:
        return NUMERIC_BAND_MAP[value_str]
    if value_str.lower() == "y":
        return "Y"
    return value_str.lower()


def compute_snr(df: pd.DataFrame, flux_col: str) -> np.ndarray:
    if "snr" in df.columns:
        return pd.to_numeric(df["snr"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    flux = pd.to_numeric(df[flux_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    for err_col in ["flux_err", "fluxerr", "flux_error", "psfluxerr", "psFluxErr"]:
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


def object_to_tensor(group: pd.DataFrame, n_bins: int, time_col: str, band_col: str, flux_col: str) -> np.ndarray:
    """
    Output shape [18, n_bins].
    Channels: 0:6 flux, 6:12 SNR, 12:18 observation mask.
    """
    tensor = np.zeros((18, n_bins), dtype=np.float32)
    counts = np.zeros((6, n_bins), dtype=np.float32)

    group = group.copy()
    group["_band_norm"] = group[band_col].map(normalize_band)

    mjd = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=np.float32)
    flux = pd.to_numeric(group[flux_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    snr = compute_snr(group, flux_col=flux_col)

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
    flux_norm = np.clip(flux / scale, -10.0, 10.0) / 10.0
    snr_norm = np.clip(snr, -20.0, 20.0) / 20.0
    bands = group["_band_norm"].to_numpy()

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

    for b in range(6):
        nz = counts[b] > 0
        tensor[b, nz] /= counts[b, nz]
        tensor[6 + b, nz] /= counts[b, nz]

    return tensor


def find_lightcurve_path(lightcurve_dir: Path, scenario: str) -> Path:
    for suffix in ["parquet", "csv"]:
        path = lightcurve_dir / f"{scenario}.{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No light-curve file found for scenario `{scenario}` in {lightcurve_dir}")


def find_feature_path(feature_dir: Path, scenario: str) -> Path:
    candidates = [
        feature_dir / f"features_v4_temporal_shape_{scenario}.parquet",
        feature_dir / f"features_v4_temporal_shape_{scenario}.csv",
        feature_dir / f"{scenario}.parquet",
        feature_dir / f"{scenario}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No feature file found for scenario `{}`. Tried:\n{}".format(scenario, "\n".join(str(p) for p in candidates)))


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except Exception:
        return str(path)


def load_reference_tabular_columns(path: Path | None, use_reference: bool) -> list[str] | None:
    if not use_reference or path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if "tabular_columns" not in data:
        print(f"[WARN] Reference tensor has no tabular_columns: {path}")
        return None
    cols = [str(c) for c in data["tabular_columns"].tolist()]
    print(f"[INFO] Using {len(cols)} reference tabular columns from {path}")
    return cols


def build_one_scenario(args: argparse.Namespace, scenario: str, reference_tabular_columns: list[str] | None = None) -> dict:
    lc_path = find_lightcurve_path(args.lightcurve_dir, scenario)
    feat_path = find_feature_path(args.feature_dir, scenario)

    print("\n" + "=" * 90)
    print(f"[SCENARIO] {scenario}")
    print(f"[LC]       {lc_path}")
    print(f"[FEATURE]  {feat_path}")

    lc = read_table(lc_path)
    object_col = detect_column(lc, OBJECT_COL_CANDIDATES, args.object_col, "object id")
    label_col = detect_column(lc, LABEL_COL_CANDIDATES, args.label_col, "label")
    time_col = detect_column(lc, TIME_COL_CANDIDATES, args.time_col, "time")
    band_col = detect_column(lc, BAND_COL_CANDIDATES, args.band_col, "band")
    flux_col = detect_column(lc, FLUX_COL_CANDIDATES, args.flux_col, "flux")

    lc[object_col] = lc[object_col].astype(np.int64)
    lc[label_col] = lc[label_col].astype(np.int64)

    object_info = (
        lc[[object_col, label_col]]
        .drop_duplicates(object_col)
        .sort_values(object_col)
        .reset_index(drop=True)
    )

    if args.max_objects is not None and args.max_objects > 0:
        object_info = object_info.head(args.max_objects).copy()

    features = read_table(feat_path)
    if "object_id" not in features.columns:
        raise ValueError(f"Feature file lacks object_id column: {feat_path}")
    features["object_id"] = features["object_id"].astype(np.int64)

    available_feature_ids = set(features["object_id"].to_numpy())
    before_feature_filter = len(object_info)
    object_info = object_info[object_info[object_col].isin(available_feature_ids)].copy()

    if len(object_info) == 0:
        raise RuntimeError(f"No overlapping objects between light curves and features for scenario {scenario}")

    object_ids = object_info[object_col].to_numpy(dtype=np.int64)
    y = object_info[label_col].to_numpy(dtype=np.int64)

    lc = lc[lc[object_col].isin(object_ids)].copy()
    grouped = lc.groupby(object_col, sort=False)
    object_to_group = {int(oid): group for oid, group in grouped}

    X_lc = np.zeros((len(object_ids), 18, args.n_bins), dtype=np.float32)
    print(f"[BUILD] objects={len(object_ids)}, observations={len(lc)}, bins={args.n_bins}")

    for i, oid in enumerate(object_ids):
        group = object_to_group.get(int(oid))
        if group is not None:
            X_lc[i] = object_to_tensor(group, args.n_bins, time_col, band_col, flux_col)
        if (i + 1) % args.progress_every == 0 or (i + 1) == len(object_ids):
            print(f"  processed {i + 1}/{len(object_ids)} objects")

    features = features.set_index("object_id").loc[object_ids]
    X_tab_df = features.drop(columns=["label"], errors="ignore")
    X_tab_df = X_tab_df.apply(pd.to_numeric, errors="coerce")

    if reference_tabular_columns is not None:
        missing_cols = [c for c in reference_tabular_columns if c not in X_tab_df.columns]
        extra_cols = [c for c in X_tab_df.columns if c not in reference_tabular_columns]
        if missing_cols:
            print(f"[WARN] {scenario}: adding {len(missing_cols)} missing reference tabular columns as zeros")
        if extra_cols:
            print(f"[INFO] {scenario}: dropping {len(extra_cols)} tabular columns not present in reference tensor")
        X_tab_df = X_tab_df.reindex(columns=reference_tabular_columns, fill_value=0.0)

    tabular_columns = X_tab_df.columns.tolist()
    X_tab_df = X_tab_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_tab = X_tab_df.to_numpy(dtype=np.float32)

    channel_names = (
        [f"flux_{b}" for b in BANDS]
        + [f"snr_{b}" for b in BANDS]
        + [f"mask_{b}" for b in BANDS]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"temporal_tensor_v1_{scenario}_{args.n_bins}bins.npz"
    meta_path = args.output_dir / f"temporal_tensor_v1_{scenario}_{args.n_bins}bins_metadata.json"

    np.savez_compressed(
        out_path,
        X_lc=X_lc,
        X_tab=X_tab,
        y=y,
        object_id=object_ids,
        channel_names=np.asarray(channel_names),
        tabular_columns=np.asarray(tabular_columns),
        bands=np.asarray(BANDS),
        scenario=np.asarray([scenario]),
    )

    counts = lc.groupby(object_col).size()
    metadata = {
        "scenario": scenario,
        "lightcurve_path": safe_rel(lc_path),
        "feature_path": safe_rel(feat_path),
        "n_objects_before_feature_filter": int(before_feature_filter),
        "n_objects": int(len(object_ids)),
        "n_observations": int(len(lc)),
        "median_points_per_object": float(counts.median()),
        "mean_points_per_object": float(counts.mean()),
        "n_bins": int(args.n_bins),
        "n_channels": 18,
        "n_tabular": int(X_tab.shape[1]),
        "used_reference_tabular_columns": bool(reference_tabular_columns is not None),
        "classes": sorted([int(v) for v in np.unique(y)]),
        "X_lc_shape": list(X_lc.shape),
        "X_tab_shape": list(X_tab.shape),
        "output_path": safe_rel(out_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] saved {out_path}")

    return {
        "scenario": scenario,
        "status": "ok",
        **metadata,
        "metadata_path": safe_rel(meta_path),
    }


def write_summary(path: Path, manifest: pd.DataFrame) -> None:
    lines = [
        "# Broker-like partial temporal tensor preparation",
        "",
        "This step builds temporal CNN tensors for each partial light-curve scenario and aligns them with the scenario-specific V4 tabular features.",
        "",
        "## Generated tensors",
        "",
    ]
    cols = [
        "scenario", "n_objects", "n_observations", "median_points_per_object",
        "n_bins", "n_tabular", "output_path",
    ]
    lines.append(manifest[[c for c in cols if c in manifest.columns]].to_markdown(index=False))
    lines += [
        "",
        "## Next step",
        "",
        "Run `experiments/train_hybrid_early_aware_partial_lightcurves.py` to train the hybrid temporal--tabular CNN with complete and partial representations.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    scenarios = parse_list(args.scenarios, DEFAULT_SCENARIOS)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.final_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    reference_tabular_columns = load_reference_tabular_columns(args.reference_tensor, args.use_reference_tabular_columns)
    for scenario in scenarios:
        out_path = args.output_dir / f"temporal_tensor_v1_{scenario}_{args.n_bins}bins.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {scenario}: {out_path} already exists. Use --overwrite to rebuild.")
            meta_path = args.output_dir / f"temporal_tensor_v1_{scenario}_{args.n_bins}bins_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                rows.append({"scenario": scenario, "status": "already_exists", **meta, "metadata_path": safe_rel(meta_path)})
            continue
        rows.append(build_one_scenario(args, scenario, reference_tabular_columns=reference_tabular_columns))

    manifest = pd.DataFrame(rows)
    manifest_path = args.results_dir / "partial_temporal_tensor_manifest.csv"
    summary_path = args.results_dir / "partial_temporal_tensor_summary.md"
    manifest.to_csv(manifest_path, index=False)
    write_summary(summary_path, manifest)

    shutil.copyfile(manifest_path, args.final_dir / manifest_path.name)
    shutil.copyfile(summary_path, args.final_dir / summary_path.name)

    print("\n" + "=" * 90)
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Summary:  {summary_path}")
    print(f"[OK] Publication copy: {args.final_dir}")


def run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)
    base = args.results_dir / "_self_test_tensor_builder"
    lc_dir = base / "lightcurves"
    feat_dir = base / "features"
    out_dir = base / "tensors"
    lc_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["full_curve_reference", "first_3_points", "window_7_days"]
    for scenario in scenarios:
        rows = []
        feats = []
        for oid in range(60):
            label = oid % 4
            n = 20 if scenario == "full_curve_reference" else (3 if scenario == "first_3_points" else 8)
            for j in range(n):
                rows.append({
                    "object_id": oid,
                    "label": label,
                    "mjd": 60000 + j + rng.normal(0, 0.05),
                    "band": rng.choice(BANDS),
                    "flux": rng.normal(label, 1.0),
                    "flux_err": 0.1 + rng.random(),
                })
            feats.append({"object_id": oid, "label": label, **{f"f{k}": rng.normal(label, 1.0) for k in range(10)}})
        pd.DataFrame(rows).to_csv(lc_dir / f"{scenario}.csv", index=False)
        pd.DataFrame(feats).to_csv(feat_dir / f"features_v4_temporal_shape_{scenario}.csv", index=False)

    test_args = argparse.Namespace(**vars(args))
    test_args.lightcurve_dir = lc_dir
    test_args.feature_dir = feat_dir
    test_args.output_dir = out_dir
    test_args.results_dir = base / "results"
    test_args.final_dir = base / "final"
    test_args.scenarios = ",".join(scenarios)
    test_args.max_objects = None
    test_args.overwrite = True
    run(test_args)
    print(f"[OK] Self-test outputs: {base}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build partial temporal tensors for hybrid early-aware training.")
    p.add_argument("--lightcurve-dir", type=Path, default=DEFAULT_LIGHTCURVE_DIR)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--reference-tensor", type=Path, default=DEFAULT_REFERENCE_TENSOR)
    p.add_argument("--use-reference-tabular-columns", action="store_true", default=True)
    p.add_argument("--no-reference-tabular-columns", dest="use_reference_tabular_columns", action="store_false")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    p.add_argument("--scenarios", type=str, default="all")
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--object-col", type=str, default=None)
    p.add_argument("--label-col", type=str, default=None)
    p.add_argument("--time-col", type=str, default=None)
    p.add_argument("--band-col", type=str, default=None)
    p.add_argument("--flux-col", type=str, default=None)
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
