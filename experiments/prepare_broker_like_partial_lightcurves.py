#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare broker-like partial light-curve datasets for AstroTrust-AI.

This is STEP 1 of the broker-like realism experiment.

It creates temporally truncated versions of the same object-level light curves:
  - full_curve_reference
  - first_N_points
  - window_X_days after the first alert

The script intentionally DOES NOT modify final_publication_summary.md.

Outputs:
  data/processed/broker_like_partial_lightcurves/*.parquet
  results/broker_like_partial_lightcurve_stress/partial_lightcurve_scenario_manifest.csv
  results/broker_like_partial_lightcurve_stress/partial_lightcurve_prepare_summary.md
  results/final_publication/broker_like_realism/partial_lightcurve_scenario_manifest.csv
  results/final_publication/broker_like_realism/partial_lightcurve_prepare_summary.md

Run from any location:
  python experiments/prepare_broker_like_partial_lightcurves.py ^
    --lightcurves data/processed/lightcurves_250k.parquet
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DATA_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_lightcurves"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "broker_like_partial_lightcurve_stress"
DEFAULT_FINAL_DIR = ROOT_DIR / "results" / "final_publication" / "broker_like_realism"

OBJECT_COL_CANDIDATES = [
    "object_id", "objectId", "diaObjectId", "dia_object_id", "oid", "id", "source_id"
]

TIME_COL_CANDIDATES = [
    "mjd", "midpointtai", "midPointTai", "time", "t", "days", "obs_time"
]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path.suffix}. Use .parquet or .csv")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {path.suffix}")


def detect_column(df: pd.DataFrame, candidates: list[str], requested: str, label: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Requested {label} column not found: {requested}")
        return requested

    lookup = {str(c).lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]

    raise ValueError(
        f"Could not detect {label} column. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def scenario_label_for_window(days: float) -> str:
    if float(days).is_integer():
        return str(int(days))
    return str(days).replace(".", "p")


def summarize(df: pd.DataFrame, object_col: str) -> dict[str, float | int]:
    counts = df.groupby(object_col).size()

    if counts.empty:
        return {
            "n_objects": 0,
            "n_observations": 0,
            "min_points_per_object": 0,
            "median_points_per_object": 0.0,
            "mean_points_per_object": 0.0,
            "max_points_per_object": 0,
        }

    return {
        "n_objects": int(counts.shape[0]),
        "n_observations": int(df.shape[0]),
        "min_points_per_object": int(counts.min()),
        "median_points_per_object": float(counts.median()),
        "mean_points_per_object": float(counts.mean()),
        "max_points_per_object": int(counts.max()),
    }


def filter_min_points(df: pd.DataFrame, object_col: str, min_points: int) -> pd.DataFrame:
    if min_points <= 1:
        return df

    counts = df.groupby(object_col).size()
    keep_objects = counts[counts >= min_points].index
    return df[df[object_col].isin(keep_objects)].copy()


def create_full_reference(df: pd.DataFrame, object_col: str, time_col: str, min_points: int) -> pd.DataFrame:
    out = df.sort_values([object_col, time_col]).copy()
    return filter_min_points(out, object_col, min_points).reset_index(drop=True)


def create_first_n(df: pd.DataFrame, object_col: str, time_col: str, n_points: int, min_points: int) -> pd.DataFrame:
    out = (
        df.sort_values([object_col, time_col])
        .groupby(object_col, sort=False)
        .head(n_points)
        .copy()
    )
    return filter_min_points(out, object_col, min_points).reset_index(drop=True)


def create_window_days(df: pd.DataFrame, object_col: str, time_col: str, days: float, min_points: int) -> pd.DataFrame:
    ordered = df.sort_values([object_col, time_col]).copy()
    first_time = ordered.groupby(object_col, sort=False)[time_col].transform("min")
    out = ordered[ordered[time_col] <= first_time + days].copy()
    return filter_min_points(out, object_col, min_points).reset_index(drop=True)


def write_summary(
    summary_path: Path,
    manifest: pd.DataFrame,
    lightcurves_path: Path,
    object_col: str,
    time_col: str,
    min_points: int,
) -> None:
    lines = [
        "# Broker-like partial light-curve preparation",
        "",
        "## Goal",
        "",
        "This preparation step creates partial light-curve datasets to evaluate broker-like realism. "
        "Each scenario keeps only the observations that would be available early in the alert stream.",
        "",
        "## Input",
        "",
        f"- Light-curve file: `{lightcurves_path}`",
        f"- Object column: `{object_col}`",
        f"- Time column: `{time_col}`",
        f"- Minimum retained observations per object: `{min_points}`",
        "",
        "## Generated scenarios",
        "",
        "| scenario | type | description | n_objects | n_observations | median points/object | output |",
        "|:--|:--|:--|--:|--:|--:|:--|",
    ]

    for row in manifest.itertuples(index=False):
        lines.append(
            f"| `{row.scenario}` | {row.scenario_type} | {row.description} | "
            f"{row.n_objects} | {row.n_observations} | {row.median_points_per_object:.1f} | `{row.output_path}` |"
        )

    lines += [
        "",
        "## Required next step",
        "",
        "Run the existing AstroTrust-AI feature-extraction and inference pipeline on each generated `.parquet` file. "
        "The critical methodological requirement is that all light-curve-derived features must be recomputed from the truncated data only. "
        "Do not reuse features derived from full curves.",
        "",
        "Suggested prediction output directory:",
        "",
        "`results/broker_like_partial_lightcurve_stress/predictions/`",
        "",
        "Suggested filenames:",
        "",
        "- `full_curve_reference_predictions.csv`",
        "- `first_5_points_predictions.csv`",
        "- `window_7_days_predictions.csv`",
        "",
        "Each prediction file should ideally contain:",
        "",
        "- object id",
        "- true label",
        "- predicted top-1 label",
        "- top-k labels or probabilities",
        "- confidence",
        "- uncertainty",
        "- novelty",
        "- rarity",
        "- true rare-class indicator",
        "",
        "After those predictions are generated, the next analysis step can compute classification degradation, uncertainty shifts, top-k retention, and rare-object enrichment under partial information.",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    lightcurves_path = args.lightcurves
    output_data_dir = args.output_data_dir
    results_dir = args.results_dir
    final_dir = args.final_dir

    output_data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(lightcurves_path)

    object_col = detect_column(df, OBJECT_COL_CANDIDATES, args.object_col, "object id")
    time_col = detect_column(df, TIME_COL_CANDIDATES, args.time_col, "time")

    if args.max_objects is not None:
        object_ids = df[object_col].drop_duplicates().head(args.max_objects)
        df = df[df[object_col].isin(object_ids)].copy()

    first_points = parse_int_list(args.first_points)
    windows_days = parse_float_list(args.windows_days)

    scenario_rows = []

    scenarios = []

    scenarios.append((
        "full_curve_reference",
        "full",
        "All available observations per object",
        create_full_reference(df, object_col, time_col, args.min_points_per_object),
    ))

    for n in first_points:
        scenarios.append((
            f"first_{n}_points",
            "first_points",
            f"First {n} observations per object",
            create_first_n(df, object_col, time_col, n, args.min_points_per_object),
        ))

    for days in windows_days:
        label = scenario_label_for_window(days)
        scenarios.append((
            f"window_{label}_days",
            "window_days",
            f"Observations within {days:g} days after first alert",
            create_window_days(df, object_col, time_col, days, args.min_points_per_object),
        ))

    for name, scenario_type, description, out_df in scenarios:
        output_path = output_data_dir / f"{name}.{args.output_format}"
        write_table(out_df, output_path)

        stats = summarize(out_df, object_col)

        scenario_rows.append({
            "scenario": name,
            "scenario_type": scenario_type,
            "description": description,
            "output_path": str(output_path.relative_to(ROOT_DIR) if str(output_path).startswith(str(ROOT_DIR)) else output_path),
            **stats,
        })

        print(
            f"[OK] {name}: "
            f"{stats['n_objects']} objects, "
            f"{stats['n_observations']} observations, "
            f"median={stats['median_points_per_object']:.1f} points/object"
        )

    manifest = pd.DataFrame(scenario_rows)

    manifest_path = results_dir / "partial_lightcurve_scenario_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    final_manifest_path = final_dir / "partial_lightcurve_scenario_manifest.csv"
    manifest.to_csv(final_manifest_path, index=False)

    summary_path = results_dir / "partial_lightcurve_prepare_summary.md"
    write_summary(
        summary_path=summary_path,
        manifest=manifest,
        lightcurves_path=lightcurves_path,
        object_col=object_col,
        time_col=time_col,
        min_points=args.min_points_per_object,
    )

    final_summary_path = final_dir / summary_path.name
    shutil.copyfile(summary_path, final_summary_path)

    print(f"[OK] Manifest saved to: {manifest_path}")
    print(f"[OK] Summary saved to: {summary_path}")
    print(f"[OK] Publication copy saved to: {final_dir}")


def run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)

    base = args.results_dir / "_self_test"
    lc_path = base / "synthetic_lightcurves.csv"

    rows = []
    for oid in range(50):
        n = int(rng.integers(5, 30))
        start = float(rng.uniform(59000, 59100))
        times = np.sort(start + rng.uniform(0, 60, size=n))
        for t in times:
            rows.append({
                "object_id": oid,
                "mjd": t,
                "flux": float(rng.normal()),
                "fluxerr": float(rng.uniform(0.01, 0.2)),
            })

    lc = pd.DataFrame(rows)
    lc_path.parent.mkdir(parents=True, exist_ok=True)
    lc.to_csv(lc_path, index=False)

    test_args = argparse.Namespace(
        lightcurves=lc_path,
        output_data_dir=base / "partial",
        results_dir=base / "results",
        final_dir=base / "final_publication" / "broker_like_realism",
        object_col="auto",
        time_col="auto",
        first_points="3,5,10",
        windows_days="7,14,30",
        min_points_per_object=2,
        output_format="csv",
        max_objects=None,
    )

    prepare(test_args)

    print("[OK] Self-test completed.")
    print(f"[OK] Synthetic outputs are under: {base}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare broker-like partial light-curve datasets."
    )

    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare partial light-curve datasets.")
    prepare_parser.add_argument("--lightcurves", type=Path, required=True)
    prepare_parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    prepare_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    prepare_parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    prepare_parser.add_argument("--object-col", type=str, default="auto")
    prepare_parser.add_argument("--time-col", type=str, default="auto")
    prepare_parser.add_argument("--first-points", type=str, default="3,5,10,20")
    prepare_parser.add_argument("--windows-days", type=str, default="2,7,14,30")
    prepare_parser.add_argument("--min-points-per-object", type=int, default=2)
    prepare_parser.add_argument("--max-objects", type=int, default=None)
    prepare_parser.add_argument("--output-format", choices=["parquet", "csv"], default="parquet")

    self_parser = subparsers.add_parser("self-test", help="Run a synthetic smoke test.")
    self_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    args = parser.parse_args()

    if args.command is None:
        args.command = "prepare"

    return args


def main() -> None:
    args = parse_args()

    if args.command == "prepare":
        prepare(args)
    elif args.command == "self-test":
        run_self_test(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
