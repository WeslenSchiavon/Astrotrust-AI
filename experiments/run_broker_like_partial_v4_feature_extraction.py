#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run V4 temporal-shape feature extraction for broker-like partial light-curve scenarios.

This is STEP 2 of the broker-like realism experiment.

It wraps the existing project script:

  src/features/feature_extraction_v4_temporal_shape.py

without modifying it.

Why a wrapper is needed:
  The existing V4 extractor expects files named:
    forced_lightcurves_{n_objects}obj.parquet
    object_context_{n_objects}obj.parquet

  The partial-light-curve preparation step generates scenario-specific files:
    first_5_points.parquet
    window_7_days.parquet
    ...

This wrapper creates a small per-scenario working directory, links or copies the
scenario file using the expected filename, runs the original V4 extractor, and
then copies the extracted features to a publication-friendly partial-features
directory.

Outputs:
  data/processed/broker_like_partial_features/
  results/broker_like_partial_lightcurve_stress/partial_feature_extraction_manifest.csv
  results/broker_like_partial_lightcurve_stress/partial_feature_extraction_summary.md
  results/final_publication/broker_like_realism/partial_feature_extraction_manifest.csv
  results/final_publication/broker_like_realism/partial_feature_extraction_summary.md

Run:
  python experiments/run_broker_like_partial_v4_feature_extraction.py

Optional:
  python experiments/run_broker_like_partial_v4_feature_extraction.py --scenarios first_5_points,first_10_points,window_7_days
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_PARTIAL_INPUT_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_lightcurves"
DEFAULT_CONTEXT_PATH = ROOT_DIR / "data" / "processed" / "elasticc2_large" / "object_context_250000obj.parquet"
DEFAULT_EXTRACTOR = ROOT_DIR / "src" / "features" / "feature_extraction_v4_temporal_shape.py"

DEFAULT_FEATURE_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_features"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "broker_like_partial_lightcurve_stress"
DEFAULT_FINAL_DIR = ROOT_DIR / "results" / "final_publication" / "broker_like_realism"

EXPECTED_SUFFIX = "250000obj"

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


def parse_scenarios(value: str | None) -> list[str]:
    if value is None or value.strip().lower() in {"", "all"}:
        return DEFAULT_SCENARIOS
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def link_or_copy(src: Path, dst: Path, mode: str = "auto") -> str:
    """
    Create dst from src.

    mode:
      auto: try hardlink first, fallback to copy
      hardlink: require hardlink
      copy: copy file
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    if mode not in {"auto", "hardlink", "copy"}:
        raise ValueError(f"Invalid link mode: {mode}")

    if mode in {"auto", "hardlink"}:
        try:
            os.link(src, dst)
            return "hardlink"
        except Exception:
            if mode == "hardlink":
                raise

    shutil.copy2(src, dst)
    return "copy"


def read_object_count_from_features(path: Path) -> tuple[int | None, int | None]:
    """
    Read only metadata-light table shape if possible.

    Pandas still needs to open the parquet file, but this is only done once per
    extracted feature file.
    """
    try:
        df = pd.read_parquet(path)
        return int(len(df)), int(df.shape[1])
    except Exception:
        return None, None


def write_summary(summary_path: Path, manifest: pd.DataFrame) -> None:
    lines = [
        "# Broker-like partial V4 feature extraction",
        "",
        "## Goal",
        "",
        "This step recomputes V4 temporal-shape/tabular features for each partial light-curve scenario. "
        "The extraction uses only the observations available in each truncated scenario, avoiding temporal leakage from the full curve.",
        "",
        "## Extracted scenarios",
        "",
        "| scenario | status | n_objects | n_features | output | runtime min |",
        "|:--|:--|--:|--:|:--|--:|",
    ]

    for row in manifest.itertuples(index=False):
        n_objects = "" if pd.isna(row.n_objects) else int(row.n_objects)
        n_features = "" if pd.isna(row.n_features) else int(row.n_features)
        runtime = "" if pd.isna(row.runtime_minutes) else f"{row.runtime_minutes:.2f}"
        lines.append(
            f"| `{row.scenario}` | {row.status} | {n_objects} | {n_features} | "
            f"`{row.output_features}` | {runtime} |"
        )

    lines += [
        "",
        "## Next step",
        "",
        "Run the partial inference/evaluation step on the generated feature files. "
        "The most important manuscript comparison should report both all-available-object results and, when possible, a common-object subset across temporal scenarios.",
        "",
        "## Methodological note",
        "",
        "The feature files generated here are scenario-specific. They should not be replaced by full-curve features, because that would leak future information into early-alert evaluation.",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_one_scenario(
    scenario: str,
    partial_input_dir: Path,
    context_path: Path,
    extractor_path: Path,
    work_dir: Path,
    feature_output_dir: Path,
    link_mode: str,
    overwrite: bool,
) -> dict:
    scenario_input = partial_input_dir / f"{scenario}.parquet"
    if not scenario_input.exists():
        return {
            "scenario": scenario,
            "status": "missing_input",
            "input_lightcurve": safe_relative(scenario_input),
            "output_features": "",
            "n_objects": None,
            "n_features": None,
            "runtime_minutes": None,
            "link_mode_lightcurve": "",
            "link_mode_context": "",
        }

    output_features = feature_output_dir / f"features_v4_temporal_shape_{scenario}.parquet"
    if output_features.exists() and not overwrite:
        n_objects, n_features = read_object_count_from_features(output_features)
        return {
            "scenario": scenario,
            "status": "already_exists",
            "input_lightcurve": safe_relative(scenario_input),
            "output_features": safe_relative(output_features),
            "n_objects": n_objects,
            "n_features": n_features,
            "runtime_minutes": 0.0,
            "link_mode_lightcurve": "",
            "link_mode_context": "",
        }

    scenario_work = work_dir / scenario
    input_root = scenario_work / "input"
    output_root = scenario_work / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    expected_lc = input_root / f"forced_lightcurves_{EXPECTED_SUFFIX}.parquet"
    expected_context = input_root / f"object_context_{EXPECTED_SUFFIX}.parquet"

    lc_mode = link_or_copy(scenario_input, expected_lc, mode=link_mode)
    ctx_mode = link_or_copy(context_path, expected_context, mode=link_mode)

    cmd = [
        sys.executable,
        str(extractor_path),
        "--n-objects",
        "250000",
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]

    print("\n" + "=" * 90)
    print(f"[RUN] Scenario: {scenario}")
    print(f"[RUN] Input: {scenario_input}")
    print(f"[RUN] Command: {' '.join(cmd)}")

    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), text=True)
    runtime_minutes = (time.time() - start) / 60.0

    raw_output = output_root / f"features_v4_temporal_shape_{EXPECTED_SUFFIX}.parquet"

    if proc.returncode != 0:
        return {
            "scenario": scenario,
            "status": f"failed_returncode_{proc.returncode}",
            "input_lightcurve": safe_relative(scenario_input),
            "output_features": "",
            "n_objects": None,
            "n_features": None,
            "runtime_minutes": runtime_minutes,
            "link_mode_lightcurve": lc_mode,
            "link_mode_context": ctx_mode,
        }

    if not raw_output.exists():
        return {
            "scenario": scenario,
            "status": "failed_missing_extractor_output",
            "input_lightcurve": safe_relative(scenario_input),
            "output_features": "",
            "n_objects": None,
            "n_features": None,
            "runtime_minutes": runtime_minutes,
            "link_mode_lightcurve": lc_mode,
            "link_mode_context": ctx_mode,
        }

    if output_features.exists():
        output_features.unlink()

    shutil.copy2(raw_output, output_features)

    n_objects, n_features = read_object_count_from_features(output_features)

    return {
        "scenario": scenario,
        "status": "ok",
        "input_lightcurve": safe_relative(scenario_input),
        "output_features": safe_relative(output_features),
        "n_objects": n_objects,
        "n_features": n_features,
        "runtime_minutes": runtime_minutes,
        "link_mode_lightcurve": lc_mode,
        "link_mode_context": ctx_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V4 feature extraction for broker-like partial light-curve scenarios."
    )
    parser.add_argument("--partial-input-dir", type=Path, default=DEFAULT_PARTIAL_INPUT_DIR)
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT_PATH)
    parser.add_argument("--extractor", type=Path, default=DEFAULT_EXTRACTOR)
    parser.add_argument("--feature-output-dir", type=Path, default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_RESULTS_DIR / "feature_extraction_work")
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "copy"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.partial_input_dir.exists():
        raise FileNotFoundError(f"Partial input dir not found: {args.partial_input_dir}")

    if not args.context_path.exists():
        raise FileNotFoundError(f"Context file not found: {args.context_path}")

    if not args.extractor.exists():
        raise FileNotFoundError(f"Extractor script not found: {args.extractor}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.final_dir.mkdir(parents=True, exist_ok=True)
    args.feature_output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = parse_scenarios(args.scenarios)

    rows = []
    for scenario in scenarios:
        row = run_one_scenario(
            scenario=scenario,
            partial_input_dir=args.partial_input_dir,
            context_path=args.context_path,
            extractor_path=args.extractor,
            work_dir=args.work_dir,
            feature_output_dir=args.feature_output_dir,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
        rows.append(row)
        print(f"[STATUS] {scenario}: {row['status']}")

    manifest = pd.DataFrame(rows)
    manifest_path = args.results_dir / "partial_feature_extraction_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    final_manifest_path = args.final_dir / manifest_path.name
    manifest.to_csv(final_manifest_path, index=False)

    summary_path = args.results_dir / "partial_feature_extraction_summary.md"
    write_summary(summary_path, manifest)

    final_summary_path = args.final_dir / summary_path.name
    shutil.copyfile(summary_path, final_summary_path)

    print("\n" + "=" * 90)
    print("[OK] Partial feature extraction wrapper finished.")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] Publication copy: {args.final_dir}")


if __name__ == "__main__":
    main()
