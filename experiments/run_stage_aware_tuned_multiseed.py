#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run 10-seed stage-aware early-aware tuned experiments for AstroTrust-AI.

Purpose
-------
This script orchestrates the final multiseed protocol for the partial-light-curve
/ broker-like route.

For each seed, it can run:

  1) Hybrid temporal-tabular CNN early-aware training
     experiments/train_hybrid_early_aware_partial_lightcurves.py

  2) Stage-aware early-aware ensemble construction
     experiments/run_stage_aware_early_aware_ensemble.py

  3) Validation-only weight tuning
     experiments/tune_stage_aware_early_aware_weights.py

Then it aggregates:
  - tuned weights across seeds;
  - tuned final-test metrics across seeds;
  - tuned-vs-fixed / tuned-vs-hybrid paired bootstrap summaries across seeds.

Important protocol choice
-------------------------
The object split is kept fixed by default with --split-seed 42.
The training/random seed changes across runs. This isolates training stochasticity
while keeping the evaluation population comparable across seeds.

Default seeds
-------------
42, 123, 777, 2025, 3407, 11, 99, 314, 2718, 9001

Recommended command
-------------------
python .\experiments\run_stage_aware_tuned_multiseed.py `
  --max-train-rows 500000 `
  --epochs 60 `
  --patience 8 `
  --grid-step 0.05 `
  --n-bootstrap 1000 `
  --log-every 100

Faster initial check with 3 seeds
---------------------------------
python .\experiments\run_stage_aware_tuned_multiseed.py `
  --seeds 42,123,777 `
  --max-train-rows 300000 `
  --epochs 35 `
  --patience 6 `
  --n-bootstrap 500 `
  --log-every 100

Resume behavior
---------------
By default, if the expected output of a step already exists, that step is skipped.
Use --force to rerun everything.

Outputs
-------
results/broker_like_stage_aware_multiseed/
  seed_<seed>/
    hybrid/
    stage_aware_ensemble/
    weight_tuning/

  multiseed_run_manifest.csv
  multiseed_tuned_point_metrics_all_seeds.csv
  multiseed_tuned_summary_stats.csv
  multiseed_tuned_weights.csv
  multiseed_tuned_weight_summary.csv
  multiseed_tuned_paired_bootstrap_all_seeds.csv
  multiseed_tuned_paired_summary_stats.csv
  multiseed_stage_aware_tuned_summary.md

Also copied to:
  results/final_publication/broker_like_realism/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SEEDS = "42,123,777,2025,3407,11,99,314,2718,9001"

DEFAULT_HYBRID_SCRIPT = ROOT_DIR / "experiments" / "train_hybrid_early_aware_partial_lightcurves.py"
DEFAULT_STAGE_SCRIPT = ROOT_DIR / "experiments" / "run_stage_aware_early_aware_ensemble.py"
DEFAULT_TUNE_SCRIPT = ROOT_DIR / "experiments" / "tune_stage_aware_early_aware_weights.py"

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "broker_like_stage_aware_multiseed"
DEFAULT_FINAL_DIR = ROOT_DIR / "results" / "final_publication" / "broker_like_realism"

PARTIAL_SCENARIOS = [
    "first_3_points",
    "first_5_points",
    "first_10_points",
    "first_20_points",
    "window_2_days",
    "window_7_days",
    "window_14_days",
    "window_30_days",
]

TRAIN_SCENARIOS = [
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

SUMMARY_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "top3_accuracy",
    "top5_accuracy",
    "rare_enrichment_5pct",
    "rare_rate_5pct",
    "ece",
    "mean_confidence",
    "mean_uncertainty",
]


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_list(value: str, default: list[str]) -> list[str]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def run_command(
    cmd: list[str],
    label: str,
    cwd: Path,
    dry_run: bool,
) -> tuple[str, float, int]:
    log("=" * 100)
    log(f"[START] {label}")
    log("[CMD] " + " ".join(str(x) for x in cmd))
    start = time.time()

    if dry_run:
        log(f"[DRY-RUN] skipped execution: {label}")
        return "dry_run", 0.0, 0

    proc = subprocess.run(cmd, cwd=str(cwd))
    elapsed = (time.time() - start) / 60.0

    if proc.returncode == 0:
        log(f"[DONE] {label} finished in {elapsed:.2f} min")
        return "ok", elapsed, proc.returncode

    log(f"[ERROR] {label} failed with return code {proc.returncode} after {elapsed:.2f} min")
    return "failed", elapsed, proc.returncode


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_dir / src.name)


def expected_hybrid_done(seed_dir: Path) -> bool:
    return (seed_dir / "hybrid" / "hybrid_early_aware_summary.md").exists()


def expected_stage_done(seed_dir: Path) -> bool:
    return (seed_dir / "stage_aware_ensemble" / "stage_aware_ensemble_summary.md").exists()


def expected_tuning_done(seed_dir: Path) -> bool:
    return (seed_dir / "weight_tuning" / "tuned_stage_aware_weight_tuning_summary.md").exists()


def build_hybrid_command(args: argparse.Namespace, seed: int, seed_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.hybrid_script),
        "--output-dir", str(seed_dir / "hybrid"),
        "--final-dir", str(seed_dir / "final_publication_copy"),
        "--scenarios", ",".join(parse_list(args.scenarios, PARTIAL_SCENARIOS)),
        "--train-scenarios", ",".join(parse_list(args.train_scenarios, TRAIN_SCENARIOS)),
        "--seed", str(seed),
        "--split-seed", str(args.split_seed),
        "--max-train-rows", str(args.max_train_rows),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers),
        "--full-baseline-mode", args.full_baseline_mode,
    ]

    if args.deterministic:
        cmd.append("--deterministic")

    if args.max_val_rows is not None:
        cmd += ["--max-val-rows", str(args.max_val_rows)]

    if args.max_train_objects is not None:
        cmd += ["--max-train-objects", str(args.max_train_objects)]

    if args.max_test_objects is not None:
        cmd += ["--max-test-objects", str(args.max_test_objects)]

    return cmd


def build_stage_command(args: argparse.Namespace, seed: int, seed_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.stage_script),
        "--hybrid-dir", str(seed_dir / "hybrid"),
        "--output-dir", str(seed_dir / "stage_aware_ensemble"),
        "--final-dir", str(seed_dir / "final_publication_copy"),
        "--scenarios", ",".join(parse_list(args.scenarios, PARTIAL_SCENARIOS)),
        "--train-scenarios", ",".join(parse_list(args.train_scenarios, TRAIN_SCENARIOS)),
        "--seed", str(seed),
        "--split-seed", str(args.split_seed),
        "--max-train-rows", str(args.max_train_rows),
        "--weight-hybrid", str(args.fixed_weight_hybrid),
        "--weight-lgbm", str(args.fixed_weight_lgbm),
        "--weight-lgbm-regularized", str(args.fixed_weight_lgbm_regularized),
        "--weight-xgboost", str(args.fixed_weight_xgboost),
    ]

    if args.skip_xgboost:
        cmd.append("--skip-xgboost")

    if args.skip_lightgbm:
        cmd.append("--skip-lightgbm")

    if args.class_weight_balanced:
        cmd.append("--class-weight-balanced")

    if args.max_train_objects is not None:
        cmd += ["--max-train-objects", str(args.max_train_objects)]

    if args.max_test_objects is not None:
        cmd += ["--max-test-objects", str(args.max_test_objects)]

    return cmd


def build_tune_command(args: argparse.Namespace, seed: int, seed_dir: Path) -> list[str]:
    # Keep tuning split fixed by default to isolate training-seed variability.
    tuning_seed = args.tuning_split_seed if args.tuning_split_seed is not None else args.split_seed

    cmd = [
        sys.executable,
        str(args.tune_script),
        "--input-dir", str(seed_dir / "stage_aware_ensemble"),
        "--output-dir", str(seed_dir / "weight_tuning"),
        "--final-dir", str(seed_dir / "final_publication_copy"),
        "--scenarios", ",".join(parse_list(args.scenarios, PARTIAL_SCENARIOS)),
        "--grid-step", str(args.grid_step),
        "--objective", args.objective,
        "--final-fraction", str(args.final_fraction),
        "--seed", str(tuning_seed),
        "--n-bootstrap", str(args.n_bootstrap),
        "--ci", str(args.ci),
        "--log-every", str(args.log_every),
    ]
    return cmd


def run_seed(args: argparse.Namespace, seed: int) -> list[dict]:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if args.skip_hybrid:
        log(f"[SKIP] hybrid training disabled for seed {seed}")
        rows.append({"seed": seed, "step": "hybrid", "status": "skipped_by_user", "minutes": 0.0, "returncode": 0})
    elif expected_hybrid_done(seed_dir) and not args.force:
        log(f"[SKIP] hybrid already exists for seed {seed}")
        rows.append({"seed": seed, "step": "hybrid", "status": "already_exists", "minutes": 0.0, "returncode": 0})
    else:
        status, minutes, rc = run_command(
            build_hybrid_command(args, seed, seed_dir),
            f"seed {seed}: hybrid early-aware training",
            ROOT_DIR,
            args.dry_run,
        )
        rows.append({"seed": seed, "step": "hybrid", "status": status, "minutes": minutes, "returncode": rc})
        if status == "failed" and not args.continue_on_error:
            return rows

    if args.skip_stage:
        log(f"[SKIP] stage ensemble disabled for seed {seed}")
        rows.append({"seed": seed, "step": "stage_ensemble", "status": "skipped_by_user", "minutes": 0.0, "returncode": 0})
    elif expected_stage_done(seed_dir) and not args.force:
        log(f"[SKIP] stage ensemble already exists for seed {seed}")
        rows.append({"seed": seed, "step": "stage_ensemble", "status": "already_exists", "minutes": 0.0, "returncode": 0})
    else:
        status, minutes, rc = run_command(
            build_stage_command(args, seed, seed_dir),
            f"seed {seed}: stage-aware early-aware ensemble",
            ROOT_DIR,
            args.dry_run,
        )
        rows.append({"seed": seed, "step": "stage_ensemble", "status": status, "minutes": minutes, "returncode": rc})
        if status == "failed" and not args.continue_on_error:
            return rows

    if args.skip_tuning:
        log(f"[SKIP] tuning disabled for seed {seed}")
        rows.append({"seed": seed, "step": "weight_tuning", "status": "skipped_by_user", "minutes": 0.0, "returncode": 0})
    elif expected_tuning_done(seed_dir) and not args.force:
        log(f"[SKIP] tuning already exists for seed {seed}")
        rows.append({"seed": seed, "step": "weight_tuning", "status": "already_exists", "minutes": 0.0, "returncode": 0})
    else:
        status, minutes, rc = run_command(
            build_tune_command(args, seed, seed_dir),
            f"seed {seed}: validation-only weight tuning",
            ROOT_DIR,
            args.dry_run,
        )
        rows.append({"seed": seed, "step": "weight_tuning", "status": status, "minutes": minutes, "returncode": rc})
        if status == "failed" and not args.continue_on_error:
            return rows

    return rows


def read_seed_outputs(output_dir: Path, seed: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    seed_dir = output_dir / f"seed_{seed}" / "weight_tuning"

    metrics_path = seed_dir / "tuned_stage_aware_point_metrics.csv"
    paired_path = seed_dir / "tuned_stage_aware_paired_bootstrap.csv"
    weights_path = seed_dir / "tuned_weights.json"

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    paired = pd.read_csv(paired_path) if paired_path.exists() else None

    weights = None
    if weights_path.exists():
        weights = json.loads(weights_path.read_text(encoding="utf-8"))

    if metrics is not None:
        metrics.insert(0, "seed", seed)

    if paired is not None:
        paired.insert(0, "seed", seed)

    return metrics, paired, weights


def aggregate_results(args: argparse.Namespace, seeds: list[int]) -> None:
    log("=" * 100)
    log("[AGGREGATE] collecting seed outputs")

    metrics_frames = []
    paired_frames = []
    weight_rows = []

    for seed in seeds:
        metrics, paired, weights = read_seed_outputs(args.output_dir, seed)

        if metrics is not None:
            metrics_frames.append(metrics)
            log(f"[AGGREGATE] metrics loaded for seed {seed}: rows={len(metrics)}")
        else:
            log(f"[WARN] no metrics found for seed {seed}")

        if paired is not None:
            paired_frames.append(paired)
            log(f"[AGGREGATE] paired bootstrap loaded for seed {seed}: rows={len(paired)}")

        if weights is not None:
            row = {"seed": seed}
            row.update({f"weight_{k}": v for k, v in weights.items()})
            weight_rows.append(row)
            log(f"[AGGREGATE] weights loaded for seed {seed}: {weights}")

    if not metrics_frames:
        log("[WARN] no metrics available to aggregate")
        return

    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    paired_all = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    weights_all = pd.DataFrame(weight_rows)

    # Main model for the final partial route.
    final_model = "ensemble_early_aware_tuned"
    final_metrics = metrics_all[metrics_all["model_name"] == final_model].copy()

    group_cols = ["model_name", "scenario"]
    summary_rows = []

    for (model_name, scenario), group in metrics_all.groupby(group_cols):
        row = {
            "model_name": model_name,
            "scenario": scenario,
            "n_seeds": int(group["seed"].nunique()),
            "n_objects_mean": float(group["n_objects"].mean()) if "n_objects" in group else np.nan,
        }
        for metric in SUMMARY_METRICS:
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
                row[f"{metric}_min"] = float(group[metric].min())
                row[f"{metric}_max"] = float(group[metric].max())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    paired_summary = pd.DataFrame()
    if not paired_all.empty:
        paired_rows = []
        for keys, group in paired_all.groupby(["target_model", "reference_model", "scenario", "metric"]):
            target, reference, scenario, metric = keys
            paired_rows.append({
                "target_model": target,
                "reference_model": reference,
                "scenario": scenario,
                "metric": metric,
                "n_seeds": int(group["seed"].nunique()),
                "delta_mean_across_seeds": float(group["delta_target_minus_reference"].mean()),
                "delta_std_across_seeds": float(group["delta_target_minus_reference"].std(ddof=1)) if len(group) > 1 else 0.0,
                "delta_min_across_seeds": float(group["delta_target_minus_reference"].min()),
                "delta_max_across_seeds": float(group["delta_target_minus_reference"].max()),
                "mean_ci_low": float(group["ci_low"].mean()) if "ci_low" in group else np.nan,
                "mean_ci_high": float(group["ci_high"].mean()) if "ci_high" in group else np.nan,
                "max_one_sided_p": float(group["one_sided_bootstrap_p_against_zero"].max()) if "one_sided_bootstrap_p_against_zero" in group else np.nan,
            })
        paired_summary = pd.DataFrame(paired_rows)

    weight_summary = pd.DataFrame()
    if not weights_all.empty:
        rows = []
        for col in [c for c in weights_all.columns if c.startswith("weight_")]:
            rows.append({
                "member": col.replace("weight_", ""),
                "weight_mean": float(weights_all[col].mean()),
                "weight_std": float(weights_all[col].std(ddof=1)) if len(weights_all) > 1 else 0.0,
                "weight_min": float(weights_all[col].min()),
                "weight_max": float(weights_all[col].max()),
                "n_seeds": int(weights_all["seed"].nunique()),
            })
        weight_summary = pd.DataFrame(rows)

    # Save outputs.
    outputs = {
        "multiseed_tuned_point_metrics_all_seeds.csv": metrics_all,
        "multiseed_tuned_summary_stats.csv": summary,
        "multiseed_tuned_weights.csv": weights_all,
        "multiseed_tuned_weight_summary.csv": weight_summary,
        "multiseed_tuned_paired_bootstrap_all_seeds.csv": paired_all,
        "multiseed_tuned_paired_summary_stats.csv": paired_summary,
    }

    for name, df in outputs.items():
        path = args.output_dir / name
        log(f"[SAVE] {path}")
        df.to_csv(path, index=False)
        copy_if_exists(path, args.final_dir)

    summary_path = args.output_dir / "multiseed_stage_aware_tuned_summary.md"
    log(f"[SAVE] {summary_path}")
    write_markdown_summary(summary_path, seeds, summary, weight_summary, paired_summary, args)
    copy_if_exists(summary_path, args.final_dir)


def write_markdown_summary(
    path: Path,
    seeds: list[int],
    summary: pd.DataFrame,
    weight_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Multiseed stage-aware tuned early-aware ensemble",
        "",
        "## Protocol",
        "",
        "This experiment repeats the final partial-light-curve route across multiple training seeds. "
        "The object split is kept fixed by `--split-seed`, while the model seed changes across runs. "
        "For each seed, the early-aware hybrid, tabular members, stage-aware ensemble, and validation-only weight tuning are rerun.",
        "",
        f"- Seeds: `{','.join(str(s) for s in seeds)}`",
        f"- Split seed: `{args.split_seed}`",
        f"- Max early-aware training rows: `{args.max_train_rows}`",
        f"- Epochs: `{args.epochs}`",
        f"- Patience: `{args.patience}`",
        f"- Grid step: `{args.grid_step}`",
        f"- Bootstrap iterations per seed: `{args.n_bootstrap}`",
        "",
    ]

    if not weight_summary.empty:
        lines += [
            "## Tuned weight stability",
            "",
            weight_summary.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]

    if not summary.empty:
        final = summary[summary["model_name"] == "ensemble_early_aware_tuned"].copy()
        keep_scenarios = ["first_3_points", "first_5_points", "first_10_points", "first_20_points", "window_7_days", "window_30_days"]
        final = final[final["scenario"].isin(keep_scenarios)]
        cols = [
            "scenario", "n_seeds",
            "accuracy_mean", "accuracy_std",
            "macro_f1_mean", "macro_f1_std",
            "top5_accuracy_mean", "top5_accuracy_std",
            "rare_enrichment_5pct_mean", "rare_enrichment_5pct_std",
            "ece_mean", "ece_std",
        ]
        cols = [c for c in cols if c in final.columns]
        lines += [
            "## Final tuned partial-route metrics across seeds",
            "",
            final[cols].to_markdown(index=False, floatfmt=".4f") if not final.empty else "_No final tuned metrics available._",
            "",
        ]

    if not paired_summary.empty:
        main = paired_summary[
            (paired_summary["target_model"] == "ensemble_early_aware_tuned")
            & (paired_summary["reference_model"].isin(["ensemble_early_aware_fixed", "hybrid_early_aware"]))
            & (paired_summary["metric"].isin(["accuracy", "macro_f1", "top5_accuracy", "rare_enrichment_5pct"]))
        ].copy()
        keep_scenarios = ["first_5_points", "first_10_points", "first_20_points", "window_30_days"]
        main = main[main["scenario"].isin(keep_scenarios)]
        cols = [
            "reference_model", "scenario", "metric", "n_seeds",
            "delta_mean_across_seeds", "delta_std_across_seeds",
            "delta_min_across_seeds", "delta_max_across_seeds",
            "max_one_sided_p",
        ]
        cols = [c for c in cols if c in main.columns]
        lines += [
            "## Paired gains across seeds",
            "",
            main[cols].to_markdown(index=False, floatfmt=".4f") if not main.empty else "_No paired summaries available._",
            "",
        ]

    lines += [
        "## Suggested manuscript wording",
        "",
        "```latex",
        r"To quantify training-seed stability of the partial-light-curve route, we repeated the validation-only tuned early-aware ensemble protocol across ten independent random seeds while keeping the object-level train/validation/test split fixed. Reported values are mean \(\pm\) standard deviation across seeds, complemented by object-level bootstrap confidence intervals within each seed.",
        "```",
        "",
        "## Interpretation note",
        "",
        "These multiseed results apply only to the partial-light-curve / stage-aware route. "
        "The complete-light-curve metrics remain attached to the original AstroTrust-AI ensemble and do not need to be replaced.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 10-seed stage-aware tuned early-aware experiments.")
    p.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)

    p.add_argument("--hybrid-script", type=Path, default=DEFAULT_HYBRID_SCRIPT)
    p.add_argument("--stage-script", type=Path, default=DEFAULT_STAGE_SCRIPT)
    p.add_argument("--tune-script", type=Path, default=DEFAULT_TUNE_SCRIPT)

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)

    p.add_argument("--scenarios", type=str, default="all")
    p.add_argument("--train-scenarios", type=str, default="all")

    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tuning-split-seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--max-val-rows", type=int, default=0)
    p.add_argument("--max-train-objects", type=int, default=None)
    p.add_argument("--max-test-objects", type=int, default=None)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--full-baseline-mode", choices=["existing_checkpoint", "train", "skip"], default="existing_checkpoint")

    p.add_argument("--fixed-weight-hybrid", type=float, default=0.70)
    p.add_argument("--fixed-weight-lgbm", type=float, default=0.15)
    p.add_argument("--fixed-weight-lgbm-regularized", type=float, default=0.10)
    p.add_argument("--fixed-weight-xgboost", type=float, default=0.05)

    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--skip-lightgbm", action="store_true")
    p.add_argument("--class-weight-balanced", action="store_true")

    p.add_argument("--grid-step", type=float, default=0.05)
    p.add_argument("--objective", choices=["combined", "macro_f1", "accuracy", "top5", "followup"], default="combined")
    p.add_argument("--final-fraction", type=float, default=0.50)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--ci", type=float, default=95.0)
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--skip-hybrid", action="store_true")
    p.add_argument("--skip-stage", action="store_true")
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    seeds = parse_int_list(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_dir.mkdir(parents=True, exist_ok=True)

    log("[START] multiseed tuned stage-aware early-aware experiment")
    log(f"[CONFIG] seeds={seeds}")
    log(f"[CONFIG] output_dir={args.output_dir}")
    log(f"[CONFIG] scenarios={parse_list(args.scenarios, PARTIAL_SCENARIOS)}")
    log(f"[CONFIG] max_train_rows={args.max_train_rows}, epochs={args.epochs}, patience={args.patience}")

    manifest_rows = []

    for i, seed in enumerate(seeds, start=1):
        log("=" * 100)
        log(f"[SEED] {i}/{len(seeds)} seed={seed}")
        rows = run_seed(args, seed)
        manifest_rows.extend(rows)

        manifest = pd.DataFrame(manifest_rows)
        manifest_path = args.output_dir / "multiseed_run_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        copy_if_exists(manifest_path, args.final_dir)

        if any(r["status"] == "failed" for r in rows) and not args.continue_on_error:
            log("[STOP] stopping because one step failed and --continue-on-error is not set")
            break

    aggregate_results(args, seeds)

    elapsed = (time.time() - start) / 60.0
    log("=" * 100)
    log(f"[DONE] multiseed run finished in {elapsed:.2f} min")
    log(f"[DONE] outputs: {args.output_dir}")
    log(f"[DONE] final copy: {args.final_dir}")


if __name__ == "__main__":
    main()
