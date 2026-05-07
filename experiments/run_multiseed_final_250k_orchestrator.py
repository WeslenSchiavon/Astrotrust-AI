from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "multiseed_250k"
DEFAULT_PLAN_PATH = DEFAULT_OUTPUT_DIR / "multiseed_plan.json"

DEFAULT_SEEDS = [42, 123, 777, 2025, 3407, 11, 99, 314, 2718, 9001]

# Candidate scripts are used only when initializing the editable JSON plan.
# The orchestrator itself executes the command templates stored in the plan.
SCRIPT_CANDIDATES = {
    "lightgbm_v4_baseline": [
        "experiments/train_lightgbm_v4_baseline_250k.py",
        "experiments/train_lightgbm_v4_baseline_250k_multiseed_ready.py",
        "experiments/train_lightgbm_v4_250k.py",
        "experiments/train_v4_temporal_shape_250k.py",
        "experiments/run_v4_temporal_shape_250k.py",
        "experiments/run_lightgbm_v4_250k.py",
    ],
    "temporal_cnn": [
        "experiments/train_temporal_cnn_250k.py",
        "experiments/train_temporal_cnn_250k_multiseed_ready.py",
    ],
    "temporal_cnn_v2": [
        "experiments/train_temporal_cnn_v2_250k.py",
        "experiments/train_temporal_cnn_v2_250k_multiseed_ready.py",
        "experiments/train_temporal_cnn_v2_residual_dilated_250k.py",
    ],
    "hybrid_temporal_tabular_cnn": [
        "experiments/train_hybrid_temporal_tabular_cnn_250k.py",
        "experiments/train_hybrid_temporal_tabular_cnn_250k_multiseed_ready.py",
    ],
    "ensemble_hybrid_dominant": [
        "experiments/run_hybrid_tabular_ensemble_250k.py",
        "experiments/run_hybrid_tabular_ensemble_250k_multiseed_ready.py",
    ],
}

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "top2_accuracy",
    "top3_accuracy",
    "top5_accuracy",
    "mean_confidence",
    "ece",
    "brier_score",
    "negative_log_likelihood",
]

METRIC_ALIASES = {
    "acc": "accuracy",
    "bal_acc": "balanced_accuracy",
    "balanced_acc": "balanced_accuracy",
    "macro-f1": "macro_f1",
    "macro f1": "macro_f1",
    "weighted-f1": "weighted_f1",
    "weighted f1": "weighted_f1",
    "top3": "top3_accuracy",
    "top_3_accuracy": "top3_accuracy",
    "top5": "top5_accuracy",
    "top_5_accuracy": "top5_accuracy",
    "brier": "brier_score",
    "nll": "negative_log_likelihood",
}

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# -----------------------------------------------------------------------------
# Time/log utilities
# -----------------------------------------------------------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    return str(timedelta(seconds=seconds))


def print_flush(message: str = "") -> None:
    print(message, flush=True)


def clean_line(line: str) -> str:
    return ANSI_ESCAPE.sub("", line.rstrip("\n\r"))


# -----------------------------------------------------------------------------
# Plan initialization and validation
# -----------------------------------------------------------------------------

def first_existing_script(candidates: list[str]) -> str | None:
    for rel in candidates:
        if (ROOT_DIR / rel).exists():
            return rel
    return None


def build_default_plan() -> dict[str, Any]:
    models = []

    for model_name in [
        "lightgbm_v4_baseline",
        "temporal_cnn",
        "temporal_cnn_v2",
        "hybrid_temporal_tabular_cnn",
        "ensemble_hybrid_dominant",
    ]:
        script = first_existing_script(SCRIPT_CANDIDATES[model_name])
        if script is None:
            script = f"EDIT_ME_SCRIPT_FOR_{model_name}.py"
            enabled = False
        else:
            enabled = True

        if model_name == "ensemble_hybrid_dominant":
            command = (
                "{python} {script} "
                "--seed {seed} "
                "--input-root {seed_dir} "
                "--output-dir {model_dir}"
            )
            notes = (
                "This command assumes the ensemble script can read seed-specific base-model outputs from --input-root. "
                "If your current ensemble script only reads fixed results folders, adapt it or set this command manually."
            )
            depends_on = [
                "lightgbm_v4_baseline",
                "temporal_cnn",
                "temporal_cnn_v2",
                "hybrid_temporal_tabular_cnn",
            ]
        else:
            command = (
                "{python} {script} "
                "--seed {seed} "
                "--output-dir {model_dir}"
            )
            notes = (
                "This command assumes the training script accepts --seed and --output-dir. "
                "If it does not, edit this command or update the training script."
            )
            depends_on = []

        models.append(
            {
                "name": model_name,
                "enabled": enabled,
                "script": script,
                "command": command,
                "depends_on": depends_on,
                "notes": notes,
                "expected_metric_files": [
                    "*metrics*.csv",
                    "*summary*.csv",
                    "*performance*.csv",
                ],
            }
        )

    return {
        "description": "AstroTrust-AI final 250k multi-seed experiment plan. Edit command templates if scripts use different CLI arguments.",
        "seeds": DEFAULT_SEEDS,
        "python": sys.executable,
        "run_order": [
            "lightgbm_v4_baseline",
            "temporal_cnn",
            "temporal_cnn_v2",
            "hybrid_temporal_tabular_cnn",
            "ensemble_hybrid_dominant",
        ],
        "models": models,
        "environment": {
            "PYTHONHASHSEED": "{seed}",
            "ASTROTRUST_SEED": "{seed}",
        },
        "reproducibility_notes": [
            "For neural models, training scripts should set Python, NumPy, and PyTorch seeds from --seed or ASTROTRUST_SEED.",
            "For LightGBM/XGBoost, training scripts should pass random_state/seed consistently.",
            "For the ensemble, recompute predictions using seed-specific base-model probabilities rather than fixed global folders.",
        ],
    }


def write_plan_template(plan_path: Path, overwrite: bool = False) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and not overwrite:
        print_flush(f"[INFO] Plan already exists: {plan_path}")
        print_flush("[INFO] Use --init-plan --overwrite-plan to regenerate it from the scripts currently present in experiments/.")
        return
    plan = build_default_plan()
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    enabled = [m["name"] for m in plan.get("models", []) if m.get("enabled", False)]
    disabled = [m["name"] for m in plan.get("models", []) if not m.get("enabled", False)]
    print_flush(f"[OK] Wrote editable multi-seed plan: {plan_path}")
    print_flush(f"[OK] Enabled models detected: {enabled}")
    if disabled:
        print_flush(f"[WARN] Disabled models because no candidate script was found: {disabled}")


def load_plan(plan_path: Path) -> dict[str, Any]:
    if not plan_path.exists():
        write_plan_template(plan_path)
        raise FileNotFoundError(
            f"Plan file was created at {plan_path}. Review/edit it, then rerun without --init-plan."
        )
    return json.loads(plan_path.read_text(encoding="utf-8"))


def model_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["name"]: m for m in plan.get("models", [])}


def selected_models(plan: dict[str, Any], models_arg: str | None) -> list[dict[str, Any]]:
    by_name = model_map(plan)
    if models_arg:
        names = [x.strip() for x in models_arg.split(",") if x.strip()]
    else:
        names = plan.get("run_order", list(by_name.keys()))

    selected = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Model {name!r} is not present in plan.")
        item = by_name[name]
        if item.get("enabled", True):
            selected.append(item)
        else:
            print_flush(f"[SKIP] Model disabled in plan: {name}")
    return selected


def validate_plan(plan: dict[str, Any], models: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for item in models:
        script = item.get("script", "")
        if script and "EDIT_ME" not in script:
            script_path = ROOT_DIR / script
            if not script_path.exists():
                warnings.append(f"Script not found for {item['name']}: {script_path}")
        if "{seed}" not in item.get("command", ""):
            warnings.append(f"Command for {item['name']} does not include {{seed}} placeholder.")
        if "{model_dir}" not in item.get("command", ""):
            warnings.append(f"Command for {item['name']} does not include {{model_dir}} placeholder.")
    return warnings


# -----------------------------------------------------------------------------
# Command execution
# -----------------------------------------------------------------------------

def render_template(text: str, context: dict[str, Any]) -> str:
    return text.format(**context)


def build_context(plan: dict[str, Any], seed: int, model: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    seed_dir = output_dir / f"seed_{seed}"
    model_dir = seed_dir / model["name"]
    return {
        "root": str(ROOT_DIR),
        "results": str(RESULTS_DIR),
        "output_dir": str(output_dir),
        "seed_dir": str(seed_dir),
        "model_dir": str(model_dir),
        "seed": seed,
        "model": model["name"],
        "script": str(ROOT_DIR / model.get("script", "")),
        "python": plan.get("python") or sys.executable,
    }


def build_env(plan: dict[str, Any], seed: int, context: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in plan.get("environment", {}).items():
        env[str(key)] = render_template(str(value), context)
    env["ASTROTRUST_MULTI_SEED"] = "1"
    env["ASTROTRUST_CURRENT_MODEL"] = str(context["model"])
    env["ASTROTRUST_OUTPUT_DIR"] = str(context["model_dir"])
    return env


def read_status(status_path: Path) -> dict[str, Any] | None:
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command_live(
    command: str,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    prefix: str,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n\n===== RUN START {now_str()} =====\n")
        log.write(f"Command: {command}\n")
        log.write(f"CWD: {cwd}\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = clean_line(raw_line)
            log.write(line + "\n")
            log.flush()
            print_flush(f"{prefix} {line}")

        return_code = process.wait()
        log.write(f"===== RUN END {now_str()} | return_code={return_code} =====\n")
        log.flush()

    return return_code


# -----------------------------------------------------------------------------
# Metrics aggregation
# -----------------------------------------------------------------------------

def normalize_metric_name(name: str) -> str:
    low = str(name).strip().lower()
    low = low.replace(" ", "_").replace("-", "_")
    return METRIC_ALIASES.get(low, low)


def extract_metrics_from_csv(path: Path) -> dict[str, float]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if df.empty:
        return {}

    out: dict[str, float] = {}
    cols_norm = {normalize_metric_name(c): c for c in df.columns}

    # Common case: one row with metric columns.
    for metric in METRIC_COLUMNS:
        if metric in cols_norm:
            value = pd.to_numeric(df[cols_norm[metric]], errors="coerce").dropna()
            if len(value):
                out[metric] = float(value.iloc[0])

    # Long format: metric/value columns.
    metric_col = None
    value_col = None
    for c in df.columns:
        cn = normalize_metric_name(c)
        if cn in ["metric", "name"]:
            metric_col = c
        if cn in ["value", "estimate", "score"]:
            value_col = c
    if metric_col is not None and value_col is not None:
        for _, row in df.iterrows():
            metric = normalize_metric_name(row[metric_col])
            if metric in METRIC_COLUMNS:
                value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    out[metric] = float(value)

    return out


def parse_metrics_from_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}

    patterns = {
        "accuracy": [r"Accuracy:\s*([0-9]*\.?[0-9]+)", r"accuracy[=:]\s*([0-9]*\.?[0-9]+)"],
        "balanced_accuracy": [r"Balanced accuracy:\s*([0-9]*\.?[0-9]+)", r"balanced_accuracy[=:]\s*([0-9]*\.?[0-9]+)"],
        "macro_f1": [r"Macro-F1:\s*([0-9]*\.?[0-9]+)", r"macro_f1[=:]\s*([0-9]*\.?[0-9]+)"],
        "weighted_f1": [r"Weighted-F1:\s*([0-9]*\.?[0-9]+)", r"weighted_f1[=:]\s*([0-9]*\.?[0-9]+)"],
        "top3_accuracy": [r"top3_accuracy[=:]\s*([0-9]*\.?[0-9]+)", r"Top-3.*?:\s*([0-9]*\.?[0-9]+)"],
        "top5_accuracy": [r"top5_accuracy[=:]\s*([0-9]*\.?[0-9]+)", r"Top-5.*?:\s*([0-9]*\.?[0-9]+)"],
        "ece": [r"ECE:\s*([0-9]*\.?[0-9]+)", r"ece[=:]\s*([0-9]*\.?[0-9]+)"],
        "brier_score": [r"Brier score:\s*([0-9]*\.?[0-9]+)", r"brier_score[=:]\s*([0-9]*\.?[0-9]+)"],
    }

    text = log_path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for metric, pats in patterns.items():
        values = []
        for pat in pats:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                try:
                    values.append(float(m.group(1)))
                except Exception:
                    pass
        if values:
            out[metric] = values[-1]
    return out


def collect_run_metrics(model_dir: Path, log_path: Path, expected_patterns: list[str]) -> tuple[dict[str, float], str]:
    # Prefer CSV metrics saved inside the model output dir.
    candidates: list[Path] = []
    for pattern in expected_patterns:
        candidates.extend(sorted(model_dir.glob(pattern)))
        candidates.extend(sorted(model_dir.rglob(pattern)))

    # Avoid giant or irrelevant files when possible.
    # When several metric files exist in the same model directory, prefer the
    # metric file whose filename explicitly contains the current model name.
    # This prevents ensemble runs from being summarized using a member model
    # such as xgboost_gpu_deeper_test_metrics.csv.
    model_token = model_dir.name.lower()
    scored = []
    for path in candidates:
        low = path.name.lower()
        score = 0
        if model_token in low:
            score += 100
        if low == f"{model_token}_test_metrics.csv":
            score += 50
        if low.endswith("_test_metrics.csv"):
            score += 20
        if "metric" in low:
            score += 4
        if "summary" in low:
            score += 2
        if "prediction" in low:
            score -= 100
        if "comparison" in low:
            score -= 10
        scored.append((score, path))
    scored.sort(reverse=True, key=lambda x: (x[0], str(x[1])))

    for _, path in scored:
        metrics = extract_metrics_from_csv(path)
        if any(m in metrics for m in ["accuracy", "macro_f1", "ece", "brier_score"]):
            return metrics, str(path)

    # Fallback: parse logs.
    metrics = parse_metrics_from_log(log_path)
    if metrics:
        return metrics, str(log_path)

    return {}, ""


def aggregate_results(output_dir: Path, plan: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    by_name = model_map(plan)

    for seed_dir in sorted(output_dir.glob("seed_*")):
        try:
            seed = int(seed_dir.name.replace("seed_", ""))
        except Exception:
            continue

        for model_name, model in by_name.items():
            model_dir = seed_dir / model_name
            status_path = model_dir / "run_status.json"
            status = read_status(status_path) or {}
            log_path = model_dir / "run.log"
            patterns = model.get("expected_metric_files", ["*metrics*.csv", "*summary*.csv"])
            metrics, source = collect_run_metrics(model_dir, log_path, patterns)

            row = {
                "seed": seed,
                "model": model_name,
                "status": status.get("status", "unknown"),
                "return_code": status.get("return_code"),
                "duration_seconds": status.get("duration_seconds"),
                "metrics_source": source,
            }
            for metric in METRIC_COLUMNS:
                row[metric] = metrics.get(metric, np.nan)
            rows.append(row)

    runs_df = pd.DataFrame(rows)
    if runs_df.empty:
        return runs_df, pd.DataFrame()

    summary_rows = []
    for model_name, group in runs_df.groupby("model"):
        completed = group[group["status"] == "completed"]
        row = {
            "model": model_name,
            "n_runs_total": int(len(group)),
            "n_completed": int(len(completed)),
            "n_failed_or_missing": int(len(group) - len(completed)),
            "mean_duration_seconds": float(pd.to_numeric(completed["duration_seconds"], errors="coerce").mean()) if len(completed) else np.nan,
        }
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(completed[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("model")
    return runs_df.sort_values(["model", "seed"]), summary_df


def write_markdown_summary(output_dir: Path, runs_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    md = "# AstroTrust-AI multi-seed stability summary\n\n"
    md += f"Generated at: `{now_str()}`\n\n"
    md += "## Per-run results\n\n"
    if runs_df.empty:
        md += "_No runs found._\n\n"
    else:
        keep = ["seed", "model", "status", "duration_seconds"] + METRIC_COLUMNS
        keep = [c for c in keep if c in runs_df.columns]
        md += runs_df[keep].to_markdown(index=False)
        md += "\n\n"

    md += "## Mean ± standard deviation across seeds\n\n"
    if summary_df.empty:
        md += "_No summary available._\n\n"
    else:
        compact_rows = []
        for _, row in summary_df.iterrows():
            compact = {
                "model": row["model"],
                "n_completed": row["n_completed"],
            }
            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "top3_accuracy",
                "top5_accuracy",
                "ece",
                "brier_score",
                "negative_log_likelihood",
            ]:
                mean = row.get(f"{metric}_mean")
                std = row.get(f"{metric}_std")
                if pd.notna(mean):
                    if pd.notna(std):
                        compact[metric] = f"{mean:.4f} ± {std:.4f}"
                    else:
                        compact[metric] = f"{mean:.4f} ± NA"
            compact_rows.append(compact)
        md += pd.DataFrame(compact_rows).to_markdown(index=False)
        md += "\n\n"

    md += "## Interpretation note\n\n"
    md += (
        "This table estimates training-run stochasticity across random seeds. "
        "It should be reported separately from bootstrap confidence intervals, which estimate test-set sampling uncertainty. "
        "For the ensemble, ensure that each seed-specific ensemble was recomputed from the corresponding seed-specific base-model probabilities.\n"
    )
    (output_dir / "multiseed_summary.md").write_text(md, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def build_run_queue(seeds: list[int], models: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return [(seed, model) for seed in seeds for model in models]


def print_eta(done_durations: list[float], completed: int, total: int, global_start: float) -> None:
    elapsed = time.time() - global_start
    remaining = total - completed
    avg = float(np.mean(done_durations)) if done_durations else None
    eta = remaining * avg if avg is not None else None
    print_flush(
        f"[PROGRESS] completed={completed}/{total} | "
        f"elapsed={format_duration(elapsed)} | "
        f"avg/run={format_duration(avg)} | "
        f"ETA={format_duration(eta)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final AstroTrust-AI 250k models under multiple random seeds with live logs and ETA.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--init-plan", action="store_true", help="Write the editable JSON plan and exit.")
    parser.add_argument("--overwrite-plan", action="store_true", help="Overwrite the JSON plan when using --init-plan.")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds. Defaults to plan seeds.")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model names to run. Defaults to plan run_order.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but do not execute.")
    parser.add_argument("--resume", action="store_true", help="Skip completed runs.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed run.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing results and exit.")
    parser.add_argument("--allow-plan-warnings", action="store_true", help="Continue even if plan validation emits warnings.")
    args = parser.parse_args()

    if args.init_plan:
        write_plan_template(args.plan, overwrite=args.overwrite_plan)
        return

    plan = load_plan(args.plan)
    models = selected_models(plan, args.models or None)

    warnings = validate_plan(plan, models)
    if warnings:
        print_flush("[WARN] Plan validation warnings:")
        for warning in warnings:
            print_flush(f"  - {warning}")
        if not args.allow_plan_warnings:
            raise SystemExit("Fix plan warnings or rerun with --allow-plan-warnings.")

    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = [int(x) for x in plan.get("seeds", DEFAULT_SEEDS)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        runs_df, summary_df = aggregate_results(args.output_dir, plan)
        runs_df.to_csv(args.output_dir / "multiseed_runs.csv", index=False)
        summary_df.to_csv(args.output_dir / "multiseed_summary_mean_std.csv", index=False)
        write_markdown_summary(args.output_dir, runs_df, summary_df)
        print_flush(f"[OK] Aggregated results saved to: {args.output_dir}")
        return

    queue = build_run_queue(seeds, models)
    total = len(queue)

    print_flush("=" * 80)
    print_flush("AstroTrust-AI final 250k multi-seed orchestrator")
    print_flush("=" * 80)
    print_flush(f"Start time: {now_str()}")
    print_flush(f"Output directory: {args.output_dir}")
    print_flush(f"Plan: {args.plan}")
    print_flush(f"Seeds: {seeds}")
    print_flush(f"Models: {[m['name'] for m in models]}")
    print_flush(f"Total runs: {total}")
    print_flush("=" * 80)

    if args.dry_run:
        print_flush("[DRY RUN] Commands that would be executed:")
        for seed, model in queue:
            context = build_context(plan, seed, model, args.output_dir)
            command = render_template(model["command"], context)
            print_flush(f"seed={seed} | model={model['name']} | command={command}")
        return

    global_start = time.time()
    completed_count = 0
    done_durations: list[float] = []
    failures = []

    status_rows = []

    for run_index, (seed, model) in enumerate(queue, start=1):
        context = build_context(plan, seed, model, args.output_dir)
        model_dir = Path(context["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_dir / "run.log"
        status_path = model_dir / "run_status.json"

        existing_status = read_status(status_path)
        if args.resume and existing_status and existing_status.get("status") == "completed":
            completed_count += 1
            duration = float(existing_status.get("duration_seconds", 0.0) or 0.0)
            if duration > 0:
                done_durations.append(duration)
            print_flush(f"[SKIP] completed already | seed={seed} | model={model['name']}")
            print_eta(done_durations, completed_count, total, global_start)
            continue

        command = render_template(model["command"], context)
        env = build_env(plan, seed, context)

        print_flush("\n" + "-" * 80)
        print_flush(f"[RUN {run_index}/{total}] seed={seed} | model={model['name']}")
        print_flush(f"[TIME] started={now_str()}")
        print_flush(f"[DIR] {model_dir}")
        print_flush(f"[LOG] {log_path}")
        print_flush(f"[CMD] {command}")
        print_flush("-" * 80)

        start = time.time()
        write_status(status_path, {
            "status": "running",
            "seed": seed,
            "model": model["name"],
            "command": command,
            "started_at": now_str(),
        })

        prefix = f"[seed={seed} model={model['name']} run={run_index}/{total}]"
        return_code = run_command_live(command, ROOT_DIR, env, log_path, prefix)
        duration = time.time() - start
        done_durations.append(duration)
        completed_count += 1

        status = "completed" if return_code == 0 else "failed"
        status_payload = {
            "status": status,
            "seed": seed,
            "model": model["name"],
            "command": command,
            "started_at": datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": now_str(),
            "duration_seconds": duration,
            "return_code": return_code,
            "log_path": str(log_path),
            "model_dir": str(model_dir),
        }
        write_status(status_path, status_payload)
        status_rows.append(status_payload)

        print_flush(f"[DONE] seed={seed} | model={model['name']} | status={status} | duration={format_duration(duration)}")
        print_eta(done_durations, completed_count, total, global_start)

        if return_code != 0:
            failures.append((seed, model["name"], return_code))
            if args.fail_fast:
                print_flush("[STOP] fail-fast enabled.")
                break

        # Save aggregate after each run so progress is never lost.
        runs_df, summary_df = aggregate_results(args.output_dir, plan)
        runs_df.to_csv(args.output_dir / "multiseed_runs.csv", index=False)
        summary_df.to_csv(args.output_dir / "multiseed_summary_mean_std.csv", index=False)
        write_markdown_summary(args.output_dir, runs_df, summary_df)

    # Final aggregation.
    runs_df, summary_df = aggregate_results(args.output_dir, plan)
    runs_df.to_csv(args.output_dir / "multiseed_runs.csv", index=False)
    summary_df.to_csv(args.output_dir / "multiseed_summary_mean_std.csv", index=False)
    write_markdown_summary(args.output_dir, runs_df, summary_df)

    print_flush("\n" + "=" * 80)
    print_flush("Multi-seed run finished")
    print_flush("=" * 80)
    print_flush(f"End time: {now_str()}")
    print_flush(f"Total elapsed: {format_duration(time.time() - global_start)}")
    print_flush(f"Failures: {len(failures)}")
    if failures:
        for seed, model_name, return_code in failures:
            print_flush(f"  - seed={seed} model={model_name} return_code={return_code}")
    print_flush("\nSaved:")
    print_flush(f"- {args.output_dir / 'multiseed_runs.csv'}")
    print_flush(f"- {args.output_dir / 'multiseed_summary_mean_std.csv'}")
    print_flush(f"- {args.output_dir / 'multiseed_summary.md'}")

    if not summary_df.empty:
        print_flush("\nMean ± std summary:")
        compact_rows = []
        for _, row in summary_df.iterrows():
            compact = {
                "model": row["model"],
                "n_completed": row["n_completed"],
            }
            for metric in ["accuracy", "macro_f1", "top3_accuracy", "top5_accuracy", "ece", "brier_score"]:
                mean = row.get(f"{metric}_mean")
                std = row.get(f"{metric}_std")
                if pd.notna(mean):
                    compact[metric] = f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f} ± NA"
            compact_rows.append(compact)
        print_flush(pd.DataFrame(compact_rows).to_string(index=False))


if __name__ == "__main__":
    main()
