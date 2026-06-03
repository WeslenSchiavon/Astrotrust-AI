#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Early-aware broker-like partial light-curve evaluation for AstroTrust-AI.

This is the practical publication-oriented experiment after the first stress test.

Core idea:
  Model A: full-trained V4 LightGBM
    - trained only on full_curve_reference features from training objects
    - evaluated on full and partial scenarios

  Model B: early-aware V4 LightGBM
    - trained on multiple representations of the same training objects:
      full curves + temporally truncated curves
    - evaluated on the same held-out object IDs for each scenario

Why this is stronger:
  The previous stress test trained on complete curves and tested on partial
  curves, creating a severe distribution shift. This script trains an
  early-aware classifier that explicitly sees partial representations during
  training while preserving object-level split integrity.

Inputs expected from Step 2:
  data/processed/broker_like_partial_features/
    features_v4_temporal_shape_full_curve_reference.parquet
    features_v4_temporal_shape_first_3_points.parquet
    features_v4_temporal_shape_first_5_points.parquet
    ...

Outputs:
  results/broker_like_early_aware_v4/
    early_aware_classification_metrics.csv
    early_aware_followup_metrics.csv
    early_aware_degradation_vs_full.csv
    early_aware_gain_over_full_trained.csv
    early_aware_training_manifest.csv
    early_aware_broker_like_summary.md
    predictions/full_trained/{scenario}_predictions.csv
    predictions/early_aware/{scenario}_predictions.csv

  results/final_publication/broker_like_realism/
    copies of the main CSV/MD outputs

Run:
  python experiments/run_broker_like_early_aware_v4_evaluation.py

Recommended first smoke test:
  python experiments/run_broker_like_early_aware_v4_evaluation.py ^
    --model random_forest ^
    --max-train-rows 100000 ^
    --max-test-objects 10000 ^
    --scenarios full_curve_reference,first_5_points,first_20_points,window_7_days,window_30_days

Recommended publication run:
  python experiments/run_broker_like_early_aware_v4_evaluation.py

Notes:
  - This is still a V4 temporal-shape/tabular experiment, not the full hybrid
    neural ensemble unless you later rerun the neural branch.
  - It is designed to strengthen the broker-like realism claim at low cost.
  - It does not modify final_publication_summary.md.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_FEATURE_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_features"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "broker_like_early_aware_v4"
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

DEFAULT_TRAIN_SCENARIOS = [
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

DEFAULT_RARE_LABELS = "4,5,6,7,11,29,30,31"
EPS = 1e-12


def parse_list(value: str, default: list[str]) -> list[str]:
    if value is None or value.strip().lower() in {"", "all"}:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_set(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


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
    raise FileNotFoundError(
        f"No feature file found for scenario `{scenario}`. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def scenario_metadata(scenario: str) -> dict[str, float]:
    """
    Numeric stage descriptors known at inference time.

    These descriptors avoid using an arbitrary string category as a model input.
    They encode whether the representation is full, first-N, or window-days,
    plus the nominal N/day value when applicable.
    """
    meta = {
        "stage_is_full": 0.0,
        "stage_is_first_points": 0.0,
        "stage_is_window_days": 0.0,
        "stage_first_n": 0.0,
        "stage_window_days": 0.0,
        "stage_log_first_n": 0.0,
        "stage_log_window_days": 0.0,
    }

    if scenario == "full_curve_reference":
        meta["stage_is_full"] = 1.0
        return meta

    if scenario.startswith("first_") and scenario.endswith("_points"):
        n = float(scenario.replace("first_", "").replace("_points", ""))
        meta["stage_is_first_points"] = 1.0
        meta["stage_first_n"] = n
        meta["stage_log_first_n"] = math.log1p(n)
        return meta

    if scenario.startswith("window_") and scenario.endswith("_days"):
        d = float(scenario.replace("window_", "").replace("_days", ""))
        meta["stage_is_window_days"] = 1.0
        meta["stage_window_days"] = d
        meta["stage_log_window_days"] = math.log1p(d)
        return meta

    return meta


def add_stage_features(df: pd.DataFrame, scenario: str, include_stage_features: bool) -> pd.DataFrame:
    df = df.copy()
    df["scenario_name"] = scenario

    if include_stage_features:
        meta = scenario_metadata(scenario)
        for key, value in meta.items():
            df[key] = value

    return df


def prepare_xy(df: pd.DataFrame, feature_columns: list[str] | None = None):
    required = {"object_id", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    object_id = df["object_id"].to_numpy()
    y = df["label"].astype(int).to_numpy()

    if feature_columns is None:
        feature_columns = [
            c for c in df.columns
            if c not in {"object_id", "label", "scenario_name"}
        ]

    X = df.reindex(columns=feature_columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # LightGBM and sklearn are happier with float32 for large tables.
    X = X.astype(np.float32)

    return object_id, y, X, feature_columns


def make_model(args: argparse.Namespace, n_classes: int):
    if args.model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError(
                "LightGBM is not available. Install it with `pip install lightgbm`, "
                "or run with `--model random_forest` for a quick smoke test."
            ) from exc

        return LGBMClassifier(
            objective="multiclass",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            class_weight="balanced" if args.class_weight_balanced else None,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=-1,
        )

    if args.model == "random_forest":
        return RandomForestClassifier(
            n_estimators=min(args.n_estimators, 300),
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced_subsample" if args.class_weight_balanced else None,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    raise ValueError(f"Unknown model: {args.model}")


def align_probabilities(probabilities: np.ndarray, model_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(all_classes)), dtype=np.float64)
    class_to_index = {int(c): i for i, c in enumerate(all_classes)}
    for local_idx, cls in enumerate(model_classes):
        if int(cls) in class_to_index:
            aligned[:, class_to_index[int(cls)]] = probabilities[:, local_idx]
    aligned = aligned / np.maximum(aligned.sum(axis=1, keepdims=True), EPS)
    return aligned


def topk_labels(probabilities: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, probabilities.shape[1])
    idx = np.argsort(probabilities, axis=1)[:, -k_eff:][:, ::-1]
    return classes[idx]


def topk_accuracy_from_prediction_frame(pred: pd.DataFrame, k: int) -> float:
    cols = [f"top{i}_label" for i in range(1, k + 1) if f"top{i}_label" in pred.columns]
    if not cols:
        return np.nan
    y_true = pred["true_label"].astype(int).to_numpy()
    top = pred[cols].astype(int).to_numpy()
    return float(np.mean([yt in row for yt, row in zip(y_true, top)]))


def fit_novelty_reference(X_train_full: pd.DataFrame) -> dict:
    values = X_train_full.to_numpy(dtype=np.float64)
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25
    keep = np.isfinite(med) & np.isfinite(iqr) & (np.abs(iqr) > EPS)

    if not np.any(keep):
        raise RuntimeError("No valid features remained for novelty reference.")

    z = (values[:, keep] - med[keep]) / (iqr[keep] + EPS)
    d_train = np.sqrt(np.mean(z * z, axis=1))

    return {
        "median": med,
        "iqr": iqr,
        "keep": keep,
        "q05": float(np.nanpercentile(d_train, 5)),
        "q95": float(np.nanpercentile(d_train, 95)),
        "n_retained_features": int(np.sum(keep)),
    }


def compute_novelty(X: pd.DataFrame, ref: dict) -> np.ndarray:
    values = X.to_numpy(dtype=np.float64)
    keep = ref["keep"]
    z = (values[:, keep] - ref["median"][keep]) / (ref["iqr"][keep] + EPS)
    d = np.sqrt(np.mean(z * z, axis=1))
    novelty = (d - ref["q05"]) / max(ref["q95"] - ref["q05"], EPS)
    return np.clip(novelty, 0.0, 1.0)


def compute_rarity(probabilities: np.ndarray, classes: np.ndarray, rare_labels: set[int]) -> np.ndarray:
    rare_cols = [i for i, c in enumerate(classes) if int(c) in rare_labels]
    if not rare_cols:
        return np.zeros(probabilities.shape[0], dtype=float)
    return probabilities[:, rare_cols].sum(axis=1)


def build_prediction_frame(
    object_id: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    novelty: np.ndarray,
    rare_labels: set[int],
    save_probabilities: bool,
) -> pd.DataFrame:
    y_pred = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    uncertainty = 1.0 - confidence
    rarity = compute_rarity(probabilities, classes, rare_labels)
    priority = 0.5 * novelty + 0.5 * rarity
    top5 = topk_labels(probabilities, classes, 5)

    out = pd.DataFrame({
        "object_id": object_id,
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "novelty": novelty,
        "rarity_score": rarity,
        "priority_novelty_rarity": priority,
        "true_is_rare": np.array([int(v) in rare_labels for v in y_true], dtype=bool),
    })

    for k in range(top5.shape[1]):
        out[f"top{k + 1}_label"] = top5[:, k]

    if save_probabilities:
        for i, cls in enumerate(classes):
            out[f"prob_{int(cls)}"] = probabilities[:, i]

    return out


def score_policies(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    novelty = pred["novelty"].to_numpy(dtype=float)
    rarity = pred["rarity_score"].to_numpy(dtype=float)
    uncertainty = pred["uncertainty"].to_numpy(dtype=float)
    return {
        "novelty_rarity": 0.5 * novelty + 0.5 * rarity,
        "rarity_only": rarity,
        "uncertainty_only": uncertainty,
        "previous_discovery": 0.1 * uncertainty + 0.8 * novelty + 0.1 * rarity,
        "fixed_discovery": 0.4 * uncertainty + 0.4 * novelty + 0.2 * rarity,
    }


def classification_metrics(pred: pd.DataFrame, model_name: str, comparison_set: str, scenario: str) -> dict:
    y_true = pred["true_label"].astype(int).to_numpy()
    y_pred = pred["predicted_label"].astype(int).to_numpy()
    correct = pred["correct"].astype(bool).to_numpy(dtype=float)
    rare = pred["true_is_rare"].astype(bool).to_numpy(dtype=float)

    out = {
        "model_name": model_name,
        "comparison_set": comparison_set,
        "scenario": scenario,
        "n_objects": int(len(pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": topk_accuracy_from_prediction_frame(pred, 2),
        "top3_accuracy": topk_accuracy_from_prediction_frame(pred, 3),
        "top5_accuracy": topk_accuracy_from_prediction_frame(pred, 5),
        "mean_correct": float(np.mean(correct)),
        "std_correct": float(np.std(correct, ddof=1)) if len(correct) > 1 else 0.0,
        "se_accuracy": float(np.std(correct, ddof=1) / np.sqrt(len(correct))) if len(correct) > 1 else 0.0,
        "mean_confidence": float(pred["confidence"].mean()),
        "std_confidence": float(pred["confidence"].std(ddof=1)) if len(pred) > 1 else 0.0,
        "mean_uncertainty": float(pred["uncertainty"].mean()),
        "std_uncertainty": float(pred["uncertainty"].std(ddof=1)) if len(pred) > 1 else 0.0,
        "mean_novelty": float(pred["novelty"].mean()),
        "std_novelty": float(pred["novelty"].std(ddof=1)) if len(pred) > 1 else 0.0,
        "mean_rarity_score": float(pred["rarity_score"].mean()),
        "std_rarity_score": float(pred["rarity_score"].std(ddof=1)) if len(pred) > 1 else 0.0,
        "rare_true_rate": float(np.mean(rare)),
        "std_true_is_rare": float(np.std(rare, ddof=1)) if len(rare) > 1 else 0.0,
    }
    return out


def followup_metrics(pred: pd.DataFrame, model_name: str, comparison_set: str, scenario: str, budgets: Iterable[float]) -> pd.DataFrame:
    rare = pred["true_is_rare"].astype(bool).to_numpy()
    baseline = float(np.mean(rare))
    policies = score_policies(pred)
    rows = []

    for policy, score in policies.items():
        order = np.argsort(score)[::-1]

        for budget in budgets:
            n_select = max(1, int(math.ceil(len(pred) * budget)))
            selected = pred.iloc[order[:n_select]]
            selected_rare = selected["true_is_rare"].astype(bool).to_numpy(dtype=float)
            selected_correct = selected["correct"].astype(bool).to_numpy(dtype=float)
            rare_rate = float(np.mean(selected_rare))
            enrichment = rare_rate / baseline if baseline > 0 else np.nan

            rows.append({
                "model_name": model_name,
                "comparison_set": comparison_set,
                "scenario": scenario,
                "policy": policy,
                "budget_fraction": float(budget),
                "n_available": int(len(pred)),
                "n_selected": int(n_select),
                "baseline_rare_rate": baseline,
                "rare_rate": rare_rate,
                "std_true_is_rare_selected": float(np.std(selected_rare, ddof=1)) if len(selected_rare) > 1 else 0.0,
                "rare_enrichment": enrichment,
                "mean_confidence": float(selected["confidence"].mean()),
                "std_confidence": float(selected["confidence"].std(ddof=1)) if len(selected) > 1 else 0.0,
                "mean_uncertainty": float(selected["uncertainty"].mean()),
                "std_uncertainty": float(selected["uncertainty"].std(ddof=1)) if len(selected) > 1 else 0.0,
                "mean_novelty": float(selected["novelty"].mean()),
                "std_novelty": float(selected["novelty"].std(ddof=1)) if len(selected) > 1 else 0.0,
                "mean_rarity_score": float(selected["rarity_score"].mean()),
                "std_rarity_score": float(selected["rarity_score"].std(ddof=1)) if len(selected) > 1 else 0.0,
                "mean_correct": float(np.mean(selected_correct)),
                "std_correct": float(np.std(selected_correct, ddof=1)) if len(selected_correct) > 1 else 0.0,
            })

    return pd.DataFrame(rows)


def sample_training_rows(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df

    rng = np.random.default_rng(random_state)
    pieces = []
    scenario_counts = df["scenario_name"].value_counts()
    quota = max(1, max_rows // max(1, len(scenario_counts)))

    for scenario, _ in scenario_counts.items():
        sub = df[df["scenario_name"] == scenario]
        n = min(len(sub), quota)
        pieces.append(sub.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))

    sampled = pd.concat(pieces, ignore_index=True)

    if len(sampled) < max_rows:
        remaining = df.drop(index=sampled.index, errors="ignore")
        n_extra = min(len(remaining), max_rows - len(sampled))
        if n_extra > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=n_extra, random_state=random_state)],
                ignore_index=True,
            )

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)

    return sampled.reset_index(drop=True)


def compute_degradation(classification: pd.DataFrame, followup: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model_name, comparison_set), subset in classification.groupby(["model_name", "comparison_set"]):
        full = subset[subset["scenario"] == "full_curve_reference"]
        if full.empty:
            continue
        ref = full.iloc[0].to_dict()

        for row in subset.itertuples(index=False):
            d = row._asdict()
            if d["scenario"] == "full_curve_reference":
                continue

            out = {
                "model_name": model_name,
                "comparison_set": comparison_set,
                "scenario": d["scenario"],
                "n_objects": d["n_objects"],
            }

            for metric in [
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "top3_accuracy", "top5_accuracy",
                "mean_confidence", "mean_uncertainty", "mean_novelty",
                "rare_true_rate"
            ]:
                if metric in ref and metric in d and pd.notna(ref[metric]) and pd.notna(d[metric]):
                    out[f"delta_{metric}"] = float(d[metric] - ref[metric])

            rows.append(out)

    degradation = pd.DataFrame(rows)

    if followup.empty or degradation.empty:
        return degradation

    main = followup[
        (followup["policy"] == "novelty_rarity")
        & (np.isclose(followup["budget_fraction"], 0.05))
    ].copy()

    enrich_rows = []
    for (model_name, comparison_set), subset in main.groupby(["model_name", "comparison_set"]):
        full = subset[subset["scenario"] == "full_curve_reference"]
        if full.empty:
            continue
        ref_enrich = float(full.iloc[0]["rare_enrichment"])

        for row in subset[subset["scenario"] != "full_curve_reference"].itertuples(index=False):
            enrich_rows.append({
                "model_name": model_name,
                "comparison_set": comparison_set,
                "scenario": row.scenario,
                "delta_rare_enrichment_5pct_novelty_rarity": float(row.rare_enrichment) - ref_enrich,
            })

    if enrich_rows:
        degradation = degradation.merge(
            pd.DataFrame(enrich_rows),
            on=["model_name", "comparison_set", "scenario"],
            how="left",
        )

    return degradation


def compute_gain_over_full_trained(classification: pd.DataFrame, followup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_model_name = "full_trained"
    early_model_name = "early_aware"

    for comparison_set in sorted(classification["comparison_set"].unique()):
        base = classification[
            (classification["comparison_set"] == comparison_set)
            & (classification["model_name"] == full_model_name)
        ]
        early = classification[
            (classification["comparison_set"] == comparison_set)
            & (classification["model_name"] == early_model_name)
        ]

        merged = early.merge(
            base,
            on=["comparison_set", "scenario"],
            suffixes=("_early", "_fulltrained"),
        )

        for row in merged.itertuples(index=False):
            d = row._asdict()
            out = {
                "comparison_set": comparison_set,
                "scenario": d["scenario"],
                "n_objects_early": d.get("n_objects_early"),
                "n_objects_fulltrained": d.get("n_objects_fulltrained"),
            }

            for metric in [
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "top3_accuracy", "top5_accuracy",
                "mean_uncertainty", "mean_novelty", "rare_true_rate"
            ]:
                a = d.get(f"{metric}_early")
                b = d.get(f"{metric}_fulltrained")
                if a is not None and b is not None and pd.notna(a) and pd.notna(b):
                    out[f"gain_{metric}"] = float(a - b)

            rows.append(out)

    gain = pd.DataFrame(rows)

    if followup.empty:
        return gain

    fmain = followup[
        (followup["policy"] == "novelty_rarity")
        & (np.isclose(followup["budget_fraction"], 0.05))
    ].copy()

    base = fmain[fmain["model_name"] == full_model_name]
    early = fmain[fmain["model_name"] == early_model_name]

    merged = early.merge(
        base,
        on=["comparison_set", "scenario", "policy", "budget_fraction"],
        suffixes=("_early", "_fulltrained"),
    )

    enrich_rows = []
    for row in merged.itertuples(index=False):
        d = row._asdict()
        enrich_rows.append({
            "comparison_set": d["comparison_set"],
            "scenario": d["scenario"],
            "gain_rare_enrichment_5pct_novelty_rarity": float(d["rare_enrichment_early"] - d["rare_enrichment_fulltrained"]),
            "gain_rare_rate_5pct_novelty_rarity": float(d["rare_rate_early"] - d["rare_rate_fulltrained"]),
        })

    if enrich_rows and not gain.empty:
        gain = gain.merge(pd.DataFrame(enrich_rows), on=["comparison_set", "scenario"], how="left")

    return gain


def write_summary(
    path: Path,
    classification: pd.DataFrame,
    followup: pd.DataFrame,
    degradation: pd.DataFrame,
    gain: pd.DataFrame,
    training_manifest: pd.DataFrame,
    split_info: dict,
    args: argparse.Namespace,
    novelty_ref: dict,
) -> None:
    lines = [
        "# Early-aware V4 broker-like partial light-curve evaluation",
        "",
        "## Protocol",
        "",
        "This experiment compares a full-curve-trained V4 probabilistic model against an early-aware V4 model. "
        "The full-trained model sees only complete light-curve features during training. "
        "The early-aware model is trained on multiple representations of the training objects, including complete and temporally truncated light curves. "
        "Both models are evaluated on the same held-out object IDs in each scenario.",
        "",
        f"- Base estimator: `{args.model}`",
        f"- Random state: `{args.random_state}`",
        f"- Train objects: `{split_info['n_train']}`",
        f"- Validation objects: `{split_info['n_validation']}`",
        f"- Held-out test objects: `{split_info['n_test']}`",
        f"- Early-aware training scenarios: `{','.join(parse_list(args.train_scenarios, DEFAULT_TRAIN_SCENARIOS))}`",
        f"- Early-aware training rows after optional cap: `{split_info['n_early_training_rows']}`",
        f"- Novelty retained features: `{novelty_ref['n_retained_features']}`",
        f"- Rare labels: `{args.rare_labels}`",
        "",
        "## Training manifest",
        "",
        training_manifest.to_markdown(index=False),
        "",
        "## Classification metrics",
        "",
    ]

    display_cols = [
        "model_name", "comparison_set", "scenario", "n_objects",
        "accuracy", "se_accuracy", "macro_f1", "top3_accuracy", "top5_accuracy",
        "mean_uncertainty", "std_uncertainty", "mean_novelty", "std_novelty",
        "rare_true_rate",
    ]
    display_cols = [c for c in display_cols if c in classification.columns]
    lines.append(classification[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    if not gain.empty:
        lines += [
            "## Early-aware gain over full-trained model",
            "",
        ]
        gain_cols = [
            "comparison_set", "scenario",
            "gain_accuracy", "gain_macro_f1", "gain_top3_accuracy", "gain_top5_accuracy",
            "gain_mean_uncertainty", "gain_rare_enrichment_5pct_novelty_rarity",
        ]
        gain_cols = [c for c in gain_cols if c in gain.columns]
        lines.append(gain[gain_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    if not followup.empty:
        main = followup[
            (followup["policy"] == "novelty_rarity")
            & (np.isclose(followup["budget_fraction"], 0.05))
        ].copy()

        lines += [
            "## Follow-up prioritization at 5% budget",
            "",
        ]
        cols = [
            "model_name", "comparison_set", "scenario", "n_available", "n_selected",
            "baseline_rare_rate", "rare_rate", "std_true_is_rare_selected",
            "rare_enrichment", "mean_uncertainty", "std_uncertainty",
            "mean_novelty", "std_novelty", "mean_correct", "std_correct",
        ]
        cols = [c for c in cols if c in main.columns]
        lines.append(main[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines += [
        "## Suggested manuscript text",
        "",
        "```latex",
        r"\paragraph{Early-aware partial-light-curve training.}",
        r"To avoid evaluating a full-curve-trained classifier under an artificial distribution shift, we trained an early-aware V4 temporal-shape model using multiple representations of the training objects. For each training object, complete and temporally truncated light-curve feature tables were included as alternative alert-stage representations, while all train--validation--test splits remained object-disjoint. The fitted model was then evaluated separately on held-out full and partial scenarios. This protocol more closely reflects a broker-like setting in which the same transient may be inspected at different stages of photometric availability.",
        "```",
        "",
        "## Interpretation caveat",
        "",
        "This is a low-cost V4 temporal-shape/tabular early-aware experiment. "
        "It strengthens the broker-like realism analysis, but it should not be described as retraining the full hybrid neural ensemble unless that branch is explicitly rerun.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_to_final(path: Path, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, final_dir / path.name)


def run_experiment(args: argparse.Namespace) -> None:
    feature_dir = args.feature_dir
    results_dir = args.results_dir
    final_dir = args.final_dir
    pred_dir = results_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    scenarios = parse_list(args.scenarios, DEFAULT_SCENARIOS)
    train_scenarios = parse_list(args.train_scenarios, DEFAULT_TRAIN_SCENARIOS)
    budgets = parse_float_list(args.budgets)
    rare_labels = parse_int_set(args.rare_labels)

    print("[LOAD] Full reference features")
    full_df = read_table(find_feature_path(feature_dir, "full_curve_reference"))
    full_df = add_stage_features(full_df, "full_curve_reference", args.include_stage_features)

    if args.max_objects is not None and args.max_objects > 0:
        full_df = full_df.head(args.max_objects).copy()

    object_id_full, y_full, X_full, feature_columns = prepare_xy(full_df)
    all_classes = np.array(sorted(np.unique(y_full)), dtype=int)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(y_full)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_full,
    )

    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=y_full[train_val_idx],
    )

    if args.max_train_objects is not None and args.max_train_objects > 0:
        train_idx = train_idx[: args.max_train_objects]
    if args.max_test_objects is not None and args.max_test_objects > 0:
        test_idx = test_idx[: args.max_test_objects]

    train_ids = set(object_id_full[train_idx])
    validation_ids = set(object_id_full[validation_idx])
    test_ids = set(object_id_full[test_idx])

    print("[SPLIT]")
    print(f"  train objects:      {len(train_ids)}")
    print(f"  validation objects: {len(validation_ids)}")
    print(f"  test objects:       {len(test_ids)}")
    print(f"  features:           {len(feature_columns)}")
    print(f"  classes:            {len(all_classes)}")

    # Novelty is anchored on full-curve training-domain representation.
    novelty_ref = fit_novelty_reference(X_full.iloc[train_idx])

    # -------------------------
    # Train full-trained model.
    # -------------------------
    X_train_full = X_full.iloc[train_idx]
    y_train_full = y_full[train_idx]

    full_model = make_model(args, n_classes=len(all_classes))
    print("\n[TRAIN] full_trained model on full-curve features only...")
    t0 = time.time()
    full_model.fit(X_train_full, y_train_full)
    full_train_minutes = (time.time() - t0) / 60.0
    print(f"[OK] full_trained runtime: {full_train_minutes:.2f} min")

    # --------------------------
    # Build early-aware training.
    # --------------------------
    early_train_pieces = []
    manifest_rows = []

    for scenario in train_scenarios:
        path = find_feature_path(feature_dir, scenario)
        print(f"[LOAD TRAIN SCENARIO] {scenario}: {path}")

        df = read_table(path)
        df = add_stage_features(df, scenario, args.include_stage_features)

        if args.max_objects is not None and args.max_objects > 0:
            df = df.head(args.max_objects).copy()

        before = len(df)
        df = df[df["object_id"].isin(train_ids)].copy()
        after = len(df)

        early_train_pieces.append(df)

        manifest_rows.append({
            "scenario": scenario,
            "source_path": safe_relative(path),
            "rows_before_train_filter": int(before),
            "rows_used_before_cap": int(after),
        })

    early_train_df = pd.concat(early_train_pieces, ignore_index=True)
    before_cap = len(early_train_df)
    early_train_df = sample_training_rows(early_train_df, args.max_train_rows, args.random_state)
    after_cap = len(early_train_df)

    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        cap_ratio = after_cap / max(before_cap, 1)
        manifest["approx_rows_used_after_global_cap"] = (manifest["rows_used_before_cap"] * cap_ratio).round().astype(int)

    _, y_train_early, X_train_early, _ = prepare_xy(early_train_df, feature_columns=feature_columns)

    early_model = make_model(args, n_classes=len(all_classes))
    print("\n[TRAIN] early_aware model on stacked full + partial training features...")
    print(f"[TRAIN] early-aware rows before cap: {before_cap}")
    print(f"[TRAIN] early-aware rows after cap:  {after_cap}")
    t0 = time.time()
    early_model.fit(X_train_early, y_train_early)
    early_train_minutes = (time.time() - t0) / 60.0
    print(f"[OK] early_aware runtime: {early_train_minutes:.2f} min")

    # Save models if joblib is available.
    try:
        import joblib
        joblib.dump(full_model, results_dir / f"full_trained_{args.model}.joblib")
        joblib.dump(early_model, results_dir / f"early_aware_{args.model}.joblib")
    except Exception as exc:
        print(f"[WARN] Could not save models with joblib: {exc}")

    split_info = {
        "n_train": len(train_ids),
        "n_validation": len(validation_ids),
        "n_test": len(test_ids),
        "n_early_training_rows": int(after_cap),
    }

    models = {
        "full_trained": full_model,
        "early_aware": early_model,
    }

    predictions: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in models}

    for scenario in scenarios:
        path = find_feature_path(feature_dir, scenario)
        print("\n" + "=" * 90)
        print(f"[EVAL SCENARIO] {scenario}")
        print(f"[LOAD] {path}")

        df = read_table(path)
        df = add_stage_features(df, scenario, args.include_stage_features)

        if args.max_objects is not None and args.max_objects > 0:
            df = df.head(args.max_objects).copy()

        obj_s, y_s, X_s, _ = prepare_xy(df, feature_columns=feature_columns)

        mask = np.array([oid in test_ids for oid in obj_s], dtype=bool)

        if args.max_test_objects is not None and args.max_test_objects > 0:
            idx = np.where(mask)[0][: args.max_test_objects]
            new_mask = np.zeros(len(mask), dtype=bool)
            new_mask[idx] = True
            mask = new_mask

        obj_eval = obj_s[mask]
        y_eval = y_s[mask]
        X_eval = X_s.loc[mask]

        if len(y_eval) == 0:
            print(f"[WARN] No held-out objects available for scenario: {scenario}")
            continue

        novelty = compute_novelty(X_eval, novelty_ref)

        for model_name, model in models.items():
            model_pred_dir = pred_dir / model_name
            model_pred_dir.mkdir(parents=True, exist_ok=True)

            raw_probs = model.predict_proba(X_eval)
            probs = align_probabilities(raw_probs, model.classes_, all_classes)
            pred = build_prediction_frame(
                object_id=obj_eval,
                y_true=y_eval,
                probabilities=probs,
                classes=all_classes,
                novelty=novelty,
                rare_labels=rare_labels,
                save_probabilities=args.save_probabilities,
            )

            out_path = model_pred_dir / f"{scenario}_predictions.csv"
            pred.to_csv(out_path, index=False)
            predictions[model_name][scenario] = pred

            print(f"[OK] {model_name}: n={len(pred)}, acc={pred['correct'].mean():.4f}, saved={out_path}")

    # Metrics for all available objects.
    classification_rows = []
    followup_frames = []

    for model_name, scen_map in predictions.items():
        for scenario, pred in scen_map.items():
            classification_rows.append(
                classification_metrics(pred, model_name, "all_available_objects", scenario)
            )
            followup_frames.append(
                followup_metrics(pred, model_name, "all_available_objects", scenario, budgets)
            )

    # Metrics for common-object subset across scenarios, per model.
    for model_name, scen_map in predictions.items():
        common_ids = None
        for pred in scen_map.values():
            ids = set(pred["object_id"].to_numpy())
            common_ids = ids if common_ids is None else common_ids.intersection(ids)
        common_ids = common_ids or set()
        print(f"[COMMON] {model_name}: {len(common_ids)} common held-out objects across evaluated scenarios")

        if common_ids:
            for scenario, pred in scen_map.items():
                common_pred = pred[pred["object_id"].isin(common_ids)].copy()
                classification_rows.append(
                    classification_metrics(pred=common_pred, model_name=model_name, comparison_set="common_object_subset", scenario=scenario)
                )
                followup_frames.append(
                    followup_metrics(pred=common_pred, model_name=model_name, comparison_set="common_object_subset", scenario=scenario, budgets=budgets)
                )

    classification = pd.DataFrame(classification_rows)
    followup = pd.concat(followup_frames, ignore_index=True) if followup_frames else pd.DataFrame()

    scenario_order = {s: i for i, s in enumerate(DEFAULT_SCENARIOS)}
    model_order = {"full_trained": 0, "early_aware": 1}

    if not classification.empty:
        classification["model_order"] = classification["model_name"].map(model_order)
        classification["scenario_order"] = classification["scenario"].map(scenario_order)
        classification = classification.sort_values(
            ["comparison_set", "scenario_order", "model_order"]
        ).drop(columns=["model_order", "scenario_order"])

    if not followup.empty:
        followup["model_order"] = followup["model_name"].map(model_order)
        followup["scenario_order"] = followup["scenario"].map(scenario_order)
        followup = followup.sort_values(
            ["comparison_set", "policy", "budget_fraction", "scenario_order", "model_order"]
        ).drop(columns=["model_order", "scenario_order"])

    degradation = compute_degradation(classification, followup)
    gain = compute_gain_over_full_trained(classification, followup)

    # Write outputs.
    classification_path = results_dir / "early_aware_classification_metrics.csv"
    followup_path = results_dir / "early_aware_followup_metrics.csv"
    degradation_path = results_dir / "early_aware_degradation_vs_full.csv"
    gain_path = results_dir / "early_aware_gain_over_full_trained.csv"
    manifest_path = results_dir / "early_aware_training_manifest.csv"
    summary_path = results_dir / "early_aware_broker_like_summary.md"

    classification.to_csv(classification_path, index=False)
    followup.to_csv(followup_path, index=False)
    degradation.to_csv(degradation_path, index=False)
    gain.to_csv(gain_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    write_summary(
        path=summary_path,
        classification=classification,
        followup=followup,
        degradation=degradation,
        gain=gain,
        training_manifest=manifest,
        split_info=split_info,
        args=args,
        novelty_ref=novelty_ref,
    )

    for path in [classification_path, followup_path, degradation_path, gain_path, manifest_path, summary_path]:
        copy_to_final(path, final_dir)

    print("\n" + "=" * 90)
    print("[OK] Early-aware broker-like V4 evaluation finished.")
    print(f"[OK] Classification: {classification_path}")
    print(f"[OK] Follow-up:       {followup_path}")
    print(f"[OK] Degradation:     {degradation_path}")
    print(f"[OK] Gain:            {gain_path}")
    print(f"[OK] Manifest:        {manifest_path}")
    print(f"[OK] Summary:         {summary_path}")
    print(f"[OK] Publication copy:{final_dir}")


def run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)
    base = args.results_dir / "_self_test_early_aware"
    feature_dir = base / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["full_curve_reference", "first_3_points", "first_5_points", "window_7_days"]
    n = 900
    n_features = 24
    n_classes = 5

    object_id = np.arange(n)
    y = rng.integers(0, n_classes, size=n)

    base_X = rng.normal(size=(n, n_features))
    base_X[:, 0] += y * 0.8
    base_X[:, 1] += y * 0.4

    for scenario in scenarios:
        if scenario == "full_curve_reference":
            noise = 0.2
            keep = np.ones(n, dtype=bool)
        elif scenario == "first_3_points":
            noise = 1.1
            keep = np.ones(n, dtype=bool)
        elif scenario == "first_5_points":
            noise = 0.8
            keep = np.ones(n, dtype=bool)
        else:
            noise = 0.7
            keep = rng.random(n) > 0.1

        Xs = base_X + rng.normal(scale=noise, size=base_X.shape)
        df = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(n_features)])
        df.insert(0, "label", y)
        df.insert(0, "object_id", object_id)
        df = df.loc[keep].copy()
        df.to_csv(feature_dir / f"features_v4_temporal_shape_{scenario}.csv", index=False)

    test_args = argparse.Namespace(
        feature_dir=feature_dir,
        results_dir=base / "results",
        final_dir=base / "final",
        scenarios=",".join(scenarios),
        train_scenarios=",".join(scenarios),
        model="random_forest",
        n_estimators=80,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        class_weight_balanced=False,
        rf_max_depth=8,
        rf_min_samples_leaf=2,
        random_state=42,
        test_size=0.25,
        validation_size=0.15,
        rare_labels="3,4",
        budgets="0.05,0.10",
        save_probabilities=False,
        include_stage_features=True,
        max_objects=None,
        max_train_objects=None,
        max_test_objects=None,
        max_train_rows=0,
        self_test=False,
    )

    run_experiment(test_args)
    print(f"[OK] Self-test outputs: {base}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a full-trained vs early-aware V4 model for broker-like partial light curves."
    )

    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)

    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--train-scenarios", type=str, default="all")

    parser.add_argument("--model", choices=["lightgbm", "random_forest"], default="lightgbm")

    parser.add_argument("--n-estimators", type=int, default=650)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weight-balanced", action="store_true")

    parser.add_argument("--rf-max-depth", type=int, default=14)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.15)

    parser.add_argument("--rare-labels", type=str, default=DEFAULT_RARE_LABELS)
    parser.add_argument("--budgets", type=str, default="0.01,0.02,0.05,0.10,0.20")

    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--include-stage-features", action="store_true", default=True)
    parser.add_argument("--no-stage-features", dest="include_stage_features", action="store_false")

    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--max-train-objects", type=int, default=None)
    parser.add_argument("--max-test-objects", type=int, default=None)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional cap on stacked early-aware training rows. 0 means no cap.",
    )

    parser.add_argument("--self-test", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
