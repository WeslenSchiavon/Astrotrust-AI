#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Broker-like partial light-curve evaluation for AstroTrust-AI.

This is STEP 3 of the broker-like realism experiment.

It trains a tabular V4 probabilistic model on the full-curve training split and
evaluates the same trained model on each partial-light-curve feature scenario.

Important methodological point:
  The model is trained only once on the full-curve training objects.
  Each partial scenario is used only at inference/evaluation time.
  This simulates the operational question:
    "What happens when a broker-like system must classify the same kind of
     object before the full light curve is available?"

Inputs expected from STEP 2:
  data/processed/broker_like_partial_features/
    features_v4_temporal_shape_full_curve_reference.parquet
    features_v4_temporal_shape_first_3_points.parquet
    features_v4_temporal_shape_first_5_points.parquet
    ...

Outputs:
  results/broker_like_partial_lightcurve_stress/
    partial_classification_metrics.csv
    partial_followup_metrics.csv
    partial_degradation_vs_full.csv
    partial_lightcurve_broker_like_evaluation_summary.md
    predictions/{scenario}_predictions.csv

  results/final_publication/broker_like_realism/
    copies of the main CSV/MD outputs

Run:
  python experiments/run_broker_like_partial_v4_evaluation.py

Quick test:
  python experiments/run_broker_like_partial_v4_evaluation.py --self-test

Notes:
  - This script does not modify final_publication_summary.md.
  - The default model is LightGBM. Use --model random_forest only for smoke tests.
  - The final manuscript should clearly state if this stress test uses the V4
    tabular temporal-shape model rather than the full hybrid neural ensemble.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_FEATURE_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_features"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "broker_like_partial_lightcurve_stress"
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

DEFAULT_RARE_LABELS = "4,5,6,7,11,29,30,31"

EPS = 1e-12


def parse_scenarios(value: str) -> list[str]:
    if value.strip().lower() in {"", "all"}:
        return DEFAULT_SCENARIOS
    return [x.strip() for x in value.split(",") if x.strip()]


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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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
        f"No feature file found for scenario {scenario}. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def prepare_xy(df: pd.DataFrame, feature_columns: list[str] | None = None):
    required = {"object_id", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    object_id = df["object_id"].to_numpy()
    y = df["label"].astype(int).to_numpy()

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"object_id", "label"}]

    X = df.reindex(columns=feature_columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    return object_id, y, X, feature_columns


def make_model(args: argparse.Namespace, n_classes: int):
    if args.model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError(
                "LightGBM is not available in this Python environment. "
                "Install it with `pip install lightgbm`, or rerun with "
                "`--model random_forest` for a quick smoke test."
            ) from exc

        return LGBMClassifier(
            objective="multiclass",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            verbose=-1,
        )

    if args.model == "random_forest":
        return RandomForestClassifier(
            n_estimators=min(args.n_estimators, 300),
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    raise ValueError(f"Unknown model: {args.model}")


def align_probabilities(probabilities: np.ndarray, model_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(all_classes)), dtype=np.float64)
    class_to_index = {int(c): i for i, c in enumerate(all_classes)}

    for local_idx, cls in enumerate(model_classes):
        global_idx = class_to_index[int(cls)]
        aligned[:, global_idx] = probabilities[:, local_idx]

    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = aligned / np.maximum(row_sum, EPS)
    return aligned


def topk_labels(probabilities: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, probabilities.shape[1])
    idx = np.argsort(probabilities, axis=1)[:, -k_eff:][:, ::-1]
    return classes[idx]


def topk_accuracy_from_probs(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, k: int) -> float:
    topk = topk_labels(probabilities, classes, k)
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def metric_bundle(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict:
    y_pred = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)

    return {
        "n_objects": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": topk_accuracy_from_probs(y_true, probabilities, classes, 2),
        "top3_accuracy": topk_accuracy_from_probs(y_true, probabilities, classes, 3),
        "top5_accuracy": topk_accuracy_from_probs(y_true, probabilities, classes, 5),
        "mean_confidence": float(confidence.mean()),
        "mean_uncertainty": float((1.0 - confidence).mean()),
    }


def fit_novelty_reference(X_train: pd.DataFrame) -> dict:
    values = X_train.to_numpy(dtype=np.float64)

    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25

    keep = np.isfinite(med) & np.isfinite(iqr) & (np.abs(iqr) > EPS)

    if not np.any(keep):
        raise RuntimeError("No valid features remained for novelty reference.")

    z = (values[:, keep] - med[keep]) / (iqr[keep] + EPS)
    d_train = np.sqrt(np.mean(z * z, axis=1))

    q05 = float(np.nanpercentile(d_train, 5))
    q95 = float(np.nanpercentile(d_train, 95))

    return {
        "median": med,
        "iqr": iqr,
        "keep": keep,
        "q05": q05,
        "q95": q95,
        "n_retained_features": int(np.sum(keep)),
    }


def compute_novelty(X: pd.DataFrame, ref: dict) -> np.ndarray:
    values = X.to_numpy(dtype=np.float64)
    keep = ref["keep"]
    med = ref["median"]
    iqr = ref["iqr"]
    z = (values[:, keep] - med[keep]) / (iqr[keep] + EPS)
    d = np.sqrt(np.mean(z * z, axis=1))

    novelty = (d - ref["q05"]) / max(ref["q95"] - ref["q05"], EPS)
    novelty = np.clip(novelty, 0.0, 1.0)
    return novelty


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


def followup_metrics_for_predictions(
    pred: pd.DataFrame,
    scenario: str,
    comparison_set: str,
    budgets: Iterable[float],
) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()

    rare = pred["true_is_rare"].astype(bool).to_numpy()
    baseline = float(np.mean(rare))

    rows = []
    policies = score_policies(pred)

    for policy, score in policies.items():
        order = np.argsort(score)[::-1]

        for budget in budgets:
            n_select = max(1, int(math.ceil(len(pred) * budget)))
            selected_idx = order[:n_select]
            selected = pred.iloc[selected_idx]

            rare_rate = float(selected["true_is_rare"].astype(bool).mean())
            enrichment = rare_rate / baseline if baseline > 0 else np.nan

            rows.append({
                "comparison_set": comparison_set,
                "scenario": scenario,
                "policy": policy,
                "budget_fraction": float(budget),
                "n_available": int(len(pred)),
                "n_selected": int(n_select),
                "baseline_rare_rate": baseline,
                "rare_rate": rare_rate,
                "rare_enrichment": enrichment,
                "mean_confidence": float(selected["confidence"].mean()),
                "mean_uncertainty": float(selected["uncertainty"].mean()),
                "mean_novelty": float(selected["novelty"].mean()),
                "mean_rarity_score": float(selected["rarity_score"].mean()),
                "mean_correct": float(selected["correct"].astype(bool).mean()),
            })

    return pd.DataFrame(rows)


def classification_metrics_for_prediction_frame(
    pred: pd.DataFrame,
    scenario: str,
    comparison_set: str,
) -> dict:
    y_true = pred["true_label"].astype(int).to_numpy()
    y_pred = pred["predicted_label"].astype(int).to_numpy()

    out = {
        "comparison_set": comparison_set,
        "scenario": scenario,
        "n_objects": int(len(pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mean_confidence": float(pred["confidence"].mean()),
        "mean_uncertainty": float(pred["uncertainty"].mean()),
        "mean_novelty": float(pred["novelty"].mean()),
        "mean_rarity_score": float(pred["rarity_score"].mean()),
        "rare_true_rate": float(pred["true_is_rare"].astype(bool).mean()),
    }

    for k in [2, 3, 5]:
        cols = [f"top{i}_label" for i in range(1, k + 1) if f"top{i}_label" in pred.columns]
        if cols:
            y_true_arr = pred["true_label"].astype(int).to_numpy()
            top = pred[cols].astype(int).to_numpy()
            out[f"top{k}_accuracy"] = float(np.mean([yt in row for yt, row in zip(y_true_arr, top)]))

    return out


def compute_degradation(classification: pd.DataFrame, followup: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for comparison_set in sorted(classification["comparison_set"].unique()):
        subset = classification[classification["comparison_set"] == comparison_set]
        full = subset[subset["scenario"] == "full_curve_reference"]

        if full.empty:
            continue

        ref = full.iloc[0].to_dict()

        for row in subset.itertuples(index=False):
            if row.scenario == "full_curve_reference":
                continue

            row_dict = row._asdict()
            out = {
                "comparison_set": comparison_set,
                "scenario": row.scenario,
                "n_objects": row_dict.get("n_objects"),
            }

            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "top3_accuracy",
                "top5_accuracy",
                "mean_confidence",
                "mean_uncertainty",
                "mean_novelty",
                "rare_true_rate",
            ]:
                if metric in ref and metric in row_dict and pd.notna(ref[metric]) and pd.notna(row_dict[metric]):
                    out[f"delta_{metric}"] = float(row_dict[metric] - ref[metric])

            rows.append(out)

    degradation = pd.DataFrame(rows)

    if not followup.empty:
        main = followup[
            (followup["policy"] == "novelty_rarity")
            & (np.isclose(followup["budget_fraction"], 0.05))
        ].copy()

        enrich_rows = []
        for comparison_set in sorted(main["comparison_set"].unique()):
            ref = main[
                (main["comparison_set"] == comparison_set)
                & (main["scenario"] == "full_curve_reference")
            ]

            if ref.empty:
                continue

            ref_enrich = float(ref.iloc[0]["rare_enrichment"])

            for row in main[
                (main["comparison_set"] == comparison_set)
                & (main["scenario"] != "full_curve_reference")
            ].itertuples(index=False):
                enrich_rows.append({
                    "comparison_set": comparison_set,
                    "scenario": row.scenario,
                    "delta_rare_enrichment_5pct_novelty_rarity": float(row.rare_enrichment) - ref_enrich,
                })

        if enrich_rows and not degradation.empty:
            degradation = degradation.merge(
                pd.DataFrame(enrich_rows),
                on=["comparison_set", "scenario"],
                how="left",
            )

    return degradation


def write_summary(
    path: Path,
    classification: pd.DataFrame,
    followup: pd.DataFrame,
    degradation: pd.DataFrame,
    split_summary: dict,
    novelty_ref: dict,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Broker-like partial light-curve evaluation",
        "",
        "## Protocol",
        "",
        "A probabilistic tabular V4 model was trained once using full-curve features from the training split. "
        "The same fitted model was then applied to full and temporally truncated feature tables. "
        "This evaluates how classification, uncertainty, top-k recovery, novelty, rarity, and follow-up prioritization degrade when only partial light curves are available.",
        "",
        f"- Model: `{args.model}`",
        f"- Random state: `{args.random_state}`",
        f"- Train objects: `{split_summary['n_train']}`",
        f"- Validation objects: `{split_summary['n_validation']}`",
        f"- Held-out test objects: `{split_summary['n_test']}`",
        f"- Novelty retained features: `{novelty_ref['n_retained_features']}`",
        f"- Rare labels: `{args.rare_labels}`",
        "",
        "## Classification metrics",
        "",
    ]

    display_cols = [
        "comparison_set", "scenario", "n_objects", "accuracy", "macro_f1",
        "top3_accuracy", "top5_accuracy", "mean_uncertainty", "mean_novelty",
        "rare_true_rate"
    ]
    display_cols = [c for c in display_cols if c in classification.columns]
    lines.append(classification[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    if not degradation.empty:
        lines += [
            "## Degradation relative to full-curve reference",
            "",
        ]
        deg_cols = [
            "comparison_set", "scenario", "n_objects",
            "delta_accuracy", "delta_macro_f1",
            "delta_top3_accuracy", "delta_top5_accuracy",
            "delta_mean_uncertainty",
            "delta_rare_enrichment_5pct_novelty_rarity",
        ]
        deg_cols = [c for c in deg_cols if c in degradation.columns]
        lines.append(degradation[deg_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    if not followup.empty:
        main_follow = followup[
            (followup["policy"] == "novelty_rarity")
            & (np.isclose(followup["budget_fraction"], 0.05))
        ].copy()

        lines += [
            "## Follow-up prioritization at 5% budget",
            "",
        ]

        cols = [
            "comparison_set", "scenario", "n_available", "n_selected",
            "baseline_rare_rate", "rare_rate", "rare_enrichment",
            "mean_uncertainty", "mean_novelty", "mean_correct"
        ]
        cols = [c for c in cols if c in main_follow.columns]
        lines.append(main_follow[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines += [
        "## Suggested manuscript text",
        "",
        "```latex",
        r"\paragraph{Broker-like partial-light-curve stress test.}",
        r"To assess whether AstroTrust-AI retains operational value under early-alert conditions, we constructed temporally truncated held-out light curves using the first \(N\) observations and fixed windows after the first alert. For each scenario, light-curve-derived V4 temporal-shape features were recomputed from the truncated observations only, avoiding leakage from future photometric points. A probabilistic model trained on the full-curve training split was then evaluated on the corresponding partial representations of the held-out objects. We report both all-available-object results and a common-object subset analysis, because very early temporal windows contain fewer objects with the minimum number of observations required for feature construction.",
        "```",
        "",
        "## Important interpretation caveat",
        "",
        "If this experiment is used with the default tabular V4 LightGBM model, describe it as a V4 temporal-shape broker-like stress test. "
        "Do not claim it is the full hybrid neural ensemble unless the hybrid/ensemble inference is also rerun for the partial scenarios.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_to_final(src: Path, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, final_dir / src.name)


def run_evaluation(args: argparse.Namespace) -> None:
    feature_dir = args.feature_dir
    results_dir = args.results_dir
    final_dir = args.final_dir
    predictions_dir = results_dir / "predictions"

    results_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    rare_labels = parse_int_set(args.rare_labels)
    scenarios = parse_scenarios(args.scenarios)
    budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]

    full_path = find_feature_path(feature_dir, "full_curve_reference")
    print(f"[LOAD] Full reference features: {full_path}")
    full_df = read_table(full_path)

    if args.max_objects is not None:
        full_df = full_df.head(args.max_objects).copy()

    object_id, y, X, feature_columns = prepare_xy(full_df)

    all_classes = np.array(sorted(np.unique(y)), dtype=int)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=y[train_val_idx],
    )

    if args.max_train_objects is not None:
        train_idx = train_idx[: args.max_train_objects]
    if args.max_test_objects is not None:
        test_idx = test_idx[: args.max_test_objects]

    train_object_ids = set(object_id[train_idx])
    validation_object_ids = set(object_id[validation_idx])
    test_object_ids = set(object_id[test_idx])

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    print("[SPLIT]")
    print(f"  train:      {len(train_object_ids)}")
    print(f"  validation: {len(validation_object_ids)}")
    print(f"  test:       {len(test_object_ids)}")
    print(f"  features:   {len(feature_columns)}")
    print(f"  classes:    {len(all_classes)}")

    novelty_ref = fit_novelty_reference(X_train)

    model = make_model(args, n_classes=len(all_classes))

    print(f"[TRAIN] Fitting {args.model} on full-curve training features...")
    model.fit(X_train, y_train)

    try:
        import joblib
        model_path = results_dir / f"broker_like_partial_{args.model}_model.joblib"
        joblib.dump(model, model_path)
        print(f"[OK] Saved model: {model_path}")
    except Exception:
        print("[WARN] Could not save model with joblib. Continuing.")

    prediction_frames: dict[str, pd.DataFrame] = {}

    for scenario in scenarios:
        scenario_path = find_feature_path(feature_dir, scenario)
        print("\n" + "=" * 90)
        print(f"[SCENARIO] {scenario}")
        print(f"[LOAD] {scenario_path}")

        df = read_table(scenario_path)

        if args.max_objects is not None:
            df = df.head(args.max_objects).copy()

        obj_s, y_s, X_s, _ = prepare_xy(df, feature_columns=feature_columns)

        mask = np.array([oid in test_object_ids for oid in obj_s], dtype=bool)

        if args.max_test_objects is not None:
            idx = np.where(mask)[0][: args.max_test_objects]
            mask = np.zeros(len(mask), dtype=bool)
            mask[idx] = True

        obj_eval = obj_s[mask]
        y_eval = y_s[mask]
        X_eval = X_s.loc[mask]

        if len(y_eval) == 0:
            print(f"[WARN] No held-out objects available for scenario {scenario}.")
            continue

        raw_probs = model.predict_proba(X_eval)
        probs = align_probabilities(raw_probs, model.classes_, all_classes)
        novelty = compute_novelty(X_eval, novelty_ref)

        pred = build_prediction_frame(
            object_id=obj_eval,
            y_true=y_eval,
            probabilities=probs,
            classes=all_classes,
            novelty=novelty,
            rare_labels=rare_labels,
            save_probabilities=args.save_probabilities,
        )

        pred_path = predictions_dir / f"{scenario}_predictions.csv"
        pred.to_csv(pred_path, index=False)
        print(f"[OK] Predictions: {pred_path}")
        print(f"[OK] Objects evaluated: {len(pred)}")

        prediction_frames[scenario] = pred

    if not prediction_frames:
        raise RuntimeError("No scenarios were evaluated.")

    classification_rows = []
    followup_frames = []

    for scenario, pred in prediction_frames.items():
        classification_rows.append(
            classification_metrics_for_prediction_frame(
                pred=pred,
                scenario=scenario,
                comparison_set="all_available_objects",
            )
        )
        followup_frames.append(
            followup_metrics_for_predictions(
                pred=pred,
                scenario=scenario,
                comparison_set="all_available_objects",
                budgets=budgets,
            )
        )

    common_ids = None
    for pred in prediction_frames.values():
        ids = set(pred["object_id"].to_numpy())
        common_ids = ids if common_ids is None else common_ids.intersection(ids)

    common_ids = common_ids or set()
    print(f"\n[COMMON] Common held-out objects across evaluated scenarios: {len(common_ids)}")

    if common_ids:
        for scenario, pred in prediction_frames.items():
            common_pred = pred[pred["object_id"].isin(common_ids)].copy()
            classification_rows.append(
                classification_metrics_for_prediction_frame(
                    pred=common_pred,
                    scenario=scenario,
                    comparison_set="common_object_subset",
                )
            )
            followup_frames.append(
                followup_metrics_for_predictions(
                    pred=common_pred,
                    scenario=scenario,
                    comparison_set="common_object_subset",
                    budgets=budgets,
                )
            )

    classification = pd.DataFrame(classification_rows)
    classification["scenario_order"] = classification["scenario"].map({s: i for i, s in enumerate(DEFAULT_SCENARIOS)})
    classification = classification.sort_values(["comparison_set", "scenario_order"]).drop(columns=["scenario_order"])

    followup = pd.concat(followup_frames, ignore_index=True) if followup_frames else pd.DataFrame()
    followup["scenario_order"] = followup["scenario"].map({s: i for i, s in enumerate(DEFAULT_SCENARIOS)})
    followup = followup.sort_values(["comparison_set", "policy", "budget_fraction", "scenario_order"]).drop(columns=["scenario_order"])

    degradation = compute_degradation(classification, followup)

    classification_path = results_dir / "partial_classification_metrics.csv"
    followup_path = results_dir / "partial_followup_metrics.csv"
    degradation_path = results_dir / "partial_degradation_vs_full.csv"
    summary_path = results_dir / "partial_lightcurve_broker_like_evaluation_summary.md"

    write_csv(classification, classification_path)
    write_csv(followup, followup_path)
    write_csv(degradation, degradation_path)

    split_summary = {
        "n_train": len(train_object_ids),
        "n_validation": len(validation_object_ids),
        "n_test": len(test_object_ids),
    }

    write_summary(
        path=summary_path,
        classification=classification,
        followup=followup,
        degradation=degradation,
        split_summary=split_summary,
        novelty_ref=novelty_ref,
        args=args,
    )

    for path in [classification_path, followup_path, degradation_path, summary_path]:
        copy_to_final(path, final_dir)

    print("\n" + "=" * 90)
    print("[OK] Broker-like partial evaluation finished.")
    print(f"[OK] Classification: {classification_path}")
    print(f"[OK] Follow-up:       {followup_path}")
    print(f"[OK] Degradation:     {degradation_path}")
    print(f"[OK] Summary:         {summary_path}")
    print(f"[OK] Publication copy:{final_dir}")


def run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)
    base = args.results_dir / "_self_test_eval"
    feature_dir = base / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["full_curve_reference", "first_3_points", "first_5_points", "window_7_days"]
    n = 600
    n_features = 20
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
            noise = 1.4
            keep = np.ones(n, dtype=bool)
        elif scenario == "first_5_points":
            noise = 1.0
            keep = np.ones(n, dtype=bool)
        else:
            noise = 0.8
            keep = rng.random(n) > 0.10

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
        model="random_forest",
        n_estimators=80,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        test_size=0.25,
        validation_size=0.15,
        rare_labels="3,4",
        budgets="0.05,0.10",
        save_probabilities=False,
        max_objects=None,
        max_train_objects=None,
        max_test_objects=None,
        rf_max_depth=8,
        rf_min_samples_leaf=2,
        self_test=False,
    )

    run_evaluation(test_args)
    print(f"[OK] Self-test outputs: {base}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate broker-like partial-light-curve feature scenarios."
    )

    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--scenarios", type=str, default="all")

    parser.add_argument("--model", choices=["lightgbm", "random_forest"], default="lightgbm")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--rf-max-depth", type=int, default=14)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.15)

    parser.add_argument("--rare-labels", type=str, default=DEFAULT_RARE_LABELS)
    parser.add_argument("--budgets", type=str, default="0.01,0.02,0.05,0.10,0.20")

    parser.add_argument("--save-probabilities", action="store_true")

    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--max-train-objects", type=int, default=None)
    parser.add_argument("--max-test-objects", type=int, default=None)

    parser.add_argument("--self-test", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
