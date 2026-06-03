#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_stage_aware_early_aware_ensemble.py

Final prototyping step for the AstroTrust-AI broker-like partial-light-curve work.

It creates two products:

1) ensemble_hybrid_dominant_early_aware
   Partial-light-curve ensemble:
     0.70 hybrid_early_aware
   + 0.15 lightgbm_early_aware
   + 0.10 lightgbm_regularized_early_aware
   + 0.05 xgboost_early_aware, if XGBoost is available.

2) stage_aware_astrotrust_ai
   Integrated prototype:
     - full_curve_reference: original full-curve model, if available
     - partial scenarios: ensemble_hybrid_dominant_early_aware

Expected previous outputs:
  results/broker_like_hybrid_early_aware/
    predictions/hybrid_early_aware/{scenario}_predictions.csv
    probabilities/hybrid_early_aware/{scenario}_probabilities.npy
    predictions/hybrid_full_trained/{scenario}_predictions.csv
    probabilities/hybrid_full_trained/{scenario}_probabilities.npy

Expected feature files:
  data/processed/broker_like_partial_features/
    features_v4_temporal_shape_{scenario}.parquet

Outputs:
  results/broker_like_stage_aware_ensemble/
    stage_aware_ensemble_classification_metrics.csv
    stage_aware_ensemble_followup_metrics.csv
    stage_aware_ensemble_gain_tables.csv
    stage_aware_ensemble_summary.md
    predictions/{model}/{scenario}_predictions.csv
    probabilities/{model}/{scenario}_probabilities.npy

Usage:
  python experiments/run_stage_aware_early_aware_ensemble.py

Safer first run without XGBoost:
  python experiments/run_stage_aware_early_aware_ensemble.py --skip-xgboost

Lower-cost run:
  python experiments/run_stage_aware_early_aware_ensemble.py --max-train-rows 300000 --skip-xgboost
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
EPS = 1e-12

DEFAULT_FEATURE_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_features"
DEFAULT_HYBRID_DIR = ROOT_DIR / "results" / "broker_like_hybrid_early_aware"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "broker_like_stage_aware_ensemble"
DEFAULT_FINAL_DIR = ROOT_DIR / "results" / "final_publication" / "broker_like_realism"

DEFAULT_OFFICIAL_FULL_PRED = ROOT_DIR / "results" / "hybrid_tabular_ensemble_250k" / "ensemble_hybrid_dominant_predictions.csv"
DEFAULT_OFFICIAL_FULL_PROB = ROOT_DIR / "results" / "hybrid_tabular_ensemble_250k" / "ensemble_hybrid_dominant_test_probabilities.npy"

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


def parse_list(value: str, default: list[str]) -> list[str]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_int_set(value: str) -> set[int]:
    return {int(x.strip()) for x in str(value).split(",") if x.strip()}


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except Exception:
        return str(path)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table file: {path}")


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
    raise FileNotFoundError("No feature file found for scenario `{}`. Tried:\n{}".format(
        scenario, "\n".join(str(p) for p in candidates)
    ))


def scenario_stage_values(scenario: str) -> dict[str, float]:
    out = {
        "stage_is_full": 0.0,
        "stage_is_first_points": 0.0,
        "stage_is_window_days": 0.0,
        "stage_first_n": 0.0,
        "stage_window_days": 0.0,
        "stage_log_first_n": 0.0,
        "stage_log_window_days": 0.0,
    }
    if scenario == "full_curve_reference":
        out["stage_is_full"] = 1.0
    elif scenario.startswith("first_") and scenario.endswith("_points"):
        n = float(scenario.replace("first_", "").replace("_points", ""))
        out["stage_is_first_points"] = 1.0
        out["stage_first_n"] = n
        out["stage_log_first_n"] = math.log1p(n)
    elif scenario.startswith("window_") and scenario.endswith("_days"):
        d = float(scenario.replace("window_", "").replace("_days", ""))
        out["stage_is_window_days"] = 1.0
        out["stage_window_days"] = d
        out["stage_log_window_days"] = math.log1p(d)
    return out


def add_stage_features(df: pd.DataFrame, scenario: str, include: bool) -> pd.DataFrame:
    df = df.copy()
    df["scenario_name"] = scenario
    if include:
        for key, value in scenario_stage_values(scenario).items():
            df[key] = value
    return df


def prepare_xy(df: pd.DataFrame, feature_columns: list[str] | None = None):
    if "object_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Feature table must contain `object_id` and `label` columns.")

    object_id = df["object_id"].to_numpy()
    y = df["label"].astype(int).to_numpy()

    if feature_columns is None:
        feature_columns = [
            c for c in df.columns
            if c not in {"object_id", "label", "scenario_name"}
        ]

    X = df.reindex(columns=feature_columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return object_id, y, X, feature_columns


def sanitize_probabilities(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.nan_to_num(p, nan=EPS, posinf=1.0, neginf=EPS)
    p = np.clip(p, EPS, 1.0)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def align_probabilities(probs: np.ndarray, model_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probs.shape[0], len(all_classes)), dtype=np.float64)
    class_to_idx = {int(c): i for i, c in enumerate(all_classes)}
    for j, cls in enumerate(model_classes):
        c = int(cls)
        if c in class_to_idx:
            aligned[:, class_to_idx[c]] = probs[:, j]
    return sanitize_probabilities(aligned)


def fit_novelty_reference(X_train: pd.DataFrame) -> dict:
    values = X_train.to_numpy(dtype=np.float64)
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25
    keep = np.isfinite(med) & np.isfinite(iqr) & (np.abs(iqr) > EPS)
    if not np.any(keep):
        raise RuntimeError("No retained feature for novelty computation.")
    z = (values[:, keep] - med[keep]) / (iqr[keep] + EPS)
    d = np.sqrt(np.mean(z * z, axis=1))
    return {
        "median": med,
        "iqr": iqr,
        "keep": keep,
        "q05": float(np.nanpercentile(d, 5)),
        "q95": float(np.nanpercentile(d, 95)),
        "n_retained_features": int(np.sum(keep)),
    }


def compute_novelty(X: pd.DataFrame, ref: dict) -> np.ndarray:
    values = X.to_numpy(dtype=np.float64)
    keep = ref["keep"]
    z = (values[:, keep] - ref["median"][keep]) / (ref["iqr"][keep] + EPS)
    d = np.sqrt(np.mean(z * z, axis=1))
    novelty = (d - ref["q05"]) / max(ref["q95"] - ref["q05"], EPS)
    return np.clip(novelty, 0.0, 1.0)


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    k = min(k, probs.shape[1])
    top = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    return float(np.mean([int(y) in row for y, row in zip(y_true, top)]))


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    p = sanitize_probabilities(probs)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    corr = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def load_hybrid_outputs(hybrid_dir: Path, model_name: str, scenario: str):
    pred_path = hybrid_dir / "predictions" / model_name / f"{scenario}_predictions.csv"
    prob_path = hybrid_dir / "probabilities" / model_name / f"{scenario}_probabilities.npy"
    if not pred_path.exists() or not prob_path.exists():
        raise FileNotFoundError(f"Missing outputs for {model_name}/{scenario}:\n{pred_path}\n{prob_path}")
    pred = pd.read_csv(pred_path)
    probs = sanitize_probabilities(np.load(prob_path))
    if len(pred) != len(probs):
        raise ValueError(f"Prediction/probability length mismatch for {model_name}/{scenario}.")
    return pred, probs


def probability_frame(object_ids: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"object_id": object_ids})
    for i in range(probs.shape[1]):
        out[f"prob_{i}"] = probs[:, i]
    return out


def align_probs_by_object(target_ids: np.ndarray, source_ids: np.ndarray, source_probs: np.ndarray) -> np.ndarray:
    src = probability_frame(source_ids, source_probs)
    target = pd.DataFrame({"object_id": target_ids})
    merged = target.merge(src, on="object_id", how="left")
    cols = [c for c in merged.columns if c.startswith("prob_")]
    if merged[cols].isna().any().any():
        missing = int(merged[cols].isna().any(axis=1).sum())
        raise ValueError(f"Could not align probabilities for {missing} objects.")
    return sanitize_probabilities(merged[cols].to_numpy(dtype=float))


def build_prediction_frame(object_ids, y_true, probs, novelty, rare_labels):
    probs = sanitize_probabilities(probs)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    rare_cols = [i for i in range(probs.shape[1]) if i in rare_labels]
    rarity = probs[:, rare_cols].sum(axis=1) if rare_cols else np.zeros(len(pred))
    top5 = np.argsort(probs, axis=1)[:, -min(5, probs.shape[1]):][:, ::-1]

    out = pd.DataFrame({
        "object_id": object_ids,
        "true_label": y_true.astype(int),
        "predicted_label": pred.astype(int),
        "correct": pred.astype(int) == y_true.astype(int),
        "confidence": conf,
        "uncertainty": 1.0 - conf,
        "novelty": novelty,
        "rarity_score": rarity,
        "priority_novelty_rarity": 0.5 * novelty + 0.5 * rarity,
        "true_is_rare": np.array([int(v) in rare_labels for v in y_true], dtype=bool),
    })
    for k in range(top5.shape[1]):
        out[f"top{k + 1}_label"] = top5[:, k]
    return out


def classification_metrics(model_name, comparison_set, scenario, pred, probs):
    y = pred["true_label"].astype(int).to_numpy()
    yp = pred["predicted_label"].astype(int).to_numpy()
    correct = pred["correct"].astype(bool).to_numpy(dtype=float)
    return {
        "model_name": model_name,
        "comparison_set": comparison_set,
        "scenario": scenario,
        "n_objects": int(len(pred)),
        "accuracy": float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, yp, average="weighted", zero_division=0)),
        "top3_accuracy": top_k_accuracy(y, probs, 3),
        "top5_accuracy": top_k_accuracy(y, probs, 5),
        "mean_confidence": float(pred["confidence"].mean()),
        "mean_uncertainty": float(pred["uncertainty"].mean()),
        "mean_novelty": float(pred["novelty"].mean()),
        "rare_true_rate": float(pred["true_is_rare"].mean()),
        "se_accuracy": float(correct.std(ddof=1) / math.sqrt(len(correct))) if len(correct) > 1 else 0.0,
        "ece": ece_score(y, probs),
    }


def followup_metrics(model_name, comparison_set, scenario, pred, budgets):
    baseline = float(pred["true_is_rare"].mean())
    scores = {
        "novelty_rarity": pred["priority_novelty_rarity"].to_numpy(dtype=float),
        "rarity_only": pred["rarity_score"].to_numpy(dtype=float),
        "uncertainty_only": pred["uncertainty"].to_numpy(dtype=float),
    }
    rows = []
    for policy, score in scores.items():
        order = np.argsort(score)[::-1]
        for budget in budgets:
            n_select = max(1, int(math.ceil(len(pred) * budget)))
            sel = pred.iloc[order[:n_select]]
            rare_rate = float(sel["true_is_rare"].mean())
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
                "rare_enrichment": rare_rate / baseline if baseline > 0 else np.nan,
                "mean_correct": float(sel["correct"].mean()),
                "mean_confidence": float(sel["confidence"].mean()),
                "mean_uncertainty": float(sel["uncertainty"].mean()),
                "mean_novelty": float(sel["novelty"].mean()),
                "mean_rarity_score": float(sel["rarity_score"].mean()),
            })
    return pd.DataFrame(rows)


def save_prediction_and_probs(output_dir, model_name, scenario, pred, probs):
    pdir = output_dir / "predictions" / model_name
    qdir = output_dir / "probabilities" / model_name
    pdir.mkdir(parents=True, exist_ok=True)
    qdir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pdir / f"{scenario}_predictions.csv", index=False)
    np.save(qdir / f"{scenario}_probabilities.npy", sanitize_probabilities(probs).astype(np.float32))


def make_model(model_name: str, args, n_classes: int):
    if model_name == "lightgbm_early_aware":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            max_depth=-1,
            min_child_samples=args.lgbm_min_child_samples,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            class_weight="balanced" if args.class_weight_balanced else None,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
    if model_name == "lightgbm_regularized_early_aware":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=args.lgbm_reg_n_estimators,
            learning_rate=args.lgbm_reg_learning_rate,
            num_leaves=args.lgbm_reg_num_leaves,
            max_depth=-1,
            min_child_samples=args.lgbm_reg_min_child_samples,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            reg_alpha=0.5,
            class_weight="balanced" if args.class_weight_balanced else None,
            random_state=args.seed + 17,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
    if model_name == "xgboost_early_aware":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            tree_method=args.xgb_tree_method,
            random_state=args.seed + 29,
            n_jobs=args.n_jobs,
        )
    raise ValueError(model_name)


def build_training_table(args, train_ids, feature_columns):
    pieces = []
    rows = []
    for scenario in parse_list(args.train_scenarios, DEFAULT_SCENARIOS):
        path = find_feature_path(args.feature_dir, scenario)
        df = add_stage_features(read_table(path), scenario, args.include_stage_features)
        before = len(df)
        df = df[df["object_id"].isin(train_ids)].copy()
        _, y, X, _ = prepare_xy(df, feature_columns)
        X = X.copy()
        X["__label__"] = y
        pieces.append(X)
        rows.append({
            "scenario": scenario,
            "source_path": safe_rel(path),
            "rows_before_train_filter": int(before),
            "rows_used_before_cap": int(len(df)),
        })

    train = pd.concat(pieces, ignore_index=True)
    if args.max_train_rows and args.max_train_rows > 0 and len(train) > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, random_state=args.seed).reset_index(drop=True)

    y_train = train.pop("__label__").astype(int).to_numpy()
    manifest = pd.DataFrame(rows)
    manifest["rows_after_global_cap"] = int(len(train))
    return train.astype(np.float32), y_train, manifest


def normalized_weights(weights: dict[str, float], available: set[str]) -> dict[str, float]:
    out = {k: float(v) for k, v in weights.items() if k in available and float(v) > 0}
    total = sum(out.values())
    if total <= 0:
        raise RuntimeError("No available ensemble weights.")
    return {k: v / total for k, v in out.items()}


def find_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Missing one of columns: {candidates}")
    return None


def try_official_full(args, target_ids, novelty, rare_labels):
    if not args.use_official_full_ensemble:
        return None
    pred_path = Path(args.official_full_predictions)
    prob_path = Path(args.official_full_probabilities)
    if not pred_path.exists() or not prob_path.exists():
        return None
    try:
        pred = pd.read_csv(pred_path)
        probs = sanitize_probabilities(np.load(prob_path))
        id_col = find_col(pred, ["object_id", "diaobjectid", "diaObjectId", "oid", "snid"], required=False)
        true_col = find_col(pred, ["true_label", "y_true", "label", "target", "true_class", "class_id"], required=False)
        if id_col is None or true_col is None or len(pred) != len(probs):
            return None
        aligned_probs = align_probs_by_object(target_ids, pred[id_col].to_numpy(), probs)
        truth = pd.DataFrame({"object_id": target_ids}).merge(
            pred[[id_col, true_col]].rename(columns={id_col: "object_id", true_col: "true_label"}),
            on="object_id",
            how="left",
        )
        if truth["true_label"].isna().any():
            return None
        frame = build_prediction_frame(target_ids, truth["true_label"].astype(int).to_numpy(), aligned_probs, novelty, rare_labels)
        return frame, aligned_probs, "official_ensemble_hybrid_dominant"
    except Exception as exc:
        print(f"[WARN] official full ensemble could not be loaded: {exc}")
        return None


def gain_table(classification, followup, target_model, reference_model):
    base = classification[classification["model_name"] == reference_model]
    targ = classification[classification["model_name"] == target_model]
    if base.empty or targ.empty:
        return pd.DataFrame()
    merged = targ.merge(base, on=["comparison_set", "scenario"], suffixes=("_target", "_reference"))
    rows = []
    for row in merged.itertuples(index=False):
        d = row._asdict()
        out = {
            "comparison_set": d["comparison_set"],
            "scenario": d["scenario"],
            "target_model": target_model,
            "reference_model": reference_model,
        }
        for m in ["accuracy", "macro_f1", "top3_accuracy", "top5_accuracy", "mean_uncertainty", "ece"]:
            out[f"gain_{m}"] = float(d[f"{m}_target"] - d[f"{m}_reference"])
        rows.append(out)
    gain = pd.DataFrame(rows)

    main = followup[
        (followup["policy"] == "novelty_rarity")
        & (np.isclose(followup["budget_fraction"], 0.05))
        & (followup["model_name"].isin([target_model, reference_model]))
    ]
    if not main.empty and not gain.empty:
        t = main[main["model_name"] == target_model]
        b = main[main["model_name"] == reference_model]
        fm = t.merge(b, on=["comparison_set", "scenario", "policy", "budget_fraction"], suffixes=("_target", "_reference"))
        erows = []
        for row in fm.itertuples(index=False):
            d = row._asdict()
            erows.append({
                "comparison_set": d["comparison_set"],
                "scenario": d["scenario"],
                "gain_rare_enrichment_5pct": float(d["rare_enrichment_target"] - d["rare_enrichment_reference"]),
                "gain_rare_rate_5pct": float(d["rare_rate_target"] - d["rare_rate_reference"]),
            })
        gain = gain.merge(pd.DataFrame(erows), on=["comparison_set", "scenario"], how="left")
    return gain


def write_summary(args, classification, followup, gain, manifest, weights_used, official_full_used):
    path = args.output_dir / "stage_aware_ensemble_summary.md"
    lines = [
        "# Stage-aware early-aware ensemble prototype",
        "",
        "## Protocol",
        "",
        "This experiment builds an early-aware ensemble for partial light curves and a stage-aware integrated AstroTrust-AI prototype.",
        "",
        f"- Early-aware tabular training rows cap: `{args.max_train_rows}`",
        f"- Include stage features: `{args.include_stage_features}`",
        f"- XGBoost skipped: `{args.skip_xgboost}`",
        f"- Official full ensemble used for full-stage route: `{official_full_used}`",
        f"- Active partial-ensemble weights: `{json.dumps(weights_used, sort_keys=True)}`",
        "",
        "## Member/training manifest",
        "",
        manifest.to_markdown(index=False) if not manifest.empty else "_No manifest._",
        "",
        "## Classification metrics",
        "",
    ]
    cols = [
        "model_name", "comparison_set", "scenario", "n_objects", "accuracy", "se_accuracy",
        "macro_f1", "top3_accuracy", "top5_accuracy", "mean_uncertainty", "mean_novelty",
        "rare_true_rate", "ece",
    ]
    cols = [c for c in cols if c in classification.columns]
    lines.append(classification[cols].to_markdown(index=False, floatfmt=".4f"))
    lines += [""]

    main = followup[
        (followup["policy"] == "novelty_rarity")
        & (np.isclose(followup["budget_fraction"], 0.05))
    ].copy()
    if not main.empty:
        lines += ["## Follow-up prioritization at 5% budget", ""]
        fcols = [
            "model_name", "comparison_set", "scenario", "n_available", "n_selected",
            "baseline_rare_rate", "rare_rate", "rare_enrichment", "mean_correct",
            "mean_uncertainty", "mean_novelty", "mean_rarity_score",
        ]
        fcols = [c for c in fcols if c in main.columns]
        lines.append(main[fcols].to_markdown(index=False, floatfmt=".4f"))
        lines += [""]

    if not gain.empty:
        lines += ["## Gain tables", ""]
        gcols = [
            "comparison_set", "scenario", "target_model", "reference_model",
            "gain_accuracy", "gain_macro_f1", "gain_top3_accuracy", "gain_top5_accuracy",
            "gain_rare_enrichment_5pct",
        ]
        gcols = [c for c in gcols if c in gain.columns]
        lines.append(gain[gcols].to_markdown(index=False, floatfmt=".4f"))
        lines += [""]

    lines += [
        "## Suggested manuscript wording",
        "",
        "```latex",
        r"\paragraph{Stage-aware early-aware inference.}",
        r"We integrate the complete-light-curve and partial-light-curve regimes through a stage-aware inference rule. Mature or complete light curves are assigned to the original full-curve AstroTrust-AI probability product, whereas sparse or temporally truncated alert-stage representations are assigned to an early-aware ensemble combining the early-aware hybrid temporal--tabular CNN with early-aware tabular members. This design avoids forcing a single classifier to operate across substantially different information regimes.",
        "```",
        "",
        "## Caveat",
        "",
        "If the default weights are used, describe this as a fixed-weight prototype inherited from the original ensemble design. A final publication claim can either report these fixed weights explicitly or add a separate validation-only weight-selection experiment.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(args):
    scenarios = parse_list(args.scenarios, DEFAULT_SCENARIOS)
    budgets = parse_float_list(args.budgets)
    rare_labels = parse_int_set(args.rare_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_dir.mkdir(parents=True, exist_ok=True)

    full_df = add_stage_features(read_table(find_feature_path(args.feature_dir, "full_curve_reference")), "full_curve_reference", args.include_stage_features)
    oid_full, y_full, X_full, feature_columns = prepare_xy(full_df)
    all_classes = np.array(sorted(np.unique(y_full)), dtype=int)

    idx_all = np.arange(len(y_full))
    train_val_idx, test_idx = train_test_split(idx_all, test_size=args.test_size, random_state=args.split_seed, stratify=y_full)
    train_idx, _ = train_test_split(train_val_idx, test_size=args.validation_size, random_state=args.split_seed, stratify=y_full[train_val_idx])
    if args.max_train_objects and args.max_train_objects > 0:
        train_idx = train_idx[:args.max_train_objects]
    train_ids = set(oid_full[train_idx])

    novelty_ref = fit_novelty_reference(X_full.iloc[train_idx])

    print("[1/4] Training early-aware tabular members...")
    X_train, y_train, manifest = build_training_table(args, train_ids, feature_columns)
    trained = {}
    member_rows = []
    model_names = []
    if not args.skip_lightgbm:
        model_names += ["lightgbm_early_aware", "lightgbm_regularized_early_aware"]
    if not args.skip_xgboost:
        model_names += ["xgboost_early_aware"]

    for name in model_names:
        t0 = time.time()
        try:
            model = make_model(name, args, n_classes=len(all_classes))
            model.fit(X_train, y_train)
            trained[name] = model
            elapsed = (time.time() - t0) / 60.0
            print(f"[OK] {name}: {elapsed:.2f} min")
            member_rows.append({"member": name, "status": "ok", "training_time_minutes": elapsed})
            try:
                import joblib
                joblib.dump(model, args.output_dir / f"{name}.joblib")
            except Exception as exc:
                print(f"[WARN] could not save {name}: {exc}")
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")
            member_rows.append({"member": name, "status": f"failed: {exc}", "training_time_minutes": np.nan})

    manifest_all = pd.concat([manifest.assign(kind="scenario_rows"), pd.DataFrame(member_rows).assign(kind="member_training")], ignore_index=True, sort=False)
    manifest_all.to_csv(args.output_dir / "stage_aware_ensemble_member_manifest.csv", index=False)

    base_weights = {
        "hybrid_early_aware": args.weight_hybrid,
        "lightgbm_early_aware": args.weight_lgbm,
        "lightgbm_regularized_early_aware": args.weight_lgbm_regularized,
        "xgboost_early_aware": args.weight_xgboost,
    }
    active_weights = normalized_weights(base_weights, {"hybrid_early_aware", *trained.keys()})

    print("[2/4] Evaluating scenarios and combining probabilities...")
    pred_map = {}
    prob_map = {}
    official_full_used = False

    for scenario in scenarios:
        print(f"\n[SCENARIO] {scenario}")
        h_pred, h_probs = load_hybrid_outputs(args.hybrid_dir, "hybrid_early_aware", scenario)
        object_ids = h_pred["object_id"].to_numpy()
        y_true = h_pred["true_label"].astype(int).to_numpy()

        df = add_stage_features(read_table(find_feature_path(args.feature_dir, scenario)), scenario, args.include_stage_features)
        oid_s, y_s, X_s, _ = prepare_xy(df, feature_columns)
        idx_table = pd.DataFrame({"object_id": oid_s, "__row": np.arange(len(oid_s))})
        order = pd.DataFrame({"object_id": object_ids}).merge(idx_table, on="object_id", how="left")
        if order["__row"].isna().any():
            raise RuntimeError(f"Could not align feature rows to hybrid predictions for {scenario}")
        rows = order["__row"].astype(int).to_numpy()
        X_eval = X_s.iloc[rows]
        novelty = compute_novelty(X_eval, novelty_ref)

        # Hybrid early-aware.
        h_frame = build_prediction_frame(object_ids, y_true, h_probs, novelty, rare_labels)
        save_prediction_and_probs(args.output_dir, "hybrid_early_aware", scenario, h_frame, h_probs)
        pred_map.setdefault("hybrid_early_aware", {})[scenario] = h_frame
        prob_map.setdefault("hybrid_early_aware", {})[scenario] = h_probs

        # Hybrid full-trained comparison.
        try:
            hf_pred0, hf_probs0 = load_hybrid_outputs(args.hybrid_dir, "hybrid_full_trained", scenario)
            hf_probs = align_probs_by_object(object_ids, hf_pred0["object_id"].to_numpy(), hf_probs0)
            hf_frame = build_prediction_frame(object_ids, y_true, hf_probs, novelty, rare_labels)
            save_prediction_and_probs(args.output_dir, "hybrid_full_trained", scenario, hf_frame, hf_probs)
            pred_map.setdefault("hybrid_full_trained", {})[scenario] = hf_frame
            prob_map.setdefault("hybrid_full_trained", {})[scenario] = hf_probs
        except Exception as exc:
            print(f"[WARN] hybrid_full_trained unavailable for {scenario}: {exc}")

        member_probs = {"hybrid_early_aware": h_probs}
        for name, model in trained.items():
            probs = align_probabilities(model.predict_proba(X_eval), model.classes_, all_classes)
            frame = build_prediction_frame(object_ids, y_true, probs, novelty, rare_labels)
            save_prediction_and_probs(args.output_dir, name, scenario, frame, probs)
            pred_map.setdefault(name, {})[scenario] = frame
            prob_map.setdefault(name, {})[scenario] = probs
            member_probs[name] = probs

        ens_probs = np.zeros_like(h_probs, dtype=np.float64)
        for name, weight in active_weights.items():
            ens_probs += weight * member_probs[name]
        ens_probs = sanitize_probabilities(ens_probs)
        ens_frame = build_prediction_frame(object_ids, y_true, ens_probs, novelty, rare_labels)
        save_prediction_and_probs(args.output_dir, "ensemble_hybrid_dominant_early_aware", scenario, ens_frame, ens_probs)
        pred_map.setdefault("ensemble_hybrid_dominant_early_aware", {})[scenario] = ens_frame
        prob_map.setdefault("ensemble_hybrid_dominant_early_aware", {})[scenario] = ens_probs

        # Stage-aware integrated route.
        if scenario == "full_curve_reference":
            official = try_official_full(args, object_ids, novelty, rare_labels)
            if official is not None:
                st_frame, st_probs, source = official
                official_full_used = True
            elif "hybrid_full_trained" in pred_map and scenario in pred_map["hybrid_full_trained"]:
                st_frame = pred_map["hybrid_full_trained"][scenario].copy()
                st_probs = prob_map["hybrid_full_trained"][scenario]
                source = "hybrid_full_trained_fallback"
            else:
                st_frame = ens_frame.copy()
                st_probs = ens_probs
                source = "early_aware_ensemble_fallback"
        else:
            st_frame = ens_frame.copy()
            st_probs = ens_probs
            source = "ensemble_hybrid_dominant_early_aware"

        st_frame["stage_aware_source"] = source
        save_prediction_and_probs(args.output_dir, "stage_aware_astrotrust_ai", scenario, st_frame, st_probs)
        pred_map.setdefault("stage_aware_astrotrust_ai", {})[scenario] = st_frame
        prob_map.setdefault("stage_aware_astrotrust_ai", {})[scenario] = st_probs

        print(f"[OK] ensemble_early acc={ens_frame['correct'].mean():.4f}, top5={top_k_accuracy(y_true, ens_probs, 5):.4f}")

    print("[3/4] Computing metrics...")
    class_rows = []
    follow_frames = []
    for model_name, scen_map in pred_map.items():
        for scenario, pred in scen_map.items():
            probs = prob_map[model_name][scenario]
            class_rows.append(classification_metrics(model_name, "all_available_objects", scenario, pred, probs))
            follow_frames.append(followup_metrics(model_name, "all_available_objects", scenario, pred, budgets))

        common = None
        for pred in scen_map.values():
            ids = set(pred["object_id"].to_numpy())
            common = ids if common is None else common.intersection(ids)
        common = common or set()
        if common:
            for scenario, pred in scen_map.items():
                mask = pred["object_id"].isin(common).to_numpy()
                pred_c = pred.loc[mask].copy()
                probs_c = prob_map[model_name][scenario][mask]
                class_rows.append(classification_metrics(model_name, "common_object_subset", scenario, pred_c, probs_c))
                follow_frames.append(followup_metrics(model_name, "common_object_subset", scenario, pred_c, budgets))

    classification = pd.DataFrame(class_rows)
    followup = pd.concat(follow_frames, ignore_index=True) if follow_frames else pd.DataFrame()

    scenario_order = {s: i for i, s in enumerate(DEFAULT_SCENARIOS)}
    model_order = {
        "hybrid_full_trained": 0,
        "hybrid_early_aware": 1,
        "lightgbm_early_aware": 2,
        "lightgbm_regularized_early_aware": 3,
        "xgboost_early_aware": 4,
        "ensemble_hybrid_dominant_early_aware": 5,
        "stage_aware_astrotrust_ai": 6,
    }
    classification["_scenario_order"] = classification["scenario"].map(scenario_order)
    classification["_model_order"] = classification["model_name"].map(model_order).fillna(99)
    classification = classification.sort_values(["comparison_set", "_scenario_order", "_model_order"]).drop(columns=["_scenario_order", "_model_order"])

    followup["_scenario_order"] = followup["scenario"].map(scenario_order)
    followup["_model_order"] = followup["model_name"].map(model_order).fillna(99)
    followup = followup.sort_values(["comparison_set", "policy", "budget_fraction", "_scenario_order", "_model_order"]).drop(columns=["_scenario_order", "_model_order"])

    gains = [
        gain_table(classification, followup, "ensemble_hybrid_dominant_early_aware", "hybrid_early_aware"),
        gain_table(classification, followup, "ensemble_hybrid_dominant_early_aware", "hybrid_full_trained"),
        gain_table(classification, followup, "stage_aware_astrotrust_ai", "hybrid_full_trained"),
    ]
    gain = pd.concat([g for g in gains if not g.empty], ignore_index=True) if any(not g.empty for g in gains) else pd.DataFrame()

    print("[4/4] Writing outputs...")
    class_path = args.output_dir / "stage_aware_ensemble_classification_metrics.csv"
    follow_path = args.output_dir / "stage_aware_ensemble_followup_metrics.csv"
    gain_path = args.output_dir / "stage_aware_ensemble_gain_tables.csv"

    classification.to_csv(class_path, index=False)
    followup.to_csv(follow_path, index=False)
    gain.to_csv(gain_path, index=False)
    summary_path = write_summary(args, classification, followup, gain, manifest_all, active_weights, official_full_used)

    for path in [class_path, follow_path, gain_path, args.output_dir / "stage_aware_ensemble_member_manifest.csv", summary_path]:
        shutil.copyfile(path, args.final_dir / path.name)

    print("\n[OK] Stage-aware early-aware ensemble finished.")
    print(f"[OK] Classification: {class_path}")
    print(f"[OK] Follow-up:       {follow_path}")
    print(f"[OK] Gains:           {gain_path}")
    print(f"[OK] Summary:         {summary_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--hybrid-dir", type=Path, default=DEFAULT_HYBRID_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    p.add_argument("--official-full-predictions", type=Path, default=DEFAULT_OFFICIAL_FULL_PRED)
    p.add_argument("--official-full-probabilities", type=Path, default=DEFAULT_OFFICIAL_FULL_PROB)
    p.add_argument("--use-official-full-ensemble", action="store_true", default=True)
    p.add_argument("--no-official-full-ensemble", dest="use_official_full_ensemble", action="store_false")

    p.add_argument("--scenarios", type=str, default="all")
    p.add_argument("--train-scenarios", type=str, default="all")
    p.add_argument("--include-stage-features", action="store_true", default=True)
    p.add_argument("--no-stage-features", dest="include_stage_features", action="store_false")
    p.add_argument("--max-train-rows", type=int, default=500000)
    p.add_argument("--max-train-objects", type=int, default=None)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--validation-size", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budgets", type=str, default="0.01,0.02,0.05,0.10,0.20")
    p.add_argument("--rare-labels", type=str, default=DEFAULT_RARE_LABELS)

    p.add_argument("--skip-lightgbm", action="store_true")
    p.add_argument("--skip-xgboost", action="store_true")
    p.add_argument("--class-weight-balanced", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument("--weight-hybrid", type=float, default=0.70)
    p.add_argument("--weight-lgbm", type=float, default=0.15)
    p.add_argument("--weight-lgbm-regularized", type=float, default=0.10)
    p.add_argument("--weight-xgboost", type=float, default=0.05)

    p.add_argument("--lgbm-n-estimators", type=int, default=650)
    p.add_argument("--lgbm-learning-rate", type=float, default=0.035)
    p.add_argument("--lgbm-num-leaves", type=int, default=63)
    p.add_argument("--lgbm-min-child-samples", type=int, default=30)

    p.add_argument("--lgbm-reg-n-estimators", type=int, default=650)
    p.add_argument("--lgbm-reg-learning-rate", type=float, default=0.030)
    p.add_argument("--lgbm-reg-num-leaves", type=int, default=31)
    p.add_argument("--lgbm-reg-min-child-samples", type=int, default=80)

    p.add_argument("--xgb-n-estimators", type=int, default=450)
    p.add_argument("--xgb-learning-rate", type=float, default=0.035)
    p.add_argument("--xgb-max-depth", type=int, default=8)
    p.add_argument("--xgb-tree-method", type=str, default="hist")
    return p.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
