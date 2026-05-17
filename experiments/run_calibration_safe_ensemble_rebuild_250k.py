#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_calibration_safe_ensemble_rebuild_250k.py

Rebuilds a calibration-safe version of the final AstroTrust-AI ensemble:

1. Uses the same ELAsTiCC 250k tensor dataset.
2. Reconstructs the three-way split:
   - train
   - validation
   - test
3. Loads the trained hybrid temporal-tabular CNN checkpoint and exports raw validation/test probabilities.
4. Trains tabular members ONLY on train, not on validation:
   - lightgbm_baseline
   - lightgbm_regularized
   - xgboost_gpu_deeper
5. Predicts validation and test probabilities for all members.
6. Reconstructs the dominant ensemble:
   0.70 hybrid + 0.15 lightgbm_baseline + 0.10 lightgbm_regularized + 0.05 xgboost_gpu_deeper
7. Fits temperature scaling ONLY on validation ensemble probabilities.
8. Applies the fitted temperature to test ensemble probabilities.
9. Reports raw-vs-temperature calibration metrics and bootstrap CIs.

This is the cleanest way to satisfy the manuscript requirement that calibration
be evaluated for an ensemble using a validation split rather than the test set.

Important methodological note:
The original ensemble test probabilities may have been produced by tabular members
trained on train+validation. This script intentionally retrains the tabular members
on train only, so it creates a calibration-safe ensemble variant. The script reports
differences against the previously official ensemble test probabilities when provided.

Usage:
python .\experiments\run_calibration_safe_ensemble_rebuild_250k.py `
  --project-root . `
  --output-dir results\ensemble_temperature_calibration_safe_rebuild_250k_final `
  --n-bootstrap 1000 `
  --n-bins 15 `
  --xgb-device cpu

Optional:
  --official-ensemble-test-probs results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_test_probabilities.npy
  --official-ensemble-test-predictions results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_predictions.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

EPS = 1e-12


def import_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=EPS, posinf=1.0, neginf=EPS)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def temp_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    probs = normalize_probs(probs)
    logp = np.log(np.clip(probs, EPS, 1.0)) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(y: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    one = np.zeros_like(probs)
    one[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - one) ** 2, axis=1)))


def ece(y: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
    probs = normalize_probs(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    n = len(y)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if np.any(mask):
            out += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())

    return float(out)


def topk_accuracy(y: np.ndarray, probs: np.ndarray, k: int) -> float:
    top = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def compute_metrics(y: np.ndarray, probs: np.ndarray, n_bins: int) -> dict[str, float]:
    probs = normalize_probs(probs)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "top3_accuracy": topk_accuracy(y, probs, 3),
        "top5_accuracy": topk_accuracy(y, probs, 5),
        "mean_confidence": float(conf.mean()),
        "calibration_gap_abs": float(abs(np.mean(pred == y) - conf.mean())),
        "ece": ece(y, probs, n_bins),
        "brier_score": brier(y, probs),
        "negative_log_likelihood": nll(y, probs),
    }


def reliability_bins(y: np.ndarray, probs: np.ndarray, n_bins: int) -> pd.DataFrame:
    probs = normalize_probs(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        rows.append({
            "bin": i,
            "bin_low": lo,
            "bin_high": hi,
            "n": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_confidence": float(conf[mask].mean()) if np.any(mask) else np.nan,
            "accuracy": float(correct[mask].mean()) if np.any(mask) else np.nan,
        })

    return pd.DataFrame(rows)


def fit_temperature(y_val: np.ndarray, probs_val: np.ndarray) -> tuple[float, float]:
    def obj(t: float) -> float:
        return nll(y_val, temp_scale_probs(probs_val, t))

    grid = np.exp(np.linspace(np.log(0.2), np.log(5.0), 300))
    values = np.array([obj(float(t)) for t in grid])
    best_i = int(values.argmin())
    best_t = float(grid[best_i])
    best_v = float(values[best_i])

    try:
        from scipy.optimize import minimize_scalar
        lo = float(grid[max(0, best_i - 4)])
        hi = float(grid[min(len(grid) - 1, best_i + 4)])
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
        if res.success and math.isfinite(float(res.fun)):
            best_t = float(res.x)
            best_v = float(res.fun)
    except Exception:
        pass

    return best_t, best_v


def bootstrap_deltas(
    y: np.ndarray,
    raw: np.ndarray,
    temp: np.ndarray,
    n_bootstrap: int,
    n_bins: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx_all = np.arange(n)

    raw_point = compute_metrics(y, raw, n_bins)
    temp_point = compute_metrics(y, temp, n_bins)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
        "top5_accuracy",
        "mean_confidence",
        "calibration_gap_abs",
        "ece",
        "brier_score",
        "negative_log_likelihood",
    ]

    lower_better = {"calibration_gap_abs", "ece", "brier_score", "negative_log_likelihood"}

    rows = []
    for metric in metric_names:
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.choice(idx_all, size=n, replace=True)
            samples[b] = (
                compute_metrics(y[idx], temp[idx], n_bins)[metric]
                - compute_metrics(y[idx], raw[idx], n_bins)[metric]
            )

        if metric in lower_better:
            p_no_improvement = float(np.mean(samples >= 0))
            direction = "lower_is_better"
        elif metric in {"mean_confidence"}:
            p_no_improvement = np.nan
            direction = "descriptive"
        else:
            p_no_improvement = float(np.mean(samples <= 0))
            direction = "higher_is_better"

        rows.append({
            "metric": metric,
            "raw": raw_point[metric],
            "temperature_scaled": temp_point[metric],
            "delta_temperature_minus_raw": temp_point[metric] - raw_point[metric],
            "bootstrap_ci_low_95": float(np.quantile(samples, 0.025)),
            "bootstrap_ci_high_95": float(np.quantile(samples, 0.975)),
            "direction": direction,
            "one_sided_bootstrap_pvalue_no_improvement": p_no_improvement,
            "n_bootstrap": n_bootstrap,
        })

    return pd.DataFrame(rows)


def align_probabilities(probabilities: np.ndarray, model_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    aligned = np.zeros((probabilities.shape[0], len(all_classes)), dtype=np.float64)

    class_to_col = {int(c): i for i, c in enumerate(model_classes)}
    for j, cls in enumerate(all_classes):
        if int(cls) in class_to_col:
            aligned[:, j] = probabilities[:, class_to_col[int(cls)]]

    return normalize_probs(aligned)


def make_feature_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=[f"feature_{i:03d}" for i in range(X.shape[1])])


def train_tabular_predict_val_test(
    name: str,
    model: Any,
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    print("\n" + "=" * 80)
    print(f"Training calibration-safe tabular member: {name}")
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    start = time.time()
    model.fit(X_train_df, y_train, sample_weight=sample_weight)
    elapsed_min = (time.time() - start) / 60.0

    print(f"Training time: {elapsed_min:.2f} min")
    val_probs = align_probabilities(model.predict_proba(X_val_df), model.classes_, all_classes)
    test_probs = align_probabilities(model.predict_proba(X_test_df), model.classes_, all_classes)

    return val_probs, test_probs, elapsed_min


def save_prob(out: Path, name: str, split: str, probs: np.ndarray) -> Path:
    path = out / f"{name}_{split}_probabilities.npy"
    np.save(path, normalize_probs(probs).astype(np.float32))
    return path


def plot_reliability(y: np.ndarray, raw: np.ndarray, temp: np.ndarray, n_bins: int, out: Path) -> None:
    raw_bins = reliability_bins(y, raw, n_bins)
    temp_bins = reliability_bins(y, temp, n_bins)
    raw_bins["variant"] = "raw"
    temp_bins["variant"] = "temperature_scaled"
    bins = pd.concat([raw_bins, temp_bins], ignore_index=True)
    bins.to_csv(out / "ensemble_calibration_safe_reliability_bins.csv", index=False)

    plt.figure(figsize=(5.5, 5.0))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    for variant, sub in bins.groupby("variant"):
        sub = sub.dropna(subset=["mean_confidence", "accuracy"])
        plt.plot(sub["mean_confidence"], sub["accuracy"], marker="o", label=variant)
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration-safe ensemble reliability")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_calibration_safe_ensemble_reliability.png", dpi=220)
    plt.close()


def plot_metrics(boot: pd.DataFrame, out: Path) -> None:
    names = ["ece", "brier_score", "negative_log_likelihood", "calibration_gap_abs"]
    sub = boot[boot["metric"].isin(names)].set_index("metric").loc[names].reset_index()
    x = np.arange(len(sub))
    width = 0.35

    plt.figure(figsize=(7.0, 4.2))
    plt.bar(x - width / 2, sub["raw"], width, label="raw")
    plt.bar(x + width / 2, sub["temperature_scaled"], width, label="temperature_scaled")
    plt.xticks(x, sub["metric"], rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.title("Calibration-safe ensemble: raw vs temperature-scaled")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_calibration_safe_ensemble_metric_comparison.png", dpi=220)
    plt.close()


def compare_to_official(
    official_probs_path: Path | None,
    official_preds_path: Path | None,
    rebuilt_probs: np.ndarray,
    object_ids_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "official_ensemble_test_probs_path": str(official_probs_path) if official_probs_path else None,
        "official_ensemble_test_predictions_path": str(official_preds_path) if official_preds_path else None,
        "compared": False,
    }

    if official_probs_path is None or not official_probs_path.exists():
        report["reason"] = "official probability file not provided or not found"
        return report

    official = normalize_probs(np.load(official_probs_path))
    if official.shape != rebuilt_probs.shape:
        report["reason"] = f"shape mismatch: official={official.shape}, rebuilt={rebuilt_probs.shape}"
        return report

    report["compared"] = True
    report["max_abs_probability_difference"] = float(np.max(np.abs(official - rebuilt_probs)))
    report["mean_abs_probability_difference"] = float(np.mean(np.abs(official - rebuilt_probs)))
    report["top1_agreement_fraction"] = float(np.mean(official.argmax(axis=1) == rebuilt_probs.argmax(axis=1)))
    report["official_accuracy_from_probs"] = float(np.mean(official.argmax(axis=1) == y_test))
    report["rebuilt_accuracy_from_probs"] = float(np.mean(rebuilt_probs.argmax(axis=1) == y_test))

    if official_preds_path is not None and official_preds_path.exists():
        pred_df = pd.read_csv(official_preds_path)
        report["official_predictions_rows"] = int(len(pred_df))
        if "object_id" in pred_df.columns:
            report["object_id_order_matches_official_predictions"] = bool(np.array_equal(pred_df["object_id"].to_numpy(), object_ids_test))
        if "true_label" in pred_df.columns:
            report["true_label_order_matches_official_predictions"] = bool(np.array_equal(pred_df["true_label"].to_numpy(), y_test))

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--tensor-path", default=None)
    parser.add_argument("--hybrid-checkpoint", default=None)
    parser.add_argument("--split-metadata", default="results/hybrid_inference_artifacts_250k/split_metadata.csv")
    parser.add_argument("--official-ensemble-test-probs", default="results/hybrid_tabular_ensemble_250k/ensemble_hybrid_dominant_test_probabilities.npy")
    parser.add_argument("--official-ensemble-test-predictions", default="results/hybrid_tabular_ensemble_250k/ensemble_hybrid_dominant_predictions.csv")
    parser.add_argument("--output-dir", default="results/ensemble_temperature_calibration_safe_rebuild_250k_final")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size-from-train", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-if-probabilities-exist", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hybrid_mod = import_from_path(
        "astrotrust_hybrid_followup_module",
        root / "experiments" / "run_hybrid_followup_policy_eval_250k.py",
    )
    ensemble_mod = import_from_path(
        "astrotrust_ensemble_scaling_module",
        root / "experiments" / "run_hybrid_tabular_ensemble_scaling_with_md.py",
    )

    tensor_path = Path(args.tensor_path) if args.tensor_path else Path(getattr(hybrid_mod, "TENSOR_PATH"))
    checkpoint_path = Path(args.hybrid_checkpoint) if args.hybrid_checkpoint else Path(getattr(hybrid_mod, "CHECKPOINT_PATH"))

    print(f"Tensor path:     {tensor_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Output dir:      {out}")

    data = np.load(tensor_path, allow_pickle=True)
    X_lc = data["X_lc"].astype(np.float32)
    X_tab = data["X_tab"].astype(np.float32)
    y = data["y"].astype(np.int64)
    object_ids = data["object_id"].astype(np.int64) if "object_id" in data else np.arange(len(y))

    train_plus_val_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_plus_val_idx,
        test_size=args.val_size_from_train,
        random_state=args.split_seed,
        stratify=y[train_plus_val_idx],
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    object_ids_val = object_ids[val_idx]
    object_ids_test = object_ids[test_idx]

    split_df = pd.DataFrame({
        "object_id": np.concatenate([object_ids[train_idx], object_ids[val_idx], object_ids[test_idx]]),
        "label": np.concatenate([y_train, y_val, y_test]),
        "split": (
            ["train"] * len(train_idx)
            + ["validation"] * len(val_idx)
            + ["test"] * len(test_idx)
        ),
    })
    split_df.to_csv(out / "calibration_safe_split_metadata.csv", index=False)

    split_check = {
        "n_total": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "split_metadata_path": args.split_metadata,
        "split_metadata_compared": False,
    }

    sm_path = root / args.split_metadata
    if sm_path.exists():
        sm = pd.read_csv(sm_path)
        split_check["split_metadata_compared"] = True
        for split_name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
            expected = set(map(int, object_ids[idx]))
            got = set(map(int, sm.loc[sm["split"].astype(str).str.lower().eq(split_name), "object_id"]))
            split_check[f"{split_name}_object_id_set_matches_existing_split_metadata"] = bool(expected == got)
            split_check[f"{split_name}_expected_n"] = int(len(expected))
            split_check[f"{split_name}_metadata_n"] = int(len(got))

    all_classes = np.asarray(sorted(np.unique(y)), dtype=int)
    n_classes = len(all_classes)

    print("\nSplit summary:")
    print(split_check)

    # Hybrid raw probabilities.
    hybrid_val_path = out / "hybrid_cnn_tabular_validation_probabilities.npy"
    hybrid_test_path = out / "hybrid_cnn_tabular_test_probabilities.npy"

    if args.skip_if_probabilities_exist and hybrid_val_path.exists() and hybrid_test_path.exists():
        print("Loading existing hybrid validation/test probabilities.")
        hybrid_val_probs = normalize_probs(np.load(hybrid_val_path))
        hybrid_test_probs = normalize_probs(np.load(hybrid_test_path))
    else:
        scaler = StandardScaler()
        X_tab_train_scaled = scaler.fit_transform(X_tab[train_idx]).astype(np.float32)
        X_tab_val_scaled = scaler.transform(X_tab[val_idx]).astype(np.float32)
        X_tab_test_scaled = scaler.transform(X_tab[test_idx]).astype(np.float32)

        X_tab_train_scaled = np.nan_to_num(X_tab_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_tab_val_scaled = np.nan_to_num(X_tab_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_tab_test_scaled = np.nan_to_num(X_tab_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        val_dataset = hybrid_mod.HybridDataset(X_lc[val_idx], X_tab_val_scaled, y_val)
        test_dataset = hybrid_mod.HybridDataset(X_lc[test_idx], X_tab_test_scaled, y_test)

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {device}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = hybrid_mod.HybridTemporalTabularCNN(
            n_channels=int(checkpoint["n_channels"]),
            n_tabular=int(checkpoint["n_tabular"]),
            n_classes=int(checkpoint["n_classes"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("\nGenerating hybrid validation/test logits...")
        logits_val, y_val_torch = hybrid_mod.get_logits(model, val_loader, device)
        logits_test, y_test_torch = hybrid_mod.get_logits(model, test_loader, device)

        if not np.array_equal(y_val_torch.numpy(), y_val):
            raise RuntimeError("Hybrid validation labels do not match split labels.")
        if not np.array_equal(y_test_torch.numpy(), y_test):
            raise RuntimeError("Hybrid test labels do not match split labels.")

        hybrid_val_probs = normalize_probs(torch.softmax(logits_val, dim=1).numpy())
        hybrid_test_probs = normalize_probs(torch.softmax(logits_test, dim=1).numpy())

        np.save(hybrid_val_path, hybrid_val_probs.astype(np.float32))
        np.save(hybrid_test_path, hybrid_test_probs.astype(np.float32))

    # Tabular model training on train only.
    X_train_df = make_feature_frame(np.nan_to_num(X_tab[train_idx], nan=0.0, posinf=0.0, neginf=0.0))
    X_val_df = make_feature_frame(np.nan_to_num(X_tab[val_idx], nan=0.0, posinf=0.0, neginf=0.0))
    X_test_df = make_feature_frame(np.nan_to_num(X_tab[test_idx], nan=0.0, posinf=0.0, neginf=0.0))

    probabilities_val_by_model = {"hybrid_cnn_tabular": hybrid_val_probs}
    probabilities_test_by_model = {"hybrid_cnn_tabular": hybrid_test_probs}

    training_rows = []
    tabular_specs = [
        ("lightgbm_baseline", ensemble_mod.make_lightgbm_baseline(n_classes)),
        ("lightgbm_regularized", ensemble_mod.make_lightgbm_regularized(n_classes)),
        ("xgboost_gpu_deeper", ensemble_mod.make_xgboost_deeper(n_classes, device=args.xgb_device)),
    ]

    for name, model_obj in tabular_specs:
        val_path = out / f"{name}_validation_probabilities.npy"
        test_path = out / f"{name}_test_probabilities.npy"

        if args.skip_if_probabilities_exist and val_path.exists() and test_path.exists():
            print(f"Loading existing probabilities for {name}.")
            val_probs = normalize_probs(np.load(val_path))
            test_probs = normalize_probs(np.load(test_path))
            elapsed = np.nan
        else:
            val_probs, test_probs, elapsed = train_tabular_predict_val_test(
                name=name,
                model=model_obj,
                X_train_df=X_train_df,
                X_val_df=X_val_df,
                X_test_df=X_test_df,
                y_train=y_train,
                all_classes=all_classes,
            )
            np.save(val_path, val_probs.astype(np.float32))
            np.save(test_path, test_probs.astype(np.float32))

        probabilities_val_by_model[name] = val_probs
        probabilities_test_by_model[name] = test_probs
        training_rows.append({
            "model": name,
            "training_time_minutes": elapsed,
            **compute_metrics(y_test, test_probs, args.n_bins),
        })

    # Save hybrid metrics as well.
    training_rows.insert(0, {
        "model": "hybrid_cnn_tabular",
        "training_time_minutes": np.nan,
        **compute_metrics(y_test, hybrid_test_probs, args.n_bins),
    })

    ensemble_configs = {
        "ensemble_hybrid_lgbm_equal": {
            "hybrid_cnn_tabular": 0.50,
            "lightgbm_baseline": 0.50,
        },
        "ensemble_hybrid_lgbm_hybrid60": {
            "hybrid_cnn_tabular": 0.60,
            "lightgbm_baseline": 0.40,
        },
        "ensemble_hybrid_lgbm_hybrid70": {
            "hybrid_cnn_tabular": 0.70,
            "lightgbm_baseline": 0.30,
        },
        "ensemble_hybrid_lgbm_lgbm60": {
            "hybrid_cnn_tabular": 0.40,
            "lightgbm_baseline": 0.60,
        },
        "ensemble_hybrid_lgbmreg_equal": {
            "hybrid_cnn_tabular": 0.50,
            "lightgbm_regularized": 0.50,
        },
        "ensemble_hybrid_xgb_equal": {
            "hybrid_cnn_tabular": 0.50,
            "xgboost_gpu_deeper": 0.50,
        },
        "ensemble_hybrid_lgbm_xgb": {
            "hybrid_cnn_tabular": 0.50,
            "lightgbm_baseline": 0.30,
            "xgboost_gpu_deeper": 0.20,
        },
        "ensemble_hybrid_all_equal": {
            "hybrid_cnn_tabular": 0.25,
            "lightgbm_baseline": 0.25,
            "lightgbm_regularized": 0.25,
            "xgboost_gpu_deeper": 0.25,
        },
        "ensemble_hybrid_dominant": {
            "hybrid_cnn_tabular": 0.70,
            "lightgbm_baseline": 0.15,
            "lightgbm_regularized": 0.10,
            "xgboost_gpu_deeper": 0.05,
        },
        "ensemble_balanced_strong": {
            "hybrid_cnn_tabular": 0.60,
            "lightgbm_baseline": 0.25,
            "lightgbm_regularized": 0.10,
            "xgboost_gpu_deeper": 0.05,
        },
    }

    for ensemble_name, weights in ensemble_configs.items():
        total_weight = sum(weights.values())
        val_probs = np.zeros_like(hybrid_val_probs, dtype=np.float64)
        test_probs = np.zeros_like(hybrid_test_probs, dtype=np.float64)

        for member, weight in weights.items():
            val_probs += (weight / total_weight) * probabilities_val_by_model[member]
            test_probs += (weight / total_weight) * probabilities_test_by_model[member]

        val_probs = normalize_probs(val_probs)
        test_probs = normalize_probs(test_probs)

        save_prob(out, ensemble_name, "validation", val_probs)
        save_prob(out, ensemble_name, "test", test_probs)

        training_rows.append({
            "model": ensemble_name,
            "training_time_minutes": np.nan,
            "members": ",".join(weights.keys()),
            "weights": ",".join([f"{k}:{v}" for k, v in weights.items()]),
            **compute_metrics(y_test, test_probs, args.n_bins),
        })

    model_metrics = pd.DataFrame(training_rows)
    model_metrics.to_csv(out / "calibration_safe_rebuilt_model_test_metrics.csv", index=False)

    ensemble_val = normalize_probs(np.load(out / "ensemble_hybrid_dominant_validation_probabilities.npy"))
    ensemble_test = normalize_probs(np.load(out / "ensemble_hybrid_dominant_test_probabilities.npy"))

    comparison_report = compare_to_official(
        official_probs_path=root / args.official_ensemble_test_probs if args.official_ensemble_test_probs else None,
        official_preds_path=root / args.official_ensemble_test_predictions if args.official_ensemble_test_predictions else None,
        rebuilt_probs=ensemble_test,
        object_ids_test=object_ids_test,
        y_test=y_test,
    )

    T, val_nll = fit_temperature(y_val, ensemble_val)
    ensemble_val_temp = temp_scale_probs(ensemble_val, T)
    ensemble_test_temp = temp_scale_probs(ensemble_test, T)

    np.save(out / "ensemble_hybrid_dominant_validation_temperature_scaled_probabilities.npy", ensemble_val_temp.astype(np.float32))
    np.save(out / "ensemble_hybrid_dominant_test_temperature_scaled_probabilities.npy", ensemble_test_temp.astype(np.float32))

    point_rows = []
    for split_name, yy, raw, temp in [
        ("validation", y_val, ensemble_val, ensemble_val_temp),
        ("test", y_test, ensemble_test, ensemble_test_temp),
    ]:
        point_rows.append({"split": split_name, "variant": "raw", **compute_metrics(yy, raw, args.n_bins)})
        point_rows.append({"split": split_name, "variant": "temperature_scaled", **compute_metrics(yy, temp, args.n_bins)})

    point_metrics = pd.DataFrame(point_rows)
    point_metrics.to_csv(out / "calibration_safe_ensemble_temperature_point_metrics.csv", index=False)

    boot = bootstrap_deltas(
        y=y_test,
        raw=ensemble_test,
        temp=ensemble_test_temp,
        n_bootstrap=args.n_bootstrap,
        n_bins=args.n_bins,
        seed=args.seed,
    )
    boot.to_csv(out / "calibration_safe_ensemble_temperature_bootstrap_deltas.csv", index=False)

    plot_reliability(y_test, ensemble_test, ensemble_test_temp, args.n_bins, out)
    plot_metrics(boot, out)

    predictions = pd.DataFrame({
        "object_id": object_ids_test,
        "true_label": y_test,
        "raw_predicted_label": ensemble_test.argmax(axis=1),
        "temperature_predicted_label": ensemble_test_temp.argmax(axis=1),
        "raw_confidence": ensemble_test.max(axis=1),
        "temperature_confidence": ensemble_test_temp.max(axis=1),
        "raw_correct": ensemble_test.argmax(axis=1) == y_test,
        "temperature_correct": ensemble_test_temp.argmax(axis=1) == y_test,
    })
    predictions.to_csv(out / "calibration_safe_ensemble_test_predictions_raw_vs_temperature.csv", index=False)

    schema = {
        "analysis": "calibration-safe ensemble rebuild and temperature scaling",
        "methodological_status": "calibration-safe ensemble variant; tabular members trained on train only, validation used only for temperature fitting",
        "tensor_path": str(tensor_path),
        "hybrid_checkpoint": str(checkpoint_path),
        "split": split_check,
        "weights": ensemble_configs["ensemble_hybrid_dominant"],
        "temperature": float(T),
        "validation_nll_at_temperature": float(val_nll),
        "n_bins": int(args.n_bins),
        "n_bootstrap": int(args.n_bootstrap),
        "official_comparison": comparison_report,
        "no_test_set_temperature_fitting": True,
        "outputs": {
            "model_metrics": str(out / "calibration_safe_rebuilt_model_test_metrics.csv"),
            "point_metrics": str(out / "calibration_safe_ensemble_temperature_point_metrics.csv"),
            "bootstrap_deltas": str(out / "calibration_safe_ensemble_temperature_bootstrap_deltas.csv"),
        },
    }
    (out / "calibration_safe_ensemble_temperature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    raw_test = point_metrics[(point_metrics["split"] == "test") & (point_metrics["variant"] == "raw")].iloc[0]
    temp_test = point_metrics[(point_metrics["split"] == "test") & (point_metrics["variant"] == "temperature_scaled")].iloc[0]

    def delta(metric: str) -> pd.Series:
        return boot[boot["metric"] == metric].iloc[0]

    d_ece = delta("ece")
    d_brier = delta("brier_score")
    d_nll = delta("negative_log_likelihood")
    d_acc = delta("accuracy")

    md = []
    md.append("# Calibration-safe ensemble rebuild and temperature scaling\n")
    md.append("This report rebuilds a calibration-safe version of the final `ensemble_hybrid_dominant` ensemble. The tabular members are trained only on the training split, the validation split is used only to fit the temperature parameter, and the held-out test split is used only for evaluation.\n")
    md.append("## Protocol\n")
    md.append(f"- Tensor dataset: `{tensor_path}`")
    md.append(f"- Hybrid checkpoint: `{checkpoint_path}`")
    md.append(f"- Train objects: `{len(train_idx)}`")
    md.append(f"- Validation objects: `{len(val_idx)}`")
    md.append(f"- Test objects: `{len(test_idx)}`")
    md.append(f"- Number of classes: `{n_classes}`")
    md.append(f"- Dominant ensemble weights: `{ensemble_configs['ensemble_hybrid_dominant']}`")
    md.append(f"- Temperature fitted on validation: `{T:.6f}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`")
    md.append("- No temperature parameter was fitted on the test set.\n")

    md.append("## Split metadata check\n")
    md.append(pd.DataFrame([split_check]).to_markdown(index=False))
    md.append("")

    md.append("## Comparison against previous official ensemble test probabilities\n")
    md.append(pd.DataFrame([comparison_report]).to_markdown(index=False))
    md.append("")
    md.append("> If the rebuilt probabilities differ from the previous official probabilities, this is expected when the previous tabular members were trained on train+validation. The rebuilt variant is the calibration-safe ensemble for temperature-scaling analysis.\n")

    md.append("## Rebuilt model test metrics\n")
    md.append(model_metrics.to_markdown(index=False))
    md.append("")

    md.append("## Ensemble raw vs temperature-scaled point metrics\n")
    md.append(point_metrics.to_markdown(index=False))
    md.append("")

    md.append("## Test-set bootstrap deltas\n")
    md.append(boot.to_markdown(index=False))
    md.append("")

    md.append("## Key interpretation\n")
    md.append(f"- Rebuilt raw ensemble test accuracy: `{raw_test['accuracy']:.6f}`.")
    md.append(f"- Rebuilt temperature-scaled ensemble test accuracy: `{temp_test['accuracy']:.6f}`.")
    md.append(f"- Accuracy delta: `{float(d_acc['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(d_acc['bootstrap_ci_low_95']):.6f}`, `{float(d_acc['bootstrap_ci_high_95']):.6f}`].")
    md.append(f"- ECE changed from `{raw_test['ece']:.6f}` to `{temp_test['ece']:.6f}`; delta `{float(d_ece['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(d_ece['bootstrap_ci_low_95']):.6f}`, `{float(d_ece['bootstrap_ci_high_95']):.6f}`].")
    md.append(f"- Brier score changed from `{raw_test['brier_score']:.6f}` to `{temp_test['brier_score']:.6f}`; delta `{float(d_brier['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(d_brier['bootstrap_ci_low_95']):.6f}`, `{float(d_brier['bootstrap_ci_high_95']):.6f}`].")
    md.append(f"- NLL changed from `{raw_test['negative_log_likelihood']:.6f}` to `{temp_test['negative_log_likelihood']:.6f}`; delta `{float(d_nll['delta_temperature_minus_raw']):.6f}` with 95% CI [`{float(d_nll['bootstrap_ci_low_95']):.6f}`, `{float(d_nll['bootstrap_ci_high_95']):.6f}`].")
    md.append("")

    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    md.append(
        f"To evaluate calibration for the final ensemble without test-set leakage, we rebuilt a calibration-safe "
        f"\\texttt{{ensemble\\_hybrid\\_dominant}} variant in which tabular members were trained only on the training split, "
        f"the validation split was used exclusively to fit the temperature parameter, and the held-out test split was used only for evaluation. "
        f"The dominant ensemble retained the same weights as the final system "
        f"(0.70 hybrid, 0.15 LightGBM baseline, 0.10 regularized LightGBM, and 0.05 XGBoost). "
        f"The fitted temperature was \\textbf{{{T:.4f}}}. On the test set, ECE changed from "
        f"\\textbf{{{raw_test['ece']:.4f}}} to \\textbf{{{temp_test['ece']:.4f}}} "
        f"(bootstrap delta = \\textbf{{{float(d_ece['delta_temperature_minus_raw']):+.4f}}}; "
        f"95\\% CI [{float(d_ece['bootstrap_ci_low_95']):.4f}, {float(d_ece['bootstrap_ci_high_95']):.4f}]). "
        f"The Brier score changed from \\textbf{{{raw_test['brier_score']:.4f}}} to "
        f"\\textbf{{{temp_test['brier_score']:.4f}}}, and the negative log-likelihood changed from "
        f"\\textbf{{{raw_test['negative_log_likelihood']:.4f}}} to "
        f"\\textbf{{{temp_test['negative_log_likelihood']:.4f}}}. "
        f"No temperature parameter was fitted on the test set."
    )
    md.append("```")
    md.append("")

    md.append("## Output files\n")
    for name in [
        "hybrid_cnn_tabular_validation_probabilities.npy",
        "hybrid_cnn_tabular_test_probabilities.npy",
        "lightgbm_baseline_validation_probabilities.npy",
        "lightgbm_baseline_test_probabilities.npy",
        "lightgbm_regularized_validation_probabilities.npy",
        "lightgbm_regularized_test_probabilities.npy",
        "xgboost_gpu_deeper_validation_probabilities.npy",
        "xgboost_gpu_deeper_test_probabilities.npy",
        "ensemble_hybrid_dominant_validation_probabilities.npy",
        "ensemble_hybrid_dominant_test_probabilities.npy",
        "ensemble_hybrid_dominant_validation_temperature_scaled_probabilities.npy",
        "ensemble_hybrid_dominant_test_temperature_scaled_probabilities.npy",
        "calibration_safe_rebuilt_model_test_metrics.csv",
        "calibration_safe_ensemble_temperature_point_metrics.csv",
        "calibration_safe_ensemble_temperature_bootstrap_deltas.csv",
        "calibration_safe_ensemble_test_predictions_raw_vs_temperature.csv",
        "calibration_safe_ensemble_temperature_schema.json",
        "fig_calibration_safe_ensemble_reliability.png",
        "fig_calibration_safe_ensemble_metric_comparison.png",
    ]:
        md.append(f"- `{out / name}`")

    summary_path = out / "calibration_safe_ensemble_temperature_summary.md"
    summary_path.write_text("\n".join(md), encoding="utf-8")

    print("\nDone.")
    print(f"Temperature: {T:.6f}")
    print(f"Summary: {summary_path}")
    print(f"Output dir: {out}")


if __name__ == "__main__":
    main()
