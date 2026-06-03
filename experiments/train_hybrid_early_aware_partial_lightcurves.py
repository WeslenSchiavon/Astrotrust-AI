#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the AstroTrust-AI hybrid temporal-tabular CNN in early-aware mode.

This script uses the same core neural architecture as the existing
`hybrid_temporal_tabular_cnn`: a temporal 1-D CNN branch over light-curve
tensors plus a tabular branch over V4 features. The difference is the training
protocol: the early-aware model is trained with complete and temporally
truncated representations of the training objects.

Required previous step:
  python experiments/build_broker_like_partial_temporal_tensors.py

Main run:
  python experiments/train_hybrid_early_aware_partial_lightcurves.py

Lower-cost first run:
  python experiments/train_hybrid_early_aware_partial_lightcurves.py ^
    --max-train-rows 300000 --epochs 35 --patience 6

Smoke test:
  python experiments/train_hybrid_early_aware_partial_lightcurves.py --self-test

Outputs:
  results/broker_like_hybrid_early_aware/hybrid_early_aware_classification_metrics.csv
  results/broker_like_hybrid_early_aware/hybrid_early_aware_followup_metrics.csv
  results/broker_like_hybrid_early_aware/hybrid_early_aware_gain_over_full_trained.csv
  results/broker_like_hybrid_early_aware/hybrid_early_aware_summary.md
  results/broker_like_hybrid_early_aware/predictions/{model}/{scenario}_predictions.csv
  results/broker_like_hybrid_early_aware/probabilities/{model}/{scenario}_probabilities.npy

Important interpretation:
  This trains the same hybrid temporal-tabular CNN family under an early-aware
  protocol. It is not yet the full dominant ensemble until combined with
  early-aware tabular members.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_TENSOR_DIR = ROOT_DIR / "data" / "processed" / "broker_like_partial_tensors"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "broker_like_hybrid_early_aware"
DEFAULT_FULL_TRAINED_CHECKPOINT = ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_250k" / "best_hybrid_temporal_tabular_cnn.pt"
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

DEFAULT_TRAIN_SCENARIOS = list(DEFAULT_SCENARIOS)
DEFAULT_RARE_LABELS = "4,5,6,7,11,29,30,31"
EPS = 1e-12


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


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


class HybridDataset(Dataset):
    def __init__(self, X_lc: np.ndarray, X_tab: np.ndarray, y: np.ndarray):
        self.X_lc = X_lc.astype(np.float32, copy=False)
        self.X_tab = X_tab.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_lc[idx]).float(),
            torch.from_numpy(self.X_tab[idx]).float(),
            torch.tensor(self.y[idx]).long(),
        )


class HybridTemporalTabularCNN(nn.Module):
    """Same architecture family as the AstroTrust-AI hybrid temporal-tabular CNN."""

    def __init__(self, n_channels=18, n_tabular=319, n_classes=32):
        super().__init__()
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.15),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.20),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.temporal_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(192, n_classes),
        )

    def forward(self, x_lc, x_tab):
        z_lc = self.temporal_branch(x_lc)
        z_lc = self.temporal_pool(z_lc)
        z_lc = self.temporal_head(z_lc)
        z_tab = self.tabular_branch(x_tab)
        return self.classifier(torch.cat([z_lc, z_tab], dim=1))


def scenario_stage_vector(scenario: str) -> np.ndarray:
    meta = np.zeros(7, dtype=np.float32)
    # [is_full, is_first, is_window, first_n, window_days, log_first_n, log_window_days]
    if scenario == "full_curve_reference":
        meta[0] = 1.0
    elif scenario.startswith("first_") and scenario.endswith("_points"):
        n = float(scenario.replace("first_", "").replace("_points", ""))
        meta[1] = 1.0
        meta[3] = n
        meta[5] = math.log1p(n)
    elif scenario.startswith("window_") and scenario.endswith("_days"):
        d = float(scenario.replace("window_", "").replace("_days", ""))
        meta[2] = 1.0
        meta[4] = d
        meta[6] = math.log1p(d)
    return meta


def add_stage_features(X_tab: np.ndarray, scenario: str, include: bool) -> np.ndarray:
    if not include:
        return X_tab.astype(np.float32, copy=False)
    stage = np.repeat(scenario_stage_vector(scenario)[None, :], X_tab.shape[0], axis=0)
    return np.concatenate([X_tab.astype(np.float32, copy=False), stage.astype(np.float32)], axis=1)


def tensor_path(tensor_dir: Path, scenario: str, n_bins: int) -> Path:
    candidates = [
        tensor_dir / f"temporal_tensor_v1_{scenario}_{n_bins}bins.npz",
        tensor_dir / f"{scenario}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No tensor file found for scenario `{}`. Tried:\n{}".format(scenario, "\n".join(str(p) for p in candidates)))


def load_tensor(tensor_dir: Path, scenario: str, n_bins: int) -> dict:
    path = tensor_path(tensor_dir, scenario, n_bins)
    data = np.load(path, allow_pickle=True)
    required = ["X_lc", "X_tab", "y", "object_id"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Tensor file {path} is missing arrays: {missing}")
    out = {
        "scenario": scenario,
        "path": path,
        "X_lc": data["X_lc"].astype(np.float32),
        "X_tab_raw": data["X_tab"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "object_id": data["object_id"].astype(np.int64),
    }
    return out


def indices_for_ids(object_ids: np.ndarray, wanted: set[int]) -> np.ndarray:
    return np.array([i for i, oid in enumerate(object_ids) if int(oid) in wanted], dtype=np.int64)


def sample_indices(n: int, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    if max_n is not None and max_n > 0 and n > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
        idx.sort()
    return idx


def build_split_arrays(
    tensors: dict[str, dict],
    scenarios: list[str],
    split_ids: set[int],
    include_stage: bool,
    max_rows: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    pieces_lc = []
    pieces_tab = []
    pieces_y = []
    pieces_oid = []
    rows = []

    quota = None
    if max_rows is not None and max_rows > 0:
        quota = max(1, int(math.ceil(max_rows / max(1, len(scenarios)))))

    for scenario in scenarios:
        t = tensors[scenario]
        idx = indices_for_ids(t["object_id"], split_ids)
        before = len(idx)
        idx = idx[sample_indices(len(idx), quota, rng)] if len(idx) > 0 else idx

        X_lc = t["X_lc"][idx]
        X_tab = add_stage_features(t["X_tab_raw"][idx], scenario, include_stage)
        y = t["y"][idx]
        oid = t["object_id"][idx]

        pieces_lc.append(X_lc)
        pieces_tab.append(X_tab)
        pieces_y.append(y)
        pieces_oid.append(oid)
        rows.append({
            "scenario": scenario,
            "available_rows_for_split": int(before),
            "rows_used_before_global_cap": int(len(idx)),
        })

    X_lc_all = np.concatenate(pieces_lc, axis=0)
    X_tab_all = np.concatenate(pieces_tab, axis=0)
    y_all = np.concatenate(pieces_y, axis=0)
    oid_all = np.concatenate(pieces_oid, axis=0)

    if max_rows is not None and max_rows > 0 and len(y_all) > max_rows:
        keep = rng.choice(np.arange(len(y_all)), size=max_rows, replace=False)
        keep.sort()
        X_lc_all = X_lc_all[keep]
        X_tab_all = X_tab_all[keep]
        y_all = y_all[keep]
        oid_all = oid_all[keep]

    return X_lc_all, X_tab_all, y_all, oid_all, rows


def sanitize_probabilities(y_prob: np.ndarray, eps: float = EPS) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), eps)
    return probs


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    k = min(k, y_prob.shape[1])
    topk = np.argpartition(-y_prob, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([int(yt) in row for yt, row in zip(y_true, topk)]))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    probs = sanitize_probabilities(y_prob)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    corr = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probs = sanitize_probabilities(y_prob)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probs = sanitize_probabilities(y_prob)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true.astype(int)])))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    probs = sanitize_probabilities(y_prob)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    return {
        "n_objects": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "top2_accuracy": top_k_accuracy(y_true, probs, 2),
        "top3_accuracy": top_k_accuracy(y_true, probs, 3),
        "top5_accuracy": top_k_accuracy(y_true, probs, 5),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "mean_uncertainty": float((1.0 - probs.max(axis=1)).mean()),
        "std_uncertainty": float((1.0 - probs.max(axis=1)).std(ddof=1)) if len(y_true) > 1 else 0.0,
        "se_accuracy": float(correct.std(ddof=1) / math.sqrt(len(correct))) if len(correct) > 1 else 0.0,
        "ece": ece_score(y_true, probs),
        "brier_score": brier_score(y_true, probs),
        "negative_log_likelihood": nll_score(y_true, probs),
    }


def fit_novelty_reference(X_tab_raw_full_train: np.ndarray) -> dict:
    values = np.asarray(X_tab_raw_full_train, dtype=np.float64)
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25
    keep = np.isfinite(med) & np.isfinite(iqr) & (np.abs(iqr) > EPS)
    if not np.any(keep):
        raise RuntimeError("No valid tabular features remained for novelty reference.")
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


def compute_novelty(X_tab_raw: np.ndarray, ref: dict) -> np.ndarray:
    values = np.asarray(X_tab_raw, dtype=np.float64)
    keep = ref["keep"]
    z = (values[:, keep] - ref["median"][keep]) / (ref["iqr"][keep] + EPS)
    d = np.sqrt(np.mean(z * z, axis=1))
    novelty = (d - ref["q05"]) / max(ref["q95"] - ref["q05"], EPS)
    return np.clip(novelty, 0.0, 1.0)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    model.train()
    running = 0.0
    n = 0
    for X_lc, X_tab, y in loader:
        X_lc = X_lc.to(device, non_blocking=True)
        X_tab = X_tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(X_lc, X_tab)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item()) * len(y)
        n += len(y)
    return running / max(n, 1)


def predict_probabilities(model, X_lc: np.ndarray, X_tab: np.ndarray, y: np.ndarray, args: argparse.Namespace, device) -> tuple[np.ndarray, np.ndarray]:
    dataset = HybridDataset(X_lc, X_tab, y)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    probs = []
    y_out = []
    with torch.no_grad():
        for xb_lc, xb_tab, yb in loader:
            xb_lc = xb_lc.to(device, non_blocking=True)
            xb_tab = xb_tab.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb_lc, xb_tab)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            y_out.append(yb.numpy())
    return np.concatenate(y_out), sanitize_probabilities(np.concatenate(probs, axis=0))


def train_model(
    model_name: str,
    X_lc_train: np.ndarray,
    X_tab_train: np.ndarray,
    y_train: np.ndarray,
    X_lc_val: np.ndarray,
    X_tab_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    args: argparse.Namespace,
    output_dir: Path,
    device,
) -> tuple[HybridTemporalTabularCNN, dict, list[dict]]:
    print("\n" + "=" * 90)
    print(f"[TRAIN] {model_name}")
    print(f"Train: X_lc={X_lc_train.shape}, X_tab={X_tab_train.shape}, y={y_train.shape}")
    print(f"Val:   X_lc={X_lc_val.shape}, X_tab={X_tab_val.shape}, y={y_val.shape}")

    train_dataset = HybridDataset(X_lc_train, X_tab_train, y_train)
    val_dataset = HybridDataset(X_lc_val, X_tab_val, y_val)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = HybridTemporalTabularCNN(
        n_channels=X_lc_train.shape[1],
        n_tabular=X_tab_train.shape[1],
        n_classes=n_classes,
    ).to(device)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    full_weights = np.ones(n_classes, dtype=np.float32)
    for cls, weight in zip(classes, class_weights):
        full_weights[int(cls)] = weight
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(full_weights, dtype=torch.float32).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 3))
    amp_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_macro = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []
    checkpoint_path = output_dir / f"best_{model_name}.pt"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        y_val_eval, p_val = predict_probabilities(model, X_lc_val, X_tab_val, y_val, args, device)
        val_metrics = compute_metrics(y_val_eval, p_val)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / f"{model_name}_training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | loss={loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_top5={val_metrics['top5_accuracy']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "epoch": epoch,
                "val_macro_f1": best_val_macro,
                "n_channels": int(X_lc_train.shape[1]),
                "n_tabular": int(X_tab_train.shape[1]),
                "n_classes": int(n_classes),
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "include_stage_features": bool(args.include_stage_features),
            }, checkpoint_path)
            print(f"  [OK] saved best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}, best_val_macro_f1={best_val_macro:.4f}")
            break

    elapsed = (time.time() - start) / 60.0
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    info = {
        "model_name": model_name,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro),
        "training_time_minutes": float(elapsed),
        "checkpoint_path": safe_rel(checkpoint_path),
    }
    return model, info, history


def build_prediction_frame(object_ids: np.ndarray, y_true: np.ndarray, probs: np.ndarray, novelty: np.ndarray, rare_labels: set[int]) -> pd.DataFrame:
    probs = sanitize_probabilities(probs)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    uncertainty = 1.0 - confidence
    rare_cols = [i for i in range(probs.shape[1]) if i in rare_labels]
    rarity = probs[:, rare_cols].sum(axis=1) if rare_cols else np.zeros(len(y_true))
    top5 = np.argsort(probs, axis=1)[:, -min(5, probs.shape[1]):][:, ::-1]

    out = pd.DataFrame({
        "object_id": object_ids,
        "true_label": y_true,
        "predicted_label": pred,
        "correct": pred == y_true,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "novelty": novelty,
        "rarity_score": rarity,
        "priority_novelty_rarity": 0.5 * novelty + 0.5 * rarity,
        "true_is_rare": np.array([int(v) in rare_labels for v in y_true], dtype=bool),
    })
    for k in range(top5.shape[1]):
        out[f"top{k + 1}_label"] = top5[:, k]
    return out


def topk_from_pred_frame(pred: pd.DataFrame, k: int) -> float:
    cols = [f"top{i}_label" for i in range(1, k + 1) if f"top{i}_label" in pred.columns]
    if not cols:
        return np.nan
    y = pred["true_label"].astype(int).to_numpy()
    top = pred[cols].astype(int).to_numpy()
    return float(np.mean([yt in row for yt, row in zip(y, top)]))


def classification_metrics_from_frame(model_name: str, comparison_set: str, scenario: str, pred: pd.DataFrame) -> dict:
    y = pred["true_label"].astype(int).to_numpy()
    yp = pred["predicted_label"].astype(int).to_numpy()
    correct = pred["correct"].astype(bool).to_numpy(dtype=float)
    rare = pred["true_is_rare"].astype(bool).to_numpy(dtype=float)
    return {
        "model_name": model_name,
        "comparison_set": comparison_set,
        "scenario": scenario,
        "n_objects": int(len(pred)),
        "accuracy": float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, yp, average="weighted", zero_division=0)),
        "top2_accuracy": topk_from_pred_frame(pred, 2),
        "top3_accuracy": topk_from_pred_frame(pred, 3),
        "top5_accuracy": topk_from_pred_frame(pred, 5),
        "mean_correct": float(correct.mean()),
        "std_correct": float(correct.std(ddof=1)) if len(correct) > 1 else 0.0,
        "se_accuracy": float(correct.std(ddof=1) / math.sqrt(len(correct))) if len(correct) > 1 else 0.0,
        "mean_confidence": float(pred["confidence"].mean()),
        "mean_uncertainty": float(pred["uncertainty"].mean()),
        "std_uncertainty": float(pred["uncertainty"].std(ddof=1)) if len(pred) > 1 else 0.0,
        "mean_novelty": float(pred["novelty"].mean()),
        "std_novelty": float(pred["novelty"].std(ddof=1)) if len(pred) > 1 else 0.0,
        "mean_rarity_score": float(pred["rarity_score"].mean()),
        "rare_true_rate": float(rare.mean()),
    }


def followup_metrics_from_frame(model_name: str, comparison_set: str, scenario: str, pred: pd.DataFrame, budgets: Iterable[float]) -> pd.DataFrame:
    baseline = float(pred["true_is_rare"].astype(bool).mean())
    policies = {
        "novelty_rarity": pred["priority_novelty_rarity"].to_numpy(dtype=float),
        "rarity_only": pred["rarity_score"].to_numpy(dtype=float),
        "uncertainty_only": pred["uncertainty"].to_numpy(dtype=float),
        "fixed_discovery": 0.4 * pred["uncertainty"].to_numpy(float) + 0.4 * pred["novelty"].to_numpy(float) + 0.2 * pred["rarity_score"].to_numpy(float),
    }
    rows = []
    for policy, score in policies.items():
        order = np.argsort(score)[::-1]
        for budget in budgets:
            nsel = max(1, int(math.ceil(len(pred) * budget)))
            sel = pred.iloc[order[:nsel]]
            rare = sel["true_is_rare"].astype(bool).to_numpy(dtype=float)
            corr = sel["correct"].astype(bool).to_numpy(dtype=float)
            rare_rate = float(rare.mean())
            rows.append({
                "model_name": model_name,
                "comparison_set": comparison_set,
                "scenario": scenario,
                "policy": policy,
                "budget_fraction": float(budget),
                "n_available": int(len(pred)),
                "n_selected": int(nsel),
                "baseline_rare_rate": baseline,
                "rare_rate": rare_rate,
                "rare_enrichment": rare_rate / baseline if baseline > 0 else np.nan,
                "mean_confidence": float(sel["confidence"].mean()),
                "mean_uncertainty": float(sel["uncertainty"].mean()),
                "mean_novelty": float(sel["novelty"].mean()),
                "mean_rarity_score": float(sel["rarity_score"].mean()),
                "mean_correct": float(corr.mean()),
            })
    return pd.DataFrame(rows)


def compute_gain(classification: pd.DataFrame, followup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison_set in classification["comparison_set"].unique():
        base = classification[(classification["comparison_set"] == comparison_set) & (classification["model_name"] == "hybrid_full_trained")]
        early = classification[(classification["comparison_set"] == comparison_set) & (classification["model_name"] == "hybrid_early_aware")]
        merged = early.merge(base, on=["comparison_set", "scenario"], suffixes=("_early", "_full"))
        for row in merged.itertuples(index=False):
            d = row._asdict()
            out = {"comparison_set": comparison_set, "scenario": d["scenario"], "n_objects": d.get("n_objects_early")}
            for m in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "top5_accuracy", "mean_uncertainty"]:
                out[f"gain_{m}"] = float(d[f"{m}_early"] - d[f"{m}_full"])
            rows.append(out)
    gain = pd.DataFrame(rows)

    main = followup[(followup["policy"] == "novelty_rarity") & (np.isclose(followup["budget_fraction"], 0.05))]
    base = main[main["model_name"] == "hybrid_full_trained"]
    early = main[main["model_name"] == "hybrid_early_aware"]
    merged = early.merge(base, on=["comparison_set", "scenario", "policy", "budget_fraction"], suffixes=("_early", "_full"))
    erows = []
    for row in merged.itertuples(index=False):
        d = row._asdict()
        erows.append({
            "comparison_set": d["comparison_set"],
            "scenario": d["scenario"],
            "gain_rare_enrichment_5pct_novelty_rarity": float(d["rare_enrichment_early"] - d["rare_enrichment_full"]),
            "early_aware_rare_enrichment_5pct_novelty_rarity": float(d["rare_enrichment_early"]),
            "full_trained_rare_enrichment_5pct_novelty_rarity": float(d["rare_enrichment_full"]),
        })
    if erows and not gain.empty:
        gain = gain.merge(pd.DataFrame(erows), on=["comparison_set", "scenario"], how="left")
    return gain


def write_summary(path: Path, classification: pd.DataFrame, followup: pd.DataFrame, gain: pd.DataFrame, train_info: list[dict], split_info: dict, args: argparse.Namespace, novelty_ref: dict) -> None:
    lines = [
        "# Hybrid early-aware broker-like partial light-curve evaluation",
        "",
        "## Protocol",
        "",
        "This experiment trains the hybrid temporal--tabular CNN under two protocols: `hybrid_full_trained`, trained only on complete light-curve representations, and `hybrid_early_aware`, trained on complete plus temporally truncated representations of the training objects. Splits remain object-disjoint.",
        "",
        f"- Train objects: `{split_info['n_train']}`",
        f"- Validation objects: `{split_info['n_val']}`",
        f"- Test objects: `{split_info['n_test']}`",
        f"- Early-aware training rows: `{split_info['n_early_train_rows']}`",
        f"- Early-aware validation rows: `{split_info['n_early_val_rows']}`",
        f"- Include stage features: `{args.include_stage_features}`",
        f"- Novelty retained features: `{novelty_ref['n_retained_features']}`",
        f"- Rare labels: `{args.rare_labels}`",
        "",
        "## Training summary",
        "",
        pd.DataFrame(train_info).to_markdown(index=False),
        "",
        "## Classification metrics",
        "",
    ]
    show_cols = ["model_name", "comparison_set", "scenario", "n_objects", "accuracy", "se_accuracy", "macro_f1", "top3_accuracy", "top5_accuracy", "mean_uncertainty", "mean_novelty", "rare_true_rate"]
    lines.append(classification[[c for c in show_cols if c in classification.columns]].to_markdown(index=False, floatfmt=".4f"))
    if not gain.empty:
        lines += ["", "## Early-aware gain over full-trained hybrid", ""]
        gain_cols = ["comparison_set", "scenario", "gain_accuracy", "gain_macro_f1", "gain_top3_accuracy", "gain_top5_accuracy", "gain_rare_enrichment_5pct_novelty_rarity"]
        lines.append(gain[[c for c in gain_cols if c in gain.columns]].to_markdown(index=False, floatfmt=".4f"))
    main = followup[(followup["policy"] == "novelty_rarity") & (np.isclose(followup["budget_fraction"], 0.05))]
    lines += ["", "## Follow-up prioritization at 5% budget", ""]
    fcols = ["model_name", "comparison_set", "scenario", "n_available", "n_selected", "baseline_rare_rate", "rare_rate", "rare_enrichment", "mean_correct"]
    lines.append(main[[c for c in fcols if c in main.columns]].to_markdown(index=False, floatfmt=".4f"))
    lines += [
        "",
        "## Suggested manuscript wording",
        "",
        "```latex",
        r"\paragraph{Hybrid early-aware partial-light-curve training.}",
        r"We further trained an early-aware variant of the hybrid temporal--tabular CNN using complete and temporally truncated representations of the training objects. This preserves the original two-branch architecture while changing the training distribution to match the evolving information state of broker alerts. The evaluation compares the early-aware hybrid against a full-curve-trained hybrid on the same held-out object IDs under each partial-light-curve scenario.",
        "```",
        "",
        "## Interpretation caveat",
        "",
        "This experiment retrains the hybrid temporal--tabular CNN family. It should be described as `hybrid_early_aware`. It becomes the full dominant ensemble only after combining it with early-aware tabular ensemble members.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed, args.deterministic)
    rng = np.random.default_rng(args.seed)
    scenarios = parse_list(args.scenarios, DEFAULT_SCENARIOS)
    train_scenarios = parse_list(args.train_scenarios, DEFAULT_TRAIN_SCENARIOS)
    budgets = parse_float_list(args.budgets)
    rare_labels = parse_int_set(args.rare_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_dir.mkdir(parents=True, exist_ok=True)
    pred_root = args.output_dir / "predictions"
    prob_root = args.output_dir / "probabilities"
    pred_root.mkdir(parents=True, exist_ok=True)
    prob_root.mkdir(parents=True, exist_ok=True)

    print("[LOAD] tensors")
    needed = sorted(set(scenarios).union(train_scenarios).union({"full_curve_reference"}), key=lambda x: DEFAULT_SCENARIOS.index(x) if x in DEFAULT_SCENARIOS else 999)
    tensors = {s: load_tensor(args.tensor_dir, s, args.n_bins) for s in needed}

    full = tensors["full_curve_reference"]
    y_full = full["y"]
    oid_full = full["object_id"]
    n_classes = int(len(np.unique(y_full)))
    n_channels = int(full["X_lc"].shape[1])
    n_tab_raw = int(full["X_tab_raw"].shape[1])
    print(f"Full reference: objects={len(y_full)}, classes={n_classes}, channels={n_channels}, tabular={n_tab_raw}")

    all_idx = np.arange(len(y_full))
    train_val_idx, test_idx = train_test_split(all_idx, test_size=args.test_size, random_state=args.split_seed, stratify=y_full)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=args.validation_size, random_state=args.split_seed, stratify=y_full[train_val_idx])

    if args.max_train_objects is not None and args.max_train_objects > 0:
        train_idx = train_idx[:args.max_train_objects]
    if args.max_test_objects is not None and args.max_test_objects > 0:
        test_idx = test_idx[:args.max_test_objects]

    train_ids = set(int(v) for v in oid_full[train_idx])
    val_ids = set(int(v) for v in oid_full[val_idx])
    test_ids = set(int(v) for v in oid_full[test_idx])

    split_info = {"n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids)}
    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    novelty_ref = fit_novelty_reference(full["X_tab_raw"][train_idx])

    # Full-trained hybrid arrays.
    X_lc_train_full = full["X_lc"][train_idx]
    X_tab_train_full_raw_stage = add_stage_features(full["X_tab_raw"][train_idx], "full_curve_reference", args.include_stage_features)
    y_train_full = y_full[train_idx]
    X_lc_val_full = full["X_lc"][val_idx]
    X_tab_val_full_raw_stage = add_stage_features(full["X_tab_raw"][val_idx], "full_curve_reference", args.include_stage_features)
    y_val_full = y_full[val_idx]

    scaler_full = StandardScaler()
    X_tab_train_full = scaler_full.fit_transform(X_tab_train_full_raw_stage).astype(np.float32)
    X_tab_val_full = scaler_full.transform(X_tab_val_full_raw_stage).astype(np.float32)
    X_tab_train_full = np.nan_to_num(X_tab_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_val_full = np.nan_to_num(X_tab_val_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Early-aware arrays.
    X_lc_train_early, X_tab_train_early_raw_stage, y_train_early, _, train_manifest_rows = build_split_arrays(
        tensors, train_scenarios, train_ids, args.include_stage_features, args.max_train_rows, rng
    )
    X_lc_val_early, X_tab_val_early_raw_stage, y_val_early, _, val_manifest_rows = build_split_arrays(
        tensors, train_scenarios, val_ids, args.include_stage_features, args.max_val_rows, rng
    )

    scaler_early = StandardScaler()
    X_tab_train_early = scaler_early.fit_transform(X_tab_train_early_raw_stage).astype(np.float32)
    X_tab_val_early = scaler_early.transform(X_tab_val_early_raw_stage).astype(np.float32)
    X_tab_train_early = np.nan_to_num(X_tab_train_early, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_val_early = np.nan_to_num(X_tab_val_early, nan=0.0, posinf=0.0, neginf=0.0)

    split_info["n_early_train_rows"] = int(len(y_train_early))
    split_info["n_early_val_rows"] = int(len(y_val_early))

    manifest = pd.DataFrame(train_manifest_rows)
    manifest.to_csv(args.output_dir / "hybrid_early_aware_training_manifest.csv", index=False)
    pd.DataFrame(val_manifest_rows).to_csv(args.output_dir / "hybrid_early_aware_validation_manifest.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_info = []
    models = {}
    scalers = {}
    model_stage_flags = {}

    if args.full_baseline_mode == "existing_checkpoint":
        ckpt_path = Path(args.full_trained_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Full-trained checkpoint not found: {ckpt_path}. "
                "Use --full-baseline-mode train to train it from scratch, or pass --full-trained-checkpoint."
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_n_tab = int(ckpt.get("n_tabular", n_tab_raw))
        if ckpt_n_tab == n_tab_raw:
            full_uses_stage = False
        elif ckpt_n_tab == n_tab_raw + 7:
            full_uses_stage = True
        else:
            raise ValueError(
                f"Checkpoint n_tabular={ckpt_n_tab}, but current tensor has raw tabular={n_tab_raw}. "
                "Rebuild partial tensors using reference tabular columns from the original tensor, "
                "or use --full-baseline-mode train."
            )
        X_full_scaler_train = add_stage_features(full["X_tab_raw"][train_idx], "full_curve_reference", full_uses_stage)
        scaler_full_existing = StandardScaler().fit(X_full_scaler_train)
        m_full = HybridTemporalTabularCNN(
            n_channels=int(ckpt.get("n_channels", n_channels)),
            n_tabular=ckpt_n_tab,
            n_classes=int(ckpt.get("n_classes", n_classes)),
        ).to(device)
        m_full.load_state_dict(ckpt["model_state_dict"])
        m_full.eval()
        models["hybrid_full_trained"] = m_full
        scalers["hybrid_full_trained"] = scaler_full_existing
        model_stage_flags["hybrid_full_trained"] = full_uses_stage
        train_info.append({
            "model_name": "hybrid_full_trained",
            "mode": "loaded_existing_checkpoint",
            "checkpoint_path": safe_rel(ckpt_path),
            "best_epoch": ckpt.get("epoch"),
            "best_val_macro_f1": ckpt.get("val_macro_f1"),
            "training_time_minutes": np.nan,
            "uses_stage_features": bool(full_uses_stage),
        })
        print(f"[OK] Loaded existing full-trained hybrid checkpoint: {ckpt_path}")

    elif args.full_baseline_mode == "train":
        m_full, info_full, _ = train_model(
            "hybrid_full_trained",
            X_lc_train_full, X_tab_train_full, y_train_full,
            X_lc_val_full, X_tab_val_full, y_val_full,
            n_classes, args, args.output_dir, device,
        )
        models["hybrid_full_trained"] = m_full
        scalers["hybrid_full_trained"] = scaler_full
        model_stage_flags["hybrid_full_trained"] = bool(args.include_stage_features)
        train_info.append(info_full)

    elif args.full_baseline_mode == "skip":
        print("[INFO] Skipping full-trained hybrid baseline.")
    else:
        raise ValueError(f"Unknown --full-baseline-mode: {args.full_baseline_mode}")

    m_early, info_early, _ = train_model(
        "hybrid_early_aware",
        X_lc_train_early, X_tab_train_early, y_train_early,
        X_lc_val_early, X_tab_val_early, y_val_early,
        n_classes, args, args.output_dir, device,
    )
    models["hybrid_early_aware"] = m_early
    scalers["hybrid_early_aware"] = scaler_early
    model_stage_flags["hybrid_early_aware"] = bool(args.include_stage_features)
    train_info.append(info_early)

    predictions = {name: {} for name in models}

    for scenario in scenarios:
        t = tensors[scenario]
        idx = indices_for_ids(t["object_id"], test_ids)
        if len(idx) == 0:
            print(f"[WARN] no test objects for scenario {scenario}")
            continue
        if args.max_test_objects is not None and args.max_test_objects > 0 and len(idx) > args.max_test_objects:
            idx = idx[:args.max_test_objects]

        X_lc_eval = t["X_lc"][idx]
        X_tab_raw_eval = t["X_tab_raw"][idx]
        y_eval = t["y"][idx]
        oid_eval = t["object_id"][idx]
        novelty = compute_novelty(X_tab_raw_eval, novelty_ref)

        for model_name, model in models.items():
            X_tab_eval_stage = add_stage_features(X_tab_raw_eval, scenario, model_stage_flags.get(model_name, args.include_stage_features))
            X_tab_eval = scalers[model_name].transform(X_tab_eval_stage).astype(np.float32)
            X_tab_eval = np.nan_to_num(X_tab_eval, nan=0.0, posinf=0.0, neginf=0.0)
            yy, probs = predict_probabilities(model, X_lc_eval, X_tab_eval, y_eval, args, device)
            pred = build_prediction_frame(oid_eval, yy, probs, novelty, rare_labels)
            predictions[model_name][scenario] = pred

            pdir = pred_root / model_name
            qdir = prob_root / model_name
            pdir.mkdir(parents=True, exist_ok=True)
            qdir.mkdir(parents=True, exist_ok=True)
            pred.to_csv(pdir / f"{scenario}_predictions.csv", index=False)
            np.save(qdir / f"{scenario}_probabilities.npy", probs.astype(np.float32))
            print(f"[EVAL] {model_name} | {scenario}: n={len(pred)}, acc={pred['correct'].mean():.4f}")

    class_rows = []
    follow_frames = []
    for model_name, scen_map in predictions.items():
        for scenario, pred in scen_map.items():
            class_rows.append(classification_metrics_from_frame(model_name, "all_available_objects", scenario, pred))
            follow_frames.append(followup_metrics_from_frame(model_name, "all_available_objects", scenario, pred, budgets))

        common = None
        for pred in scen_map.values():
            ids = set(int(v) for v in pred["object_id"].to_numpy())
            common = ids if common is None else common.intersection(ids)
        common = common or set()
        print(f"[COMMON] {model_name}: {len(common)} objects")
        if common:
            for scenario, pred in scen_map.items():
                cp = pred[pred["object_id"].isin(common)].copy()
                class_rows.append(classification_metrics_from_frame(model_name, "common_object_subset", scenario, cp))
                follow_frames.append(followup_metrics_from_frame(model_name, "common_object_subset", scenario, cp, budgets))

    classification = pd.DataFrame(class_rows)
    followup = pd.concat(follow_frames, ignore_index=True) if follow_frames else pd.DataFrame()

    scenario_order = {s: i for i, s in enumerate(DEFAULT_SCENARIOS)}
    model_order = {"hybrid_full_trained": 0, "hybrid_early_aware": 1}
    if not classification.empty:
        classification["_s"] = classification["scenario"].map(scenario_order)
        classification["_m"] = classification["model_name"].map(model_order)
        classification = classification.sort_values(["comparison_set", "_s", "_m"]).drop(columns=["_s", "_m"])
    if not followup.empty:
        followup["_s"] = followup["scenario"].map(scenario_order)
        followup["_m"] = followup["model_name"].map(model_order)
        followup = followup.sort_values(["comparison_set", "policy", "budget_fraction", "_s", "_m"]).drop(columns=["_s", "_m"])

    gain = compute_gain(classification, followup) if "hybrid_full_trained" in models else pd.DataFrame()

    class_path = args.output_dir / "hybrid_early_aware_classification_metrics.csv"
    follow_path = args.output_dir / "hybrid_early_aware_followup_metrics.csv"
    gain_path = args.output_dir / "hybrid_early_aware_gain_over_full_trained.csv"
    info_path = args.output_dir / "hybrid_early_aware_training_summary.csv"
    summary_path = args.output_dir / "hybrid_early_aware_summary.md"

    classification.to_csv(class_path, index=False)
    followup.to_csv(follow_path, index=False)
    gain.to_csv(gain_path, index=False)
    pd.DataFrame(train_info).to_csv(info_path, index=False)
    write_summary(summary_path, classification, followup, gain, train_info, split_info, args, novelty_ref)

    for path in [class_path, follow_path, gain_path, info_path, args.output_dir / "hybrid_early_aware_training_manifest.csv", summary_path]:
        if path.exists():
            shutil.copyfile(path, args.final_dir / path.name)

    print("\n" + "=" * 90)
    print("[OK] Hybrid early-aware training/evaluation finished.")
    print(f"[OK] Classification: {class_path}")
    print(f"[OK] Follow-up:       {follow_path}")
    print(f"[OK] Gain:            {gain_path}")
    print(f"[OK] Summary:         {summary_path}")
    print(f"[OK] Publication copy:{args.final_dir}")


def run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)
    base = args.output_dir / "_self_test_hybrid_early_aware"
    tensor_dir = base / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["full_curve_reference", "first_3_points", "first_5_points", "window_7_days"]
    n = 80
    n_classes = 4
    n_tab = 12
    n_bins = 8
    oids = np.arange(n, dtype=np.int64)
    y = (oids % n_classes).astype(np.int64)
    base_tab = rng.normal(size=(n, n_tab)).astype(np.float32)
    base_tab[:, 0] += y * 0.8
    base_lc = rng.normal(size=(n, 18, n_bins)).astype(np.float32)
    base_lc[:, 0, :] += y[:, None] * 0.08

    for scenario in scenarios:
        noise = 0.2 if scenario == "full_curve_reference" else 0.8
        keep = np.ones(n, dtype=bool)
        if scenario == "window_7_days":
            keep = rng.random(n) > 0.1
        X_lc = base_lc + rng.normal(scale=noise, size=base_lc.shape).astype(np.float32)
        X_tab = base_tab + rng.normal(scale=noise, size=base_tab.shape).astype(np.float32)
        np.savez_compressed(
            tensor_dir / f"temporal_tensor_v1_{scenario}_{n_bins}bins.npz",
            X_lc=X_lc[keep],
            X_tab=X_tab[keep],
            y=y[keep],
            object_id=oids[keep],
            tabular_columns=np.asarray([f"f{i}" for i in range(n_tab)]),
        )

    test_args = argparse.Namespace(**vars(args))
    test_args.tensor_dir = tensor_dir
    test_args.output_dir = base / "results"
    test_args.final_dir = base / "final"
    test_args.scenarios = ",".join(scenarios)
    test_args.train_scenarios = ",".join(scenarios)
    test_args.n_bins = n_bins
    test_args.epochs = 1
    test_args.patience = 1
    test_args.batch_size = 32
    test_args.max_train_rows = 120
    test_args.max_val_rows = 60
    test_args.max_test_objects = 20
    test_args.full_baseline_mode = "train"
    run(test_args)
    print(f"[OK] Self-test outputs: {base}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hybrid temporal-tabular CNN in early-aware partial-light-curve mode.")
    p.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSOR_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    p.add_argument("--scenarios", type=str, default="all")
    p.add_argument("--train-scenarios", type=str, default="all")
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--validation-size", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--include-stage-features", action="store_true", default=True)
    p.add_argument("--no-stage-features", dest="include_stage_features", action="store_false")
    p.add_argument("--max-train-rows", type=int, default=500000, help="Cap stacked early-aware training rows. 0 means no cap.")
    p.add_argument("--max-val-rows", type=int, default=0, help="Optional cap on stacked early-aware validation rows. 0 means no cap.")
    p.add_argument("--max-train-objects", type=int, default=None)
    p.add_argument("--max-test-objects", type=int, default=None)
    p.add_argument("--full-baseline-mode", choices=["existing_checkpoint", "train", "skip"], default="existing_checkpoint")
    p.add_argument("--full-trained-checkpoint", type=Path, default=DEFAULT_FULL_TRAINED_CHECKPOINT)
    p.add_argument("--rare-labels", type=str, default=DEFAULT_RARE_LABELS)
    p.add_argument("--budgets", type=str, default="0.01,0.02,0.05,0.10,0.20")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Normalize 0 caps to None.
    if args.max_train_rows is not None and args.max_train_rows <= 0:
        args.max_train_rows = None
    if args.max_val_rows is not None and args.max_val_rows <= 0:
        args.max_val_rows = None
    if args.self_test:
        run_self_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
