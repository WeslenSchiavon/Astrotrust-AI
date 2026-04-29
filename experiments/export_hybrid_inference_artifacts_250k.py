from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]

TENSOR_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "temporal_tensor_v1_250000obj_64bins.npz"
)

CHECKPOINT_PATH = (
    ROOT_DIR
    / "results"
    / "hybrid_temporal_tabular_cnn_250k"
    / "best_hybrid_temporal_tabular_cnn.pt"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "hybrid_inference_artifacts_250k"

RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE_FROM_TRAIN = 0.15
TEMPERATURE = 1.1872


class HybridTemporalTabularCNN(nn.Module):
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
        z = torch.cat([z_lc, z_tab], dim=1)
        return self.classifier(z)


def load_class_metadata():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    counts = counts.sort_values("rarity_rank").copy()

    label_to_name = dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))
    global_counts = dict(zip(counts["label"].astype(int), counts["n_objects"].astype(int)))

    n_rare = max(1, int(np.ceil(len(counts) * 0.25)))
    rare_classes = counts.head(n_rare)["label"].astype(int).tolist()

    return counts, label_to_name, global_counts, rare_classes


def split_indices(y):
    all_idx = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y[train_idx],
    )

    return train_idx, val_idx, test_idx


def fit_mahalanobis_models(X_train_scaled, y_train):
    models = {}

    for cls in sorted(np.unique(y_train)):
        X_cls = X_train_scaled[y_train == cls]

        print(f"  class {int(cls):02d}: fitting LedoitWolf on {len(X_cls)} objects")

        cov = LedoitWolf()
        cov.fit(X_cls)

        models[int(cls)] = {
            "mean": cov.location_.astype(np.float32),
            "precision": cov.precision_.astype(np.float32),
            "n_train": int(len(X_cls)),
        }

    return models


def mahalanobis_squared(X, mean, precision):
    diff = X - mean
    return np.sum(diff @ precision * diff, axis=1)


def compute_mahalanobis_novelty(X_eval, models):
    distances = []

    for model in models.values():
        d = mahalanobis_squared(X_eval, model["mean"], model["precision"])
        distances.append(d)

    distances = np.vstack(distances).T
    return np.min(distances, axis=1)


def compute_rarity_score_normalization(global_counts):
    labels = sorted(global_counts.keys())
    raw_values = np.asarray([1.0 / np.log1p(global_counts[label]) for label in labels], dtype=np.float32)

    return {
        "labels": labels,
        "raw_values": raw_values.tolist(),
        "min": float(raw_values.min()),
        "max": float(raw_values.max()),
    }


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TENSOR_PATH.exists():
        raise FileNotFoundError(TENSOR_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not CLASS_COUNTS_PATH.exists():
        raise FileNotFoundError(CLASS_COUNTS_PATH)

    print(f"Loading tensor dataset: {TENSOR_PATH}")
    data = np.load(TENSOR_PATH, allow_pickle=True)

    X_tab = data["X_tab"].astype(np.float32)
    y = data["y"].astype(np.int64)
    object_ids = data["object_id"].astype(np.int64)

    if "tabular_columns" in data:
        tabular_columns = [str(v) for v in data["tabular_columns"].tolist()]
    else:
        tabular_columns = [f"feature_{i}" for i in range(X_tab.shape[1])]

    if "channel_names" in data:
        channel_names = [str(v) for v in data["channel_names"].tolist()]
    else:
        channel_names = [f"channel_{i}" for i in range(18)]

    if "bands" in data:
        bands = [str(v) for v in data["bands"].tolist()]
    else:
        bands = ["u", "g", "r", "i", "z", "Y"]

    print("\nDataset summary:")
    print(f"Objects: {len(y)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"X_tab shape: {X_tab.shape}")

    train_idx, val_idx, test_idx = split_indices(y)

    print("\nSplits:")
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")

    print("\nFitting StandardScaler on training tabular features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_tab[train_idx]).astype(np.float32)
    X_val_scaled = scaler.transform(X_tab[val_idx]).astype(np.float32)

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = OUTPUT_DIR / "tabular_standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Saved scaler: {scaler_path}")

    print("\nFitting class-wise Mahalanobis novelty models on training features...")
    novelty_models = fit_mahalanobis_models(X_train_scaled, y[train_idx])

    novelty_path = OUTPUT_DIR / "mahalanobis_novelty_models.joblib"
    joblib.dump(novelty_models, novelty_path)
    print(f"[OK] Saved novelty models: {novelty_path}")

    print("\nComputing novelty normalization from validation split...")
    val_raw_novelty = compute_mahalanobis_novelty(X_val_scaled, novelty_models)
    novelty_norm = {
        "min": float(np.nanmin(val_raw_novelty)),
        "max": float(np.nanmax(val_raw_novelty)),
        "source": "validation_split",
    }

    novelty_norm_path = OUTPUT_DIR / "novelty_normalization.json"
    save_json(novelty_norm_path, novelty_norm)
    print(f"[OK] Saved novelty normalization: {novelty_norm_path}")

    print("\nLoading class metadata and global rarity...")
    class_counts, label_to_name, global_counts, rare_classes = load_class_metadata()

    class_counts_out = OUTPUT_DIR / "class_counts.csv"
    class_counts.to_csv(class_counts_out, index=False)
    print(f"[OK] Saved class counts: {class_counts_out}")

    label_metadata = {
        "label_to_name": {str(k): v for k, v in label_to_name.items()},
        "global_counts": {str(k): int(v) for k, v in global_counts.items()},
        "rare_classes": [int(v) for v in rare_classes],
        "rarity_score_normalization": compute_rarity_score_normalization(global_counts),
    }

    label_metadata_path = OUTPUT_DIR / "label_and_rarity_metadata.json"
    save_json(label_metadata_path, label_metadata)
    print(f"[OK] Saved label metadata: {label_metadata_path}")

    print("\nCopying hybrid checkpoint...")
    checkpoint_out = OUTPUT_DIR / "best_hybrid_temporal_tabular_cnn.pt"
    shutil.copy2(CHECKPOINT_PATH, checkpoint_out)
    print(f"[OK] Copied checkpoint: {checkpoint_out}")

    temperature_metadata = {
        "temperature": TEMPERATURE,
        "source": "hybrid_followup_policy_eval_250k validation temperature scaling",
        "note": "Probabilities are calibrated as softmax(logits / temperature).",
    }

    temperature_path = OUTPUT_DIR / "temperature_scaling.json"
    save_json(temperature_path, temperature_metadata)
    print(f"[OK] Saved temperature metadata: {temperature_path}")

    split_metadata = pd.DataFrame({
        "object_id": object_ids,
        "label": y,
        "split": "unused",
    })
    split_metadata.loc[train_idx, "split"] = "train"
    split_metadata.loc[val_idx, "split"] = "validation"
    split_metadata.loc[test_idx, "split"] = "test"

    split_path = OUTPUT_DIR / "split_metadata.csv"
    split_metadata.to_csv(split_path, index=False)
    print(f"[OK] Saved split metadata: {split_path}")

    inference_metadata = {
        "artifact_version": "hybrid_inference_artifacts_250k_v1",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_size_from_train": VAL_SIZE_FROM_TRAIN,
        "n_tabular_features": int(X_tab.shape[1]),
        "n_channels": 18,
        "n_bins": 64,
        "n_classes": int(len(np.unique(y))),
        "bands": bands,
        "channel_names": channel_names,
        "tabular_columns": tabular_columns,
        "final_followup_policy": {
            "name": "novelty_rarity",
            "w_uncertainty": 0.0,
            "w_novelty": 0.5,
            "w_rarity": 0.5,
            "formula": "P(x) = 0.5*N(x) + 0.5*R(y_hat)",
        },
        "files": {
            "checkpoint": checkpoint_out.name,
            "scaler": scaler_path.name,
            "novelty_models": novelty_path.name,
            "novelty_normalization": novelty_norm_path.name,
            "temperature_scaling": temperature_path.name,
            "label_metadata": label_metadata_path.name,
            "class_counts": class_counts_out.name,
            "split_metadata": split_path.name,
        },
    }

    inference_metadata_path = OUTPUT_DIR / "inference_metadata.json"
    save_json(inference_metadata_path, inference_metadata)
    print(f"[OK] Saved inference metadata: {inference_metadata_path}")

    print("\nExport completed.")
    print(f"Artifacts directory: {OUTPUT_DIR}")
    print("\nExpected files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
