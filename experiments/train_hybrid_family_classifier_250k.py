from __future__ import annotations

from pathlib import Path
import argparse
import json
import time
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_TENSOR_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "temporal_tensor_v1_250000obj_64bins.npz"
)

DEFAULT_SPLIT_METADATA = (
    ROOT_DIR
    / "results"
    / "hybrid_inference_artifacts_250k"
    / "split_metadata.csv"
)

DEFAULT_LABEL_METADATA = (
    ROOT_DIR
    / "results"
    / "hybrid_inference_artifacts_250k"
    / "label_and_rarity_metadata.json"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "hybrid_family_classifier_250k"

RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Taxonomy mapping
# -----------------------------------------------------------------------------

def map_to_family(class_name: str) -> str:
    low = str(class_name).strip().lower()

    if low in ["clagn"] or "agn" in low:
        return "AGN"
    if "cepheid" in low or low == "rrl" or "d-sct" in low or "dsct" in low:
        return "Periodic variable"
    if low == "eb" or "eclips" in low:
        return "Eclipsing binary"
    if "mdwarf" in low or "flare" in low:
        return "Stellar flare"
    if "dwarf-nova" in low or "dwarf nova" in low:
        return "Cataclysmic variable"
    if "ulens" in low or "microlens" in low:
        return "Microlensing"
    if low.startswith("kn") or "kilonova" in low:
        return "Kilonova"
    if "slsn" in low:
        return "Superluminous SN"
    if "tde" in low:
        return "TDE"
    if "ilot" in low:
        return "ILOT"
    if "pisn" in low:
        return "PISN"
    if "snia" in low or "sn ia" in low:
        return "SN Ia"
    if "snii" in low or "sn ii" in low or "sniib" in low or "sniin" in low:
        return "SN II"
    if "snib" in low or "snic" in low or "sn ib" in low or "sn ic" in low:
        return "SN Ib/c"
    if "cart" in low:
        return "CART"

    return "Other / unknown"


# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------

def find_array_key(data, candidates: list[str]):
    keys = list(data.keys())
    lower_map = {k.lower(): k for k in keys}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_label_names(path: Path, n_classes: int) -> dict[int, str]:
    if not path.exists():
        return {i: str(i) for i in range(n_classes)}

    meta = json.loads(path.read_text(encoding="utf-8"))

    for key in ["label_to_class", "label_to_name", "class_names", "classes"]:
        if key not in meta:
            continue

        value = meta[key]

        if isinstance(value, dict):
            mapping = {}
            for k, v in value.items():
                try:
                    mapping[int(k)] = str(v)
                except Exception:
                    pass
            if mapping:
                for i in range(n_classes):
                    mapping.setdefault(i, str(i))
                return mapping

        if isinstance(value, list):
            mapping = {i: str(v) for i, v in enumerate(value)}
            for i in range(n_classes):
                mapping.setdefault(i, str(i))
            return mapping

    return {i: str(i) for i in range(n_classes)}


def load_tensor_dataset(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    print("NPZ keys:", list(data.keys()))

    x_lc_key = find_array_key(data, ["X_lc", "x_lc", "lightcurves", "temporal", "X"])
    x_tab_key = find_array_key(data, ["X_tab", "x_tab", "tabular", "tabular_features"])
    y_key = find_array_key(data, ["y", "labels", "target", "targets"])
    obj_key = find_array_key(data, ["object_ids", "object_id", "object_ids_str", "ids"])

    if x_lc_key is None or x_tab_key is None or y_key is None:
        raise ValueError(
            f"Could not find X_lc/X_tab/y in {npz_path}. Available keys: {list(data.keys())}"
        )

    X_lc = np.asarray(data[x_lc_key], dtype=np.float32)
    X_tab = np.asarray(data[x_tab_key], dtype=np.float32)
    y = np.asarray(data[y_key]).astype(int)

    if obj_key is not None:
        object_ids = np.asarray(data[obj_key]).astype(str)
    else:
        object_ids = np.arange(len(y)).astype(str)

    X_lc = np.nan_to_num(X_lc, nan=0.0, posinf=1e10, neginf=-1e10)
    X_lc = np.clip(X_lc, -1e10, 1e10).astype(np.float32)

    X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=1e10, neginf=-1e10)
    X_tab = np.clip(X_tab, -1e10, 1e10).astype(np.float32)

    return X_lc, X_tab, y, object_ids, data


def load_splits(data, split_metadata_path: Path, n_samples: int):
    for train_key, val_key, test_key in [
        ("train_idx", "val_idx", "test_idx"),
        ("train_indices", "val_indices", "test_indices"),
        ("idx_train", "idx_val", "idx_test"),
    ]:
        if train_key in data and val_key in data and test_key in data:
            return (
                np.asarray(data[train_key]).astype(int),
                np.asarray(data[val_key]).astype(int),
                np.asarray(data[test_key]).astype(int),
                "npz_indices",
            )

    if split_metadata_path.exists():
        split_df = pd.read_csv(split_metadata_path)
        lower = {str(c).lower(): c for c in split_df.columns}

        split_col = None
        for c in ["split", "split_name", "dataset_split", "partition"]:
            if c in lower:
                split_col = lower[c]
                break

        idx_col = None
        for c in ["index", "idx", "row_index", "sample_index", "array_index"]:
            if c in lower:
                idx_col = lower[c]
                break

        if split_col is not None:
            tmp = split_df.copy()

            if idx_col is None:
                tmp["_idx"] = np.arange(len(tmp))
                idx_col = "_idx"

            split_values = tmp[split_col].astype(str).str.lower()
            train_idx = tmp.loc[split_values.str.contains("train"), idx_col].astype(int).to_numpy()
            val_idx = tmp.loc[split_values.str.contains("val"), idx_col].astype(int).to_numpy()
            test_idx = tmp.loc[split_values.str.contains("test"), idx_col].astype(int).to_numpy()

            if len(train_idx) and len(val_idx) and len(test_idx):
                return train_idx, val_idx, test_idx, "split_metadata_csv"

    warnings.warn(
        "Could not recover original split. Falling back to deterministic random split. "
        "Use this only for exploratory runs."
    )
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(0.6375 * n_samples)
    n_val = int(0.1125 * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx, "fallback_random"


# -----------------------------------------------------------------------------
# Dataset and model
# -----------------------------------------------------------------------------

class FamilyTensorDataset(Dataset):
    def __init__(self, X_lc, X_tab, y_family, indices):
        self.X_lc = X_lc
        self.X_tab = X_tab
        self.y_family = y_family
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return (
            torch.from_numpy(self.X_lc[idx]).float(),
            torch.from_numpy(self.X_tab[idx]).float(),
            torch.tensor(self.y_family[idx], dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.10):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TemporalCNNBranch(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 96, embedding_dim: int = 192, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(hidden, kernel_size=5, dilation=1, dropout=dropout),
            ResidualConvBlock(hidden, kernel_size=5, dilation=2, dropout=dropout),
            ResidualConvBlock(hidden, kernel_size=5, dilation=4, dropout=dropout),
            ResidualConvBlock(hidden, kernel_size=3, dilation=8, dropout=dropout),
        )
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        avg = self.pool_avg(x).squeeze(-1)
        mx = self.pool_max(x).squeeze(-1)
        return self.proj(torch.cat([avg, mx], dim=1))


class TabularBranch(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int = 192, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class HybridFamilyClassifier(nn.Module):
    def __init__(
        self,
        lc_channels: int,
        tab_features: int,
        n_families: int,
        temporal_dim: int = 192,
        tabular_dim: int = 192,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.temporal = TemporalCNNBranch(lc_channels, embedding_dim=temporal_dim, dropout=dropout)
        self.tabular = TabularBranch(tab_features, embedding_dim=tabular_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(temporal_dim + tabular_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, n_families),
        )

    def forward(self, x_lc, x_tab):
        z_lc = self.temporal(x_lc)
        z_tab = self.tabular(x_tab)
        z = torch.cat([z_lc, z_tab], dim=1)
        return self.head(z)


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def evaluate_model(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    indices = []

    with torch.no_grad():
        for x_lc, x_tab, y, idx in loader:
            x_lc = x_lc.to(device, non_blocking=True)
            x_tab = x_tab.to(device, non_blocking=True)

            logits = model(x_lc, x_tab)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.append(y.numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
            indices.append(idx.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    indices = np.concatenate(indices)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "mean_confidence": float(np.max(y_prob, axis=1).mean()),
    }

    return metrics, y_true, y_pred, y_prob, indices


def main():
    parser = argparse.ArgumentParser(description="Train a hybrid temporal-tabular classifier for astronomical families.")
    parser.add_argument("--tensor-dataset", type=Path, default=DEFAULT_TENSOR_DATASET)
    parser.add_argument("--split-metadata", type=Path, default=DEFAULT_SPLIT_METADATA)
    parser.add_argument("--label-metadata", type=Path, default=DEFAULT_LABEL_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_lc, X_tab, y_fine, object_ids, npz_data = load_tensor_dataset(args.tensor_dataset)
    n_fine_classes = int(np.max(y_fine)) + 1
    label_names = load_label_names(args.label_metadata, n_classes=n_fine_classes)

    fine_names = np.array([label_names[int(v)] for v in y_fine])
    family_names = np.array([map_to_family(v) for v in fine_names])

    family_encoder = LabelEncoder()
    y_family = family_encoder.fit_transform(family_names)
    n_families = len(family_encoder.classes_)

    train_idx, val_idx, test_idx, split_source = load_splits(npz_data, args.split_metadata, len(y_fine))

    print("Dataset summary:")
    print(f"Objects: {len(y_fine)}")
    print(f"X_lc: {X_lc.shape}")
    print(f"X_tab: {X_tab.shape}")
    print(f"Fine classes: {n_fine_classes}")
    print(f"Families: {n_families} -> {list(family_encoder.classes_)}")
    print(f"Split source: {split_source}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Scale tabular features using train only.
    print("Fitting StandardScaler on train tabular features...")
    scaler = StandardScaler()
    X_tab_scaled = X_tab.copy()
    X_tab_scaled[train_idx] = scaler.fit_transform(X_tab[train_idx])
    X_tab_scaled[val_idx] = scaler.transform(X_tab[val_idx])
    X_tab_scaled[test_idx] = scaler.transform(X_tab[test_idx])
    X_tab_scaled = np.nan_to_num(X_tab_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    X_tab_scaled = np.clip(X_tab_scaled, -1e6, 1e6).astype(np.float32)

    joblib.dump(scaler, args.output_dir / "family_tabular_standard_scaler.joblib")

    train_ds = FamilyTensorDataset(X_lc, X_tab_scaled, y_family, train_idx)
    val_ds = FamilyTensorDataset(X_lc, X_tab_scaled, y_family, val_idx)
    test_ds = FamilyTensorDataset(X_lc, X_tab_scaled, y_family, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = HybridFamilyClassifier(
        lc_channels=X_lc.shape[1],
        tab_features=X_tab_scaled.shape[1],
        n_families=n_families,
        dropout=args.dropout,
    ).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        classes = np.unique(y_family[train_idx])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_family[train_idx],
        )
        full_weights = np.ones(n_families, dtype=np.float32)
        for cls, w in zip(classes, weights):
            full_weights[int(cls)] = float(w)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(full_weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
    )

    scaler_amp = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_macro_f1 = -np.inf
    best_epoch = -1
    patience_counter = 0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for x_lc, x_tab, y, _ in train_loader:
            x_lc = x_lc.to(device, non_blocking=True)
            x_tab = x_tab.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_lc, x_tab)
                loss = criterion(logits, y)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            batch_size = len(y)
            running_loss += float(loss.item()) * batch_size
            n_seen += batch_size

        train_loss = running_loss / max(n_seen, 1)
        val_metrics, _, _, _, _ = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1 + 1e-5:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "family_classes": family_encoder.classes_.tolist(),
                    "label_names": label_names,
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                args.output_dir / "best_hybrid_family_classifier.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.2f} min")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val Macro-F1: {best_val_macro_f1:.4f}")

    pd.DataFrame(history).to_csv(args.output_dir / "family_training_history.csv", index=False)

    print("\nLoading best model for final test evaluation...")
    checkpoint = torch.load(
        args.output_dir / "best_hybrid_family_classifier.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true_family, y_pred_family, y_prob_family, test_indices = evaluate_model(model, test_loader, device)

    test_metrics_df = pd.DataFrame([{**test_metrics, "best_epoch": best_epoch, "best_val_macro_f1": best_val_macro_f1}])
    test_metrics_df.to_csv(args.output_dir / "family_test_metrics.csv", index=False)

    y_true_family_names = family_encoder.inverse_transform(y_true_family)
    y_pred_family_names = family_encoder.inverse_transform(y_pred_family)

    report = classification_report(
        y_true_family_names,
        y_pred_family_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(args.output_dir / "family_classification_report.csv")

    cm = confusion_matrix(y_true_family_names, y_pred_family_names, labels=family_encoder.classes_)
    pd.DataFrame(cm, index=family_encoder.classes_, columns=family_encoder.classes_).to_csv(
        args.output_dir / "family_confusion_matrix.csv"
    )

    predictions = pd.DataFrame({
        "object_id": object_ids[test_indices],
        "true_fine_label": y_fine[test_indices],
        "true_fine_class_name": [label_names[int(v)] for v in y_fine[test_indices]],
        "true_family_label": y_true_family,
        "true_family_name": y_true_family_names,
        "predicted_family_label": y_pred_family,
        "predicted_family_name": y_pred_family_names,
        "confidence": y_prob_family.max(axis=1),
        "correct": y_true_family == y_pred_family,
    })
    predictions.to_csv(args.output_dir / "family_test_predictions.csv", index=False)
    np.save(args.output_dir / "family_test_probabilities.npy", y_prob_family)

    # Save metadata useful for future hierarchical experiments.
    metadata = {
        "family_classes": family_encoder.classes_.tolist(),
        "label_names": {str(k): v for k, v in label_names.items()},
        "fine_label_to_family": {str(k): map_to_family(v) for k, v in label_names.items()},
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "split_source": split_source,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
    }
    (args.output_dir / "family_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFinal family test results:")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:       {test_metrics['weighted_f1']:.4f}")
    print(f"Mean confidence:   {test_metrics['mean_confidence']:.4f}")

    print("\nSaved:")
    print(f"- {args.output_dir / 'best_hybrid_family_classifier.pt'}")
    print(f"- {args.output_dir / 'family_training_history.csv'}")
    print(f"- {args.output_dir / 'family_test_metrics.csv'}")
    print(f"- {args.output_dir / 'family_test_predictions.csv'}")
    print(f"- {args.output_dir / 'family_test_probabilities.npy'}")
    print(f"- {args.output_dir / 'family_classification_report.csv'}")
    print(f"- {args.output_dir / 'family_model_metadata.json'}")


if __name__ == "__main__":
    main()
