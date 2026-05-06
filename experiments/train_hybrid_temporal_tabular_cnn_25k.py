from pathlib import Path
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


ROOT_DIR = Path(__file__).resolve().parents[1]

TENSOR_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "temporal_tensor_v1_25000obj_64bins.npz"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_25k"

RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE_FROM_TRAIN = 0.15

BATCH_SIZE = 512
EPOCHS = 80
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0


class HybridDataset(Dataset):
    def __init__(self, X_lc, X_tab, y):
        self.X_lc = X_lc
        self.X_tab = X_tab
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_lc = torch.from_numpy(self.X_lc[idx]).float()
        x_tab = torch.from_numpy(self.X_tab[idx]).float()
        y = torch.tensor(self.y[idx]).long()
        return x_lc, x_tab, y


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
            nn.Linear(256 + 256, 384),
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


def load_label_names():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))


def evaluate_model(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for X_lc_batch, X_tab_batch, y_batch in loader:
            X_lc_batch = X_lc_batch.to(device, non_blocking=True)
            X_tab_batch = X_tab_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(X_lc_batch, X_tab_batch)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.append(y_batch.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "mean_confidence": float(np.max(y_prob, axis=1).mean()),
    }

    return metrics, y_true, y_pred, y_prob


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()

    running_loss = 0.0
    n_samples = 0

    for X_lc_batch, X_tab_batch, y_batch in loader:
        X_lc_batch = X_lc_batch.to(device, non_blocking=True)
        X_tab_batch = X_tab_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(X_lc_batch, X_tab_batch)
            loss = criterion(logits, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = len(y_batch)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / n_samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tensor dataset: {TENSOR_PATH}")

    data = np.load(TENSOR_PATH, allow_pickle=True)

    if "X_tab" not in data:
        raise ValueError(
            "X_tab not found in tensor dataset. "
            "Rebuild with --include-tabular."
        )

    X_lc = data["X_lc"].astype(np.float32)
    X_tab = data["X_tab"].astype(np.float32)
    y = data["y"].astype(np.int64)
    object_ids = data["object_id"].astype(np.int64)

    n_objects = len(y)
    n_classes = len(np.unique(y))
    n_channels = X_lc.shape[1]
    n_tabular = X_tab.shape[1]

    print("\nDataset summary:")
    print(f"Objects: {n_objects}")
    print(f"Classes: {n_classes}")
    print(f"X_lc shape: {X_lc.shape}")
    print(f"X_tab shape: {X_tab.shape}")
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index())

    train_idx, test_idx = train_test_split(
        np.arange(n_objects),
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

    print("\nSplits:")
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")

    # Fit tabular scaler only on train.
    scaler_tab = StandardScaler()
    X_tab_train = scaler_tab.fit_transform(X_tab[train_idx]).astype(np.float32)
    X_tab_val = scaler_tab.transform(X_tab[val_idx]).astype(np.float32)
    X_tab_test = scaler_tab.transform(X_tab[test_idx]).astype(np.float32)

    # Replace any accidental NaN/Inf after scaling.
    X_tab_train = np.nan_to_num(X_tab_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_val = np.nan_to_num(X_tab_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_test = np.nan_to_num(X_tab_test, nan=0.0, posinf=0.0, neginf=0.0)

    train_dataset = HybridDataset(X_lc[train_idx], X_tab_train, y[train_idx])
    val_dataset = HybridDataset(X_lc[val_idx], X_tab_val, y[val_idx])
    test_dataset = HybridDataset(X_lc[test_idx], X_tab_test, y[test_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = HybridTemporalTabularCNN(
        n_channels=n_channels,
        n_tabular=n_tabular,
        n_classes=n_classes,
    ).to(device)

    classes = np.unique(y[train_idx])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y[train_idx],
    )

    full_weights = np.ones(n_classes, dtype=np.float32)

    for cls, weight in zip(classes, class_weights):
        full_weights[int(cls)] = weight

    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    history = []

    best_model_path = OUTPUT_DIR / "best_hybrid_temporal_tabular_cnn.pt"

    print("\nTraining hybrid temporal + tabular CNN...")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )

        val_metrics, _, _, _ = evaluate_model(model, val_loader, device)

        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_val_macro_f1,
                    "n_channels": n_channels,
                    "n_tabular": n_tabular,
                    "n_classes": n_classes,
                },
                best_model_path,
            )

            print(f"  [OK] New best model saved: val_macro_f1={best_val_macro_f1:.4f}")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    elapsed = time.time() - start_time

    print(f"\nTraining finished in {elapsed / 60:.2f} min")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val Macro-F1: {best_val_macro_f1:.4f}")

    print("\nLoading best model for final test evaluation...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    print("\nFinal test results:")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:       {test_metrics['weighted_f1']:.4f}")
    print(f"Mean confidence:   {test_metrics['mean_confidence']:.4f}")

    label_to_name = load_label_names()
    labels = sorted(np.unique(y_true))
    target_names = [label_to_name.get(int(label), str(label)) for label in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    with open(OUTPUT_DIR / "hybrid_temporal_tabular_cnn_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame([{
        "model": "hybrid_temporal_tabular_cnn",
        "n_objects": n_objects,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_channels": n_channels,
        "n_tabular": n_tabular,
        "n_classes": n_classes,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "training_time_minutes": elapsed / 60,
        **test_metrics,
    }]).to_csv(OUTPUT_DIR / "hybrid_temporal_tabular_cnn_test_metrics.csv", index=False)

    pd.DataFrame({
        "object_id": object_ids[test_idx],
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "confidence": np.max(y_prob, axis=1),
    }).to_csv(OUTPUT_DIR / "hybrid_temporal_tabular_cnn_test_predictions.csv", index=False)

    np.save(OUTPUT_DIR / "hybrid_temporal_tabular_cnn_test_probabilities.npy", y_prob)

    print("\nSaved:")
    print(f"- {best_model_path}")
    print(f"- {OUTPUT_DIR / 'training_history.csv'}")
    print(f"- {OUTPUT_DIR / 'hybrid_temporal_tabular_cnn_test_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'hybrid_temporal_tabular_cnn_test_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'hybrid_temporal_tabular_cnn_test_probabilities.npy'}")


if __name__ == "__main__":
    main()