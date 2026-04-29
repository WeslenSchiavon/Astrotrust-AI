from pathlib import Path
import time
import json

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
from sklearn.utils.class_weight import compute_class_weight


ROOT_DIR = Path(__file__).resolve().parents[1]

TENSOR_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "temporal_tensor_v1_250000obj_64bins.npz"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "temporal_cnn_250k"

RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE_FROM_TRAIN = 0.15

BATCH_SIZE = 512
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0


class LightCurveDataset(Dataset):
    def __init__(self, X_lc, y):
        self.X_lc = X_lc
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X_lc[idx]).float()
        y = torch.tensor(self.y[idx]).long()
        return x, y


class TemporalCNN(nn.Module):
    def __init__(self, n_channels=18, n_classes=32):
        super().__init__()

        self.features = nn.Sequential(
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

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def load_label_names():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))


def evaluate_model(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
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

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(X_batch)
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

    X_lc = data["X_lc"].astype(np.float32)
    y = data["y"].astype(np.int64)
    object_ids = data["object_id"].astype(np.int64)

    n_objects = len(y)
    n_classes = len(np.unique(y))
    n_channels = X_lc.shape[1]

    print("\nDataset summary:")
    print(f"Objects: {n_objects}")
    print(f"Classes: {n_classes}")
    print(f"X_lc shape: {X_lc.shape}")
    print(f"Class distribution:")
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

    train_dataset = LightCurveDataset(X_lc[train_idx], y[train_idx])
    val_dataset = LightCurveDataset(X_lc[val_idx], y[val_idx])
    test_dataset = LightCurveDataset(X_lc[test_idx], y[test_idx])

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

    model = TemporalCNN(
        n_channels=n_channels,
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

    best_model_path = OUTPUT_DIR / "best_temporal_cnn.pt"

    print("\nTraining temporal CNN...")

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

    with open(OUTPUT_DIR / "temporal_cnn_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame([{
        "model": "temporal_cnn_lc_only",
        "n_objects": n_objects,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_channels": n_channels,
        "n_classes": n_classes,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "training_time_minutes": elapsed / 60,
        **test_metrics,
    }]).to_csv(OUTPUT_DIR / "temporal_cnn_test_metrics.csv", index=False)

    pd.DataFrame({
        "object_id": object_ids[test_idx],
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "confidence": np.max(y_prob, axis=1),
    }).to_csv(OUTPUT_DIR / "temporal_cnn_test_predictions.csv", index=False)

    np.save(OUTPUT_DIR / "temporal_cnn_test_probabilities.npy", y_prob)

    print("\nSaved:")
    print(f"- {best_model_path}")
    print(f"- {OUTPUT_DIR / 'training_history.csv'}")
    print(f"- {OUTPUT_DIR / 'temporal_cnn_test_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'temporal_cnn_test_predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'temporal_cnn_test_probabilities.npy'}")


if __name__ == "__main__":
    main()