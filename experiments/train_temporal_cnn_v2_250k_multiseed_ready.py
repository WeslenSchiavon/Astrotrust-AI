from pathlib import Path
import time
import argparse

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

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "temporal_cnn_v2_250k"

DEFAULT_SPLIT_SEED = 42
TEST_SIZE = 0.25
VAL_SIZE_FROM_TRAIN = 0.15

BATCH_SIZE = 512
EPOCHS = 120
PATIENCE = 18
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
LABEL_SMOOTHING = 0.02
GRAD_CLIP_NORM = 3.0


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def top_k_accuracy(y_true, y_prob, k: int) -> float:
    k = min(k, y_prob.shape[1])
    topk = np.argpartition(-y_prob, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def multiclass_brier_score(y_true, y_prob, n_classes: int) -> float:
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def negative_log_likelihood(y_true, y_prob, eps: float = 1e-12) -> float:
    probs = np.clip(y_prob[np.arange(len(y_true)), y_true.astype(int)], eps, 1.0)
    return float(-np.mean(np.log(probs)))


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)
    return float(ece)


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


class ResidualDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout):
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        residual = self.proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.act(out)
        out = self.dropout(out)

        return out


class TemporalCNNv2(nn.Module):
    def __init__(self, n_channels=18, n_classes=32):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.blocks = nn.Sequential(
            ResidualDilatedBlock(64, 128, dilation=1, dropout=0.10),
            ResidualDilatedBlock(128, 128, dilation=2, dropout=0.10),
            ResidualDilatedBlock(128, 192, dilation=4, dropout=0.15),
            ResidualDilatedBlock(192, 256, dilation=8, dropout=0.15),
            ResidualDilatedBlock(256, 256, dilation=16, dropout=0.20),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(192, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        avg = self.avg_pool(x)
        mx = self.max_pool(x)

        x = torch.cat([avg, mx], dim=1)
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
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(X_batch)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.append(y_batch.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    n_classes = y_prob.shape[1]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "top2_accuracy": top_k_accuracy(y_true, y_prob, 2),
        "top3_accuracy": top_k_accuracy(y_true, y_prob, 3),
        "top5_accuracy": top_k_accuracy(y_true, y_prob, 5),
        "mean_confidence": float(np.max(y_prob, axis=1).mean()),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=15),
        "brier_score": multiclass_brier_score(y_true, y_prob, n_classes=n_classes),
        "negative_log_likelihood": negative_log_likelihood(y_true, y_prob),
    }

    return metrics, y_true, y_pred, y_prob


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()

    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        batch_size = len(y_batch)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / n_samples


def main():
    parser = argparse.ArgumentParser(description="Train temporal CNN 250k with multi-seed support.")
    parser.add_argument("--seed", type=int, default=42, help="Training random seed.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED, help="Fixed split seed for train/val/test.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where outputs are saved.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic CuDNN settings when possible.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed, deterministic=args.deterministic)

    print(f"Training seed: {args.seed}")
    print(f"Split seed:    {args.split_seed}")
    print(f"Output dir:    {output_dir}")
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
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index())

    train_idx, test_idx = train_test_split(
        np.arange(n_objects),
        test_size=TEST_SIZE,
        random_state=args.split_seed,
        stratify=y,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=args.split_seed,
        stratify=y[train_idx],
    )

    print("\nSplits:")
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")

    train_dataset = LightCurveDataset(X_lc[train_idx], y[train_idx])
    val_dataset = LightCurveDataset(X_lc[val_idx], y[val_idx])
    test_dataset = LightCurveDataset(X_lc[test_idx], y[test_idx])

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
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

    model = TemporalCNNv2(
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

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    history = []
    best_model_path = output_dir / "best_temporal_cnn_v2.pt"

    print("\nTraining TemporalCNN v2...")

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

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

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
    print(f"Top-3 accuracy:    {test_metrics['top3_accuracy']:.4f}")
    print(f"Top-5 accuracy:    {test_metrics['top5_accuracy']:.4f}")
    print(f"ECE:               {test_metrics['ece']:.4f}")
    print(f"Brier score:       {test_metrics['brier_score']:.4f}")
    print(f"NLL:               {test_metrics['negative_log_likelihood']:.4f}")

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

    with open(output_dir / "temporal_cnn_v2_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame([{
        "model": "temporal_cnn_v2_residual_dilated",
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
    }]).to_csv(output_dir / "temporal_cnn_v2_test_metrics.csv", index=False)

    pd.DataFrame({
        "object_id": object_ids[test_idx],
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "confidence": np.max(y_prob, axis=1),
    }).to_csv(output_dir / "temporal_cnn_v2_test_predictions.csv", index=False)

    np.save(output_dir / "temporal_cnn_v2_test_probabilities.npy", y_prob)

    print("\nSaved:")
    print(f"- {best_model_path}")
    print(f"- {output_dir / 'training_history.csv'}")
    print(f"- {output_dir / 'temporal_cnn_v2_test_metrics.csv'}")
    print(f"- {output_dir / 'temporal_cnn_v2_test_predictions.csv'}")
    print(f"- {output_dir / 'temporal_cnn_v2_test_probabilities.npy'}")


if __name__ == "__main__":
    main()