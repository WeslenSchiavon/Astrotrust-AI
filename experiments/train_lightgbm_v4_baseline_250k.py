from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm") from exc


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "features_v4_temporal_shape_250000obj.parquet"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "lightgbm_v4_baseline_250k"

TEST_SIZE = 0.25


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correctness = (predictions == y_true).astype(float)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == 0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences > left) & (confidences <= right)

        if not np.any(mask):
            continue

        bin_weight = float(np.mean(mask))
        bin_acc = float(np.mean(correctness[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def brier_multiclass(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    one_hot = np.zeros((len(y_true), n_classes), dtype=np.float32)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    topk = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def load_label_names() -> dict[int, str]:
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))


def prepare_dataset(features_path: Path):
    df = pd.read_parquet(features_path)

    if "label" not in df.columns:
        raise ValueError(f"Column 'label' not found in {features_path}")

    y = df["label"].astype(int).to_numpy()

    drop_cols = [c for c in ["object_id", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(np.float32)

    if "object_id" in df.columns:
        object_ids = df["object_id"].to_numpy()
    else:
        object_ids = np.arange(len(df))

    return X, y, object_ids, df


def make_lightgbm_v4_baseline(n_classes: int, seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def align_probabilities(probabilities: np.ndarray, model_classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(all_classes)), dtype=np.float32)
    class_to_col = {int(cls): idx for idx, cls in enumerate(all_classes)}

    for local_idx, cls in enumerate(model_classes):
        aligned[:, class_to_col[int(cls)]] = probabilities[:, local_idx]

    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = aligned / np.maximum(row_sum, 1e-12)
    return aligned.astype(np.float32)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    all_classes: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    y_pred = all_classes[np.argmax(probabilities, axis=1)]

    n_classes = len(all_classes)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "top2_accuracy": topk_accuracy(y_true, probabilities, 2),
        "top3_accuracy": topk_accuracy(y_true, probabilities, 3),
        "top5_accuracy": topk_accuracy(y_true, probabilities, 5),
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
        "ece": expected_calibration_error(y_true, probabilities, n_bins=15),
        "brier_score": brier_multiclass(y_true, probabilities, n_classes=n_classes),
        "negative_log_likelihood": log_loss(y_true, probabilities, labels=all_classes),
    }

    return metrics, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the AstroTrust-AI LightGBM v4 baseline on the 250k v4 temporal-shape feature table."
    )
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Training/model random seed.")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed split seed. Keep fixed for training-stability experiments.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training seed: {args.seed}")
    print(f"Split seed:    {args.split_seed}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Features path: {args.features_path}")

    start_total = time.time()

    X, y, object_ids, df = prepare_dataset(args.features_path)
    all_classes = np.asarray(sorted(np.unique(y)), dtype=int)
    n_classes = len(all_classes)

    print("\nDataset summary:")
    print(f"Objects:  {len(y)}")
    print(f"Classes:  {n_classes}")
    print(f"Features: {X.shape[1]}")
    print("\nClass distribution:")
    print(pd.Series(y).value_counts().sort_index())

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=y,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    object_ids_test = object_ids[test_idx]

    print("\nSplits:")
    print(f"Train: {len(train_idx)}")
    print(f"Test:  {len(test_idx)}")

    model = make_lightgbm_v4_baseline(n_classes=n_classes, seed=args.seed)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print("\nTraining LightGBM v4 baseline...")
    start_train = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_elapsed = time.time() - start_train
    print(f"Training time: {train_elapsed / 60:.2f} min")

    print("\nPredicting test probabilities...")
    raw_probabilities = model.predict_proba(X_test)
    probabilities = align_probabilities(
        probabilities=raw_probabilities,
        model_classes=model.classes_,
        all_classes=all_classes,
    )

    metrics, y_pred = evaluate_probabilities(y_test, probabilities, all_classes)

    print("\nFinal test results:")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:       {metrics['weighted_f1']:.4f}")
    print(f"Top-3 accuracy:    {metrics['top3_accuracy']:.4f}")
    print(f"Top-5 accuracy:    {metrics['top5_accuracy']:.4f}")
    print(f"Mean confidence:   {metrics['mean_confidence']:.4f}")
    print(f"ECE:               {metrics['ece']:.4f}")
    print(f"Brier score:       {metrics['brier_score']:.4f}")
    print(f"NLL:               {metrics['negative_log_likelihood']:.4f}")

    label_to_name = load_label_names()
    labels = sorted(np.unique(y_test))
    target_names = [label_to_name.get(int(label), str(label)) for label in labels]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    metrics_row = {
        "model": "lightgbm_v4_baseline",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "n_objects": len(y),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "training_time_minutes": train_elapsed / 60,
        "total_time_minutes": (time.time() - start_total) / 60,
        **metrics,
    }

    pd.DataFrame([metrics_row]).to_csv(
        args.output_dir / "lightgbm_v4_baseline_test_metrics.csv",
        index=False,
    )

    pd.DataFrame({
        "object_id": object_ids_test,
        "true_label": y_test,
        "predicted_label": y_pred,
        "correct": y_test == y_pred,
        "confidence": np.max(probabilities, axis=1),
    }).to_csv(args.output_dir / "lightgbm_v4_baseline_predictions.csv", index=False)

    np.save(
        args.output_dir / "lightgbm_v4_baseline_test_probabilities.npy",
        probabilities.astype(np.float32),
    )

    with open(args.output_dir / "lightgbm_v4_baseline_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    metadata = {
        "model": "lightgbm_v4_baseline",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "features_path": str(args.features_path),
        "output_dir": str(args.output_dir),
        "hyperparameters": model.get_params(),
        "metrics": metrics_row,
    }

    (args.output_dir / "lightgbm_v4_baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"- {args.output_dir / 'lightgbm_v4_baseline_test_metrics.csv'}")
    print(f"- {args.output_dir / 'lightgbm_v4_baseline_predictions.csv'}")
    print(f"- {args.output_dir / 'lightgbm_v4_baseline_test_probabilities.npy'}")
    print(f"- {args.output_dir / 'lightgbm_v4_baseline_classification_report.txt'}")
    print(f"- {args.output_dir / 'lightgbm_v4_baseline_metadata.json'}")


if __name__ == "__main__":
    main()
