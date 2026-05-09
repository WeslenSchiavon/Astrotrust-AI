from pathlib import Path
import argparse
import random
import time

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_CONFIGS = {
    "25k": {
        "tensor": ROOT_DIR / "data" / "processed" / "elasticc2_large" / "temporal_tensor_v1_25000obj_64bins.npz",
        "hybrid_dir": ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_25k",
        "output_dir": ROOT_DIR / "results" / "hybrid_tabular_ensemble_25k",
    },
    "100k": {
        "tensor": ROOT_DIR / "data" / "processed" / "elasticc2_large" / "temporal_tensor_v1_100000obj_64bins.npz",
        "hybrid_dir": ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_100k",
        "output_dir": ROOT_DIR / "results" / "hybrid_tabular_ensemble_100k",
    },
    "250k": {
        "tensor": ROOT_DIR / "data" / "processed" / "elasticc2_large" / "temporal_tensor_v1_250000obj_64bins.npz",
        "hybrid_dir": ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_250k",
        "output_dir": ROOT_DIR / "results" / "hybrid_tabular_ensemble_250k_scaling",
    },
}

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

RANDOM_STATE = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sanitize_probabilities(y_prob, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps)
    probs = np.clip(probs, eps, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = probs / np.maximum(row_sum, eps)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), eps)
    return probs.astype(np.float32)


def top_k_accuracy(y_true, y_prob, all_classes, k: int) -> float:
    probs = sanitize_probabilities(y_prob)
    k = min(k, probs.shape[1])
    topk_cols = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    topk_labels = all_classes[topk_cols]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk_labels)]))


def multiclass_brier_score(y_true, y_prob, all_classes) -> float:
    probs = sanitize_probabilities(y_prob)
    class_to_col = {int(cls): i for i, cls in enumerate(all_classes)}
    y_onehot = np.zeros((len(y_true), len(all_classes)), dtype=np.float32)
    y_cols = np.array([class_to_col[int(y)] for y in y_true], dtype=int)
    y_onehot[np.arange(len(y_true)), y_cols] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def negative_log_likelihood(y_true, y_prob, all_classes, eps: float = 1e-12) -> float:
    probs = sanitize_probabilities(y_prob, eps=eps)
    class_to_col = {int(cls): i for i, cls in enumerate(all_classes)}
    y_cols = np.array([class_to_col[int(y)] for y in y_true], dtype=int)
    true_probs = np.clip(probs[np.arange(len(y_true)), y_cols], eps, 1.0)
    return float(-np.mean(np.log(true_probs)))


def expected_calibration_error(y_true, y_prob, all_classes, n_bins: int = 15) -> float:
    probs = sanitize_probabilities(y_prob)
    confidences = np.max(probs, axis=1)
    pred_cols = np.argmax(probs, axis=1)
    predictions = all_classes[pred_cols]
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


def load_label_names():
    if not CLASS_COUNTS_PATH.exists():
        return {}

    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))


def load_tensor_dataset(tensor_path: Path):
    data = np.load(tensor_path, allow_pickle=True)

    if "X_tab" not in data:
        raise ValueError(
            "X_tab not found in tensor dataset. "
            "Rebuild temporal tensor dataset with --include-tabular."
        )

    X_tab = data["X_tab"].astype(np.float32)
    y = data["y"].astype(np.int64)

    if "object_id" in data:
        object_ids = data["object_id"]
    elif "object_ids" in data:
        object_ids = data["object_ids"]
    else:
        object_ids = np.arange(len(y))

    return X_tab, y, object_ids


def make_feature_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=[f"feature_{i:03d}" for i in range(X.shape[1])])


def make_lightgbm_baseline(n_classes):
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
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def make_lightgbm_regularized(n_classes):
    return LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def make_xgboost_deeper(n_classes, device: str):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=900,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=2,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=1.0,
        max_bin=256,
        tree_method="hist",
        device=device,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def align_probabilities(probabilities, model_classes, all_classes):
    aligned = np.zeros((probabilities.shape[0], len(all_classes)), dtype=np.float32)
    class_to_col = {int(cls): idx for idx, cls in enumerate(all_classes)}

    for local_idx, cls in enumerate(model_classes):
        global_idx = class_to_col[int(cls)]
        aligned[:, global_idx] = probabilities[:, local_idx]

    return sanitize_probabilities(aligned)


def evaluate_probabilities(name, y_true, probabilities, all_classes):
    probabilities = sanitize_probabilities(probabilities)
    y_pred = all_classes[np.argmax(probabilities, axis=1)]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    mean_confidence = float(np.max(probabilities, axis=1).mean())

    top2 = top_k_accuracy(y_true, probabilities, all_classes, 2)
    top3 = top_k_accuracy(y_true, probabilities, all_classes, 3)
    top5 = top_k_accuracy(y_true, probabilities, all_classes, 5)
    ece = expected_calibration_error(y_true, probabilities, all_classes, n_bins=15)
    brier = multiclass_brier_score(y_true, probabilities, all_classes)
    nll = negative_log_likelihood(y_true, probabilities, all_classes)

    print("\n" + "=" * 80)
    print(f"Results: {name}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")
    print(f"Top-3 accuracy:    {top3:.4f}")
    print(f"Top-5 accuracy:    {top5:.4f}")
    print(f"Mean confidence:   {mean_confidence:.4f}")
    print(f"ECE:               {ece:.4f}")
    print(f"Brier score:       {brier:.4f}")
    print(f"NLL:               {nll:.4f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "mean_confidence": mean_confidence,
        "ece": ece,
        "brier_score": brier,
        "negative_log_likelihood": nll,
    }, y_pred


def train_predict_proba(name, model, X_train_df, X_test_df, y_train, all_classes):
    print("\n" + "=" * 80)
    print(f"Training {name}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    start = time.time()

    model.fit(
        X_train_df,
        y_train,
        sample_weight=sample_weight,
    )

    elapsed = time.time() - start
    print(f"Training time: {elapsed / 60:.2f} min")

    probabilities = model.predict_proba(X_test_df)
    probabilities = align_probabilities(
        probabilities=probabilities,
        model_classes=model.classes_,
        all_classes=all_classes,
    )

    return probabilities, elapsed / 60


def save_metrics(output_dir: Path, name, metrics):
    pd.DataFrame([metrics]).to_csv(output_dir / f"{name}_test_metrics.csv", index=False)


def save_report(output_dir: Path, name, y_true, y_pred):
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

    with open(output_dir / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


def save_predictions(output_dir: Path, name, object_ids_test, y_true, y_pred, probabilities):
    pd.DataFrame({
        "object_id": object_ids_test,
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "confidence": np.max(sanitize_probabilities(probabilities), axis=1),
    }).to_csv(output_dir / f"{name}_predictions.csv", index=False)


def save_probabilities(output_dir: Path, name, probabilities):
    np.save(
        output_dir / f"{name}_test_probabilities.npy",
        sanitize_probabilities(probabilities).astype(np.float32),
    )


def check_hybrid_alignment(hybrid_preds_path: Path, object_ids_test, y_test):
    if not hybrid_preds_path.exists():
        print("[WARN] Hybrid predictions file not found. Skipping alignment check.")
        return

    preds = pd.read_csv(hybrid_preds_path)

    if len(preds) != len(y_test):
        raise ValueError(
            f"Hybrid predictions length mismatch: {len(preds)} vs {len(y_test)}"
        )

    if "object_id" in preds.columns:
        expected = object_ids_test.astype(str)
        observed = preds["object_id"].astype(str).to_numpy()
        same_objects = np.all(observed == expected)
        print(f"Hybrid object_id alignment: {same_objects}")

        if not same_objects:
            raise ValueError("Hybrid predictions are not aligned with current test split.")

    if "true_label" in preds.columns:
        same_labels = np.all(preds["true_label"].to_numpy(dtype=np.int64) == y_test)
        print(f"Hybrid true_label alignment: {same_labels}")

        if not same_labels:
            raise ValueError("Hybrid labels are not aligned with current test split.")


def resolve_paths(args):
    cfg = DATASET_CONFIGS[args.dataset_size]

    tensor_path = Path(args.tensor_path) if args.tensor_path else cfg["tensor"]
    hybrid_dir = cfg["hybrid_dir"]

    hybrid_probs = (
        Path(args.hybrid_probs)
        if args.hybrid_probs
        else hybrid_dir / "hybrid_temporal_tabular_cnn_test_probabilities.npy"
    )

    hybrid_preds = (
        Path(args.hybrid_preds)
        if args.hybrid_preds
        else hybrid_dir / "hybrid_temporal_tabular_cnn_test_predictions.csv"
    )

    output_dir = Path(args.output_dir) if args.output_dir else cfg["output_dir"]

    return tensor_path, hybrid_probs, hybrid_preds, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run AstroTrust-AI hybrid + tabular ensemble for dataset-size scaling."
    )
    parser.add_argument("--dataset-size", choices=["25k", "100k", "250k"], required=True)
    parser.add_argument("--tensor-path", type=Path, default=None)
    parser.add_argument("--hybrid-probs", type=Path, default=None)
    parser.add_argument("--hybrid-preds", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--xgb-device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    global RANDOM_STATE
    RANDOM_STATE = int(args.seed)

    set_global_seed(args.seed)
    tensor_path, hybrid_probs_path, hybrid_preds_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AstroTrust-AI scaling ensemble")
    print("=" * 80)
    print(f"Dataset size:  {args.dataset_size}")
    print(f"Training seed: {args.seed}")
    print(f"Split seed:    {args.split_seed}")
    print(f"Tensor path:   {tensor_path}")
    print(f"Hybrid probs:  {hybrid_probs_path}")
    print(f"Hybrid preds:  {hybrid_preds_path}")
    print(f"Output dir:    {output_dir}")
    print(f"XGBoost device:{args.xgb_device}")

    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")

    if not hybrid_probs_path.exists():
        raise FileNotFoundError(f"Hybrid probabilities file not found: {hybrid_probs_path}")

    print(f"\nLoading tensor dataset: {tensor_path}")
    X_tab, y, object_ids = load_tensor_dataset(tensor_path)

    print("\nDataset summary:")
    print(f"Objects: {len(y)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"X_tab shape: {X_tab.shape}")

    print("\nClass distribution:")
    print(pd.Series(y).value_counts().sort_index())

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=y,
    )

    X_train = X_tab[train_idx]
    X_test = X_tab[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    object_ids_test = object_ids[test_idx]

    all_classes = np.asarray(sorted(np.unique(y)), dtype=int)
    n_classes = len(all_classes)

    print("\nSplits:")
    print(f"Train: {len(train_idx)}")
    print(f"Test:  {len(test_idx)}")
    print(f"Classes: {n_classes}")

    print(f"\nLoading hybrid probabilities: {hybrid_probs_path}")
    hybrid_probs = sanitize_probabilities(np.load(hybrid_probs_path))

    if hybrid_probs.shape[0] != len(test_idx):
        raise ValueError(
            f"Hybrid probabilities length mismatch: "
            f"{hybrid_probs.shape[0]} vs {len(test_idx)}. "
            f"Check --tensor-path, --split-seed, and --test-size."
        )

    check_hybrid_alignment(hybrid_preds_path, object_ids_test, y_test)

    X_train_df = make_feature_frame(X_train)
    X_test_df = make_feature_frame(X_test)

    probabilities_by_model = {
        "hybrid_cnn_tabular": hybrid_probs,
    }

    results = []

    metrics, y_pred = evaluate_probabilities(
        name="hybrid_cnn_tabular",
        y_true=y_test,
        probabilities=hybrid_probs,
        all_classes=all_classes,
    )
    results.append(metrics)
    save_metrics(output_dir, "hybrid_cnn_tabular", metrics)
    save_predictions(output_dir, "hybrid_cnn_tabular", object_ids_test, y_test, y_pred, hybrid_probs)
    save_probabilities(output_dir, "hybrid_cnn_tabular", hybrid_probs)
    save_report(output_dir, "hybrid_cnn_tabular", y_test, y_pred)

    tabular_models = [
        ("lightgbm_baseline", make_lightgbm_baseline(n_classes)),
        ("lightgbm_regularized", make_lightgbm_regularized(n_classes)),
        ("xgboost_gpu_deeper", make_xgboost_deeper(n_classes, device=args.xgb_device)),
    ]

    for name, model in tabular_models:
        probs, train_time = train_predict_proba(
            name=name,
            model=model,
            X_train_df=X_train_df,
            X_test_df=X_test_df,
            y_train=y_train,
            all_classes=all_classes,
        )

        probabilities_by_model[name] = probs

        metrics, y_pred = evaluate_probabilities(
            name=name,
            y_true=y_test,
            probabilities=probs,
            all_classes=all_classes,
        )

        metrics["training_time_minutes"] = train_time
        results.append(metrics)

        save_metrics(output_dir, name, metrics)
        save_predictions(output_dir, name, object_ids_test, y_test, y_pred, probs)
        save_probabilities(output_dir, name, probs)
        save_report(output_dir, name, y_test, y_pred)

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
        print("\n" + "=" * 80)
        print(f"Evaluating {ensemble_name}")
        print(f"Weights: {weights}")

        total_weight = sum(weights.values())
        probs = np.zeros_like(hybrid_probs, dtype=np.float32)

        for member_name, weight in weights.items():
            probs += (weight / total_weight) * probabilities_by_model[member_name]

        probs = sanitize_probabilities(probs)

        metrics, y_pred = evaluate_probabilities(
            name=ensemble_name,
            y_true=y_test,
            probabilities=probs,
            all_classes=all_classes,
        )

        metrics["members"] = ",".join(weights.keys())
        metrics["weights"] = ",".join([f"{k}:{v}" for k, v in weights.items()])
        results.append(metrics)

        save_metrics(output_dir, ensemble_name, metrics)
        save_predictions(output_dir, ensemble_name, object_ids_test, y_test, y_pred, probs)
        save_probabilities(output_dir, ensemble_name, probs)
        save_report(output_dir, ensemble_name, y_test, y_pred)

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)

    comparison_path = output_dir / f"hybrid_tabular_ensemble_comparison_{args.dataset_size}.csv"
    results_df.to_csv(comparison_path, index=False)

    final_model_name = "ensemble_hybrid_dominant"
    final_rows = results_df.loc[results_df["model"] == final_model_name].copy()

    if final_rows.empty:
        raise RuntimeError(f"Could not find {final_model_name} in ensemble results.")

    final_metrics_path = output_dir / f"{final_model_name}_test_metrics.csv"
    final_rows.to_csv(final_metrics_path, index=False)

    selected_metrics_path = output_dir / "final_selected_ensemble_test_metrics.csv"
    final_rows.to_csv(selected_metrics_path, index=False)

    print("\n" + "=" * 80)
    print("Hybrid + tabular ensemble comparison:")
    print(results_df)

    print("\nFinal selected ensemble metrics:")
    print(final_rows.to_string(index=False))

    print("\nSaved:")
    print(f"- {comparison_path}")
    print(f"- {final_metrics_path}")
    print(f"- {selected_metrics_path}")


if __name__ == "__main__":
    main()
