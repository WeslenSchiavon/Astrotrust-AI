from pathlib import Path
import time
import argparse
import random

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

TENSOR_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "temporal_tensor_v1_250000obj_64bins.npz"
)

DEFAULT_HYBRID_PROBS_PATH = (
    ROOT_DIR
    / "results"
    / "hybrid_temporal_tabular_cnn_250k"
    / "hybrid_temporal_tabular_cnn_test_probabilities.npy"
)

DEFAULT_HYBRID_PREDS_PATH = (
    ROOT_DIR
    / "results"
    / "hybrid_temporal_tabular_cnn_250k"
    / "hybrid_temporal_tabular_cnn_test_predictions.csv"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "hybrid_tabular_ensemble_250k"

HYBRID_PROBS_PATH = DEFAULT_HYBRID_PROBS_PATH
HYBRID_PREDS_PATH = DEFAULT_HYBRID_PREDS_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

RANDOM_STATE = 42
DEFAULT_SPLIT_SEED = 42
TEST_SIZE = 0.25


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def load_label_names():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    return dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))


def load_tensor_dataset():
    data = np.load(TENSOR_PATH, allow_pickle=True)

    if "X_tab" not in data:
        raise ValueError(
            "X_tab not found in tensor dataset. "
            "Rebuild temporal tensor dataset with --include-tabular."
        )

    X_tab = data["X_tab"].astype(np.float32)
    y = data["y"].astype(np.int64)
    object_ids = data["object_id"].astype(np.int64)

    return X_tab, y, object_ids


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


def make_xgboost_deeper(n_classes):
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
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def align_probabilities(probabilities, model_classes, all_classes):
    aligned = np.zeros((probabilities.shape[0], len(all_classes)), dtype=np.float32)

    class_to_col = {
        int(cls): idx
        for idx, cls in enumerate(all_classes)
    }

    for local_idx, cls in enumerate(model_classes):
        global_idx = class_to_col[int(cls)]
        aligned[:, global_idx] = probabilities[:, local_idx]

    return aligned


def evaluate_probabilities(name, y_true, probabilities, all_classes):
    y_pred = all_classes[np.argmax(probabilities, axis=1)]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    mean_confidence = float(np.max(probabilities, axis=1).mean())
    top2 = top_k_accuracy(y_true, probabilities, 2)
    top3 = top_k_accuracy(y_true, probabilities, 3)
    top5 = top_k_accuracy(y_true, probabilities, 5)
    ece = expected_calibration_error(y_true, probabilities, n_bins=15)
    brier = multiclass_brier_score(y_true, probabilities, n_classes=len(all_classes))
    nll = negative_log_likelihood(y_true, probabilities)

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


def train_predict_proba(name, model, X_train, X_test, y_train, all_classes):
    print("\n" + "=" * 80)
    print(f"Training {name}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    start = time.time()

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )

    elapsed = time.time() - start

    print(f"Training time: {elapsed / 60:.2f} min")

    probabilities = model.predict_proba(X_test)

    probabilities = align_probabilities(
        probabilities=probabilities,
        model_classes=model.classes_,
        all_classes=all_classes,
    )

    return probabilities, elapsed / 60


def save_metrics(name, metrics):
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / f"{name}_test_metrics.csv", index=False)


def save_report(name, y_true, y_pred):
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

    with open(OUTPUT_DIR / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)


def save_predictions(name, object_ids_test, y_true, y_pred, probabilities):
    pd.DataFrame({
        "object_id": object_ids_test,
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "confidence": np.max(probabilities, axis=1),
    }).to_csv(OUTPUT_DIR / f"{name}_predictions.csv", index=False)


def save_probabilities(name, probabilities):
    np.save(
        OUTPUT_DIR / f"{name}_test_probabilities.npy",
        probabilities.astype(np.float32),
    )


def check_hybrid_alignment(object_ids_test, y_test):
    if not HYBRID_PREDS_PATH.exists():
        print("[WARN] Hybrid predictions file not found. Skipping alignment check.")
        return

    preds = pd.read_csv(HYBRID_PREDS_PATH)

    if len(preds) != len(y_test):
        raise ValueError(
            f"Hybrid predictions length mismatch: {len(preds)} vs {len(y_test)}"
        )

    if "object_id" in preds.columns:
        same_objects = np.all(preds["object_id"].to_numpy(dtype=np.int64) == object_ids_test)
        print(f"Hybrid object_id alignment: {same_objects}")

        if not same_objects:
            raise ValueError("Hybrid predictions are not aligned with current test split.")

    if "true_label" in preds.columns:
        same_labels = np.all(preds["true_label"].to_numpy(dtype=np.int64) == y_test)
        print(f"Hybrid true_label alignment: {same_labels}")

        if not same_labels:
            raise ValueError("Hybrid labels are not aligned with current test split.")


def main():
    parser = argparse.ArgumentParser(description="Run hybrid + tabular ensemble with multi-seed support.")
    parser.add_argument("--seed", type=int, default=42, help="Training random seed for tabular members.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED, help="Fixed split seed for train/test alignment.")
    parser.add_argument("--input-root", type=Path, default=None, help="Seed-specific root directory containing hybrid outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where outputs are saved.")
    parser.add_argument("--hybrid-probs", type=Path, default=None, help="Optional path to hybrid test probabilities.")
    parser.add_argument("--hybrid-preds", type=Path, default=None, help="Optional path to hybrid test predictions CSV.")
    args = parser.parse_args()

    global OUTPUT_DIR, HYBRID_PROBS_PATH, HYBRID_PREDS_PATH, RANDOM_STATE
    RANDOM_STATE = int(args.seed)
    OUTPUT_DIR = Path(args.output_dir)

    if args.hybrid_probs is not None:
        HYBRID_PROBS_PATH = Path(args.hybrid_probs)
    elif args.input_root is not None:
        HYBRID_PROBS_PATH = Path(args.input_root) / "hybrid_temporal_tabular_cnn" / "hybrid_temporal_tabular_cnn_test_probabilities.npy"
    else:
        HYBRID_PROBS_PATH = DEFAULT_HYBRID_PROBS_PATH

    if args.hybrid_preds is not None:
        HYBRID_PREDS_PATH = Path(args.hybrid_preds)
    elif args.input_root is not None:
        HYBRID_PREDS_PATH = Path(args.input_root) / "hybrid_temporal_tabular_cnn" / "hybrid_temporal_tabular_cnn_test_predictions.csv"
    else:
        HYBRID_PREDS_PATH = DEFAULT_HYBRID_PREDS_PATH

    set_global_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training seed: {args.seed}")
    print(f"Split seed:    {args.split_seed}")
    print(f"Input root:    {args.input_root}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Hybrid probs:  {HYBRID_PROBS_PATH}")
    print(f"Hybrid preds:  {HYBRID_PREDS_PATH}")
    print(f"Loading tensor dataset: {TENSOR_PATH}")

    X_tab, y, object_ids = load_tensor_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(y)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"X_tab shape: {X_tab.shape}")

    print("\nClass distribution:")
    print(pd.Series(y).value_counts().sort_index())

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
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

    if not HYBRID_PROBS_PATH.exists():
        raise FileNotFoundError(HYBRID_PROBS_PATH)

    print(f"\nLoading hybrid probabilities: {HYBRID_PROBS_PATH}")
    hybrid_probs = np.load(HYBRID_PROBS_PATH).astype(np.float32)

    if hybrid_probs.shape[0] != len(test_idx):
        raise ValueError(
            f"Hybrid probabilities length mismatch: "
            f"{hybrid_probs.shape[0]} vs {len(test_idx)}"
        )

    check_hybrid_alignment(object_ids_test, y_test)

    probabilities_by_model = {
        "hybrid_cnn_tabular": hybrid_probs,
    }

    results = []

    # Evaluate loaded hybrid.
    metrics, y_pred = evaluate_probabilities(
        name="hybrid_cnn_tabular",
        y_true=y_test,
        probabilities=hybrid_probs,
        all_classes=all_classes,
    )
    results.append(metrics)
    save_metrics("hybrid_cnn_tabular", metrics)
    save_predictions("hybrid_cnn_tabular", object_ids_test, y_test, y_pred, hybrid_probs)
    save_probabilities("hybrid_cnn_tabular", hybrid_probs)
    save_report("hybrid_cnn_tabular", y_test, y_pred)

    tabular_models = [
        ("lightgbm_baseline", make_lightgbm_baseline(n_classes)),
        ("lightgbm_regularized", make_lightgbm_regularized(n_classes)),
        ("xgboost_gpu_deeper", make_xgboost_deeper(n_classes)),
    ]

    for name, model in tabular_models:
        probs, train_time = train_predict_proba(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
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

        save_metrics(name, metrics)
        save_predictions(name, object_ids_test, y_test, y_pred, probs)
        save_probabilities(name, probs)
        save_report(name, y_test, y_pred)

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

        metrics, y_pred = evaluate_probabilities(
            name=ensemble_name,
            y_true=y_test,
            probabilities=probs,
            all_classes=all_classes,
        )

        metrics["members"] = ",".join(weights.keys())
        metrics["weights"] = ",".join([f"{k}:{v}" for k, v in weights.items()])
        results.append(metrics)

        save_metrics(ensemble_name, metrics)
        save_predictions(ensemble_name, object_ids_test, y_test, y_pred, probs)
        save_probabilities(ensemble_name, probs)
        save_report(ensemble_name, y_test, y_pred)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("macro_f1", ascending=False)

    output_path = OUTPUT_DIR / "hybrid_tabular_ensemble_comparison_250k.csv"
    results_df.to_csv(output_path, index=False)

    # Save the configured final model in an unambiguous file. The multi-seed
    # orchestrator prioritizes files containing the model directory name, so
    # this prevents it from summarizing a member model such as xgboost_gpu_deeper.
    final_model_name = "ensemble_hybrid_dominant"
    final_rows = results_df.loc[results_df["model"] == final_model_name].copy()
    if final_rows.empty:
        raise RuntimeError(f"Could not find {final_model_name} in ensemble results.")

    final_metrics_path = OUTPUT_DIR / f"{final_model_name}_test_metrics.csv"
    final_rows.to_csv(final_metrics_path, index=False)

    # Alias used for quick manual inspection.
    selected_metrics_path = OUTPUT_DIR / "final_selected_ensemble_test_metrics.csv"
    final_rows.to_csv(selected_metrics_path, index=False)

    print("\n" + "=" * 80)
    print("Hybrid + tabular ensemble comparison:")
    print(results_df)

    print("\nFinal selected ensemble metrics:")
    print(final_rows.to_string(index=False))

    print("\nSaved:")
    print(f"- {output_path}")
    print(f"- {final_metrics_path}")
    print(f"- {selected_metrics_path}")


if __name__ == "__main__":
    main()