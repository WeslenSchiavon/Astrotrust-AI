from pathlib import Path

import sys
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

FEATURES_PATH = ROOT_DIR / Path("data/processed/elasticc2/features_1000obj.parquet")
RESULTS_DIR = ROOT_DIR / Path("results/novelty_detection_mahalanobis")

MIN_OBJECTS_PER_CLASS = 20
TEST_SIZE = 0.25
RANDOM_STATE = 42


def mahalanobis_squared(X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = X - mean
    return np.sum(diff @ precision * diff, axis=1)


def recall_at_top_fraction(y_true_ood: np.ndarray, novelty_scores: np.ndarray, top_fraction: float) -> float:
    n = len(y_true_ood)
    k = max(1, int(np.ceil(n * top_fraction)))

    top_indices = np.argsort(novelty_scores)[::-1][:k]

    total_ood = np.sum(y_true_ood == 1)
    if total_ood == 0:
        return 0.0

    recovered_ood = np.sum(y_true_ood[top_indices] == 1)

    return recovered_ood / total_ood


def fit_class_mahalanobis_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    models = {}

    for cls in np.unique(y_train):
        X_cls = X_train[y_train == cls]

        # LedoitWolf is more stable than ordinary covariance for small datasets.
        covariance_model = LedoitWolf()
        covariance_model.fit(X_cls)

        models[cls] = {
            "mean": covariance_model.location_,
            "precision": covariance_model.precision_,
            "n": len(X_cls),
        }

    return models


def compute_novelty_scores(X_eval: np.ndarray, class_models: dict) -> np.ndarray:
    all_distances = []

    for cls, model in class_models.items():
        distances = mahalanobis_squared(
            X_eval,
            model["mean"],
            model["precision"],
        )
        all_distances.append(distances)

    all_distances = np.vstack(all_distances).T

    # If an object is far from every known class, it is more novel.
    min_distance_to_known_class = np.min(all_distances, axis=1)

    return min_distance_to_known_class


def run_single_held_out_experiment(df: pd.DataFrame, held_out_class: int) -> dict:
    known_df = df[df["label"] != held_out_class].copy()
    ood_df = df[df["label"] == held_out_class].copy()

    y_known = known_df["label"].to_numpy()
    X_known = known_df.drop(columns=["object_id", "label"])

    X_known_train, X_known_test, y_known_train, y_known_test = train_test_split(
        X_known,
        y_known,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_known,
    )

    X_ood = ood_df.drop(columns=["object_id", "label"])

    scaler = StandardScaler()
    X_known_train_scaled = scaler.fit_transform(X_known_train)
    X_known_test_scaled = scaler.transform(X_known_test)
    X_ood_scaled = scaler.transform(X_ood)

    class_models = fit_class_mahalanobis_models(
        X_known_train_scaled,
        y_known_train,
    )

    X_eval = np.vstack([X_known_test_scaled, X_ood_scaled])

    novelty_scores = compute_novelty_scores(X_eval, class_models)

    y_true_ood = np.concatenate([
        np.zeros(len(X_known_test_scaled), dtype=int),
        np.ones(len(X_ood_scaled), dtype=int),
    ])

    eval_labels = np.concatenate([
        y_known_test,
        np.full(len(X_ood_scaled), held_out_class),
    ])

    eval_kind = np.array(
        ["known"] * len(X_known_test_scaled) + ["held_out_ood"] * len(X_ood_scaled)
    )

    roc_auc = roc_auc_score(y_true_ood, novelty_scores)
    avg_precision = average_precision_score(y_true_ood, novelty_scores)

    recall_top_5 = recall_at_top_fraction(y_true_ood, novelty_scores, top_fraction=0.05)
    recall_top_10 = recall_at_top_fraction(y_true_ood, novelty_scores, top_fraction=0.10)
    recall_top_20 = recall_at_top_fraction(y_true_ood, novelty_scores, top_fraction=0.20)

    known_scores = novelty_scores[y_true_ood == 0]
    ood_scores = novelty_scores[y_true_ood == 1]

    predictions = pd.DataFrame({
        "held_out_class": held_out_class,
        "eval_label": eval_labels,
        "kind": eval_kind,
        "is_ood": y_true_ood,
        "novelty_score": novelty_scores,
    })

    predictions_path = RESULTS_DIR / f"predictions_heldout_{held_out_class}.csv"
    predictions.to_csv(predictions_path, index=False)

    return {
        "held_out_class": held_out_class,
        "n_known_train": len(X_known_train),
        "n_known_test": len(X_known_test),
        "n_ood_test": len(X_ood),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "recall_at_top_5_percent": recall_top_5,
        "recall_at_top_10_percent": recall_top_10,
        "recall_at_top_20_percent": recall_top_20,
        "mean_novelty_known": float(np.mean(known_scores)),
        "mean_novelty_ood": float(np.mean(ood_scores)),
        "median_novelty_known": float(np.median(known_scores)),
        "median_novelty_ood": float(np.median(ood_scores)),
        "predictions_file": str(predictions_path),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    class_counts = df["label"].value_counts().sort_index()
    valid_classes = class_counts[class_counts >= MIN_OBJECTS_PER_CLASS].index.tolist()

    df = df[df["label"].isin(valid_classes)].copy()

    print("\nValid classes used in Mahalanobis novelty experiment:")
    print(df["label"].value_counts().sort_index())

    print(f"\nNumber of valid classes: {len(valid_classes)}")
    print(f"Number of objects: {len(df)}")

    results = []

    for held_out_class in valid_classes:
        print("\n" + "=" * 80)
        print(f"Holding out class {held_out_class} as unknown/OOD")

        result = run_single_held_out_experiment(df, held_out_class)
        results.append(result)

        print(f"ROC-AUC:             {result['roc_auc']:.4f}")
        print(f"Average Precision:   {result['average_precision']:.4f}")
        print(f"Recall@Top 5%:       {result['recall_at_top_5_percent']:.4f}")
        print(f"Recall@Top 10%:      {result['recall_at_top_10_percent']:.4f}")
        print(f"Recall@Top 20%:      {result['recall_at_top_20_percent']:.4f}")
        print(f"Mean novelty known:  {result['mean_novelty_known']:.4f}")
        print(f"Mean novelty OOD:    {result['mean_novelty_ood']:.4f}")

    results_df = pd.DataFrame(results)

    summary_path = RESULTS_DIR / "novelty_detection_mahalanobis_summary.csv"
    results_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("Summary:")
    print(
        results_df[
            [
                "held_out_class",
                "n_ood_test",
                "roc_auc",
                "average_precision",
                "recall_at_top_10_percent",
                "mean_novelty_known",
                "mean_novelty_ood",
            ]
        ].sort_values("roc_auc", ascending=False)
    )

    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()