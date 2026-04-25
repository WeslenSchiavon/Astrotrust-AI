from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT_DIR / Path("data/processed/elasticc2/features_v3_contextual_1000obj.parquet")
RESULTS_DIR = ROOT_DIR / Path("results/followup_prioritization_v3_calibrated_hgb")

MIN_OBJECTS_PER_CLASS = 20
RANDOM_STATE = 42

W_UNCERTAINTY = 0.40
W_NOVELTY = 0.40
W_RARITY = 0.20

BUDGETS = [0.05, 0.10, 0.20]
N_RANDOM_RUNS = 1000


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n_classes = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropy / np.log(n_classes)


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_v = np.min(values)
    max_v = np.max(values)

    if np.isclose(min_v, max_v):
        return np.zeros_like(values)

    return (values - min_v) / (max_v - min_v)


def mahalanobis_squared(X: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = X - mean
    return np.sum(diff @ precision * diff, axis=1)


def fit_class_mahalanobis_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    models = {}

    for cls in np.unique(y_train):
        X_cls = X_train[y_train == cls]

        covariance_model = LedoitWolf()
        covariance_model.fit(X_cls)

        models[cls] = {
            "mean": covariance_model.location_,
            "precision": covariance_model.precision_,
            "n": len(X_cls),
        }

    return models


def compute_mahalanobis_novelty(X_eval: np.ndarray, class_models: dict) -> np.ndarray:
    all_distances = []

    for _, model in class_models.items():
        distances = mahalanobis_squared(
            X_eval,
            model["mean"],
            model["precision"],
        )
        all_distances.append(distances)

    all_distances = np.vstack(all_distances).T

    # Far from all known classes = more novel.
    min_distance = np.min(all_distances, axis=1)

    return min_distance


def compute_rarity_weights(y_train_original: np.ndarray, predicted_labels_original: np.ndarray) -> np.ndarray:
    train_counts = pd.Series(y_train_original).value_counts()

    rarity = []

    for label in predicted_labels_original:
        count = train_counts.get(label, 1)
        rarity.append(1.0 / np.log1p(count))

    return minmax_scale(np.asarray(rarity))


def priority_category(score: float) -> str:
    if score >= 0.75:
        return "very_high"
    if score >= 0.50:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def make_base_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def summarize_selection(df: pd.DataFrame) -> dict:
    return {
        "accuracy": df["correct"].mean(),
        "error_rate": 1.0 - df["correct"].mean(),
        "rare_true_rate": df["true_is_rare"].mean(),
        "mean_uncertainty": df["uncertainty_score"].mean(),
        "mean_novelty": df["novelty_score"].mean(),
        "mean_rarity": df["rarity_score"].mean(),
        "mean_priority": df["priority_score"].mean(),
    }


def compute_enrichment_vs_random(ranking: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for budget in BUDGETS:
        n = len(ranking)
        k = max(1, int(np.ceil(n * budget)))

        top_selected = ranking.sort_values("priority_score", ascending=False).head(k)
        top_summary = summarize_selection(top_selected)

        random_summaries = []

        for _ in range(N_RANDOM_RUNS):
            random_selected = ranking.sample(
                n=k,
                replace=False,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            random_summaries.append(summarize_selection(random_selected))

        random_df = pd.DataFrame(random_summaries)
        random_mean = random_df.mean()
        random_std = random_df.std()

        rows.append({
            "budget_fraction": budget,
            "n_selected": k,

            "top_accuracy": top_summary["accuracy"],
            "random_accuracy_mean": random_mean["accuracy"],
            "random_accuracy_std": random_std["accuracy"],

            "top_error_rate": top_summary["error_rate"],
            "random_error_rate_mean": random_mean["error_rate"],
            "random_error_rate_std": random_std["error_rate"],

            "top_rare_true_rate": top_summary["rare_true_rate"],
            "random_rare_true_rate_mean": random_mean["rare_true_rate"],
            "random_rare_true_rate_std": random_std["rare_true_rate"],

            "rare_enrichment": (
                top_summary["rare_true_rate"] / random_mean["rare_true_rate"]
                if random_mean["rare_true_rate"] > 0 else np.nan
            ),

            "top_mean_uncertainty": top_summary["mean_uncertainty"],
            "random_mean_uncertainty": random_mean["mean_uncertainty"],

            "top_mean_novelty": top_summary["mean_novelty"],
            "random_mean_novelty": random_mean["mean_novelty"],

            "top_mean_priority": top_summary["mean_priority"],
            "random_mean_priority": random_mean["mean_priority"],
        })

    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    class_counts = df["label"].value_counts().sort_index()
    valid_classes = class_counts[class_counts >= MIN_OBJECTS_PER_CLASS].index.tolist()

    df = df[df["label"].isin(valid_classes)].copy()

    print("\nClasses used:")
    print(df["label"].value_counts().sort_index())

    filtered_counts = df["label"].value_counts()
    rare_threshold = filtered_counts.quantile(0.25)
    rare_classes = filtered_counts[filtered_counts <= rare_threshold].index.tolist()

    print(f"\nRare-class threshold: {rare_threshold}")
    print(f"Rare classes: {rare_classes}")

    object_ids = df["object_id"].to_numpy()
    y_original = df["label"].to_numpy()

    X = df.drop(columns=["object_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_original)

    # Split into train/calibration/test.
    X_train_cal, X_test, y_train_cal, y_test, obj_train_cal, obj_test = train_test_split(
        X,
        y_encoded,
        object_ids,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    X_train, X_cal, y_train, y_cal, obj_train, obj_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        obj_train_cal,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_cal,
    )

    y_train_original = label_encoder.inverse_transform(y_train)
    y_test_original = label_encoder.inverse_transform(y_test)

    print(f"\nTrain size: {len(X_train)}")
    print(f"Calibration size: {len(X_cal)}")
    print(f"Test size: {len(X_test)}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print("\nTraining base HistGradientBoosting...")
    base_model = make_base_model()
    base_model.fit(X_train, y_train, sample_weight=sample_weight)

    print("Fitting sigmoid calibration...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv="prefit",
    )
    calibrated_model.fit(X_cal, y_cal)

    probabilities = calibrated_model.predict_proba(X_test)

    y_pred = np.argmax(probabilities, axis=1)
    predicted_labels_original = label_encoder.inverse_transform(y_pred)

    confidence = np.max(probabilities, axis=1)
    uncertainty = normalized_entropy(probabilities)

    print("Fitting Mahalanobis novelty models...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_models = fit_class_mahalanobis_models(X_train_scaled, y_train)
    raw_novelty = compute_mahalanobis_novelty(X_test_scaled, class_models)
    novelty_score = minmax_scale(raw_novelty)

    rarity_score = compute_rarity_weights(
        y_train_original=y_train_original,
        predicted_labels_original=predicted_labels_original,
    )

    priority_score = (
        W_UNCERTAINTY * uncertainty
        + W_NOVELTY * novelty_score
        + W_RARITY * rarity_score
    )

    correct = predicted_labels_original == y_test_original

    ranking = pd.DataFrame({
        "object_id": obj_test,
        "true_label": y_test_original,
        "predicted_label": predicted_labels_original,
        "correct": correct,
        "confidence": confidence,
        "uncertainty_score": uncertainty,
        "raw_novelty": raw_novelty,
        "novelty_score": novelty_score,
        "rarity_score": rarity_score,
        "priority_score": priority_score,
    })

    ranking["priority_category"] = ranking["priority_score"].apply(priority_category)
    ranking["true_is_rare"] = ranking["true_label"].isin(rare_classes)
    ranking["predicted_is_rare"] = ranking["predicted_label"].isin(rare_classes)

    ranking = ranking.sort_values("priority_score", ascending=False).reset_index(drop=True)
    ranking["priority_rank"] = np.arange(1, len(ranking) + 1)

    accuracy = accuracy_score(y_test_original, predicted_labels_original)
    balanced_acc = balanced_accuracy_score(y_test_original, predicted_labels_original)
    macro_f1 = f1_score(y_test_original, predicted_labels_original, average="macro")
    weighted_f1 = f1_score(y_test_original, predicted_labels_original, average="weighted")

    print("\nClassification performance:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")

    print("\nTop 20 candidates by follow-up priority:")
    cols = [
        "priority_rank",
        "object_id",
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
        "uncertainty_score",
        "novelty_score",
        "rarity_score",
        "priority_score",
        "priority_category",
    ]
    print(ranking[cols].head(20))

    print("\nFollow-up budget evaluation:")
    budget_rows = []
    for budget in BUDGETS:
        n = len(ranking)
        k = max(1, int(np.ceil(n * budget)))
        selected = ranking.head(k)

        budget_rows.append({
            "budget_fraction": budget,
            "n_selected": k,
            "selected_accuracy": selected["correct"].mean(),
            "selected_error_rate": 1.0 - selected["correct"].mean(),
            "selected_mean_uncertainty": selected["uncertainty_score"].mean(),
            "selected_mean_novelty": selected["novelty_score"].mean(),
            "selected_mean_rarity": selected["rarity_score"].mean(),
            "selected_rare_true_rate": selected["true_is_rare"].mean(),
        })

    budget_df = pd.DataFrame(budget_rows)
    print(budget_df)

    print("\nComputing enrichment vs random selection...")
    enrichment_df = compute_enrichment_vs_random(ranking)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\nFollow-up enrichment summary:")
    print(enrichment_df)

    metrics = pd.DataFrame([{
        "model": "hist_gradient_boosting_sigmoid_calibrated",
        "feature_set": "features_v3_contextual",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_test": len(y_test_original),
        "n_classes": len(np.unique(y_original)),
        "w_uncertainty": W_UNCERTAINTY,
        "w_novelty": W_NOVELTY,
        "w_rarity": W_RARITY,
    }])

    ranking.to_csv(RESULTS_DIR / "followup_priority_ranking.csv", index=False)
    budget_df.to_csv(RESULTS_DIR / "followup_budget_evaluation.csv", index=False)
    enrichment_df.to_csv(RESULTS_DIR / "followup_enrichment_vs_random.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "classification_metrics.csv", index=False)

    print("\nSaved results:")
    print(f"- {RESULTS_DIR / 'followup_priority_ranking.csv'}")
    print(f"- {RESULTS_DIR / 'followup_budget_evaluation.csv'}")
    print(f"- {RESULTS_DIR / 'followup_enrichment_vs_random.csv'}")
    print(f"- {RESULTS_DIR / 'classification_metrics.csv'}")


if __name__ == "__main__":
    main()