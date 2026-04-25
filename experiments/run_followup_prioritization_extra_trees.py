from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH =  ROOT_DIR / Path("data/processed/elasticc2/features_1000obj.parquet")
RESULTS_DIR =  ROOT_DIR / Path("results/followup_prioritization_extra_trees")

MIN_OBJECTS_PER_CLASS = 20
TEST_SIZE = 0.25
RANDOM_STATE = 42

W_UNCERTAINTY = 0.40
W_NOVELTY = 0.40
W_RARITY = 0.20


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n_classes = probabilities.shape[1]

    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    max_entropy = np.log(n_classes)

    return entropy / max_entropy


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

    for cls, model in class_models.items():
        distances = mahalanobis_squared(
            X_eval,
            model["mean"],
            model["precision"],
        )
        all_distances.append(distances)

    all_distances = np.vstack(all_distances).T
    min_distance = np.min(all_distances, axis=1)

    return min_distance


def compute_rarity_weights(y_train: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    train_counts = pd.Series(y_train).value_counts()

    rarity = []

    for label in predicted_labels:
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


def evaluate_top_budget(ranking: pd.DataFrame, budget_fraction: float) -> dict:
    n = len(ranking)
    k = max(1, int(np.ceil(n * budget_fraction)))

    selected = ranking.sort_values("priority_score", ascending=False).head(k)

    return {
        "budget_fraction": budget_fraction,
        "n_selected": k,
        "selected_accuracy": selected["correct"].mean(),
        "selected_error_rate": 1.0 - selected["correct"].mean(),
        "selected_mean_uncertainty": selected["uncertainty_score"].mean(),
        "selected_mean_novelty": selected["novelty_score"].mean(),
        "selected_mean_rarity": selected["rarity_score"].mean(),
        "selected_rare_true_rate": selected["true_is_rare"].mean(),
    }


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

    y = df["label"].to_numpy()
    object_ids = df["object_id"].to_numpy()
    X = df.drop(columns=["object_id", "label"])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    X_train, X_test, y_train, y_test, obj_train, obj_test = train_test_split(
        X,
        y,
        object_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Extra Trees classifier...")

    classifier = ExtraTreesClassifier(
        n_estimators=800,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    classifier.fit(X_train, y_train)

    probabilities = classifier.predict_proba(X_test)
    predicted_labels = classifier.classes_[np.argmax(probabilities, axis=1)]

    confidence = np.max(probabilities, axis=1)
    uncertainty = normalized_entropy(probabilities)

    print("Fitting Mahalanobis novelty models...")

    class_models = fit_class_mahalanobis_models(X_train_scaled, y_train)
    raw_novelty = compute_mahalanobis_novelty(X_test_scaled, class_models)

    novelty_score = minmax_scale(raw_novelty)
    rarity_score = compute_rarity_weights(y_train, predicted_labels)

    priority_score = (
        W_UNCERTAINTY * uncertainty
        + W_NOVELTY * novelty_score
        + W_RARITY * rarity_score
    )

    correct = predicted_labels == y_test

    ranking = pd.DataFrame({
        "object_id": obj_test,
        "true_label": y_test,
        "predicted_label": predicted_labels,
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

    accuracy = accuracy_score(y_test, predicted_labels)
    balanced_acc = balanced_accuracy_score(y_test, predicted_labels)
    macro_f1 = f1_score(y_test, predicted_labels, average="macro")
    weighted_f1 = f1_score(y_test, predicted_labels, average="weighted")

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

    budget_results = []

    for budget_fraction in [0.05, 0.10, 0.20]:
        budget_results.append(evaluate_top_budget(ranking, budget_fraction))

    budget_df = pd.DataFrame(budget_results)

    print("\nFollow-up budget evaluation:")
    print(budget_df)

    metrics = pd.DataFrame([{
        "model": "extra_trees",
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_test": len(y_test),
        "n_classes": len(np.unique(y)),
        "w_uncertainty": W_UNCERTAINTY,
        "w_novelty": W_NOVELTY,
        "w_rarity": W_RARITY,
    }])

    ranking.to_csv(RESULTS_DIR / "followup_priority_ranking.csv", index=False)
    budget_df.to_csv(RESULTS_DIR / "followup_budget_evaluation.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "classification_metrics.csv", index=False)

    print("\nSaved results:")
    print(f"- {RESULTS_DIR / 'followup_priority_ranking.csv'}")
    print(f"- {RESULTS_DIR / 'followup_budget_evaluation.csv'}")
    print(f"- {RESULTS_DIR / 'classification_metrics.csv'}")


if __name__ == "__main__":
    main()