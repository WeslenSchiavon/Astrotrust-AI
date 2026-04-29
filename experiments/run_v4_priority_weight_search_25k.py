from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "features_v4_temporal_shape_25000obj.parquet"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "v4_priority_weight_search_25k"

RANDOM_STATE = 42
BUDGETS = [0.05, 0.10, 0.20]
N_RANDOM_RUNS = 1000


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n_classes = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropy / np.log(n_classes)


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_v = np.nanmin(values)
    max_v = np.nanmax(values)

    if np.isclose(min_v, max_v):
        return np.zeros_like(values)

    return (values - min_v) / (max_v - min_v)


def prepare_dataset():
    df = pd.read_parquet(FEATURES_PATH)

    object_ids = df["object_id"].to_numpy()
    y = df["label"].astype(int).to_numpy()

    X = df.drop(columns=["object_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.astype(float)

    return X, y, object_ids


def load_global_rarity():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    counts = counts.sort_values("rarity_rank").copy()

    n_rare = max(1, int(np.ceil(len(counts) * 0.25)))
    rare_classes = counts.head(n_rare)["label"].astype(int).tolist()

    global_counts = dict(
        zip(
            counts["label"].astype(int),
            counts["n_objects"].astype(int),
        )
    )

    class_names = dict(
        zip(
            counts["label"].astype(int),
            counts["class_name"].astype(str),
        )
    )

    print("\nGlobal rarity definition:")
    print(f"Total classes: {len(counts)}")
    print(f"Rare classes selected: {n_rare}")
    print(counts.head(n_rare)[["label", "class_name", "n_objects", "fraction", "rarity_rank"]])

    return global_counts, rare_classes, class_names


def make_lightgbm():
    return LGBMClassifier(
        objective="multiclass",
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


def mahalanobis_squared(X, mean, precision):
    diff = X - mean
    return np.sum(diff @ precision * diff, axis=1)


def fit_class_mahalanobis_models(X_train, y_train):
    models = {}

    for cls in np.unique(y_train):
        X_cls = X_train[y_train == cls]

        covariance_model = LedoitWolf()
        covariance_model.fit(X_cls)

        models[int(cls)] = {
            "mean": covariance_model.location_,
            "precision": covariance_model.precision_,
            "n": len(X_cls),
        }

    return models


def compute_mahalanobis_novelty(X_eval, class_models):
    all_distances = []

    for _, model in class_models.items():
        distances = mahalanobis_squared(
            X_eval,
            model["mean"],
            model["precision"],
        )
        all_distances.append(distances)

    all_distances = np.vstack(all_distances).T
    return np.min(all_distances, axis=1)


def compute_global_rarity_weights(global_counts, predicted_labels):
    rarity = []

    for label in predicted_labels:
        count = global_counts.get(int(label), None)

        if count is None:
            rarity.append(0.0)
        else:
            rarity.append(1.0 / np.log1p(count))

    return minmax_scale(np.asarray(rarity))


def predict_from_probabilities(probabilities, classes):
    pred_idx = np.argmax(probabilities, axis=1)
    return np.asarray(classes)[pred_idx].astype(int)


def build_scoring_table(
    split_name,
    object_ids,
    y_true,
    probabilities,
    classes,
    uncertainty,
    novelty_score,
    raw_novelty,
    global_counts,
    rare_classes,
    class_names,
):
    y_pred = predict_from_probabilities(probabilities, classes)
    confidence = np.max(probabilities, axis=1)
    rarity_score = compute_global_rarity_weights(global_counts, y_pred)

    df = pd.DataFrame({
        "split": split_name,
        "object_id": object_ids,
        "true_label": y_true,
        "true_class_name": [class_names.get(int(v), str(v)) for v in y_true],
        "predicted_label": y_pred,
        "predicted_class_name": [class_names.get(int(v), str(v)) for v in y_pred],
        "correct": y_true == y_pred,
        "confidence": confidence,
        "uncertainty_score": uncertainty,
        "raw_novelty": raw_novelty,
        "novelty_score": novelty_score,
        "rarity_score": rarity_score,
    })

    df["true_is_rare"] = df["true_label"].isin(rare_classes)
    df["predicted_is_rare"] = df["predicted_label"].isin(rare_classes)

    return df


def summarize_selection(df):
    return {
        "accuracy": df["correct"].mean(),
        "error_rate": 1.0 - df["correct"].mean(),
        "rare_true_rate": df["true_is_rare"].mean(),
        "mean_uncertainty": df["uncertainty_score"].mean(),
        "mean_novelty": df["novelty_score"].mean(),
        "mean_rarity": df["rarity_score"].mean(),
        "mean_priority": df["priority_score"].mean(),
    }


def compute_enrichment_vs_random(ranking, budgets=BUDGETS, n_random_runs=N_RANDOM_RUNS):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for budget in budgets:
        n = len(ranking)
        k = max(1, int(np.ceil(n * budget)))

        top_selected = ranking.sort_values("priority_score", ascending=False).head(k)
        top_summary = summarize_selection(top_selected)

        random_summaries = []

        for _ in range(n_random_runs):
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

            "top_mean_rarity": top_summary["mean_rarity"],
            "random_mean_rarity": random_mean["mean_rarity"],

            "top_mean_priority": top_summary["mean_priority"],
            "random_mean_priority": random_mean["mean_priority"],
        })

    return pd.DataFrame(rows)


def apply_priority_score(df, w_uncertainty, w_novelty, w_rarity):
    out = df.copy()

    out["priority_score"] = (
        w_uncertainty * out["uncertainty_score"]
        + w_novelty * out["novelty_score"]
        + w_rarity * out["rarity_score"]
    )

    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)

    return out


def objective_from_enrichment(enrichment_df):
    """
    Science utility objective.

    We emphasize limited follow-up budgets:
    top 5% has weight 0.50,
    top 10% has weight 0.35,
    top 20% has weight 0.15.
    """
    weights = {
        0.05: 0.50,
        0.10: 0.35,
        0.20: 0.15,
    }

    score = 0.0

    for budget, w in weights.items():
        row = enrichment_df[enrichment_df["budget_fraction"] == budget]
        if not row.empty:
            score += w * float(row.iloc[0]["rare_enrichment"])

    return score


def make_weight_grid(step=0.10):
    values = np.round(np.arange(0.0, 1.0 + step, step), 2)

    rows = []

    for wu, wn, wr in itertools.product(values, values, values):
        if np.isclose(wu + wn + wr, 1.0):
            rows.append((float(wu), float(wn), float(wr)))

    return rows


def evaluate_weight_configuration(df, name, wu, wn, wr, n_random_runs=N_RANDOM_RUNS):
    ranking = apply_priority_score(
        df,
        w_uncertainty=wu,
        w_novelty=wn,
        w_rarity=wr,
    )

    enrichment = compute_enrichment_vs_random(
        ranking,
        budgets=BUDGETS,
        n_random_runs=n_random_runs,
    )

    enrichment["configuration"] = name
    enrichment["w_uncertainty"] = wu
    enrichment["w_novelty"] = wn
    enrichment["w_rarity"] = wr
    enrichment["objective"] = objective_from_enrichment(enrichment)

    return enrichment, ranking


def evaluate_classification(y_true, probabilities, classes):
    y_pred = predict_from_probabilities(probabilities, classes)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_counts, rare_classes, class_names = load_global_rarity()

    print(f"\nLoading features from: {FEATURES_PATH}")

    X, y, object_ids = prepare_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(X)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Features: {X.shape[1]}")

    # Final test is held out.
    X_train_all, X_test, y_train_all, y_test, obj_train_all, obj_test = train_test_split(
        X,
        y,
        object_ids,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split remaining data into model-training, calibration, and priority-validation.
    X_train_cal, X_priority_val, y_train_cal, y_priority_val, obj_train_cal, obj_priority_val = train_test_split(
        X_train_all,
        y_train_all,
        obj_train_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_all,
    )

    X_train, X_cal, y_train, y_cal, obj_train, obj_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        obj_train_cal,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_cal,
    )

    print("\nSplits:")
    print(f"Train size:              {len(X_train)}")
    print(f"Calibration size:        {len(X_cal)}")
    print(f"Priority validation:     {len(X_priority_val)}")
    print(f"Final test size:         {len(X_test)}")

    print("\nTraining LightGBM v4 baseline...")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    base_model = make_lightgbm()
    base_model.fit(X_train, y_train, sample_weight=sample_weight)

    print("\nFitting sigmoid calibration...")
    sigmoid = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv="prefit",
    )
    sigmoid.fit(X_cal, y_cal)

    print("\nComputing probabilities...")

    val_probs = sigmoid.predict_proba(X_priority_val)
    test_probs = sigmoid.predict_proba(X_test)
    classes = sigmoid.classes_

    val_classification = evaluate_classification(y_priority_val, val_probs, classes)
    test_classification = evaluate_classification(y_test, test_probs, classes)

    print("\nClassification performance:")
    print("Priority validation:")
    print(val_classification)
    print("Final test:")
    print(test_classification)

    print("\nFitting Mahalanobis novelty models...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_priority_val)
    X_test_scaled = scaler.transform(X_test)

    class_models = fit_class_mahalanobis_models(X_train_scaled, y_train)

    val_raw_novelty = compute_mahalanobis_novelty(X_val_scaled, class_models)
    test_raw_novelty = compute_mahalanobis_novelty(X_test_scaled, class_models)

    # Normalize novelty jointly using validation+test scale from this experiment.
    # In a stricter deployment setting, this would be fixed from validation only.
    combined_novelty = np.concatenate([val_raw_novelty, test_raw_novelty])
    combined_novelty_scaled = minmax_scale(combined_novelty)

    val_novelty_score = combined_novelty_scaled[:len(val_raw_novelty)]
    test_novelty_score = combined_novelty_scaled[len(val_raw_novelty):]

    val_uncertainty = normalized_entropy(val_probs)
    test_uncertainty = normalized_entropy(test_probs)

    val_scoring = build_scoring_table(
        split_name="priority_validation",
        object_ids=obj_priority_val,
        y_true=y_priority_val,
        probabilities=val_probs,
        classes=classes,
        uncertainty=val_uncertainty,
        novelty_score=val_novelty_score,
        raw_novelty=val_raw_novelty,
        global_counts=global_counts,
        rare_classes=rare_classes,
        class_names=class_names,
    )

    test_scoring = build_scoring_table(
        split_name="final_test",
        object_ids=obj_test,
        y_true=y_test,
        probabilities=test_probs,
        classes=classes,
        uncertainty=test_uncertainty,
        novelty_score=test_novelty_score,
        raw_novelty=test_raw_novelty,
        global_counts=global_counts,
        rare_classes=rare_classes,
        class_names=class_names,
    )

    val_scoring.to_csv(OUTPUT_DIR / "priority_validation_scoring_table.csv", index=False)
    test_scoring.to_csv(OUTPUT_DIR / "final_test_scoring_table.csv", index=False)

    # Ablation configurations.
    ablations = [
        ("uncertainty_only", 1.0, 0.0, 0.0),
        ("novelty_only", 0.0, 1.0, 0.0),
        ("rarity_only", 0.0, 0.0, 1.0),
        ("uncertainty_novelty", 0.5, 0.5, 0.0),
        ("uncertainty_rarity", 0.5, 0.0, 0.5),
        ("novelty_rarity", 0.0, 0.5, 0.5),
        ("fixed_current", 0.4, 0.4, 0.2),
    ]

    print("\nRunning ablation on priority validation...")

    val_ablation_rows = []

    for name, wu, wn, wr in ablations:
        enrichment, _ = evaluate_weight_configuration(
            val_scoring,
            name=name,
            wu=wu,
            wn=wn,
            wr=wr,
            n_random_runs=N_RANDOM_RUNS,
        )
        val_ablation_rows.append(enrichment)

    val_ablation_df = pd.concat(val_ablation_rows, ignore_index=True)
    val_ablation_df.to_csv(OUTPUT_DIR / "validation_ablation_summary.csv", index=False)

    print("\nValidation ablation summary:")
    print(val_ablation_df.sort_values(["budget_fraction", "rare_enrichment"], ascending=[True, False]))

    print("\nRunning grid search on priority validation...")

    grid_rows = []
    weight_grid = make_weight_grid(step=0.10)

    for i, (wu, wn, wr) in enumerate(weight_grid, start=1):
        name = f"grid_{i:03d}"

        enrichment, _ = evaluate_weight_configuration(
            val_scoring,
            name=name,
            wu=wu,
            wn=wn,
            wr=wr,
            n_random_runs=300,  # faster for grid search
        )

        objective = objective_from_enrichment(enrichment)

        grid_rows.append({
            "configuration": name,
            "w_uncertainty": wu,
            "w_novelty": wn,
            "w_rarity": wr,
            "objective": objective,
            "top5_enrichment": float(enrichment[enrichment["budget_fraction"] == 0.05]["rare_enrichment"].iloc[0]),
            "top10_enrichment": float(enrichment[enrichment["budget_fraction"] == 0.10]["rare_enrichment"].iloc[0]),
            "top20_enrichment": float(enrichment[enrichment["budget_fraction"] == 0.20]["rare_enrichment"].iloc[0]),
        })

        if i % 10 == 0 or i == len(weight_grid):
            print(f"  evaluated {i}/{len(weight_grid)} weight configurations")

    grid_df = pd.DataFrame(grid_rows)
    grid_df = grid_df.sort_values("objective", ascending=False).reset_index(drop=True)
    grid_df.to_csv(OUTPUT_DIR / "validation_weight_search.csv", index=False)

    best = grid_df.iloc[0]

    best_wu = float(best["w_uncertainty"])
    best_wn = float(best["w_novelty"])
    best_wr = float(best["w_rarity"])

    print("\nBest validation weights:")
    print(best)

    # Final test evaluation:
    print("\nEvaluating selected configurations on final test...")

    test_configs = ablations + [
        ("validation_optimized", best_wu, best_wn, best_wr),
    ]

    test_rows = []
    ranking_paths = []

    for name, wu, wn, wr in test_configs:
        enrichment, ranking = evaluate_weight_configuration(
            test_scoring,
            name=name,
            wu=wu,
            wn=wn,
            wr=wr,
            n_random_runs=N_RANDOM_RUNS,
        )

        test_rows.append(enrichment)

        ranking_path = OUTPUT_DIR / f"final_test_ranking_{name}.csv"
        ranking.to_csv(ranking_path, index=False)
        ranking_paths.append(ranking_path)

    test_summary = pd.concat(test_rows, ignore_index=True)
    test_summary.to_csv(OUTPUT_DIR / "final_test_ablation_and_optimized_summary.csv", index=False)

    selected = pd.DataFrame([{
        "selected_by": "priority_validation",
        "objective": float(best["objective"]),
        "w_uncertainty": best_wu,
        "w_novelty": best_wn,
        "w_rarity": best_wr,
        "validation_top5_enrichment": float(best["top5_enrichment"]),
        "validation_top10_enrichment": float(best["top10_enrichment"]),
        "validation_top20_enrichment": float(best["top20_enrichment"]),
        **{f"test_{k}": v for k, v in test_classification.items()},
    }])

    selected.to_csv(OUTPUT_DIR / "selected_priority_weights.csv", index=False)

    print("\nFinal test ablation and optimized summary:")
    print(test_summary.sort_values(["budget_fraction", "rare_enrichment"], ascending=[True, False]))

    print("\nSelected weights:")
    print(selected)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'priority_validation_scoring_table.csv'}")
    print(f"- {OUTPUT_DIR / 'final_test_scoring_table.csv'}")
    print(f"- {OUTPUT_DIR / 'validation_ablation_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'validation_weight_search.csv'}")
    print(f"- {OUTPUT_DIR / 'final_test_ablation_and_optimized_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'selected_priority_weights.csv'}")
    for path in ranking_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()