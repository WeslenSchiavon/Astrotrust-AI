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
    ROOT_DIR / "data" / "processed" / "elasticc2_large"
    / "features_v4_temporal_shape_25000obj.parquet"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR / "data" / "processed" / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "final_v4_strict_policy_eval_25k"

RANDOM_STATE = 42
BUDGETS = [0.05, 0.10, 0.20]
N_RANDOM_RUNS = 1000


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    n_classes = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropy / np.log(n_classes)


def minmax_fit(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    return np.nanmin(values), np.nanmax(values)


def minmax_transform(values: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if np.isclose(min_v, max_v):
        return np.zeros_like(values)

    scaled = (values - min_v) / (max_v - min_v)
    return np.clip(scaled, 0.0, 1.0)


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

    global_counts = dict(zip(counts["label"].astype(int), counts["n_objects"].astype(int)))
    class_names = dict(zip(counts["label"].astype(int), counts["class_name"].astype(str)))

    print("\nGlobal rarity definition:")
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


def predict_from_probabilities(probabilities, classes):
    pred_idx = np.argmax(probabilities, axis=1)
    return np.asarray(classes)[pred_idx].astype(int)


def compute_global_rarity_weights(global_counts, predicted_labels):
    values = []

    for label in predicted_labels:
        count = global_counts.get(int(label), None)

        if count is None:
            values.append(0.0)
        else:
            values.append(1.0 / np.log1p(count))

    values = np.asarray(values, dtype=float)
    min_v, max_v = minmax_fit(values)
    return minmax_transform(values, min_v, max_v)


def mahalanobis_squared(X, mean, precision):
    diff = X - mean
    return np.sum(diff @ precision * diff, axis=1)


def fit_class_mahalanobis_models(X_train, y_train):
    models = {}

    for cls in np.unique(y_train):
        X_cls = X_train[y_train == cls]

        cov = LedoitWolf()
        cov.fit(X_cls)

        models[int(cls)] = {
            "mean": cov.location_,
            "precision": cov.precision_,
        }

    return models


def compute_mahalanobis_novelty(X_eval, class_models):
    distances = []

    for model in class_models.values():
        d = mahalanobis_squared(X_eval, model["mean"], model["precision"])
        distances.append(d)

    distances = np.vstack(distances).T
    return np.min(distances, axis=1)


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


def apply_priority_score(df, config_name, wu, wn, wr):
    out = df.copy()

    out["configuration"] = config_name
    out["w_uncertainty"] = wu
    out["w_novelty"] = wn
    out["w_rarity"] = wr

    out["priority_score"] = (
        wu * out["uncertainty_score"]
        + wn * out["novelty_score"]
        + wr * out["rarity_score"]
    )

    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)

    return out


def summarize_selection(df):
    return {
        "accuracy": df["correct"].mean(),
        "error_rate": 1.0 - df["correct"].mean(),
        "rare_true_rate": df["true_is_rare"].mean(),
        "predicted_rare_rate": df["predicted_is_rare"].mean(),
        "mean_uncertainty": df["uncertainty_score"].mean(),
        "mean_novelty": df["novelty_score"].mean(),
        "mean_rarity": df["rarity_score"].mean(),
        "mean_priority": df["priority_score"].mean(),
    }


def compute_enrichment_vs_random(ranking):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for budget in BUDGETS:
        n = len(ranking)
        k = max(1, int(np.ceil(n * budget)))

        top_selected = ranking.head(k)
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
            "top_error_rate": top_summary["error_rate"],
            "random_error_rate_mean": random_mean["error_rate"],
            "top_rare_true_rate": top_summary["rare_true_rate"],
            "random_rare_true_rate_mean": random_mean["rare_true_rate"],
            "rare_enrichment": (
                top_summary["rare_true_rate"] / random_mean["rare_true_rate"]
                if random_mean["rare_true_rate"] > 0 else np.nan
            ),
            "top_predicted_rare_rate": top_summary["predicted_rare_rate"],
            "top_mean_uncertainty": top_summary["mean_uncertainty"],
            "random_mean_uncertainty": random_mean["mean_uncertainty"],
            "top_mean_novelty": top_summary["mean_novelty"],
            "random_mean_novelty": random_mean["mean_novelty"],
            "top_mean_rarity": top_summary["mean_rarity"],
            "random_mean_rarity": random_mean["mean_rarity"],
            "top_mean_priority": top_summary["mean_priority"],
            "random_mean_priority": random_mean["mean_priority"],
            "random_rare_true_rate_std": random_std["rare_true_rate"],
        })

    return pd.DataFrame(rows)


def weighted_objective(enrichment_df, mode):
    weights = {0.05: 0.50, 0.10: 0.35, 0.20: 0.15}

    if mode == "rare_retrieval":
        target = "rare_enrichment"
    elif mode == "discovery":
        # A balanced objective: rare enrichment + uncertainty + novelty.
        # It is computed after normalization across configurations outside this function.
        raise ValueError("Discovery objective is computed after collecting configurations.")
    else:
        raise ValueError(mode)

    score = 0.0

    for budget, w in weights.items():
        row = enrichment_df[enrichment_df["budget_fraction"] == budget]
        if not row.empty:
            score += w * float(row.iloc[0][target])

    return score


def make_weight_grid(step=0.10):
    values = np.round(np.arange(0.0, 1.0 + step, step), 2)

    rows = []
    for wu, wn, wr in itertools.product(values, values, values):
        if np.isclose(wu + wn + wr, 1.0):
            rows.append((float(wu), float(wn), float(wr)))

    return rows


def config_summary_from_enrichment(config_name, wu, wn, wr, enrichment):
    weights = {0.05: 0.50, 0.10: 0.35, 0.20: 0.15}

    row = {
        "configuration": config_name,
        "w_uncertainty": wu,
        "w_novelty": wn,
        "w_rarity": wr,
    }

    for budget in BUDGETS:
        b = enrichment[enrichment["budget_fraction"] == budget].iloc[0]
        key = int(budget * 100)
        row[f"top{key}_rare_enrichment"] = float(b["rare_enrichment"])
        row[f"top{key}_rare_rate"] = float(b["top_rare_true_rate"])
        row[f"top{key}_uncertainty"] = float(b["top_mean_uncertainty"])
        row[f"top{key}_novelty"] = float(b["top_mean_novelty"])

    row["weighted_rare_enrichment"] = sum(
        weights[b] * row[f"top{int(b * 100)}_rare_enrichment"]
        for b in BUDGETS
    )

    row["weighted_uncertainty"] = sum(
        weights[b] * row[f"top{int(b * 100)}_uncertainty"]
        for b in BUDGETS
    )

    row["weighted_novelty"] = sum(
        weights[b] * row[f"top{int(b * 100)}_novelty"]
        for b in BUDGETS
    )

    return row


def normalize_column(df, col):
    min_v = df[col].min()
    max_v = df[col].max()

    if np.isclose(min_v, max_v):
        return np.zeros(len(df))

    return (df[col] - min_v) / (max_v - min_v)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_counts, rare_classes, class_names = load_global_rarity()

    print(f"\nLoading features from: {FEATURES_PATH}")
    X, y, object_ids = prepare_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(X)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Features: {X.shape[1]}")

    X_train_all, X_test, y_train_all, y_test, obj_train_all, obj_test = train_test_split(
        X, y, object_ids,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_cal, X_priority_val, y_train_cal, y_priority_val, obj_train_cal, obj_priority_val = train_test_split(
        X_train_all, y_train_all, obj_train_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_all,
    )

    X_train, X_cal, y_train, y_cal, obj_train, obj_cal = train_test_split(
        X_train_cal, y_train_cal, obj_train_cal,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_cal,
    )

    print("\nSplits:")
    print(f"Train:              {len(X_train)}")
    print(f"Calibration:        {len(X_cal)}")
    print(f"Priority validation:{len(X_priority_val)}")
    print(f"Final test:         {len(X_test)}")

    print("\nTraining LightGBM v4...")
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = make_lightgbm()
    model.fit(X_train, y_train, sample_weight=sample_weight)

    print("Calibrating with sigmoid...")
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv="prefit",
    )
    calibrated.fit(X_cal, y_cal)

    val_probs = calibrated.predict_proba(X_priority_val)
    test_probs = calibrated.predict_proba(X_test)
    classes = calibrated.classes_

    val_pred = predict_from_probabilities(val_probs, classes)
    test_pred = predict_from_probabilities(test_probs, classes)

    print("\nClassification:")
    print("Priority validation:")
    print({
        "accuracy": accuracy_score(y_priority_val, val_pred),
        "balanced_accuracy": balanced_accuracy_score(y_priority_val, val_pred),
        "macro_f1": f1_score(y_priority_val, val_pred, average="macro"),
        "weighted_f1": f1_score(y_priority_val, val_pred, average="weighted"),
    })
    print("Final test:")
    print({
        "accuracy": accuracy_score(y_test, test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "macro_f1": f1_score(y_test, test_pred, average="macro"),
        "weighted_f1": f1_score(y_test, test_pred, average="weighted"),
    })

    print("\nFitting novelty models...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_priority_val)
    X_test_scaled = scaler.transform(X_test)

    novelty_models = fit_class_mahalanobis_models(X_train_scaled, y_train)

    val_raw_novelty = compute_mahalanobis_novelty(X_val_scaled, novelty_models)
    test_raw_novelty = compute_mahalanobis_novelty(X_test_scaled, novelty_models)

    # Strict protocol: novelty normalization is fitted on priority validation only.
    novelty_min, novelty_max = minmax_fit(val_raw_novelty)
    val_novelty = minmax_transform(val_raw_novelty, novelty_min, novelty_max)
    test_novelty = minmax_transform(test_raw_novelty, novelty_min, novelty_max)

    val_scoring = build_scoring_table(
        split_name="priority_validation",
        object_ids=obj_priority_val,
        y_true=y_priority_val,
        probabilities=val_probs,
        classes=classes,
        uncertainty=normalized_entropy(val_probs),
        novelty_score=val_novelty,
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
        uncertainty=normalized_entropy(test_probs),
        novelty_score=test_novelty,
        raw_novelty=test_raw_novelty,
        global_counts=global_counts,
        rare_classes=rare_classes,
        class_names=class_names,
    )

    val_scoring.to_csv(OUTPUT_DIR / "priority_validation_scoring_table.csv", index=False)
    test_scoring.to_csv(OUTPUT_DIR / "final_test_scoring_table.csv", index=False)

    print("\nEvaluating policies on priority validation...")

    named_configs = [
        ("uncertainty_only", 1.0, 0.0, 0.0),
        ("novelty_only", 0.0, 1.0, 0.0),
        ("rarity_only", 0.0, 0.0, 1.0),
        ("uncertainty_novelty", 0.5, 0.5, 0.0),
        ("uncertainty_rarity", 0.5, 0.0, 0.5),
        ("novelty_rarity", 0.0, 0.5, 0.5),
        ("fixed_discovery", 0.4, 0.4, 0.2),
    ]

    validation_rows = []

    for name, wu, wn, wr in named_configs:
        ranking = apply_priority_score(val_scoring, name, wu, wn, wr)
        enrichment = compute_enrichment_vs_random(ranking)
        validation_rows.append(config_summary_from_enrichment(name, wu, wn, wr, enrichment))

    # Grid search for validation policies.
    for i, (wu, wn, wr) in enumerate(make_weight_grid(step=0.10), start=1):
        name = f"grid_{i:03d}"
        ranking = apply_priority_score(val_scoring, name, wu, wn, wr)
        enrichment = compute_enrichment_vs_random(ranking)
        validation_rows.append(config_summary_from_enrichment(name, wu, wn, wr, enrichment))

    validation_summary = pd.DataFrame(validation_rows)

    validation_summary["rare_retrieval_score"] = validation_summary["weighted_rare_enrichment"]

    validation_summary["rare_norm"] = normalize_column(validation_summary, "weighted_rare_enrichment")
    validation_summary["uncertainty_norm"] = normalize_column(validation_summary, "weighted_uncertainty")
    validation_summary["novelty_norm"] = normalize_column(validation_summary, "weighted_novelty")

    validation_summary["discovery_utility_score"] = (
        0.40 * validation_summary["rare_norm"]
        + 0.30 * validation_summary["uncertainty_norm"]
        + 0.30 * validation_summary["novelty_norm"]
    )

    validation_summary.to_csv(OUTPUT_DIR / "priority_validation_policy_search.csv", index=False)

    rare_best = validation_summary.sort_values("rare_retrieval_score", ascending=False).iloc[0]
    discovery_best = validation_summary.sort_values("discovery_utility_score", ascending=False).iloc[0]

    selected = pd.DataFrame([
        {
            "policy": "rare_class_retrieval",
            "selected_configuration": rare_best["configuration"],
            "w_uncertainty": rare_best["w_uncertainty"],
            "w_novelty": rare_best["w_novelty"],
            "w_rarity": rare_best["w_rarity"],
            "validation_score": rare_best["rare_retrieval_score"],
        },
        {
            "policy": "discovery_oriented_triage",
            "selected_configuration": discovery_best["configuration"],
            "w_uncertainty": discovery_best["w_uncertainty"],
            "w_novelty": discovery_best["w_novelty"],
            "w_rarity": discovery_best["w_rarity"],
            "validation_score": discovery_best["discovery_utility_score"],
        },
    ])

    selected.to_csv(OUTPUT_DIR / "selected_policies_from_validation.csv", index=False)

    print("\nSelected policies from validation:")
    print(selected)

    print("\nEvaluating selected policies on final test...")

    test_rows = []
    test_enrichments = []

    final_configs = [
        (
            selected.iloc[0]["selected_configuration"],
            float(selected.iloc[0]["w_uncertainty"]),
            float(selected.iloc[0]["w_novelty"]),
            float(selected.iloc[0]["w_rarity"]),
        ),
        (
            selected.iloc[1]["selected_configuration"],
            float(selected.iloc[1]["w_uncertainty"]),
            float(selected.iloc[1]["w_novelty"]),
            float(selected.iloc[1]["w_rarity"]),
        ),
        ("fixed_discovery", 0.4, 0.4, 0.2),
        ("rarity_only", 0.0, 0.0, 1.0),
        ("novelty_rarity", 0.0, 0.5, 0.5),
    ]

    # Deduplicate while preserving order.
    seen = set()
    unique_final_configs = []
    for cfg in final_configs:
        key = cfg
        if key not in seen:
            unique_final_configs.append(cfg)
            seen.add(key)

    for name, wu, wn, wr in unique_final_configs:
        ranking = apply_priority_score(test_scoring, name, wu, wn, wr)
        enrichment = compute_enrichment_vs_random(ranking)

        enrichment["configuration"] = name
        enrichment["w_uncertainty"] = wu
        enrichment["w_novelty"] = wn
        enrichment["w_rarity"] = wr

        test_enrichments.append(enrichment)
        test_rows.append(config_summary_from_enrichment(name, wu, wn, wr, enrichment))

        ranking.to_csv(OUTPUT_DIR / f"final_test_ranking_{name}.csv", index=False)

    test_summary = pd.DataFrame(test_rows)
    test_enrichment_full = pd.concat(test_enrichments, ignore_index=True)

    test_summary.to_csv(OUTPUT_DIR / "final_test_policy_summary.csv", index=False)
    test_enrichment_full.to_csv(OUTPUT_DIR / "final_test_policy_enrichment_by_budget.csv", index=False)

    print("\nFinal test policy summary:")
    print(test_summary.sort_values("weighted_rare_enrichment", ascending=False))

    print("\nFinal test enrichment by budget:")
    print(test_enrichment_full.sort_values(["budget_fraction", "rare_enrichment"], ascending=[True, False]))

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'priority_validation_scoring_table.csv'}")
    print(f"- {OUTPUT_DIR / 'final_test_scoring_table.csv'}")
    print(f"- {OUTPUT_DIR / 'priority_validation_policy_search.csv'}")
    print(f"- {OUTPUT_DIR / 'selected_policies_from_validation.csv'}")
    print(f"- {OUTPUT_DIR / 'final_test_policy_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'final_test_policy_enrichment_by_budget.csv'}")


if __name__ == "__main__":
    main()