from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
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

OUTPUT_DIR = ROOT_DIR / "results" / "v4_lightgbm_calibration_followup_25k"

RANDOM_STATE = 42
N_BINS = 10
BUDGETS = [0.05, 0.10, 0.20]
N_RANDOM_RUNS = 1000

W_UNCERTAINTY = 0.40
W_NOVELTY = 0.40
W_RARITY = 0.20


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


def multiclass_brier_score(y_true, probabilities, classes):
    class_to_col = {int(cls): i for i, cls in enumerate(classes)}

    y_onehot = np.zeros_like(probabilities, dtype=float)

    for i, label in enumerate(y_true):
        col = class_to_col[int(label)]
        y_onehot[i, col] = 1.0

    return np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))


def expected_calibration_error(y_true, y_pred, confidences, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        n_bin = int(np.sum(mask))

        if n_bin == 0:
            rows.append({
                "bin": i,
                "lower": lower,
                "upper": upper,
                "n": 0,
                "accuracy": np.nan,
                "confidence": np.nan,
                "gap": np.nan,
            })
            continue

        bin_accuracy = np.mean(y_true[mask] == y_pred[mask])
        bin_confidence = np.mean(confidences[mask])
        gap = abs(bin_accuracy - bin_confidence)

        ece += (n_bin / len(y_true)) * gap

        rows.append({
            "bin": i,
            "lower": lower,
            "upper": upper,
            "n": n_bin,
            "accuracy": bin_accuracy,
            "confidence": bin_confidence,
            "gap": gap,
        })

    return ece, pd.DataFrame(rows)


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


def predict_from_probabilities(probabilities, classes):
    pred_idx = np.argmax(probabilities, axis=1)
    return np.asarray(classes)[pred_idx].astype(int)


def evaluate_probabilistic_model(name, probabilities, classes, y_test, class_names, output_dir):
    y_pred = predict_from_probabilities(probabilities, classes)
    confidence = np.max(probabilities, axis=1)
    uncertainty = normalized_entropy(probabilities)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    brier = multiclass_brier_score(y_test, probabilities, classes)
    ece, bins = expected_calibration_error(
        y_true=y_test,
        y_pred=y_pred,
        confidences=confidence,
        n_bins=N_BINS,
    )

    print("\n" + "=" * 80)
    print(f"Model: {name}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Weighted-F1:       {weighted_f1:.4f}")
    print(f"Brier score:       {brier:.4f}")
    print(f"ECE:               {ece:.4f}")
    print(f"Mean confidence:   {confidence.mean():.4f}")
    print(f"Mean uncertainty:  {uncertainty.mean():.4f}")

    labels = sorted(np.unique(y_test))
    target_names = [class_names.get(int(label), str(label)) for label in labels]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_pred,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "correct": y_test == y_pred,
    }).to_csv(model_dir / "predictions.csv", index=False)

    bins.to_csv(model_dir / "calibration_bins.csv", index=False)

    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "model": name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "brier_score": brier,
        "ece": ece,
        "mean_confidence": confidence.mean(),
        "mean_uncertainty": uncertainty.mean(),
    }, y_pred, confidence, uncertainty


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


def priority_category(score):
    if score >= 0.75:
        return "very_high"
    if score >= 0.50:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


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


def compute_enrichment_vs_random(ranking):
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

            "top_mean_rarity": top_summary["mean_rarity"],
            "random_mean_rarity": random_mean["mean_rarity"],

            "top_mean_priority": top_summary["mean_priority"],
            "random_mean_priority": random_mean["mean_priority"],
        })

    return pd.DataFrame(rows)


def run_followup(
    name,
    y_test,
    object_ids_test,
    y_pred,
    confidence,
    uncertainty,
    novelty_score,
    raw_novelty,
    global_counts,
    rare_classes,
    class_names,
    output_dir,
):
    rarity_score = compute_global_rarity_weights(
        global_counts=global_counts,
        predicted_labels=y_pred,
    )

    priority_score = (
        W_UNCERTAINTY * uncertainty
        + W_NOVELTY * novelty_score
        + W_RARITY * rarity_score
    )

    ranking = pd.DataFrame({
        "object_id": object_ids_test,
        "true_label": y_test,
        "true_class_name": [class_names.get(int(v), str(v)) for v in y_test],
        "predicted_label": y_pred,
        "predicted_class_name": [class_names.get(int(v), str(v)) for v in y_pred],
        "correct": y_test == y_pred,
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

    print("\n" + "=" * 80)
    print(f"Follow-up priority results: {name}")

    print("\nTop 20 candidates:")
    cols = [
        "priority_rank",
        "object_id",
        "true_class_name",
        "predicted_class_name",
        "correct",
        "confidence",
        "uncertainty_score",
        "novelty_score",
        "rarity_score",
        "priority_score",
        "true_is_rare",
        "predicted_is_rare",
    ]
    print(ranking[cols].head(20))

    print("\nComputing enrichment vs random selection...")
    enrichment = compute_enrichment_vs_random(ranking)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("\nFollow-up enrichment summary:")
    print(enrichment)

    followup_dir = output_dir / name / "followup"
    followup_dir.mkdir(parents=True, exist_ok=True)

    ranking.to_csv(followup_dir / "followup_priority_ranking.csv", index=False)
    enrichment.to_csv(followup_dir / "followup_enrichment_vs_random.csv", index=False)

    return enrichment


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_counts, rare_classes, class_names = load_global_rarity()

    print(f"\nLoading features from: {FEATURES_PATH}")

    X, y, object_ids = prepare_dataset()

    print("\nDataset summary:")
    print(f"Objects: {len(X)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Features: {X.shape[1]}")

    X_train_cal, X_test, y_train_cal, y_test, obj_train_cal, obj_test = train_test_split(
        X,
        y,
        object_ids,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_cal, y_train, y_cal, obj_train, obj_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        obj_train_cal,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_cal,
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Calibration size: {len(X_cal)}")
    print(f"Test size: {len(X_test)}")

    print("\nTraining LightGBM v4 baseline...")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    base_model = make_lightgbm()
    base_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
    )

    print("\nFitting Mahalanobis novelty model...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_models = fit_class_mahalanobis_models(X_train_scaled, y_train)
    raw_novelty = compute_mahalanobis_novelty(X_test_scaled, class_models)
    novelty_score = minmax_scale(raw_novelty)

    results = []
    followup_rows = []

    # Uncalibrated
    probs_uncalibrated = base_model.predict_proba(X_test)
    classes_uncalibrated = base_model.classes_

    metrics, y_pred, confidence, uncertainty = evaluate_probabilistic_model(
        name="lightgbm_v4_uncalibrated",
        probabilities=probs_uncalibrated,
        classes=classes_uncalibrated,
        y_test=y_test,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
    )
    results.append(metrics)

    enrichment = run_followup(
        name="lightgbm_v4_uncalibrated",
        y_test=y_test,
        object_ids_test=obj_test,
        y_pred=y_pred,
        confidence=confidence,
        uncertainty=uncertainty,
        novelty_score=novelty_score,
        raw_novelty=raw_novelty,
        global_counts=global_counts,
        rare_classes=rare_classes,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
    )
    enrichment["model"] = "lightgbm_v4_uncalibrated"
    followup_rows.append(enrichment)

    # Sigmoid calibration
    print("\nFitting sigmoid calibration...")
    sigmoid = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv="prefit",
    )
    sigmoid.fit(X_cal, y_cal)

    probs_sigmoid = sigmoid.predict_proba(X_test)
    classes_sigmoid = sigmoid.classes_

    metrics, y_pred, confidence, uncertainty = evaluate_probabilistic_model(
        name="lightgbm_v4_sigmoid",
        probabilities=probs_sigmoid,
        classes=classes_sigmoid,
        y_test=y_test,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
    )
    results.append(metrics)

    enrichment = run_followup(
        name="lightgbm_v4_sigmoid",
        y_test=y_test,
        object_ids_test=obj_test,
        y_pred=y_pred,
        confidence=confidence,
        uncertainty=uncertainty,
        novelty_score=novelty_score,
        raw_novelty=raw_novelty,
        global_counts=global_counts,
        rare_classes=rare_classes,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
    )
    enrichment["model"] = "lightgbm_v4_sigmoid"
    followup_rows.append(enrichment)

    # Isotonic calibration
    print("\nFitting isotonic calibration...")
    isotonic = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv="prefit",
    )
    isotonic.fit(X_cal, y_cal)

    probs_isotonic = isotonic.predict_proba(X_test)
    classes_isotonic = isotonic.classes_

    metrics, y_pred, confidence, uncertainty = evaluate_probabilistic_model(
        name="lightgbm_v4_isotonic",
        probabilities=probs_isotonic,
        classes=classes_isotonic,
        y_test=y_test,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
    )
    results.append(metrics)

    enrichment = run_followup(
        name="lightgbm_v4_isotonic",
        y_test=y_test,
        object_ids_test=obj_test,
        y_pred=y_pred,
        confidence=confidence,
        uncertainty=uncertainty,
        novelty_score=novelty_score,
        raw_novelty=raw_novelty,
        global_counts=global_counts,
        rare_classes=rare_classes,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
    )
    enrichment["model"] = "lightgbm_v4_isotonic"
    followup_rows.append(enrichment)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["macro_f1", "ece"], ascending=[False, True])

    followup_df = pd.concat(followup_rows, ignore_index=True)

    results_df.to_csv(OUTPUT_DIR / "v4_lightgbm_calibration_summary.csv", index=False)
    followup_df.to_csv(OUTPUT_DIR / "v4_lightgbm_followup_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("V4 LightGBM calibration summary:")
    print(results_df)

    print("\n" + "=" * 80)
    print("V4 LightGBM follow-up summary:")
    print(followup_df.sort_values(["budget_fraction", "rare_enrichment"], ascending=[True, False]))

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'v4_lightgbm_calibration_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'v4_lightgbm_followup_summary.csv'}")


if __name__ == "__main__":
    main()