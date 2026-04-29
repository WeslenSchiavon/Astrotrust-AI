from pathlib import Path
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]

TENSOR_PATH = (
    ROOT_DIR / "data" / "processed" / "elasticc2_large"
    / "temporal_tensor_v1_250000obj_64bins.npz"
)

CHECKPOINT_PATH = (
    ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_250k"
    / "best_hybrid_temporal_tabular_cnn.pt"
)

CLASS_COUNTS_PATH = (
    ROOT_DIR / "data" / "processed" / "elasticc2_large"
    / "full_class_counts.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "hybrid_followup_policy_eval_250k"

RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE_FROM_TRAIN = 0.15

BATCH_SIZE = 512
NUM_WORKERS = 0
BUDGETS = [0.05, 0.10, 0.20]
N_RANDOM_RUNS = 1000


class HybridDataset(Dataset):
    def __init__(self, X_lc, X_tab, y):
        self.X_lc = X_lc
        self.X_tab = X_tab
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_lc[idx]).float(),
            torch.from_numpy(self.X_tab[idx]).float(),
            torch.tensor(self.y[idx]).long(),
        )


class HybridTemporalTabularCNN(nn.Module):
    def __init__(self, n_channels=18, n_tabular=319, n_classes=32):
        super().__init__()

        self.temporal_branch = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.15),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.20),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.temporal_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(192, n_classes),
        )

    def forward(self, x_lc, x_tab):
        z_lc = self.temporal_branch(x_lc)
        z_lc = self.temporal_pool(z_lc)
        z_lc = self.temporal_head(z_lc)

        z_tab = self.tabular_branch(x_tab)

        z = torch.cat([z_lc, z_tab], dim=1)
        return self.classifier(z)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        temperature = torch.exp(self.log_temperature)
        return logits / temperature


def normalized_entropy(probabilities):
    eps = 1e-12
    k = probabilities.shape[1]
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropy / np.log(k)


def multiclass_brier_score(y_true, probabilities, classes):
    class_to_col = {int(cls): i for i, cls in enumerate(classes)}
    y_onehot = np.zeros_like(probabilities, dtype=float)

    for i, label in enumerate(y_true):
        y_onehot[i, class_to_col[int(label)]] = 1.0

    return np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))


def expected_calibration_error(y_true, probabilities, classes, n_bins=10):
    y_pred = classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)

    ece = 0.0
    rows = []

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        n = int(mask.sum())

        if n == 0:
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

        acc = np.mean(y_true[mask] == y_pred[mask])
        conf = np.mean(confidence[mask])
        gap = abs(acc - conf)

        ece += (n / len(y_true)) * gap

        rows.append({
            "bin": i,
            "lower": lower,
            "upper": upper,
            "n": n,
            "accuracy": acc,
            "confidence": conf,
            "gap": gap,
        })

    return ece, pd.DataFrame(rows)


def evaluate_probabilities(name, y_true, probabilities, classes):
    y_pred = classes[np.argmax(probabilities, axis=1)]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
        "mean_uncertainty": float(normalized_entropy(probabilities).mean()),
        "brier_score": multiclass_brier_score(y_true, probabilities, classes),
    }

    ece, bins = expected_calibration_error(y_true, probabilities, classes)
    metrics["ece"] = ece

    return metrics, bins


def get_logits(model, loader, device):
    model.eval()

    all_logits = []
    all_y = []

    with torch.no_grad():
        for x_lc, x_tab, y in loader:
            x_lc = x_lc.to(device, non_blocking=True)
            x_tab = x_tab.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x_lc, x_tab)

            all_logits.append(logits.float().cpu())
            all_y.append(y)

    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)


def fit_temperature(logits_val, y_val):
    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.05,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_val), y_val)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(torch.exp(scaler.log_temperature).detach().cpu().item())
    return scaler, temperature


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


def minmax_fit(values):
    return np.nanmin(values), np.nanmax(values)


def minmax_transform(values, min_v, max_v):
    if np.isclose(min_v, max_v):
        return np.zeros_like(values, dtype=np.float32)

    out = (values - min_v) / (max_v - min_v)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_global_rarity():
    counts = pd.read_csv(CLASS_COUNTS_PATH)
    counts = counts.sort_values("rarity_rank").copy()

    n_rare = max(1, int(np.ceil(len(counts) * 0.25)))
    rare_classes = counts.head(n_rare)["label"].astype(int).tolist()

    global_counts = dict(
        zip(counts["label"].astype(int), counts["n_objects"].astype(int))
    )

    return global_counts, rare_classes


def compute_global_rarity_weights(global_counts, predicted_labels):
    values = []

    for label in predicted_labels:
        count = global_counts.get(int(label), None)
        values.append(0.0 if count is None else 1.0 / np.log1p(count))

    values = np.asarray(values, dtype=np.float32)
    min_v, max_v = minmax_fit(values)
    return minmax_transform(values, min_v, max_v)


def build_scoring_table(
    split_name,
    object_ids,
    y_true,
    probabilities,
    classes,
    novelty_score,
    raw_novelty,
    global_counts,
    rare_classes,
):
    y_pred = classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    uncertainty = normalized_entropy(probabilities)
    rarity = compute_global_rarity_weights(global_counts, y_pred)

    df = pd.DataFrame({
        "split": split_name,
        "object_id": object_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
        "confidence": confidence,
        "uncertainty_score": uncertainty,
        "raw_novelty": raw_novelty,
        "novelty_score": novelty_score,
        "rarity_score": rarity,
    })

    df["true_is_rare"] = df["true_label"].isin(rare_classes)
    df["predicted_is_rare"] = df["predicted_label"].isin(rare_classes)

    return df


def apply_priority_score(df, name, wu, wn, wr):
    out = df.copy()
    out["configuration"] = name
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

        top = ranking.head(k)
        top_summary = summarize_selection(top)

        random_summaries = []

        for _ in range(N_RANDOM_RUNS):
            sample = ranking.sample(
                n=k,
                replace=False,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            random_summaries.append(summarize_selection(sample))

        random_df = pd.DataFrame(random_summaries)
        random_mean = random_df.mean()

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
        })

    return pd.DataFrame(rows)


def config_summary(name, wu, wn, wr, enrichment):
    weights = {0.05: 0.50, 0.10: 0.35, 0.20: 0.15}

    row = {
        "configuration": name,
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

    print(f"Loading tensor dataset: {TENSOR_PATH}")
    data = np.load(TENSOR_PATH, allow_pickle=True)

    X_lc = data["X_lc"].astype(np.float32)
    X_tab = data["X_tab"].astype(np.float32)
    y = data["y"].astype(np.int64)
    object_ids = data["object_id"].astype(np.int64)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE_FROM_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y[train_idx],
    )

    print("\nSplits:")
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")

    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab[train_idx]).astype(np.float32)
    X_tab_val = scaler.transform(X_tab[val_idx]).astype(np.float32)
    X_tab_test = scaler.transform(X_tab[test_idx]).astype(np.float32)

    X_tab_train = np.nan_to_num(X_tab_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_val = np.nan_to_num(X_tab_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_tab_test = np.nan_to_num(X_tab_test, nan=0.0, posinf=0.0, neginf=0.0)

    val_dataset = HybridDataset(X_lc[val_idx], X_tab_val, y[val_idx])
    test_dataset = HybridDataset(X_lc[test_idx], X_tab_test, y[test_idx])

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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = HybridTemporalTabularCNN(
        n_channels=int(checkpoint["n_channels"]),
        n_tabular=int(checkpoint["n_tabular"]),
        n_classes=int(checkpoint["n_classes"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    classes = np.asarray(sorted(np.unique(y)), dtype=int)

    print("\nGenerating validation/test logits...")
    logits_val, y_val_torch = get_logits(model, val_loader, device)
    logits_test, y_test_torch = get_logits(model, test_loader, device)

    y_val = y_val_torch.numpy()
    y_test = y_test_torch.numpy()

    probs_val_raw = torch.softmax(logits_val, dim=1).numpy()
    probs_test_raw = torch.softmax(logits_test, dim=1).numpy()

    print("\nFitting temperature scaling on validation logits...")
    temperature_model, temperature = fit_temperature(logits_val, y_val_torch)
    print(f"Selected temperature: {temperature:.4f}")

    probs_val_temp = torch.softmax(temperature_model(logits_val), dim=1).detach().numpy()
    probs_test_temp = torch.softmax(temperature_model(logits_test), dim=1).detach().numpy()

    calibration_rows = []

    for name, yy, probs in [
        ("hybrid_raw_val", y_val, probs_val_raw),
        ("hybrid_temp_val", y_val, probs_val_temp),
        ("hybrid_raw_test", y_test, probs_test_raw),
        ("hybrid_temp_test", y_test, probs_test_temp),
    ]:
        metrics, bins = evaluate_probabilities(name, yy, probs, classes)
        calibration_rows.append(metrics)
        bins.to_csv(OUTPUT_DIR / f"{name}_calibration_bins.csv", index=False)

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(OUTPUT_DIR / "hybrid_calibration_summary.csv", index=False)

    print("\nCalibration summary:")
    print(calibration_df)

    print("\nFitting Mahalanobis novelty models on train tabular features...")
    novelty_models = fit_class_mahalanobis_models(X_tab_train, y[train_idx])

    raw_novelty_val = compute_mahalanobis_novelty(X_tab_val, novelty_models)
    raw_novelty_test = compute_mahalanobis_novelty(X_tab_test, novelty_models)

    novelty_min, novelty_max = minmax_fit(raw_novelty_val)

    novelty_val = minmax_transform(raw_novelty_val, novelty_min, novelty_max)
    novelty_test = minmax_transform(raw_novelty_test, novelty_min, novelty_max)

    global_counts, rare_classes = load_global_rarity()

    # Use temperature-scaled probabilities for final policy evaluation.
    val_scoring = build_scoring_table(
        split_name="validation",
        object_ids=object_ids[val_idx],
        y_true=y_val,
        probabilities=probs_val_temp,
        classes=classes,
        novelty_score=novelty_val,
        raw_novelty=raw_novelty_val,
        global_counts=global_counts,
        rare_classes=rare_classes,
    )

    test_scoring = build_scoring_table(
        split_name="test",
        object_ids=object_ids[test_idx],
        y_true=y_test,
        probabilities=probs_test_temp,
        classes=classes,
        novelty_score=novelty_test,
        raw_novelty=raw_novelty_test,
        global_counts=global_counts,
        rare_classes=rare_classes,
    )

    val_scoring.to_csv(OUTPUT_DIR / "hybrid_validation_scoring_table.csv", index=False)
    test_scoring.to_csv(OUTPUT_DIR / "hybrid_test_scoring_table.csv", index=False)

    candidate_configs = [
        ("uncertainty_only", 1.0, 0.0, 0.0),
        ("novelty_only", 0.0, 1.0, 0.0),
        ("rarity_only", 0.0, 0.0, 1.0),
        ("uncertainty_novelty", 0.5, 0.5, 0.0),
        ("uncertainty_rarity", 0.5, 0.0, 0.5),
        ("novelty_rarity", 0.0, 0.5, 0.5),
        ("previous_discovery", 0.1, 0.8, 0.1),
        ("fixed_discovery", 0.4, 0.4, 0.2),
    ]

    print("\nEvaluating candidate policies on validation...")
    validation_rows = []

    for name, wu, wn, wr in candidate_configs:
        ranking = apply_priority_score(val_scoring, name, wu, wn, wr)
        enrichment = compute_enrichment_vs_random(ranking)
        validation_rows.append(config_summary(name, wu, wn, wr, enrichment))

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

    validation_summary.to_csv(OUTPUT_DIR / "hybrid_validation_policy_summary.csv", index=False)

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

    selected.to_csv(OUTPUT_DIR / "hybrid_selected_policies_from_validation.csv", index=False)

    print("\nSelected policies:")
    print(selected)

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
        ("previous_discovery", 0.1, 0.8, 0.1),
        ("fixed_discovery", 0.4, 0.4, 0.2),
        ("rarity_only", 0.0, 0.0, 1.0),
        ("novelty_rarity", 0.0, 0.5, 0.5),
    ]

    seen = set()
    unique_configs = []

    for cfg in final_configs:
        if cfg not in seen:
            unique_configs.append(cfg)
            seen.add(cfg)

    test_rows = []
    test_enrichments = []

    print("\nEvaluating selected policies on final test...")

    for name, wu, wn, wr in unique_configs:
        ranking = apply_priority_score(test_scoring, name, wu, wn, wr)
        enrichment = compute_enrichment_vs_random(ranking)

        enrichment["configuration"] = name
        enrichment["w_uncertainty"] = wu
        enrichment["w_novelty"] = wn
        enrichment["w_rarity"] = wr

        test_rows.append(config_summary(name, wu, wn, wr, enrichment))
        test_enrichments.append(enrichment)

        ranking.to_csv(OUTPUT_DIR / f"hybrid_test_ranking_{name}.csv", index=False)

    test_summary = pd.DataFrame(test_rows)
    test_enrichment = pd.concat(test_enrichments, ignore_index=True)

    test_summary.to_csv(OUTPUT_DIR / "hybrid_test_policy_summary.csv", index=False)
    test_enrichment.to_csv(OUTPUT_DIR / "hybrid_test_policy_enrichment_by_budget.csv", index=False)

    print("\nHybrid test policy summary:")
    print(test_summary.sort_values("weighted_rare_enrichment", ascending=False))

    print("\nHybrid test enrichment by budget:")
    print(test_enrichment.sort_values(["budget_fraction", "rare_enrichment"], ascending=[True, False]))

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'hybrid_calibration_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'hybrid_validation_policy_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'hybrid_selected_policies_from_validation.csv'}")
    print(f"- {OUTPUT_DIR / 'hybrid_test_policy_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'hybrid_test_policy_enrichment_by_budget.csv'}")


if __name__ == "__main__":
    main()