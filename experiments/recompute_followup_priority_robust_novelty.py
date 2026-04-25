from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_RANKING_PATH = ROOT_DIR / Path(
    "results/followup_prioritization_v3_calibrated_hgb/followup_priority_ranking.csv"
)

RESULTS_DIR = ROOT_DIR / Path("results/followup_prioritization_v3_calibrated_hgb_robust_novelty")

BUDGETS = [0.05, 0.10, 0.20]
N_RANDOM_RUNS = 1000
RANDOM_STATE = 42

# Testaremos algumas combinações de pesos.
WEIGHT_CONFIGS = {
    "balanced": {
        "w_uncertainty": 0.40,
        "w_novelty": 0.40,
        "w_rarity": 0.20,
    },
    "uncertainty_focused": {
        "w_uncertainty": 0.50,
        "w_novelty": 0.30,
        "w_rarity": 0.20,
    },
    "novelty_focused": {
        "w_uncertainty": 0.30,
        "w_novelty": 0.50,
        "w_rarity": 0.20,
    },
    "rare_focused": {
        "w_uncertainty": 0.35,
        "w_novelty": 0.35,
        "w_rarity": 0.30,
    },
}


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_v = np.min(values)
    max_v = np.max(values)

    if np.isclose(min_v, max_v):
        return np.zeros_like(values)

    return (values - min_v) / (max_v - min_v)


def robust_novelty_scale(raw_novelty: np.ndarray) -> np.ndarray:
    """
    Robust novelty scaling:
    1. guarantees non-negative values;
    2. applies log1p to reduce extreme tails;
    3. clips between percentiles 1 and 99;
    4. applies min-max scaling.
    """
    x = np.asarray(raw_novelty, dtype=float)

    # Mahalanobis distances should be non-negative, but this keeps the function safe.
    x = np.maximum(x, 0.0)

    x = np.log1p(x)

    lower = np.percentile(x, 1)
    upper = np.percentile(x, 99)

    x = np.clip(x, lower, upper)

    return minmax_scale(x)


def priority_category(score: float) -> str:
    if score >= 0.75:
        return "very_high"
    if score >= 0.50:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


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


def compute_enrichment_vs_random(ranking: pd.DataFrame, config_name: str) -> pd.DataFrame:
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
            "config": config_name,
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


def recompute_priority(base_ranking: pd.DataFrame, config_name: str, weights: dict) -> pd.DataFrame:
    ranking = base_ranking.copy()

    ranking["novelty_score"] = robust_novelty_scale(ranking["raw_novelty"].to_numpy())

    ranking["priority_score"] = (
        weights["w_uncertainty"] * ranking["uncertainty_score"]
        + weights["w_novelty"] * ranking["novelty_score"]
        + weights["w_rarity"] * ranking["rarity_score"]
    )

    ranking["priority_category"] = ranking["priority_score"].apply(priority_category)

    ranking = ranking.sort_values("priority_score", ascending=False).reset_index(drop=True)
    ranking["priority_rank"] = np.arange(1, len(ranking) + 1)
    ranking["priority_config"] = config_name

    return ranking


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading ranking from: {INPUT_RANKING_PATH}")
    base_ranking = pd.read_csv(INPUT_RANKING_PATH)

    required_columns = [
        "raw_novelty",
        "uncertainty_score",
        "rarity_score",
        "correct",
        "true_is_rare",
    ]

    missing = [col for col in required_columns if col not in base_ranking.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_enrichment = []

    for config_name, weights in WEIGHT_CONFIGS.items():
        print("\n" + "=" * 80)
        print(f"Recomputing priority with config: {config_name}")
        print(weights)

        ranking = recompute_priority(base_ranking, config_name, weights)

        output_ranking_path = RESULTS_DIR / f"followup_priority_ranking_{config_name}.csv"
        ranking.to_csv(output_ranking_path, index=False)

        print("\nTop 20 candidates:")
        display_cols = [
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
        print(ranking[display_cols].head(20))

        enrichment = compute_enrichment_vs_random(ranking, config_name)
        all_enrichment.append(enrichment)

        print("\nEnrichment summary:")
        print(enrichment)

    all_enrichment_df = pd.concat(all_enrichment, ignore_index=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("\n" + "=" * 80)
    print("All configurations summary:")
    print(
        all_enrichment_df[
            [
                "config",
                "budget_fraction",
                "n_selected",
                "top_rare_true_rate",
                "random_rare_true_rate_mean",
                "rare_enrichment",
                "top_accuracy",
                "random_accuracy_mean",
                "top_mean_uncertainty",
                "random_mean_uncertainty",
                "top_mean_novelty",
                "random_mean_novelty",
                "top_mean_priority",
                "random_mean_priority",
            ]
        ].sort_values(["budget_fraction", "rare_enrichment"], ascending=[True, False])
    )

    output_summary_path = RESULTS_DIR / "robust_novelty_priority_comparison.csv"
    all_enrichment_df.to_csv(output_summary_path, index=False)

    print(f"\nSaved summary: {output_summary_path}")


if __name__ == "__main__":
    main()