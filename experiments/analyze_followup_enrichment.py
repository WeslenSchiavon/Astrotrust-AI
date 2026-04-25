from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

RANKING_PATH = ROOT_DIR /  Path("results/followup_prioritization/followup_priority_ranking.csv")
RESULTS_DIR =  ROOT_DIR / Path("results/followup_prioritization")

BUDGETS = [0.05, 0.10, 0.20]
N_RANDOM_RUNS = 1000
RANDOM_STATE = 42


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(RANKING_PATH)
    rng = np.random.default_rng(RANDOM_STATE)

    global_summary = summarize_selection(ranking)

    print("\nGlobal test-set summary:")
    for k, v in global_summary.items():
        print(f"{k}: {v:.4f}")

    rows = []

    for budget in BUDGETS:
        n = len(ranking)
        k = max(1, int(np.ceil(n * budget)))

        top_selected = ranking.sort_values("priority_score", ascending=False).head(k)
        top_summary = summarize_selection(top_selected)

        random_summaries = []

        for _ in range(N_RANDOM_RUNS):
            random_selected = ranking.sample(n=k, replace=False, random_state=int(rng.integers(0, 1_000_000_000)))
            random_summaries.append(summarize_selection(random_selected))

        random_df = pd.DataFrame(random_summaries)
        random_mean = random_df.mean()
        random_std = random_df.std()

        row = {
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
        }

        rows.append(row)

    enrichment = pd.DataFrame(rows)

    print("\nFollow-up enrichment summary:")
    print(enrichment)

    output_path = RESULTS_DIR / "followup_enrichment_vs_random.csv"
    enrichment.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()