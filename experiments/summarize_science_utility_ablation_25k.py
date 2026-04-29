from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT_DIR
    / "results"
    / "v4_priority_weight_search_25k"
    / "final_test_ablation_and_optimized_summary.csv"
)

OUTPUT_DIR = ROOT_DIR / "results" / "v4_science_utility_ablation_25k"

BUDGET_WEIGHTS = {
    0.05: 0.50,
    0.10: 0.35,
    0.20: 0.15,
}


def weighted_average(df: pd.DataFrame, column: str) -> float:
    total = 0.0
    weight_sum = 0.0

    for budget, weight in BUDGET_WEIGHTS.items():
        row = df[df["budget_fraction"] == budget]

        if row.empty:
            continue

        total += weight * float(row.iloc[0][column])
        weight_sum += weight

    if weight_sum == 0:
        return np.nan

    return total / weight_sum


def minmax(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()

    if np.isclose(min_v, max_v):
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - min_v) / (max_v - min_v)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    configs = []

    for config, group in df.groupby("configuration"):
        group = group.sort_values("budget_fraction")

        configs.append({
            "configuration": config,
            "w_uncertainty": float(group.iloc[0]["w_uncertainty"]),
            "w_novelty": float(group.iloc[0]["w_novelty"]),
            "w_rarity": float(group.iloc[0]["w_rarity"]),

            "weighted_rare_enrichment": weighted_average(group, "rare_enrichment"),
            "weighted_top_rare_true_rate": weighted_average(group, "top_rare_true_rate"),
            "weighted_random_rare_true_rate": weighted_average(group, "random_rare_true_rate_mean"),

            "weighted_top_accuracy": weighted_average(group, "top_accuracy"),
            "weighted_top_error_rate": weighted_average(group, "top_error_rate"),

            "weighted_mean_uncertainty": weighted_average(group, "top_mean_uncertainty"),
            "weighted_mean_novelty": weighted_average(group, "top_mean_novelty"),
            "weighted_mean_rarity": weighted_average(group, "top_mean_rarity"),

            "top5_rare_enrichment": float(group[group["budget_fraction"] == 0.05]["rare_enrichment"].iloc[0]),
            "top10_rare_enrichment": float(group[group["budget_fraction"] == 0.10]["rare_enrichment"].iloc[0]),
            "top20_rare_enrichment": float(group[group["budget_fraction"] == 0.20]["rare_enrichment"].iloc[0]),

            "top5_rare_rate": float(group[group["budget_fraction"] == 0.05]["top_rare_true_rate"].iloc[0]),
            "top10_rare_rate": float(group[group["budget_fraction"] == 0.10]["top_rare_true_rate"].iloc[0]),
            "top20_rare_rate": float(group[group["budget_fraction"] == 0.20]["top_rare_true_rate"].iloc[0]),
        })

    summary = pd.DataFrame(configs)

    # Policy 1: rare-class retrieval.
    # This score intentionally rewards known rare-class concentration.
    summary["rare_retrieval_score"] = summary["weighted_rare_enrichment"]

    # Policy 2: discovery-oriented triage.
    # This score rewards a broader scientific behavior:
    # rare candidates + uncertain candidates + novel candidates.
    summary["rare_norm"] = minmax(summary["weighted_rare_enrichment"])
    summary["uncertainty_norm"] = minmax(summary["weighted_mean_uncertainty"])
    summary["novelty_norm"] = minmax(summary["weighted_mean_novelty"])

    summary["discovery_utility_score"] = (
        0.40 * summary["rare_norm"]
        + 0.30 * summary["uncertainty_norm"]
        + 0.30 * summary["novelty_norm"]
    )

    summary = summary.sort_values(
        ["discovery_utility_score", "rare_retrieval_score"],
        ascending=False,
    ).reset_index(drop=True)

    rare_best = summary.sort_values("rare_retrieval_score", ascending=False).iloc[0]
    discovery_best = summary.sort_values("discovery_utility_score", ascending=False).iloc[0]

    output_summary = OUTPUT_DIR / "science_utility_ablation_summary.csv"
    output_selected = OUTPUT_DIR / "selected_science_policies.csv"
    output_md = OUTPUT_DIR / "science_utility_ablation_summary.md"

    summary.to_csv(output_summary, index=False)

    selected = pd.DataFrame([
        {
            "policy": "rare_class_retrieval",
            "selected_configuration": rare_best["configuration"],
            "w_uncertainty": rare_best["w_uncertainty"],
            "w_novelty": rare_best["w_novelty"],
            "w_rarity": rare_best["w_rarity"],
            "weighted_rare_enrichment": rare_best["weighted_rare_enrichment"],
            "weighted_top_rare_true_rate": rare_best["weighted_top_rare_true_rate"],
            "weighted_mean_uncertainty": rare_best["weighted_mean_uncertainty"],
            "weighted_mean_novelty": rare_best["weighted_mean_novelty"],
            "score": rare_best["rare_retrieval_score"],
        },
        {
            "policy": "discovery_oriented_triage",
            "selected_configuration": discovery_best["configuration"],
            "w_uncertainty": discovery_best["w_uncertainty"],
            "w_novelty": discovery_best["w_novelty"],
            "w_rarity": discovery_best["w_rarity"],
            "weighted_rare_enrichment": discovery_best["weighted_rare_enrichment"],
            "weighted_top_rare_true_rate": discovery_best["weighted_top_rare_true_rate"],
            "weighted_mean_uncertainty": discovery_best["weighted_mean_uncertainty"],
            "weighted_mean_novelty": discovery_best["weighted_mean_novelty"],
            "score": discovery_best["discovery_utility_score"],
        },
    ])

    selected.to_csv(output_selected, index=False)

    lines = []
    lines.append("# Science-utility ablation - AstroTrust-AI v4\n")
    lines.append("This analysis separates rare-class retrieval from discovery-oriented triage.\n")

    lines.append("## Selected policies\n")
    lines.append(selected.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Full science utility summary\n")
    display_cols = [
        "configuration",
        "w_uncertainty",
        "w_novelty",
        "w_rarity",
        "weighted_rare_enrichment",
        "weighted_top_rare_true_rate",
        "weighted_mean_uncertainty",
        "weighted_mean_novelty",
        "weighted_mean_rarity",
        "rare_retrieval_score",
        "discovery_utility_score",
        "top5_rare_enrichment",
        "top10_rare_enrichment",
        "top20_rare_enrichment",
    ]

    lines.append(summary[display_cols].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Interpretation\n")
    lines.append(
        "If the objective is purely rare-class retrieval, the best policy is expected to emphasize global rarity. "
        "If the objective is discovery-oriented triage, uncertainty and novelty remain scientifically meaningful because they select candidates that are more ambiguous or unusual, not only candidates predicted as known rare classes.\n"
    )

    output_md.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Saved:")
    print(f"- {output_summary}")
    print(f"- {output_selected}")
    print(f"- {output_md}")

    print("\nSelected science policies:")
    print(selected)

    print("\nScience utility summary:")
    display_cols = [
        "configuration",
        "w_uncertainty",
        "w_novelty",
        "w_rarity",
        "weighted_rare_enrichment",
        "weighted_top_rare_true_rate",
        "weighted_mean_uncertainty",
        "weighted_mean_novelty",
        "weighted_mean_rarity",
        "rare_retrieval_score",
        "discovery_utility_score",
        "top5_rare_enrichment",
        "top10_rare_enrichment",
        "top20_rare_enrichment",
    ]
    print(summary[display_cols])


if __name__ == "__main__":
    main()