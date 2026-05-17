#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality rare-class enrichment figure by follow-up budget.

Official source:
  - Clean ensemble follow-up policy ablation
  - final ensemble_hybrid_dominant probabilities
  - explicit rare-class labels
  - independent robust feature-space novelty score

Figure:
  - x-axis: follow-up budget (%)
  - y-axis: rare-class enrichment
  - clean policies only:
      * Rarity only
      * Novelty + Rarity
      * Uncertainty + Novelty + Rarity
      * Novelty only
      * Random
  - shaded bands: 95% bootstrap confidence intervals

Output:
  results/final_publication/figures/fig_rare_class_enrichment_by_budget.png

Run:
  python experiments/plot_rare_class_enrichment_by_budget.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_dataframe() -> pd.DataFrame:
    """Official clean ensemble follow-up values from the final publication package."""
    rows = [
        # budget_pct, policy, enrichment, ci_low, ci_high
        (1,  "random",                     1.050100, 0.873449, 1.246380),
        (2,  "random",                     1.099170, 0.961775, 1.217060),
        (5,  "random",                     1.011550, 0.932986, 1.086190),
        (10, "random",                     0.989948, 0.938854, 1.046910),
        (20, "random",                     0.984546, 0.941801, 1.020400),

        (1,  "rarity_only",                5.603810, 5.603810, 5.603810),
        (2,  "rarity_only",                5.603810, 5.603810, 5.603810),
        (5,  "rarity_only",                5.603810, 5.603810, 5.603810),
        (10, "rarity_only",                5.600860, 5.597920, 5.603810),
        (20, "rarity_only",                4.899650, 4.825990, 4.978230),

        (1,  "novelty_only",               1.030470, 0.853821, 1.216940),
        (2,  "novelty_only",               1.050100, 0.927426, 1.187500),
        (5,  "novelty_only",               0.997804, 0.917273, 1.074460),
        (10, "novelty_only",               0.909416, 0.857365, 0.966402),
        (20, "novelty_only",               0.850000, 0.814141, 0.886853),

        (1,  "novelty_rarity",             5.348650, 5.250510, 5.436970),
        (2,  "novelty_rarity",             5.289760, 5.201440, 5.358460),
        (5,  "novelty_rarity",             5.275790, 5.220800, 5.320970),
        (10, "novelty_rarity",             5.128480, 5.087210, 5.168740),
        (20, "novelty_rarity",             4.771490, 4.697790, 4.831400),

        (1,  "uncertainty_novelty_rarity", 4.592970, 4.386870, 4.779430),
        (2,  "uncertainty_novelty_rarity", 4.946270, 4.838320, 5.054230),
        (5,  "uncertainty_novelty_rarity", 5.212940, 5.155980, 5.267940),
        (10, "uncertainty_novelty_rarity", 5.127500, 5.086250, 5.170710),
        (20, "uncertainty_novelty_rarity", 4.776400, 4.698800, 4.808330),
    ]

    return pd.DataFrame(
        rows,
        columns=["budget_pct", "policy", "rare_enrichment", "ci_low", "ci_high"],
    )


def configure_style() -> None:
    """Journal-friendly figure style."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure(df: pd.DataFrame) -> Path:
    """Create and save the rare-class enrichment figure."""
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_rare_class_enrichment_by_budget.png"

    fig, ax = plt.subplots(figsize=(8.15, 4.95))

    policy_order = [
        "rarity_only",
        "novelty_rarity",
        "uncertainty_novelty_rarity",
        "novelty_only",
        "random",
    ]

    style = {
        "rarity_only": {
            "color": "#0072B2",
            "marker": "s",
            "label": "Rarity only",
            "lw": 2.2,
            "z": 6,
            "ls": "-",
        },
        "novelty_rarity": {
            "color": "#D55E00",
            "marker": "o",
            "label": "Novelty + Rarity",
            "lw": 2.3,
            "z": 7,
            "ls": "-",
        },
        "uncertainty_novelty_rarity": {
            "color": "#6A3D9A",
            "marker": "P",
            "label": "Uncertainty + Novelty + Rarity",
            "lw": 1.9,
            "z": 5,
            "ls": "-",
        },
        "novelty_only": {
            "color": "#009E73",
            "marker": "^",
            "label": "Novelty only",
            "lw": 1.8,
            "z": 4,
            "ls": "-",
        },
        "random": {
            "color": "#7A7A7A",
            "marker": "D",
            "label": "Random",
            "lw": 1.6,
            "z": 3,
            "ls": "--",
        },
    }

    for policy in policy_order:
        sub = df[df["policy"] == policy].sort_values("budget_pct")
        st = style[policy]

        ax.fill_between(
            sub["budget_pct"],
            sub["ci_low"],
            sub["ci_high"],
            color=st["color"],
            alpha=0.12 if policy != "random" else 0.09,
            zorder=st["z"] - 1,
        )

        ax.plot(
            sub["budget_pct"],
            sub["rare_enrichment"],
            color=st["color"],
            marker=st["marker"],
            markersize=6.0,
            linewidth=st["lw"],
            linestyle=st["ls"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=st["label"],
            zorder=st["z"],
        )

    ax.axhline(
        1.0,
        color="#333333",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        alpha=0.85,
        zorder=2,
    )

    ax.set_xlabel("Follow-up budget (%)")
    ax.set_ylabel("Rare-class enrichment")
    ax.set_title("Rare-class enrichment by follow-up budget", pad=20)

    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%"])
    ax.set_xlim(0.6, 20.7)
    ax.set_ylim(0.0, 6.05)

    ax.grid(axis="y", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", linewidth=0.35, alpha=0.14)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.20),
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
        fontsize=7.9,
    )

    ax.text(
        0.015,
        1.02,
        "Clean ensemble follow-up ablation:\n"
        "novelty + rarity keeps strong rare-class enrichment.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    ax.text(
        20.55,
        1.15,
        "random expectation ≈ 1×",
        ha="right",
        va="bottom",
        fontsize=7.9,
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.22",
            fc="white",
            ec="#D0D0D0",
            lw=0.6,
            alpha=0.94,
        ),
    )

    ax.text(
        0.985,
        1.02,
        "Shaded bands: 95% bootstrap CI\n"
        "Values above 1× outperform random selection",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    ax.annotate(
        "At 5% budget:\n"
        "novelty + rarity ≈ 5.28×\n"
        "rarity only ≈ 5.60×",
        xy=(5, 5.275790),
        xytext=(7.8, 4.55),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
            zorder=12,
        ),
        bbox=dict(
            boxstyle="round,pad=0.32",
            fc="white",
            ec="#BDBDBD",
            lw=0.7,
            alpha=0.96,
        ),
        fontsize=8.1,
        ha="left",
        va="center",
        zorder=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    df = build_dataframe()
    output_path = plot_figure(df)
    print(f"[OK] Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
