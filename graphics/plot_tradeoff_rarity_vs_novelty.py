#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality trade-off figure:
rare-class enrichment vs. novelty for the CLEAN ensemble follow-up ablation.

Official source:
  - Clean ensemble follow-up policy ablation
  - final ensemble_hybrid_dominant probabilities
  - explicit rare-class labels
  - independent robust feature-space novelty score

Figure:
  - x-axis: mean novelty score
  - y-axis: rare-class enrichment
  - each point: one policy at one follow-up budget
  - color/marker: policy
  - marker size: follow-up budget
  - dashed y=1 line: random-enrichment reference

Output:
  results/final_publication/figures/fig_tradeoff_rarity_vs_novelty.png

Run:
  python experiments/plot_tradeoff_rarity_vs_novelty.py
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
        # budget_fraction, policy, rare_enrichment, mean_novelty
        (0.01, "random",                     1.050100, 0.027082),
        (0.02, "random",                     1.099170, 0.028001),
        (0.05, "random",                     1.011550, 0.027820),
        (0.10, "random",                     0.989948, 0.027224),
        (0.20, "random",                     0.984546, 0.028825),

        (0.01, "rarity_only",                5.603810, 0.022995),
        (0.02, "rarity_only",                5.603810, 0.028230),
        (0.05, "rarity_only",                5.603810, 0.035644),
        (0.10, "rarity_only",                5.600860, 0.034025),
        (0.20, "rarity_only",                4.899650, 0.029462),

        (0.01, "novelty_only",               1.030470, 1.000000),
        (0.02, "novelty_only",               1.050100, 0.852241),
        (0.05, "novelty_only",               0.997804, 0.495029),
        (0.10, "novelty_only",               0.909416, 0.277266),
        (0.20, "novelty_only",               0.850000, 0.145159),

        (0.01, "novelty_rarity",             5.348650, 0.481414),
        (0.02, "novelty_rarity",             5.289760, 0.289847),
        (0.05, "novelty_rarity",             5.275790, 0.154284),
        (0.10, "novelty_rarity",             5.128480, 0.132696),
        (0.20, "novelty_rarity",             4.771490, 0.104333),

        (0.01, "uncertainty_novelty_rarity", 4.592970, 0.528159),
        (0.02, "uncertainty_novelty_rarity", 4.946270, 0.295099),
        (0.05, "uncertainty_novelty_rarity", 5.212940, 0.140876),
        (0.10, "uncertainty_novelty_rarity", 5.127500, 0.101653),
        (0.20, "uncertainty_novelty_rarity", 4.776400, 0.084394),
    ]

    df = pd.DataFrame(
        rows,
        columns=["budget_fraction", "policy", "rare_enrichment", "mean_novelty"],
    )
    df["budget_pct"] = (df["budget_fraction"] * 100).round().astype(int)
    return df


def configure_style() -> None:
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


def size_map(budget_pct: int) -> float:
    return {
        1: 55,
        2: 78,
        5: 120,
        10: 175,
        20: 260,
    }[budget_pct]


def plot_figure(df: pd.DataFrame) -> Path:
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_tradeoff_rarity_vs_novelty.png"

    fig, ax = plt.subplots(figsize=(8.25, 5.0))

    style = {
        "rarity_only": {
            "color": "#0072B2", "marker": "s", "label": "Rarity only", "z": 6,
        },
        "novelty_rarity": {
            "color": "#D55E00", "marker": "o", "label": "Novelty + Rarity", "z": 7,
        },
        "uncertainty_novelty_rarity": {
            "color": "#6A3D9A", "marker": "P", "label": "Uncertainty + Novelty + Rarity", "z": 5,
        },
        "novelty_only": {
            "color": "#009E73", "marker": "^", "label": "Novelty only", "z": 4,
        },
        "random": {
            "color": "#7A7A7A", "marker": "D", "label": "Random", "z": 3,
        },
    }

    policy_order = [
        "rarity_only",
        "novelty_rarity",
        "uncertainty_novelty_rarity",
        "novelty_only",
        "random",
    ]

    for policy in policy_order:
        sub = df[df["policy"] == policy].sort_values("budget_fraction").copy()
        st = style[policy]

        ax.scatter(
            sub["mean_novelty"],
            sub["rare_enrichment"],
            s=[size_map(v) for v in sub["budget_pct"]],
            c=st["color"],
            marker=st["marker"],
            edgecolors="white",
            linewidths=0.9,
            alpha=0.93,
            label=st["label"],
            zorder=st["z"],
        )

        ax.plot(
            sub["mean_novelty"],
            sub["rare_enrichment"],
            color=st["color"],
            linewidth=1.45,
            alpha=0.52,
            zorder=st["z"] - 1,
        )

    ax.axhline(
        1.0,
        color="#333333",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.85,
        zorder=1,
    )

    ax.text(
        1.08,
        1.15,
        "random expectation ≈ 1×",
        ha="right",
        va="bottom",
        fontsize=7.9,
        color="#333333",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Mean novelty score")
    ax.set_ylabel("Rare-class enrichment")
    ax.set_title("Trade-off between rare-class enrichment and novelty", pad=10)

    ax.grid(axis="y", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", linewidth=0.35, alpha=0.14, which="both")

    ax.set_xlim(0.018, 1.15)
    ax.set_ylim(0.0, 6.05)

    ax.text(
        0.07,
        0.62,
        "Novelty + Rarity sacrifices modest rare enrichment\n"
        "relative to Rarity only, while selecting substantially\n"
        "more novel candidates.",
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    ax.annotate(
        "5% budget:\n"
        "Novelty + Rarity ≈ 5.28×\n"
        "Rarity only ≈ 5.60×",
        xy=(0.154284, 5.275790),
        xytext=(0.080, 4),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
            zorder=20,
        ),
        bbox=dict(
            boxstyle="round,pad=0.32",
            fc="white",
            ec="#BDBDBD",
            lw=0.7,
            alpha=0.96,
        ),
        fontsize=8.0,
        ha="left",
        va="center",
        zorder=20,
    )

    ax.annotate(
        "Novelty-only maximizes novelty,\nbut does not enrich rare classes.",
        xy=(0.495029, 0.997804),
        xytext=(0.2, 2.05),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
            zorder=20,
        ),
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
        fontsize=7.8,
        ha="left",
        va="center",
        zorder=20,
    )

    policy_legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.005, 0.02),
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
        labelspacing=0.65,
        handletextpad=0.35,
        handleheight=1.25,
        title="Policy",
        title_fontsize=8.3,
    )
    ax.add_artist(policy_legend)

    budget_handles = []
    budget_labels = []
    for pct in [1, 2, 5, 10, 20]:
        handle = ax.scatter(
            [],
            [],
            s=size_map(pct),
            c="#B0B0B0",
            edgecolors="#666666",
            linewidths=0.7,
            alpha=0.90,
        )
        budget_handles.append(handle)
        budget_labels.append(f"{pct}%")

    ax.legend(
        budget_handles,
        budget_labels,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.23),
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
        title="Budget",
        title_fontsize=8.3,
        scatterpoints=1,
        labelspacing=1.15,
        handletextpad=0.8,
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
