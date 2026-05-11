#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality rare-class enrichment figure by follow-up budget.

Figure:
  - x-axis: follow-up budget (%)
  - y-axis: rare-class enrichment
  - one line per policy:
      * Novelty + Rarity
      * Rarity only
      * Previous discovery
      * Fixed discovery
  - shaded bands: 95% confidence intervals

Output:
  results/final_publication/figures/fig_rare_class_enrichment_by_budget.png

Run from any location:
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
    """Fixed final values from the publication summary."""
    rows = [
        # budget_pct, policy, enrichment, ci_low, ci_high
        (1,  "novelty_rarity",     5.51548, 5.45462, 5.57411),
        (2,  "novelty_rarity",     5.35333, 5.27840, 5.41685),
        (5,  "novelty_rarity",     5.18544, 5.13436, 5.24008),
        (10, "novelty_rarity",     5.33963, 5.30877, 5.36934),
        (20, "novelty_rarity",     4.74988, 4.71319, 4.78900),

        (1,  "rarity_only",        5.50567, 5.44040, 5.56364),
        (2,  "rarity_only",        5.31404, 5.24094, 5.38421),
        (5,  "rarity_only",        5.21490, 5.16233, 5.26560),
        (10, "rarity_only",        5.34748, 5.31636, 5.38046),
        (20, "rarity_only",        4.75725, 4.72011, 4.79487),

        (1,  "previous_discovery", 4.34761, 4.16328, 4.53076),
        (2,  "previous_discovery", 3.48212, 3.33338, 3.63705),
        (5,  "previous_discovery", 2.22345, 2.13131, 2.32060),
        (10, "previous_discovery", 2.14096, 2.06666, 2.21454),
        (20, "previous_discovery", 2.42871, 2.37851, 2.47926),

        (1,  "fixed_discovery",    0.765494, 0.612580, 0.922786),
        (2,  "fixed_discovery",    0.540245, 0.452677, 0.631447),
        (5,  "fixed_discovery",    0.388908, 0.337892, 0.436691),
        (10, "fixed_discovery",    0.430156, 0.394471, 0.468333),
        (20, "fixed_discovery",    0.672732, 0.639637, 0.705510),
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
        "legend.fontsize": 8.2,
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

    fig, ax = plt.subplots(figsize=(7.9, 4.9))

    policy_order = [
        "novelty_rarity",
        "rarity_only",
        "previous_discovery",
        "fixed_discovery",
    ]

    style = {
        "novelty_rarity": {
            "color": "#D55E00",
            "marker": "o",
            "label": "Novelty + Rarity",
            "lw": 2.3,
            "z": 5,
        },
        "rarity_only": {
            "color": "#0072B2",
            "marker": "s",
            "label": "Rarity only",
            "lw": 2.1,
            "z": 4,
        },
        "previous_discovery": {
            "color": "#009E73",
            "marker": "^",
            "label": "Previous discovery",
            "lw": 1.9,
            "z": 3,
        },
        "fixed_discovery": {
            "color": "#7A7A7A",
            "marker": "D",
            "label": "Fixed discovery",
            "lw": 1.8,
            "z": 2,
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
            alpha=0.16,
            zorder=st["z"] - 1,
        )

        ax.plot(
            sub["budget_pct"],
            sub["rare_enrichment"],
            color=st["color"],
            marker=st["marker"],
            markersize=6.2,
            linewidth=st["lw"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=st["label"],
            zorder=st["z"],
        )

    ax.set_xlabel("Follow-up budget (%)")
    ax.set_ylabel("Rare-class enrichment")
    ax.set_title("Rare-class enrichment by follow-up budget", pad=20)

    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1%", "2%", "5%", "10%", "20%"])
    ax.set_xlim(0.6, 20.7)
    ax.set_ylim(0.0, 5.9)

    ax.grid(axis="y", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", linewidth=0.35, alpha=0.14)

    ax.legend(
        loc="lower left",
        # O primeiro valor (0.02) é a distância da esquerda (x)
        # O segundo valor (0.15) empurra a legenda para cima (y)
        bbox_to_anchor=(0.02, 0.13), 
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
        fontsize=8.2
    )

    # Random-selection baseline. Rare enrichment is expected to be approximately 1x
    # under a random follow-up selection.
    ax.axhline(
        1.0,
        color="#333333",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        alpha=0.85,
        zorder=2,
    )


    ax.text(
        0.015,
        1.02,
        "Novelty + rarity remains close to rarity only,\n"
        "while explicitly incorporating novelty.",
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
        1.07,
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



    # Highlight the central 5% result.
    x_anno = 5
    y_anno = 5.18544
    ax.annotate(
        "At 5% budget:\nnovelty + rarity ≈ 5.19×",
        xy=(x_anno, y_anno),
        xytext=(8.2, 4.85),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
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
