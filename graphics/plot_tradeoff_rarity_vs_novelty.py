#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality trade-off figure:
rarity vs. novelty.

Figure:
  - x-axis: mean novelty
  - y-axis: rare enrichment
  - each point: one policy at one follow-up budget
  - color: policy
  - marker size: follow-up budget
  - dashed y=1 line: random-enrichment reference

Output:
  results/final_publication/figures/fig_tradeoff_rarity_vs_novelty.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_dataframe() -> pd.DataFrame:
    rows = [
        (0.01, "novelty_rarity",     5.51548,  0.00391174),
        (0.02, "novelty_rarity",     5.35333,  0.00204289),
        (0.05, "novelty_rarity",     5.18544,  0.00103484),
        (0.10, "novelty_rarity",     5.33963,  0.000623659),
        (0.20, "novelty_rarity",     4.74988,  0.000417280),

        (0.01, "rarity_only",        5.50567,  0.000247397),
        (0.02, "rarity_only",        5.31404,  0.000207447),
        (0.05, "rarity_only",        5.21490,  0.000175030),
        (0.10, "rarity_only",        5.34748,  0.000180667),
        (0.20, "rarity_only",        4.75725,  0.000256028),

        (0.01, "previous_discovery", 4.34761,  0.00494020),
        (0.02, "previous_discovery", 3.48212,  0.00257993),
        (0.05, "previous_discovery", 2.22345,  0.00118535),
        (0.10, "previous_discovery", 2.14096,  0.000697452),
        (0.20, "previous_discovery", 2.42871,  0.000436393),

        (0.01, "fixed_discovery",    0.765494, 0.00318325),
        (0.02, "fixed_discovery",    0.540245, 0.00170083),
        (0.05, "fixed_discovery",    0.388908, 0.000802187),
        (0.10, "fixed_discovery",    0.430156, 0.000491217),
        (0.20, "fixed_discovery",    0.672732, 0.000372344),
    ]
    df = pd.DataFrame(
        rows,
        columns=["budget_fraction", "policy", "rare_enrichment", "mean_novelty"],
    )
    df["budget_pct"] = (df["budget_fraction"] * 100).astype(int)
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
        "legend.fontsize": 8.2,
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

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    style = {
        "novelty_rarity": {
            "color": "#D55E00", "marker": "o", "label": "Novelty + Rarity", "z": 6,
        },
        "rarity_only": {
            "color": "#0072B2", "marker": "s", "label": "Rarity only", "z": 5,
        },
        "previous_discovery": {
            "color": "#009E73", "marker": "^", "label": "Previous discovery", "z": 4,
        },
        "fixed_discovery": {
            "color": "#7A7A7A", "marker": "D", "label": "Fixed discovery", "z": 3,
        },
    }

    policy_order = [
        "novelty_rarity",
        "rarity_only",
        "previous_discovery",
        "fixed_discovery",
    ]

    for policy in policy_order:
        sub = df[df["policy"] == policy].copy()
        st = style[policy]

        ax.scatter(
            sub["mean_novelty"],
            sub["rare_enrichment"],
            s=[size_map(v) for v in sub["budget_pct"]],
            c=st["color"],
            marker=st["marker"],
            edgecolors="white",
            linewidths=0.9,
            alpha=0.92,
            label=st["label"],
            zorder=st["z"],
        )

        ax.plot(
            sub["mean_novelty"],
            sub["rare_enrichment"],
            color=st["color"],
            linewidth=1.4,
            alpha=0.55,
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
        0.00495,
        1.06,
        "random expectation ≈ 1×",
        ha="right",
        va="bottom",
        fontsize=7.9,
        color="#333333",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Mean novelty score")
    ax.set_ylabel("Rare-class enrichment")
    ax.set_title("Trade-off between rarity and novelty", pad=10)

    ax.grid(axis="y", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", linewidth=0.35, alpha=0.14, which="both")

    ax.set_xlim(1.4e-4, 6.2e-3)
    ax.set_ylim(0.0, 5.95)

    # Move the main note into an empty region of the plot.
    ax.text(
        0.75e-3,
        1.90,
        "Novelty + Rarity sacrifices very little rare enrichment\n"
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
        "Preferred trade-off:\n"
        "higher novelty with nearly unchanged rare enrichment",
        xy=(0.00103484, 5.12544),
        xytext=(0.0008, 4.35),
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
        fontsize=8.0,
        ha="left",
        va="center",
    )

    policy_legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.005, 0.01), # Sobe a legenda em relação à base
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
        # --- AJUSTES PARA MELHORAR O ESPAÇAMENTO DOS SÍMBOLOS ---
        labelspacing=0.7,          # Aumenta o espaço vertical entre as linhas da legenda
        handletextpad=0.2,         # Aumenta o espaço entre o símbolo (triângulo) e o texto
        handleheight=1.5,          # Dá mais altura para o símbolo não tocar na linha de cima/baixo
        # -------------------------------------------------------
        title="Policy",
        title_fontsize=8.3,
    )
    ax.add_artist(policy_legend)

    budget_handles = []
    budget_labels = []
    for pct in [1, 2, 5, 10, 20]:
        handle = ax.scatter([], [], s=size_map(pct), c="#B0B0B0",
                            edgecolors="#666666", linewidths=0.7, alpha=0.90)
        budget_handles.append(handle)
        budget_labels.append(f"{pct}%")

    ax.legend(
        budget_handles,
        budget_labels,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.2), # Sobe a legenda para o mesmo nível da Policy
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
        title="Budget",
        title_fontsize=8.3,
        scatterpoints=1,
        labelspacing=1.2,            # Aumenta o espaço para acomodar os círculos maiores
        handletextpad=0.8            # Mais espaço entre o círculo e o texto (%)
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
