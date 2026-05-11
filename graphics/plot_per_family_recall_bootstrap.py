#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality per-family top-1 recall figure with 95% CI.

Figure:
  - horizontal bars
  - y-axis: astronomical families
  - x-axis: top-1 recall by family
  - bars sorted from highest to lowest recall
  - error bars: 95% bootstrap confidence intervals

Output:
  results/final_publication/figures/fig_per_family_recall_bootstrap.png

Run from any location:
  python experiments/plot_per_family_recall_bootstrap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_dataframe() -> pd.DataFrame:
    """Fixed final values from the publication summary."""
    rows = [
        {"family": "AGN",   "recall": 0.994880, "ci_low": 0.991807, "ci_high": 0.997952, "support": 1953},
        {"family": "Other", "recall": 0.974829, "ci_low": 0.971839, "ci_high": 0.977477, "support": 11720},
        {"family": "CV",    "recall": 0.973252, "ci_low": 0.966756, "ci_high": 0.979366, "support": 2617},
        {"family": "KN",    "recall": 0.955696, "ci_low": 0.945570, "ci_high": 0.965823, "support": 1580},
        {"family": "uLens", "recall": 0.948525, "ci_low": 0.942549, "ci_high": 0.954502, "support": 5187},
        {"family": "TDE",   "recall": 0.921659, "ci_low": 0.908845, "ci_high": 0.933436, "support": 1953},
        {"family": "SLSN",  "recall": 0.885174, "ci_low": 0.873547, "ci_high": 0.896802, "support": 2752},
        {"family": "SNIa",  "recall": 0.799112, "ci_low": 0.788018, "ci_high": 0.809182, "support": 5859},
        {"family": "SNII",  "recall": 0.704838, "ci_low": 0.696988, "ci_high": 0.713288, "support": 11719},
        {"family": "SNIbc", "recall": 0.676088, "ci_low": 0.667071, "ci_high": 0.686129, "support": 9765},
        {"family": "CART",  "recall": 0.665643, "ci_low": 0.643113, "ci_high": 0.686124, "support": 1953},
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("recall", ascending=False).reset_index(drop=True)


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
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure(df: pd.DataFrame) -> Path:
    """Create and save the per-family recall figure."""
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_per_family_recall_bootstrap.png"

    fig, ax = plt.subplots(figsize=(8.3, 5.2))

    # Reverse so the highest value appears at the top in barh.
    df_plot = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df_plot))

    bar_color = "#0072B2"
    edge_color = "#0A4C7A"

    lower_err = df_plot["recall"] - df_plot["ci_low"]
    upper_err = df_plot["ci_high"] - df_plot["recall"]

    bars = ax.barh(
        y,
        df_plot["recall"],
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.9,
        xerr=[lower_err, upper_err],
        ecolor="#222222",
        capsize=4,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["family"])
    ax.set_xlabel("Top-1 recall")
    ax.set_ylabel("Astronomical family")
    ax.set_title("Per-family top-1 recall with 95% bootstrap confidence intervals", pad=10)

    ax.set_xlim(0.60, 1.005)
    ax.grid(axis="x", linewidth=0.5, alpha=0.28, zorder=1)

    # Main interpretation note
    ax.text(
        0.015,
        0.98,
        "Higher-recall families are more reliable,\nwhile lower-recall families remain more challenging",
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

   
    # Add value labels and support
    for bar, row in zip(bars, df_plot.itertuples(index=False)):
        # Calcula a extremidade direita da barra de erro para posicionar o texto após ela
        right_edge_of_error = row.ci_high if hasattr(row, 'ci_high') else row.recall
        
        ax.text(
            right_edge_of_error + 0.002, # Desloca 0.008 para a direita do fim da barra de erro
            bar.get_y() + bar.get_height() / 2.0,
            f"{row.recall:.3f}  (n={row.support})",
            ha="left",
            va="center",
            fontsize=7.8,
            color="#111111",
        )

    # Lightly emphasize the strongest and weakest families.
    best_row = df.iloc[0]
    worst_row = df.iloc[-1]
    ax.text(
        0.95,
        0.98,
        f"Highest: {best_row.family} ≈ {best_row.recall:.3f}\nLowest: {worst_row.family} ≈ {worst_row.recall:.3f}",
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
