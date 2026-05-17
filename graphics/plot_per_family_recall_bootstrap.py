#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality per-family top-1 recall figure with 95% bootstrap CIs.

Official source:
  - ensemble_hybrid_dominant
  - Top-k/family bootstrap analysis
  - Metric: family_recall_top1_from_fine_top1
  - Definition: top-1 fine prediction mapped to astronomical family.

Figure:
  - horizontal bars
  - y-axis: astronomical families
  - x-axis: top-1 recall by family
  - bars sorted from highest to lowest recall
  - error bars: 95% bootstrap confidence intervals

Output:
  results/final_publication/figures/fig_per_family_recall_bootstrap.png

Run:
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
    """Official ensemble values from the final top-k/family bootstrap analysis."""
    rows = [
        {"family": "AGN",   "recall": 0.996416, "ci_low": 0.993344, "ci_high": 0.998976, "support": 1953},
        {"family": "CV",    "recall": 0.984333, "ci_low": 0.979366, "ci_high": 0.988919, "support": 2617},
        {"family": "Other", "recall": 0.974147, "ci_low": 0.971329, "ci_high": 0.976962, "support": 11720},
        {"family": "KN",    "recall": 0.961392, "ci_low": 0.951899, "ci_high": 0.970886, "support": 1580},
        {"family": "uLens", "recall": 0.950646, "ci_low": 0.944862, "ci_high": 0.956434, "support": 5187},
        {"family": "TDE",   "recall": 0.929339, "ci_low": 0.917563, "ci_high": 0.939580, "support": 1953},
        {"family": "SLSN",  "recall": 0.899346, "ci_low": 0.888808, "ci_high": 0.910247, "support": 2752},
        {"family": "SNIa",  "recall": 0.808670, "ci_low": 0.798430, "ci_high": 0.818740, "support": 5859},
        {"family": "SNII",  "recall": 0.703388, "ci_low": 0.694684, "ci_high": 0.712010, "support": 11719},
        {"family": "SNIbc", "recall": 0.689401, "ci_low": 0.680387, "ci_high": 0.698927, "support": 9765},
        {"family": "CART",  "recall": 0.667179, "ci_low": 0.645673, "ci_high": 0.687673, "support": 1953},
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

    fig, ax = plt.subplots(figsize=(8.45, 5.25))

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
    ax.set_xlabel("Top-1 family recall")
    ax.set_ylabel("Astronomical family")
    ax.set_title("Per-family top-1 recall with 95% bootstrap confidence intervals", pad=10)

    ax.set_xlim(0.60, 1.005)
    ax.grid(axis="x", linewidth=0.5, alpha=0.28, zorder=1)

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

    for bar, row in zip(bars, df_plot.itertuples(index=False)):
        right_edge_of_error = row.ci_high

        ax.text(
            right_edge_of_error + 0.002,
            bar.get_y() + bar.get_height() / 2.0,
            f"{row.recall:.3f}  (n={row.support})",
            ha="left",
            va="center",
            fontsize=7.8,
            color="#111111",
        )

    best_row = df.iloc[0]
    worst_row = df.iloc[-1]
    ax.text(
        0.95,
        0.98,
        f"Highest: {best_row.family} ≈ {best_row.recall:.3f}\n"
        f"Lowest: {worst_row.family} ≈ {worst_row.recall:.3f}",
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
