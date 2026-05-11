#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality grouped bar chart for:

Top-k fine vs. family-level performance with 95% bootstrap confidence intervals.

Figure:
  - x-axis: Top-1, Top-2, Top-3, Top-5
  - y-axis: Accuracy
  - two bars per group:
      * fine-grained labels
      * astronomical families
  - error bars: 95% bootstrap confidence intervals

Output:
  results/final_publication/figures/fig_topk_fine_vs_family_bootstrap.png

Run from any location:
  python experiments/plot_topk_fine_vs_family_bootstrap.py
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
        {
            "k_label": "Top-1",
            "fine": 0.673420,
            "fine_ci_low": 0.669474,
            "fine_ci_high": 0.677207,
            "family": 0.831172,
            "family_ci_low": 0.828069,
            "family_ci_high": 0.834065,
        },
        {
            "k_label": "Top-2",
            "fine": 0.818290,
            "fine_ci_low": 0.815433,
            "fine_ci_high": 0.821534,
            "family": 0.916646,
            "family_ci_low": 0.914350,
            "family_ci_high": 0.918960,
        },
        {
            "k_label": "Top-3",
            "fine": 0.886501,
            "fine_ci_low": 0.883907,
            "fine_ci_high": 0.889043,
            "family": 0.955239,
            "family_ci_low": 0.953591,
            "family_ci_high": 0.956886,
        },
        {
            "k_label": "Top-5",
            "fine": 0.947019,
            "fine_ci_low": 0.945266,
            "fine_ci_high": 0.948842,
            "family": 0.986260,
            "family_ci_low": 0.985295,
            "family_ci_high": 0.987206,
        },
    ]
    return pd.DataFrame(rows)


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
        "legend.fontsize": 8.4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure(df: pd.DataFrame) -> Path:
    """Create and save the grouped top-k figure."""
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_topk_fine_vs_family_bootstrap.png"

    fig, ax = plt.subplots(figsize=(7.45, 4.7))

    x = np.arange(len(df))
    width = 0.34

    fine_color = "#0072B2"
    family_color = "#D55E00"
    fine_edge = "#0A4C7A"
    family_edge = "#8C3B00"

    fine_lower = df["fine"] - df["fine_ci_low"]
    fine_upper = df["fine_ci_high"] - df["fine"]
    family_lower = df["family"] - df["family_ci_low"]
    family_upper = df["family_ci_high"] - df["family"]

    fine_bars = ax.bar(
        x - width / 2,
        df["fine"],
        width=width,
        color=fine_color,
        edgecolor=fine_edge,
        linewidth=0.9,
        label="Fine-grained labels",
        yerr=[fine_lower, fine_upper],
        ecolor="#222222",
        capsize=4,
        zorder=3,
    )

    family_bars = ax.bar(
        x + width / 2,
        df["family"],
        width=width,
        color=family_color,
        edgecolor=family_edge,
        linewidth=0.9,
        label="Astronomical families",
        yerr=[family_lower, family_upper],
        ecolor="#222222",
        capsize=4,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["k_label"])
    ax.set_xlabel("Prediction cutoff")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-k fine-grained vs. family-level performance", pad=40)

    ax.set_ylim(0.62, 1.005)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
    )

    ax.text(
        0.015,
        0.96,
        "Top-1 fine performance is moderate,\nwhile top-k and family-level accuracy are very strong",
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
        0.985,
        1.13,
        "Top-5:\nfine ≈ 0.9470\nfamily ≈ 0.9863",
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

    for bars, values in [(fine_bars, df["fine"]), (family_bars, df["family"])]:
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.0065,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color="#111111",
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
