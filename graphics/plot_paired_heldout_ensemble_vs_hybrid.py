#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality paired held-out comparison figure:
ensemble vs. hybrid.

Figure:
  - x-axis: Accuracy, Macro-F1, Weighted-F1, Balanced accuracy
  - y-axis: delta_ensemble_minus_hybrid
  - bars: held-out gain of ensemble over hybrid
  - error bars: 95% bootstrap confidence intervals
  - horizontal reference line at zero

Output:
  results/final_publication/figures/fig_paired_heldout_ensemble_vs_hybrid.png

Run from any location:
  python experiments/plot_paired_heldout_ensemble_vs_hybrid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_dataframe() -> pd.DataFrame:
    """Fixed final values from the paired held-out prediction comparison."""
    rows = [
        {
            "metric": "Accuracy",
            "delta": 0.0111641,
            "ci_low": 0.00851458,
            "ci_high": 0.0136006,
        },
        {
            "metric": "Macro-F1",
            "delta": 0.00809182,
            "ci_low": 0.00493472,
            "ci_high": 0.0110950,
        },
        {
            "metric": "Weighted-F1",
            "delta": 0.00871472,
            "ci_low": 0.00597360,
            "ci_high": 0.0114660,
        },
        {
            "metric": "Balanced\naccuracy",
            "delta": 0.00841391,
            "ci_low": 0.00538186,
            "ci_high": 0.0112256,
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
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure(df: pd.DataFrame) -> Path:
    """Create and save the paired held-out comparison figure."""
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_paired_heldout_ensemble_vs_hybrid.png"

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    x = range(len(df))
    bar_color = "#0B5FA5"
    edge_color = "#083B66"

    lower_err = df["delta"] - df["ci_low"]
    upper_err = df["ci_high"] - df["delta"]

    bars = ax.bar(
        x,
        df["delta"],
        width=0.68,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.9,
        yerr=[lower_err, upper_err],
        ecolor="#222222",
        capsize=4,
        zorder=3,
    )

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["metric"])
    ax.set_ylabel("Ensemble gain over hybrid")
    ax.set_xlabel("Metric")
    ax.set_title("Paired held-out comparison: ensemble vs. hybrid", pad=10)

    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)

    ymax = max(df["ci_high"]) + 0.0036
    ax.set_ylim(-0.0008, ymax)

    # Subtle explanatory note inside the plot.
    ax.text(
        0.015,
        0.98,
        "Bars show $\\Delta$ = ensemble $-$ hybrid\nError bars: 95% bootstrap CI",
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    # McNemar significance annotation.
    ax.text(
        0.985,
        0.98,
        "Accuracy gain = +0.0112\nMcNemar exact $p = 6.36\\times10^{-16}$\nHeld-out objects = 57,058",
        transform=ax.transAxes,
        fontsize=8.0,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    # Annotate bar heights.
    for bar, delta in zip(bars, df["delta"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.00035,
            f"+{delta:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.1,
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
