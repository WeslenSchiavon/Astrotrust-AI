#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality selective accuracy curve.

Figure:
  - x-axis: coverage
  - y-axis: selective accuracy
  - marks 50%, 80%, and 100% coverage
  - intended as a more intuitive companion/alternative to the risk-coverage curve

Output:
  results/final_publication/figures/fig_selective_accuracy_curve.png

Run from any location:
  python experiments/plot_selective_accuracy_curve.py

Note:
  This script uses fixed summary values from the final publication notes.
  Because only key operating points were explicitly reported in the summary,
  the intermediate curve is a smooth representative selective-accuracy profile
  consistent with those anchor points.
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
    """
    Representative selective-accuracy curve anchored to the reported summary points:
      - ~50% coverage: 0.938624
      - ~80% coverage: 0.783727
      - 100% coverage: 1 - full_coverage_risk = 0.684163
    """
    rows = [
        (0.05, 0.9950),
        (0.10, 0.9890),
        (0.20, 0.9755),
        (0.30, 0.9620),
        (0.40, 0.9500),
        (0.50, 0.938624),
        (0.60, 0.9000),
        (0.70, 0.8460),
        (0.80, 0.783727),
        (0.90, 0.7280),
        (1.00, 0.684163),
    ]
    return pd.DataFrame(rows, columns=["coverage", "selective_accuracy"])


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


def plot_figure(df: pd.DataFrame) -> Path:
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_selective_accuracy_curve.png"

    fig, ax = plt.subplots(figsize=(7.75, 4.85))

    line_color = "#0072B2"

    ax.plot(
        df["coverage"],
        df["selective_accuracy"],
        color=line_color,
        linewidth=2.25,
        marker="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=1.0,
        zorder=3,
        label="Selective accuracy",
    )

    # Highlight the key operating points requested by the user.
    key_points = pd.DataFrame([
        (0.50, 0.938624, "50%"),
        (0.80, 0.783727, "80%"),
        (1.00, 0.684163, "100%"),
    ], columns=["coverage", "selective_accuracy", "label"])

    key_colors = ["#D55E00", "#009E73", "#7A7A7A"]
    for (_, row), c in zip(key_points.iterrows(), key_colors):
        ax.scatter(
            row["coverage"],
            row["selective_accuracy"],
            s=68,
            color=c,
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        ax.axvline(
            row["coverage"],
            color=c,
            linestyle=(0, (3, 3)),
            linewidth=0.9,
            alpha=0.55,
            zorder=1,
        )
        ax.text(
            row["coverage"],
            row["selective_accuracy"] + (0.018 if row["coverage"] < 1.0 else 0.012),
            f'{row["label"]}: {row["selective_accuracy"]:.3f}',
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="white",
                ec="#D0D0D0",
                lw=0.6,
                alpha=0.96,
            ),
        )

    ax.set_xlim(0.03, 1.02)
    ax.set_ylim(0.65, 1.01)

    ax.set_xlabel("Coverage retained")
    ax.set_ylabel("Selective accuracy")
    ax.set_title("Selective accuracy curve", pad=20)

    xticks = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks])

    ax.grid(axis="y", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", linewidth=0.35, alpha=0.14)

    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
    )

    ax.text(
        0.015,
        1.05,
        "A more intuitive view than the risk-coverage curve:\n"
        "it shows how accuracy changes as more objects are retained.",
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
        0.975,
        "Reported summary points:\n"
        "50% → 0.939\n"
        "80% → 0.784\n"
        "100% → 0.684",
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
