#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality single-panel multi-seed stability figure.

Figure:
  - x-axis: models
  - y-axis: mean score across 10 independent random seeds
  - two metrics shown in the same panel:
      * Accuracy
      * Macro-F1
  - point: mean
  - error bar: standard deviation across seeds

Output:
  results/final_publication/figures/fig_multiseed_stability_single_panel.png

Run from any location:
  python experiments/plot_multiseed_stability_single_panel.py

Suggested caption note:
  Points and error bars show mean ± standard deviation across ten independent
  random seeds.

Interpretation note:
  This figure measures stability across repeated training runs, whereas the
  paired held-out comparison figure measures paired performance differences on
  the same held-out test set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_dataframe() -> pd.DataFrame:
    """Fixed final values from the completed multi-seed stability analysis."""
    rows = [
        {
            "model": "Ensemble hybrid\ndominant",
            "accuracy_mean": 0.6815,
            "accuracy_std": 0.0018,
            "macro_f1_mean": 0.6735,
            "macro_f1_std": 0.0023,
        },
        {
            "model": "Hybrid temporal-\ntabular CNN",
            "accuracy_mean": 0.6698,
            "accuracy_std": 0.0029,
            "macro_f1_mean": 0.6643,
            "macro_f1_std": 0.0030,
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
        "legend.fontsize": 8.3,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure(df: pd.DataFrame) -> Path:
    """Create and save the single-panel multi-seed stability figure."""
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_multiseed_stability_single_panel.png"

    fig, ax = plt.subplots(figsize=(6.1, 3.6))

    x = [0, 0.5]
    offset = 0.10

    # Metric encodings
    acc_color = "#D55E00"
    f1_color = "#0072B2"

    # Accuracy series
    ax.errorbar(
        [v - offset for v in x],
        df["accuracy_mean"],
        yerr=df["accuracy_std"],
        fmt="o",
        color=acc_color,
        ecolor="#222222",
        elinewidth=1.15,
        capsize=4,
        capthick=1.15,
        markersize=7.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Accuracy",
        zorder=3,
    )

    # Macro-F1 series
    ax.errorbar(
        [v + offset for v in x],
        df["macro_f1_mean"],
        yerr=df["macro_f1_std"],
        fmt="s",
        color=f1_color,
        ecolor="#222222",
        elinewidth=1.15,
        capsize=4,
        capthick=1.15,
        markersize=7.4,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Macro-F1",
        zorder=3,
    )

    # Value annotations
    for i, row in df.iterrows():
        ax.text(
            x[i] - offset,
            row["accuracy_mean"] + row["accuracy_std"] + 0.0008,
            f'{row["accuracy_mean"]:.4f} ± {row["accuracy_std"]:.4f}',
            ha="center",
            va="bottom",
            fontsize=7.9,
            color="#222222",
        )
        ax.text(
            x[i] + offset,
            row["macro_f1_mean"] + row["macro_f1_std"] + 0.0008,
            f'{row["macro_f1_mean"]:.4f} ± {row["macro_f1_std"]:.4f}',
            ha="center",
            va="bottom",
            fontsize=7.9,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_xlim(x[0] - 0.25, x[1] + 0.25)
    ax.set_ylabel("Mean test score")
    ax.set_xlabel("Model")
    ax.set_title("Multi-seed stability", pad=10)

    # Tight but readable y-range
    y_min = min(
        (df["accuracy_mean"] - df["accuracy_std"]).min(),
        (df["macro_f1_mean"] - df["macro_f1_std"]).min(),
    )
    y_max = max(
        (df["accuracy_mean"] + df["accuracy_std"]).max(),
        (df["macro_f1_mean"] + df["macro_f1_std"]).max(),
    )
    pad = max(0.0035, (y_max - y_min) * 0.42)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
    )

    # Main message annotation
    ax.text(
        0.985,
        0.97,
        "10/10 matched seeds:\nensemble > hybrid",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    # Clarify what the error bars represent
    ax.text(
        0.015,
        0.97,
        "Points and error bars show mean ± SD\nacross 10 independent random seeds",
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
