#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality hierarchical upper-bound figure.

Official source:
  - final ensemble_hybrid_dominant hierarchical upper-bound analysis
  - original_top1
  - family_mass_then_subclass
  - oracle_true_family_then_subclass

Figure:
  - Panel A: fine-grained accuracy
  - Panel B: family-level accuracy
  - Error bars: 95% bootstrap confidence intervals

Output:
  results/final_publication/figures/fig_hierarchical_upper_bound_analysis.png

Run:
  python experiments/plot_hierarchical_upper_bound_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_data():
    """Official ensemble hierarchical upper-bound values."""
    strategies = [
        "Original\ntop-1",
        "Family-mass\nthen subclass",
        "Oracle true family\nthen subclass",
    ]

    fine_accuracy = np.array([0.684163, 0.679677, 0.788864])
    fine_ci_low = np.array([0.680290, 0.675627, 0.785499])
    fine_ci_high = np.array([0.687671, 0.683603, 0.792143])

    family_accuracy = np.array([0.835904, 0.845753, 1.000000])
    family_ci_low = np.array([0.832959, 0.842598, 1.000000])
    family_ci_high = np.array([0.838901, 0.848506, 1.000000])

    deltas = {
        "family_mass_fine_delta": -0.00448666,
        "family_mass_family_delta": 0.00984963,
        "oracle_fine_headroom": 0.104700,
        "oracle_family_headroom": 0.164096,
    }

    return (
        strategies,
        fine_accuracy,
        fine_ci_low,
        fine_ci_high,
        family_accuracy,
        family_ci_low,
        family_ci_high,
        deltas,
    )


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.8,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_bar_labels(ax, bars, values, offset=0.010):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(val + offset, ax.get_ylim()[1] - 0.008),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#111111",
            zorder=6,
        )


def make_yerr(values, ci_low, ci_high):
    lower = values - ci_low
    upper = ci_high - values
    return np.vstack([lower, upper])


def plot_figure(
    strategies,
    fine_accuracy,
    fine_ci_low,
    fine_ci_high,
    family_accuracy,
    family_ci_low,
    family_ci_high,
    deltas,
) -> Path:
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_hierarchical_upper_bound_analysis.png"

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.8), constrained_layout=True)

    colors = ["#0072B2", "#56B4E9", "#D55E00"]
    edgecolors = ["#0A4C7A", "#2E7FA4", "#8C3B00"]
    x = np.arange(len(strategies))

    # -------------------------
    # Panel A: fine accuracy
    # -------------------------
    ax = axes[0]
    bars_a = ax.bar(
        x,
        fine_accuracy,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.9,
        width=0.62,
        yerr=make_yerr(fine_accuracy, fine_ci_low, fine_ci_high),
        ecolor="#222222",
        capsize=4,
        zorder=3,
    )

    ax.set_title("A. Fine-grained accuracy", pad=10)
    ax.set_ylabel("Fine accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0.64, 0.815)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)
    add_bar_labels(ax, bars_a, fine_accuracy, offset=0.007)

    ax.text(
        0.02,
        0.98,
        "Simple hierarchical reranking\ndoes not improve fine accuracy.\nError bars: 95% bootstrap CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
        zorder=7,
    )

    ax.annotate(
        f"Family-mass reranking: {deltas['family_mass_fine_delta']:+.4f}",
        xy=(x[1], fine_accuracy[1]),
        xycoords="data",
        xytext=(0.17, 0.17),
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
            connectionstyle="arc3,rad=-0.10",
        ),
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white",
            ec="#CFCFCF",
            lw=0.7,
            alpha=0.97,
        ),
        fontsize=7.9,
        ha="left",
        va="center",
        zorder=8,
    )

    ax.annotate(
        f"Oracle headroom: +{deltas['oracle_fine_headroom']:.4f}",
        xy=(x[2], fine_accuracy[2]),
        xycoords="data",
        xytext=(0.58, 0.64),
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
            connectionstyle="arc3,rad=0.10",
        ),
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white",
            ec="#CFCFCF",
            lw=0.7,
            alpha=0.97,
        ),
        fontsize=7.9,
        ha="left",
        va="center",
        zorder=8,
    )

    # -------------------------
    # Panel B: family accuracy
    # -------------------------
    ax = axes[1]
    bars_b = ax.bar(
        x,
        family_accuracy,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.9,
        width=0.62,
        yerr=make_yerr(family_accuracy, family_ci_low, family_ci_high),
        ecolor="#222222",
        capsize=4,
        zorder=3,
    )

    ax.set_title("B. Family-level accuracy", pad=10)
    ax.set_ylabel("Family accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0.80, 1.035)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)
    add_bar_labels(ax, bars_b, family_accuracy, offset=0.009)

    ax.text(
        0.02,
        0.98,
        "Family-mass reranking improves family accuracy,\nwhile the oracle reveals remaining headroom.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
        zorder=7,
    )

    ax.annotate(
        f"Family-mass gain: +{deltas['family_mass_family_delta']:.4f}",
        xy=(x[1], family_accuracy[1]),
        xycoords="data",
        xytext=(0.40, 0.12),
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
            connectionstyle="arc3,rad=0.10",
        ),
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white",
            ec="#CFCFCF",
            lw=0.7,
            alpha=0.97,
        ),
        fontsize=7.9,
        ha="left",
        va="center",
        zorder=8,
    )

    fig.suptitle("Hierarchical upper-bound analysis", fontsize=12.5, y=1.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    (
        strategies,
        fine_accuracy,
        fine_ci_low,
        fine_ci_high,
        family_accuracy,
        family_ci_low,
        family_ci_high,
        deltas,
    ) = build_data()

    output_path = plot_figure(
        strategies,
        fine_accuracy,
        fine_ci_low,
        fine_ci_high,
        family_accuracy,
        family_ci_low,
        family_ci_high,
        deltas,
    )
    print(f"[OK] Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
