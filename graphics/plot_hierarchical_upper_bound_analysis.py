#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_data():
    strategies = [
        "Original\ntop-1",
        "Family-mass\nthen subclass",
        "Oracle true family\nthen subclass",
    ]
    fine_accuracy = [0.673, 0.669, 0.785]
    family_accuracy = [0.827, 0.836, 1.000]
    return strategies, fine_accuracy, family_accuracy


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


def add_bar_labels(ax, bars, values, offset=0.012):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(val + offset, ax.get_ylim()[1] - 0.01),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#111111",
            zorder=6,
        )


def plot_figure(strategies, fine_accuracy, family_accuracy) -> Path:
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_hierarchical_upper_bound_analysis.png"

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.8), constrained_layout=True)

    colors = ["#0072B2", "#56B4E9", "#D55E00"]
    edgecolors = ["#0A4C7A", "#2E7FA4", "#8C3B00"]
    x = np.arange(len(strategies))

    # Panel A
    ax = axes[0]
    bars_a = ax.bar(
        x, fine_accuracy,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.9,
        width=0.62,
        zorder=3,
    )
    ax.set_title("A. Fine-grained accuracy", pad=10)
    ax.set_ylabel("Fine accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0.60, 0.83)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)
    add_bar_labels(ax, bars_a, fine_accuracy, offset=0.008)

    ax.text(
        0.02, 0.98,
        "Simple hierarchical reranking\ndoes not improve fine accuracy.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white", ec="#D0D0D0", lw=0.7, alpha=0.96,
        ),
        zorder=7,
    )

    ax.annotate(
        "Family-mass reranking: -0.004",
        xy=(x[1], fine_accuracy[1]),
        xycoords="data",
        xytext=(0.18, 0.09),
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
        "Oracle headroom: +0.112",
        xy=(x[2], fine_accuracy[2]),
        xycoords="data",
        xytext=(0.60, 0.67),
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

    # Panel B
    ax = axes[1]
    bars_b = ax.bar(
        x, family_accuracy,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.9,
        width=0.62,
        zorder=3,
    )
    ax.set_title("B. Family-level accuracy", pad=10)
    ax.set_ylabel("Family accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0.75, 1.04)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)
    add_bar_labels(ax, bars_b, family_accuracy, offset=0.010)

    ax.text(
        0.02, 0.98,
        "The oracle scenario reveals substantial headroom\nif astronomical family modeling is improved.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white", ec="#D0D0D0", lw=0.7, alpha=0.96,
        ),
        zorder=7,
    )

    fig.suptitle("Hierarchical upper-bound analysis", fontsize=12.5, y=1.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    strategies, fine_accuracy, family_accuracy = build_data()
    output_path = plot_figure(strategies, fine_accuracy, family_accuracy)
    print(f"[OK] Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
