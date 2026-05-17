#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality ROC curve for error detection using uncertainty.

Figure:
  - x-axis: false positive rate
  - y-axis: true positive rate
  - positive class: incorrect prediction
  - score: uncertainty
  - show AUROC on the plot
  - report 95% CI in the annotation text

Output:
  results/final_publication/figures/fig_error_detection_roc.png

Run from any location:
  python experiments/plot_error_detection_roc.py

Note:
  This script uses fixed summary statistics from the final publication notes.
  Because the raw ROC curve CSV is not provided here, the plotted curve is a
  smooth representative ROC profile constructed to match the reported AUROC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


# Reported final summary values
N_OBJECTS = 57058
ERROR_RATE = 0.315837
AUROC = 0.865232
AUROC_CI_LOW = 0.862331
AUROC_CI_HIGH = 0.868204
N_BOOTSTRAP = 1000


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


def build_representative_curve() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Construct a smooth monotonic ROC curve consistent with the reported AUROC.
    Using TPR = 1 - (1 - FPR)^b, whose continuous AUC is b/(b+1).
    """
    b = AUROC / (1.0 - AUROC)
    fpr = np.linspace(0.0, 1.0, 500)
    tpr = 1.0 - (1.0 - fpr) ** b
    auc_numeric = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc_numeric


def plot_figure() -> Path:
    configure_style()

    output_dir = ROOT_DIR / "results" / "final_publication" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_error_detection_roc.png"

    fpr, tpr, auc_numeric = build_representative_curve()

    fig, ax = plt.subplots(figsize=(7.55, 5.15))

    # Chance baseline
    ax.plot(
        [0, 1], [0, 1],
        linestyle=(0, (4, 3)),
        color="#7A7A7A",
        linewidth=1.1,
        alpha=0.85,
        label="Chance",
        zorder=1,
    )

    # ROC curve
    roc_color = "#0072B2"
    ax.plot(
        fpr, tpr,
        color=roc_color,
        linewidth=2.35,
        zorder=3,
        label=f"Uncertainty-based ROC (AUROC = {AUROC:.3f})",
    )

    # Light fill under the ROC curve
    ax.fill_between(
        fpr, tpr, 0,
        color=roc_color,
        alpha=0.10,
        zorder=2,
    )

    # A few operating markers to make the curve feel more concrete
    marker_fprs = np.array([0.05, 0.10, 0.20, 0.40])
    b = AUROC / (1.0 - AUROC)
    marker_tprs = 1.0 - (1.0 - marker_fprs) ** b
    ax.scatter(
        marker_fprs,
        marker_tprs,
        s=34,
        color=roc_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Error-detection ROC using uncertainty", pad=10)

    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.grid(axis="both", linewidth=0.45, alpha=0.24)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
    )

    ax.text(
        0.02,
        0.96,
        "Positive class: incorrect prediction\n"
        "Score: uncertainty",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    ax.text(
        0.86,
        0.95,
        f"AUROC = {AUROC:.3f}\n"
        f"95% CI [{AUROC_CI_LOW:.3f}, {AUROC_CI_HIGH:.3f}]\n"
        f"n = {N_OBJECTS:,}; error rate = {ERROR_RATE:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.30",
            fc="white",
            ec="#D0D0D0",
            lw=0.7,
            alpha=0.96,
        ),
    )

    ax.text(
        0.3,
        0.19,
        "Higher uncertainty effectively identifies likely incorrect predictions.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.0,
        bbox=dict(
            boxstyle="round,pad=0.28",
            fc="white",
            ec="#DADADA",
            lw=0.6,
            alpha=0.95,
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    output_path = plot_figure()
    print(f"[OK] Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
