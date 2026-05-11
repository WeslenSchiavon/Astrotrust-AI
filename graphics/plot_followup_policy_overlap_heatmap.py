#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality overlap heatmap for the follow-up policy ablation.

Figure:
  - x-axis: follow-up budget
  - y-axis: policy compared against the final Novelty + Rarity policy
  - color: Jaccard overlap
  - annotations: exact overlap values

This corrected version uses the final Jaccard-overlap values from:
  results/final_publication/followup_policy_ablation/followup_policy_ablation_overlap_vs_reference.csv

If the CSV is not found, it falls back to the final publication values.

Outputs:
  results/final_publication/figures/fig_followup_policy_overlap_heatmap.png
  results/final_publication/figures/fig_followup_policy_overlap_heatmap.pdf
  results/final_publication/figures/fig_followup_policy_overlap_heatmap_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

BUDGET_ORDER = [0.01, 0.02, 0.05, 0.10, 0.20]
BUDGET_LABELS = ["1%", "2%", "5%", "10%", "20%"]

POLICY_ORDER = ["rarity_only", "previous_discovery", "fixed_discovery"]
POLICY_LABELS = {
    "rarity_only": "Rarity only",
    "previous_discovery": "Previous discovery",
    "fixed_discovery": "Fixed discovery",
}

# Final Jaccard-overlap values from the publication summary.
# Rows: rarity_only, previous_discovery, fixed_discovery.
# Columns: 1%, 2%, 5%, 10%, 20% follow-up budget.
FALLBACK_JACCARD_VALUES = np.array([
    [0.912898,   0.815434,   0.847798,  0.915729, 0.922345],
    [0.449239,   0.339988,   0.191480,  0.245852, 0.297334],
    [0.00705467, 0.00928793, 0.0327602, 0.041526, 0.0700422],
], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot follow-up policy overlap heatmap using final Jaccard-overlap values."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT_DIR
        / "results"
        / "final_publication"
        / "followup_policy_ablation"
        / "followup_policy_ablation_overlap_vs_reference.csv",
        help="CSV containing overlap metrics. Default: final_publication follow-up ablation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "final_publication" / "figures",
        help="Directory where the figure and markdown summary will be saved.",
    )
    parser.add_argument(
        "--metric",
        choices=["jaccard_overlap", "overlap_fraction_of_reference"],
        default="jaccard_overlap",
        help="Overlap metric to plot. For the manuscript heatmap, use jaccard_overlap.",
    )
    return parser.parse_args()


def _format_budget_label(value: float) -> str:
    return f"{int(round(value * 100))}%"


def _as_float_budget(series: pd.Series) -> pd.Series:
    """Convert budget values such as 0.05 or '5%' to fractions such as 0.05."""
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(float)
        return values.where(values <= 1, values / 100.0)

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .astype(float)
    )
    return cleaned.where(cleaned <= 1, cleaned / 100.0)


def build_data(input_csv: Path, metric: str):
    """
    Build the heatmap matrix.

    Preferred behavior:
      - Read the final overlap CSV.
      - Use the requested metric column, normally jaccard_overlap.

    Fallback:
      - Use the final hard-coded publication values if the CSV is not available.
    """
    if input_csv.exists():
        df = pd.read_csv(input_csv)

        required = {"budget_fraction", "policy", metric}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df["budget_fraction"] = _as_float_budget(df["budget_fraction"])
        df = df[df["policy"].isin(POLICY_ORDER)]

        if df.empty:
            raise ValueError(
                "No rows found for the compared policies: "
                f"{', '.join(POLICY_ORDER)}"
            )

        pivot = df.pivot_table(
            index="policy",
            columns="budget_fraction",
            values=metric,
            aggfunc="first",
        )

        missing_policies = [p for p in POLICY_ORDER if p not in pivot.index]
        missing_budgets = [b for b in BUDGET_ORDER if b not in pivot.columns]
        if missing_policies or missing_budgets:
            raise ValueError(
                "Input CSV does not contain the expected policies/budgets. "
                f"Missing policies: {missing_policies}; missing budgets: {missing_budgets}"
            )

        values = pivot.loc[POLICY_ORDER, BUDGET_ORDER].to_numpy(dtype=float)
        source = str(input_csv)
    else:
        values = FALLBACK_JACCARD_VALUES.copy()
        source = "fallback final publication values"

    return BUDGET_LABELS, [POLICY_LABELS[p] for p in POLICY_ORDER], values, source


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure(budgets: list[str], policies: list[str], values: np.ndarray, output_dir: Path) -> tuple[Path, Path]:
    configure_style()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "fig_followup_policy_overlap_heatmap.png"
    output_pdf = output_dir / "fig_followup_policy_overlap_heatmap.pdf"

    fig, ax = plt.subplots(figsize=(7.45, 4.55))

    im = ax.imshow(
        values,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels(budgets)
    ax.set_yticks(np.arange(len(policies)))
    ax.set_yticklabels(policies)

    ax.set_xlabel("Follow-up budget")
    ax.set_ylabel("Compared policy")
    ax.set_title("Selection overlap with Novelty + Rarity", pad=10)

    # Annotate each cell.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt_color = "white" if val < 0.55 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8.4,
                color=txt_color,
                fontweight="bold" if i == 0 else "normal",
            )

    # Subtle cell borders.
    ax.set_xticks(np.arange(-0.5, len(budgets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(policies), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Jaccard overlap")

    fig.text(
        0.5,
        0.02,
        "Novelty + Rarity is highly similar to Rarity only, but much less similar to Previous discovery and Fixed discovery.",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_png, bbox_inches="tight")
    #fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return output_png, output_pdf


def write_markdown_summary(
    output_dir: Path,
    budgets: Iterable[str],
    policies: Iterable[str],
    values: np.ndarray,
    source: str,
    metric: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_md = output_dir / "fig_followup_policy_overlap_heatmap_summary.md"

    lines = []
    lines.append("# Follow-up policy overlap heatmap\n")
    lines.append(f"Source: `{source}`\n")
    lines.append(f"Metric plotted: `{metric}`\n")
    lines.append("\n## Heatmap values\n")
    lines.append("| Compared policy | " + " | ".join(budgets) + " |")
    lines.append("|:--|" + "--:|" * len(list(budgets)))
    for policy, row in zip(policies, values):
        lines.append("| " + policy + " | " + " | ".join(f"{v:.6f}" for v in row) + " |")

    lines.append("\n## Interpretation\n")
    lines.append(
        "The final Novelty + Rarity policy has high Jaccard overlap with the Rarity only ranking, "
        "but substantially lower overlap with Previous discovery and Fixed discovery. This supports the "
        "interpretation that the selected policy preserves the rare-candidate prioritization behavior of "
        "Rarity only while adding an explicit novelty component.\n"
    )

    #output_md.write_text("\n".join(lines), encoding="utf-8")
    return output_md


def main() -> None:
    args = parse_args()
    budgets, policies, values, source = build_data(args.input_csv, args.metric)
    output_png, output_pdf = plot_figure(budgets, policies, values, args.output_dir)
    output_md = write_markdown_summary(args.output_dir, budgets, policies, values, source, args.metric)

    print(f"[OK] Figure saved to: {output_png}")
    #print(f"[OK] PDF saved to: {output_pdf}")
    #print(f"[OK] Summary saved to: {output_md}")


if __name__ == "__main__":
    main()
