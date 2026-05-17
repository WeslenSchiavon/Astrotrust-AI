#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a publication-quality overlap heatmap for the CLEAN ensemble follow-up policy ablation.

Official source:
  - Clean ensemble follow-up policy ablation
  - reference policy: novelty_rarity
  - metric: Jaccard overlap
  - final ensemble_hybrid_dominant probabilities
  - independent robust feature-space novelty score

Figure:
  - x-axis: follow-up budget
  - y-axis: policy compared against Novelty + Rarity
  - color: Jaccard overlap
  - annotations: exact overlap values

Outputs:
  results/final_publication/figures/fig_followup_policy_overlap_heatmap.png
  results/final_publication/figures/fig_followup_policy_overlap_heatmap_summary.md

Run:
  python experiments/plot_followup_policy_overlap_heatmap.py

Optional:
  python experiments/plot_followup_policy_overlap_heatmap.py `
    --input-csv results/final_publication/clean_ensemble_followup_policy_ablation/clean_followup_policy_overlap_vs_reference.csv
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

POLICY_ORDER = [
    "rarity_only",
    "uncertainty_novelty_rarity",
    "novelty_only",
    "random",
]

POLICY_LABELS = {
    "rarity_only": "Rarity only",
    "uncertainty_novelty_rarity": "Uncertainty + \nNovelty + Rarity",
    "novelty_only": "Novelty only",
    "random": "Random",
}

# Official clean ensemble Jaccard-overlap values from:
# clean_followup_policy_ablation_summary.md / clean_followup_policy_overlap_vs_reference.csv
# Rows: rarity_only, uncertainty_novelty_rarity, novelty_only, random
# Columns: 1%, 2%, 5%, 10%, 20%.
FALLBACK_JACCARD_VALUES = np.array([
    [0.0251346, 0.0728041, 0.379927, 0.743355, 0.821548],
    [0.380895,  0.228618,  0.258769, 0.434930, 0.910438],
    [0.116325,  0.134625,  0.124113, 0.127000, 0.145381],
    [0.00263389, 0.0137594, 0.0264436, 0.053059, 0.109146],
], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clean follow-up policy overlap heatmap using Jaccard overlap."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing clean overlap metrics. Expected columns include "
            "budget_fraction, policy, and jaccard_overlap. If omitted, the script tries "
            "known final-publication paths and then falls back to official hard-coded values."
        ),
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


def candidate_input_paths(user_path: Path | None) -> list[Path]:
    paths = []
    if user_path is not None:
        paths.append(user_path)

    paths.extend([
        ROOT_DIR
        / "results"
        / "final_publication"
        / "clean_ensemble_followup_policy_ablation"
        / "clean_followup_policy_overlap_vs_reference.csv",
        ROOT_DIR
        / "results"
        / "ensemble_followup_policy_ablation_clean_250k_final"
        / "clean_followup_policy_overlap_vs_reference.csv",
    ])
    return paths


def _as_float_budget(series: pd.Series) -> pd.Series:
    """Convert budget values such as 0.05, 5, or '5%' to fractions such as 0.05."""
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


def read_clean_overlap_csv(input_csv: Path, metric: str) -> tuple[np.ndarray, str]:
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
            "No rows found for the expected clean policies: "
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
    return values, str(input_csv)


def build_data(input_csv: Path | None, metric: str):
    """
    Build the heatmap matrix.

    Preferred:
      - read clean overlap CSV from final_publication or results directory.

    Fallback:
      - use official hard-coded clean ensemble values.
    """
    for path in candidate_input_paths(input_csv):
        if path.exists():
            try:
                values, source = read_clean_overlap_csv(path, metric)
                return BUDGET_LABELS, [POLICY_LABELS[p] for p in POLICY_ORDER], values, source
            except Exception as exc:
                if input_csv is not None and path == input_csv:
                    raise
                print(f"[WARN] Could not use {path}: {exc}")

    values = FALLBACK_JACCARD_VALUES.copy()
    source = "fallback official clean ensemble values"
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
        "ytick.labelsize": 8.7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_figure(
    budgets: list[str],
    policies: list[str],
    values: np.ndarray,
    output_dir: Path,
) -> Path:
    configure_style()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "fig_followup_policy_overlap_heatmap.png"

    fig, ax = plt.subplots(figsize=(8.15, 4.95))

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

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt_color = "white" if val < 0.52 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8.3,
                color=txt_color,
                fontweight="bold" if val >= 0.35 else "normal",
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
        0.022,
        "The clean Novelty + Rarity ranking differs strongly from Rarity only at small budgets, "
        "but converges at larger budgets; overlap with Random and Novelty only remains low.",
        ha="center",
        va="bottom",
        fontsize=7.9,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    return output_png


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

    budgets = list(budgets)
    policies = list(policies)

    lines = []
    lines.append("# Clean follow-up policy overlap heatmap\n")
    lines.append(f"Source: `{source}`\n")
    lines.append(f"Metric plotted: `{metric}`\n")
    lines.append("Reference policy: `novelty_rarity`\n")
    lines.append("\n## Heatmap values\n")
    lines.append("| Compared policy | " + " | ".join(budgets) + " |")
    lines.append("|:--|" + "--:|" * len(budgets))

    for policy, row in zip(policies, values):
        lines.append("| " + policy + " | " + " | ".join(f"{v:.6f}" for v in row) + " |")

    lines.append("\n## Interpretation\n")
    lines.append(
        "The clean Novelty + Rarity policy has low Jaccard overlap with Rarity only at small budgets, "
        "showing that the novelty term substantially changes the highest-priority candidate set. "
        "At larger budgets, the overlap with Rarity only increases because both policies include a broader "
        "set of high-rarity candidates. Overlap with Random and Novelty only remains low, indicating that "
        "Novelty + Rarity is not equivalent to either random selection or novelty-only exploration.\n"
    )

    output_md.write_text("\n".join(lines), encoding="utf-8")
    return output_md


def main() -> None:
    args = parse_args()
    budgets, policies, values, source = build_data(args.input_csv, args.metric)
    output_png = plot_figure(budgets, policies, values, args.output_dir)
    output_md = write_markdown_summary(args.output_dir, budgets, policies, values, source, args.metric)

    print(f"[OK] Figure saved to: {output_png}")
    print(f"[OK] Summary saved to: {output_md}")
    print(f"[INFO] Source: {source}")


if __name__ == "__main__":
    main()
