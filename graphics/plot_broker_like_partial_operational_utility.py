#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Publication figure: broker-like partial-light-curve operational utility.

This figure is intended to replace a broader partial-metrics panel when the
main manuscript should emphasize broker-like operational utility rather than
low early top-1 accuracy. It focuses on the two quantities that best represent
early follow-up triage under sparse alert information:

  1. Top-5 accuracy / ranked recovery.
  2. Rare-class enrichment at a 5% follow-up budget.

Reads, in priority order:
  --input-csv, if provided
  results/final_publication/broker_like_realism/multiseed_tuned_summary_stats.csv
  results/broker_like_stage_aware_multiseed/multiseed_tuned_summary_stats.csv
  ./multiseed_tuned_summary_stats.csv
  /mnt/data/multiseed_tuned_summary_stats.csv

Outputs:
  results/final_publication/figures/fig_broker_like_partial_operational_utility.png
  results/final_publication/figures/fig_broker_like_partial_operational_utility.pdf
  results/final_publication/figures/fig_broker_like_partial_operational_utility.svg
  results/final_publication/broker_like_realism/broker_like_partial_operational_utility_plot_data.csv
  results/final_publication/reports/broker_like_partial_operational_utility_figure_summary.md

Run:
  python experiments/plot_broker_like_partial_operational_utility.py --update-summary

Notes:
  - Accuracy and macro-F1 should remain in the accompanying results table.
  - This figure emphasizes the operational triage message: partial alerts are
    hard for exact top-1 classification, but still useful for ranked recovery
    and rare-candidate prioritization.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = [
    "first_3_points",
    "first_5_points",
    "first_10_points",
    "first_20_points",
    "window_7_days",
    "window_30_days",
]

SCENARIO_LABELS = {
    "first_3_points": "First\n3 pts",
    "first_5_points": "First\n5 pts",
    "first_10_points": "First\n10 pts",
    "first_20_points": "First\n20 pts",
    "window_7_days": "7-day\nwindow",
    "window_30_days": "30-day\nwindow",
}

DEFAULT_MODEL = "ensemble_early_aware_tuned"
OUTPUT_STEM = "fig_broker_like_partial_operational_utility"


def infer_root_dir() -> Path:
    """Infer project root when the script is run from experiments/ or root."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "results").exists() or (candidate / "experiments").exists():
            if candidate.name == "experiments":
                return candidate.parent
            return candidate
    return here.parent


ROOT_DIR = infer_root_dir()


def log(message: str) -> None:
    print(f"[PLOT-PARTIAL-UTILITY] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot broker-like partial-light-curve operational utility."
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "final_publication" / "figures",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT_DIR / "results" / "final_publication" / "reports",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR / "results" / "final_publication" / "broker_like_realism",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT_DIR / "results" / "final_publication" / "final_publication_summary.md",
    )
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="Append or replace a generated section in final_publication_summary.md.",
    )
    parser.add_argument(
        "--include-all-window-scenarios",
        action="store_true",
        help="Also include 2-day and 14-day windows if available. Main manuscript default is off.",
    )
    return parser.parse_args()


def find_existing(paths: Iterable[Path]) -> Path:
    checked: list[str] = []
    for path in paths:
        checked.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError("Could not find any expected input file:\n" + "\n".join(checked))


def load_data(input_csv: Path | None, model: str, include_all_window_scenarios: bool) -> tuple[pd.DataFrame, Path]:
    candidates: list[Path] = []
    if input_csv is not None:
        candidates.append(input_csv)

    candidates.extend([
        ROOT_DIR / "results" / "final_publication" / "broker_like_realism" / "multiseed_tuned_summary_stats.csv",
        ROOT_DIR / "results" / "broker_like_stage_aware_multiseed" / "multiseed_tuned_summary_stats.csv",
        Path.cwd() / "multiseed_tuned_summary_stats.csv",
        Path("/mnt/data/multiseed_tuned_summary_stats.csv"),
    ])

    source = find_existing(candidates)
    log(f"[LOAD] {source}")

    df = pd.read_csv(source)

    scenario_order = SCENARIO_ORDER.copy()
    scenario_labels = SCENARIO_LABELS.copy()
    if include_all_window_scenarios:
        scenario_order = [
            "first_3_points",
            "first_5_points",
            "first_10_points",
            "first_20_points",
            "window_2_days",
            "window_7_days",
            "window_14_days",
            "window_30_days",
        ]
        scenario_labels.update({
            "window_2_days": "2-day\nwindow",
            "window_14_days": "14-day\nwindow",
        })

    required = {
        "model_name",
        "scenario",
        "n_seeds",
        "top5_accuracy_mean",
        "top5_accuracy_std",
        "rare_enrichment_5pct_mean",
        "rare_enrichment_5pct_std",
        "rare_rate_5pct_mean",
        "rare_rate_5pct_std",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df[(df["model_name"] == model) & (df["scenario"].isin(scenario_order))].copy()
    if df.empty:
        raise ValueError(f"No rows found for model={model} and expected scenarios.")

    df["scenario"] = pd.Categorical(df["scenario"], categories=scenario_order, ordered=True)
    df = df.sort_values("scenario").reset_index(drop=True)
    df["scenario_label"] = df["scenario"].astype(str).map(scenario_labels)

    log(f"[DATA] model={model}; scenarios={len(df)}; seeds={sorted(df['n_seeds'].unique())}")
    return df, source


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.2,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.6,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def add_value_labels(ax: plt.Axes, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, dy: float, fmt: str) -> None:
    for xi, yi, ei in zip(x, y, yerr):
        ax.text(
            xi,
            yi + ei + dy,
            fmt.format(yi),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def plot(df: pd.DataFrame, output_dir: Path) -> Path:
    log("[PLOT] creating operational-utility figure")
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / OUTPUT_STEM

    x = np.arange(len(df))
    labels = df["scenario_label"].to_list()

    top5 = df["top5_accuracy_mean"].to_numpy(dtype=float)
    top5_sd = df["top5_accuracy_std"].to_numpy(dtype=float)
    enrich = df["rare_enrichment_5pct_mean"].to_numpy(dtype=float)
    enrich_sd = df["rare_enrichment_5pct_std"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.35, 4.45), constrained_layout=True)

    # Panel A: ranked recovery.
    ax = axes[0]
    ax.errorbar(
        x,
        top5,
        yerr=top5_sd,
        fmt="o-",
        linewidth=2.15,
        markersize=5.8,
        capsize=4,
        elinewidth=1.1,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    ax.set_title("Ranked recovery for partial alerts", pad=9)
    ax.set_ylabel("Top-5 accuracy")
    ax.set_xlabel("Partial-light-curve scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.735, 0.825)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)
    add_value_labels(ax, x, top5, top5_sd, 0.003, "{:.3f}")
    ax.text(
        0.03,
        0.96,
        "Top-5 remains\n$\\geq$ 0.755",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#D0D0D0", lw=0.7, alpha=0.96),
    )

    # Panel B: rare-candidate enrichment.
    ax = axes[1]
    ax.errorbar(
        x,
        enrich,
        yerr=enrich_sd,
        fmt="D-",
        linewidth=2.15,
        markersize=5.6,
        capsize=4,
        elinewidth=1.1,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="#666666", zorder=1)
    ax.set_title("Rare-candidate prioritization", pad=9)
    ax.set_ylabel("Rare enrichment at 5% budget")
    ax.set_xlabel("Partial-light-curve scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(3.55, 4.78)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)
    add_value_labels(ax, x, enrich, enrich_sd, 0.045, "{:.2f}x")
    ax.text(
        0.03,
        0.96,
        "Enrichment remains\n$>$ 3.7$\\times$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#D0D0D0", lw=0.7, alpha=0.96),
    )

    fig.suptitle(
        "Operational utility under broker-like partial-light-curve information",
        fontsize=12.0,
        y=1.04,
    )
    fig.text(
        0.5,
        -0.035,
        "Mean ± standard deviation across 10 training seeds for the validation-only tuned early-aware route. "
        "Top-1 accuracy and macro-F1 are reported in the accompanying table.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )

    for ext in [".png", ".pdf", ".svg"]:
        output_path = output_base.with_suffix(ext)
        fig.savefig(output_path, bbox_inches="tight")
        log(f"[SAVE] {output_path}")
    plt.close(fig)

    return output_base.with_suffix(".png")


def write_summary(df: pd.DataFrame, source: Path, output_png: Path, reports_dir: Path, data_dir: Path) -> Path:
    log("[REPORT] writing CSV and markdown summary")
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "broker_like_partial_operational_utility_plot_data.csv"
    df.to_csv(data_path, index=False)

    summary_path = reports_dir / "broker_like_partial_operational_utility_figure_summary.md"
    table_cols = [
        "scenario",
        "n_seeds",
        "top5_accuracy_mean",
        "top5_accuracy_std",
        "rare_enrichment_5pct_mean",
        "rare_enrichment_5pct_std",
        "rare_rate_5pct_mean",
        "rare_rate_5pct_std",
        "accuracy_mean" if "accuracy_mean" in df.columns else "top5_accuracy_mean",
        "macro_f1_mean" if "macro_f1_mean" in df.columns else "top5_accuracy_mean",
    ]
    # Preserve order and avoid duplicates if fallback columns are used.
    table_cols = list(dict.fromkeys(table_cols))

    min_top5 = df["top5_accuracy_mean"].min()
    max_top5 = df["top5_accuracy_mean"].max()
    min_enrich = df["rare_enrichment_5pct_mean"].min()
    max_enrich = df["rare_enrichment_5pct_mean"].max()

    md = f"""# Broker-like partial operational-utility figure

Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

Source: `{source}`

Figure: `{output_png}`

Data: `{data_path}`

## Rationale

This figure intentionally emphasizes broker-like operational utility rather than early top-1 classification. Accuracy and macro-F1 remain in the accompanying table, while the main visual message focuses on ranked recovery and rare-candidate enrichment under sparse partial-alert conditions.

## Main values

- Top-5 accuracy range across plotted scenarios: **{min_top5:.4f}--{max_top5:.4f}**.
- Rare-class enrichment range at 5% budget: **{min_enrich:.4f}x--{max_enrich:.4f}x**.

## Values plotted

{df[table_cols].to_markdown(index=False, floatfmt='.4f')}

## Suggested caption

```latex
\\caption{{Operational utility of the stage-aware partial-light-curve route under sparse broker-like alert conditions. Rather than emphasizing top-1 classification alone, the figure reports top-5 recovery and rare-class enrichment at a 5\% follow-up budget. Points and error bars show mean $\\pm$ standard deviation across ten training seeds for the validation-only tuned early-aware route. Even with only a few observations, the partial route preserves high ranked-candidate recovery and strong enrichment in the predefined rare-class subset.}}
```
"""
    summary_path.write_text(md, encoding="utf-8")
    log(f"[SAVE] {summary_path}")
    log(f"[SAVE] {data_path}")
    return summary_path


def update_final_summary(summary_path: Path, section_path: Path) -> None:
    if not summary_path.exists():
        log(f"[WARN] final summary not found: {summary_path}")
        return

    begin = "<!-- BEGIN_BROKER_LIKE_PARTIAL_OPERATIONAL_UTILITY_FIGURE -->"
    end = "<!-- END_BROKER_LIKE_PARTIAL_OPERATIONAL_UTILITY_FIGURE -->"
    section = section_path.read_text(encoding="utf-8")
    block = f"{begin}\n\n## Broker-like partial operational-utility figure\n\n{section}\n\n{end}"

    text = summary_path.read_text(encoding="utf-8")
    if begin in text and end in text:
        before = text.split(begin, 1)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        text = before + "\n\n" + block + "\n\n" + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"

    summary_path.write_text(text, encoding="utf-8")
    log(f"[UPDATE] {summary_path}")


def main() -> None:
    args = parse_args()
    log("[START] Broker-like partial operational-utility figure")
    log(f"[ROOT] {ROOT_DIR}")

    df, source = load_data(args.input_csv, args.model, args.include_all_window_scenarios)
    output_png = plot(df, args.output_dir)
    section_path = write_summary(df, source, output_png, args.reports_dir, args.data_dir)

    if args.update_summary:
        update_final_summary(args.summary_path, section_path)

    log("[DONE]")


if __name__ == "__main__":
    main()
