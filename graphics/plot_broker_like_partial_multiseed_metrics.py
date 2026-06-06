#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Publication figure: broker-like partial-light-curve multiseed metrics.

Reads:
  results/final_publication/broker_like_realism/multiseed_tuned_summary_stats.csv
or:
  results/broker_like_stage_aware_multiseed/multiseed_tuned_summary_stats.csv

Outputs:
  results/final_publication/figures/fig_broker_like_partial_multiseed_metrics.png
  results/final_publication/figures/fig_broker_like_partial_multiseed_metrics.pdf
  results/final_publication/figures/fig_broker_like_partial_multiseed_metrics.svg
  results/final_publication/broker_like_realism/broker_like_partial_multiseed_metrics_plot_data.csv
  results/final_publication/reports/broker_like_partial_multiseed_metrics_figure_summary.md

Run:
  python experiments/plot_broker_like_partial_multiseed_metrics.py

This is the main figure for the broker-like partial-light-curve evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

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

METRIC_SPECS = [
    ("accuracy", "Accuracy", 0.30, 0.47),
    ("macro_f1", "Macro-F1", 0.28, 0.44),
    ("top5_accuracy", "Top-5 accuracy", 0.72, 0.82),
    ("rare_enrichment_5pct", "Rare enrichment at 5% budget", 3.55, 4.75),
]


def log(message: str) -> None:
    print(f"[PLOT-PARTIAL-METRICS] {message}", flush=True)


def find_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find any expected input file:\n" + "\n".join(str(p) for p in paths))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "figures")
    p.add_argument("--reports-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "reports")
    p.add_argument("--data-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "broker_like_realism")
    p.add_argument("--model", type=str, default="ensemble_early_aware_tuned")
    p.add_argument("--update-summary", action="store_true")
    return p.parse_args()


def load_data(input_csv: Path | None, model: str) -> tuple[pd.DataFrame, Path]:
    candidates = []
    if input_csv is not None:
        candidates.append(input_csv)

    candidates.extend([
        ROOT_DIR / "results" / "final_publication" / "broker_like_realism" / "multiseed_tuned_summary_stats.csv",
        ROOT_DIR / "results" / "broker_like_stage_aware_multiseed" / "multiseed_tuned_summary_stats.csv",
        Path.cwd() / "multiseed_tuned_summary_stats.csv",
        Path("/mnt/data/multiseed_tuned_summary_stats.csv"),
    ])

    path = find_existing(candidates)
    log(f"[LOAD] {path}")

    df = pd.read_csv(path)
    df = df[(df["model_name"] == model) & (df["scenario"].isin(SCENARIO_ORDER))].copy()
    if df.empty:
        raise ValueError(f"No rows found for model={model}")

    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df = df.sort_values("scenario").reset_index(drop=True)
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)

    return df, path


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.4,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.4,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.6,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def plot(df: pd.DataFrame, output_dir: Path) -> Path:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "fig_broker_like_partial_multiseed_metrics"

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 5.9), constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(len(df))

    for ax, (metric, label, ymin, ymax) in zip(axes, METRIC_SPECS):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        ax.errorbar(
            x,
            df[mean_col],
            yerr=df[std_col],
            fmt="o-",
            linewidth=2.0,
            markersize=5.4,
            capsize=4,
            elinewidth=1.05,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

        ax.set_title(label, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["scenario_label"])
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)

        for xi, mean, sd in zip(x, df[mean_col], df[std_col]):
            ax.text(
                xi,
                mean + sd + (ymax - ymin) * 0.025,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
            )

    fig.suptitle(
        "Broker-like partial-light-curve performance across 10 training seeds",
        fontsize=12.2,
        y=1.03,
    )

    fig.text(
        0.5,
        -0.025,
        "Points and error bars show mean ± standard deviation across 10 training seeds. "
        "The complete-light-curve route remains the original ensemble_hybrid_dominant model.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )

    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(output_base.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)

    return output_base.with_suffix(".png")


def write_summary(df: pd.DataFrame, source: Path, output_png: Path, reports_dir: Path, data_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "broker_like_partial_multiseed_metrics_plot_data.csv"
    df.to_csv(data_path, index=False)

    summary_path = reports_dir / "broker_like_partial_multiseed_metrics_figure_summary.md"

    table_cols = [
        "scenario", "n_seeds",
        "accuracy_mean", "accuracy_std",
        "macro_f1_mean", "macro_f1_std",
        "top5_accuracy_mean", "top5_accuracy_std",
        "rare_enrichment_5pct_mean", "rare_enrichment_5pct_std",
        "ece_mean", "ece_std",
    ]

    md = f"""# Broker-like partial-light-curve multiseed metrics figure

Generated at: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`

Source: `{source}`

Figure: `{output_png}`

Data: `{data_path}`

## Values plotted

{df[table_cols].to_markdown(index=False, floatfmt=".4f")}

## Suggested caption

```latex
\\caption{{Broker-like partial-light-curve performance of the stage-aware AstroTrust-AI route. Each point reports the validation-only tuned early-aware ensemble evaluated across ten training seeds, with error bars denoting mean $\\pm$ standard deviation across seeds. The scenarios simulate early broker decisions using either the first $N$ observations or fixed windows after the first alert. Despite the expected degradation relative to complete light curves, the partial route preserves high top-5 recovery and strong rare-class enrichment under a 5\\% follow-up budget.}}
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

    begin = "<!-- BEGIN_BROKER_LIKE_PARTIAL_MULTISEED_FIGURE -->"
    end = "<!-- END_BROKER_LIKE_PARTIAL_MULTISEED_FIGURE -->"
    section = section_path.read_text(encoding="utf-8")
    block = f"{begin}\n\n## Broker-like partial-light-curve multiseed figure\n\n{section}\n\n{end}"

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
    log("[START] Broker-like partial multiseed metrics figure")

    df, source = load_data(args.input_csv, args.model)
    output_png = plot(df, args.output_dir)
    log(f"[SAVE] {output_png}")

    section_path = write_summary(df, source, output_png, args.reports_dir, args.data_dir)

    if args.update_summary:
        update_final_summary(ROOT_DIR / "results" / "final_publication" / "final_publication_summary.md", section_path)

    log("[DONE]")


if __name__ == "__main__":
    main()
