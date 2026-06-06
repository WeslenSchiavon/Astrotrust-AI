#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Publication figure: validation-only tuned early-aware ensemble gains.

Reads:
  results/final_publication/broker_like_realism/multiseed_tuned_paired_summary_stats.csv
or:
  results/broker_like_stage_aware_multiseed/multiseed_tuned_paired_summary_stats.csv

Outputs:
  results/final_publication/figures/fig_broker_like_tuned_vs_fixed_gains.png
  results/final_publication/figures/fig_broker_like_tuned_vs_fixed_gains.pdf
  results/final_publication/figures/fig_broker_like_tuned_vs_fixed_gains.svg
  results/final_publication/broker_like_realism/broker_like_tuned_vs_fixed_gains_plot_data.csv
  results/final_publication/reports/broker_like_tuned_vs_fixed_gains_figure_summary.md

Run:
  python experiments/plot_broker_like_tuned_vs_fixed_gains.py
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

SCENARIO_ORDER = ["first_5_points", "first_10_points", "first_20_points", "window_30_days"]
SCENARIO_LABELS = {
    "first_5_points": "First\n5 pts",
    "first_10_points": "First\n10 pts",
    "first_20_points": "First\n20 pts",
    "window_30_days": "30-day\nwindow",
}


def log(message: str) -> None:
    print(f"[PLOT-TUNED-GAINS] {message}", flush=True)


def find_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find input CSV:\n" + "\n".join(str(p) for p in paths))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "figures")
    p.add_argument("--reports-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "reports")
    p.add_argument("--data-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "broker_like_realism")
    p.add_argument("--reference-model", type=str, default="ensemble_early_aware_fixed")
    p.add_argument("--target-model", type=str, default="ensemble_early_aware_tuned")
    p.add_argument("--update-summary", action="store_true")
    return p.parse_args()


def load_data(input_csv: Path | None, target_model: str, reference_model: str) -> tuple[pd.DataFrame, Path]:
    candidates = []
    if input_csv is not None:
        candidates.append(input_csv)

    candidates.extend([
        ROOT_DIR / "results" / "final_publication" / "broker_like_realism" / "multiseed_tuned_paired_summary_stats.csv",
        ROOT_DIR / "results" / "broker_like_stage_aware_multiseed" / "multiseed_tuned_paired_summary_stats.csv",
        Path.cwd() / "multiseed_tuned_paired_summary_stats.csv",
        Path("/mnt/data/multiseed_tuned_paired_summary_stats.csv"),
    ])

    path = find_existing(candidates)
    log(f"[LOAD] {path}")

    df = pd.read_csv(path)
    df = df[
        (df["target_model"] == target_model)
        & (df["reference_model"] == reference_model)
        & (df["scenario"].isin(SCENARIO_ORDER))
        & (df["metric"].isin(["accuracy", "macro_f1", "rare_enrichment_5pct"]))
    ].copy()

    if df.empty:
        raise ValueError("No paired gain rows found for the requested target/reference.")

    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    return df.sort_values(["metric", "scenario"]).reset_index(drop=True), path


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.4,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.7,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def metric_panel(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str, ylim_pad: float):
    sub = df[df["metric"] == metric].sort_values("scenario").copy()
    x = np.arange(len(sub))
    y = sub["delta_mean_across_seeds"].to_numpy()
    low = sub["mean_ci_low"].to_numpy()
    high = sub["mean_ci_high"].to_numpy()

    lower = y - low
    upper = high - y

    bars = ax.bar(
        x,
        y,
        width=0.64,
        yerr=[lower, upper],
        capsize=4,
        linewidth=0.9,
        edgecolor="#333333",
        zorder=3,
    )

    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#555555", zorder=2)
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["scenario_label"])
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)

    y_min = min(0.0, float(np.nanmin(low))) - ylim_pad
    y_max = max(0.0, float(np.nanmax(high))) + ylim_pad
    ax.set_ylim(y_min, y_max)

    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (y_max - y_min) * 0.035,
            f"+{value:.3f}" if metric == "rare_enrichment_5pct" else f"+{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def plot(df: pd.DataFrame, output_dir: Path, reference_model: str) -> Path:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "fig_broker_like_tuned_vs_fixed_gains"

    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.75), constrained_layout=True)

    metric_panel(axes[0], df, "accuracy", "Accuracy gain", "Δ accuracy", 0.0015)
    metric_panel(axes[1], df, "macro_f1", "Macro-F1 gain", "Δ macro-F1", 0.0015)
    metric_panel(axes[2], df, "rare_enrichment_5pct", "Rare-enrichment gain", "Δ enrichment", 0.025)

    ref_label = reference_model.replace("_", " ")
    fig.suptitle(
        f"Validation-only tuned early-aware ensemble gains over {ref_label}",
        fontsize=12.0,
        y=1.06,
    )
    fig.text(
        0.5,
        -0.04,
        "Bars show the mean paired gain across 10 training seeds; error bars show the mean object-level 95% bootstrap CI across seeds.",
        ha="center",
        va="bottom",
        fontsize=8.3,
    )

    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(output_base.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)

    return output_base.with_suffix(".png")


def write_summary(df: pd.DataFrame, source: Path, output_png: Path, reports_dir: Path, data_dir: Path, reference_model: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "broker_like_tuned_vs_fixed_gains_plot_data.csv"
    df.to_csv(data_path, index=False)

    summary_path = reports_dir / "broker_like_tuned_vs_fixed_gains_figure_summary.md"

    cols = [
        "reference_model", "scenario", "metric", "n_seeds",
        "delta_mean_across_seeds", "delta_std_across_seeds",
        "delta_min_across_seeds", "delta_max_across_seeds",
        "mean_ci_low", "mean_ci_high", "max_one_sided_p",
    ]

    md = f"""# Broker-like tuned-vs-fixed gains figure

Generated at: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`

Source: `{source}`

Reference model: `{reference_model}`

Figure: `{output_png}`

Data: `{data_path}`

## Values plotted

{df[cols].to_markdown(index=False, floatfmt=".4f")}

## Suggested caption

```latex
\\caption{{Effect of validation-only weight tuning on the broker-like partial-light-curve route. Bars show the mean paired gain of the tuned early-aware ensemble over the fixed-weight early-aware ensemble across ten training seeds. Error bars denote the mean object-level 95\\% bootstrap confidence interval across seeds. Tuning consistently improves top-1 accuracy, Macro-F1, and rare-class enrichment, while top-5 recovery remains high and is reported separately.}}
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

    begin = "<!-- BEGIN_BROKER_LIKE_TUNED_GAINS_FIGURE -->"
    end = "<!-- END_BROKER_LIKE_TUNED_GAINS_FIGURE -->"
    section = section_path.read_text(encoding="utf-8")
    block = f"{begin}\n\n## Broker-like tuned-vs-fixed gains figure\n\n{section}\n\n{end}"

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
    log("[START] Tuned-vs-fixed gains figure")

    df, source = load_data(args.input_csv, args.target_model, args.reference_model)
    output_png = plot(df, args.output_dir, args.reference_model)
    log(f"[SAVE] {output_png}")

    section_path = write_summary(df, source, output_png, args.reports_dir, args.data_dir, args.reference_model)

    if args.update_summary:
        update_final_summary(ROOT_DIR / "results" / "final_publication" / "final_publication_summary.md", section_path)

    log("[DONE]")


if __name__ == "__main__":
    main()
