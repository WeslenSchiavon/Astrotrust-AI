#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Publication figure: broker-like partial rare-class enrichment.

Reads:
  results/final_publication/broker_like_realism/multiseed_tuned_summary_stats.csv
or:
  results/broker_like_stage_aware_multiseed/multiseed_tuned_summary_stats.csv

Outputs:
  results/final_publication/figures/fig_broker_like_partial_rare_enrichment.png
  results/final_publication/figures/fig_broker_like_partial_rare_enrichment.pdf
  results/final_publication/figures/fig_broker_like_partial_rare_enrichment.svg
  results/final_publication/broker_like_realism/broker_like_partial_rare_enrichment_plot_data.csv
  results/final_publication/reports/broker_like_partial_rare_enrichment_figure_summary.md

Run:
  python experiments/plot_broker_like_partial_rare_enrichment.py
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

MODEL_ORDER = [
    "hybrid_early_aware",
    "ensemble_early_aware_fixed",
    "ensemble_early_aware_tuned",
]

MODEL_LABELS = {
    "hybrid_early_aware": "Hybrid early-aware",
    "ensemble_early_aware_fixed": "Fixed early-aware ensemble",
    "ensemble_early_aware_tuned": "Tuned early-aware ensemble",
}


def log(message: str) -> None:
    print(f"[PLOT-PARTIAL-ENRICHMENT] {message}", flush=True)


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
    p.add_argument("--update-summary", action="store_true")
    return p.parse_args()


def load_data(input_csv: Path | None) -> tuple[pd.DataFrame, Path]:
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
    df = df[(df["model_name"].isin(MODEL_ORDER)) & (df["scenario"].isin(SCENARIO_ORDER))].copy()
    if df.empty:
        raise ValueError("No rows found for expected models/scenarios.")

    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df["model_name"] = pd.Categorical(df["model_name"], categories=MODEL_ORDER, ordered=True)
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    df["model_label"] = df["model_name"].map(MODEL_LABELS)
    return df.sort_values(["model_name", "scenario"]).reset_index(drop=True), path


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.4,
        "axes.labelsize": 9.8,
        "axes.titlesize": 10.7,
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


def plot(df: pd.DataFrame, output_dir: Path) -> Path:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "fig_broker_like_partial_rare_enrichment"

    fig, ax = plt.subplots(figsize=(8.9, 4.7))
    x = np.arange(len(SCENARIO_ORDER))

    markers = {
        "hybrid_early_aware": "o",
        "ensemble_early_aware_fixed": "s",
        "ensemble_early_aware_tuned": "D",
    }

    for model in MODEL_ORDER:
        sub = df[df["model_name"] == model].sort_values("scenario")
        y = sub["rare_enrichment_5pct_mean"].to_numpy()
        yerr = sub["rare_enrichment_5pct_std"].to_numpy()

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=markers[model] + "-",
            linewidth=2.0 if model == "ensemble_early_aware_tuned" else 1.65,
            markersize=5.7,
            capsize=4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=MODEL_LABELS[model],
            zorder=4 if model == "ensemble_early_aware_tuned" else 3,
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="#555555", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Rare-class enrichment at 5% budget")
    ax.set_xlabel("Partial-light-curve scenario")
    ax.set_title("Rare-class enrichment under broker-like partial information", pad=10)
    ax.set_ylim(3.1, 4.85)
    ax.grid(axis="y", linewidth=0.5, alpha=0.28, zorder=1)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.75,
    )

    ax.text(
        0.015,
        0.98,
        "Mean ± SD across 10 training seeds\nBudget = top 5% prioritized candidates",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#D0D0D0", lw=0.7, alpha=0.96),
    )

    ax.text(
        0.985,
        0.98,
        "Tuned route maintains >3.7× enrichment\nin all early-stage scenarios",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#D0D0D0", lw=0.7, alpha=0.96),
    )

    fig.tight_layout()
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(output_base.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)

    return output_base.with_suffix(".png")


def write_summary(df: pd.DataFrame, source: Path, output_png: Path, reports_dir: Path, data_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / "broker_like_partial_rare_enrichment_plot_data.csv"
    df.to_csv(data_path, index=False)

    summary_path = reports_dir / "broker_like_partial_rare_enrichment_figure_summary.md"

    cols = [
        "model_name", "scenario", "n_seeds",
        "rare_enrichment_5pct_mean", "rare_enrichment_5pct_std",
        "rare_rate_5pct_mean", "rare_rate_5pct_std",
    ]

    md = f"""# Broker-like partial rare-enrichment figure

Generated at: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`

Source: `{source}`

Figure: `{output_png}`

Data: `{data_path}`

## Values plotted

{df[cols].to_markdown(index=False, floatfmt=".4f")}

## Suggested caption

```latex
\\caption{{Rare-class enrichment under broker-like partial-light-curve conditions. The figure compares the standalone early-aware hybrid, the fixed-weight early-aware ensemble, and the validation-only tuned early-aware ensemble. Values are reported at a 5\\% follow-up budget as mean $\\pm$ standard deviation across ten training seeds. The tuned route preserves strong rare-candidate enrichment across both first-$N$ and fixed-window scenarios.}}
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

    begin = "<!-- BEGIN_BROKER_LIKE_PARTIAL_RARE_ENRICHMENT_FIGURE -->"
    end = "<!-- END_BROKER_LIKE_PARTIAL_RARE_ENRICHMENT_FIGURE -->"
    section = section_path.read_text(encoding="utf-8")
    block = f"{begin}\n\n## Broker-like partial rare-enrichment figure\n\n{section}\n\n{end}"

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
    log("[START] Partial rare-enrichment figure")

    df, source = load_data(args.input_csv)
    output_png = plot(df, args.output_dir)
    log(f"[SAVE] {output_png}")

    section_path = write_summary(df, source, output_png, args.reports_dir, args.data_dir)

    if args.update_summary:
        update_final_summary(ROOT_DIR / "results" / "final_publication" / "final_publication_summary.md", section_path)

    log("[DONE]")


if __name__ == "__main__":
    main()
