#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Publication figure: AstroTrust-AI conceptual pipeline with stage-aware routing.

This is an updated conceptual diagram that adds the broker-like stage-aware route:

  complete light curve  -> ensemble_hybrid_dominant
  partial alert         -> validation-only tuned early-aware ensemble

Output:
  results/final_publication/figures/fig_astrotrust_ai_conceptual_pipeline_stage_aware.png
  results/final_publication/figures/fig_astrotrust_ai_conceptual_pipeline_stage_aware.pdf
  results/final_publication/reports/astrotrust_ai_conceptual_pipeline_stage_aware_summary.md

Run:
  python experiments/plot_astrotrust_ai_conceptual_pipeline_stage_aware.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT_DIR = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[PLOT-STAGE-PIPELINE] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "figures")
    p.add_argument("--reports-dir", type=Path, default=ROOT_DIR / "results" / "final_publication" / "reports")
    return p.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.4,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def wrap(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def add_box(ax, xy, wh, title, body, face="#F8FAFC", edge="#64748B", title_size=10.5, body_size=8.4):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w/2, y + h - 0.028,
        title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
        zorder=3,
    )
    ax.text(
        x + 0.025, y + h - 0.075,
        wrap(body, width=max(24, int(w*95))),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=body_size,
        color="#334155",
        linespacing=1.22,
        zorder=3,
    )
    return {
        "left": (x, y + h/2),
        "right": (x + w, y + h/2),
        "top": (x + w/2, y + h),
        "bottom": (x + w/2, y),
        "center": (x + w/2, y + h/2),
    }


def add_arrow(ax, start, end, color="#64748B", lw=1.2, rad=0.0):
    arrow = FancyArrowPatch(
        start, end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
        zorder=5,
    )
    ax.add_patch(arrow)


def plot(output_dir: Path) -> Path:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "fig_astrotrust_ai_conceptual_pipeline_stage_aware"

    fig = plt.figure(figsize=(13.2, 6.9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(
        0.5, 0.955,
        "AstroTrust-AI stage-aware broker-like inference",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=18.0,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5, 0.905,
        "Complete light curves use the original dominant ensemble, while partial alert states are routed to a validation-only tuned early-aware ensemble.",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=10.5,
        color="#475569",
    )

    input_box = add_box(
        ax, (0.05, 0.57), (0.20, 0.20),
        "Alert/light-curve input",
        "Multi-band observations, object metadata, feature completeness checks, and alert-stage information.",
        face="#FFFFFF",
    )

    stage_box = add_box(
        ax, (0.34, 0.56), (0.20, 0.22),
        "Stage-aware router",
        "Determines whether the object is represented by a complete/well-sampled light curve or by an early partial alert state.",
        face="#EFF6FF",
        edge="#2563EB",
    )

    full_box = add_box(
        ax, (0.66, 0.68), (0.25, 0.17),
        "Complete-route model",
        "Use ensemble_hybrid_dominant with the original final weights for complete light-curve inference.",
        face="#F8FAFC",
        edge="#64748B",
    )

    partial_box = add_box(
        ax, (0.66, 0.43), (0.25, 0.19),
        "Partial-route model",
        "Use validation-only tuned early-aware ensemble trained with first-N and fixed-window partial light-curve representations.",
        face="#FFF7ED",
        edge="#D97706",
    )

    trust_box = add_box(
        ax, (0.22, 0.20), (0.26, 0.19),
        "Trust and reliability outputs",
        "Class probabilities, confidence, uncertainty, calibration diagnostics, novelty score, and domain-reliability flags.",
        face="#F0FDF4",
        edge="#15803D",
    )

    follow_box = add_box(
        ax, (0.58, 0.18), (0.28, 0.21),
        "Broker-like prioritization",
        "Rank candidates for follow-up using rarity, novelty, uncertainty, and the operational stage of the alert.",
        face="#FDF2F8",
        edge="#BE185D",
    )

    add_arrow(ax, input_box["right"], stage_box["left"])
    add_arrow(ax, stage_box["right"], full_box["left"], rad=0.10)
    add_arrow(ax, stage_box["right"], partial_box["left"], rad=-0.10)
    add_arrow(ax, full_box["bottom"], follow_box["top"], rad=-0.10)
    add_arrow(ax, partial_box["bottom"], follow_box["top"], rad=0.10)
    add_arrow(ax, stage_box["bottom"], trust_box["top"], rad=0.00)
    add_arrow(ax, trust_box["right"], follow_box["left"])

    ax.text(
        0.575, 0.755,
        "complete / mature state",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#334155",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD5E1", lw=0.6),
    )
    ax.text(
        0.575, 0.505,
        "partial / early alert state",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#92400E",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#FDBA74", lw=0.6),
    )

    ax.text(
        0.5, 0.07,
        "The stage-aware design preserves the final complete-light-curve classifier while adding an operational partial-light-curve route for early broker decisions.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#475569",
    )

    for ext in [".png", ".pdf"]:
        fig.savefig(output_base.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)

    return output_base.with_suffix(".png")


def write_summary(output_png: Path, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "astrotrust_ai_conceptual_pipeline_stage_aware_summary.md"

    md = f"""# AstroTrust-AI stage-aware conceptual pipeline figure

Generated at: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`

Figure: `{output_png}`

## Suggested caption

```latex
\\caption{{Stage-aware AstroTrust-AI inference architecture. Complete or mature light curves are routed to the original \\texttt{{ensemble\\_hybrid\\_dominant}} probability product, whereas partial alert states are routed to a validation-only tuned early-aware ensemble trained with first-$N$ and fixed-window partial-light-curve representations. Both routes feed a common trust and prioritization layer that reports class probabilities, uncertainty, novelty, rarity, and follow-up priority.}}
```
"""
    summary_path.write_text(md, encoding="utf-8")
    log(f"[SAVE] {summary_path}")
    return summary_path


def main() -> None:
    args = parse_args()
    log("[START] Stage-aware conceptual pipeline figure")
    output_png = plot(args.output_dir)
    log(f"[SAVE] {output_png}")
    write_summary(output_png, args.reports_dir)
    log("[DONE]")


if __name__ == "__main__":
    main()
