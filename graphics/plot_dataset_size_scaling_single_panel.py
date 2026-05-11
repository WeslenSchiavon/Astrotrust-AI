#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-panel publication-quality dataset-size scaling figure for AstroTrust-AI.

This version uses one unified graph:
  - x-axis: nominal dataset size: 25k, 100k, 250k
  - y-axis: score
  - color: model family
  - line style: metric
      solid  = Accuracy
      dashed = Macro-F1

Outputs:
  results/final_publication/figures/fig_dataset_size_scaling_single_panel.png
  results/final_publication/figures/fig_dataset_size_scaling_single_panel.pdf
  results/final_publication/figures/fig_dataset_size_scaling_single_panel.svg
  results/final_publication/model_performance/dataset_size_scaling_single_panel_data.csv
  results/final_publication/reports/dataset_size_scaling_single_panel_summary.md

It also updates:
  results/final_publication/final_publication_summary.md

Run from the project root:
  python experiments/plot_dataset_size_scaling_single_panel.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


def build_scaling_dataframe() -> pd.DataFrame:
    """Final values from results/final_publication/final_publication_summary.md."""
    rows = [
        # model, dataset_size, n_objects_nominal, accuracy, macro_f1
        ("LightGBM v4 baseline", "25k", 25_000, 0.577760, 0.568219),
        ("LightGBM v4 baseline", "100k", 100_000, 0.620806, 0.611582),
        ("LightGBM v4 baseline", "250k", 250_000, 0.634074, 0.624910),

        ("Hybrid temporal-tabular CNN", "25k", 25_000, 0.544480, 0.539075),
        ("Hybrid temporal-tabular CNN", "100k", 100_000, 0.636887, 0.627761),
        ("Hybrid temporal-tabular CNN", "250k", 250_000, 0.673420, 0.668218),

        ("Ensemble hybrid dominant", "25k", 25_000, 0.574720, 0.565683),
        ("Ensemble hybrid dominant", "100k", 100_000, 0.652845, 0.640088),
        ("Ensemble hybrid dominant", "250k", 250_000, 0.684163, 0.676580),
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "dataset_size",
            "n_objects_nominal",
            "accuracy",
            "macro_f1",
        ],
    )

    dataset_order = {"25k": 0, "100k": 1, "250k": 2}
    df["dataset_order"] = df["dataset_size"].map(dataset_order)

    return df.sort_values(["model", "dataset_order"]).reset_index(drop=True)


def configure_publication_style() -> None:
    """Clean, compact style suitable for a journal figure."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def plot_single_panel(df: pd.DataFrame, output_base: Path) -> None:
    """Generate the single-panel Accuracy/Macro-F1 scaling figure."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    configure_publication_style()

    model_order = [
        "LightGBM v4 baseline",
        "Hybrid temporal-tabular CNN",
        "Ensemble hybrid dominant",
    ]

    # Color-blind friendly palette. Different line styles and markers keep the
    # figure interpretable even in grayscale print.
    model_colors = {
        "LightGBM v4 baseline": "#4D4D4D",
        "Hybrid temporal-tabular CNN": "#0072B2",
        "Ensemble hybrid dominant": "#D55E00",
    }

    metric_specs = {
        "accuracy": {
            "label": "Accuracy",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.15,
            "markersize": 5.4,
        },
        "macro_f1": {
            "label": "Macro-F1",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 1.85,
            "markersize": 4.9,
        },
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.55))

    for model in model_order:
        model_df = df[df["model"] == model].sort_values("n_objects_nominal")

        for metric, spec in metric_specs.items():
            is_ensemble = model == "Ensemble hybrid dominant"
            lw = spec["linewidth"] + (0.35 if is_ensemble else 0.0)
            zorder = 5 if is_ensemble else 3

            ax.plot(
                model_df["n_objects_nominal"],
                model_df[metric],
                color=model_colors[model],
                linestyle=spec["linestyle"],
                marker=spec["marker"],
                linewidth=lw,
                markersize=spec["markersize"],
                markeredgewidth=0.8,
                markeredgecolor="white",
                label=f"{model} — {spec['label']}",
                zorder=zorder,
            )

    # Use a log-scaled x-axis because the dataset sizes are multiplicative.
    ax.set_xscale("log")
    ax.set_xticks([25_000, 100_000, 250_000])
    ax.set_xticklabels(["25k", "100k", "250k"])

    ax.set_ylim(0.525, 0.700)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.set_xlabel("Nominal training-set size")
    ax.set_ylabel("Test performance score")
    ax.set_title("Dataset-size scaling of AstroTrust-AI models", pad=10)

    ax.grid(axis="y", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", linewidth=0.35, alpha=0.12)

    # Direct final-performance annotation for the main message.
    best_text = (
        "Best final configuration at 250k\n"
        "Ensemble: Accuracy = 0.684, Macro-F1 = 0.677"
    )
    ax.annotate(
        best_text,
        xy=(250_000, 0.684163),
        xytext=(50_000, 0.690),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.8,
            color="#333333",
            shrinkA=3,
            shrinkB=3,
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="#BDBDBD",
            lw=0.7,
            alpha=0.96,
        ),
        fontsize=8.2,
        ha="left",
        va="center",
    )

    # Endpoint labels: model names only, avoiding a crowded legend.
    endpoint_offsets = {
        "LightGBM v4 baseline": -0.002,
        "Hybrid temporal-tabular CNN": 0.000,
        "Ensemble hybrid dominant": 0.002,
    }

    for model in model_order:
        model_df = df[(df["model"] == model) & (df["dataset_size"] == "250k")].iloc[0]
        ax.text(
            260_000,
            model_df["accuracy"] + endpoint_offsets[model],
            model.replace(" temporal-tabular ", "\n")
                 .replace(" hybrid dominant", "\nhybrid dominant")
                 .replace(" v4 ", "\nv4 "),
            color=model_colors[model],
            fontsize=7.8,
            va="center",
            ha="left",
        )

    # Add separate compact legends: one for models, one for metrics.
    model_handles = [
        Line2D([0], [0], color=model_colors[m], lw=2.5, label=m)
        for m in model_order
    ]

    metric_handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            lw=2.0,
            linestyle=metric_specs["accuracy"]["linestyle"],
            marker=metric_specs["accuracy"]["marker"],
            markersize=5,
            label="Accuracy",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            lw=2.0,
            linestyle=metric_specs["macro_f1"]["linestyle"],
            marker=metric_specs["macro_f1"]["marker"],
            markersize=5,
            label="Macro-F1",
        ),
    ]

    legend_1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.8,
    )
    legend_1.get_title().set_fontsize(8.8)
    ax.add_artist(legend_1)

    legend_2 = ax.legend(
        handles=metric_handles,
        title="Metric",
        loc="upper left",
        frameon=True,
        framealpha=0.96,
        edgecolor="#D0D0D0",
        borderpad=0.8,
    )
    legend_2.get_title().set_fontsize(8.8)

    # Footnote-style note inside the figure.
    ax.text(
        0.01,
        -0.19,
        "Note: 25k/100k ensemble rows are supplementary fixed-ensemble scaling checks; 250k is the final large-scale configuration.",
        transform=ax.transAxes,
        fontsize=7.6,
        ha="left",
        va="top",
    )

    fig.tight_layout()

    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def compute_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gains from 25k to 250k."""
    rows = []

    for model, model_df in df.groupby("model"):
        model_df = model_df.set_index("dataset_size")
        rows.append({
            "model": model,
            "accuracy_25k": model_df.loc["25k", "accuracy"],
            "accuracy_250k": model_df.loc["250k", "accuracy"],
            "accuracy_gain_25k_to_250k": (
                model_df.loc["250k", "accuracy"] - model_df.loc["25k", "accuracy"]
            ),
            "macro_f1_25k": model_df.loc["25k", "macro_f1"],
            "macro_f1_250k": model_df.loc["250k", "macro_f1"],
            "macro_f1_gain_25k_to_250k": (
                model_df.loc["250k", "macro_f1"] - model_df.loc["25k", "macro_f1"]
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "accuracy_gain_25k_to_250k",
        ascending=False,
    )


def write_summary(
    df: pd.DataFrame,
    gain_df: pd.DataFrame,
    summary_path: Path,
    figure_png: Path,
    figure_pdf: Path,
    figure_svg: Path,
    csv_path: Path,
) -> str:
    """Write a Markdown summary for the figure."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    best_250k = (
        df[df["dataset_size"] == "250k"]
        .sort_values("accuracy", ascending=False)
        .iloc[0]
    )

    gain_lines = []
    for _, row in gain_df.iterrows():
        gain_lines.append(
            f'| {row["model"]} | '
            f'{row["accuracy_gain_25k_to_250k"]:.6f} | '
            f'{row["macro_f1_gain_25k_to_250k"]:.6f} |'
        )

    md = f"""# Dataset-size scaling: single-panel figure

Generated at: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`

## Generated files

- PNG: `{figure_png}`
- PDF: `{figure_pdf}`
- SVG: `{figure_svg}`
- CSV: `{csv_path}`

## Figure design

This figure uses one unified panel:

- color encodes the model family;
- solid lines with circle markers encode Accuracy;
- dashed lines with square markers encode Macro-F1;
- the x-axis uses a logarithmic scale because the nominal dataset sizes increase multiplicatively.

## Main result

At 250k, the strongest final configuration is `{best_250k["model"]}`, with Accuracy = `{best_250k["accuracy"]:.6f}` and Macro-F1 = `{best_250k["macro_f1"]:.6f}`.

## Scaling gains from 25k to 250k

| model | Accuracy gain | Macro-F1 gain |
|:--|--:|--:|
{chr(10).join(gain_lines)}

## Interpretation

The single-panel scaling figure shows that increasing the dataset size benefits the hybrid models more strongly than the purely tabular LightGBM baseline. The standalone hybrid temporal-tabular CNN exhibits the largest absolute gain from 25k to 250k, while the ensemble hybrid dominant reaches the best final performance at 250k.

The 25k and 100k ensemble rows should be described as supplementary fixed-ensemble scaling checks. The 250k ensemble row corresponds to the final large-scale AstroTrust-AI configuration.

## Suggested LaTeX caption

```latex
\\caption{{Single-panel dataset-size scaling of AstroTrust-AI classification performance. The figure reports held-out Accuracy and Macro-F1 across 25k, 100k, and 250k nominal training objects for the LightGBM v4 baseline, the hybrid temporal--tabular CNN, and the dominant hybrid ensemble. Color encodes the model family, whereas line style encodes the metric. The hybrid models benefit more strongly from increased dataset size than the tabular baseline, and the ensemble achieves the strongest final performance at 250k.}}
```
"""

    summary_path.write_text(md, encoding="utf-8")
    return md


def update_final_publication_summary(final_summary_path: Path, section_md: str) -> None:
    """Insert or replace a generated section in final_publication_summary.md."""
    if not final_summary_path.exists():
        print(f"[WARN] Could not update missing file: {final_summary_path}")
        return

    begin = "<!-- BEGIN_DATASET_SIZE_SCALING_SINGLE_PANEL_FIGURE -->"
    end = "<!-- END_DATASET_SIZE_SCALING_SINGLE_PANEL_FIGURE -->"

    block = f"""{begin}

## Dataset-size scaling single-panel figure

{section_md.split("## Figure design", 1)[1].strip()}

{end}
"""

    text = final_summary_path.read_text(encoding="utf-8")

    if begin in text and end in text:
        before = text.split(begin, 1)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        new_text = f"{before}\n\n{block.strip()}\n\n{after}"
    else:
        new_text = text.rstrip() + "\n\n" + block.strip() + "\n"

    final_summary_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    final_publication_dir = project_root / "results" / "final_publication"
    figures_dir = final_publication_dir / "figures"
    reports_dir = final_publication_dir / "reports"
    model_performance_dir = final_publication_dir / "model_performance"

    output_base = figures_dir / "fig_dataset_size_scaling_single_panel"
    csv_path = model_performance_dir / "dataset_size_scaling_single_panel_data.csv"
    summary_path = reports_dir / "dataset_size_scaling_single_panel_summary.md"
    final_summary_path = final_publication_dir / "final_publication_summary.md"

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_performance_dir.mkdir(parents=True, exist_ok=True)

    df = build_scaling_dataframe()
    df.to_csv(csv_path, index=False)

    plot_single_panel(df, output_base)

    gain_df = compute_gain_table(df)

    section_md = write_summary(
        df=df,
        gain_df=gain_df,
        summary_path=summary_path,
        figure_png=output_base.with_suffix(".png"),
        figure_pdf=output_base.with_suffix(".pdf"),
        figure_svg=output_base.with_suffix(".svg"),
        csv_path=csv_path,
    )

    update_final_publication_summary(final_summary_path, section_md)

    print("[OK] Single-panel dataset-size scaling figure generated.")
    print(f"[OK] PNG: {output_base.with_suffix('.png')}")
    print(f"[OK] PDF: {output_base.with_suffix('.pdf')}")
    print(f"[OK] SVG: {output_base.with_suffix('.svg')}")
    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] Updated: {final_summary_path}")


if __name__ == "__main__":
    main()
