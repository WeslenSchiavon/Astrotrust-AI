from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "final_publication" / "dataset_size_scaling"


def read_first_row(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV file: {path}")
    return df.iloc[0].to_dict()


def read_final_model_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "final_ai_summary_hybrid" / "final_model_performance_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def get_model_row(summary: pd.DataFrame, model: str, dataset: str) -> dict:
    mask = (
        (summary["model"].astype(str) == model)
        & (summary["dataset"].astype(str) == dataset)
    )
    if not mask.any():
        raise ValueError(f"Could not find model={model}, dataset={dataset} in final summary")
    return summary.loc[mask].iloc[0].to_dict()


def add_row(rows: list[dict], model: str, dataset_size: str, source: str, row: dict):
    rows.append(
        {
            "model": model,
            "dataset_size": dataset_size,
            "n_objects_nominal": int(dataset_size.replace("k", "")) * 1000,
            "accuracy": float(row["accuracy"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "weighted_f1": float(row["weighted_f1"]),
            "source": source,
        }
    )


def build_clean_scaling_table() -> pd.DataFrame:
    final_summary = read_final_model_summary()
    rows: list[dict] = []

    # Comparable LightGBM baseline scaling from the final summary.
    for dataset_size in ["25k", "100k", "250k"]:
        row = get_model_row(final_summary, "lightgbm_v4_baseline", dataset_size)
        add_row(
            rows,
            model="LightGBM v4 baseline",
            dataset_size=dataset_size,
            source="final_model_performance_summary.csv",
            row=row,
        )

    # Comparable hybrid temporal-tabular CNN scaling.
    hybrid_25k = read_first_row(
        RESULTS_DIR
        / "hybrid_temporal_tabular_cnn_25k"
        / "hybrid_temporal_tabular_cnn_test_metrics.csv"
    )
    add_row(
        rows,
        model="Hybrid temporal-tabular CNN",
        dataset_size="25k",
        source="hybrid_temporal_tabular_cnn_25k/hybrid_temporal_tabular_cnn_test_metrics.csv",
        row=hybrid_25k,
    )

    hybrid_100k = read_first_row(
        RESULTS_DIR
        / "hybrid_temporal_tabular_cnn_100k"
        / "hybrid_temporal_tabular_cnn_test_metrics.csv"
    )
    add_row(
        rows,
        model="Hybrid temporal-tabular CNN",
        dataset_size="100k",
        source="hybrid_temporal_tabular_cnn_100k/hybrid_temporal_tabular_cnn_test_metrics.csv",
        row=hybrid_100k,
    )

    row = get_model_row(final_summary, "hybrid_temporal_tabular_cnn", "250k")
    add_row(
        rows,
        model="Hybrid temporal-tabular CNN",
        dataset_size="250k",
        source="final_model_performance_summary.csv",
        row=row,
    )

    # Final ensemble is available only at 250k and should be shown as the final best model, not as a scaling curve.
    row = get_model_row(final_summary, "ensemble_hybrid_dominant", "250k")
    add_row(
        rows,
        model="Ensemble hybrid dominant",
        dataset_size="250k",
        source="final_model_performance_summary.csv",
        row=row,
    )

    df = pd.DataFrame(rows)
    size_order = {"25k": 25_000, "100k": 100_000, "250k": 250_000}
    model_order = {
        "LightGBM v4 baseline": 0,
        "Hybrid temporal-tabular CNN": 1,
        "Ensemble hybrid dominant": 2,
    }
    df["_model_order"] = df["model"].map(model_order)
    df["_size_order"] = df["dataset_size"].map(size_order)
    df = df.sort_values(["_model_order", "_size_order"]).drop(columns=["_model_order", "_size_order"])
    return df


def build_scaling_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["LightGBM v4 baseline", "Hybrid temporal-tabular CNN"]:
        group = df[df["model"] == model].sort_values("n_objects_nominal")
        if group.empty:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "model": model,
                "first_dataset_size": first["dataset_size"],
                "last_dataset_size": last["dataset_size"],
                "accuracy_gain": float(last["accuracy"] - first["accuracy"]),
                "macro_f1_gain": float(last["macro_f1"] - first["macro_f1"]),
                "accuracy_gain_25k_to_100k": gain_between(group, "25k", "100k", "accuracy"),
                "accuracy_gain_100k_to_250k": gain_between(group, "100k", "250k", "accuracy"),
                "macro_f1_gain_25k_to_100k": gain_between(group, "25k", "100k", "macro_f1"),
                "macro_f1_gain_100k_to_250k": gain_between(group, "100k", "250k", "macro_f1"),
            }
        )
    return pd.DataFrame(rows)


def gain_between(group: pd.DataFrame, a: str, b: str, metric: str):
    row_a = group[group["dataset_size"] == a]
    row_b = group[group["dataset_size"] == b]
    if row_a.empty or row_b.empty:
        return None
    return float(row_b.iloc[0][metric] - row_a.iloc[0][metric])


def plot_scaling(df: pd.DataFrame, metric: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    plot_df = df[df["model"].isin(["LightGBM v4 baseline", "Hybrid temporal-tabular CNN"])].copy()
    for model, group in plot_df.groupby("model"):
        group = group.sort_values("n_objects_nominal")
        ax.plot(group["n_objects_nominal"], group[metric], marker="o", label=model)

    ax.set_xlabel("Dataset size (objects)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Dataset-size scaling: {metric.replace('_', ' ')}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    md = "# Clean dataset-size scaling analysis\n\n"
    md += "This table includes only directly comparable final experiments. Exploratory, calibration-only, family-level, and older small-sample experiments are excluded.\n\n"
    md += "## Clean scaling table\n\n"
    md += df.drop(columns=["source"]).to_markdown(index=False)
    md += "\n\n## Scaling gains\n\n"
    md += summary.to_markdown(index=False)
    md += "\n\n## Suggested interpretation for the manuscript\n\n"
    md += (
        "The dataset-size analysis shows that both the tabular LightGBM baseline and the hybrid temporal-tabular CNN benefit from larger ELAsTiCC/LSST-like training sets. "
        "The LightGBM baseline improves steadily from 25k to 250k objects, while the hybrid model exhibits stronger scaling and only surpasses the tabular baseline once sufficient training data are available. "
        "This supports the interpretation that the hybrid architecture requires larger samples to exploit the complementarity between temporal light-curve tensors and tabular features. "
        "At 250k objects, the hybrid model clearly outperforms the tabular baseline, and the final hybrid-dominant ensemble provides the best raw top-1 performance.\n"
    )
    (output_dir / "clean_dataset_size_scaling_summary.md").write_text(md, encoding="utf-8")


def update_final_publication_summary(output_dir: Path, df: pd.DataFrame, summary: pd.DataFrame):
    final_summary_path = RESULTS_DIR / "final_publication" / "final_publication_summary.md"
    if not final_summary_path.exists():
        return

    current = final_summary_path.read_text(encoding="utf-8")
    marker = "\n## Dataset-size scaling\n\n"
    section = marker
    section += "The clean dataset-size scaling analysis compares only directly comparable final runs.\n\n"
    section += df.drop(columns=["source"]).to_markdown(index=False)
    section += "\n\nScaling gains:\n\n"
    section += summary.to_markdown(index=False)
    section += "\n"

    if marker in current:
        current = current.split(marker)[0].rstrip() + section
    else:
        current = current.rstrip() + "\n" + section

    final_summary_path.write_text(current, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate clean dataset-size scaling table for the AstroTrust-AI manuscript.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = build_clean_scaling_table()
    summary = build_scaling_summary(df)

    df.to_csv(args.output_dir / "table_dataset_size_scaling_clean.csv", index=False)
    summary.to_csv(args.output_dir / "table_dataset_size_scaling_gains.csv", index=False)

    # Also copy to final_publication/tables for manuscript use.
    tables_dir = RESULTS_DIR / "final_publication" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "table_dataset_size_scaling_clean.csv", index=False)
    summary.to_csv(tables_dir / "table_dataset_size_scaling_gains.csv", index=False)

    plot_scaling(df, "accuracy", args.output_dir / "fig_clean_dataset_size_accuracy.png")
    plot_scaling(df, "macro_f1", args.output_dir / "fig_clean_dataset_size_macro_f1.png")

    # Also copy figures to final_publication/figures.
    figures_dir = RESULTS_DIR / "final_publication" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_scaling(df, "accuracy", figures_dir / "fig_clean_dataset_size_accuracy.png")
    plot_scaling(df, "macro_f1", figures_dir / "fig_clean_dataset_size_macro_f1.png")

    write_markdown(df, summary, args.output_dir)
    update_final_publication_summary(args.output_dir, df, summary)

    print("[OK] Clean dataset-size scaling table generated.")
    print(f"Output directory: {args.output_dir}")
    print("\nClean table:")
    print(df.drop(columns=["source"]).to_string(index=False))
    print("\nScaling gains:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"- {args.output_dir / 'table_dataset_size_scaling_clean.csv'}")
    print(f"- {args.output_dir / 'table_dataset_size_scaling_gains.csv'}")
    print(f"- {args.output_dir / 'fig_clean_dataset_size_accuracy.png'}")
    print(f"- {args.output_dir / 'fig_clean_dataset_size_macro_f1.png'}")
    print(f"- {args.output_dir / 'clean_dataset_size_scaling_summary.md'}")
    print(f"- {tables_dir / 'table_dataset_size_scaling_clean.csv'}")
    print(f"- {tables_dir / 'table_dataset_size_scaling_gains.csv'}")


if __name__ == "__main__":
    main()
