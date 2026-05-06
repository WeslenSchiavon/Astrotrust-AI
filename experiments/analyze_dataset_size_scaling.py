from __future__ import annotations

from pathlib import Path
import argparse
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "dataset_size_scaling_analysis"

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "ece",
    "brier_score",
    "mean_confidence",
]

DATASET_SIZE_PATTERN = re.compile(r"(?P<size>\d+)(?P<unit>k|K|m|M)?")


def parse_dataset_size(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip().lower()

    if text in ["25k", "100k", "250k", "500k"]:
        return int(text.replace("k", "")) * 1000

    if text.endswith("obj"):
        text = text.replace("obj", "")

    match = DATASET_SIZE_PATTERN.search(text)
    if not match:
        return None

    size = int(match.group("size"))
    unit = match.group("unit")

    if unit is None:
        if size in [25, 100, 250, 500]:
            return size * 1000
        return size

    if unit.lower() == "k":
        return size * 1000

    if unit.lower() == "m":
        return size * 1_000_000

    return size


def infer_dataset_size_from_path(path: Path) -> int | None:
    parts = [path.stem, path.parent.name, str(path)]

    priority_patterns = [
        r"(?P<size>\d+)k",
        r"(?P<size>\d+)000obj",
        r"(?P<size>\d+)000_obj",
        r"(?P<size>\d+)obj",
    ]

    for text in parts:
        low = str(text).lower()

        for pattern in priority_patterns:
            match = re.search(pattern, low)
            if match:
                size = int(match.group("size"))

                if "k" in pattern:
                    return size * 1000

                if size in [25, 100, 250, 500]:
                    return size * 1000

                return size

    return None


def normalize_model_name(name: str) -> str:
    low = str(name).lower()

    if "ensemble_hybrid" in low:
        return "ensemble_hybrid_dominant"
    if "hybrid_temporal_tabular" in low or "hybrid" in low:
        return "hybrid_temporal_tabular_cnn"
    if "lightgbm" in low or "lgbm" in low:
        return "lightgbm_v4_baseline"
    if "temporal_cnn_v2" in low or "residual" in low or "dilated" in low:
        return "temporal_cnn_v2_residual_dilated"
    if "temporal_cnn" in low or "cnn" in low:
        return "temporal_cnn_lc_only"
    if "randomforest" in low or "random_forest" in low or "rf" == low:
        return "random_forest"
    if "extra" in low and "tree" in low:
        return "extra_trees"
    if "histgradient" in low or "hist_gradient" in low:
        return "hist_gradient_boosting"

    return str(name)


def load_final_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "final_ai_summary_hybrid" / "final_model_performance_summary.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for _, row in df.iterrows():
        dataset = row.get("dataset")
        dataset_size = parse_dataset_size(dataset)

        model = row.get("model", row.get("name", "unknown"))

        rows.append(
            {
                "source_file": str(path),
                "model": str(model),
                "model_group": normalize_model_name(str(model)),
                "dataset_label": str(dataset),
                "dataset_size": dataset_size,
                "family": row.get("family", None),
                **{c: row.get(c, np.nan) for c in METRIC_COLUMNS if c in row.index},
            }
        )

    return pd.DataFrame(rows)


def load_metric_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    metric_cols_present = [c for c in METRIC_COLUMNS if c in df.columns]

    if not metric_cols_present:
        return pd.DataFrame()

    rows = []

    for _, row in df.iterrows():
        model_raw = (
            row.get("model")
            or row.get("name")
            or row.get("experiment")
            or path.parent.name
        )

        dataset_raw = (
            row.get("dataset")
            or row.get("dataset_label")
            or row.get("n_objects")
            or row.get("train_size")
            or path.parent.name
            or path.stem
        )

        dataset_size = parse_dataset_size(dataset_raw)
        if dataset_size is None:
            dataset_size = infer_dataset_size_from_path(path)

        rows.append(
            {
                "source_file": str(path),
                "model": str(model_raw),
                "model_group": normalize_model_name(str(model_raw)),
                "dataset_label": str(dataset_raw),
                "dataset_size": dataset_size,
                "family": row.get("family", None),
                **{c: row.get(c, np.nan) for c in metric_cols_present},
            }
        )

    return pd.DataFrame(rows)


def discover_metric_files() -> pd.DataFrame:
    patterns = [
        "**/*metrics*.csv",
        "**/*summary*.csv",
        "**/*performance*.csv",
    ]

    frames = []

    for pattern in patterns:
        for path in RESULTS_DIR.glob(pattern):
            # Avoid reading outputs generated by this script.
            if "dataset_size_scaling_analysis" in str(path):
                continue
            if path.name.startswith("table_"):
                continue
            if path.name in ["final_publication_key_numbers.csv"]:
                continue

            frame = load_metric_file(path)
            if not frame.empty:
                frames.append(frame)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()


def combine_and_clean() -> pd.DataFrame:
    frames = []

    final_summary = load_final_summary()
    if not final_summary.empty:
        frames.append(final_summary)

    discovered = discover_metric_files()
    if not discovered.empty:
        frames.append(discovered)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Keep rows with a dataset size and at least one main metric.
    df = df[df["dataset_size"].notna()].copy()
    df["dataset_size"] = df["dataset_size"].astype(int)

    for col in METRIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove exact duplicates; keep the best macro_f1 for each model_group + dataset_size if repeated.
    sort_cols = [c for c in ["model_group", "dataset_size", "macro_f1", "accuracy"] if c in df.columns]
    if "macro_f1" in df.columns:
        df = df.sort_values(["model_group", "dataset_size", "macro_f1", "accuracy"], ascending=[True, True, False, False])
    df = df.drop_duplicates(subset=["model_group", "dataset_size"], keep="first")
    df = df.sort_values(["model_group", "dataset_size"]).reset_index(drop=True)

    return df


def create_scaling_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_group, group in df.groupby("model_group"):
        group = group.sort_values("dataset_size")

        if len(group) == 0:
            continue

        first = group.iloc[0]
        last = group.iloc[-1]

        row = {
            "model_group": model_group,
            "n_points": len(group),
            "min_dataset_size": int(first["dataset_size"]),
            "max_dataset_size": int(last["dataset_size"]),
            "first_accuracy": first.get("accuracy"),
            "last_accuracy": last.get("accuracy"),
            "delta_accuracy": last.get("accuracy") - first.get("accuracy") if pd.notna(first.get("accuracy")) and pd.notna(last.get("accuracy")) else np.nan,
            "first_macro_f1": first.get("macro_f1"),
            "last_macro_f1": last.get("macro_f1"),
            "delta_macro_f1": last.get("macro_f1") - first.get("macro_f1") if pd.notna(first.get("macro_f1")) and pd.notna(last.get("macro_f1")) else np.nan,
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values("model_group")


def plot_metric_scaling(df: pd.DataFrame, metric: str, output_path: Path):
    if metric not in df.columns or df[metric].dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_group, group in df.groupby("model_group"):
        group = group.sort_values("dataset_size")
        if group[metric].dropna().empty:
            continue
        ax.plot(group["dataset_size"], group[metric], marker="o", label=model_group)

    ax.set_xlabel("Dataset size (objects)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Dataset-size scaling: {metric.replace('_', ' ')}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_markdown(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    md = "# Dataset-size scaling analysis\n\n"
    md += (
        "This analysis summarizes how model performance changes with the number of ELAsTiCC/LSST-like objects used in the experiment. "
        "It is intended to support the Results section by showing whether performance gains saturate as the dataset grows.\n\n"
    )

    md += "## Aggregated scaling table\n\n"
    md += df.to_markdown(index=False) if not df.empty else "_No rows found._"
    md += "\n\n"

    md += "## Model-level scaling summary\n\n"
    md += summary.to_markdown(index=False) if not summary.empty else "_No summary available._"
    md += "\n\n"

    md += "## Interpretation notes\n\n"
    md += (
        "- A monotonic improvement with dataset size supports the claim that larger training sets improve generalization.\n"
        "- A plateau suggests that model capacity, feature representation, or class ambiguity has become the limiting factor.\n"
        "- If only LightGBM has multiple dataset sizes, the section should be framed as a tabular-baseline scaling analysis, not as a full scaling law for all architectures.\n"
        "- For a stronger manuscript, the same 25k/100k/250k comparison should ideally be repeated for the hybrid model if computationally feasible.\n"
    )

    path = output_dir / "dataset_size_scaling_summary.md"
    path.write_text(md, encoding="utf-8")


def identify_missing_recommended_runs(df: pd.DataFrame) -> pd.DataFrame:
    recommended_models = [
        "lightgbm_v4_baseline",
        "temporal_cnn_lc_only",
        "hybrid_temporal_tabular_cnn",
        "ensemble_hybrid_dominant",
    ]
    recommended_sizes = [25_000, 100_000, 250_000]

    existing = set(zip(df.get("model_group", []), df.get("dataset_size", [])))
    rows = []

    for model in recommended_models:
        for size in recommended_sizes:
            rows.append(
                {
                    "model_group": model,
                    "dataset_size": size,
                    "available": (model, size) in existing,
                    "recommended": not ((model, size) in existing),
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset-size scaling for AstroTrust-AI experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = combine_and_clean()

    if df.empty:
        raise FileNotFoundError(
            "No metric rows with dataset_size were found. Check results/final_ai_summary_hybrid or metric CSV files."
        )

    summary = create_scaling_summary(df)
    missing = identify_missing_recommended_runs(df)

    df.to_csv(args.output_dir / "dataset_size_scaling_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "dataset_size_scaling_model_summary.csv", index=False)
    missing.to_csv(args.output_dir / "dataset_size_scaling_missing_recommended_runs.csv", index=False)

    plot_metric_scaling(df, "accuracy", args.output_dir / "fig_dataset_size_accuracy.png")
    plot_metric_scaling(df, "macro_f1", args.output_dir / "fig_dataset_size_macro_f1.png")
    plot_metric_scaling(df, "balanced_accuracy", args.output_dir / "fig_dataset_size_balanced_accuracy.png")

    build_markdown(df, summary, args.output_dir)

    print("[OK] Saved dataset-size scaling analysis to:")
    print(args.output_dir)

    print("\nScaling metrics:")
    print(df[[c for c in ["model_group", "dataset_size", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "source_file"] if c in df.columns]].to_string(index=False))

    print("\nScaling summary:")
    print(summary.to_string(index=False))

    print("\nRecommended missing runs:")
    print(missing[missing["recommended"]].to_string(index=False))


if __name__ == "__main__":
    main()
