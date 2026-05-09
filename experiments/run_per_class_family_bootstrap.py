"""
AstroTrust-AI: per-class and per-family bootstrap confidence intervals.

Run locally from the project root. The script always writes a Markdown summary.

Example (adjust paths if needed):

python experiments/run_per_class_family_bootstrap.py ^
  --predictions results/hybrid_tabular_ensemble_250k/ensemble_hybrid_dominant_predictions.csv ^
  --baseline-predictions results/hybrid_temporal_tabular_cnn_250k/hybrid_temporal_tabular_cnn_test_predictions.csv ^
  --label-map data/processed/elasticc2_large/full_class_counts.csv ^
  --output-dir results/per_class_family_bootstrap ^
  --final-summary results/final_publication/final_publication_summary.md ^
  --n-bootstrap 5000

Prediction CSV must contain true_label and predicted_label. If object_id exists in both
files, baseline alignment is done by object_id; otherwise row order is used.

Label map is optional. If it contains family/coarse_family, per-family CIs are produced.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime
import re
import numpy as np
import pandas as pd

def _progress_every_default():
    return int(globals().get('_ASTROTRUST_PROGRESS_EVERY', 250))


METRICS = ["precision", "recall", "f1"]


def norm_label(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    s = str(x)
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except Exception:
            return s
    return s


def load_predictions(path: Path, true_col: str, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [true_col, pred_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {path}")
        df[col] = df[col].map(norm_label)
    if "object_id" in df.columns:
        df["object_id"] = df["object_id"].astype(str)
    return df


def load_label_map(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["label", "class_name", "family"])
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("label map must contain a 'label' column")
    df["label"] = df["label"].map(norm_label)
    if "class_name" not in df.columns:
        df["class_name"] = df["label"]
    for c in ["family", "coarse_family", "class_family", "superclass"]:
        if c in df.columns:
            df["family"] = df[c].astype(str)
            break
    return df


def infer_family(class_name: str) -> str:
    t = str(class_name).lower()
    if "snia" in t or ("sn" in t and "ia" in t): return "SNIa"
    if "snii" in t or "sn ii" in t: return "SNII"
    if "snib" in t or "snic" in t or "ibc" in t: return "SNIbc"
    if "slsn" in t: return "SLSN"
    if "kn" in t or "kilonova" in t: return "KN"
    if "tde" in t: return "TDE"
    if "agn" in t: return "AGN"
    if "ulens" in t or "microlens" in t: return "uLens"
    if "dwarf" in t or "nova" in t or "cv" in t: return "CV"
    if "cart" in t: return "CART"
    return "Other"


def build_metadata(labels, label_map, infer_family_flag=False):
    meta = pd.DataFrame({"label": [str(x) for x in labels]})
    if label_map is not None and not label_map.empty:
        meta = meta.merge(label_map, on="label", how="left")
    if "class_name" not in meta.columns:
        meta["class_name"] = meta["label"]
    meta["class_name"] = meta["class_name"].fillna(meta["label"])
    if "family" not in meta.columns:
        meta["family"] = np.nan
    if infer_family_flag:
        meta["family"] = meta["family"].fillna(meta["class_name"].map(infer_family))
    return meta


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = b > 0
    out[m] = a[m] / b[m]
    return out


def metric_arrays(y_true, y_pred, labels):
    tp = np.array([np.sum((y_true == lab) & (y_pred == lab)) for lab in labels], dtype=float)
    fp = np.array([np.sum((y_true != lab) & (y_pred == lab)) for lab in labels], dtype=float)
    fn = np.array([np.sum((y_true == lab) & (y_pred != lab)) for lab in labels], dtype=float)
    support = np.array([np.sum(y_true == lab) for lab in labels], dtype=float)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"support": support, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def point_metrics(y_true, y_pred, labels):
    arr = metric_arrays(y_true, y_pred, labels)
    df = pd.DataFrame({"label": labels})
    for k, v in arr.items():
        df[k] = v
    for k in ["support", "tp", "fp", "fn"]:
        df[k] = df[k].fillna(0).astype(int)
    return df


def bootstrap_ci_metrics(y_true, y_pred, labels, n_boot, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    dist = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        arr = metric_arrays(y_true[idx], y_pred[idx], labels)
        for m in METRICS:
            dist[m].append(arr[m])
    rows = []
    for i, lab in enumerate(labels):
        row = {"label": lab}
        for m in METRICS:
            vals = np.array([x[i] for x in dist[m]], dtype=float)
            row[f"{m}_ci_low"] = float(np.nanquantile(vals, alpha / 2))
            row[f"{m}_ci_high"] = float(np.nanquantile(vals, 1 - alpha / 2))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_diff_metrics(y_true, y_model, y_base, labels, n_boot, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_model = metric_arrays(y_true, y_model, labels)
    point_base = metric_arrays(y_true, y_base, labels)
    dist = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma = metric_arrays(y_true[idx], y_model[idx], labels)
        mb = metric_arrays(y_true[idx], y_base[idx], labels)
        for m in METRICS:
            dist[m].append(ma[m] - mb[m])
    rows = []
    for i, lab in enumerate(labels):
        row = {"label": lab}
        for m in METRICS:
            vals = np.array([x[i] for x in dist[m]], dtype=float)
            row[f"{m}_diff"] = float(point_model[m][i] - point_base[m][i])
            row[f"{m}_diff_ci_low"] = float(np.nanquantile(vals, alpha / 2))
            row[f"{m}_diff_ci_high"] = float(np.nanquantile(vals, 1 - alpha / 2))
            row[f"{m}_diff_supported_positive"] = bool(row[f"{m}_diff_ci_low"] > 0)
            row[f"{m}_diff_supported_negative"] = bool(row[f"{m}_diff_ci_high"] < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def add_metadata(df, meta):
    out = df.merge(meta, on="label", how="left")
    front = [c for c in ["label", "class_name", "family", "support", "tp", "fp", "fn"] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def align_with_baseline(primary, baseline, true_col, pred_col):
    if "object_id" in primary.columns and "object_id" in baseline.columns:
        merged = primary[["object_id", true_col, pred_col]].merge(
            baseline[["object_id", true_col, pred_col]],
            on="object_id", suffixes=("_model", "_baseline"), how="inner"
        )
        if len(merged) != len(primary) or len(merged) != len(baseline):
            raise ValueError(f"object_id alignment mismatch: primary={len(primary)}, baseline={len(baseline)}, merged={len(merged)}")
        if not np.all(merged[f"{true_col}_model"].values == merged[f"{true_col}_baseline"].values):
            raise ValueError("true_label mismatch after object_id alignment")
        return (
            merged[f"{true_col}_model"].values.astype(object),
            merged[f"{pred_col}_model"].values.astype(object),
            merged[f"{pred_col}_baseline"].values.astype(object),
        )
    if len(primary) != len(baseline):
        raise ValueError("No object_id columns and files have different lengths")
    if not np.all(primary[true_col].values == baseline[true_col].values):
        raise ValueError("No object_id columns and true_label columns do not match row-wise")
    return primary[true_col].values.astype(object), primary[pred_col].values.astype(object), baseline[pred_col].values.astype(object)


def map_to_family(y, meta):
    d = dict(zip(meta["label"].astype(str), meta["family"].fillna("Unknown").astype(str)))
    return np.array([d.get(str(x), "Unknown") for x in y], dtype=object)


def markdown_table(df, cols, n=None):
    use = df[cols].copy()
    if n is not None:
        use = use.head(n)
    return use.to_markdown(index=False, floatfmt=".4f")


def write_md(path, class_metrics, class_diff, family_metrics, family_diff, model_name, baseline_name, n_boot):
    lines = []
    lines.append("# AstroTrust-AI Per-class and Per-family Bootstrap Summary")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append(f"Bootstrap resamples: `{n_boot}`")
    lines.append(f"Primary model: `{model_name}`")
    if baseline_name:
        lines.append(f"Baseline model: `{baseline_name}`")
    lines.append("")
    lines.append("## Lowest-F1 classes")
    lines.append("")
    cols = [c for c in ["label", "class_name", "family", "support", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high"] if c in class_metrics.columns]
    lines.append(markdown_table(class_metrics.sort_values("f1"), cols, 10))
    lines.append("")
    lines.append("## Highest-F1 classes")
    lines.append("")
    lines.append(markdown_table(class_metrics.sort_values("f1", ascending=False), cols, 10))
    lines.append("")
    if class_diff is not None:
        merged = class_diff.merge(class_metrics[[c for c in ["label", "class_name", "family", "support"] if c in class_metrics.columns]], on="label", how="left")
        dcols = [c for c in ["label", "class_name", "family", "support", "f1_diff", "f1_diff_ci_low", "f1_diff_ci_high", "f1_diff_supported_positive", "f1_diff_supported_negative"] if c in merged.columns]
        lines.append("## Strongest class-level F1 gains over baseline")
        lines.append("")
        lines.append(markdown_table(merged.sort_values("f1_diff", ascending=False), dcols, 10))
        lines.append("")
        lines.append("## Strongest class-level F1 losses relative to baseline")
        lines.append("")
        lines.append(markdown_table(merged.sort_values("f1_diff"), dcols, 10))
        lines.append("")
        lines.append("## Class-level paired-difference summary")
        lines.append("")
        lines.append(f"- Supported positive F1 differences: `{int(merged['f1_diff_supported_positive'].sum())}`")
        lines.append(f"- Supported negative F1 differences: `{int(merged['f1_diff_supported_negative'].sum())}`")
        lines.append("")
    if family_metrics is not None:
        fcols = [c for c in ["label", "support", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high"] if c in family_metrics.columns]
        lines.append("## Per-family metrics")
        lines.append("")
        lines.append(markdown_table(family_metrics.sort_values("f1", ascending=False), fcols))
        lines.append("")
    if family_diff is not None:
        fdcols = [c for c in ["label", "f1_diff", "f1_diff_ci_low", "f1_diff_ci_high", "f1_diff_supported_positive", "f1_diff_supported_negative"] if c in family_diff.columns]
        lines.append("## Per-family paired differences")
        lines.append("")
        lines.append(markdown_table(family_diff.sort_values("f1_diff", ascending=False), fdcols))
        lines.append("")
    lines.append("## Suggested manuscript sentence")
    lines.append("")
    lines.append("> To complement aggregate metrics, we computed class-wise bootstrap confidence intervals for precision, recall, and F1. This analysis identifies lower-sensitivity classes and checks whether model improvements are broadly distributed across the taxonomy or concentrated in specific classes.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def update_final_summary(final_summary, output_dir, class_metrics, class_diff, family_metrics, model_name, baseline_name):
    text = final_summary.read_text(encoding="utf-8")
    scope = "Update scope: consolidated final bootstrap, calibration, follow-up, hierarchical, multi-seed stability, seed-level paired tests, completed dataset-size scaling, and per-class/per-family bootstrap sensitivity analysis."
    if "Update scope:" in text:
        text = re.sub(r"Update scope: .*", scope, text)
    else:
        text = text.replace("Updated at: `2026-05-09`", f"Updated at: `2026-05-09`\n\n{scope}")
    supported_gain = supported_loss = "NA"
    if class_diff is not None:
        supported_gain = int(class_diff["f1_diff_supported_positive"].sum())
        supported_loss = int(class_diff["f1_diff_supported_negative"].sum())
    key = f"| per_class_bootstrap | n_classes | {len(class_metrics)} | class-wise precision/recall/F1 bootstrap intervals |\n"
    if class_diff is not None:
        key += f"| per_class_bootstrap | supported_f1_gains_vs_baseline | {supported_gain} | classes with 95% bootstrap CI above zero |\n"
        key += f"| per_class_bootstrap | supported_f1_losses_vs_baseline | {supported_loss} | classes with 95% bootstrap CI below zero |"
    if "per_class_bootstrap | n_classes" not in text:
        marker = "| seed_level_paired_test | ensemble_nll_reduction"
        m = re.search(r"\| seed_level_paired_test \| ensemble_nll_reduction \|.*\|", text)
        if m:
            text = text[:m.end()] + "\n" + key + text[m.end():]
        else:
            marker2 = "| dataset_scaling | ensemble_hybrid_dominant_250k | accuracy = 0.6842; Macro-F1 = 0.6766 | final large-scale configuration |"
            text = text.replace(marker2, marker2 + "\n" + key)
    low = class_metrics.sort_values("f1").head(5)
    cols = [c for c in ["label", "class_name", "family", "support", "f1", "f1_ci_low", "f1_ci_high"] if c in low.columns]
    section = []
    section.append("## Per-class and per-family bootstrap sensitivity analysis")
    section.append("")
    section.append(f"Class-wise bootstrap confidence intervals were computed for `{model_name}` to quantify precision, recall, and F1 uncertainty at the individual-class level.")
    if baseline_name:
        section.append(f"Paired bootstrap differences against `{baseline_name}` were also computed for class-level sensitivity comparisons.")
    section.append("")
    section.append("Lowest-F1 classes:")
    section.append("")
    section.append(low[cols].to_markdown(index=False, floatfmt=".4f"))
    section.append("")
    if family_metrics is not None:
        fcols = [c for c in ["label", "support", "f1", "f1_ci_low", "f1_ci_high"] if c in family_metrics.columns]
        section.append("Per-family summary:")
        section.append("")
        section.append(family_metrics.sort_values("f1", ascending=False)[fcols].to_markdown(index=False, floatfmt=".4f"))
        section.append("")
    block = "\n\n" + "\n".join(section) + "\n"
    if "## Per-class and per-family bootstrap sensitivity analysis" not in text:
        if "\n## Suggested manuscript claims" in text:
            text = text.replace("\n## Suggested manuscript claims", block + "\n## Suggested manuscript claims")
        else:
            text += block
    out = output_dir / "final_publication_summary_with_perclass_bootstrap.md"
    out.write_text(text, encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--baseline-predictions", type=Path)
    p.add_argument("--label-map", type=Path)
    p.add_argument("--true-col", default="true_label")
    p.add_argument("--pred-col", default="predicted_label")
    p.add_argument("--model-name", default="ensemble_hybrid_dominant")
    p.add_argument("--baseline-name", default="hybrid_temporal_tabular_cnn")
    p.add_argument("--output-dir", type=Path, default=Path("results/per_class_family_bootstrap"))
    p.add_argument("--final-summary", type=Path)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=250, help="Print progress every N bootstrap iterations.")
    p.add_argument("--infer-family-from-class-name", action="store_true")
    args = p.parse_args()
    globals()["_ASTROTRUST_PROGRESS_EVERY"] = args.progress_every

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Starting per-class/per-family bootstrap analysis", flush=True)
    print(f"[INFO] n_bootstrap={args.n_bootstrap}", flush=True)
    print(f"[INFO] progress_every={args.progress_every}", flush=True)
    print("[INFO] Loading primary predictions...", flush=True)
    primary = load_predictions(args.predictions, args.true_col, args.pred_col)
    print(f"[INFO] Primary rows: {len(primary)}", flush=True)
    print("[INFO] Loading label map...", flush=True)
    label_map = load_label_map(args.label_map)
    print(f"[INFO] Label-map rows: {len(label_map)}", flush=True)

    y_true = primary[args.true_col].values.astype(object)
    y_pred = primary[args.pred_col].values.astype(object)
    labels = np.array(sorted(pd.unique(pd.Series(np.concatenate([y_true, y_pred]))).astype(str)), dtype=object)
    meta = build_metadata(labels, label_map, args.infer_family_from_class_name)

    class_point = point_metrics(y_true, y_pred, labels)
    class_ci = bootstrap_ci_metrics(y_true, y_pred, labels, args.n_bootstrap, args.seed)
    class_metrics = add_metadata(class_point.merge(class_ci, on="label"), meta)
    class_metrics_path = args.output_dir / "per_class_bootstrap_metrics.csv"
    class_metrics.to_csv(class_metrics_path, index=False)

    class_diff = None
    family_diff = None
    baseline_name = None
    if args.baseline_predictions:
        baseline_name = args.baseline_name
        baseline = load_predictions(args.baseline_predictions, args.true_col, args.pred_col)
        yt, yp, yb = align_with_baseline(primary, baseline, args.true_col, args.pred_col)
        labels_diff = np.array(sorted(pd.unique(pd.Series(np.concatenate([yt, yp, yb]))).astype(str)), dtype=object)
        meta_diff = build_metadata(labels_diff, label_map, args.infer_family_from_class_name)
        class_diff = bootstrap_diff_metrics(yt, yp, yb, labels_diff, args.n_bootstrap, args.seed + 1)
        class_diff = add_metadata(class_diff, meta_diff)
        class_diff.to_csv(args.output_dir / "per_class_model_minus_baseline_bootstrap.csv", index=False)

    family_metrics = None
    if "family" in meta.columns and meta["family"].notna().any():
        meta["family"] = meta["family"].fillna("Unknown")
        yt_f = map_to_family(y_true, meta)
        yp_f = map_to_family(y_pred, meta)
        fam_labels = np.array(sorted(pd.unique(pd.Series(np.concatenate([yt_f, yp_f]))).astype(str)), dtype=object)
        fam_point = point_metrics(yt_f, yp_f, fam_labels)
        fam_ci = bootstrap_ci_metrics(yt_f, yp_f, fam_labels, args.n_bootstrap, args.seed + 2)
        family_metrics = fam_point.merge(fam_ci, on="label")
        family_metrics.to_csv(args.output_dir / "per_family_bootstrap_metrics.csv", index=False)
        if args.baseline_predictions:
            baseline = load_predictions(args.baseline_predictions, args.true_col, args.pred_col)
            yt, yp, yb = align_with_baseline(primary, baseline, args.true_col, args.pred_col)
            yt_f = map_to_family(yt, meta)
            yp_f = map_to_family(yp, meta)
            yb_f = map_to_family(yb, meta)
            fam_labels_diff = np.array(sorted(pd.unique(pd.Series(np.concatenate([yt_f, yp_f, yb_f]))).astype(str)), dtype=object)
            family_diff = bootstrap_diff_metrics(yt_f, yp_f, yb_f, fam_labels_diff, args.n_bootstrap, args.seed + 3)
            family_diff.to_csv(args.output_dir / "per_family_model_minus_baseline_bootstrap.csv", index=False)

    md_path = args.output_dir / "per_class_family_bootstrap_summary.md"
    write_md(md_path, class_metrics, class_diff, family_metrics, family_diff, args.model_name, baseline_name, args.n_bootstrap)

    print("Saved:")
    print(f"- {class_metrics_path}")
    if class_diff is not None:
        print(f"- {args.output_dir / 'per_class_model_minus_baseline_bootstrap.csv'}")
    if family_metrics is not None:
        print(f"- {args.output_dir / 'per_family_bootstrap_metrics.csv'}")
    if family_diff is not None:
        print(f"- {args.output_dir / 'per_family_model_minus_baseline_bootstrap.csv'}")
    print(f"- {md_path}")

    if args.final_summary:
        out = update_final_summary(args.final_summary, args.output_dir, class_metrics, class_diff, family_metrics, args.model_name, baseline_name)
        print(f"- {out}")


if __name__ == "__main__":
    main()
