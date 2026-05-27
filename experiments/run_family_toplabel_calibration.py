"""
AstroTrust-AI family-wise top-label calibration diagnostics.

Purpose
-------
Check whether a low aggregate ECE may hide calibration heterogeneity across
astronomical families.

This script uses only object-level true labels, correctness, and top-label
confidence. Therefore, it computes TOP-LABEL ECE, not full multiclass ECE.

Run from the project root.

Example
-------
python experiments/run_family_toplabel_calibration.py ^
  --predictions results/hybrid_tabular_ensemble_250k/ensemble_hybrid_dominant_predictions.csv ^
  --label-map data/processed/elasticc2_large/full_class_counts.csv ^
  --output-dir results/family_toplabel_calibration ^
  --final-summary results/final_publication/final_publication_summary.md ^
  --n-bins 15 ^
  --binning uniform ^
  --infer-family-from-class-name

Outputs
-------
global_toplabel_calibration.csv
family_toplabel_calibration.csv
family_toplabel_calibration_bins.csv
family_toplabel_calibration_summary.md
final_publication_summary_with_family_calibration.md, if --final-summary is supplied
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

TRUE_CANDIDATES = [
    "true_label", "y_true", "target", "label_true", "class_true", "true_class",
    "true_class_id", "target_label", "label"
]
CONFIDENCE_CANDIDATES = [
    "confidence", "top1_probability", "top_probability", "max_probability",
    "probability", "conf", "p_max"
]
CORRECT_CANDIDATES = ["correct", "is_correct", "top1_correct"]


def norm_label(x) -> str:
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
            pass
    return s


def find_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], role: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit {role} column not found: {explicit}\nAvailable columns: {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer {role} column. Please pass --{role.replace('_', '-')}.\n"
        f"Candidates tried: {candidates}\nAvailable columns: {list(df.columns)}"
    )


def infer_family_from_name(name: str) -> str:
    text = str(name).lower()
    if "slsn" in text:
        return "SLSN"
    if "snia" in text or ("sn" in text and "ia" in text):
        return "SNIa"
    if "snii" in text or "sn ii" in text:
        return "SNII"
    if "snib" in text or "snic" in text or "ibc" in text:
        return "SNIbc"
    if "kn" in text or "kilonova" in text:
        return "KN"
    if "tde" in text:
        return "TDE"
    if "agn" in text:
        return "AGN"
    if "ulens" in text or "microlens" in text:
        return "uLens"
    if "dwarf" in text or "nova" in text or "cv" in text:
        return "CV"
    if "cart" in text:
        return "CART"
    return "Other"


def load_label_map(path: Path | None, infer_family: bool) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["label", "class_name", "family"])
    lm = pd.read_csv(path)
    if "label" not in lm.columns:
        raise ValueError(f"Label map must contain a 'label' column. Available columns: {list(lm.columns)}")
    lm = lm.copy()
    lm["label"] = lm["label"].map(norm_label)

    if "class_name" not in lm.columns:
        for alt in ["name", "class", "class_label", "target_name"]:
            if alt in lm.columns:
                lm["class_name"] = lm[alt].astype(str)
                break
    if "class_name" not in lm.columns:
        lm["class_name"] = lm["label"]

    fam_col = None
    for c in ["family", "coarse_family", "class_family", "superclass", "coarse_class"]:
        if c in lm.columns:
            fam_col = c
            break
    lm["family"] = lm[fam_col].astype(str) if fam_col is not None else np.nan

    if infer_family:
        lm["family"] = lm["family"].where(lm["family"].notna(), lm["class_name"].map(infer_family_from_name))
    lm["family"] = lm["family"].fillna("Unknown")
    return lm[["label", "class_name", "family"]].drop_duplicates("label")


def make_bins(conf: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    conf = np.asarray(conf, dtype=float)
    if binning == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if binning == "quantile":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        return edges
    raise ValueError("binning must be 'uniform' or 'quantile'")


def compute_toplabel_ece(df: pd.DataFrame, confidence_col: str, correct_col: str, n_bins: int, binning: str):
    d = df[[confidence_col, correct_col]].dropna().copy()
    d[confidence_col] = pd.to_numeric(d[confidence_col], errors="coerce")
    d[correct_col] = pd.to_numeric(d[correct_col], errors="coerce")
    d = d.dropna()

    if len(d) == 0:
        return {"n": 0, "accuracy": np.nan, "mean_confidence": np.nan, "ece": np.nan,
                "mce": np.nan, "n_bins": n_bins, "binning": binning}, pd.DataFrame()

    conf = d[confidence_col].to_numpy(dtype=float)
    corr = d[correct_col].to_numpy(dtype=float)
    edges = make_bins(conf, n_bins=n_bins, binning=binning)

    rows = []
    ece = 0.0
    mce = 0.0
    n = len(d)

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": i, "bin_low": lo, "bin_high": hi, "n": 0,
                         "mean_confidence": np.nan, "accuracy": np.nan,
                         "abs_gap": np.nan, "weighted_gap": 0.0})
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(corr[mask].mean())
        gap = abs(bin_acc - bin_conf)
        weighted = (count / n) * gap
        ece += weighted
        mce = max(mce, gap)
        rows.append({"bin": i, "bin_low": lo, "bin_high": hi, "n": count,
                     "mean_confidence": bin_conf, "accuracy": bin_acc,
                     "abs_gap": gap, "weighted_gap": weighted})

    summary = {"n": n, "accuracy": float(corr.mean()), "mean_confidence": float(conf.mean()),
               "ece": float(ece), "mce": float(mce), "n_bins": n_bins, "binning": binning}
    return summary, pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    d = df[cols].copy()
    if n is not None:
        d = d.head(n)
    return d.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--label-map", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/family_toplabel_calibration"))
    parser.add_argument("--final-summary", type=Path)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--binning", choices=["uniform", "quantile"], default="uniform")
    parser.add_argument("--true-col")
    parser.add_argument("--confidence-col")
    parser.add_argument("--correct-col")
    parser.add_argument("--infer-family-from-class-name", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading predictions...", flush=True)
    df = pd.read_csv(args.predictions)
    print(f"[INFO] Prediction rows: {len(df)}", flush=True)

    true_col = find_column(df, args.true_col, TRUE_CANDIDATES, "true_col")
    conf_col = find_column(df, args.confidence_col, CONFIDENCE_CANDIDATES, "confidence_col")
    correct_col = find_column(df, args.correct_col, CORRECT_CANDIDATES, "correct_col")

    label_map = load_label_map(args.label_map, infer_family=args.infer_family_from_class_name)

    df = df.copy()
    df["_true_label"] = df[true_col].map(norm_label)
    if not label_map.empty:
        df = df.merge(label_map, left_on="_true_label", right_on="label", how="left")
    else:
        df["class_name"] = df["_true_label"]
        df["family"] = "Unknown"
    df["family"] = df["family"].fillna("Unknown")

    if df[correct_col].dtype == bool:
        df[correct_col] = df[correct_col].astype(float)
    else:
        mapped = df[correct_col].astype(str).str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
        numeric = pd.to_numeric(df[correct_col], errors="coerce")
        df[correct_col] = mapped.fillna(numeric)

    print(f"[INFO] true_col={true_col}", flush=True)
    print(f"[INFO] confidence_col={conf_col}", flush=True)
    print(f"[INFO] correct_col={correct_col}", flush=True)
    print(f"[INFO] n_bins={args.n_bins}, binning={args.binning}", flush=True)

    global_summary, global_bins = compute_toplabel_ece(
        df, confidence_col=conf_col, correct_col=correct_col,
        n_bins=args.n_bins, binning=args.binning
    )
    pd.DataFrame([global_summary]).to_csv(args.output_dir / "global_toplabel_calibration.csv", index=False)
    global_bins.to_csv(args.output_dir / "global_toplabel_calibration_bins.csv", index=False)

    family_rows = []
    family_bin_rows = []
    for family, g in df.groupby("family"):
        s, bins = compute_toplabel_ece(g, confidence_col=conf_col, correct_col=correct_col,
                                       n_bins=args.n_bins, binning=args.binning)
        s["family"] = family
        family_rows.append(s)
        if not bins.empty:
            bins = bins.copy()
            bins["family"] = family
            family_bin_rows.append(bins)

    family_df = pd.DataFrame(family_rows)
    family_df = family_df[["family", "n", "accuracy", "mean_confidence", "ece", "mce", "n_bins", "binning"]]
    family_df = family_df.sort_values(["ece", "n"], ascending=[False, False])
    family_df.to_csv(args.output_dir / "family_toplabel_calibration.csv", index=False)

    if family_bin_rows:
        pd.concat(family_bin_rows, ignore_index=True).to_csv(
            args.output_dir / "family_toplabel_calibration_bins.csv", index=False
        )

    lines = []
    lines.append("# AstroTrust-AI Family-wise Top-label Calibration Diagnostics")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Metric definition")
    lines.append("")
    lines.append(
        f"This diagnostic reports top-label ECE using {args.n_bins} {args.binning} bins. "
        "For each bin, the absolute difference between mean top-label confidence and "
        "empirical top-1 accuracy is weighted by the bin fraction. This is not a full "
        "multiclass ECE and does not prove class-wise calibration."
    )
    lines.append("")
    lines.append("## Global top-label calibration")
    lines.append("")
    lines.append(md_table(pd.DataFrame([global_summary]), ["n", "accuracy", "mean_confidence", "ece", "mce", "n_bins", "binning"]))
    lines.append("")
    lines.append("## Family-wise top-label calibration")
    lines.append("")
    lines.append(md_table(family_df, ["family", "n", "accuracy", "mean_confidence", "ece", "mce"]))
    lines.append("")
    lines.append("## Compact interpretation")
    lines.append("")
    if not family_df.empty:
        worst = family_df.iloc[0]
        lines.append(
            f"The largest family-wise top-label ECE is observed for `{worst['family']}` "
            f"(ECE={worst['ece']:.4f}, n={int(worst['n'])}). "
            "Use this diagnostic to state whether the low aggregate ECE masks family-level heterogeneity."
        )
    lines.append("")

    summary_path = args.output_dir / "family_toplabel_calibration_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    if args.final_summary is not None:
        original = args.final_summary.read_text(encoding="utf-8")
        block = []
        block.append("## Family-wise top-label calibration diagnostic")
        block.append("")
        block.append(
            f"Top-label ECE was computed with {args.n_bins} {args.binning} bins using "
            "top-label confidence and empirical top-1 correctness. This diagnostic is "
            "global/family-wise and should not be interpreted as full class-wise or multiclass calibration."
        )
        block.append("")
        block.append("Global:")
        block.append("")
        block.append(md_table(pd.DataFrame([global_summary]), ["n", "accuracy", "mean_confidence", "ece", "mce"]))
        block.append("")
        block.append("By family:")
        block.append("")
        block.append(md_table(family_df, ["family", "n", "accuracy", "mean_confidence", "ece", "mce"]))
        block.append("")
        updated = original
        if "## Family-wise top-label calibration diagnostic" not in updated:
            updated += "\n\n" + "\n".join(block) + "\n"
        (args.output_dir / "final_publication_summary_with_family_calibration.md").write_text(updated, encoding="utf-8")

    print("[DONE] Family-wise top-label calibration diagnostics completed.", flush=True)
    print(f"[SAVED] {args.output_dir / 'global_toplabel_calibration.csv'}", flush=True)
    print(f"[SAVED] {args.output_dir / 'family_toplabel_calibration.csv'}", flush=True)
    print(f"[SAVED] {summary_path}", flush=True)
    if args.final_summary:
        print(f"[SAVED] {args.output_dir / 'final_publication_summary_with_family_calibration.md'}", flush=True)


if __name__ == "__main__":
    main()
