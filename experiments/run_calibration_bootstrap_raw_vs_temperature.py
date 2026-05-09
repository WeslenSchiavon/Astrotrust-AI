#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_calibration_bootstrap_raw_vs_temperature.py

Bootstrap pareado para comparar predições raw vs temperature-scaled.

Objetivo:
- fortalecer a afirmação de que AstroTrust-AI é calibrado;
- testar se temperature scaling reduz ECE, Brier e NLL de forma robusta;
- gerar CSVs, figuras e relatório Markdown.

O script NÃO treina modelos. Ele apenas lê arquivos object-level.

Uso típico:
python ./experiments/run_calibration_bootstrap_raw_vs_temperature.py \
  --raw-predictions results/final_publication/temperature_scaled_hybrid_250k/hybrid_temporal_tabular_cnn_raw_test_predictions.csv \
  --calibrated-predictions results/final_publication/temperature_scaled_hybrid_250k/hybrid_temporal_tabular_cnn_temperature_scaled_test_predictions.csv \
  --output-dir results/calibration_bootstrap_raw_vs_temperature_250k_final \
  --n-bootstrap 1000

Se os nomes dos arquivos forem diferentes, primeiro localize com:
Get-ChildItem -Recurse -File results -Include "*raw*test*prediction*.csv","*temperature*scaled*test*prediction*.csv","*calibrat*prediction*.csv" | Select-Object FullName
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRUE_CANDIDATES = ["true_label", "y_true", "true_class", "target", "label", "class"]
PRED_CANDIDATES = ["predicted_label", "pred_label", "y_pred", "prediction", "predicted_class", "top1_class"]
ID_CANDIDATES = ["object_id", "diaobject_id", "objectId", "id", "snid"]
CONF_CANDIDATES = ["confidence", "top1_probability", "top1_prob", "max_probability", "max_prob", "probability", "score"]


PROB_PREFIXES = [
    "prob_", "proba_", "probability_", "p_", "pclass_", "class_prob_",
    "class_probability_", "pred_proba_", "softmax_", "logitprob_"
]


LOWER_IS_BETTER = {
    "ece": True,
    "brier_score": True,
    "negative_log_likelihood": True,
    "calibration_gap_abs": True,
    "mean_confidence": False,
    "accuracy": False,
    "macro_f1": False,
    "weighted_f1": False,
}


def find_col(df: pd.DataFrame, explicit: Optional[str], candidates: List[str], required: bool = True) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit column `{explicit}` not found. Available columns: {list(df.columns)}")
        return explicit
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Could not infer column. Candidates={candidates}. Available={list(df.columns)}")
    return None


def normalize_label(x) -> str:
    if pd.isna(x):
        return "nan"
    s = str(x).strip()
    # Normalize numeric labels like 1.0 -> 1
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def parse_prob_label(col: str) -> str:
    s = str(col)
    low = s.lower()
    for pref in PROB_PREFIXES:
        if low.startswith(pref):
            return s[len(pref):]
    # Common forms: prob[12], p(12), class_12_probability
    m = re.search(r"(\d+)$", s)
    if m:
        return m.group(1)
    return s


def detect_probability_columns(
    df: pd.DataFrame,
    true_values: Optional[pd.Series] = None,
    prob_cols_regex: Optional[str] = None,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    exclude = set(exclude or [])
    if prob_cols_regex:
        pat = re.compile(prob_cols_regex)
        cols = [c for c in df.columns if pat.search(c)]
        if cols:
            return cols

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        low = c.lower()
        if any(low.startswith(p) for p in PROB_PREFIXES):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)

    # If no prefixed probability columns, try columns whose names match true labels.
    if not cols and true_values is not None:
        labels = set(normalize_label(v) for v in true_values.dropna().unique())
        for c in df.columns:
            if c in exclude:
                continue
            if normalize_label(c) in labels and pd.api.types.is_numeric_dtype(df[c]):
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.between(-1e-8, 1 + 1e-8).mean() > 0.95:
                    cols.append(c)

    return cols


def align_pair(
    raw: pd.DataFrame,
    cal: pd.DataFrame,
    id_col_raw: Optional[str],
    id_col_cal: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if id_col_raw and id_col_cal:
        raw2 = raw.copy()
        cal2 = cal.copy()
        raw2[id_col_raw] = raw2[id_col_raw].astype(str)
        cal2[id_col_cal] = cal2[id_col_cal].astype(str)
        common = sorted(set(raw2[id_col_raw]) & set(cal2[id_col_cal]))
        if not common:
            raise ValueError("No common IDs between raw and calibrated files.")
        raw2 = raw2.drop_duplicates(id_col_raw).set_index(id_col_raw).loc[common].reset_index()
        cal2 = cal2.drop_duplicates(id_col_cal).set_index(id_col_cal).loc[common].reset_index()
        return raw2, cal2

    n = min(len(raw), len(cal))
    if len(raw) != len(cal):
        print(f"[WARN] No ID column used; aligning by row order and truncating to {n} rows.")
    return raw.iloc[:n].reset_index(drop=True), cal.iloc[:n].reset_index(drop=True)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    weights = []
    for lab in labels:
        tp = np.sum((y_true == lab) & (y_pred == lab))
        fp = np.sum((y_true != lab) & (y_pred == lab))
        fn = np.sum((y_true == lab) & (y_pred != lab))
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp / denom)
        support = np.sum(y_true == lab)
        f1s.append(f1)
        weights.append(support)
    macro = float(np.mean(f1s)) if f1s else float("nan")
    weighted = float(np.average(f1s, weights=weights)) if np.sum(weights) > 0 else float("nan")
    return macro, weighted


def ece_score(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, n_bins: int = 15) -> float:
    correct = (y_true == y_pred).astype(float)
    conf = np.asarray(conf, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def reliability_bins(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    correct = (y_true == y_pred).astype(float)
    conf = np.asarray(conf, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            rows.append({
                "bin": i, "bin_low": lo, "bin_high": hi, "n": 0,
                "accuracy": np.nan, "mean_confidence": np.nan, "gap": np.nan
            })
        else:
            rows.append({
                "bin": i, "bin_low": lo, "bin_high": hi, "n": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "mean_confidence": float(conf[mask].mean()),
                "gap": float(abs(correct[mask].mean() - conf[mask].mean())),
            })
    return pd.DataFrame(rows)


def probability_matrix(df: pd.DataFrame, prob_cols: List[str]) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    if not prob_cols:
        return None, {}
    probs = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # Clip and renormalize if close but not exact.
    probs = np.clip(probs, 1e-15, 1.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(row_sum <= 0, 1.0, row_sum)
    label_to_idx = {normalize_label(parse_prob_label(c)): i for i, c in enumerate(prob_cols)}
    return probs, label_to_idx


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, label_to_idx: Dict[str, int]) -> float:
    n, k = probs.shape
    y_one = np.zeros((n, k), dtype=float)
    valid = np.zeros(n, dtype=bool)
    for i, lab in enumerate(y_true):
        idx = label_to_idx.get(normalize_label(lab))
        if idx is not None:
            y_one[i, idx] = 1.0
            valid[i] = True
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.sum((probs[valid] - y_one[valid]) ** 2, axis=1)))


def nll_score(y_true: np.ndarray, probs: np.ndarray, label_to_idx: Dict[str, int]) -> float:
    vals = []
    for i, lab in enumerate(y_true):
        idx = label_to_idx.get(normalize_label(lab))
        if idx is not None:
            vals.append(-np.log(max(float(probs[i, idx]), 1e-15)))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def top_conf_from_probs(probs: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if probs is None:
        return None
    return probs.max(axis=1)


def pred_from_probs(probs: Optional[np.ndarray], label_to_idx: Dict[str, int]) -> Optional[np.ndarray]:
    if probs is None:
        return None
    inv = {v: k for k, v in label_to_idx.items()}
    idx = probs.argmax(axis=1)
    return np.array([inv.get(int(i), str(i)) for i in idx], dtype=object)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    label_to_idx: Dict[str, int],
    n_bins: int,
) -> Dict[str, float]:
    out = {}

    if probs is not None:
        prob_conf = top_conf_from_probs(probs)
        prob_pred = pred_from_probs(probs, label_to_idx)
        if conf is None:
            conf = prob_conf
        if y_pred is None:
            y_pred = prob_pred

    if y_pred is not None:
        out["accuracy"] = accuracy(y_true, y_pred)
        macro, weighted = f1_scores(y_true, y_pred)
        out["macro_f1"] = macro
        out["weighted_f1"] = weighted

    if conf is not None and y_pred is not None:
        out["mean_confidence"] = float(np.mean(conf))
        out["calibration_gap_abs"] = float(abs(np.mean(conf) - accuracy(y_true, y_pred)))
        out["ece"] = ece_score(y_true, y_pred, conf, n_bins=n_bins)

    if probs is not None:
        out["brier_score"] = multiclass_brier(y_true, probs, label_to_idx)
        out["negative_log_likelihood"] = nll_score(y_true, probs, label_to_idx)

    return out


def bootstrap_compare(
    y_true: np.ndarray,
    raw_y_pred: Optional[np.ndarray],
    cal_y_pred: Optional[np.ndarray],
    raw_conf: Optional[np.ndarray],
    cal_conf: Optional[np.ndarray],
    raw_probs: Optional[np.ndarray],
    cal_probs: Optional[np.ndarray],
    raw_label_to_idx: Dict[str, int],
    cal_label_to_idx: Dict[str, int],
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx_all = np.arange(n)

    raw_point = compute_metrics(y_true, raw_y_pred, raw_conf, raw_probs, raw_label_to_idx, n_bins)
    cal_point = compute_metrics(y_true, cal_y_pred, cal_conf, cal_probs, cal_label_to_idx, n_bins)
    metrics = sorted(set(raw_point.keys()) & set(cal_point.keys()))

    boot = {m: [] for m in metrics}
    for _ in range(n_bootstrap):
        idx = rng.choice(idx_all, size=n, replace=True)

        rb = compute_metrics(
            y_true[idx],
            raw_y_pred[idx] if raw_y_pred is not None else None,
            raw_conf[idx] if raw_conf is not None else None,
            raw_probs[idx] if raw_probs is not None else None,
            raw_label_to_idx,
            n_bins,
        )
        cb = compute_metrics(
            y_true[idx],
            cal_y_pred[idx] if cal_y_pred is not None else None,
            cal_conf[idx] if cal_conf is not None else None,
            cal_probs[idx] if cal_probs is not None else None,
            cal_label_to_idx,
            n_bins,
        )
        for m in metrics:
            boot[m].append(cb[m] - rb[m])

    rows = []
    for m in metrics:
        samples = np.array(boot[m], dtype=float)
        samples = samples[np.isfinite(samples)]
        point_delta = cal_point[m] - raw_point[m]
        ci_low = float(np.quantile(samples, 0.025)) if len(samples) else float("nan")
        ci_high = float(np.quantile(samples, 0.975)) if len(samples) else float("nan")
        lower_better = LOWER_IS_BETTER.get(m, False)

        if lower_better:
            p_improve = float(np.mean(samples >= 0)) if len(samples) else float("nan")  # calibrated not lower
            direction = "lower_is_better"
        else:
            p_improve = float(np.mean(samples <= 0)) if len(samples) else float("nan")  # calibrated not higher
            direction = "higher_is_better"

        p_two_sided = float(min(1.0, 2.0 * min(np.mean(samples <= 0), np.mean(samples >= 0)))) if len(samples) else float("nan")

        rows.append({
            "metric": m,
            "raw": raw_point[m],
            "temperature_scaled": cal_point[m],
            "delta_temperature_minus_raw": point_delta,
            "bootstrap_ci_low_95": ci_low,
            "bootstrap_ci_high_95": ci_high,
            "direction": direction,
            "one_sided_bootstrap_pvalue_no_improvement": p_improve,
            "two_sided_bootstrap_pvalue": p_two_sided,
            "n_bootstrap": n_bootstrap,
        })

    return pd.DataFrame(rows)


def plot_reliability(raw_bins: pd.DataFrame, cal_bins: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(raw_bins["mean_confidence"], raw_bins["accuracy"], marker="o", label="Raw")
    plt.plot(cal_bins["mean_confidence"], cal_bins["accuracy"], marker="o", label="Temperature-scaled")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_deltas(results: pd.DataFrame, out_path: Path) -> None:
    metrics = results["metric"].tolist()
    x = np.arange(len(metrics))
    y = results["delta_temperature_minus_raw"].to_numpy()
    lo = results["bootstrap_ci_low_95"].to_numpy()
    hi = results["bootstrap_ci_high_95"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])
    plt.figure(figsize=(10, 5))
    plt.axhline(0, linestyle="--")
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, metrics, rotation=45, ha="right")
    plt.ylabel("Temperature-scaled minus raw")
    plt.title("Paired bootstrap calibration deltas")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--calibrated-predictions", required=True)
    parser.add_argument("--output-dir", default="results/calibration_bootstrap_raw_vs_temperature_250k_final")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--raw-id-col", default=None)
    parser.add_argument("--calibrated-id-col", default=None)
    parser.add_argument("--true-col", default=None)
    parser.add_argument("--raw-true-col", default=None)
    parser.add_argument("--calibrated-true-col", default=None)
    parser.add_argument("--raw-pred-col", default=None)
    parser.add_argument("--calibrated-pred-col", default=None)
    parser.add_argument("--raw-confidence-col", default=None)
    parser.add_argument("--calibrated-confidence-col", default=None)
    parser.add_argument("--raw-prob-cols-regex", default=None)
    parser.add_argument("--calibrated-prob-cols-regex", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_path = Path(args.raw_predictions)
    cal_path = Path(args.calibrated_predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    cal = pd.read_csv(cal_path)

    raw_id_col = args.raw_id_col or args.id_col
    cal_id_col = args.calibrated_id_col or args.id_col

    if raw_id_col is None:
        raw_id_col = find_col(raw, None, ID_CANDIDATES, required=False)
    if cal_id_col is None:
        cal_id_col = find_col(cal, None, ID_CANDIDATES, required=False)

    raw, cal = align_pair(raw, cal, raw_id_col, cal_id_col)

    raw_true_col = args.raw_true_col or args.true_col
    cal_true_col = args.calibrated_true_col or args.true_col
    raw_true_col = find_col(raw, raw_true_col, TRUE_CANDIDATES, required=True)
    cal_true_col = find_col(cal, cal_true_col, TRUE_CANDIDATES, required=True)

    raw_y_true = np.array([normalize_label(v) for v in raw[raw_true_col].to_numpy()], dtype=object)
    cal_y_true = np.array([normalize_label(v) for v in cal[cal_true_col].to_numpy()], dtype=object)

    if not np.array_equal(raw_y_true, cal_y_true):
        raise ValueError("Raw and calibrated true labels are not aligned. Check ID columns or file ordering.")

    y_true = raw_y_true

    raw_pred_col = find_col(raw, args.raw_pred_col, PRED_CANDIDATES, required=False)
    cal_pred_col = find_col(cal, args.calibrated_pred_col, PRED_CANDIDATES, required=False)

    raw_y_pred = None if raw_pred_col is None else np.array([normalize_label(v) for v in raw[raw_pred_col].to_numpy()], dtype=object)
    cal_y_pred = None if cal_pred_col is None else np.array([normalize_label(v) for v in cal[cal_pred_col].to_numpy()], dtype=object)

    raw_conf_col = find_col(raw, args.raw_confidence_col, CONF_CANDIDATES, required=False)
    cal_conf_col = find_col(cal, args.calibrated_confidence_col, CONF_CANDIDATES, required=False)

    raw_conf = None if raw_conf_col is None else pd.to_numeric(raw[raw_conf_col], errors="coerce").to_numpy(dtype=float)
    cal_conf = None if cal_conf_col is None else pd.to_numeric(cal[cal_conf_col], errors="coerce").to_numpy(dtype=float)

    exclude_raw = [c for c in [raw_id_col, raw_true_col, raw_pred_col, raw_conf_col] if c]
    exclude_cal = [c for c in [cal_id_col, cal_true_col, cal_pred_col, cal_conf_col] if c]

    raw_prob_cols = detect_probability_columns(raw, raw[raw_true_col], args.raw_prob_cols_regex, exclude_raw)
    cal_prob_cols = detect_probability_columns(cal, cal[cal_true_col], args.calibrated_prob_cols_regex, exclude_cal)

    raw_probs, raw_label_to_idx = probability_matrix(raw, raw_prob_cols)
    cal_probs, cal_label_to_idx = probability_matrix(cal, cal_prob_cols)

    if raw_conf is None and raw_probs is None:
        raise ValueError("Raw file has neither confidence column nor probability columns.")
    if cal_conf is None and cal_probs is None:
        raise ValueError("Calibrated file has neither confidence column nor probability columns.")

    results = bootstrap_compare(
        y_true=y_true,
        raw_y_pred=raw_y_pred,
        cal_y_pred=cal_y_pred,
        raw_conf=raw_conf,
        cal_conf=cal_conf,
        raw_probs=raw_probs,
        cal_probs=cal_probs,
        raw_label_to_idx=raw_label_to_idx,
        cal_label_to_idx=cal_label_to_idx,
        n_bins=args.n_bins,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    results_path = out_dir / "calibration_bootstrap_raw_vs_temperature.csv"
    results.to_csv(results_path, index=False)

    # Reliability bins if possible.
    raw_metrics = compute_metrics(y_true, raw_y_pred, raw_conf, raw_probs, raw_label_to_idx, args.n_bins)
    cal_metrics = compute_metrics(y_true, cal_y_pred, cal_conf, cal_probs, cal_label_to_idx, args.n_bins)

    reliability_written = False
    if "ece" in raw_metrics and "ece" in cal_metrics:
        if raw_conf is None and raw_probs is not None:
            raw_conf_plot = top_conf_from_probs(raw_probs)
        else:
            raw_conf_plot = raw_conf
        if cal_conf is None and cal_probs is not None:
            cal_conf_plot = top_conf_from_probs(cal_probs)
        else:
            cal_conf_plot = cal_conf

        raw_pred_plot = raw_y_pred if raw_y_pred is not None else pred_from_probs(raw_probs, raw_label_to_idx)
        cal_pred_plot = cal_y_pred if cal_y_pred is not None else pred_from_probs(cal_probs, cal_label_to_idx)

        raw_bins = reliability_bins(y_true, raw_pred_plot, raw_conf_plot, args.n_bins)
        cal_bins = reliability_bins(y_true, cal_pred_plot, cal_conf_plot, args.n_bins)
        raw_bins["model"] = "raw"
        cal_bins["model"] = "temperature_scaled"
        bins = pd.concat([raw_bins, cal_bins], ignore_index=True)
        bins.to_csv(out_dir / "calibration_reliability_bins.csv", index=False)
        plot_reliability(raw_bins, cal_bins, out_dir / "fig_calibration_reliability_raw_vs_temperature.png")
        reliability_written = True

    plot_metric_deltas(results, out_dir / "fig_calibration_metric_deltas.png")

    schema = {
        "raw_predictions": str(raw_path),
        "calibrated_predictions": str(cal_path),
        "n_objects": int(len(y_true)),
        "raw_id_col": raw_id_col,
        "calibrated_id_col": cal_id_col,
        "raw_true_col": raw_true_col,
        "calibrated_true_col": cal_true_col,
        "raw_pred_col": raw_pred_col,
        "calibrated_pred_col": cal_pred_col,
        "raw_confidence_col": raw_conf_col,
        "calibrated_confidence_col": cal_conf_col,
        "raw_probability_columns": raw_prob_cols,
        "calibrated_probability_columns": cal_prob_cols,
        "n_bootstrap": args.n_bootstrap,
        "n_bins": args.n_bins,
        "seed": args.seed,
    }
    (out_dir / "calibration_bootstrap_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Markdown
    def row_for(metric: str) -> Optional[pd.Series]:
        sub = results[results["metric"] == metric]
        return None if sub.empty else sub.iloc[0]

    md = []
    md.append("# Calibration bootstrap: raw vs temperature-scaled\n")
    md.append("This report compares raw and temperature-scaled predictions using paired bootstrap resampling.\n")
    md.append("## Inputs\n")
    md.append(f"- Raw predictions: `{raw_path}`")
    md.append(f"- Temperature-scaled predictions: `{cal_path}`")
    md.append(f"- Number of aligned objects: `{len(y_true)}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`")
    md.append(f"- ECE bins: `{args.n_bins}`")
    md.append(f"- Raw confidence column: `{raw_conf_col}`")
    md.append(f"- Calibrated confidence column: `{cal_conf_col}`")
    md.append(f"- Raw probability columns detected: `{len(raw_prob_cols)}`")
    md.append(f"- Calibrated probability columns detected: `{len(cal_prob_cols)}`\n")

    md.append("## Main results\n")
    md.append(results.to_markdown(index=False))
    md.append("")

    md.append("## Key interpretation\n")
    ece = row_for("ece")
    brier = row_for("brier_score")
    nll = row_for("negative_log_likelihood")
    gap = row_for("calibration_gap_abs")

    if ece is not None:
        md.append(
            f"- ECE changed from `{ece['raw']:.6g}` to `{ece['temperature_scaled']:.6g}` "
            f"(delta = `{ece['delta_temperature_minus_raw']:.6g}`, 95% CI "
            f"[`{ece['bootstrap_ci_low_95']:.6g}`, `{ece['bootstrap_ci_high_95']:.6g}`])."
        )
    if brier is not None:
        md.append(
            f"- Brier score changed from `{brier['raw']:.6g}` to `{brier['temperature_scaled']:.6g}` "
            f"(delta = `{brier['delta_temperature_minus_raw']:.6g}`, 95% CI "
            f"[`{brier['bootstrap_ci_low_95']:.6g}`, `{brier['bootstrap_ci_high_95']:.6g}`])."
        )
    if nll is not None:
        md.append(
            f"- NLL changed from `{nll['raw']:.6g}` to `{nll['temperature_scaled']:.6g}` "
            f"(delta = `{nll['delta_temperature_minus_raw']:.6g}`, 95% CI "
            f"[`{nll['bootstrap_ci_low_95']:.6g}`, `{nll['bootstrap_ci_high_95']:.6g}`])."
        )
    if gap is not None:
        md.append(
            f"- Absolute calibration gap changed from `{gap['raw']:.6g}` to `{gap['temperature_scaled']:.6g}` "
            f"(delta = `{gap['delta_temperature_minus_raw']:.6g}`)."
        )
    md.append("")

    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    if ece is not None:
        text = (
            "We assessed the statistical effect of temperature scaling using paired bootstrap resampling over the held-out test objects. "
            f"Temperature scaling reduced the expected calibration error from \\textbf{{{ece['raw']:.4f}}} to \\textbf{{{ece['temperature_scaled']:.4f}}} "
            f"(bootstrap delta = {ece['delta_temperature_minus_raw']:.4f}; 95\\% CI [{ece['bootstrap_ci_low_95']:.4f}, {ece['bootstrap_ci_high_95']:.4f}])."
        )
        if brier is not None:
            text += (
                f" It also changed the Brier score from \\textbf{{{brier['raw']:.4f}}} to \\textbf{{{brier['temperature_scaled']:.4f}}} "
                f"(delta = {brier['delta_temperature_minus_raw']:.4f}; 95\\% CI [{brier['bootstrap_ci_low_95']:.4f}, {brier['bootstrap_ci_high_95']:.4f}])."
            )
        md.append(text)
    else:
        md.append("Temperature scaling was evaluated using paired bootstrap resampling over the held-out test objects.")
    md.append("```\n")

    md.append("## Output files\n")
    for p in [
        results_path,
        out_dir / "calibration_reliability_bins.csv",
        out_dir / "calibration_bootstrap_schema.json",
        out_dir / "fig_calibration_reliability_raw_vs_temperature.png",
        out_dir / "fig_calibration_metric_deltas.png",
    ]:
        if p.exists():
            md.append(f"- `{p}`")

    md_path = out_dir / "calibration_bootstrap_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")
    print(f"Summary: {md_path}")
    if not raw_prob_cols or not cal_prob_cols:
        print("[WARN] Probability columns were not detected in one or both files. Brier/NLL may be unavailable.")
    if not reliability_written:
        print("[WARN] Reliability bins/figure not written because ECE inputs were unavailable.")


if __name__ == "__main__":
    main()
