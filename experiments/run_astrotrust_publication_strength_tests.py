#!/usr/bin/env python3
"""
AstroTrust-AI publication-strength statistical analyses.

This script adds five publication-oriented analyses around the final AstroTrust-AI
predictions:

1) Paired seed-level tests: ensemble_hybrid_dominant vs hybrid_temporal_tabular_cnn.
2) Per-class and per-family bootstrap confidence intervals.
3) Permutation test for rare-class enrichment under a follow-up ranking policy.
4) Risk-coverage / selective-classification curves.
5) AUROC/AUPRC for error detection using uncertainty.

The script is intentionally robust to slightly different CSV schemas. Use explicit
--true-col, --pred-col, --score-col, --confidence-col, --uncertainty-col, etc. when needed.

Example:
python experiments/run_astrotrust_publication_strength_tests.py \
  --ensemble-predictions results/hybrid_tabular_ensemble_250k/ensemble_hybrid_dominant_predictions.csv \
  --hybrid-predictions results/hybrid_temporal_tabular_cnn_250k/hybrid_temporal_tabular_cnn_test_predictions.csv \
  --seed-summary results/multi_seed_stability/per_seed_metrics.csv \
  --label-map data/processed/elasticc2_large/full_class_counts.csv \
  --output-dir results/publication_strength_tests \
  --n-bootstrap 5000 \
  --n-permutations 10000 \
  --infer-family-from-class-name
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc

import matplotlib.pyplot as plt


TRUE_CANDIDATES = [
    "true_label", "true_class", "y_true", "target", "label", "class", "class_true",
    "true", "truth", "sim_type", "type",
]
PRED_CANDIDATES = [
    "pred_label", "pred_class", "y_pred", "prediction", "predicted_class", "top1_class",
    "pred", "predicted_label",
]
ID_CANDIDATES = [
    "object_id", "diaobject_id", "diaObjectId", "objectId", "id", "snid", "source_id",
]
CONFIDENCE_CANDIDATES = ["confidence", "top1_probability", "max_probability", "max_prob", "prob_top1"]
UNCERTAINTY_CANDIDATES = ["uncertainty", "predictive_uncertainty", "entropy", "error_score"]
SCORE_CANDIDATES = ["priority_score", "score", "followup_score", "rank_score", "novelty_rarity_score"]
RARE_CANDIDATES = ["is_true_rare", "true_rare", "is_rare", "rare", "rare_true"]
FAMILY_CANDIDATES = ["family", "class_family", "coarse_family", "family_name", "coarse_label"]
CLASS_MAP_CANDIDATES = ["class", "class_name", "label", "target", "true_label", "original_class", "type"]
COUNT_CANDIDATES = ["count", "n", "n_objects", "n_samples", "class_count", "full_count", "num_objects"]


@dataclass
class ColumnSpec:
    true_col: str
    pred_col: str
    id_col: Optional[str]
    confidence_col: Optional[str]
    uncertainty_col: Optional[str]
    score_col: Optional[str]
    rare_col: Optional[str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_column(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str], required: bool = False, label: str = "column") -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested {label} '{explicit}' was not found. Available columns: {list(df.columns)[:40]}")
        return explicit
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Could not infer {label}. Provide it explicitly. Available columns: {list(df.columns)[:80]}")
    return None


def detect_probability_columns(df: pd.DataFrame, explicit_cols: Optional[str], prefixes: Sequence[str], exclude: Sequence[Optional[str]]) -> List[str]:
    if explicit_cols:
        cols = [c.strip() for c in explicit_cols.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Probability columns not found: {missing}")
        return cols

    exclude_set = {c for c in exclude if c}
    prefixes = [p for p in prefixes if p]
    prob_cols: List[str] = []
    for col in df.columns:
        if col in exclude_set:
            continue
        if any(str(col).startswith(p) for p in prefixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                prob_cols.append(col)

    # Fallback: common class-probability pattern, excluding obvious scalar metric columns.
    if not prob_cols:
        scalar_like = set(
            [
                "confidence", "uncertainty", "priority_score", "novelty", "rarity",
                "matched_fraction", "time_span", "n_rows", "n_bands", "accuracy",
                "macro_f1", "weighted_f1", "balanced_accuracy", "ece", "brier_score", "nll",
            ]
        )
        numeric_cols = [
            c for c in df.columns
            if c not in exclude_set
            and pd.api.types.is_numeric_dtype(df[c])
            and str(c).lower() not in scalar_like
        ]
        # Use this fallback only when it looks like a probability matrix.
        if len(numeric_cols) >= 3:
            vals = df[numeric_cols].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.any():
                vmin = np.nanmin(vals)
                vmax = np.nanmax(vals)
                row_sums = np.nansum(vals, axis=1)
                if vmin >= -1e-6 and vmax <= 1.0 + 1e-6 and np.nanmedian(row_sums) > 0.5:
                    prob_cols = numeric_cols

    return prob_cols


def probability_label_from_col(col: str, prefixes: Sequence[str]) -> str:
    name = str(col)
    for p in prefixes:
        if p and name.startswith(p):
            return name[len(p):]
    return name


def add_confidence_uncertainty(df: pd.DataFrame, spec: ColumnSpec, prob_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    if spec.confidence_col:
        out["_confidence"] = pd.to_numeric(out[spec.confidence_col], errors="coerce")
    elif prob_cols:
        out["_confidence"] = np.nanmax(out[prob_cols].to_numpy(dtype=float), axis=1)
    else:
        raise ValueError("No confidence column and no probability columns were detected. Provide --confidence-col or --prob-cols.")

    if spec.uncertainty_col:
        out["_uncertainty"] = pd.to_numeric(out[spec.uncertainty_col], errors="coerce")
    else:
        out["_uncertainty"] = 1.0 - out["_confidence"].astype(float)

    return out


def infer_family_from_class_name(class_name: str) -> str:
    """Heuristic family mapping for ELAsTiCC-like labels."""
    s = normalize_name(class_name)
    if not s:
        return "Unknown"
    low = s.lower()
    if low.startswith("sn") or "supernova" in low:
        if "ia" in low:
            return "SNIa-like"
        return "SN-like"
    if low.startswith("kn") or "kilonova" in low:
        return "Kilonova"
    if "tde" in low:
        return "TDE"
    if "agn" in low:
        return "AGN"
    if "lens" in low or "ulens" in low or "microlens" in low:
        return "Microlensing"
    if "cart" in low:
        return "CART"
    if "dwarf" in low or "nova" in low:
        return "Variable/Nova"
    if "rrlyr" in low or "cepheid" in low or "periodic" in low:
        return "Periodic variable"
    if "eb" == low or "eclips" in low:
        return "Eclipsing binary"
    return re.split(r"[-_ ]", s)[0]


def load_label_maps(
    label_map_path: Optional[Path],
    family_col: Optional[str],
    class_col: Optional[str],
    rare_col: Optional[str],
    rare_count_threshold: Optional[int],
    rare_quantile: float,
    infer_family: bool,
) -> Tuple[Dict[str, str], Dict[str, bool], Optional[pd.DataFrame]]:
    family_map: Dict[str, str] = {}
    rare_map: Dict[str, bool] = {}
    label_df: Optional[pd.DataFrame] = None

    if label_map_path and label_map_path.exists():
        label_df = pd.read_csv(label_map_path)
        c_col = find_column(label_df, class_col, CLASS_MAP_CANDIDATES, required=True, label="class column in label map")
        f_col = find_column(label_df, family_col, FAMILY_CANDIDATES, required=False, label="family column in label map")
        if f_col:
            for _, row in label_df.iterrows():
                family_map[normalize_name(row[c_col])] = normalize_name(row[f_col])
        elif infer_family:
            for v in label_df[c_col].dropna().unique():
                family_map[normalize_name(v)] = infer_family_from_class_name(v)

        r_col = find_column(label_df, rare_col, RARE_CANDIDATES, required=False, label="rare column in label map") if rare_col else find_column(label_df, None, RARE_CANDIDATES, required=False, label="rare column in label map")
        if r_col:
            for _, row in label_df.iterrows():
                val = row[r_col]
                if isinstance(val, str):
                    is_rare = val.strip().lower() in {"1", "true", "yes", "rare", "y"}
                else:
                    is_rare = bool(val)
                rare_map[normalize_name(row[c_col])] = is_rare
        else:
            count_col = find_column(label_df, None, COUNT_CANDIDATES, required=False, label="count column in label map")
            if count_col:
                counts = pd.to_numeric(label_df[count_col], errors="coerce")
                threshold = rare_count_threshold
                if threshold is None:
                    threshold = float(np.nanquantile(counts, rare_quantile))
                for _, row in label_df.iterrows():
                    rare_map[normalize_name(row[c_col])] = float(row[count_col]) <= float(threshold)

    return family_map, rare_map, label_df


def basic_metrics(y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    y_true = np.asarray([normalize_name(x) for x in y_true])
    y_pred = np.asarray([normalize_name(x) for x in y_pred])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def paired_seed_tests(seed_summary_path: Path, output_dir: Path, ensemble_name: str, hybrid_name: str, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.read_csv(seed_summary_path)
    seed_col = find_column(df, None, ["seed", "random_seed", "run_seed"], required=True, label="seed column")
    model_col = find_column(df, None, ["model", "model_name", "name"], required=True, label="model column")

    metric_candidates = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy",
        "top5_accuracy", "ece", "brier_score", "negative_log_likelihood", "nll",
    ]
    rows = []
    for metric in metric_candidates:
        if metric not in df.columns:
            continue
        sub = df[df[model_col].isin([ensemble_name, hybrid_name])][[seed_col, model_col, metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        piv = sub.pivot_table(index=seed_col, columns=model_col, values=metric, aggfunc="first").dropna()
        if ensemble_name not in piv.columns or hybrid_name not in piv.columns or len(piv) < 2:
            continue
        diff = (piv[ensemble_name] - piv[hybrid_name]).to_numpy(dtype=float)
        boot_means = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_bootstrap)])
        ci_low, ci_high = percentile_ci(boot_means)
        t_p = np.nan
        w_p = np.nan
        sign_p = np.nan
        if stats is not None:
            try:
                t_p = float(stats.ttest_rel(piv[ensemble_name], piv[hybrid_name], nan_policy="omit").pvalue)
            except Exception:
                pass
            try:
                if np.any(diff != 0):
                    w_p = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception:
                pass
            try:
                n_pos = int(np.sum(diff > 0))
                n_nonzero = int(np.sum(diff != 0))
                if n_nonzero > 0:
                    # two-sided exact binomial sign test around p=0.5
                    if hasattr(stats, "binomtest"):
                        sign_p = float(stats.binomtest(k=n_pos, n=n_nonzero, p=0.5, alternative="two-sided").pvalue)
            except Exception:
                pass
        rows.append({
            "metric": metric,
            "n_paired_seeds": len(diff),
            "ensemble_mean": float(np.mean(piv[ensemble_name])),
            "hybrid_mean": float(np.mean(piv[hybrid_name])),
            "mean_delta_ensemble_minus_hybrid": float(np.mean(diff)),
            "sd_delta": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "paired_t_pvalue": t_p,
            "wilcoxon_pvalue": w_p,
            "sign_test_pvalue": sign_p,
            "n_positive_deltas": int(np.sum(diff > 0)),
            "n_negative_deltas": int(np.sum(diff < 0)),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "paired_seed_level_tests.csv", index=False)
    return out


def paired_prediction_tests(
    ensemble_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    ensemble_spec: ColumnSpec,
    hybrid_spec: ColumnSpec,
    output_dir: Path,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if ensemble_spec.id_col and hybrid_spec.id_col:
        e = ensemble_df[[ensemble_spec.id_col, ensemble_spec.true_col, ensemble_spec.pred_col]].copy()
        h = hybrid_df[[hybrid_spec.id_col, hybrid_spec.true_col, hybrid_spec.pred_col]].copy()
        e.columns = ["_id", "true", "ensemble_pred"]
        h.columns = ["_id", "hybrid_true", "hybrid_pred"]
        merged = e.merge(h, on="_id", how="inner")
        # Prefer the ensemble truth column but verify consistency where possible.
        if (merged["true"].astype(str) != merged["hybrid_true"].astype(str)).mean() > 0.01:
            print("Warning: more than 1% of merged rows have inconsistent true labels between files.")
    else:
        n = min(len(ensemble_df), len(hybrid_df))
        merged = pd.DataFrame({
            "true": ensemble_df[ensemble_spec.true_col].iloc[:n].to_numpy(),
            "ensemble_pred": ensemble_df[ensemble_spec.pred_col].iloc[:n].to_numpy(),
            "hybrid_pred": hybrid_df[hybrid_spec.pred_col].iloc[:n].to_numpy(),
        })

    y = merged["true"].map(normalize_name).to_numpy()
    e_pred = merged["ensemble_pred"].map(normalize_name).to_numpy()
    h_pred = merged["hybrid_pred"].map(normalize_name).to_numpy()
    e_correct = e_pred == y
    h_correct = h_pred == y

    rows = []
    for metric in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]:
        if metric == "accuracy":
            e_val = accuracy_score(y, e_pred)
            h_val = accuracy_score(y, h_pred)
        elif metric == "macro_f1":
            e_val = f1_score(y, e_pred, average="macro", zero_division=0)
            h_val = f1_score(y, h_pred, average="macro", zero_division=0)
        elif metric == "weighted_f1":
            e_val = f1_score(y, e_pred, average="weighted", zero_division=0)
            h_val = f1_score(y, h_pred, average="weighted", zero_division=0)
        else:
            e_val = balanced_accuracy_score(y, e_pred)
            h_val = balanced_accuracy_score(y, h_pred)

        boot = []
        n = len(y)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yy, ep, hp = y[idx], e_pred[idx], h_pred[idx]
            if metric == "accuracy":
                boot.append(accuracy_score(yy, ep) - accuracy_score(yy, hp))
            elif metric == "macro_f1":
                boot.append(f1_score(yy, ep, average="macro", zero_division=0) - f1_score(yy, hp, average="macro", zero_division=0))
            elif metric == "weighted_f1":
                boot.append(f1_score(yy, ep, average="weighted", zero_division=0) - f1_score(yy, hp, average="weighted", zero_division=0))
            else:
                boot.append(balanced_accuracy_score(yy, ep) - balanced_accuracy_score(yy, hp))
        ci_low, ci_high = percentile_ci(np.asarray(boot))
        p_boot = float((1 + np.sum(np.asarray(boot) <= 0)) / (len(boot) + 1)) if e_val >= h_val else float((1 + np.sum(np.asarray(boot) >= 0)) / (len(boot) + 1))
        rows.append({
            "metric": metric,
            "n_objects": len(y),
            "ensemble": float(e_val),
            "hybrid": float(h_val),
            "delta_ensemble_minus_hybrid": float(e_val - h_val),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "one_sided_bootstrap_pvalue_against_zero": p_boot,
        })

    # McNemar test for paired correctness.
    b = int(np.sum(e_correct & ~h_correct))  # ensemble right, hybrid wrong
    c = int(np.sum(~e_correct & h_correct))  # ensemble wrong, hybrid right
    mcnemar_p = np.nan
    if stats is not None and (b + c) > 0:
        # Exact two-sided binomial McNemar.
        if hasattr(stats, "binomtest"):
            mcnemar_p = float(stats.binomtest(k=min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)
    rows.append({
        "metric": "mcnemar_accuracy_correctness",
        "n_objects": len(y),
        "ensemble": float(np.mean(e_correct)),
        "hybrid": float(np.mean(h_correct)),
        "delta_ensemble_minus_hybrid": float(np.mean(e_correct) - np.mean(h_correct)),
        "bootstrap_ci_low": np.nan,
        "bootstrap_ci_high": np.nan,
        "one_sided_bootstrap_pvalue_against_zero": np.nan,
        "mcnemar_b_ensemble_correct_hybrid_wrong": b,
        "mcnemar_c_ensemble_wrong_hybrid_correct": c,
        "mcnemar_exact_pvalue": mcnemar_p,
    })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "paired_prediction_level_tests.csv", index=False)
    return out


def bootstrap_per_group_ci(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    output_path: Path,
    n_bootstrap: int,
    rng: np.random.Generator,
    group_label: str = "class",
) -> pd.DataFrame:
    y_true = df[true_col].map(normalize_name).to_numpy()
    y_pred = df[pred_col].map(normalize_name).to_numpy()
    labels = sorted(set(y_true) | set(y_pred))

    # Point estimates.
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    point = {
        lab: {"precision": p[i], "recall": r[i], "f1": f[i], "support": support[i]}
        for i, lab in enumerate(labels)
    }

    boot_precision = {lab: [] for lab in labels}
    boot_recall = {lab: [] for lab in labels}
    boot_f1 = {lab: [] for lab in labels}

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        pp = y_pred[idx]
        bp, br, bf, _ = precision_recall_fscore_support(yy, pp, labels=labels, zero_division=0)
        for i, lab in enumerate(labels):
            boot_precision[lab].append(bp[i])
            boot_recall[lab].append(br[i])
            boot_f1[lab].append(bf[i])

    rows = []
    for lab in labels:
        for metric_name, boot_dict in [
            ("precision", boot_precision), ("recall", boot_recall), ("f1", boot_f1)
        ]:
            ci_low, ci_high = percentile_ci(np.asarray(boot_dict[lab]))
            rows.append({
                group_label: lab,
                "metric": metric_name,
                "point_estimate": float(point[lab][metric_name]),
                "bootstrap_mean": float(np.mean(boot_dict[lab])),
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "support": int(point[lab]["support"]),
            })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def add_family_columns(df: pd.DataFrame, true_col: str, pred_col: str, family_map: Dict[str, str], infer_family: bool) -> Optional[pd.DataFrame]:
    if not family_map and not infer_family:
        return None
    out = df.copy()

    def fam(x):
        key = normalize_name(x)
        if key in family_map:
            return family_map[key]
        return infer_family_from_class_name(key) if infer_family else "Unknown"

    out["_true_family"] = out[true_col].map(fam)
    out["_pred_family"] = out[pred_col].map(fam)
    return out


def define_true_rare(
    df: pd.DataFrame,
    true_col: str,
    rare_col: Optional[str],
    rare_map: Dict[str, bool],
    rare_classes: Optional[str],
) -> pd.Series:
    if rare_col and rare_col in df.columns:
        s = df[rare_col]
        if s.dtype == object:
            return s.astype(str).str.lower().isin(["1", "true", "yes", "rare", "y"])
        return s.astype(bool)

    explicit = set()
    if rare_classes:
        explicit = {normalize_name(x) for x in rare_classes.split(",") if normalize_name(x)}

    if explicit:
        return df[true_col].map(lambda x: normalize_name(x) in explicit)

    if rare_map:
        return df[true_col].map(lambda x: bool(rare_map.get(normalize_name(x), False)))

    raise ValueError(
        "Could not define true rare classes. Provide --rare-col in predictions, --rare-classes, "
        "or --label-map with a rare/count column."
    )


def permutation_rare_enrichment(
    df: pd.DataFrame,
    true_col: str,
    score_col: str,
    rare_col: Optional[str],
    rare_map: Dict[str, bool],
    rare_classes: Optional[str],
    output_dir: Path,
    n_permutations: int,
    rng: np.random.Generator,
    budgets: Sequence[float],
) -> pd.DataFrame:
    work = df.copy()
    work["_true_rare"] = define_true_rare(work, true_col, rare_col, rare_map, rare_classes).astype(bool)
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["_score"])
    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    rare = work["_true_rare"].to_numpy(dtype=bool)
    scores = work["_score"].to_numpy(dtype=float)
    baseline = float(np.mean(rare))
    if baseline <= 0:
        raise ValueError("No rare objects found. Check rare-class definition.")

    rows = []
    n = len(work)
    for budget in budgets:
        k = max(1, int(round(n * budget)))
        selected = np.arange(n)[:k]
        obs_rate = float(np.mean(rare[selected]))
        obs_enrichment = obs_rate / baseline
        null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            rare_perm = rng.permutation(rare)
            null[i] = float(np.mean(rare_perm[selected]) / baseline)
        p_right = float((1 + np.sum(null >= obs_enrichment)) / (n_permutations + 1))
        ci_low, ci_high = percentile_ci(null)
        rows.append({
            "budget_fraction": budget,
            "n_selected": k,
            "baseline_rare_rate": baseline,
            "observed_rare_rate": obs_rate,
            "observed_rare_enrichment": obs_enrichment,
            "permutation_null_mean": float(np.mean(null)),
            "permutation_null_ci_low_95": ci_low,
            "permutation_null_ci_high_95": ci_high,
            "one_sided_pvalue_enrichment_gt_random": p_right,
            "n_permutations": n_permutations,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "permutation_rare_enrichment.csv", index=False)

    # Plot enrichment by budget with null interval.
    plt.figure(figsize=(7, 4.5))
    plt.plot(out["budget_fraction"] * 100, out["observed_rare_enrichment"], marker="o", label="Observed policy")
    plt.plot(out["budget_fraction"] * 100, out["permutation_null_mean"], linestyle="--", label="Permutation null")
    plt.fill_between(out["budget_fraction"] * 100, out["permutation_null_ci_low_95"], out["permutation_null_ci_high_95"], alpha=0.2)
    plt.xlabel("Follow-up budget (%)")
    plt.ylabel("Rare-class enrichment")
    plt.title("Permutation test for rare-class enrichment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_permutation_rare_enrichment.png", dpi=300)
    plt.close()

    return out


def risk_coverage_analysis(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    confidence_col_internal: str,
    output_dir: Path,
    coverages: Sequence[float],
) -> pd.DataFrame:
    work = df.copy()
    work["_true"] = work[true_col].map(normalize_name)
    work["_pred"] = work[pred_col].map(normalize_name)
    work["_confidence"] = pd.to_numeric(work[confidence_col_internal], errors="coerce")
    work = work.dropna(subset=["_confidence"]).sort_values("_confidence", ascending=False).reset_index(drop=True)

    n = len(work)
    rows = []
    for cov in coverages:
        k = max(1, int(round(n * cov)))
        sub = work.iloc[:k]
        acc = accuracy_score(sub["_true"], sub["_pred"])
        rows.append({
            "coverage": cov,
            "n_accepted": k,
            "selective_accuracy": float(acc),
            "selective_risk": float(1.0 - acc),
            "selective_macro_f1": float(f1_score(sub["_true"], sub["_pred"], average="macro", zero_division=0)),
        })

    out = pd.DataFrame(rows)
    aurc = float(np.trapezoid(out["selective_risk"], out["coverage"]))
    out["aurc"] = aurc
    out.to_csv(output_dir / "risk_coverage_curve.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(out["coverage"] * 100, out["selective_risk"], marker="o", markersize=3)
    plt.xlabel("Coverage: accepted predictions (%)")
    plt.ylabel("Selective risk: error rate among accepted predictions")
    plt.title(f"Risk-coverage curve (AURC={aurc:.4f})")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_risk_coverage_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(out["coverage"] * 100, out["selective_accuracy"], marker="o", markersize=3)
    plt.xlabel("Coverage: accepted predictions (%)")
    plt.ylabel("Selective accuracy")
    plt.title("Selective-classification accuracy by coverage")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_selective_accuracy_curve.png", dpi=300)
    plt.close()

    return out


def error_detection_auroc(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    uncertainty_col_internal: str,
    output_dir: Path,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    work = df.copy()
    y_true = work[true_col].map(normalize_name).to_numpy()
    y_pred = work[pred_col].map(normalize_name).to_numpy()
    errors = (y_true != y_pred).astype(int)
    scores = pd.to_numeric(work[uncertainty_col_internal], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(scores)
    errors = errors[mask]
    scores = scores[mask]

    if len(np.unique(errors)) < 2:
        raise ValueError("Error-detection AUROC requires at least one correct and one incorrect prediction.")

    auroc = float(roc_auc_score(errors, scores))
    auprc = float(average_precision_score(errors, scores))
    baseline_error_rate = float(np.mean(errors))

    boot_auroc = []
    boot_auprc = []
    n = len(errors)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(errors[idx])) < 2:
            continue
        boot_auroc.append(roc_auc_score(errors[idx], scores[idx]))
        boot_auprc.append(average_precision_score(errors[idx], scores[idx]))
    auroc_low, auroc_high = percentile_ci(np.asarray(boot_auroc))
    auprc_low, auprc_high = percentile_ci(np.asarray(boot_auprc))

    fpr, tpr, thresholds = roc_curve(errors, scores)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df.to_csv(output_dir / "error_detection_roc_curve.csv", index=False)

    plt.figure(figsize=(5.5, 5.5))
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Error detection using uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_error_detection_roc.png", dpi=300)
    plt.close()

    out = pd.DataFrame([
        {
            "task": "detect_incorrect_predictions",
            "positive_label": "prediction_error",
            "score": "uncertainty",
            "n_objects": n,
            "error_rate": baseline_error_rate,
            "auroc": auroc,
            "auroc_ci_low_95": auroc_low,
            "auroc_ci_high_95": auroc_high,
            "auprc": auprc,
            "auprc_ci_low_95": auprc_low,
            "auprc_ci_high_95": auprc_high,
            "n_bootstrap": n_bootstrap,
        }
    ])
    out.to_csv(output_dir / "error_detection_auroc.csv", index=False)
    return out


def write_markdown_summary(
    output_dir: Path,
    args: argparse.Namespace,
    paired_seed: Optional[pd.DataFrame],
    paired_pred: Optional[pd.DataFrame],
    per_class: Optional[pd.DataFrame],
    per_family: Optional[pd.DataFrame],
    perm: Optional[pd.DataFrame],
    rc: Optional[pd.DataFrame],
    error_auc: Optional[pd.DataFrame],
) -> None:
    md = []
    md.append("# AstroTrust-AI publication-strength statistical analyses\n")
    md.append("This report was generated by `run_astrotrust_publication_strength_tests.py`.\n")
    md.append("\n## Inputs\n")
    for key in ["ensemble_predictions", "hybrid_predictions", "seed_summary", "label_map", "followup_ranking"]:
        val = getattr(args, key, None)
        if val:
            md.append(f"- `{key}`: `{val}`\n")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`\n")
    md.append(f"- Permutations: `{args.n_permutations}`\n")

    if paired_seed is not None and not paired_seed.empty:
        md.append("\n## 1. Paired seed-level test: ensemble vs hybrid\n")
        tmp = paired_seed.copy()
        show_cols = ["metric", "n_paired_seeds", "mean_delta_ensemble_minus_hybrid", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_pvalue", "wilcoxon_pvalue", "n_positive_deltas", "n_negative_deltas"]
        md.append(tmp[show_cols].to_markdown(index=False))
        md.append("\n")
        md.append("Interpretation: a positive delta means that `ensemble_hybrid_dominant` outperformed `hybrid_temporal_tabular_cnn` across matched random seeds.\n")

    if paired_pred is not None and not paired_pred.empty:
        md.append("\n## 1b. Paired held-out prediction test\n")
        md.append(paired_pred.to_markdown(index=False))
        md.append("\n")
        md.append("Interpretation: this complements the seed-level test by comparing predictions on the same held-out objects. The McNemar row tests paired correctness differences.\n")

    if per_class is not None and not per_class.empty:
        md.append("\n## 2. Per-class bootstrap confidence intervals\n")
        md.append("Output file: `per_class_bootstrap_ci.csv`. Use this as a supplementary table and highlight low-support/high-uncertainty classes in the manuscript.\n")
        top = per_class[per_class["metric"].eq("f1")].sort_values("support", ascending=False).head(12)
        md.append(top.to_markdown(index=False))
        md.append("\n")

    if per_family is not None and not per_family.empty:
        md.append("\n## 2b. Per-family bootstrap confidence intervals\n")
        md.append("Output file: `per_family_bootstrap_ci.csv`. This is usually stronger for the main paper than a large per-class table.\n")
        top = per_family[per_family["metric"].eq("f1")].sort_values("support", ascending=False)
        md.append(top.to_markdown(index=False))
        md.append("\n")

    if perm is not None and not perm.empty:
        md.append("\n## 3. Permutation test for rare-class enrichment\n")
        md.append(perm.to_markdown(index=False))
        md.append("\n")
        md.append("Interpretation: small one-sided p-values indicate that the follow-up score ranks rare objects substantially above random ranking.\n")

    if rc is not None and not rc.empty:
        md.append("\n## 4. Risk-coverage / selective classification\n")
        full_risk = float(rc.loc[rc["coverage"].idxmax(), "selective_risk"])
        best_50 = rc.iloc[(rc["coverage"] - 0.50).abs().argsort()[:1]].iloc[0]
        best_80 = rc.iloc[(rc["coverage"] - 0.80).abs().argsort()[:1]].iloc[0]
        md.append(f"- Full-coverage risk: `{full_risk:.6f}`\n")
        md.append(f"- Risk at ~50% coverage: `{best_50['selective_risk']:.6f}`; selective accuracy `{best_50['selective_accuracy']:.6f}`\n")
        md.append(f"- Risk at ~80% coverage: `{best_80['selective_risk']:.6f}`; selective accuracy `{best_80['selective_accuracy']:.6f}`\n")
        md.append(f"- AURC: `{float(rc['aurc'].iloc[0]):.6f}`\n")
        md.append("\nFigures: `fig_risk_coverage_curve.png`, `fig_selective_accuracy_curve.png`.\n")

    if error_auc is not None and not error_auc.empty:
        md.append("\n## 5. AUROC for error detection using uncertainty\n")
        md.append(error_auc.to_markdown(index=False))
        md.append("\n")
        md.append("Interpretation: AUROC above 0.5 indicates that uncertainty tends to be higher for incorrect predictions than for correct predictions.\n")
        md.append("\nFigure: `fig_error_detection_roc.png`.\n")

    md.append("\n## Suggested manuscript wording\n")
    md.append("```latex\n")
    md.append(
        "Beyond aggregate performance, we evaluated AstroTrust-AI under a set of paired and resampling-based reliability analyses. "
        "First, seed-level paired tests compared the final ensemble against the standalone hybrid temporal--tabular model across matched random seeds. "
        "Second, non-parametric bootstrap resampling was used to estimate class- and family-level confidence intervals, exposing where performance is stable and where low-support classes remain uncertain. "
        "Third, the rare-candidate follow-up policy was evaluated with a permutation test that compares the observed rare-object enrichment against random rankings under the same observing budget. "
        "Finally, we quantified operational abstention behavior through risk--coverage curves and measured whether predictive uncertainty can detect likely classification errors using AUROC/AUPRC.\n"
    )
    md.append("```\n")

    (output_dir / "publication_strength_tests_summary.md").write_text("".join(md), encoding="utf-8")


def parse_budgets(text: str) -> List[float]:
    vals = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        val = float(p)
        if val > 1:
            val /= 100.0
        vals.append(val)
    return vals


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AstroTrust-AI publication-strength statistical tests.")
    p.add_argument("--ensemble-predictions", required=True, type=Path, help="CSV with final ensemble predictions and, ideally, probabilities.")
    p.add_argument("--hybrid-predictions", type=Path, help="CSV with standalone hybrid predictions for paired held-out comparison.")
    p.add_argument("--seed-summary", type=Path, help="CSV with per-seed metrics for paired seed-level tests.")
    p.add_argument("--label-map", type=Path, help="CSV mapping fine classes to families and/or rare labels/counts.")
    p.add_argument("--followup-ranking", type=Path, help="Optional per-object follow-up ranking CSV for the rare-enrichment permutation test. If omitted, ensemble predictions are used.")
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--true-col", help="True-label column in ensemble predictions.")
    p.add_argument("--pred-col", help="Predicted-label column in ensemble predictions.")
    p.add_argument("--hybrid-true-col", help="True-label column in hybrid predictions.")
    p.add_argument("--hybrid-pred-col", help="Predicted-label column in hybrid predictions.")
    p.add_argument("--id-col", help="Object ID column for merging ensemble and hybrid predictions.")
    p.add_argument("--hybrid-id-col", help="Object ID column in hybrid predictions.")

    p.add_argument("--confidence-col", help="Confidence column. If omitted, max probability is used.")
    p.add_argument("--uncertainty-col", help="Uncertainty column. If omitted, 1 - confidence is used.")
    p.add_argument("--score-col", help="Ranking/priority score column for rare-enrichment permutation test.")
    p.add_argument("--rare-col", help="Rare-class boolean column in predictions, ranking file, or label map.")
    p.add_argument("--ranking-true-col", help="True-label column in --followup-ranking. Defaults to --true-col or auto-detection.")
    p.add_argument("--ranking-score-col", help="Ranking/priority score column in --followup-ranking. Defaults to --score-col or auto-detection.")
    p.add_argument("--ranking-rare-col", help="Rare-class boolean column in --followup-ranking. Defaults to --rare-col or auto-detection.")
    p.add_argument("--rare-classes", help="Comma-separated list of true class names considered rare.")
    p.add_argument("--rare-count-threshold", type=int, help="Define rare classes as classes with count <= threshold in label map.")
    p.add_argument("--rare-quantile", type=float, default=0.25, help="If using counts and no threshold, define rare as bottom quantile. Default: 0.25.")

    p.add_argument("--label-map-class-col", help="Class-name column in label map.")
    p.add_argument("--label-map-family-col", help="Family column in label map.")
    p.add_argument("--infer-family-from-class-name", action="store_true", help="Infer coarse families from fine class names when no family map is present.")

    p.add_argument("--prob-cols", help="Comma-separated class probability columns.")
    p.add_argument("--prob-prefixes", default="prob_,p_,proba_", help="Comma-separated prefixes for class probability columns.")

    p.add_argument("--ensemble-name", default="ensemble_hybrid_dominant")
    p.add_argument("--hybrid-name", default="hybrid_temporal_tabular_cnn")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budgets", default="0.01,0.02,0.05,0.10,0.20", help="Budget fractions or percentages for rare-enrichment test.")
    p.add_argument("--coverage-grid", default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0", help="Coverage fractions or percentages for risk-coverage curve.")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    rng = np.random.default_rng(args.seed)

    ensemble_df = pd.read_csv(args.ensemble_predictions)
    true_col = find_column(ensemble_df, args.true_col, TRUE_CANDIDATES, required=True, label="true-label column")
    pred_col = find_column(ensemble_df, args.pred_col, PRED_CANDIDATES, required=True, label="predicted-label column")
    id_col = find_column(ensemble_df, args.id_col, ID_CANDIDATES, required=False, label="object-id column")
    confidence_col = find_column(ensemble_df, args.confidence_col, CONFIDENCE_CANDIDATES, required=False, label="confidence column")
    uncertainty_col = find_column(ensemble_df, args.uncertainty_col, UNCERTAINTY_CANDIDATES, required=False, label="uncertainty column")
    score_col = find_column(ensemble_df, args.score_col, SCORE_CANDIDATES, required=False, label="ranking score column")
    rare_col_pred = find_column(ensemble_df, args.rare_col, RARE_CANDIDATES, required=False, label="rare column") if args.rare_col else find_column(ensemble_df, None, RARE_CANDIDATES, required=False, label="rare column")

    prob_prefixes = [p.strip() for p in args.prob_prefixes.split(",") if p.strip()]
    prob_cols = detect_probability_columns(
        ensemble_df,
        args.prob_cols,
        prob_prefixes,
        exclude=[true_col, pred_col, id_col, confidence_col, uncertainty_col, score_col, rare_col_pred],
    )

    spec = ColumnSpec(
        true_col=true_col,
        pred_col=pred_col,
        id_col=id_col,
        confidence_col=confidence_col,
        uncertainty_col=uncertainty_col,
        score_col=score_col,
        rare_col=rare_col_pred,
    )
    ensemble_df = add_confidence_uncertainty(ensemble_df, spec, prob_cols)

    family_map, rare_map, label_df = load_label_maps(
        args.label_map,
        args.label_map_family_col,
        args.label_map_class_col,
        args.rare_col,
        args.rare_count_threshold,
        args.rare_quantile,
        args.infer_family_from_class_name,
    )

    # Save detected schema.
    schema = {
        "true_col": true_col,
        "pred_col": pred_col,
        "id_col": id_col,
        "confidence_col": confidence_col,
        "uncertainty_col": uncertainty_col,
        "internal_confidence": "_confidence",
        "internal_uncertainty": "_uncertainty",
        "score_col": score_col,
        "rare_col": rare_col_pred,
        "n_probability_columns_detected": len(prob_cols),
        "probability_columns_sample": prob_cols[:20],
        "n_family_map_entries": len(family_map),
        "n_rare_map_entries": len(rare_map),
    }
    (args.output_dir / "detected_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Core point metrics for the ensemble.
    pd.DataFrame([basic_metrics(ensemble_df[true_col], ensemble_df[pred_col])]).to_csv(args.output_dir / "ensemble_point_metrics.csv", index=False)

    paired_seed = None
    if args.seed_summary and args.seed_summary.exists():
        paired_seed = paired_seed_tests(args.seed_summary, args.output_dir, args.ensemble_name, args.hybrid_name, args.n_bootstrap, rng)

    paired_pred = None
    if args.hybrid_predictions and args.hybrid_predictions.exists():
        hybrid_df = pd.read_csv(args.hybrid_predictions)
        h_true_col = find_column(hybrid_df, args.hybrid_true_col or args.true_col, TRUE_CANDIDATES, required=True, label="hybrid true-label column")
        h_pred_col = find_column(hybrid_df, args.hybrid_pred_col, PRED_CANDIDATES, required=True, label="hybrid predicted-label column")
        h_id_col = find_column(hybrid_df, args.hybrid_id_col or args.id_col, ID_CANDIDATES, required=False, label="hybrid object-id column")
        hybrid_spec = ColumnSpec(h_true_col, h_pred_col, h_id_col, None, None, None, None)
        paired_pred = paired_prediction_tests(ensemble_df, hybrid_df, spec, hybrid_spec, args.output_dir, args.n_bootstrap, rng)

    per_class = bootstrap_per_group_ci(
        ensemble_df,
        true_col,
        pred_col,
        args.output_dir / "per_class_bootstrap_ci.csv",
        args.n_bootstrap,
        rng,
        group_label="class",
    )

    per_family = None
    fam_df = add_family_columns(ensemble_df, true_col, pred_col, family_map, args.infer_family_from_class_name)
    if fam_df is not None:
        per_family = bootstrap_per_group_ci(
            fam_df,
            "_true_family",
            "_pred_family",
            args.output_dir / "per_family_bootstrap_ci.csv",
            args.n_bootstrap,
            rng,
            group_label="family",
        )

    perm = None
    # The rare-enrichment permutation test can be computed either from the main
    # prediction table or from a separate per-object follow-up ranking table.
    perm_df = ensemble_df
    perm_true_col = true_col
    perm_score_col = score_col
    perm_rare_col = rare_col_pred
    if args.followup_ranking and args.followup_ranking.exists():
        perm_df = pd.read_csv(args.followup_ranking)
        perm_true_col = find_column(perm_df, args.ranking_true_col or args.true_col, TRUE_CANDIDATES, required=True, label="true-label column in follow-up ranking")
        perm_score_col = find_column(perm_df, args.ranking_score_col or args.score_col, SCORE_CANDIDATES, required=True, label="ranking score column in follow-up ranking")
        if args.ranking_rare_col or args.rare_col:
            perm_rare_col = find_column(perm_df, args.ranking_rare_col or args.rare_col, RARE_CANDIDATES, required=False, label="rare column in follow-up ranking")
        else:
            perm_rare_col = find_column(perm_df, None, RARE_CANDIDATES, required=False, label="rare column in follow-up ranking")

    if perm_score_col:
        try:
            perm = permutation_rare_enrichment(
                perm_df,
                perm_true_col,
                perm_score_col,
                perm_rare_col,
                rare_map,
                args.rare_classes,
                args.output_dir,
                args.n_permutations,
                rng,
                parse_budgets(args.budgets),
            )
        except Exception as exc:
            (args.output_dir / "permutation_rare_enrichment_SKIPPED.txt").write_text(str(exc), encoding="utf-8")
            print(f"Rare-enrichment permutation skipped: {exc}")
    else:
        (args.output_dir / "permutation_rare_enrichment_SKIPPED.txt").write_text(
            "No ranking score column detected. Provide --score-col priority_score, --ranking-score-col, or --followup-ranking.",
            encoding="utf-8",
        )
        print("Rare-enrichment permutation skipped: no ranking score column detected.")

    rc = risk_coverage_analysis(
        ensemble_df,
        true_col,
        pred_col,
        "_confidence",
        args.output_dir,
        parse_budgets(args.coverage_grid),
    )

    error_auc = error_detection_auroc(
        ensemble_df,
        true_col,
        pred_col,
        "_uncertainty",
        args.output_dir,
        args.n_bootstrap,
        rng,
    )

    write_markdown_summary(args.output_dir, args, paired_seed, paired_pred, per_class, per_family, perm, rc, error_auc)
    print(f"Done. Results written to: {args.output_dir}")
    print(f"Summary: {args.output_dir / 'publication_strength_tests_summary.md'}")


if __name__ == "__main__":
    main()
