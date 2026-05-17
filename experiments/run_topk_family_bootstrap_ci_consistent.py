#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_topk_family_bootstrap_ci_consistent.py

Calcula ICs bootstrap para métricas top-k fine-grained e family-level usando
a MESMA definição operacional da tabela original de top-k/family analysis:

1. fine top-k:
   verifica se a classe fina verdadeira está entre os top-k fine labels.

2. family top-k:
   pega os top-k fine labels, mapeia cada classe fina para sua família,
   e verifica se a família verdadeira aparece entre essas famílias.
   Ou seja, NÃO soma probabilidades por família antes do top-k.

Essa definição mantém consistência com métricas como:
- fine_accuracy
- fine_top3_accuracy
- fine_top5_accuracy
- family_accuracy
- family_top3_family_accuracy
- family_top5_family_accuracy

Uso:
python .\experiments\run_topk_family_bootstrap_ci_consistent.py `
  --predictions results\final_publication\temperature_scaled_hybrid_250k\hybrid_temporal_tabular_cnn_temperature_scaled_test_predictions.csv `
  --probabilities results\final_publication\temperature_scaled_hybrid_250k\hybrid_temporal_tabular_cnn_temperature_scaled_test_probabilities.npy `
  --class-family-map results\per_class_family_bootstrap\per_class_bootstrap_metrics.csv `
  --output-dir results\topk_family_bootstrap_ci_250k_consistent_final `
  --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_COL_CANDIDATES = [
    "class", "class_id", "true_label", "label", "target", "fine_class",
    "fine_label", "elasticc_class", "class_number"
]

FAMILY_COL_CANDIDATES = [
    "family", "class_family", "coarse_family", "coarse_label",
    "family_name", "true_family", "astrophysical_family"
]


def normalize_label(x) -> str:
    if pd.isna(x):
        return "nan"
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


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
        raise ValueError(f"Could not infer column. Candidates={candidates}. Available columns={list(df.columns)}")
    return None


def load_class_family_map(path: Path, class_col: Optional[str], family_col: Optional[str]) -> Dict[str, str]:
    df = pd.read_csv(path)
    ccol = find_col(df, class_col, CLASS_COL_CANDIDATES, required=True)
    fcol = find_col(df, family_col, FAMILY_COL_CANDIDATES, required=True)

    mapping = {}
    for _, row in df[[ccol, fcol]].dropna().drop_duplicates().iterrows():
        mapping[normalize_label(row[ccol])] = str(row[fcol]).strip()

    if not mapping:
        raise ValueError(f"No valid class-family mapping could be read from {path}")

    return mapping


def infer_class_labels(n_classes: int, explicit_labels: Optional[str]) -> List[str]:
    if explicit_labels:
        labels = [normalize_label(x) for x in explicit_labels.split(",")]
        if len(labels) != n_classes:
            raise ValueError(f"--class-labels has {len(labels)} labels, but probabilities have {n_classes} classes.")
        return labels
    return [str(i) for i in range(n_classes)]


def topk_indices_sorted(probs: np.ndarray, k: int) -> np.ndarray:
    k = min(k, probs.shape[1])
    part = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    part_probs = np.take_along_axis(probs, part, axis=1)
    order = np.argsort(-part_probs, axis=1)
    return np.take_along_axis(part, order, axis=1)


def bootstrap_ci(values_by_metric: Dict[str, np.ndarray], n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(next(iter(values_by_metric.values())))
    idx_all = np.arange(n)

    rows = []
    for metric, vals in values_by_metric.items():
        vals = vals.astype(float)
        boot = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            idx = rng.choice(idx_all, size=n, replace=True)
            boot[b] = vals[idx].mean()

        rows.append({
            "metric": metric,
            "point_estimate": float(vals.mean()),
            "bootstrap_mean": float(boot.mean()),
            "bootstrap_std": float(boot.std(ddof=1)),
            "ci_low_95": float(np.quantile(boot, 0.025)),
            "ci_high_95": float(np.quantile(boot, 0.975)),
            "n_objects": n,
            "n_bootstrap": n_bootstrap,
        })

    order = {
        "fine_accuracy": 0,
        "fine_top2_accuracy": 1,
        "fine_top3_accuracy": 2,
        "fine_top5_accuracy": 3,
        "family_accuracy": 4,
        "family_top2_family_accuracy": 5,
        "family_top3_family_accuracy": 6,
        "family_top5_family_accuracy": 7,
    }
    df = pd.DataFrame(rows)
    df["_order"] = df["metric"].map(order).fillna(999)
    return df.sort_values(["_order", "metric"]).drop(columns=["_order"])


def per_family_recall_ci(
    true_family: np.ndarray,
    pred_family_top1: np.ndarray,
    family_names: List[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for fam in family_names:
        mask = true_family == fam
        support = int(mask.sum())
        if support == 0:
            continue

        correct = (pred_family_top1[mask] == fam).astype(float)
        idx_all = np.arange(support)
        boot = np.empty(n_bootstrap, dtype=float)

        for b in range(n_bootstrap):
            idx = rng.choice(idx_all, size=support, replace=True)
            boot[b] = correct[idx].mean()

        rows.append({
            "family": fam,
            "metric": "family_recall_top1_from_fine_top1",
            "point_estimate": float(correct.mean()),
            "bootstrap_mean": float(boot.mean()),
            "bootstrap_std": float(boot.std(ddof=1)),
            "ci_low_95": float(np.quantile(boot, 0.025)),
            "ci_high_95": float(np.quantile(boot, 0.975)),
            "support": support,
            "n_bootstrap": n_bootstrap,
        })

    return pd.DataFrame(rows).sort_values("point_estimate", ascending=False)


def plot_main_ci(df: pd.DataFrame, out_path: Path) -> None:
    p = df.copy()
    x = np.arange(len(p))
    y = p["point_estimate"].to_numpy()
    lo = p["ci_low_95"].to_numpy()
    hi = p["ci_high_95"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])

    plt.figure(figsize=(10, 5))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
    plt.xticks(x, p["metric"], rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(max(0, y.min() - 0.05), min(1.01, y.max() + 0.05))
    plt.title("Bootstrap CIs for fine top-k and mapped-family top-k performance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_family_ci(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    p = df.sort_values("point_estimate", ascending=True)
    y_pos = np.arange(len(p))
    x = p["point_estimate"].to_numpy()
    lo = p["ci_low_95"].to_numpy()
    hi = p["ci_high_95"].to_numpy()
    xerr = np.vstack([x - lo, hi - x])

    plt.figure(figsize=(8, max(4, 0.35 * len(p))))
    plt.errorbar(x, y_pos, xerr=xerr, fmt="o", capsize=4)
    plt.yticks(y_pos, p["family"])
    plt.xlabel("Family recall from fine top-1 mapped to family")
    plt.title("Per-family bootstrap confidence intervals")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--class-family-map", required=True)
    parser.add_argument("--output-dir", default="results/topk_family_bootstrap_ci_250k_consistent_final")
    parser.add_argument("--true-col", default="true_label")
    parser.add_argument("--pred-col", default="predicted_label")
    parser.add_argument("--class-map-class-col", default=None)
    parser.add_argument("--class-map-family-col", default=None)
    parser.add_argument("--class-labels", default=None)
    parser.add_argument("--topk", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    probabilities_path = Path(args.probabilities)
    map_path = Path(args.class_family_map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_path)
    probs = np.load(probabilities_path).astype(np.float64)

    if len(df) != probs.shape[0]:
        raise ValueError(f"Row mismatch: predictions={len(df)} probabilities={probs.shape[0]}")

    if args.true_col not in df.columns:
        raise ValueError(f"True label column `{args.true_col}` not found. Available={list(df.columns)}")

    n_objects, n_classes = probs.shape
    class_labels = infer_class_labels(n_classes, args.class_labels)
    label_to_idx = {normalize_label(lab): i for i, lab in enumerate(class_labels)}

    y_true_label = np.array([normalize_label(v) for v in df[args.true_col].to_numpy()], dtype=object)
    missing_true = sorted(set(lab for lab in y_true_label if lab not in label_to_idx))
    if missing_true:
        raise ValueError(
            f"Some true labels are not present in probability class labels: {missing_true[:20]}. "
            "Use --class-labels if the probability order is not 0..n_classes-1."
        )

    y_true_idx = np.array([label_to_idx[lab] for lab in y_true_label], dtype=int)

    class_to_family = load_class_family_map(map_path, args.class_map_class_col, args.class_map_family_col)

    missing_map = [lab for lab in class_labels if normalize_label(lab) not in class_to_family]
    if missing_map:
        raise ValueError(
            f"Some class labels are missing from the class-family map: {missing_map[:20]}. "
            "Fix the map or pass --class-labels with labels matching the map."
        )

    class_family = np.array([class_to_family[normalize_label(lab)] for lab in class_labels], dtype=object)
    true_family = np.array([class_family[i] for i in y_true_idx], dtype=object)
    family_names = sorted(set(class_family))

    top1_idx = probs.argmax(axis=1)
    pred_family_top1 = class_family[top1_idx]

    values = {}
    values["fine_accuracy"] = (top1_idx == y_true_idx).astype(float)
    values["family_accuracy"] = (pred_family_top1 == true_family).astype(float)

    object_level = pd.DataFrame({
        "object_id": df["object_id"] if "object_id" in df.columns else np.arange(n_objects),
        "true_label": y_true_label,
        "true_family": true_family,
        "predicted_label_from_probs": [class_labels[i] for i in top1_idx],
        "predicted_family_from_fine_top1": pred_family_top1,
        "fine_accuracy": values["fine_accuracy"].astype(int),
        "family_accuracy": values["family_accuracy"].astype(int),
    })

    for k in args.topk:
        topk = topk_indices_sorted(probs, k)

        fine_hit = np.any(topk == y_true_idx[:, None], axis=1)
        values[f"fine_top{k}_accuracy"] = fine_hit.astype(float)

        family_hit = np.zeros(n_objects, dtype=bool)
        for i in range(n_objects):
            fams_in_topk = set(class_family[topk[i]])
            family_hit[i] = true_family[i] in fams_in_topk

        values[f"family_top{k}_family_accuracy"] = family_hit.astype(float)
        object_level[f"fine_top{k}_accuracy"] = fine_hit.astype(int)
        object_level[f"family_top{k}_family_accuracy"] = family_hit.astype(int)

    main_ci = bootstrap_ci(values, args.n_bootstrap, args.seed)
    main_path = out_dir / "topk_family_bootstrap_ci.csv"
    main_ci.to_csv(main_path, index=False)

    per_family = per_family_recall_ci(
        true_family=true_family,
        pred_family_top1=pred_family_top1,
        family_names=family_names,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
    )
    per_family_path = out_dir / "per_family_recall_bootstrap_ci.csv"
    per_family.to_csv(per_family_path, index=False)

    indicators_path = out_dir / "topk_family_object_level_indicators.csv"
    object_level.to_csv(indicators_path, index=False)

    plot_main_ci(main_ci, out_dir / "fig_topk_family_bootstrap_ci.png")
    plot_per_family_ci(per_family, out_dir / "fig_per_family_recall_bootstrap_ci.png")

    schema = {
        "definition": "family-level metrics are computed by mapping fine top-k labels to families; probabilities are NOT summed by family before top-k",
        "predictions": str(predictions_path),
        "probabilities": str(probabilities_path),
        "class_family_map": str(map_path),
        "n_objects": int(n_objects),
        "n_classes": int(n_classes),
        "n_families": int(len(family_names)),
        "families": family_names,
        "class_labels": class_labels,
        "topk": args.topk,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    schema_path = out_dir / "topk_family_bootstrap_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def get(metric: str) -> pd.Series:
        sub = main_ci[main_ci["metric"] == metric]
        if sub.empty:
            raise ValueError(f"Metric not found: {metric}")
        return sub.iloc[0]

    fine_acc = get("fine_accuracy")
    fine_top3 = get("fine_top3_accuracy")
    fine_top5 = get("fine_top5_accuracy")
    fam_acc = get("family_accuracy")
    fam_top3 = get("family_top3_family_accuracy")
    fam_top5 = get("family_top5_family_accuracy")

    md = []
    md.append("# Top-k and family-level bootstrap confidence intervals\n")
    md.append("This report estimates bootstrap confidence intervals for fine-grained top-k and family-level performance.\n")
    md.append("## Definition used\n")
    md.append("Family-level top-k metrics are computed by mapping the fine-grained top-k predicted classes to their astronomical families and checking whether the true family appears in that mapped set. Probabilities are **not** summed by family before selecting top-k families.\n")
    md.append("This definition is intended to be consistent with the original top-k/family-level table in the final publication summary.\n")

    md.append("## Inputs\n")
    md.append(f"- Predictions CSV: `{predictions_path}`")
    md.append(f"- Probability matrix: `{probabilities_path}`")
    md.append(f"- Class-family map: `{map_path}`")
    md.append(f"- Objects: `{n_objects}`")
    md.append(f"- Fine classes: `{n_classes}`")
    md.append(f"- Families: `{len(family_names)}`")
    md.append(f"- Families detected: `{', '.join(family_names)}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`\n")

    md.append("## Main top-k and family-level bootstrap CIs\n")
    md.append(main_ci.to_markdown(index=False))
    md.append("")

    md.append("## Per-family top-1 recall bootstrap CIs\n")
    md.append(per_family.to_markdown(index=False))
    md.append("")

    md.append("## Key interpretation\n")
    md.append(f"- Fine top-1 accuracy is `{fine_acc['point_estimate']:.6f}` (95% CI [`{fine_acc['ci_low_95']:.6f}`, `{fine_acc['ci_high_95']:.6f}`]).")
    md.append(f"- Fine top-3 accuracy is `{fine_top3['point_estimate']:.6f}` (95% CI [`{fine_top3['ci_low_95']:.6f}`, `{fine_top3['ci_high_95']:.6f}`]).")
    md.append(f"- Fine top-5 accuracy is `{fine_top5['point_estimate']:.6f}` (95% CI [`{fine_top5['ci_low_95']:.6f}`, `{fine_top5['ci_high_95']:.6f}`]).")
    md.append(f"- Family top-1 accuracy, obtained by mapping fine top-1 predictions to families, is `{fam_acc['point_estimate']:.6f}` (95% CI [`{fam_acc['ci_low_95']:.6f}`, `{fam_acc['ci_high_95']:.6f}`]).")
    md.append(f"- Family top-3 accuracy, obtained by mapping fine top-3 predictions to families, is `{fam_top3['point_estimate']:.6f}` (95% CI [`{fam_top3['ci_low_95']:.6f}`, `{fam_top3['ci_high_95']:.6f}`]).")
    md.append(f"- Family top-5 accuracy, obtained by mapping fine top-5 predictions to families, is `{fam_top5['point_estimate']:.6f}` (95% CI [`{fam_top5['ci_low_95']:.6f}`, `{fam_top5['ci_high_95']:.6f}`]).\n")

    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    md.append(
        "We quantified uncertainty in the top-$k$ and family-level metrics using non-parametric bootstrap resampling over the held-out test objects. "
        f"Although fine-grained top-1 accuracy was \\textbf{{{fine_acc['point_estimate']:.4f}}} "
        f"(95\\% CI [{fine_acc['ci_low_95']:.4f}, {fine_acc['ci_high_95']:.4f}]), "
        f"the fine-grained top-3 and top-5 accuracies increased to \\textbf{{{fine_top3['point_estimate']:.4f}}} "
        f"(95\\% CI [{fine_top3['ci_low_95']:.4f}, {fine_top3['ci_high_95']:.4f}]) and "
        f"\\textbf{{{fine_top5['point_estimate']:.4f}}} "
        f"(95\\% CI [{fine_top5['ci_low_95']:.4f}, {fine_top5['ci_high_95']:.4f}]), respectively. "
        "For family-level evaluation, we mapped the fine-grained top-$k$ predictions to their corresponding astronomical families. "
        f"Under this definition, family top-1 accuracy reached \\textbf{{{fam_acc['point_estimate']:.4f}}} "
        f"(95\\% CI [{fam_acc['ci_low_95']:.4f}, {fam_acc['ci_high_95']:.4f}]), while family top-3 and top-5 accuracies reached "
        f"\\textbf{{{fam_top3['point_estimate']:.4f}}} "
        f"(95\\% CI [{fam_top3['ci_low_95']:.4f}, {fam_top3['ci_high_95']:.4f}]) and "
        f"\\textbf{{{fam_top5['point_estimate']:.4f}}} "
        f"(95\\% CI [{fam_top5['ci_low_95']:.4f}, {fam_top5['ci_high_95']:.4f}]). "
        "These results support the use of AstroTrust-AI as a broker-like triage tool: even when the exact fine-grained subclass is uncertain, the correct class often remains within the high-probability candidate set or within the correct astrophysical family."
    )
    md.append("```\n")

    md.append("## Output files\n")
    for p in [
        main_path,
        per_family_path,
        indicators_path,
        schema_path,
        out_dir / "fig_topk_family_bootstrap_ci.png",
        out_dir / "fig_per_family_recall_bootstrap_ci.png",
    ]:
        md.append(f"- `{p}`")

    md_path = out_dir / "topk_family_bootstrap_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
