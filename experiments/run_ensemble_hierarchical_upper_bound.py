#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hierarchical upper-bound analysis for the final ensemble (`ensemble_hybrid_dominant`).

Strategies:
1. original_top1: argmax over final ensemble fine-class probabilities.
2. family_mass_then_subclass: choose the family with largest summed probability mass,
   then choose the highest-probability subclass inside that family.
3. oracle_true_family_then_subclass: restrict prediction to the true family and choose
   the highest-probability subclass. This is an upper bound, not a deployable model.

Recommended usage:
python .\experiments\run_ensemble_hierarchical_upper_bound.py `
  --predictions results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_predictions.csv `
  --probabilities results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_test_probabilities.npy `
  --object-family-indicators results\ensemble_topk_family_bootstrap_ci_250k_final\topk_family_object_level_indicators.csv `
  --output-dir results\ensemble_hierarchical_upper_bound_250k_final `
  --n-bootstrap 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def detect_column(cols: Iterable[str], candidates: List[str], label: str) -> str:
    cols = list(cols)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in cols:
        cl = c.lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    raise ValueError(f"Could not detect {label}. Available columns: {cols}")


def load_family_map(object_family_indicators: Path | None, class_family_map: Path | None) -> Tuple[Dict[int, str], dict]:
    if object_family_indicators is not None:
        df = pd.read_csv(object_family_indicators)
        label_col = detect_column(df.columns, ["true_label", "label", "class_label", "fine_label", "class_id"], "label column")
        family_col = detect_column(df.columns, ["true_family", "family", "class_family", "astronomical_family"], "family column")
        tmp = df[[label_col, family_col]].dropna().drop_duplicates()
        tmp[label_col] = tmp[label_col].astype(int)
        mapping = {int(lab): str(sub[family_col].astype(str).value_counts().index[0]) for lab, sub in tmp.groupby(label_col)}
        return mapping, {
            "family_map_source": str(object_family_indicators),
            "family_map_method": "derived from object-level true_label -> true_family indicators",
            "label_col": label_col,
            "family_col": family_col,
            "n_label_family_rows": int(len(tmp)),
        }

    if class_family_map is not None:
        df = pd.read_csv(class_family_map)
        label_col = detect_column(df.columns, ["true_label", "label", "class_label", "fine_label", "class_id", "target"], "label column")
        family_col = detect_column(df.columns, ["true_family", "family", "class_family", "astronomical_family", "family_name"], "family column")
        tmp = df[[label_col, family_col]].dropna().drop_duplicates()
        tmp[label_col] = tmp[label_col].astype(int)
        mapping = {int(lab): str(sub[family_col].astype(str).value_counts().index[0]) for lab, sub in tmp.groupby(label_col)}
        return mapping, {
            "family_map_source": str(class_family_map),
            "family_map_method": "derived from class-family map",
            "label_col": label_col,
            "family_col": family_col,
            "n_label_family_rows": int(len(tmp)),
        }

    raise ValueError("Provide --object-family-indicators or --class-family-map.")


def build_family_structures(label_to_family: Dict[int, str], n_classes: int):
    missing = [i for i in range(n_classes) if i not in label_to_family]
    if missing:
        raise ValueError(f"Family mapping missing labels: {missing}")
    family_names = sorted(set(label_to_family.values()))
    family_to_idx = {fam: i for i, fam in enumerate(family_names)}
    class_to_family_idx = np.array([family_to_idx[label_to_family[i]] for i in range(n_classes)], dtype=int)
    family_to_classes = {fam: np.array([i for i in range(n_classes) if label_to_family[i] == fam], dtype=int) for fam in family_names}
    return class_to_family_idx, family_names, family_to_classes


def predict_family_mass_then_subclass(probs: np.ndarray, family_names: List[str], family_to_classes: Dict[str, np.ndarray]):
    n = probs.shape[0]
    fam_mass = np.zeros((n, len(family_names)), dtype=float)
    for j, fam in enumerate(family_names):
        fam_mass[:, j] = probs[:, family_to_classes[fam]].sum(axis=1)
    pred_family_idx = fam_mass.argmax(axis=1)
    pred_fine = np.empty(n, dtype=int)
    for j, fam in enumerate(family_names):
        mask = pred_family_idx == j
        if np.any(mask):
            cls = family_to_classes[fam]
            pred_fine[mask] = cls[probs[np.ix_(mask, cls)].argmax(axis=1)]
    return pred_fine, pred_family_idx, fam_mass


def predict_oracle_true_family(probs: np.ndarray, true_family_idx: np.ndarray, family_names: List[str], family_to_classes: Dict[str, np.ndarray]):
    n = probs.shape[0]
    pred_fine = np.empty(n, dtype=int)
    for j, fam in enumerate(family_names):
        mask = true_family_idx == j
        if np.any(mask):
            cls = family_to_classes[fam]
            pred_fine[mask] = cls[probs[np.ix_(mask, cls)].argmax(axis=1)]
    return pred_fine


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    vals = []
    for lab in labels:
        mask = y_true == lab
        if np.any(mask):
            vals.append(float(np.mean(y_pred[mask] == lab)))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_accuracy(y_true_fine, y_true_family, predictions, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true_fine)
    rows = []
    for strat, pred in predictions.items():
        fine_pred = pred["fine_pred"]
        family_pred = pred["family_pred"]
        fine_samples = np.empty(n_bootstrap)
        family_samples = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            fine_samples[b] = np.mean(y_true_fine[idx] == fine_pred[idx])
            family_samples[b] = np.mean(y_true_family[idx] == family_pred[idx])
        rows.append({
            "strategy": strat,
            "metric": "fine_accuracy",
            "estimate": float(np.mean(y_true_fine == fine_pred)),
            "ci_low_95": float(np.quantile(fine_samples, 0.025)),
            "ci_high_95": float(np.quantile(fine_samples, 0.975)),
            "bootstrap_mean": float(np.mean(fine_samples)),
            "bootstrap_std": float(np.std(fine_samples, ddof=1)),
            "n_bootstrap": n_bootstrap,
        })
        rows.append({
            "strategy": strat,
            "metric": "family_accuracy",
            "estimate": float(np.mean(y_true_family == family_pred)),
            "ci_low_95": float(np.quantile(family_samples, 0.025)),
            "ci_high_95": float(np.quantile(family_samples, 0.975)),
            "bootstrap_mean": float(np.mean(family_samples)),
            "bootstrap_std": float(np.std(family_samples, ddof=1)),
            "n_bootstrap": n_bootstrap,
        })
    return pd.DataFrame(rows)


def paired_bootstrap_deltas(y_true_fine, y_true_family, predictions, reference_strategy, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true_fine)
    rows = []
    ref = predictions[reference_strategy]
    for strat, pred in predictions.items():
        if strat == reference_strategy:
            continue
        configs = [
            ("fine_accuracy", y_true_fine, ref["fine_pred"], pred["fine_pred"]),
            ("family_accuracy", y_true_family, ref["family_pred"], pred["family_pred"]),
        ]
        for metric_name, y_true, ref_pred, comp_pred in configs:
            point = float(np.mean(y_true == comp_pred) - np.mean(y_true == ref_pred))
            samples = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                samples[b] = np.mean(y_true[idx] == comp_pred[idx]) - np.mean(y_true[idx] == ref_pred[idx])
            p_le = np.mean(samples <= 0)
            p_ge = np.mean(samples >= 0)
            rows.append({
                "reference_strategy": reference_strategy,
                "comparison_strategy": strat,
                "metric": metric_name,
                "comparison_minus_reference": point,
                "ci_low_95": float(np.quantile(samples, 0.025)),
                "ci_high_95": float(np.quantile(samples, 0.975)),
                "two_sided_bootstrap_pvalue": float(min(1.0, 2.0 * min(p_le, p_ge))),
                "n_bootstrap": n_bootstrap,
            })
    return pd.DataFrame(rows)


def make_plots(metrics: pd.DataFrame, deltas: pd.DataFrame, out_dir: Path):
    order = ["original_top1", "family_mass_then_subclass", "oracle_true_family_then_subclass"]
    label_map = {
        "original_top1": "Original top-1",
        "family_mass_then_subclass": "Family-mass\nthen subclass",
        "oracle_true_family_then_subclass": "Oracle true family\nthen subclass",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in [(axes[0], "fine_accuracy", "Fine-grained accuracy"), (axes[1], "family_accuracy", "Family-level accuracy")]:
        sub = metrics[metrics["metric"] == metric].set_index("strategy").loc[order].reset_index()
        x = np.arange(len(sub))
        y = sub["estimate"].to_numpy()
        yerr = np.vstack([y - sub["ci_low_95"].to_numpy(), sub["ci_high_95"].to_numpy() - y])
        ax.bar(x, y, yerr=yerr, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([label_map[s] for s in sub["strategy"]])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, val in enumerate(y):
            ax.text(i, min(1.02, val + 0.025), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    try:
        fm = deltas[(deltas["comparison_strategy"] == "family_mass_then_subclass") & (deltas["metric"] == "fine_accuracy")].iloc[0]["comparison_minus_reference"]
        oracle = deltas[(deltas["comparison_strategy"] == "oracle_true_family_then_subclass") & (deltas["metric"] == "fine_accuracy")].iloc[0]["comparison_minus_reference"]
        axes[0].text(0.5, 0.08, f"Family-mass delta: {fm:+.3f}", ha="center", fontsize=8)
        axes[0].text(1.55, 0.16, f"Oracle headroom: {oracle:+.3f}", ha="center", fontsize=8)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_dir / "fig_hierarchical_upper_bound_two_panel.png", dpi=220)
    plt.close(fig)

    for metric, fname, title in [
        ("fine_accuracy", "fig_hierarchical_fine_accuracy.png", "Hierarchical upper-bound: fine-grained accuracy"),
        ("family_accuracy", "fig_hierarchical_family_accuracy.png", "Hierarchical upper-bound: family-level accuracy"),
    ]:
        sub = metrics[metrics["metric"] == metric].set_index("strategy").loc[order].reset_index()
        plt.figure(figsize=(6.2, 4.2))
        x = np.arange(len(sub))
        y = sub["estimate"].to_numpy()
        yerr = np.vstack([y - sub["ci_low_95"].to_numpy(), sub["ci_high_95"].to_numpy() - y])
        plt.bar(x, y, yerr=yerr, capsize=4)
        plt.xticks(x, [label_map[s] for s in sub["strategy"]])
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.grid(axis="y", alpha=0.25)
        for i, val in enumerate(y):
            plt.text(i, min(1.02, val + 0.025), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--object-family-indicators")
    parser.add_argument("--class-family-map")
    parser.add_argument("--output-dir", default="results/ensemble_hierarchical_upper_bound_250k_final")
    parser.add_argument("--id-col", default="object_id")
    parser.add_argument("--true-col", default="true_label")
    parser.add_argument("--pred-col", default="predicted_label")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.predictions)
    prob_path = Path(args.probabilities)
    pred_df = pd.read_csv(pred_path)
    probs = normalize_probs(np.load(prob_path))
    n, n_classes = probs.shape

    if len(pred_df) != n:
        raise ValueError(f"Predictions/probabilities row mismatch: {len(pred_df)} vs {n}")
    for c in [args.id_col, args.true_col]:
        if c not in pred_df.columns:
            raise ValueError(f"Missing required predictions column: {c}")

    label_to_family, family_schema = load_family_map(
        Path(args.object_family_indicators) if args.object_family_indicators else None,
        Path(args.class_family_map) if args.class_family_map else None,
    )
    class_to_family_idx, family_names, family_to_classes = build_family_structures(label_to_family, n_classes)

    object_ids = pred_df[args.id_col].to_numpy()
    y_true = pred_df[args.true_col].to_numpy(dtype=int)
    y_true_family = class_to_family_idx[y_true]

    original_fine = probs.argmax(axis=1).astype(int)
    mismatch = None
    if args.pred_col in pred_df.columns:
        mismatch = int(np.sum(pred_df[args.pred_col].to_numpy(dtype=int) != original_fine))
    original_family = class_to_family_idx[original_fine]

    fam_mass_fine, fam_mass_family, family_mass_matrix = predict_family_mass_then_subclass(probs, family_names, family_to_classes)
    oracle_fine = predict_oracle_true_family(probs, y_true_family, family_names, family_to_classes)
    oracle_family = y_true_family.copy()

    predictions = {
        "original_top1": {"fine_pred": original_fine, "family_pred": original_family},
        "family_mass_then_subclass": {"fine_pred": fam_mass_fine, "family_pred": fam_mass_family},
        "oracle_true_family_then_subclass": {"fine_pred": oracle_fine, "family_pred": oracle_family},
    }

    metrics = bootstrap_accuracy(y_true, y_true_family, predictions, args.n_bootstrap, args.seed)
    deltas = paired_bootstrap_deltas(y_true, y_true_family, predictions, "original_top1", args.n_bootstrap, args.seed + 100)

    labels = np.arange(n_classes)
    point_rows = []
    for strat, pred in predictions.items():
        point_rows.append({
            "strategy": strat,
            "fine_accuracy": float(np.mean(y_true == pred["fine_pred"])),
            "fine_macro_recall": macro_recall(y_true, pred["fine_pred"], labels),
            "family_accuracy": float(np.mean(y_true_family == pred["family_pred"])),
            "family_macro_recall": macro_recall(y_true_family, pred["family_pred"], np.arange(len(family_names))),
            "n_objects": n,
            "n_classes": n_classes,
            "n_families": len(family_names),
        })
    point_metrics = pd.DataFrame(point_rows)

    object_level = pd.DataFrame({
        "object_id": object_ids,
        "true_label": y_true,
        "true_family": [family_names[i] for i in y_true_family],
        "original_top1_label": original_fine,
        "original_top1_family": [family_names[i] for i in original_family],
        "original_top1_correct": y_true == original_fine,
        "family_mass_label": fam_mass_fine,
        "family_mass_family": [family_names[i] for i in fam_mass_family],
        "family_mass_fine_correct": y_true == fam_mass_fine,
        "family_mass_family_correct": y_true_family == fam_mass_family,
        "oracle_true_family_label": oracle_fine,
        "oracle_true_family": [family_names[i] for i in oracle_family],
        "oracle_true_family_fine_correct": y_true == oracle_fine,
        "confidence_original_top1": probs.max(axis=1),
        "family_mass_confidence": family_mass_matrix.max(axis=1),
        "oracle_true_family_subclass_probability": probs[np.arange(n), oracle_fine],
    })

    per_family_rows = []
    for j, fam in enumerate(family_names):
        mask = y_true_family == j
        row = {"family": fam, "support": int(mask.sum())}
        for strat, pred in predictions.items():
            row[f"{strat}_family_recall"] = float(np.mean(pred["family_pred"][mask] == j))
            row[f"{strat}_fine_accuracy_within_true_family"] = float(np.mean(pred["fine_pred"][mask] == y_true[mask]))
        per_family_rows.append(row)
    per_family = pd.DataFrame(per_family_rows).sort_values("support", ascending=False)

    metrics.to_csv(out_dir / "hierarchical_bootstrap_accuracy.csv", index=False)
    deltas.to_csv(out_dir / "hierarchical_paired_bootstrap_deltas_vs_original.csv", index=False)
    point_metrics.to_csv(out_dir / "hierarchical_strategy_point_metrics.csv", index=False)
    object_level.to_csv(out_dir / "hierarchical_object_level_predictions.csv", index=False)
    per_family.to_csv(out_dir / "hierarchical_per_family_metrics.csv", index=False)
    make_plots(metrics, deltas, out_dir)

    schema = {
        "analysis": "ensemble hierarchical upper-bound",
        "model": "ensemble_hybrid_dominant",
        "predictions": str(pred_path),
        "probabilities": str(prob_path),
        "n_objects": int(n),
        "n_classes": int(n_classes),
        "n_families": int(len(family_names)),
        "families": family_names,
        "class_to_family": {str(k): v for k, v in label_to_family.items()},
        "probability_class_order_assumption": "probability column j corresponds to class label j",
        "csv_predicted_label_mismatch_with_argmax_probabilities": mismatch,
        "strategies": {
            "original_top1": "argmax over final ensemble class probabilities",
            "family_mass_then_subclass": "sum probabilities by family, choose max-mass family, then choose highest-probability fine class within that family",
            "oracle_true_family_then_subclass": "restrict to the true family and choose highest-probability fine class; upper bound, not deployable",
        },
        "bootstrap": {"n_bootstrap": int(args.n_bootstrap), "seed": int(args.seed), "method": "non-parametric resampling over held-out test objects"},
        "family_mapping": family_schema,
    }
    (out_dir / "hierarchical_upper_bound_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def get_est(strategy, metric):
        return metrics[(metrics["strategy"] == strategy) & (metrics["metric"] == metric)].iloc[0]

    orig_fine = get_est("original_top1", "fine_accuracy")
    fm_fine = get_est("family_mass_then_subclass", "fine_accuracy")
    oracle_fine_m = get_est("oracle_true_family_then_subclass", "fine_accuracy")
    orig_family = get_est("original_top1", "family_accuracy")
    fm_family = get_est("family_mass_then_subclass", "family_accuracy")
    oracle_family_m = get_est("oracle_true_family_then_subclass", "family_accuracy")
    fm_delta_fine = deltas[(deltas["comparison_strategy"] == "family_mass_then_subclass") & (deltas["metric"] == "fine_accuracy")].iloc[0]
    oracle_delta_fine = deltas[(deltas["comparison_strategy"] == "oracle_true_family_then_subclass") & (deltas["metric"] == "fine_accuracy")].iloc[0]

    md = []
    md.append("# Ensemble hierarchical upper-bound analysis\n")
    md.append("This report evaluates whether hierarchical family structure provides headroom for the final `ensemble_hybrid_dominant` model.\n")
    md.append("## Inputs and protocol\n")
    md.append(f"- Predictions: `{pred_path}`")
    md.append(f"- Probabilities: `{prob_path}`")
    md.append(f"- Objects: `{n}`")
    md.append(f"- Classes: `{n_classes}`")
    md.append(f"- Families: `{len(family_names)}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`")
    md.append("- Strategy 1: `original_top1` = standard ensemble top-1 fine prediction.")
    md.append("- Strategy 2: `family_mass_then_subclass` = choose the family with largest summed probability mass, then the highest-probability subclass within that family.")
    md.append("- Strategy 3: `oracle_true_family_then_subclass` = restrict prediction to the true family; this is an upper bound, not a deployable model.\n")
    md.append("## Bootstrap accuracy summary\n")
    md.append(metrics.to_markdown(index=False))
    md.append("\n## Paired bootstrap deltas versus original top-1\n")
    md.append(deltas.to_markdown(index=False))
    md.append("\n## Point metrics\n")
    md.append(point_metrics.to_markdown(index=False))
    md.append("\n## Per-family metrics\n")
    md.append(per_family.to_markdown(index=False))
    md.append("\n## Key interpretation\n")
    md.append(f"- Original fine accuracy: `{float(orig_fine['estimate']):.6f}` with 95% CI [`{float(orig_fine['ci_low_95']):.6f}`, `{float(orig_fine['ci_high_95']):.6f}`].")
    md.append(f"- Family-mass fine accuracy: `{float(fm_fine['estimate']):.6f}` with 95% CI [`{float(fm_fine['ci_low_95']):.6f}`, `{float(fm_fine['ci_high_95']):.6f}`].")
    md.append(f"- Oracle true-family fine accuracy: `{float(oracle_fine_m['estimate']):.6f}` with 95% CI [`{float(oracle_fine_m['ci_low_95']):.6f}`, `{float(oracle_fine_m['ci_high_95']):.6f}`].")
    md.append(f"- Family-mass delta versus original fine accuracy: `{float(fm_delta_fine['comparison_minus_reference']):.6f}` with 95% CI [`{float(fm_delta_fine['ci_low_95']):.6f}`, `{float(fm_delta_fine['ci_high_95']):.6f}`].")
    md.append(f"- Oracle true-family headroom versus original fine accuracy: `{float(oracle_delta_fine['comparison_minus_reference']):.6f}` with 95% CI [`{float(oracle_delta_fine['ci_low_95']):.6f}`, `{float(oracle_delta_fine['ci_high_95']):.6f}`].")
    md.append(f"- Original family accuracy: `{float(orig_family['estimate']):.6f}`; family-mass family accuracy: `{float(fm_family['estimate']):.6f}`; oracle family accuracy: `{float(oracle_family_m['estimate']):.6f}`.")
    md.append("- Interpretation: a simple family-mass reranking is not guaranteed to improve fine-grained classification, but the oracle true-family result quantifies the recoverable headroom if family-level assignment improves.\n")
    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    md.append(
        f"We further quantified the hierarchical headroom of the final ensemble by comparing the original fine-grained top-1 prediction with two family-aware variants. "
        f"The standard ensemble achieved a fine-grained accuracy of \\textbf{{{float(orig_fine['estimate']):.4f}}} "
        f"(95\\% CI [{float(orig_fine['ci_low_95']):.4f}, {float(orig_fine['ci_high_95']):.4f}]). "
        f"A simple family-mass reranking strategy achieved \\textbf{{{float(fm_fine['estimate']):.4f}}} "
        f"(95\\% CI [{float(fm_fine['ci_low_95']):.4f}, {float(fm_fine['ci_high_95']):.4f}]), corresponding to a delta of "
        f"\\textbf{{{float(fm_delta_fine['comparison_minus_reference']):+.4f}}}. "
        f"When the prediction was restricted to the true astronomical family, the fine-grained upper bound increased to "
        f"\\textbf{{{float(oracle_fine_m['estimate']):.4f}}} "
        f"(95\\% CI [{float(oracle_fine_m['ci_low_95']):.4f}, {float(oracle_fine_m['ci_high_95']):.4f}]), yielding an oracle headroom of "
        f"\\textbf{{{float(oracle_delta_fine['comparison_minus_reference']):+.4f}}}. "
        f"These results indicate that naive hierarchical reranking alone is insufficient, but that improved family-level modeling could recover a substantial fraction of the remaining fine-grained errors."
    )
    md.append("```\n")
    md.append("## Output files\n")
    for name in [
        "hierarchical_bootstrap_accuracy.csv",
        "hierarchical_paired_bootstrap_deltas_vs_original.csv",
        "hierarchical_strategy_point_metrics.csv",
        "hierarchical_object_level_predictions.csv",
        "hierarchical_per_family_metrics.csv",
        "hierarchical_upper_bound_schema.json",
        "fig_hierarchical_upper_bound_two_panel.png",
        "fig_hierarchical_fine_accuracy.png",
        "fig_hierarchical_family_accuracy.png",
    ]:
        md.append(f"- `{out_dir / name}`")

    summary_path = out_dir / "hierarchical_upper_bound_summary.md"
    summary_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Done. Results written to: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
