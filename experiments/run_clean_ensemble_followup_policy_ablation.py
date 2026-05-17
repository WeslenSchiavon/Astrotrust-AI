#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_clean_ensemble_followup_policy_ablation.py

Refaz a ablação de follow-up de forma metodologicamente mais limpa.

Princípios:
1. Raridade é definida por uma lista explícita de classes raras.
2. rarity_score = soma das probabilidades do ensemble nas classes raras.
3. uncertainty_score = 1 - max_probability do ensemble.
4. novelty_score é calculado a partir de features independentes do objeto
   ou lido de uma coluna explicitamente fornecida.
5. O ranking NÃO é reaproveitado de políticas antigas.

Uso com novelty por features:
python .\experiments\run_clean_ensemble_followup_policy_ablation.py `
  --predictions results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_predictions.csv `
  --probabilities results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_test_probabilities.npy `
  --feature-file CAMINHO_DO_ARQUIVO_DE_FEATURES.csv `
  --rare-labels 0,1,2 `
  --output-dir results\ensemble_followup_policy_ablation_clean_250k_final `
  --n-bootstrap 1000

Uso com novelty_score já existente e explicitamente aceito:
python .\experiments\run_clean_ensemble_followup_policy_ablation.py `
  --predictions ... `
  --probabilities ... `
  --novelty-file CAMINHO.csv `
  --novelty-col novelty_score `
  --rare-labels 0,1,2 `
  --output-dir results\ensemble_followup_policy_ablation_clean_250k_final `
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
DEFAULT_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def parse_rare_labels(s: str | None, json_path: str | None) -> List[int]:
    if s:
        return sorted([int(x.strip()) for x in s.split(",") if x.strip() != ""])
    if json_path:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return sorted([int(x) for x in data])
        if "rare_labels" in data:
            return sorted([int(x) for x in data["rare_labels"]])
        if "consensus_rare_labels" in data:
            return sorted([int(x) for x in data["consensus_rare_labels"]])
    raise ValueError("You must provide --rare-labels or --rare-labels-json.")


def minmax01(x: np.ndarray, clip_quantile: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if clip_quantile is not None:
        hi = np.nanquantile(x, clip_quantile)
        x = np.minimum(x, hi)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < EPS:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def compute_feature_novelty(
    eval_features: pd.DataFrame,
    feature_cols: List[str],
    reference_features: pd.DataFrame | None = None,
    clip_quantile: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if not feature_cols:
        raise ValueError("No feature columns selected for novelty computation.")

    X_eval = eval_features[feature_cols].apply(pd.to_numeric, errors="coerce")
    if reference_features is None:
        X_ref = X_eval.copy()
        stats_source = "evaluation_feature_file"
    else:
        X_ref = reference_features[feature_cols].apply(pd.to_numeric, errors="coerce")
        stats_source = "reference_feature_file"

    med = X_ref.median(axis=0, skipna=True)
    q25 = X_ref.quantile(0.25)
    q75 = X_ref.quantile(0.75)
    iqr = (q75 - q25).replace(0, np.nan)

    valid_cols = [c for c in feature_cols if pd.notna(iqr[c]) and abs(float(iqr[c])) > EPS]
    if not valid_cols:
        raise ValueError("All selected feature columns have zero/invalid IQR.")

    med = med[valid_cols]
    iqr = iqr[valid_cols]

    X = X_eval[valid_cols].copy()
    X = X.fillna(med)
    Z = (X - med) / iqr
    Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    novelty_raw = np.sqrt(np.mean(np.square(Z.to_numpy(dtype=float)), axis=1))
    novelty_score = minmax01(novelty_raw, clip_quantile=clip_quantile)

    schema = {
        "novelty_method": "robust_feature_distance",
        "feature_stats_source": stats_source,
        "n_features_selected_initial": len(feature_cols),
        "n_features_used_after_iqr_filter": len(valid_cols),
        "feature_columns_used": valid_cols,
        "novelty_raw_definition": "sqrt(mean(robust_zscore^2))",
        "robust_zscore_definition": "(x - median_reference) / IQR_reference",
        "novelty_score_normalization": f"min-max after clipping raw novelty at q={clip_quantile}",
    }
    return novelty_raw, novelty_score, schema


def select_feature_columns(df: pd.DataFrame, id_col: str, explicit_cols: str | None) -> List[str]:
    if explicit_cols:
        requested = [c.strip() for c in explicit_cols.split(",") if c.strip()]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing: {missing}")
        return requested

    exclude_tokens = [
        id_col.lower(), "object_id", "objectid", "diaobjectid",
        "true", "label", "target", "class", "pred", "prob", "score",
        "rank", "rare", "novelty", "uncertainty", "correct", "split"
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = []
    for c in numeric_cols:
        cl = c.lower()
        if any(tok == cl or tok in cl for tok in exclude_tokens):
            continue
        selected.append(c)
    return selected


def selected_indices(scores: np.ndarray, budget: float, idx: np.ndarray | None = None) -> np.ndarray:
    if idx is None:
        idx = np.arange(len(scores))
    k = max(1, int(np.ceil(len(idx) * budget)))
    local_scores = scores[idx]
    order = np.argsort(-local_scores, kind="mergesort")
    return idx[order[:k]]


def mean_on_selection(values: np.ndarray, selected: np.ndarray) -> float:
    if len(selected) == 0:
        return float("nan")
    return float(np.nanmean(values[selected]))


def summarize_policy(
    policy: str,
    scores: np.ndarray,
    values: Dict[str, np.ndarray],
    budgets: List[float],
    baseline_rare_rate: float,
) -> List[dict]:
    rows = []
    n = len(scores)
    for b in budgets:
        sel = selected_indices(scores, b)
        rare_rate = mean_on_selection(values["true_is_rare"], sel)
        rows.append({
            "budget_fraction": b,
            "policy": policy,
            "n_objects": n,
            "n_selected": len(sel),
            "baseline_rare_rate": baseline_rare_rate,
            "rare_rate": rare_rate,
            "rare_enrichment": rare_rate / baseline_rare_rate if baseline_rare_rate > 0 else np.nan,
            "mean_novelty": mean_on_selection(values["novelty_score"], sel),
            "mean_uncertainty": mean_on_selection(values["uncertainty_score"], sel),
            "mean_confidence": mean_on_selection(values["confidence"], sel),
            "mean_correct": mean_on_selection(values["correct"], sel),
            "mean_rarity_score": mean_on_selection(values["rarity_score"], sel),
        })
    return rows


def bootstrap_summary(
    policy_scores: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
    budgets: List[float],
    n_bootstrap: int,
    seed: int,
    baseline_rare_rate: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(next(iter(policy_scores.values())))
    idx_all = np.arange(n)

    metric_names = ["true_is_rare", "novelty_score", "uncertainty_score", "confidence", "correct", "rarity_score"]
    metric_out_names = {
        "true_is_rare": "rare_rate",
        "novelty_score": "mean_novelty",
        "uncertainty_score": "mean_uncertainty",
        "confidence": "mean_confidence",
        "correct": "mean_correct",
        "rarity_score": "mean_rarity_score",
    }

    rows = []
    for budget in budgets:
        for policy, score in policy_scores.items():
            boot = {name: np.empty(n_bootstrap, dtype=float) for name in metric_names}
            for r in range(n_bootstrap):
                idx = rng.choice(idx_all, size=n, replace=True)
                sel = selected_indices(score, budget, idx=idx)
                for name in metric_names:
                    boot[name][r] = mean_on_selection(values[name], sel)

            point_sel = selected_indices(score, budget)
            base = {
                "budget_fraction": budget,
                "policy": policy,
                "n_objects": n,
                "n_selected": int(len(point_sel)),
                "baseline_rare_rate": baseline_rare_rate,
                "n_bootstrap": n_bootstrap,
            }
            for name in metric_names:
                out = metric_out_names[name]
                point = mean_on_selection(values[name], point_sel)
                base[out] = point
                base[f"{out}_ci_low_95"] = float(np.nanquantile(boot[name], 0.025))
                base[f"{out}_ci_high_95"] = float(np.nanquantile(boot[name], 0.975))
            base["rare_enrichment"] = base["rare_rate"] / baseline_rare_rate if baseline_rare_rate > 0 else np.nan
            base["rare_enrichment_ci_low_95"] = base["rare_rate_ci_low_95"] / baseline_rare_rate if baseline_rare_rate > 0 else np.nan
            base["rare_enrichment_ci_high_95"] = base["rare_rate_ci_high_95"] / baseline_rare_rate if baseline_rare_rate > 0 else np.nan
            rows.append(base)

    return pd.DataFrame(rows)


def bootstrap_pairwise(
    policy_scores: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
    reference_policy: str,
    budgets: List[float],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(next(iter(policy_scores.values())))
    idx_all = np.arange(n)

    metric_names = {
        "rare_rate": "true_is_rare",
        "novelty": "novelty_score",
        "uncertainty": "uncertainty_score",
        "confidence": "confidence",
        "correct": "correct",
        "rarity_score": "rarity_score",
    }

    rows = []
    ref_score = policy_scores[reference_policy]

    for budget in budgets:
        ref_sel = selected_indices(ref_score, budget)
        for comp_policy, comp_score in policy_scores.items():
            if comp_policy == reference_policy:
                continue
            comp_sel = selected_indices(comp_score, budget)
            for metric_out, value_name in metric_names.items():
                point = mean_on_selection(values[value_name], ref_sel) - mean_on_selection(values[value_name], comp_sel)
                diffs = np.empty(n_bootstrap, dtype=float)
                for r in range(n_bootstrap):
                    idx = rng.choice(idx_all, size=n, replace=True)
                    rs = selected_indices(ref_score, budget, idx=idx)
                    cs = selected_indices(comp_score, budget, idx=idx)
                    diffs[r] = mean_on_selection(values[value_name], rs) - mean_on_selection(values[value_name], cs)

                p_le = np.nanmean(diffs <= 0)
                p_ge = np.nanmean(diffs >= 0)
                p_two = float(min(1.0, 2.0 * min(p_le, p_ge)))

                rows.append({
                    "budget_fraction": budget,
                    "reference_policy": reference_policy,
                    "comparison_policy": comp_policy,
                    "metric": metric_out,
                    "reference_minus_comparison": point,
                    "bootstrap_ci_low_95": float(np.nanquantile(diffs, 0.025)),
                    "bootstrap_ci_high_95": float(np.nanquantile(diffs, 0.975)),
                    "two_sided_bootstrap_pvalue": p_two,
                    "n_bootstrap": n_bootstrap,
                })
    return pd.DataFrame(rows)


def overlap_table(policy_scores: Dict[str, np.ndarray], budgets: List[float], reference_policy: str) -> pd.DataFrame:
    rows = []
    ref_score = policy_scores[reference_policy]
    for budget in budgets:
        ref_sel = set(selected_indices(ref_score, budget).tolist())
        for policy, score in policy_scores.items():
            sel = set(selected_indices(score, budget).tolist())
            inter = len(ref_sel & sel)
            union = len(ref_sel | sel)
            rows.append({
                "budget_fraction": budget,
                "reference_policy": reference_policy,
                "policy": policy,
                "reference_n_selected": len(ref_sel),
                "policy_n_selected": len(sel),
                "intersection_n": inter,
                "union_n": union,
                "jaccard_overlap": inter / union if union else np.nan,
                "overlap_fraction_of_reference": inter / len(ref_sel) if ref_sel else np.nan,
                "overlap_fraction_of_policy": inter / len(sel) if sel else np.nan,
            })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    for metric, ylabel, fname in [
        ("rare_enrichment", "Rare-class enrichment", "fig_clean_policy_rare_enrichment_by_budget.png"),
        ("mean_novelty", "Mean novelty score", "fig_clean_policy_novelty_by_budget.png"),
        ("mean_correct", "Mean correctness", "fig_clean_policy_correctness_by_budget.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for policy, sub in summary.groupby("policy"):
            sub = sub.sort_values("budget_fraction")
            plt.plot(sub["budget_fraction"] * 100, sub[metric], marker="o", label=policy)
        plt.xlabel("Follow-up budget (%)")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by policy")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--feature-file")
    parser.add_argument("--reference-feature-file")
    parser.add_argument("--feature-cols")
    parser.add_argument("--novelty-file")
    parser.add_argument("--novelty-col")
    parser.add_argument("--rare-labels")
    parser.add_argument("--rare-labels-json")
    parser.add_argument("--output-dir", default="results/ensemble_followup_policy_ablation_clean_250k_final")
    parser.add_argument("--id-col", default="object_id")
    parser.add_argument("--true-col", default="true_label")
    parser.add_argument("--pred-col", default="predicted_label")
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-policy", default="novelty_rarity")
    parser.add_argument("--novelty-clip-quantile", type=float, default=0.99)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.predictions)
    prob_path = Path(args.probabilities)

    pred_df = pd.read_csv(pred_path)
    probs = normalize_probs(np.load(prob_path))

    if len(pred_df) != probs.shape[0]:
        raise ValueError(f"Predictions/probabilities mismatch: {len(pred_df)} vs {probs.shape[0]}")
    for c in [args.id_col, args.true_col]:
        if c not in pred_df.columns:
            raise ValueError(f"Missing required predictions column: {c}")

    rare_labels = parse_rare_labels(args.rare_labels, args.rare_labels_json)
    if min(rare_labels) < 0 or max(rare_labels) >= probs.shape[1]:
        raise ValueError(f"Rare labels outside probability matrix range 0..{probs.shape[1]-1}: {rare_labels}")

    base = pred_df[[args.id_col, args.true_col]].copy()
    base[args.pred_col] = probs.argmax(axis=1)
    base["confidence"] = probs.max(axis=1)
    base["uncertainty_score"] = 1.0 - base["confidence"]
    base["correct"] = base[args.pred_col].to_numpy().astype(int) == base[args.true_col].to_numpy().astype(int)
    base["true_is_rare"] = np.isin(base[args.true_col].to_numpy().astype(int), rare_labels)
    base["predicted_is_rare"] = np.isin(base[args.pred_col].to_numpy().astype(int), rare_labels)
    base["rarity_score_raw"] = probs[:, rare_labels].sum(axis=1)
    base["rarity_score"] = minmax01(base["rarity_score_raw"].to_numpy())

    novelty_schema = {}
    if args.feature_file:
        feature_path = Path(args.feature_file)
        feat_df = read_table(feature_path)
        if args.id_col not in feat_df.columns:
            raise ValueError(f"Feature file missing id column `{args.id_col}`: {feature_path}")

        cols = select_feature_columns(feat_df, args.id_col, args.feature_cols)
        ref_df = None
        if args.reference_feature_file:
            ref_df = read_table(Path(args.reference_feature_file))
            missing = [c for c in cols if c not in ref_df.columns]
            if missing:
                raise ValueError(f"Reference feature file missing selected feature columns: {missing[:20]}")
        novelty_raw, novelty_score, novelty_schema = compute_feature_novelty(
            eval_features=feat_df.set_index(args.id_col).loc[base[args.id_col]].reset_index(),
            feature_cols=cols,
            reference_features=ref_df,
            clip_quantile=args.novelty_clip_quantile,
        )
        base["novelty_raw"] = novelty_raw
        base["novelty_score"] = novelty_score
        novelty_schema["feature_file"] = str(feature_path)
        novelty_schema["reference_feature_file"] = args.reference_feature_file
    elif args.novelty_file and args.novelty_col:
        nov_path = Path(args.novelty_file)
        nov_df = read_table(nov_path)
        if args.id_col not in nov_df.columns or args.novelty_col not in nov_df.columns:
            raise ValueError(f"Novelty file must contain `{args.id_col}` and `{args.novelty_col}`.")
        nov_aux = nov_df[[args.id_col, args.novelty_col]].drop_duplicates(subset=[args.id_col])
        base = base.merge(nov_aux, on=args.id_col, how="left", validate="one_to_one")
        if base[args.novelty_col].isna().any():
            raise ValueError("Missing novelty values for some prediction objects.")
        base["novelty_raw"] = pd.to_numeric(base[args.novelty_col], errors="coerce").fillna(0).to_numpy()
        base["novelty_score"] = minmax01(base["novelty_raw"].to_numpy(), clip_quantile=args.novelty_clip_quantile)
        novelty_schema = {
            "novelty_method": "provided_column",
            "novelty_file": str(nov_path),
            "novelty_col": args.novelty_col,
            "warning": "Use this only if the column is object-level and independent of the old hybrid model ranking.",
        }
    else:
        raise ValueError("Provide either --feature-file or --novelty-file + --novelty-col.")

    # Normalize uncertainty too for balanced policies.
    base["uncertainty_score_norm"] = minmax01(base["uncertainty_score"].to_numpy())
    base["confidence_score_norm"] = minmax01(base["confidence"].to_numpy())

    rng = np.random.default_rng(args.seed)
    random_score = rng.random(len(base))

    # Clean policies.
    policy_scores = {
        "random": random_score,
        "rarity_only": base["rarity_score"].to_numpy(),
        "novelty_only": base["novelty_score"].to_numpy(),
        "uncertainty_only": base["uncertainty_score_norm"].to_numpy(),
        "novelty_rarity": 0.5 * base["novelty_score"].to_numpy() + 0.5 * base["rarity_score"].to_numpy(),
        "uncertainty_rarity": 0.5 * base["uncertainty_score_norm"].to_numpy() + 0.5 * base["rarity_score"].to_numpy(),
        "uncertainty_novelty_rarity": (
            base["uncertainty_score_norm"].to_numpy()
            + base["novelty_score"].to_numpy()
            + base["rarity_score"].to_numpy()
        ) / 3.0,
    }

    if args.reference_policy not in policy_scores:
        raise ValueError(f"Reference policy `{args.reference_policy}` not available. Options: {list(policy_scores)}")

    values = {
        "true_is_rare": base["true_is_rare"].astype(float).to_numpy(),
        "novelty_score": base["novelty_score"].to_numpy(dtype=float),
        "uncertainty_score": base["uncertainty_score"].to_numpy(dtype=float),
        "confidence": base["confidence"].to_numpy(dtype=float),
        "correct": base["correct"].astype(float).to_numpy(),
        "rarity_score": base["rarity_score"].to_numpy(dtype=float),
    }
    baseline_rare_rate = float(np.mean(values["true_is_rare"]))

    # Save object-level table and rankings.
    base_out = base.copy()
    for policy, score in policy_scores.items():
        base_out[f"priority_score_{policy}"] = score
        ranks = np.empty(len(score), dtype=int)
        ranks[np.argsort(-score, kind="mergesort")] = np.arange(1, len(score) + 1)
        base_out[f"priority_rank_{policy}"] = ranks

        ranking_cols = [
            args.id_col, args.true_col, args.pred_col, "correct",
            "true_is_rare", "predicted_is_rare", "confidence",
            "uncertainty_score", "novelty_raw", "novelty_score",
            "rarity_score_raw", "rarity_score",
            f"priority_score_{policy}", f"priority_rank_{policy}",
        ]
        rank_df = base_out[ranking_cols].sort_values(f"priority_rank_{policy}")
        rank_df.to_csv(out_dir / f"clean_ensemble_test_ranking_{policy}.csv", index=False)

    base_out.to_csv(out_dir / "clean_ensemble_followup_object_level.csv", index=False)

    summary = bootstrap_summary(
        policy_scores=policy_scores,
        values=values,
        budgets=args.budgets,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 100,
        baseline_rare_rate=baseline_rare_rate,
    )
    pairwise = bootstrap_pairwise(
        policy_scores=policy_scores,
        values=values,
        reference_policy=args.reference_policy,
        budgets=args.budgets,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 200,
    )
    overlap = overlap_table(policy_scores, args.budgets, args.reference_policy)

    summary.to_csv(out_dir / "clean_followup_policy_summary_by_budget.csv", index=False)
    pairwise.to_csv(out_dir / "clean_followup_policy_pairwise_vs_reference.csv", index=False)
    overlap.to_csv(out_dir / "clean_followup_policy_overlap_vs_reference.csv", index=False)

    plot_summary(summary, out_dir)

    schema = {
        "analysis": "clean ensemble follow-up policy ablation",
        "model": "ensemble_hybrid_dominant",
        "predictions": str(pred_path),
        "probabilities": str(prob_path),
        "rare_labels_explicit": rare_labels,
        "rarity_score_definition": "sum of final ensemble probabilities over explicitly predefined rare labels, min-max normalized for combined policies",
        "uncertainty_score_definition": "1 - max ensemble probability",
        "novelty": novelty_schema,
        "policies": {
            "random": "deterministic random score with fixed seed",
            "rarity_only": "rarity_score",
            "novelty_only": "novelty_score",
            "uncertainty_only": "normalized uncertainty_score",
            "novelty_rarity": "0.5*novelty_score + 0.5*rarity_score",
            "uncertainty_rarity": "0.5*uncertainty_score_norm + 0.5*rarity_score",
            "uncertainty_novelty_rarity": "(uncertainty_score_norm + novelty_score + rarity_score)/3",
        },
        "budgets": args.budgets,
        "n_objects": int(len(base)),
        "n_classes": int(probs.shape[1]),
        "baseline_rare_rate": baseline_rare_rate,
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_method": "resample test objects with replacement; rerank policies within each bootstrap sample; compute selected-set metrics",
        "seed": int(args.seed),
    }
    (out_dir / "clean_followup_policy_ablation_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    key_budget = 0.05 if 0.05 in args.budgets else args.budgets[0]
    def get_row(policy: str) -> pd.Series:
        return summary[(summary["budget_fraction"] == key_budget) & (summary["policy"] == policy)].iloc[0]
    nr = get_row("novelty_rarity")
    ro = get_row("rarity_only")
    rand = get_row("random")
    unr = get_row("uncertainty_novelty_rarity")
    diff_nr_ro_rare = pairwise[
        (pairwise["budget_fraction"] == key_budget)
        & (pairwise["comparison_policy"] == "rarity_only")
        & (pairwise["metric"] == "rare_rate")
    ].iloc[0]
    diff_nr_ro_nov = pairwise[
        (pairwise["budget_fraction"] == key_budget)
        & (pairwise["comparison_policy"] == "rarity_only")
        & (pairwise["metric"] == "novelty")
    ].iloc[0]

    md = []
    md.append("# Clean ensemble follow-up policy ablation\n")
    md.append("This report evaluates follow-up policies reconstructed from the final `ensemble_hybrid_dominant` probabilities with explicit rare-class labels and independent novelty scores.\n")
    md.append("## Inputs and protocol\n")
    md.append(f"- Predictions: `{pred_path}`")
    md.append(f"- Probabilities: `{prob_path}`")
    md.append(f"- Objects: `{len(base)}`")
    md.append(f"- Classes: `{probs.shape[1]}`")
    md.append(f"- Explicit rare labels: `{rare_labels}`")
    md.append(f"- Baseline rare rate: `{baseline_rare_rate:.6f}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`")
    md.append("- Bootstrap method: resample objects, rerank within each bootstrap sample, and recompute selected-set metrics.")
    md.append(f"- Novelty method: `{novelty_schema.get('novelty_method')}`\n")

    md.append("## Summary by budget\n")
    md.append(summary.to_markdown(index=False))
    md.append("\n## Pairwise comparisons versus reference\n")
    md.append(pairwise.to_markdown(index=False))
    md.append("\n## Ranking overlap\n")
    md.append(overlap.to_markdown(index=False))

    md.append("\n## Key interpretation\n")
    md.append(f"- At the {key_budget:.0%} budget, `novelty_rarity` selects `{int(nr['n_selected'])}` objects.")
    md.append(f"- `novelty_rarity` rare rate is `{float(nr['rare_rate']):.6f}` with 95% CI [`{float(nr['rare_rate_ci_low_95']):.6f}`, `{float(nr['rare_rate_ci_high_95']):.6f}`], corresponding to enrichment `{float(nr['rare_enrichment']):.6f}`.")
    md.append(f"- `rarity_only` rare rate is `{float(ro['rare_rate']):.6f}`, corresponding to enrichment `{float(ro['rare_enrichment']):.6f}`.")
    md.append(f"- `random` rare rate is `{float(rand['rare_rate']):.6f}`, corresponding to enrichment `{float(rand['rare_enrichment']):.6f}`.")
    md.append(f"- `uncertainty_novelty_rarity` rare rate is `{float(unr['rare_rate']):.6f}`, corresponding to enrichment `{float(unr['rare_enrichment']):.6f}`.")
    md.append(f"- Difference `novelty_rarity - rarity_only` for rare rate: `{float(diff_nr_ro_rare['reference_minus_comparison']):.6f}` with 95% CI [`{float(diff_nr_ro_rare['bootstrap_ci_low_95']):.6f}`, `{float(diff_nr_ro_rare['bootstrap_ci_high_95']):.6f}`].")
    md.append(f"- Difference `novelty_rarity - rarity_only` for novelty: `{float(diff_nr_ro_nov['reference_minus_comparison']):.6f}` with 95% CI [`{float(diff_nr_ro_nov['bootstrap_ci_low_95']):.6f}`, `{float(diff_nr_ro_nov['bootstrap_ci_high_95']):.6f}`].\n")

    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    md.append(
        f"We performed a clean follow-up policy ablation using the final ensemble probabilities, an explicit predefined rare-class set, and a model-independent novelty score. "
        f"At a 5\\% follow-up budget, \\texttt{{novelty\\_rarity}} selected \\textbf{{{int(nr['n_selected'])}}} objects and achieved a rare-object rate of "
        f"\\textbf{{{float(nr['rare_rate']):.4f}}} (95\\% CI [{float(nr['rare_rate_ci_low_95']):.4f}, {float(nr['rare_rate_ci_high_95']):.4f}]), "
        f"corresponding to a rare-class enrichment of \\textbf{{{float(nr['rare_enrichment']):.2f}$\\times$}} over the test-set baseline. "
        f"The \\texttt{{rarity\\_only}} policy reached a rare-object rate of \\textbf{{{float(ro['rare_rate']):.4f}}}, whereas a random policy reached \\textbf{{{float(rand['rare_rate']):.4f}}}. "
        f"Compared with \\texttt{{rarity\\_only}}, \\texttt{{novelty\\_rarity}} changed the rare-object rate by "
        f"\\textbf{{{float(diff_nr_ro_rare['reference_minus_comparison']):.4f}}} "
        f"(95\\% CI [{float(diff_nr_ro_rare['bootstrap_ci_low_95']):.4f}, {float(diff_nr_ro_rare['bootstrap_ci_high_95']):.4f}]) "
        f"while changing the novelty score by \\textbf{{{float(diff_nr_ro_nov['reference_minus_comparison']):.4f}}} "
        f"(95\\% CI [{float(diff_nr_ro_nov['bootstrap_ci_low_95']):.4f}, {float(diff_nr_ro_nov['bootstrap_ci_high_95']):.4f}]). "
        f"Thus, the follow-up module should be interpreted as a rare-class prioritization mechanism over a predefined target set, with novelty acting as an explicit secondary criterion for broker-like triage."
    )
    md.append("```\n")

    md.append("## Output files\n")
    for p in [
        out_dir / "clean_followup_policy_summary_by_budget.csv",
        out_dir / "clean_followup_policy_pairwise_vs_reference.csv",
        out_dir / "clean_followup_policy_overlap_vs_reference.csv",
        out_dir / "clean_ensemble_followup_object_level.csv",
        out_dir / "clean_followup_policy_ablation_schema.json",
        out_dir / "fig_clean_policy_rare_enrichment_by_budget.png",
        out_dir / "fig_clean_policy_novelty_by_budget.png",
        out_dir / "fig_clean_policy_correctness_by_budget.png",
    ]:
        md.append(f"- `{p}`")

    summary_md = out_dir / "clean_followup_policy_ablation_summary.md"
    summary_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")
    print(f"Summary: {summary_md}")


if __name__ == "__main__":
    main()
