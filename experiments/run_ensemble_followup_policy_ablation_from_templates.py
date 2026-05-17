#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
run_ensemble_followup_policy_ablation_from_templates.py

Refaz a ablação de políticas de follow-up usando o modelo final:
`ensemble_hybrid_dominant`.

Ideia:
- Usa os rankings híbridos antigos apenas como TEMPLATE para recuperar:
  object_id, raw_novelty, novelty_score, true_is_rare e pesos das políticas.
- Recalcula, para o ensemble:
  confidence, uncertainty_score, predicted_label, correct, predicted_is_rare,
  rarity_score = soma das probabilidades do ensemble nas classes raras.
- Reconstrói os rankings:
  novelty_rarity, rarity_only, previous_discovery, fixed_discovery
  usando os pesos salvos nos templates.
- Roda bootstrap de ablação em budgets fixos.

Uso:
python .\experiments\run_ensemble_followup_policy_ablation_from_templates.py `
  --predictions results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_predictions.csv `
  --probabilities results\hybrid_tabular_ensemble_250k\ensemble_hybrid_dominant_test_probabilities.npy `
  --template-dir results\hybrid_followup_policy_eval_250k `
  --output-dir results\ensemble_followup_policy_ablation_250k_final `
  --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_POLICIES = ["novelty_rarity", "rarity_only", "previous_discovery", "fixed_discovery"]
DEFAULT_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
EPS = 1e-12


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def as_bool_float(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(float).to_numpy()
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map({
            "true": 1, "false": 0,
            "1": 1, "0": 0,
            "yes": 1, "no": 0,
            "y": 1, "n": 0,
        })
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).to_numpy()


def as_bool_series(series: pd.Series) -> pd.Series:
    return pd.Series(as_bool_float(series), index=series.index).astype(bool)


def infer_rare_labels(template_df: pd.DataFrame, true_col: str, rare_col: str) -> List[int]:
    tmp = template_df[[true_col, rare_col]].copy()
    tmp[rare_col] = as_bool_float(tmp[rare_col])
    grouped = tmp.groupby(true_col)[rare_col].mean()
    rare_labels = sorted([int(k) for k, v in grouped.items() if float(v) >= 0.5])
    if not rare_labels:
        raise ValueError("Could not infer rare labels from template true_is_rare column.")
    return rare_labels


def get_policy_weights(template_policy_df: pd.DataFrame, policy: str) -> Tuple[float, float, float]:
    """
    Returns (w_uncertainty, w_novelty, w_rarity).
    Prefer weights saved in template columns. Fallbacks preserve expected policy semantics.
    """
    cols = ["w_uncertainty", "w_novelty", "w_rarity"]
    if all(c in template_policy_df.columns for c in cols):
        vals = template_policy_df[cols].dropna()
        if len(vals) > 0:
            first = vals.iloc[0]
            return float(first["w_uncertainty"]), float(first["w_novelty"]), float(first["w_rarity"])

    # Conservative fallbacks if templates do not contain weights.
    fallback = {
        "novelty_rarity": (0.0, 0.5, 0.5),
        "rarity_only": (0.0, 0.0, 1.0),
        "previous_discovery": (0.5, 0.5, 0.0),
        "fixed_discovery": (1.0, 0.0, 0.0),
    }
    return fallback.get(policy, (0.0, 0.5, 0.5))


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def selected_mask_by_score(df: pd.DataFrame, budget: float, score_col: str = "priority_score") -> np.ndarray:
    n = len(df)
    k = max(1, int(round(n * budget)))
    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(-np.inf).to_numpy()
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def safe_mean(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.nanmean(x))


def selected_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return safe_mean(values[mask])


def bootstrap_metric(values: np.ndarray, mask: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> np.ndarray:
    n = len(values)
    idx_all = np.arange(n)
    out = np.full(n_bootstrap, np.nan)
    for b in range(n_bootstrap):
        idx = rng.choice(idx_all, size=n, replace=True)
        mb = mask[idx]
        if int(mb.sum()) > 0:
            out[b] = np.nanmean(values[idx][mb])
    return out


def bootstrap_pairwise_diff(values: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> np.ndarray:
    n = len(values)
    idx_all = np.arange(n)
    out = np.full(n_bootstrap, np.nan)
    for b in range(n_bootstrap):
        idx = rng.choice(idx_all, size=n, replace=True)
        ma = mask_a[idx]
        mb = mask_b[idx]
        if int(ma.sum()) > 0 and int(mb.sum()) > 0:
            out[b] = np.nanmean(values[idx][ma]) - np.nanmean(values[idx][mb])
    return out


def ci95(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def pvalue_two_sided(samples: np.ndarray) -> float:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan")
    p_le = np.mean(samples <= 0)
    p_ge = np.mean(samples >= 0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def pvalue_noninferior(samples: np.ndarray, margin: float) -> float:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan")
    return float(np.mean(samples <= -abs(margin)))


def build_ensemble_rankings(
    pred_path: Path,
    prob_path: Path,
    template_dir: Path,
    output_dir: Path,
    policies: List[str],
    id_col: str,
    true_col: str,
    pred_col: str,
    template_prefix: str,
    output_prefix: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ensemble_df = pd.read_csv(pred_path)
    probs = normalize_probs(np.load(prob_path))

    if len(ensemble_df) != probs.shape[0]:
        raise ValueError(f"Predictions/probabilities row mismatch: {len(ensemble_df)} vs {probs.shape[0]}")
    for col in [id_col, true_col]:
        if col not in ensemble_df.columns:
            raise ValueError(f"Missing column in ensemble predictions: {col}")

    pred_from_probs = probs.argmax(axis=1).astype(int)
    if pred_col in ensemble_df.columns:
        pred_csv = ensemble_df[pred_col].to_numpy().astype(int)
        if not np.array_equal(pred_csv, pred_from_probs):
            mismatch = int((pred_csv != pred_from_probs).sum())
            print(f"WARNING: {mismatch} predicted labels differ from argmax(probabilities). Using argmax(probabilities).")

    ensemble_base = ensemble_df[[id_col, true_col]].copy()
    ensemble_base[pred_col] = pred_from_probs
    ensemble_base["correct"] = ensemble_base[pred_col].to_numpy().astype(int) == ensemble_base[true_col].to_numpy().astype(int)
    ensemble_base["confidence"] = probs.max(axis=1)
    ensemble_base["uncertainty_score"] = 1.0 - ensemble_base["confidence"].to_numpy()

    # Load templates.
    templates = {}
    for policy in policies:
        p = template_dir / f"{template_prefix}_{policy}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing template ranking for policy `{policy}`: {p}")
        templates[policy] = pd.read_csv(p)

    ref_template = templates["novelty_rarity"] if "novelty_rarity" in templates else next(iter(templates.values()))
    for col in [id_col, true_col, "true_is_rare"]:
        if col not in ref_template.columns:
            raise ValueError(f"Reference template missing required column `{col}`")

    rare_labels = infer_rare_labels(ref_template, true_col, "true_is_rare")
    rare_probs = probs[:, rare_labels].sum(axis=1)
    ensemble_base["rarity_score"] = rare_probs
    ensemble_base["predicted_is_rare"] = np.isin(ensemble_base[pred_col].to_numpy().astype(int), rare_labels)

    # Copy novelty / rare truth from reference template.
    keep_cols = [id_col]
    for c in ["split", "raw_novelty", "novelty_score", "true_is_rare"]:
        if c in ref_template.columns:
            keep_cols.append(c)

    aux = ref_template[keep_cols].drop_duplicates(subset=[id_col], keep="first").copy()
    merged = ensemble_base.merge(aux, on=id_col, how="left", validate="one_to_one")

    if merged["true_is_rare"].isna().any():
        raise ValueError("Some ensemble objects were not found in the template rankings; cannot recover true_is_rare/novelty.")

    if "novelty_score" not in merged.columns:
        if "raw_novelty" in merged.columns:
            raw = pd.to_numeric(merged["raw_novelty"], errors="coerce").fillna(0).to_numpy()
            lo, hi = np.nanmin(raw), np.nanmax(raw)
            merged["novelty_score"] = (raw - lo) / (hi - lo + EPS)
        else:
            merged["novelty_score"] = 0.0
            merged["raw_novelty"] = 0.0

    if "raw_novelty" not in merged.columns:
        merged["raw_novelty"] = merged["novelty_score"]

    if "split" not in merged.columns:
        merged["split"] = "test"

    merged["true_is_rare"] = as_bool_series(merged["true_is_rare"])

    rankings = {}
    weights_used = {}

    for policy in policies:
        wu, wn, wr = get_policy_weights(templates[policy], policy)
        weights_used[policy] = {
            "w_uncertainty": wu,
            "w_novelty": wn,
            "w_rarity": wr,
        }

        df = merged.copy()
        df["configuration"] = policy
        df["w_uncertainty"] = wu
        df["w_novelty"] = wn
        df["w_rarity"] = wr
        df["priority_score"] = (
            wu * pd.to_numeric(df["uncertainty_score"], errors="coerce").fillna(0).to_numpy()
            + wn * pd.to_numeric(df["novelty_score"], errors="coerce").fillna(0).to_numpy()
            + wr * pd.to_numeric(df["rarity_score"], errors="coerce").fillna(0).to_numpy()
        )
        df["priority_rank"] = rank_desc(df["priority_score"].to_numpy())

        out_cols = [
            "split", id_col, true_col, pred_col, "correct", "confidence",
            "uncertainty_score", "raw_novelty", "novelty_score", "rarity_score",
            "true_is_rare", "predicted_is_rare", "configuration",
            "w_uncertainty", "w_novelty", "w_rarity",
            "priority_score", "priority_rank",
        ]
        df = df[out_cols].sort_values("priority_rank").reset_index(drop=True)
        out_path = output_dir / f"{output_prefix}_{policy}.csv"
        df.to_csv(out_path, index=False)
        rankings[policy] = df

    schema = {
        "generation": "ensemble rankings reconstructed from hybrid ranking templates",
        "ensemble_predictions": str(pred_path),
        "ensemble_probabilities": str(prob_path),
        "template_dir": str(template_dir),
        "template_prefix": template_prefix,
        "output_prefix": output_prefix,
        "rare_labels_inferred_from_template_true_is_rare": rare_labels,
        "rarity_score_definition": "sum of final ensemble probabilities over inferred rare labels",
        "uncertainty_score_definition": "1 - max ensemble probability",
        "novelty_score_source": f"{template_dir}\\{template_prefix}_novelty_rarity.csv",
        "weights_used": weights_used,
        "n_objects": int(len(merged)),
        "n_classes": int(probs.shape[1]),
    }
    (output_dir / "ensemble_followup_ranking_generation_schema.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    return rankings, schema


def run_ablation(
    rankings: Dict[str, pd.DataFrame],
    output_dir: Path,
    budgets: List[float],
    n_bootstrap: int,
    seed: int,
    reference_policy: str,
    id_col: str,
    noninferiority_margin_rare_rate: float,
) -> None:
    rng = np.random.default_rng(seed)

    if reference_policy not in rankings:
        raise ValueError(f"Reference policy not found: {reference_policy}")

    # Align by common objects and common order.
    common = None
    for df in rankings.values():
        ids = set(df[id_col].astype(str))
        common = ids if common is None else common & ids
    common_ids = sorted(common)
    if not common_ids:
        raise ValueError("No common object IDs across ensemble policy rankings.")

    aligned = {}
    for policy, df in rankings.items():
        tmp = df.copy()
        tmp[id_col] = tmp[id_col].astype(str)
        tmp = tmp[tmp[id_col].isin(common_ids)].set_index(id_col).loc[common_ids].reset_index()
        aligned[policy] = tmp

    ref_df = aligned[reference_policy]
    rare = as_bool_float(ref_df["true_is_rare"])
    baseline_rare_rate = float(np.mean(rare))

    aux_values = {
        "novelty": pd.to_numeric(ref_df["novelty_score"], errors="coerce").to_numpy(),
        "raw_novelty": pd.to_numeric(ref_df["raw_novelty"], errors="coerce").to_numpy(),
        "uncertainty": pd.to_numeric(ref_df["uncertainty_score"], errors="coerce").to_numpy(),
        "confidence": pd.to_numeric(ref_df["confidence"], errors="coerce").to_numpy(),
        "correct": as_bool_float(ref_df["correct"]),
        "rarity_score": pd.to_numeric(ref_df["rarity_score"], errors="coerce").to_numpy(),
    }

    selected_masks = {
        policy: {budget: selected_mask_by_score(df, budget) for budget in budgets}
        for policy, df in aligned.items()
    }

    summary_rows = []
    pairwise_rows = []
    overlap_rows = []

    for budget in budgets:
        ref_mask = selected_masks[reference_policy][budget]
        ref_k = int(ref_mask.sum())

        for policy, df in aligned.items():
            mask = selected_masks[policy][budget]
            rare_rate = selected_mean(rare, mask)
            rare_enrichment = rare_rate / baseline_rare_rate if baseline_rare_rate > 0 else float("nan")
            rare_boot = bootstrap_metric(rare, mask, rng, n_bootstrap)
            rare_ci_low, rare_ci_high = ci95(rare_boot)

            row = {
                "budget_fraction": budget,
                "policy": policy,
                "n_objects_common": len(common_ids),
                "n_selected": int(mask.sum()),
                "baseline_rare_rate": baseline_rare_rate,
                "rare_rate": rare_rate,
                "rare_rate_ci_low_95": rare_ci_low,
                "rare_rate_ci_high_95": rare_ci_high,
                "rare_enrichment": rare_enrichment,
                "rare_enrichment_ci_low_95": rare_ci_low / baseline_rare_rate if baseline_rare_rate > 0 else float("nan"),
                "rare_enrichment_ci_high_95": rare_ci_high / baseline_rare_rate if baseline_rare_rate > 0 else float("nan"),
                "n_bootstrap": n_bootstrap,
            }
            for label, vals in aux_values.items():
                row[f"mean_{label}"] = selected_mean(vals, mask)
            summary_rows.append(row)

            inter = int(np.logical_and(ref_mask, mask).sum())
            union = int(np.logical_or(ref_mask, mask).sum())
            overlap_rows.append({
                "budget_fraction": budget,
                "reference_policy": reference_policy,
                "policy": policy,
                "reference_n_selected": ref_k,
                "policy_n_selected": int(mask.sum()),
                "intersection_n": inter,
                "union_n": union,
                "jaccard_overlap": inter / union if union > 0 else float("nan"),
                "overlap_fraction_of_reference": inter / ref_k if ref_k > 0 else float("nan"),
                "overlap_fraction_of_policy": inter / int(mask.sum()) if int(mask.sum()) > 0 else float("nan"),
            })

        for policy in aligned:
            if policy == reference_policy:
                continue

            ref_mask = selected_masks[reference_policy][budget]
            other_mask = selected_masks[policy][budget]

            metrics = {"rare_rate": rare, **aux_values}
            for metric, values in metrics.items():
                diff_boot = bootstrap_pairwise_diff(values, ref_mask, other_mask, rng, n_bootstrap)
                diff = selected_mean(values, ref_mask) - selected_mean(values, other_mask)
                lo, hi = ci95(diff_boot)
                pairwise_rows.append({
                    "budget_fraction": budget,
                    "reference_policy": reference_policy,
                    "comparison_policy": policy,
                    "metric": metric,
                    "reference_minus_comparison": diff,
                    "bootstrap_ci_low_95": lo,
                    "bootstrap_ci_high_95": hi,
                    "two_sided_bootstrap_pvalue": pvalue_two_sided(diff_boot),
                    "noninferiority_margin": noninferiority_margin_rare_rate if metric == "rare_rate" else np.nan,
                    "one_sided_pvalue_reference_more_than_margin_worse": (
                        pvalue_noninferior(diff_boot, noninferiority_margin_rare_rate)
                        if metric == "rare_rate" else np.nan
                    ),
                    "n_bootstrap": n_bootstrap,
                })

    summary_df = pd.DataFrame(summary_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    summary_path = output_dir / "followup_policy_ablation_summary_by_budget.csv"
    pairwise_path = output_dir / "followup_policy_ablation_pairwise_vs_reference.csv"
    overlap_path = output_dir / "followup_policy_ablation_overlap_vs_reference.csv"

    summary_df.to_csv(summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    schema = {
        "policy_files": {
            policy: str(output_dir / f"ensemble_test_ranking_{policy}.csv")
            for policy in aligned
        },
        "policies_loaded": list(aligned.keys()),
        "reference_policy": reference_policy,
        "id_col": id_col,
        "score_col": "priority_score",
        "rank_col": "priority_rank",
        "rare_col": "true_is_rare",
        "budgets": budgets,
        "n_bootstrap": n_bootstrap,
        "n_common_objects": len(common_ids),
        "baseline_rare_rate": baseline_rare_rate,
        "model": "ensemble_hybrid_dominant",
    }
    (output_dir / "followup_policy_ablation_schema.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    # Figures.
    try:
        pivot = summary_df.pivot(index="budget_fraction", columns="policy", values="rare_enrichment")
        plt.figure(figsize=(8, 5))
        for col in pivot.columns:
            plt.plot(pivot.index * 100, pivot[col], marker="o", label=col)
        plt.xlabel("Follow-up budget (%)")
        plt.ylabel("Rare-class enrichment")
        plt.title("Rare-class enrichment by follow-up policy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "fig_policy_rare_enrichment_by_budget.png", dpi=200)
        plt.close()

        pivot_nov = summary_df.pivot(index="budget_fraction", columns="policy", values="mean_novelty")
        plt.figure(figsize=(8, 5))
        for col in pivot_nov.columns:
            plt.plot(pivot_nov.index * 100, pivot_nov[col], marker="o", label=col)
        plt.xlabel("Follow-up budget (%)")
        plt.ylabel("Mean novelty score")
        plt.title("Mean novelty by follow-up policy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "fig_policy_novelty_by_budget.png", dpi=200)
        plt.close()

        budget_for_tradeoff = 0.05 if 0.05 in budgets else budgets[0]
        trade = summary_df[summary_df["budget_fraction"] == budget_for_tradeoff].copy()
        plt.figure(figsize=(7, 5))
        plt.scatter(trade["mean_novelty"], trade["rare_enrichment"])
        for _, r in trade.iterrows():
            plt.annotate(r["policy"], (r["mean_novelty"], r["rare_enrichment"]))
        plt.xlabel("Mean novelty score")
        plt.ylabel("Rare-class enrichment")
        plt.title(f"Policy trade-off at {budget_for_tradeoff:.0%} budget")
        plt.tight_layout()
        plt.savefig(output_dir / "fig_policy_tradeoff_rare_vs_novelty.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Figure generation failed: {exc}")

    # Markdown summary.
    key_budget = 0.05 if 0.05 in budgets else budgets[0]
    def pick_summary(policy: str, budget: float = key_budget) -> pd.Series:
        row = summary_df[(summary_df["budget_fraction"] == budget) & (summary_df["policy"] == policy)]
        if row.empty:
            raise ValueError(f"Missing summary row: {policy}, budget={budget}")
        return row.iloc[0]

    def pick_pairwise(policy: str, metric: str, budget: float = key_budget) -> pd.Series:
        row = pairwise_df[
            (pairwise_df["budget_fraction"] == budget)
            & (pairwise_df["comparison_policy"] == policy)
            & (pairwise_df["metric"] == metric)
        ]
        if row.empty:
            raise ValueError(f"Missing pairwise row: comparison={policy}, metric={metric}, budget={budget}")
        return row.iloc[0]

    nr5 = pick_summary(reference_policy, key_budget)
    ro5 = pick_summary("rarity_only", key_budget) if "rarity_only" in aligned else None
    nr_vs_ro_rare = pick_pairwise("rarity_only", "rare_rate", key_budget) if "rarity_only" in aligned else None
    nr_vs_ro_nov = pick_pairwise("rarity_only", "novelty", key_budget) if "rarity_only" in aligned else None

    md = []
    md.append("# Ensemble follow-up policy ablation bootstrap\n")
    md.append("This report compares object-level follow-up policies reconstructed for the final `ensemble_hybrid_dominant` model.\n")
    md.append("## Inputs\n")
    md.append(f"- Common objects across policies: `{len(common_ids)}`")
    md.append(f"- Baseline rare rate: `{baseline_rare_rate:.6f}`")
    md.append(f"- Bootstrap iterations: `{n_bootstrap}`")
    md.append(f"- Reference policy: `{reference_policy}`")
    md.append("- Rarity score: sum of final ensemble probabilities over rare labels inferred from the template `true_is_rare` column.")
    md.append("- Novelty score: reused from the original follow-up ranking template because novelty is object-level metadata, not a model prediction.\n")

    md.append("## Summary by budget\n")
    cols = [
        "budget_fraction", "policy", "n_selected", "rare_rate", "rare_rate_ci_low_95",
        "rare_rate_ci_high_95", "rare_enrichment", "rare_enrichment_ci_low_95",
        "rare_enrichment_ci_high_95", "mean_novelty", "mean_uncertainty",
        "mean_confidence", "mean_correct"
    ]
    md.append(summary_df[cols].to_markdown(index=False))
    md.append("")

    md.append("## Pairwise bootstrap comparisons versus reference\n")
    md.append(pairwise_df.to_markdown(index=False))
    md.append("")

    md.append("## Overlap with reference policy\n")
    md.append(overlap_df.to_markdown(index=False))
    md.append("")

    md.append("## Key interpretation\n")
    md.append(f"- At the {key_budget:.0%} follow-up budget, `{reference_policy}` selects `{int(nr5['n_selected'])}` objects.")
    md.append(f"- `{reference_policy}` rare rate is `{float(nr5['rare_rate']):.6f}`, corresponding to rare enrichment `{float(nr5['rare_enrichment']):.6f}`.")
    if ro5 is not None:
        md.append(f"- `rarity_only` rare rate is `{float(ro5['rare_rate']):.6f}`, corresponding to rare enrichment `{float(ro5['rare_enrichment']):.6f}`.")
    if nr_vs_ro_rare is not None:
        md.append(f"- The paired bootstrap rare-rate difference `{reference_policy} - rarity_only` is `{float(nr_vs_ro_rare['reference_minus_comparison']):.6f}` with 95% CI [`{float(nr_vs_ro_rare['bootstrap_ci_low_95']):.6f}`, `{float(nr_vs_ro_rare['bootstrap_ci_high_95']):.6f}`].")
    if nr_vs_ro_nov is not None:
        md.append(f"- The paired bootstrap novelty difference `{reference_policy} - rarity_only` is `{float(nr_vs_ro_nov['reference_minus_comparison']):.6f}` with 95% CI [`{float(nr_vs_ro_nov['bootstrap_ci_low_95']):.6f}`, `{float(nr_vs_ro_nov['bootstrap_ci_high_95']):.6f}`].")
    md.append("")

    md.append("## Suggested manuscript wording\n")
    md.append("```latex")
    if ro5 is not None and nr_vs_ro_rare is not None and nr_vs_ro_nov is not None:
        md.append(
            f"We repeated the follow-up policy ablation using rankings reconstructed from the final ensemble probabilities. "
            f"At a 5\\% follow-up budget, the final \\texttt{{novelty\\_rarity}} policy selected \\textbf{{{int(nr5['n_selected'])}}} objects and achieved a rare-object rate of "
            f"\\textbf{{{float(nr5['rare_rate']):.4f}}}, corresponding to a rare-class enrichment of \\textbf{{{float(nr5['rare_enrichment']):.2f}$\\times$}} over the test-set baseline. "
            f"The purely rarity-driven baseline reached a rare-object rate of \\textbf{{{float(ro5['rare_rate']):.4f}}} and enrichment of \\textbf{{{float(ro5['rare_enrichment']):.2f}$\\times$}}. "
            f"The paired bootstrap difference in rare-object rate between \\texttt{{novelty\\_rarity}} and \\texttt{{rarity\\_only}} was "
            f"\\textbf{{{float(nr_vs_ro_rare['reference_minus_comparison']):.4f}}} "
            f"(95\\% CI [{float(nr_vs_ro_rare['bootstrap_ci_low_95']):.4f}, {float(nr_vs_ro_rare['bootstrap_ci_high_95']):.4f}]), "
            f"while the novelty score was higher for \\texttt{{novelty\\_rarity}} by \\textbf{{{float(nr_vs_ro_nov['reference_minus_comparison']):.4f}}} "
            f"(95\\% CI [{float(nr_vs_ro_nov['bootstrap_ci_low_95']):.4f}, {float(nr_vs_ro_nov['bootstrap_ci_high_95']):.4f}]). "
            f"This supports the final policy as a broker-like trade-off: it preserves strong rare-object enrichment while explicitly incorporating novelty."
        )
    else:
        md.append(
            f"We repeated the follow-up policy ablation using rankings reconstructed from the final ensemble probabilities. "
            f"At the 5\\% follow-up budget, the final policy achieved a rare-object rate of "
            f"\\textbf{{{float(nr5['rare_rate']):.4f}}} and rare-class enrichment of \\textbf{{{float(nr5['rare_enrichment']):.2f}$\\times$}}."
        )
    md.append("```\n")

    md.append("## Output files\n")
    for p in [
        summary_path, pairwise_path, overlap_path,
        output_dir / "followup_policy_ablation_schema.json",
        output_dir / "ensemble_followup_ranking_generation_schema.json",
        output_dir / "fig_policy_rare_enrichment_by_budget.png",
        output_dir / "fig_policy_novelty_by_budget.png",
        output_dir / "fig_policy_tradeoff_rare_vs_novelty.png",
    ]:
        if p.exists():
            md.append(f"- `{p}`")
    for policy in aligned:
        md.append(f"- `{output_dir / f'ensemble_test_ranking_{policy}.csv'}`")

    md_path = output_dir / "followup_policy_ablation_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--template-dir", required=True)
    parser.add_argument("--output-dir", default="results/ensemble_followup_policy_ablation_250k_final")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-policy", default="novelty_rarity")
    parser.add_argument("--id-col", default="object_id")
    parser.add_argument("--true-col", default="true_label")
    parser.add_argument("--pred-col", default="predicted_label")
    parser.add_argument("--template-prefix", default="hybrid_test_ranking")
    parser.add_argument("--output-prefix", default="ensemble_test_ranking")
    parser.add_argument("--noninferiority-margin-rare-rate", type=float, default=0.01)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    prob_path = Path(args.probabilities)
    template_dir = Path(args.template_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings, schema = build_ensemble_rankings(
        pred_path=pred_path,
        prob_path=prob_path,
        template_dir=template_dir,
        output_dir=output_dir,
        policies=args.policies,
        id_col=args.id_col,
        true_col=args.true_col,
        pred_col=args.pred_col,
        template_prefix=args.template_prefix,
        output_prefix=args.output_prefix,
    )

    run_ablation(
        rankings=rankings,
        output_dir=output_dir,
        budgets=args.budgets,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference_policy=args.reference_policy,
        id_col=args.id_col,
        noninferiority_margin_rare_rate=args.noninferiority_margin_rare_rate,
    )

    print(f"Done. Results written to: {output_dir}")
    print(f"Summary: {output_dir / 'followup_policy_ablation_summary.md'}")


if __name__ == "__main__":
    main()
