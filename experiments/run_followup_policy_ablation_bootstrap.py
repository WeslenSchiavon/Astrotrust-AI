#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_followup_policy_ablation_bootstrap.py

Compara estatisticamente políticas de follow-up usando rankings object-level.
Objetivo principal: justificar a política final `novelty_rarity` contra alternativas
como `rarity_only`, `previous_discovery` e `fixed_discovery`.

O script NÃO treina modelos e NÃO inventa dados. Ele usa os rankings já gerados.

Saídas:
- followup_policy_ablation_summary_by_budget.csv
- followup_policy_ablation_pairwise_vs_reference.csv
- followup_policy_ablation_overlap_vs_reference.csv
- followup_policy_ablation_summary.md
- fig_policy_rare_enrichment_by_budget.png
- fig_policy_novelty_by_budget.png
- fig_policy_tradeoff_rare_vs_novelty.png

Exemplo:
python .\\experiments\\run_followup_policy_ablation_bootstrap.py `
  --policy-ranking novelty_rarity=results\\hybrid_followup_policy_eval_250k\\hybrid_test_ranking_novelty_rarity.csv `
  --policy-ranking rarity_only=results\\hybrid_followup_policy_eval_250k\\hybrid_test_ranking_rarity_only.csv `
  --policy-ranking previous_discovery=results\\hybrid_followup_policy_eval_250k\\hybrid_test_ranking_previous_discovery.csv `
  --policy-ranking fixed_discovery=results\\hybrid_followup_policy_eval_250k\\hybrid_test_ranking_fixed_discovery.csv `
  --reference-policy novelty_rarity `
  --output-dir results\\followup_policy_ablation_bootstrap_250k `
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


DEFAULT_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]


def parse_policy_arg(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        path = Path(arg)
        name = path.stem
        for prefix in ["hybrid_test_ranking_", "followup_priority_ranking_"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name, path
    name, path = arg.split("=", 1)
    return name.strip(), Path(path.strip())


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


def infer_policy_name(df: pd.DataFrame, fallback: str) -> str:
    if "configuration" in df.columns:
        vals = df["configuration"].dropna().astype(str).unique()
        if len(vals) == 1:
            return vals[0]
    return fallback


def load_rankings(
    policy_args: List[str],
    id_col: str,
    score_col: str,
    rank_col: str,
) -> Dict[str, pd.DataFrame]:
    out = {}
    for arg in policy_args:
        fallback_name, path = parse_policy_arg(arg)
        if not path.exists():
            raise FileNotFoundError(f"Missing ranking file: {path}")

        df = pd.read_csv(path)
        name = infer_policy_name(df, fallback_name)

        if id_col not in df.columns:
            raise ValueError(f"{path} does not contain id column `{id_col}`")

        if score_col not in df.columns and rank_col not in df.columns:
            raise ValueError(f"{path} must contain `{score_col}` or `{rank_col}`")

        # Keep one row per object. Prefer best rank / highest score.
        if rank_col in df.columns:
            df = df.sort_values(rank_col, ascending=True)
        elif score_col in df.columns:
            df = df.sort_values(score_col, ascending=False)

        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
        out[name] = df

    return out


def align_to_common_objects(rankings: Dict[str, pd.DataFrame], id_col: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    common = None
    for df in rankings.values():
        ids = set(df[id_col].astype(str))
        common = ids if common is None else common & ids

    if not common:
        raise ValueError("No common object IDs across policy ranking files.")

    common_ids = sorted(common)
    aligned = {}
    for name, df in rankings.items():
        tmp = df.copy()
        tmp[id_col] = tmp[id_col].astype(str)
        tmp = tmp[tmp[id_col].isin(common_ids)].copy()
        tmp = tmp.set_index(id_col).loc[common_ids].reset_index()
        aligned[name] = tmp

    return common_ids, aligned


def get_selected_mask(df: pd.DataFrame, budget: float, score_col: str, rank_col: str) -> np.ndarray:
    n = len(df)
    k = max(1, int(round(n * budget)))

    if rank_col in df.columns:
        order = np.argsort(pd.to_numeric(df[rank_col], errors="coerce").to_numpy())
    else:
        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(-np.inf).to_numpy()
        order = np.argsort(-scores)

    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def safe_mean(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.nanmean(x))


def selected_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return safe_mean(values[mask])


def bootstrap_metric_for_policy(
    values: np.ndarray,
    selected_mask: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> np.ndarray:
    n = len(values)
    out = np.full(n_bootstrap, np.nan)
    idx_all = np.arange(n)
    for b in range(n_bootstrap):
        idx = rng.choice(idx_all, size=n, replace=True)
        denom = selected_mask[idx].sum()
        if denom > 0:
            out[b] = np.nanmean(values[idx][selected_mask[idx]])
    return out


def bootstrap_pairwise_diff(
    values: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> np.ndarray:
    n = len(values)
    out = np.full(n_bootstrap, np.nan)
    idx_all = np.arange(n)
    for b in range(n_bootstrap):
        idx = rng.choice(idx_all, size=n, replace=True)
        va = values[idx][mask_a[idx]]
        vb = values[idx][mask_b[idx]]
        if len(va) > 0 and len(vb) > 0:
            out[b] = np.nanmean(va) - np.nanmean(vb)
    return out


def ci95(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def pvalue_two_sided_against_zero(samples: np.ndarray) -> float:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan")
    p_le = np.mean(samples <= 0)
    p_ge = np.mean(samples >= 0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def pvalue_noninferior(samples: np.ndarray, margin: float) -> float:
    """
    One-sided bootstrap probability that diff <= -margin.
    If small, A is non-inferior to B within the chosen margin.
    """
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan")
    return float(np.mean(samples <= -abs(margin)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ranking", action="append", required=True,
                        help="Policy ranking as name=path. Repeat for each policy.")
    parser.add_argument("--reference-policy", default="novelty_rarity")
    parser.add_argument("--output-dir", default="results/followup_policy_ablation_bootstrap_250k")

    parser.add_argument("--id-col", default="object_id")
    parser.add_argument("--score-col", default="priority_score")
    parser.add_argument("--rank-col", default="priority_rank")
    parser.add_argument("--rare-col", default="true_is_rare")
    parser.add_argument("--novelty-col", default="novelty_score")
    parser.add_argument("--raw-novelty-col", default="raw_novelty")
    parser.add_argument("--uncertainty-col", default="uncertainty_score")
    parser.add_argument("--confidence-col", default="confidence")
    parser.add_argument("--correct-col", default="correct")

    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # Non-inferiority margin for rare enrichment/rate comparison.
    # margin=0.01 means novelty_rarity is considered practically non-inferior
    # if rare_rate is not more than 1 percentage point below rarity_only.
    parser.add_argument("--noninferiority-margin-rare-rate", type=float, default=0.01)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rankings = load_rankings(args.policy_ranking, args.id_col, args.score_col, args.rank_col)
    common_ids, rankings = align_to_common_objects(rankings, args.id_col)

    if args.reference_policy not in rankings:
        raise ValueError(
            f"Reference policy `{args.reference_policy}` not found. "
            f"Available policies: {sorted(rankings)}"
        )

    # Use the reference file for object-level labels and auxiliary metrics.
    ref_df = rankings[args.reference_policy]
    required = [args.rare_col]
    for c in required:
        if c not in ref_df.columns:
            raise ValueError(f"Reference ranking is missing required column `{c}`")

    rare = as_bool_float(ref_df[args.rare_col])
    baseline_rare_rate = float(np.mean(rare))

    aux_values = {}
    for label, col in [
        ("novelty", args.novelty_col),
        ("raw_novelty", args.raw_novelty_col),
        ("uncertainty", args.uncertainty_col),
        ("confidence", args.confidence_col),
        ("correct", args.correct_col),
        ("priority_score", args.score_col),
    ]:
        if col in ref_df.columns:
            if label == "correct":
                aux_values[label] = as_bool_float(ref_df[col])
            else:
                aux_values[label] = pd.to_numeric(ref_df[col], errors="coerce").to_numpy()

    rng = np.random.default_rng(args.seed)

    selected_masks = {
        policy: {
            budget: get_selected_mask(df, budget, args.score_col, args.rank_col)
            for budget in args.budgets
        }
        for policy, df in rankings.items()
    }

    summary_rows = []
    pairwise_rows = []
    overlap_rows = []

    for budget in args.budgets:
        ref_mask = selected_masks[args.reference_policy][budget]
        ref_k = int(ref_mask.sum())

        for policy, df in rankings.items():
            mask = selected_masks[policy][budget]
            selected_n = int(mask.sum())
            rare_rate = selected_mean(rare, mask)
            rare_enrichment = rare_rate / baseline_rare_rate if baseline_rare_rate > 0 else float("nan")

            rare_boot = bootstrap_metric_for_policy(rare, mask, rng, args.n_bootstrap)
            rare_ci_low, rare_ci_high = ci95(rare_boot)
            enrich_ci_low = rare_ci_low / baseline_rare_rate if baseline_rare_rate > 0 else float("nan")
            enrich_ci_high = rare_ci_high / baseline_rare_rate if baseline_rare_rate > 0 else float("nan")

            row = {
                "budget_fraction": budget,
                "policy": policy,
                "n_objects_common": len(common_ids),
                "n_selected": selected_n,
                "baseline_rare_rate": baseline_rare_rate,
                "rare_rate": rare_rate,
                "rare_rate_ci_low_95": rare_ci_low,
                "rare_rate_ci_high_95": rare_ci_high,
                "rare_enrichment": rare_enrichment,
                "rare_enrichment_ci_low_95": enrich_ci_low,
                "rare_enrichment_ci_high_95": enrich_ci_high,
                "n_bootstrap": args.n_bootstrap,
            }

            for label, vals in aux_values.items():
                row[f"mean_{label}"] = selected_mean(vals, mask)

            summary_rows.append(row)

            inter = int(np.logical_and(ref_mask, mask).sum())
            union = int(np.logical_or(ref_mask, mask).sum())
            overlap_rows.append({
                "budget_fraction": budget,
                "reference_policy": args.reference_policy,
                "policy": policy,
                "reference_n_selected": ref_k,
                "policy_n_selected": selected_n,
                "intersection_n": inter,
                "union_n": union,
                "jaccard_overlap": inter / union if union > 0 else float("nan"),
                "overlap_fraction_of_reference": inter / ref_k if ref_k > 0 else float("nan"),
                "overlap_fraction_of_policy": inter / selected_n if selected_n > 0 else float("nan"),
            })

        # Pairwise comparisons: reference vs each other policy.
        for policy in rankings:
            if policy == args.reference_policy:
                continue

            ref_mask = selected_masks[args.reference_policy][budget]
            other_mask = selected_masks[policy][budget]

            # Main comparison: rare rate difference.
            rare_diff_boot = bootstrap_pairwise_diff(rare, ref_mask, other_mask, rng, args.n_bootstrap)
            rare_diff = selected_mean(rare, ref_mask) - selected_mean(rare, other_mask)
            ci_low, ci_high = ci95(rare_diff_boot)

            row = {
                "budget_fraction": budget,
                "reference_policy": args.reference_policy,
                "comparison_policy": policy,
                "metric": "rare_rate",
                "reference_minus_comparison": rare_diff,
                "bootstrap_ci_low_95": ci_low,
                "bootstrap_ci_high_95": ci_high,
                "two_sided_bootstrap_pvalue": pvalue_two_sided_against_zero(rare_diff_boot),
                "noninferiority_margin": args.noninferiority_margin_rare_rate,
                "one_sided_pvalue_reference_more_than_margin_worse": pvalue_noninferior(
                    rare_diff_boot,
                    args.noninferiority_margin_rare_rate
                ),
                "n_bootstrap": args.n_bootstrap,
            }
            pairwise_rows.append(row)

            # Additional operational metrics.
            for label in ["novelty", "raw_novelty", "uncertainty", "confidence", "correct"]:
                if label not in aux_values:
                    continue
                vals = aux_values[label]
                diff_boot = bootstrap_pairwise_diff(vals, ref_mask, other_mask, rng, args.n_bootstrap)
                diff = selected_mean(vals, ref_mask) - selected_mean(vals, other_mask)
                cilo, cihi = ci95(diff_boot)
                pairwise_rows.append({
                    "budget_fraction": budget,
                    "reference_policy": args.reference_policy,
                    "comparison_policy": policy,
                    "metric": label,
                    "reference_minus_comparison": diff,
                    "bootstrap_ci_low_95": cilo,
                    "bootstrap_ci_high_95": cihi,
                    "two_sided_bootstrap_pvalue": pvalue_two_sided_against_zero(diff_boot),
                    "noninferiority_margin": np.nan,
                    "one_sided_pvalue_reference_more_than_margin_worse": np.nan,
                    "n_bootstrap": args.n_bootstrap,
                })

    summary_df = pd.DataFrame(summary_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    summary_path = out_dir / "followup_policy_ablation_summary_by_budget.csv"
    pairwise_path = out_dir / "followup_policy_ablation_pairwise_vs_reference.csv"
    overlap_path = out_dir / "followup_policy_ablation_overlap_vs_reference.csv"

    summary_df.to_csv(summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    schema = {
        "policy_files": {k: str(v) for k, v in [parse_policy_arg(a) for a in args.policy_ranking]},
        "policies_loaded": list(rankings.keys()),
        "reference_policy": args.reference_policy,
        "id_col": args.id_col,
        "score_col": args.score_col,
        "rank_col": args.rank_col,
        "rare_col": args.rare_col,
        "budgets": args.budgets,
        "n_bootstrap": args.n_bootstrap,
        "n_common_objects": len(common_ids),
        "baseline_rare_rate": baseline_rare_rate,
    }
    (out_dir / "followup_policy_ablation_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    # Figures
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
        plt.savefig(out_dir / "fig_policy_rare_enrichment_by_budget.png", dpi=200)
        plt.close()

        if "mean_novelty" in summary_df.columns:
            pivot_nov = summary_df.pivot(index="budget_fraction", columns="policy", values="mean_novelty")
            plt.figure(figsize=(8, 5))
            for col in pivot_nov.columns:
                plt.plot(pivot_nov.index * 100, pivot_nov[col], marker="o", label=col)
            plt.xlabel("Follow-up budget (%)")
            plt.ylabel("Mean novelty score")
            plt.title("Mean novelty by follow-up policy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "fig_policy_novelty_by_budget.png", dpi=200)
            plt.close()

            budget_for_tradeoff = 0.05 if 0.05 in args.budgets else args.budgets[0]
            trade = summary_df[summary_df["budget_fraction"] == budget_for_tradeoff].copy()
            plt.figure(figsize=(7, 5))
            plt.scatter(trade["mean_novelty"], trade["rare_enrichment"])
            for _, r in trade.iterrows():
                plt.annotate(r["policy"], (r["mean_novelty"], r["rare_enrichment"]))
            plt.xlabel("Mean novelty score")
            plt.ylabel("Rare-class enrichment")
            plt.title(f"Policy trade-off at {budget_for_tradeoff:.0%} budget")
            plt.tight_layout()
            plt.savefig(out_dir / "fig_policy_tradeoff_rare_vs_novelty.png", dpi=200)
            plt.close()
    except Exception as exc:
        print(f"[WARN] Could not generate one or more figures: {exc}")

    # Markdown summary
    md = []
    md.append("# Follow-up policy ablation bootstrap\n")
    md.append("This report compares object-level follow-up policies under fixed observing budgets.\n")
    md.append("The goal is to justify the final `novelty_rarity` policy against simpler alternatives such as `rarity_only`.\n")

    md.append("## Inputs\n")
    for policy, df in rankings.items():
        md.append(f"- `{policy}`: {len(df)} aligned objects")
    md.append(f"- Common objects across policies: `{len(common_ids)}`")
    md.append(f"- Baseline rare rate: `{baseline_rare_rate:.6f}`")
    md.append(f"- Bootstrap iterations: `{args.n_bootstrap}`")
    md.append(f"- Reference policy: `{args.reference_policy}`\n")

    md.append("## Summary by budget\n")
    cols = [
        "budget_fraction", "policy", "n_selected", "rare_rate", "rare_rate_ci_low_95",
        "rare_rate_ci_high_95", "rare_enrichment", "rare_enrichment_ci_low_95",
        "rare_enrichment_ci_high_95"
    ]
    extra_cols = [c for c in ["mean_novelty", "mean_uncertainty", "mean_confidence", "mean_correct"] if c in summary_df.columns]
    md.append(summary_df[cols + extra_cols].to_markdown(index=False))
    md.append("")

    md.append("## Pairwise bootstrap comparisons versus reference\n")
    md.append(pairwise_df.to_markdown(index=False))
    md.append("")

    md.append("## Overlap with reference policy\n")
    md.append(overlap_df.to_markdown(index=False))
    md.append("")

    md.append("## Suggested manuscript interpretation\n")
    md.append(
        "The ablation quantifies whether the final `novelty_rarity` policy preserves the rare-class enrichment "
        "of a purely rarity-driven ranking while adding explicit sensitivity to novelty. A practically useful outcome "
        "is that `novelty_rarity` remains statistically close to `rarity_only` in rare-object enrichment, but yields "
        "a more balanced broker-like prioritization objective because it incorporates both rarity and novelty."
    )
    md.append("")

    md.append("## Output files\n")
    for p in [
        summary_path, pairwise_path, overlap_path,
        out_dir / "followup_policy_ablation_schema.json",
        out_dir / "fig_policy_rare_enrichment_by_budget.png",
        out_dir / "fig_policy_novelty_by_budget.png",
        out_dir / "fig_policy_tradeoff_rare_vs_novelty.png",
    ]:
        if p.exists():
            md.append(f"- `{p}`")

    md_path = out_dir / "followup_policy_ablation_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
