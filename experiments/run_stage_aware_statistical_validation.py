#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical validation for the AstroTrust-AI stage-aware prototype.

Computes the same statistical evidence used for previous model blocks:
point metrics, bootstrap confidence intervals, paired bootstrap differences,
McNemar tests, follow-up bootstrap CIs, and permutation tests for rare-object
enrichment.

Default input:
  results/broker_like_stage_aware_ensemble/
    predictions/{model}/{scenario}_predictions.csv
    probabilities/{model}/{scenario}_probabilities.npy

Run:
  python .\experiments\run_stage_aware_statistical_validation.py `
    --input-dir results\broker_like_stage_aware_ensemble `
    --output-dir results\broker_like_stage_aware_statistical_validation `
    --n-bootstrap 1000 `
    --n-permutations 5000

Outputs are also copied to:
  results/final_publication/broker_like_realism/
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

try:
    from scipy import stats
except Exception:
    stats = None

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "results" / "broker_like_stage_aware_ensemble"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "broker_like_stage_aware_statistical_validation"
DEFAULT_FINAL_DIR = ROOT_DIR / "results" / "final_publication" / "broker_like_realism"

DEFAULT_SCENARIOS = [
    "full_curve_reference", "first_3_points", "first_5_points", "first_10_points",
    "first_20_points", "window_2_days", "window_7_days", "window_14_days", "window_30_days",
]
DEFAULT_MODELS = [
    "hybrid_full_trained", "hybrid_early_aware",
    "ensemble_hybrid_dominant_early_aware", "stage_aware_astrotrust_ai",
]
DEFAULT_PAIRS = [
    "ensemble_hybrid_dominant_early_aware:hybrid_early_aware",
    "ensemble_hybrid_dominant_early_aware:hybrid_full_trained",
    "stage_aware_astrotrust_ai:hybrid_full_trained",
    "stage_aware_astrotrust_ai:hybrid_early_aware",
]
EPS = 1e-12


def parse_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_pairs(value: str | None) -> list[tuple[str, str]]:
    out = []
    for item in parse_list(value, DEFAULT_PAIRS):
        a, b = item.split(":", 1)
        out.append((a.strip(), b.strip()))
    return out


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def scenario_order(s: str) -> int:
    return {v: i for i, v in enumerate(DEFAULT_SCENARIOS)}.get(s, 999)


def model_order(m: str) -> int:
    return {v: i for i, v in enumerate(DEFAULT_MODELS)}.get(m, 999)


def sanitize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=EPS, posinf=1.0, neginf=EPS)
    p = np.clip(p, EPS, 1.0)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def ece_score(y: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
    probs = sanitize_probs(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    corr = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def topk_correct(y: np.ndarray, probs: np.ndarray, k: int) -> np.ndarray:
    k = min(k, probs.shape[1])
    top = np.argsort(probs, axis=1)[:, -k:]
    return np.array([int(t) in row for t, row in zip(y, top)], dtype=bool)


def metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray, n_bins: int) -> dict[str, float]:
    corr = pred.astype(int) == y.astype(int)
    return {
        "accuracy": float(np.mean(corr)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "top3_accuracy": float(np.mean(topk_correct(y, probs, 3))),
        "top5_accuracy": float(np.mean(topk_correct(y, probs, 5))),
        "ece": ece_score(y, probs, n_bins),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "mean_uncertainty": float((1.0 - probs.max(axis=1)).mean()),
    }


def ci(vals: np.ndarray, level: float) -> tuple[float, float]:
    a = (100.0 - level) / 2.0
    return float(np.nanpercentile(vals, a)), float(np.nanpercentile(vals, 100.0 - a))


def load_pair(input_dir: Path, model: str, scenario: str) -> tuple[pd.DataFrame, np.ndarray]:
    pred_path = input_dir / "predictions" / model / f"{scenario}_predictions.csv"
    prob_path = input_dir / "probabilities" / model / f"{scenario}_probabilities.npy"
    if not pred_path.exists() or not prob_path.exists():
        raise FileNotFoundError(f"Missing {model}/{scenario}: {pred_path} or {prob_path}")
    pred = pd.read_csv(pred_path)
    probs = sanitize_probs(np.load(prob_path))
    if len(pred) != len(probs):
        raise ValueError(f"Length mismatch for {model}/{scenario}")
    return pred, probs


def subset_common(pred: pd.DataFrame, probs: np.ndarray, ids: set) -> tuple[pd.DataFrame, np.ndarray]:
    mask = pred["object_id"].isin(ids).to_numpy()
    return pred.loc[mask].reset_index(drop=True), probs[mask]


def align(a: pd.DataFrame, pa: np.ndarray, b: pd.DataFrame, pb: np.ndarray):
    ia = pd.DataFrame({"object_id": a["object_id"].to_numpy(), "ia": np.arange(len(a))})
    ib = pd.DataFrame({"object_id": b["object_id"].to_numpy(), "ib": np.arange(len(b))})
    m = ia.merge(ib, on="object_id", how="inner")
    if m.empty:
        raise ValueError("No common objects")
    xa = m["ia"].to_numpy(dtype=int)
    xb = m["ib"].to_numpy(dtype=int)
    return a.iloc[xa].reset_index(drop=True), pa[xa], b.iloc[xb].reset_index(drop=True), pb[xb]


def bootstrap_metric_ci(pred: pd.DataFrame, probs: np.ndarray, model: str, scenario: str, comp: str, n_boot: int, level: float, rng, n_bins: int) -> pd.DataFrame:
    y = pred["true_label"].astype(int).to_numpy()
    yp = pred["predicted_label"].astype(int).to_numpy()
    n = len(y)
    names = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "top5_accuracy", "ece"]
    vals = {k: [] for k in names}
    point = metrics(y, yp, probs, n_bins)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mm = metrics(y[idx], yp[idx], probs[idx], n_bins)
        for k in names:
            vals[k].append(mm[k])
    rows = []
    for k in names:
        arr = np.asarray(vals[k], dtype=float)
        lo, hi = ci(arr, level)
        rows.append({"model_name": model, "comparison_set": comp, "scenario": scenario, "metric": k,
                     "n_objects": n, "point_estimate": point[k], "bootstrap_mean": float(arr.mean()),
                     "ci_level": level, "ci_low": lo, "ci_high": hi, "bootstrap_sd": float(arr.std(ddof=1))})
    return pd.DataFrame(rows)


def paired_bootstrap(tpred, tprob, rpred, rprob, target, ref, scenario, comp, n_boot, level, rng, n_bins):
    tpred, tprob, rpred, rprob = align(tpred, tprob, rpred, rprob)
    y = tpred["true_label"].astype(int).to_numpy()
    yt = tpred["predicted_label"].astype(int).to_numpy()
    yr = rpred["predicted_label"].astype(int).to_numpy()
    n = len(y)
    names = ["accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "top5_accuracy", "ece"]
    mt = metrics(y, yt, tprob, n_bins)
    mr = metrics(y, yr, rprob, n_bins)
    rows = []
    for name in names:
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            vals.append(metrics(y[idx], yt[idx], tprob[idx], n_bins)[name] - metrics(y[idx], yr[idx], rprob[idx], n_bins)[name])
        vals = np.asarray(vals, dtype=float)
        lo, hi = ci(vals, level)
        delta = mt[name] - mr[name]
        p = float(np.mean(vals <= 0)) if delta >= 0 else float(np.mean(vals >= 0))
        rows.append({"target_model": target, "reference_model": ref, "comparison_set": comp, "scenario": scenario,
                     "metric": name, "n_common_objects": n, "target_point": mt[name], "reference_point": mr[name],
                     "delta_target_minus_reference": delta, "bootstrap_mean_delta": float(vals.mean()),
                     "ci_level": level, "ci_low": lo, "ci_high": hi, "bootstrap_sd": float(vals.std(ddof=1)),
                     "one_sided_bootstrap_p_against_zero": p})
    return pd.DataFrame(rows)


def mcnemar(tpred, tprob, rpred, rprob, target, ref, scenario, comp):
    tpred, tprob, rpred, rprob = align(tpred, tprob, rpred, rprob)
    y = tpred["true_label"].astype(int).to_numpy()
    ct = tpred["predicted_label"].astype(int).to_numpy() == y
    cr = rpred["predicted_label"].astype(int).to_numpy() == y
    b = int(np.sum(ct & ~cr))
    c = int(np.sum(~ct & cr))
    p = np.nan
    if stats is not None and (b + c) > 0:
        p = float(stats.binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)
    return {"target_model": target, "reference_model": ref, "comparison_set": comp, "scenario": scenario,
            "n_common_objects": len(y), "target_accuracy": float(ct.mean()), "reference_accuracy": float(cr.mean()),
            "delta_accuracy": float(ct.mean() - cr.mean()), "target_correct_reference_wrong": b,
            "target_wrong_reference_correct": c, "mcnemar_exact_pvalue": p}


def score(pred: pd.DataFrame, policy: str) -> np.ndarray:
    if policy == "novelty_rarity":
        if "priority_novelty_rarity" in pred.columns:
            return pred["priority_novelty_rarity"].to_numpy(float)
        return 0.5 * pred["novelty"].to_numpy(float) + 0.5 * pred["rarity_score"].to_numpy(float)
    if policy == "rarity_only":
        return pred["rarity_score"].to_numpy(float)
    if policy == "uncertainty_only":
        return pred["uncertainty"].to_numpy(float)
    raise ValueError(policy)


def follow_point(pred: pd.DataFrame, scores: np.ndarray, budget: float) -> dict:
    n = len(pred)
    k = max(1, int(math.ceil(n * budget)))
    order = np.argsort(scores)[::-1]
    sel = pred.iloc[order[:k]]
    base = float(pred["true_is_rare"].astype(bool).mean())
    rr = float(sel["true_is_rare"].astype(bool).mean())
    return {"n_available": n, "n_selected": k, "baseline_rare_rate": base, "rare_rate": rr,
            "rare_enrichment": rr / base if base > 0 else np.nan, "mean_correct": float(sel["correct"].mean()),
            "mean_uncertainty": float(sel["uncertainty"].mean()), "mean_novelty": float(sel["novelty"].mean()),
            "mean_rarity_score": float(sel["rarity_score"].mean())}


def follow_boot(pred, model, scenario, comp, policies, budgets, n_boot, level, rng):
    rows = []
    n = len(pred)
    for pol in policies:
        sfull = score(pred, pol)
        for budget in budgets:
            point = follow_point(pred, sfull, budget)
            rrs, ens, mcs = [], [], []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                val = follow_point(pred.iloc[idx].reset_index(drop=True), sfull[idx], budget)
                rrs.append(val["rare_rate"]); ens.append(val["rare_enrichment"]); mcs.append(val["mean_correct"])
            rrlo, rrhi = ci(np.asarray(rrs), level)
            elo, ehi = ci(np.asarray(ens), level)
            mlo, mhi = ci(np.asarray(mcs), level)
            rows.append({"model_name": model, "comparison_set": comp, "scenario": scenario, "policy": pol,
                         "budget_fraction": budget, **point, "ci_level": level,
                         "rare_rate_ci_low": rrlo, "rare_rate_ci_high": rrhi,
                         "rare_enrichment_ci_low": elo, "rare_enrichment_ci_high": ehi,
                         "mean_correct_ci_low": mlo, "mean_correct_ci_high": mhi})
    return pd.DataFrame(rows)


def perm_test(pred, model, scenario, comp, policy, budget, n_perm, rng):
    s = score(pred, policy)
    point = follow_point(pred, s, budget)
    rare = pred["true_is_rare"].astype(bool).to_numpy()
    n = len(rare); k = point["n_selected"]
    vals = np.asarray([rare[rng.choice(n, size=k, replace=False)].mean() for _ in range(n_perm)], dtype=float)
    lo, hi = ci(vals, 95.0)
    obs = point["rare_rate"]
    return {"model_name": model, "comparison_set": comp, "scenario": scenario, "policy": policy,
            "budget_fraction": budget, "n_available": n, "n_selected": k,
            "observed_rare_rate": obs, "observed_rare_enrichment": point["rare_enrichment"],
            "null_mean_rare_rate": float(vals.mean()), "null_ci_low": lo, "null_ci_high": hi,
            "permutation_pvalue_observed_ge_null": float((np.sum(vals >= obs) + 1) / (len(vals) + 1)),
            "n_permutations": n_perm}


def write_summary(path, point, boot, paired, mc, follow, perm, args):
    lines = ["# Stage-aware statistical validation", "", "## Protocol", "",
             "This report provides point estimates, bootstrap confidence intervals, paired bootstrap differences, McNemar tests, and follow-up prioritization uncertainty for the stage-aware AstroTrust-AI prototype.", "",
             f"- Bootstrap iterations: `{args.n_bootstrap}`", f"- Permutations: `{args.n_permutations}`", f"- CI level: `{args.ci}`", ""]
    lines.append("## Key point metrics")
    lines.append("")
    key = point[(point["comparison_set"] == "all_available_objects") &
                (point["scenario"].isin(["full_curve_reference", "first_5_points", "first_10_points", "first_20_points", "window_7_days", "window_30_days"]))]
    cols = ["model_name", "scenario", "n_objects", "accuracy", "macro_f1", "top3_accuracy", "top5_accuracy", "ece"]
    lines.append(key[cols].to_markdown(index=False, floatfmt=".4f") if not key.empty else "_No point metrics._")
    lines.append("\n## Paired bootstrap differences")
    showp = paired[(paired["comparison_set"] == "all_available_objects") & (paired["metric"].isin(["accuracy", "macro_f1", "top5_accuracy"]))]
    cols = ["target_model", "reference_model", "scenario", "metric", "delta_target_minus_reference", "ci_low", "ci_high", "one_sided_bootstrap_p_against_zero"]
    lines.append(showp[cols].to_markdown(index=False, floatfmt=".4f") if not showp.empty else "_No paired differences._")
    lines.append("\n## Follow-up bootstrap at 5% budget")
    showf = follow[(follow["comparison_set"] == "all_available_objects") & (follow["policy"] == "novelty_rarity") & (np.isclose(follow["budget_fraction"], 0.05))]
    cols = ["model_name", "scenario", "rare_enrichment", "rare_enrichment_ci_low", "rare_enrichment_ci_high", "mean_correct", "mean_correct_ci_low", "mean_correct_ci_high"]
    lines.append(showf[cols].to_markdown(index=False, floatfmt=".4f") if not showf.empty else "_No follow-up bootstrap._")
    lines.append("\n## Manuscript constraint")
    lines.append("```latex")
    lines.append(r"All broker-like stage-aware results are reported with object-level bootstrap confidence intervals and paired tests against the corresponding full-curve-trained and early-aware baselines. The complete-light-curve claim remains attached to the original \texttt{ensemble\_hybrid\_dominant}, whereas partial-light-curve claims are attached to the early-aware stage-aware route.")
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    ap.add_argument("--models", default="all")
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--pairs", default="all")
    ap.add_argument("--budgets", default="0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--policies", default="novelty_rarity,rarity_only,uncertainty_only")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-permutations", type=int, default=5000)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    models = parse_list(args.models, DEFAULT_MODELS)
    scenarios = parse_list(args.scenarios, DEFAULT_SCENARIOS)
    pairs = parse_pairs(args.pairs)
    budgets = parse_float_list(args.budgets)
    policies = parse_list(args.policies, ["novelty_rarity", "rarity_only", "uncertainty_only"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for m in models:
        loaded[m] = {}
        for s in scenarios:
            try:
                loaded[m][s] = load_pair(args.input_dir, m, s)
                print(f"[OK] loaded {m}/{s}")
            except Exception as exc:
                print(f"[WARN] skip {m}/{s}: {exc}")

    common = {}
    for m, smap in loaded.items():
        ids = None
        for pred, _p in smap.values():
            ss = set(pred["object_id"].to_numpy())
            ids = ss if ids is None else ids.intersection(ss)
        common[m] = ids or set()

    point_rows, boot_frames, follow_frames, perm_rows = [], [], [], []
    for m, smap in loaded.items():
        for s, (pred, probs) in smap.items():
            datasets = [("all_available_objects", pred, probs)]
            if common[m]: datasets.append(("common_object_subset", *subset_common(pred, probs, common[m])))
            for comp, ppred, pprob in datasets:
                if len(ppred) < 2: continue
                y = ppred["true_label"].astype(int).to_numpy(); yp = ppred["predicted_label"].astype(int).to_numpy()
                point_rows.append({"model_name": m, "comparison_set": comp, "scenario": s, "n_objects": len(ppred),
                                   **metrics(y, yp, pprob, args.n_bins),
                                   "rare_true_rate": float(ppred["true_is_rare"].mean()),
                                   "mean_novelty": float(ppred["novelty"].mean()),
                                   "mean_rarity_score": float(ppred["rarity_score"].mean())})
                boot_frames.append(bootstrap_metric_ci(ppred, pprob, m, s, comp, args.n_bootstrap, args.ci, rng, args.n_bins))
                follow_frames.append(follow_boot(ppred, m, s, comp, policies, budgets, args.n_bootstrap, args.ci, rng))
                if args.n_permutations > 0:
                    perm_rows.append(perm_test(ppred, m, s, comp, "novelty_rarity", 0.05, args.n_permutations, rng))

    pair_frames, mc_rows = [], []
    for target, ref in pairs:
        for s in sorted(set(loaded.get(target, {})).intersection(loaded.get(ref, {})), key=scenario_order):
            tp, tprob = loaded[target][s]; rp, rprob = loaded[ref][s]
            try:
                pair_frames.append(paired_bootstrap(tp, tprob, rp, rprob, target, ref, s, "all_available_objects", args.n_bootstrap, args.ci, rng, args.n_bins))
                mc_rows.append(mcnemar(tp, tprob, rp, rprob, target, ref, s, "all_available_objects"))
            except Exception as exc:
                print(f"[WARN] paired failed {target}:{ref}/{s}: {exc}")
            both = common.get(target, set()).intersection(common.get(ref, set()))
            if both:
                ctp, ctprob = subset_common(tp, tprob, both); crp, crprob = subset_common(rp, rprob, both)
                try:
                    pair_frames.append(paired_bootstrap(ctp, ctprob, crp, crprob, target, ref, s, "common_object_subset", args.n_bootstrap, args.ci, rng, args.n_bins))
                    mc_rows.append(mcnemar(ctp, ctprob, crp, crprob, target, ref, s, "common_object_subset"))
                except Exception as exc:
                    print(f"[WARN] paired common failed {target}:{ref}/{s}: {exc}")

    point = pd.DataFrame(point_rows)
    boot = pd.concat(boot_frames, ignore_index=True) if boot_frames else pd.DataFrame()
    paired = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    mc = pd.DataFrame(mc_rows)
    follow = pd.concat(follow_frames, ignore_index=True) if follow_frames else pd.DataFrame()
    perm = pd.DataFrame(perm_rows)

    for df in [point, boot, follow, perm]:
        if not df.empty:
            df["_so"] = df["scenario"].map(scenario_order)
            if "model_name" in df: df["_mo"] = df["model_name"].map(model_order)
            df.sort_values([c for c in ["comparison_set", "_so", "_mo", "metric", "policy", "budget_fraction"] if c in df], inplace=True)
            df.drop(columns=[c for c in ["_so", "_mo"] if c in df], inplace=True)
    for df in [paired, mc]:
        if not df.empty:
            df["_so"] = df["scenario"].map(scenario_order)
            df.sort_values([c for c in ["comparison_set", "target_model", "reference_model", "_so", "metric"] if c in df], inplace=True)
            df.drop(columns=["_so"], inplace=True)

    outputs = {
        "stage_aware_point_metrics.csv": point,
        "stage_aware_bootstrap_metric_ci.csv": boot,
        "stage_aware_paired_bootstrap_differences.csv": paired,
        "stage_aware_mcnemar_tests.csv": mc,
        "stage_aware_followup_bootstrap_ci.csv": follow,
        "stage_aware_followup_permutation_tests.csv": perm,
    }
    for name, df in outputs.items():
        p = args.output_dir / name
        df.to_csv(p, index=False)
        shutil.copyfile(p, args.final_dir / name)
        print(f"[OK] saved {p}")

    summary = args.output_dir / "stage_aware_statistical_validation_summary.md"
    write_summary(summary, point, boot, paired, mc, follow, perm, args)
    shutil.copyfile(summary, args.final_dir / summary.name)
    print(f"[OK] saved {summary}")
    print(f"[OK] copied outputs to {args.final_dir}")


if __name__ == "__main__":
    main()
