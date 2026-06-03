#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast statistical validation for AstroTrust-AI stage-aware partial light-curve results.
Adds terminal progress logs and avoids expensive repeated sklearn calls in bootstrap loops.

Recommended:
python .\experiments\run_stage_aware_statistical_validation_fast.py `
  --input-dir results\broker_like_stage_aware_ensemble `
  --output-dir results\broker_like_stage_aware_statistical_validation_fast `
  --partial-only `
  --n-bootstrap 1000 `
  --n-permutations 5000 `
  --log-every 100
"""
from __future__ import annotations

import argparse, math, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "results" / "broker_like_stage_aware_ensemble"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "broker_like_stage_aware_statistical_validation_fast"
DEFAULT_FINAL_DIR = ROOT_DIR / "results" / "final_publication" / "broker_like_realism"
ALL_SCENARIOS = ["full_curve_reference","first_3_points","first_5_points","first_10_points","first_20_points","window_2_days","window_7_days","window_14_days","window_30_days"]
PARTIAL_SCENARIOS = [s for s in ALL_SCENARIOS if s != "full_curve_reference"]
DEFAULT_MODELS = ["hybrid_full_trained","hybrid_early_aware","ensemble_hybrid_dominant_early_aware","stage_aware_astrotrust_ai"]
DEFAULT_PAIRS = ["ensemble_hybrid_dominant_early_aware:hybrid_early_aware","ensemble_hybrid_dominant_early_aware:hybrid_full_trained","stage_aware_astrotrust_ai:hybrid_full_trained"]
EPS = 1e-12

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def parse_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or str(value).strip().lower() in {"", "all"}:
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]

def parse_pairs(value: str | None) -> list[tuple[str, str]]:
    pairs=[]
    for item in parse_list(value, DEFAULT_PAIRS):
        if ":" not in item: raise ValueError(f"Invalid pair `{item}`. Use target:reference.")
        a,b=item.split(":",1); pairs.append((a.strip(),b.strip()))
    return pairs

def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]

def sanitize_probabilities(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,dtype=np.float64); p=np.nan_to_num(p,nan=EPS,posinf=1.0,neginf=EPS); p=np.clip(p,EPS,1.0)
    return p/np.maximum(p.sum(axis=1,keepdims=True),EPS)

def load_prediction_and_probs(input_dir: Path, model: str, scenario: str):
    pred_path=input_dir/"predictions"/model/f"{scenario}_predictions.csv"
    prob_path=input_dir/"probabilities"/model/f"{scenario}_probabilities.npy"
    if not pred_path.exists() or not prob_path.exists(): raise FileNotFoundError(f"Missing {model}/{scenario}")
    pred=pd.read_csv(pred_path); probs=sanitize_probabilities(np.load(prob_path))
    if len(pred)!=len(probs): raise ValueError(f"Length mismatch for {model}/{scenario}: {len(pred)} vs {len(probs)}")
    return pred, probs

def topk_correct(y: np.ndarray, probs: np.ndarray, k: int) -> np.ndarray:
    k=min(k, probs.shape[1]); top=np.argpartition(probs, kth=probs.shape[1]-k, axis=1)[:,-k:]
    return np.array([yy in row for yy,row in zip(y,top)], dtype=bool)

def confusion_fast(y: np.ndarray, yp: np.ndarray, n_classes: int, weights: np.ndarray | None = None) -> np.ndarray:
    idx=y.astype(int)*n_classes+yp.astype(int)
    return np.bincount(idx, weights=weights, minlength=n_classes*n_classes).reshape(n_classes,n_classes).astype(float)

def f1_from_cm(cm: np.ndarray) -> tuple[float,float]:
    tp=np.diag(cm); support=cm.sum(axis=1); predsum=cm.sum(axis=0)
    denom=2*tp+(predsum-tp)+(support-tp)
    f1=np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
    total=support.sum(); return float(np.mean(f1)), float(np.sum(f1*support)/total) if total>0 else np.nan

def balanced_acc_from_cm(cm: np.ndarray) -> float:
    support=cm.sum(axis=1); rec=np.divide(np.diag(cm),support,out=np.zeros_like(support,dtype=float),where=support>0)
    return float(np.mean(rec))

def ece_from_arrays(correct: np.ndarray, conf: np.ndarray, n_bins: int=15, weights: np.ndarray | None=None) -> float:
    if weights is None: weights=np.ones(len(correct),dtype=float)
    total_w=float(weights.sum()); edges=np.linspace(0,1,n_bins+1); ece=0.0
    if total_w<=0: return np.nan
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        mask=(conf>=lo)&(conf<=hi) if i==n_bins-1 else (conf>=lo)&(conf<hi)
        if not mask.any(): continue
        w=weights[mask]; sw=float(w.sum())
        if sw<=0: continue
        acc=float(np.sum(correct[mask]*w)/sw); cavg=float(np.sum(conf[mask]*w)/sw)
        ece += (sw/total_w)*abs(acc-cavg)
    return float(ece)

def metric_bundle(y, yp, probs, n_bins, weights=None, top3=None, top5=None) -> dict:
    conf=probs.max(axis=1); corr=(yp==y).astype(float)
    if weights is None: weights=np.ones(len(y),dtype=float)
    wsum=float(weights.sum()); cm=confusion_fast(y,yp,probs.shape[1],weights=weights); macro,weighted=f1_from_cm(cm)
    if top3 is None: top3=topk_correct(y,probs,3)
    if top5 is None: top5=topk_correct(y,probs,5)
    return {"accuracy":float(np.sum(corr*weights)/wsum),"balanced_accuracy":balanced_acc_from_cm(cm),"macro_f1":macro,"weighted_f1":weighted,"top3_accuracy":float(np.sum(top3.astype(float)*weights)/wsum),"top5_accuracy":float(np.sum(top5.astype(float)*weights)/wsum),"ece":ece_from_arrays(corr,conf,n_bins,weights),"mean_confidence":float(np.sum(conf*weights)/wsum),"mean_uncertainty":float(np.sum((1-conf)*weights)/wsum)}

def bootstrap_weights(n:int, rng) -> np.ndarray:
    return rng.multinomial(n, np.full(n,1.0/n))

def percentile_ci(x, level):
    a=(100.0-level)/2.0; return float(np.nanpercentile(x,a)), float(np.nanpercentile(x,100-a))

def bootstrap_metric_ci(pred, probs, model, scenario, n_boot, ci, rng, n_bins, log_every):
    y=pred["true_label"].astype(int).to_numpy(); yp=pred["predicted_label"].astype(int).to_numpy()
    top3=topk_correct(y,probs,3); top5=topk_correct(y,probs,5)
    point=metric_bundle(y,yp,probs,n_bins,top3=top3,top5=top5)
    metrics=["accuracy","balanced_accuracy","macro_f1","weighted_f1","top3_accuracy","top5_accuracy","ece"]
    vals={m:np.empty(n_boot) for m in metrics}
    for b in range(n_boot):
        if log_every>0 and (b+1)%log_every==0: log(f"[BOOT metrics] {model}/{scenario}: {b+1}/{n_boot}")
        w=bootstrap_weights(len(y),rng); mb=metric_bundle(y,yp,probs,n_bins,weights=w,top3=top3,top5=top5)
        for m in metrics: vals[m][b]=mb[m]
    rows=[]
    for m in metrics:
        lo,hi=percentile_ci(vals[m],ci)
        rows.append({"model_name":model,"scenario":scenario,"metric":m,"n_objects":len(y),"point_estimate":point[m],"bootstrap_mean":float(np.nanmean(vals[m])),"bootstrap_sd":float(np.nanstd(vals[m],ddof=1)),"ci_level":ci,"ci_low":lo,"ci_high":hi})
    return pd.DataFrame(rows), point

def align_pair(pred_t, prob_t, pred_r, prob_r):
    a=pd.DataFrame({"object_id":pred_t["object_id"].to_numpy(),"ia":np.arange(len(pred_t))})
    b=pd.DataFrame({"object_id":pred_r["object_id"].to_numpy(),"ib":np.arange(len(pred_r))})
    m=a.merge(b,on="object_id",how="inner"); ia=m["ia"].to_numpy(int); ib=m["ib"].to_numpy(int)
    return pred_t.iloc[ia].reset_index(drop=True), prob_t[ia], pred_r.iloc[ib].reset_index(drop=True), prob_r[ib]

def paired_bootstrap(pred_t,prob_t,pred_r,prob_r,target,reference,scenario,n_boot,ci,rng,n_bins,log_every):
    t,pt,r,pr=align_pair(pred_t,prob_t,pred_r,prob_r)
    y=t["true_label"].astype(int).to_numpy(); yt=t["predicted_label"].astype(int).to_numpy(); yr=r["predicted_label"].astype(int).to_numpy()
    top3t=topk_correct(y,pt,3); top5t=topk_correct(y,pt,5); top3r=topk_correct(y,pr,3); top5r=topk_correct(y,pr,5)
    point_t=metric_bundle(y,yt,pt,n_bins,top3=top3t,top5=top5t); point_r=metric_bundle(y,yr,pr,n_bins,top3=top3r,top5=top5r)
    metrics=["accuracy","macro_f1","weighted_f1","top3_accuracy","top5_accuracy","ece"]; vals={m:np.empty(n_boot) for m in metrics}
    for b in range(n_boot):
        if log_every>0 and (b+1)%log_every==0: log(f"[BOOT paired] {target} vs {reference}/{scenario}: {b+1}/{n_boot}")
        w=bootstrap_weights(len(y),rng)
        mt=metric_bundle(y,yt,pt,n_bins,weights=w,top3=top3t,top5=top5t); mr=metric_bundle(y,yr,pr,n_bins,weights=w,top3=top3r,top5=top5r)
        for m in metrics: vals[m][b]=mt[m]-mr[m]
    rows=[]
    for m in metrics:
        lo,hi=percentile_ci(vals[m],ci); delta=point_t[m]-point_r[m]
        p_one=float(np.mean(vals[m]<=0)) if delta>=0 else float(np.mean(vals[m]>=0))
        rows.append({"target_model":target,"reference_model":reference,"scenario":scenario,"metric":m,"n_common_objects":len(y),"target_point":point_t[m],"reference_point":point_r[m],"delta_target_minus_reference":float(delta),"bootstrap_mean_delta":float(np.nanmean(vals[m])),"bootstrap_sd":float(np.nanstd(vals[m],ddof=1)),"ci_level":ci,"ci_low":lo,"ci_high":hi,"one_sided_bootstrap_p_against_zero":p_one})
    return pd.DataFrame(rows)

def mcnemar(pred_t,prob_t,pred_r,prob_r,target,reference,scenario):
    t,pt,r,pr=align_pair(pred_t,prob_t,pred_r,prob_r); y=t["true_label"].astype(int).to_numpy()
    ct=t["predicted_label"].astype(int).to_numpy()==y; cr=r["predicted_label"].astype(int).to_numpy()==y
    b=int(np.sum(ct & ~cr)); c=int(np.sum(~ct & cr)); p=np.nan
    if stats is not None and b+c>0: p=float(stats.binomtest(min(b,c), n=b+c, p=0.5).pvalue)
    return {"target_model":target,"reference_model":reference,"scenario":scenario,"n_common_objects":len(y),"target_accuracy":float(np.mean(ct)),"reference_accuracy":float(np.mean(cr)),"delta_accuracy":float(np.mean(ct)-np.mean(cr)),"target_correct_reference_wrong":b,"target_wrong_reference_correct":c,"mcnemar_exact_pvalue":p}

def policy_score(pred, policy):
    if policy=="novelty_rarity":
        return pred["priority_novelty_rarity"].to_numpy(float) if "priority_novelty_rarity" in pred.columns else 0.5*pred["novelty"].to_numpy(float)+0.5*pred["rarity_score"].to_numpy(float)
    if policy=="rarity_only": return pred["rarity_score"].to_numpy(float)
    if policy=="uncertainty_only": return pred["uncertainty"].to_numpy(float)
    raise ValueError(policy)

def followup_bootstrap(pred,model,scenario,policies,budgets,n_boot,n_perm,ci,rng,log_every):
    rows=[]; perm_rows=[]; rare_all=pred["true_is_rare"].astype(bool).to_numpy(); correct_all=pred["correct"].astype(bool).to_numpy()
    n=len(pred); n_rare=int(rare_all.sum()); base=float(n_rare/n)
    for policy in policies:
        order=np.argsort(policy_score(pred,policy))[::-1]
        for budget in budgets:
            k=max(1,int(math.ceil(n*budget))); sel=order[:k]; sel_rare=rare_all[sel]; sel_corr=correct_all[sel]
            obs_rate=float(sel_rare.mean()); obs_enrich=obs_rate/base if base>0 else np.nan; obs_corr=float(sel_corr.mean())
            rr=np.empty(n_boot); en=np.empty(n_boot); mc=np.empty(n_boot)
            for b in range(n_boot):
                if log_every>0 and (b+1)%log_every==0: log(f"[BOOT followup] {model}/{scenario}/{policy}: {b+1}/{n_boot}")
                isel=rng.integers(0,k,size=k); ibase=rng.integers(0,n,size=n)
                rrate=float(sel_rare[isel].mean()); brate=float(rare_all[ibase].mean())
                rr[b]=rrate; en[b]=rrate/brate if brate>0 else np.nan; mc[b]=float(sel_corr[isel].mean())
            rr_lo,rr_hi=percentile_ci(rr,ci); en_lo,en_hi=percentile_ci(en,ci); mc_lo,mc_hi=percentile_ci(mc,ci)
            rows.append({"model_name":model,"scenario":scenario,"policy":policy,"budget_fraction":budget,"n_available":n,"n_selected":k,"baseline_rare_rate":base,"rare_rate":obs_rate,"rare_enrichment":obs_enrich,"mean_correct":obs_corr,"rare_rate_bootstrap_mean":float(np.nanmean(rr)),"rare_rate_bootstrap_sd":float(np.nanstd(rr,ddof=1)),"rare_rate_ci_low":rr_lo,"rare_rate_ci_high":rr_hi,"rare_enrichment_bootstrap_mean":float(np.nanmean(en)),"rare_enrichment_bootstrap_sd":float(np.nanstd(en,ddof=1)),"rare_enrichment_ci_low":en_lo,"rare_enrichment_ci_high":en_hi,"mean_correct_ci_low":mc_lo,"mean_correct_ci_high":mc_hi})
            if policy=="novelty_rarity" and abs(budget-0.05)<1e-9 and n_perm>0:
                draws=rng.hypergeometric(ngood=n_rare, nbad=n-n_rare, nsample=k, size=n_perm); null=draws/k
                lo,hi=percentile_ci(null,95.0); pval=float((np.sum(null>=obs_rate)+1)/(len(null)+1))
                perm_rows.append({"model_name":model,"scenario":scenario,"policy":policy,"budget_fraction":budget,"n_available":n,"n_selected":k,"observed_rare_rate":obs_rate,"observed_rare_enrichment":obs_enrich,"null_mean_rare_rate":float(null.mean()),"null_ci_low":lo,"null_ci_high":hi,"permutation_pvalue_observed_ge_null":pval,"n_permutations":n_perm})
    return pd.DataFrame(rows), pd.DataFrame(perm_rows)

def write_summary(path,point,ci,paired,mc,fci,perm,args):
    lines=["# Fast stage-aware statistical validation","",f"- Bootstrap iterations: `{args.n_bootstrap}`",f"- Permutation draws: `{args.n_permutations}`",f"- CI level: `{args.ci}`","","## Key point metrics",""]
    show=point[point["scenario"].isin(["first_3_points","first_5_points","first_10_points","first_20_points","window_7_days","window_30_days"])]
    cols=["model_name","scenario","n_objects","accuracy","macro_f1","top3_accuracy","top5_accuracy","ece"]
    lines.append(show[cols].to_markdown(index=False,floatfmt=".4f") if not show.empty else "_No metrics._")
    lines += ["","## Main paired differences",""]
    showp=paired[(paired["metric"].isin(["accuracy","macro_f1","top5_accuracy"])) & (paired["scenario"].isin(["first_5_points","first_10_points","first_20_points","window_30_days"]))]
    pcols=["target_model","reference_model","scenario","metric","delta_target_minus_reference","ci_low","ci_high","one_sided_bootstrap_p_against_zero"]
    lines.append(showp[pcols].to_markdown(index=False,floatfmt=".4f") if not showp.empty else "_No paired metrics._")
    lines += ["","## Follow-up enrichment at 5% budget",""]
    showf=fci[(fci["policy"]=="novelty_rarity") & (np.isclose(fci["budget_fraction"],0.05)) & (fci["scenario"].isin(["first_5_points","first_10_points","first_20_points","window_30_days"]))]
    fcols=["model_name","scenario","rare_enrichment","rare_enrichment_ci_low","rare_enrichment_ci_high","mean_correct","mean_correct_ci_low","mean_correct_ci_high"]
    lines.append(showf[fcols].to_markdown(index=False,floatfmt=".4f") if not showf.empty else "_No follow-up metrics._")
    path.write_text("\n".join(lines),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT_DIR); ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); ap.add_argument("--final-dir",type=Path,default=DEFAULT_FINAL_DIR)
    ap.add_argument("--models",type=str,default="all"); ap.add_argument("--scenarios",type=str,default="all"); ap.add_argument("--partial-only",action="store_true"); ap.add_argument("--pairs",type=str,default="all")
    ap.add_argument("--budgets",type=str,default="0.05"); ap.add_argument("--policies",type=str,default="novelty_rarity"); ap.add_argument("--n-bootstrap",type=int,default=1000); ap.add_argument("--n-permutations",type=int,default=5000); ap.add_argument("--ci",type=float,default=95.0); ap.add_argument("--n-bins",type=int,default=15); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--log-every",type=int,default=100)
    args=ap.parse_args(); start=time.time(); rng=np.random.default_rng(args.seed)
    models=parse_list(args.models,DEFAULT_MODELS); scenarios=parse_list(args.scenarios, PARTIAL_SCENARIOS if args.partial_only else ALL_SCENARIOS); pairs=parse_pairs(args.pairs); budgets=parse_float_list(args.budgets); policies=parse_list(args.policies,["novelty_rarity"])
    args.output_dir.mkdir(parents=True,exist_ok=True); args.final_dir.mkdir(parents=True,exist_ok=True)
    log(f"[START] models={models}"); log(f"[START] scenarios={scenarios}"); log(f"[START] n_bootstrap={args.n_bootstrap}, n_permutations={args.n_permutations}")
    loaded={}
    for model in models:
        loaded[model]={}
        for scenario in scenarios:
            log(f"[LOAD] {model}/{scenario}")
            try:
                loaded[model][scenario]=load_prediction_and_probs(args.input_dir,model,scenario); log(f"[OK] loaded n={len(loaded[model][scenario][0])}")
            except Exception as exc: log(f"[WARN] skipped {model}/{scenario}: {exc}")
    point_rows=[]; ci_frames=[]; follow_frames=[]; perm_frames=[]; total=sum(len(v) for v in loaded.values()); job=0
    for model, scen_map in loaded.items():
        for scenario,(pred,probs) in scen_map.items():
            job+=1; log(f"[METRICS] job {job}/{total}: {model}/{scenario}")
            cdf,point=bootstrap_metric_ci(pred,probs,model,scenario,args.n_bootstrap,args.ci,rng,args.n_bins,args.log_every)
            point_rows.append({"model_name":model,"scenario":scenario,"n_objects":len(pred),**point,"rare_true_rate":float(pred["true_is_rare"].astype(bool).mean()),"mean_novelty":float(pred["novelty"].mean()),"mean_rarity_score":float(pred["rarity_score"].mean())}); ci_frames.append(cdf)
            log(f"[FOLLOWUP] {model}/{scenario}"); fci,fp=followup_bootstrap(pred,model,scenario,policies,budgets,args.n_bootstrap,args.n_permutations,args.ci,rng,args.log_every); follow_frames.append(fci)
            if not fp.empty: perm_frames.append(fp)
    paired_frames=[]; mc_rows=[]
    for target,reference in pairs:
        if target not in loaded or reference not in loaded: continue
        for scenario in [s for s in scenarios if s in loaded[target] and s in loaded[reference]]:
            log(f"[PAIRED] {target} vs {reference}/{scenario}")
            pred_t,prob_t=loaded[target][scenario]; pred_r,prob_r=loaded[reference][scenario]
            paired_frames.append(paired_bootstrap(pred_t,prob_t,pred_r,prob_r,target,reference,scenario,args.n_bootstrap,args.ci,rng,args.n_bins,args.log_every)); mc_rows.append(mcnemar(pred_t,prob_t,pred_r,prob_r,target,reference,scenario))
    point_df=pd.DataFrame(point_rows); ci_df=pd.concat(ci_frames,ignore_index=True) if ci_frames else pd.DataFrame(); paired_df=pd.concat(paired_frames,ignore_index=True) if paired_frames else pd.DataFrame(); mc_df=pd.DataFrame(mc_rows); fci_df=pd.concat(follow_frames,ignore_index=True) if follow_frames else pd.DataFrame(); perm_df=pd.concat(perm_frames,ignore_index=True) if perm_frames else pd.DataFrame()
    outputs={"stage_aware_fast_point_metrics.csv":point_df,"stage_aware_fast_bootstrap_metric_ci.csv":ci_df,"stage_aware_fast_paired_bootstrap_differences.csv":paired_df,"stage_aware_fast_mcnemar_tests.csv":mc_df,"stage_aware_fast_followup_bootstrap_ci.csv":fci_df,"stage_aware_fast_followup_permutation_tests.csv":perm_df}
    for name,df in outputs.items():
        out=args.output_dir/name; log(f"[SAVE] {out}"); df.to_csv(out,index=False); shutil.copyfile(out,args.final_dir/name)
    summary_path=args.output_dir/"stage_aware_fast_statistical_validation_summary.md"; log(f"[SAVE] {summary_path}"); write_summary(summary_path,point_df,ci_df,paired_df,mc_df,fci_df,perm_df,args); shutil.copyfile(summary_path,args.final_dir/summary_path.name)
    log(f"[DONE] Finished in {(time.time()-start)/60:.2f} minutes"); log(f"[DONE] copied outputs to {args.final_dir}")
if __name__=="__main__": main()
