"""
AstroTrust-AI follow-up policy diversity diagnostics.

Compare rarity_only and novelty_rarity follow-up queues to answer:
If rarity_only reaches higher rare-object purity, why use novelty_rarity?

The script measures whether novelty_rarity increases feature-space atypicality,
rare-class coverage, rare-family coverage, entropy/effective diversity, and how
much it differs from the rarity_only queue.

Run from the project root.

Example:
python experiments/run_followup_policy_diversity.py ^
  --predictions results/hybrid_tabular_ensemble_250k/ensemble_hybrid_dominant_predictions.csv ^
  --label-map data/processed/elasticc2_large/full_class_counts.csv ^
  --output-dir results/followup_policy_diversity ^
  --final-summary results/final_publication/final_publication_summary.md ^
  --budgets 0.01 0.05 0.10 0.20 ^
  --infer-family-from-class-name
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import math

import numpy as np
import pandas as pd


TRUE_CANDIDATES = [
    "true_label", "y_true", "target", "label_true", "class_true", "true_class",
    "true_class_id", "target_label", "label"
]
NOVELTY_CANDIDATES = [
    "novelty_score", "novelty", "feature_novelty", "feature_space_novelty",
    "N_i", "N", "mean_novelty"
]
RARITY_CANDIDATES = [
    "rarity_score", "rare_probability_mass", "rare_prob_mass", "rare_mass",
    "rare_class_probability_mass", "R_i", "R", "rare_score"
]
PRIORITY_CANDIDATES = [
    "priority_score", "priority", "followup_priority", "novelty_rarity_score",
    "S_i", "S"
]
ID_CANDIDATES = ["object_id", "diaobject_id", "objectId", "id"]


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
        f"Could not infer {role} column. Please pass --{role.replace('_', '-')}-col.\n"
        f"Candidates tried: {candidates}\nAvailable columns: {list(df.columns)}"
    )


def find_optional_column(df: pd.DataFrame, explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit column not found: {explicit}\nAvailable columns: {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    return None


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
    if fam_col is not None:
        lm["family"] = lm[fam_col].astype(str)
    else:
        lm["family"] = np.nan

    if infer_family:
        lm["family"] = lm["family"].where(lm["family"].notna(), lm["class_name"].map(infer_family_from_name))

    return lm[["label", "class_name", "family"]].drop_duplicates("label")


def entropy_and_effective(values: pd.Series, possible_n: int | None = None) -> tuple[float, float, float, int]:
    counts = values.dropna().astype(str).value_counts()
    n_unique = int(len(counts))
    if counts.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    p = counts.values / counts.values.sum()
    h = float(-(p * np.log(p)).sum())
    eff = float(np.exp(h))
    denom = math.log(possible_n if possible_n and possible_n > 1 else max(n_unique, 2))
    h_norm = float(h / denom) if denom > 0 else np.nan
    return h, h_norm, eff, n_unique


def safe_mean(s: pd.Series) -> float:
    if len(s) == 0:
        return np.nan
    return float(pd.to_numeric(s, errors="coerce").mean())


def safe_median(s: pd.Series) -> float:
    if len(s) == 0:
        return np.nan
    return float(pd.to_numeric(s, errors="coerce").median())


def build_scores(df: pd.DataFrame, novelty_col: str, rarity_col: str, priority_col: str | None, rank_normalize: bool) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    out["_novelty"] = pd.to_numeric(out[novelty_col], errors="coerce")
    out["_rarity"] = pd.to_numeric(out[rarity_col], errors="coerce")

    if rank_normalize:
        out["_novelty_used"] = out["_novelty"].rank(pct=True)
        out["_rarity_used"] = out["_rarity"].rank(pct=True)
    else:
        out["_novelty_used"] = out["_novelty"]
        out["_rarity_used"] = out["_rarity"]

    if priority_col is not None:
        out["_novelty_rarity_score"] = pd.to_numeric(out[priority_col], errors="coerce")
        policy_source = f"existing priority column '{priority_col}'"
    else:
        out["_novelty_rarity_score"] = 0.5 * out["_novelty_used"] + 0.5 * out["_rarity_used"]
        policy_source = "computed as 0.5 * novelty + 0.5 * rarity"

    out["_rarity_only_score"] = out["_rarity_used"]
    return out, policy_source


def select_queue(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    return df.sort_values(score_col, ascending=False, kind="mergesort").head(k).copy()


def queue_metrics(
    selected: pd.DataFrame,
    policy_name: str,
    budget: float,
    rare_labels: set[str],
    all_rare_labels_count: int,
) -> dict:
    rare_df = selected[selected["_is_true_rare"]].copy()
    rare_n = len(rare_df)

    class_h, class_hn, class_eff, class_cov = entropy_and_effective(
        rare_df["_true_label"], possible_n=all_rare_labels_count
    )
    fam_h, fam_hn, fam_eff, fam_cov = entropy_and_effective(rare_df["family"], possible_n=None)

    rare_counts = rare_df["_true_label"].value_counts()
    max_class_share = float(rare_counts.iloc[0] / rare_n) if rare_n > 0 and len(rare_counts) else np.nan
    top_rare_class = str(rare_counts.index[0]) if len(rare_counts) else ""

    family_counts = rare_df["family"].astype(str).value_counts()
    max_family_share = float(family_counts.iloc[0] / rare_n) if rare_n > 0 and len(family_counts) else np.nan
    top_family = str(family_counts.index[0]) if len(family_counts) else ""

    return {
        "budget": budget,
        "budget_pct": budget * 100,
        "policy": policy_name,
        "selected_n": len(selected),
        "rare_n": rare_n,
        "rare_rate": rare_n / len(selected) if len(selected) else np.nan,
        "mean_novelty": safe_mean(selected["_novelty"]),
        "median_novelty": safe_median(selected["_novelty"]),
        "mean_rarity_score": safe_mean(selected["_rarity"]),
        "median_rarity_score": safe_median(selected["_rarity"]),
        "rare_class_coverage": class_cov,
        "rare_class_entropy": class_h,
        "rare_class_entropy_norm": class_hn,
        "rare_class_effective_n": class_eff,
        "max_rare_class_share": max_class_share,
        "top_rare_class": top_rare_class,
        "rare_family_coverage": fam_cov,
        "rare_family_entropy": fam_h,
        "rare_family_entropy_norm": fam_hn,
        "rare_family_effective_n": fam_eff,
        "max_rare_family_share": max_family_share,
        "top_rare_family": top_family,
    }


def distribution_table(selected_by_policy: dict[str, pd.DataFrame], group_col: str, rare_only: bool = True) -> pd.DataFrame:
    rows = []
    for policy, selected in selected_by_policy.items():
        df = selected[selected["_is_true_rare"]] if rare_only else selected
        total = len(df)
        counts = df[group_col].fillna("Unknown").astype(str).value_counts()
        for value, n in counts.items():
            rows.append({
                "policy": policy,
                group_col: value,
                "n": int(n),
                "share": float(n / total) if total else np.nan,
            })
    return pd.DataFrame(rows)


def summarize_unique(name: str, df: pd.DataFrame) -> dict:
    rare_df = df[df["_is_true_rare"]]
    _, hn, eff, cov = entropy_and_effective(rare_df["_true_label"], None)
    return {
        "subset": name,
        "n": len(df),
        "rare_n": len(rare_df),
        "rare_rate": len(rare_df) / len(df) if len(df) else np.nan,
        "mean_novelty": safe_mean(df["_novelty"]),
        "mean_rarity_score": safe_mean(df["_rarity"]),
        "rare_class_coverage": cov,
        "rare_class_entropy_norm": hn,
        "rare_class_effective_n": eff,
    }


def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    d = df[cols].copy()
    if n is not None:
        d = d.head(n)
    return d.to_markdown(index=False, floatfmt=".4f")


def write_summary(
    output_dir: Path,
    metrics: pd.DataFrame,
    overlap_rows: list[dict],
    class_dist_5: pd.DataFrame | None,
    family_dist_5: pd.DataFrame | None,
    unique_5: pd.DataFrame | None,
    policy_source: str,
    score_cols: dict,
    final_summary: Path | None,
) -> Path:
    lines = []
    lines.append("# AstroTrust-AI Follow-up Policy Diversity Diagnostics")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Score columns")
    lines.append("")
    lines.append(f"- Novelty column: `{score_cols['novelty_col']}`")
    lines.append(f"- Rarity column: `{score_cols['rarity_col']}`")
    lines.append(f"- Novelty-rarity policy source: `{policy_source}`")
    lines.append("")
    lines.append("## Policy-level diversity summary")
    lines.append("")
    show_cols = [
        "budget_pct", "policy", "selected_n", "rare_rate", "mean_novelty",
        "rare_class_coverage", "rare_class_entropy_norm", "rare_class_effective_n",
        "max_rare_class_share", "rare_family_coverage", "rare_family_entropy_norm",
        "rare_family_effective_n",
    ]
    show_cols = [c for c in show_cols if c in metrics.columns]
    lines.append(md_table(metrics.sort_values(["budget", "policy"]), show_cols))
    lines.append("")

    ov = pd.DataFrame(overlap_rows)
    if not ov.empty:
        lines.append("## Queue overlap at each budget")
        lines.append("")
        lines.append(md_table(ov, ["budget_pct", "intersection_n", "jaccard", "novelty_rarity_unique_n", "rarity_only_unique_n"]))
        lines.append("")

    if unique_5 is not None and not unique_5.empty:
        lines.append("## Objects selected by one 5% policy but not the other")
        lines.append("")
        lines.append(md_table(unique_5, list(unique_5.columns)))
        lines.append("")

    if class_dist_5 is not None and not class_dist_5.empty:
        lines.append("## Rare-class distribution at 5% budget")
        lines.append("")
        lines.append(md_table(class_dist_5, list(class_dist_5.columns), n=30))
        lines.append("")

    if family_dist_5 is not None and not family_dist_5.empty:
        lines.append("## Rare-family distribution at 5% budget")
        lines.append("")
        lines.append(md_table(family_dist_5, list(family_dist_5.columns), n=30))
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append(
        "If `rarity_only` has higher rare-object rate but lower novelty, lower "
        "rare-class/family coverage, lower entropy, or higher maximum class share, "
        "then it should be interpreted as a purity-maximizing retrieval baseline. "
        "`novelty_rarity` can then be justified as a discovery-oriented triage "
        "policy that sacrifices some rare purity to increase atypicality and/or "
        "diversity of the follow-up queue."
    )
    lines.append("")

    out = output_dir / "followup_policy_diversity_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    if final_summary is not None:
        text = final_summary.read_text(encoding="utf-8")
        block = []
        block.append("## Follow-up policy diversity diagnostics")
        block.append("")
        block.append("A dedicated diversity analysis compares the purity-oriented `rarity_only` queue against the discovery-oriented `novelty_rarity` queue. The diagnostic reports rare-object rate, mean novelty, rare-class and rare-family coverage, entropy/effective diversity, maximum class concentration, and queue overlap across follow-up budgets.")
        block.append("")
        five = metrics[np.isclose(metrics["budget"], 0.05)]
        if not five.empty:
            block.append("At the 5% budget:")
            block.append("")
            block.append(md_table(five[show_cols], show_cols))
            block.append("")
        if not ov.empty:
            five_ov = ov[np.isclose(ov["budget"], 0.05)]
            if not five_ov.empty:
                block.append("5% queue overlap:")
                block.append("")
                block.append(md_table(five_ov, ["budget_pct", "intersection_n", "jaccard", "novelty_rarity_unique_n", "rarity_only_unique_n"]))
                block.append("")

        final_text = text
        if "## Follow-up policy diversity diagnostics" not in final_text:
            marker = "\n## Suggested manuscript claims"
            if marker in final_text:
                final_text = final_text.replace(marker, "\n\n" + "\n".join(block) + "\n" + marker)
            else:
                final_text = final_text + "\n\n" + "\n".join(block) + "\n"

        updated = output_dir / "final_publication_summary_with_followup_diversity.md"
        updated.write_text(final_text, encoding="utf-8")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--label-map", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/followup_policy_diversity"))
    parser.add_argument("--final-summary", type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20])
    parser.add_argument("--rare-labels", default="4,5,6,7,11,29,30,31")
    parser.add_argument("--true-col")
    parser.add_argument("--novelty-col")
    parser.add_argument("--rarity-col")
    parser.add_argument("--priority-col")
    parser.add_argument("--id-col")
    parser.add_argument("--rank-normalize", action="store_true")
    parser.add_argument("--infer-family-from-class-name", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading predictions...", flush=True)
    df = pd.read_csv(args.predictions)
    print(f"[INFO] Prediction rows: {len(df)}", flush=True)

    true_col = find_column(df, args.true_col, TRUE_CANDIDATES, "true_col")
    novelty_col = find_column(df, args.novelty_col, NOVELTY_CANDIDATES, "novelty_col")
    rarity_col = find_column(df, args.rarity_col, RARITY_CANDIDATES, "rarity_col")
    priority_col = find_optional_column(df, args.priority_col, PRIORITY_CANDIDATES)
    id_col = find_optional_column(df, args.id_col, ID_CANDIDATES)
    if id_col is None:
        df = df.copy()
        df["_row_id"] = np.arange(len(df)).astype(str)
        id_col = "_row_id"
        print("[WARN] No object_id-like column found; using row index for queue overlap.", flush=True)

    label_map = load_label_map(args.label_map, infer_family=args.infer_family_from_class_name)

    df = df.copy()
    df["_true_label"] = df[true_col].map(norm_label)
    rare_labels = {norm_label(x.strip()) for x in args.rare_labels.split(",") if x.strip()}
    df["_is_true_rare"] = df["_true_label"].isin(rare_labels)

    if not label_map.empty:
        df = df.merge(label_map, left_on="_true_label", right_on="label", how="left")
    else:
        df["class_name"] = df["_true_label"]
        df["family"] = "Unknown"

    df["class_name"] = df["class_name"].fillna(df["_true_label"])
    df["family"] = df["family"].fillna("Unknown")

    df, policy_source = build_scores(df, novelty_col, rarity_col, priority_col, args.rank_normalize)

    print(f"[INFO] true_col={true_col}", flush=True)
    print(f"[INFO] novelty_col={novelty_col}", flush=True)
    print(f"[INFO] rarity_col={rarity_col}", flush=True)
    print(f"[INFO] novelty_rarity policy source={policy_source}", flush=True)
    print(f"[INFO] true rare baseline={df['_is_true_rare'].mean():.6f}", flush=True)

    rows = []
    overlap_rows = []
    selected_5 = {}
    total_n = len(df)

    for budget in args.budgets:
        k = max(1, int(round(total_n * budget)))
        print(f"[INFO] Processing budget={budget:.4f} k={k}", flush=True)

        q_nr = select_queue(df, "_novelty_rarity_score", k)
        q_r = select_queue(df, "_rarity_only_score", k)

        rows.append(queue_metrics(q_nr, "novelty_rarity", budget, rare_labels, len(rare_labels)))
        rows.append(queue_metrics(q_r, "rarity_only", budget, rare_labels, len(rare_labels)))

        ids_nr = set(q_nr[id_col].astype(str))
        ids_r = set(q_r[id_col].astype(str))
        inter = ids_nr & ids_r
        union = ids_nr | ids_r
        overlap_rows.append({
            "budget": budget,
            "budget_pct": budget * 100,
            "intersection_n": len(inter),
            "jaccard": len(inter) / len(union) if union else np.nan,
            "novelty_rarity_unique_n": len(ids_nr - ids_r),
            "rarity_only_unique_n": len(ids_r - ids_nr),
        })

        if np.isclose(budget, 0.05):
            selected_5["novelty_rarity"] = q_nr
            selected_5["rarity_only"] = q_r

    metrics = pd.DataFrame(rows)
    metrics_path = args.output_dir / "policy_diversity_by_budget.csv"
    metrics.to_csv(metrics_path, index=False)

    class_dist_5 = None
    family_dist_5 = None
    unique_5 = None

    if selected_5:
        nr = selected_5["novelty_rarity"]
        ro = selected_5["rarity_only"]

        keep_cols = [
            id_col, "_true_label", "class_name", "family", "_is_true_rare",
            "_novelty", "_rarity", "_novelty_rarity_score", "_rarity_only_score"
        ]
        keep_cols = [c for c in keep_cols if c in nr.columns]
        nr[keep_cols].to_csv(args.output_dir / "selected_queue_budget_5pct_novelty_rarity.csv", index=False)
        ro[keep_cols].to_csv(args.output_dir / "selected_queue_budget_5pct_rarity_only.csv", index=False)

        class_dist_5 = distribution_table(selected_5, "_true_label", rare_only=True)
        if not label_map.empty:
            class_dist_5 = class_dist_5.merge(label_map, left_on="_true_label", right_on="label", how="left")
        class_dist_5.to_csv(args.output_dir / "rare_class_distribution_budget_5pct.csv", index=False)

        family_dist_5 = distribution_table(selected_5, "family", rare_only=True)
        family_dist_5.to_csv(args.output_dir / "rare_family_distribution_budget_5pct.csv", index=False)

        ids_nr = set(nr[id_col].astype(str))
        ids_ro = set(ro[id_col].astype(str))
        nr_unique = nr[nr[id_col].astype(str).isin(ids_nr - ids_ro)]
        ro_unique = ro[ro[id_col].astype(str).isin(ids_ro - ids_nr)]
        unique_5 = pd.DataFrame([
            summarize_unique("selected_only_by_novelty_rarity", nr_unique),
            summarize_unique("selected_only_by_rarity_only", ro_unique),
        ])
        unique_5.to_csv(args.output_dir / "novelty_unique_vs_rarity_unique_budget_5pct.csv", index=False)

    summary_path = write_summary(
        output_dir=args.output_dir,
        metrics=metrics,
        overlap_rows=overlap_rows,
        class_dist_5=class_dist_5,
        family_dist_5=family_dist_5,
        unique_5=unique_5,
        policy_source=policy_source,
        score_cols={
            "novelty_col": novelty_col,
            "rarity_col": rarity_col,
            "priority_col": priority_col or "",
        },
        final_summary=args.final_summary,
    )

    print("[DONE] Follow-up policy diversity diagnostics completed.", flush=True)
    print(f"[SAVED] {metrics_path}", flush=True)
    print(f"[SAVED] {summary_path}", flush=True)
    if args.final_summary:
        print(f"[SAVED] {args.output_dir / 'final_publication_summary_with_followup_diversity.md'}", flush=True)


if __name__ == "__main__":
    main()
