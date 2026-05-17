#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scan_clean_followup_inputs.py

Procura insumos para refazer a ablação de follow-up de forma mais limpa:
- arquivos candidatos de features/metadados com object_id;
- rankings antigos apenas para auditar colunas e inferir o conjunto de classes raras;
- gera rare_class_labels_candidate.json para inspeção.

Uso:
python .\experiments\scan_clean_followup_inputs.py `
  --results-dir results `
  --output-dir results\clean_followup_input_scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_EXT = {".csv", ".parquet", ".pq"}
FEATURE_KEYWORDS = [
    "feature", "features", "context", "contextual", "tabular",
    "object", "metadata", "meta", "test"
]
RANKING_KEYWORDS = ["ranking", "followup", "follow", "novelty", "rarity", "priority"]


def safe_read_head(path: Path, nrows: int = 2000) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=nrows)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def inspect_file(path: Path) -> dict | None:
    df = safe_read_head(path)
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    lower_cols = {c.lower(): c for c in cols}

    id_candidates = [
        c for c in cols
        if c.lower() in {"object_id", "objectid", "diaobjectid", "dia_object_id", "id"}
        or "object" in c.lower() and "id" in c.lower()
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_like = [
        c for c in cols
        if any(k in c.lower() for k in ["label", "target", "class", "truth", "pred"])
    ]

    return {
        "path": str(path),
        "extension": path.suffix.lower(),
        "n_columns_head": len(cols),
        "columns_preview": ", ".join(cols[:30]),
        "id_candidates": ", ".join(id_candidates),
        "n_numeric_columns_head": len(numeric_cols),
        "numeric_columns_preview": ", ".join(numeric_cols[:30]),
        "label_like_columns": ", ".join(label_like[:30]),
        "n_rows_head_read": int(len(df)),
    }


def infer_rare_labels_from_rankings(results_dir: Path) -> dict:
    ranking_files = []
    for p in results_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        name = p.name.lower()
        if any(k in name for k in RANKING_KEYWORDS):
            ranking_files.append(p)

    outputs = []
    rare_candidates = {}

    for p in ranking_files:
        df = safe_read_head(p, nrows=200000)
        if df is None or df.empty:
            continue

        cols = set(df.columns)
        info = {
            "path": str(p),
            "columns": list(df.columns),
            "n_rows_read": int(len(df)),
        }

        if {"true_label", "true_is_rare"}.issubset(cols):
            tmp = df[["true_label", "true_is_rare"]].copy()
            if tmp["true_is_rare"].dtype == object:
                tmp["true_is_rare"] = tmp["true_is_rare"].astype(str).str.lower().map({
                    "true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0
                })
            tmp["true_is_rare"] = pd.to_numeric(tmp["true_is_rare"], errors="coerce").fillna(0)
            grouped = tmp.groupby("true_label")["true_is_rare"].mean()
            rare_labels = sorted([int(k) for k, v in grouped.items() if float(v) >= 0.5])
            nonrare_labels = sorted([int(k) for k, v in grouped.items() if float(v) < 0.5])
            info["rare_labels_inferred"] = rare_labels
            info["nonrare_labels_inferred"] = nonrare_labels
            rare_candidates[str(p)] = rare_labels

        outputs.append(info)

    consensus = None
    if rare_candidates:
        vals = list(rare_candidates.values())
        first = vals[0]
        if all(v == first for v in vals):
            consensus = first

    return {
        "ranking_files": outputs,
        "rare_candidates_by_file": rare_candidates,
        "consensus_rare_labels": consensus,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/clean_followup_input_scan")
    parser.add_argument("--max-files", type=int, default=300)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for p in results_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in CANDIDATE_EXT:
            continue
        low = str(p).lower()
        if any(k in low for k in FEATURE_KEYWORDS):
            info = inspect_file(p)
            if info is not None:
                candidates.append(info)
        if len(candidates) >= args.max_files:
            break

    cand_df = pd.DataFrame(candidates)
    cand_csv = out_dir / "candidate_feature_metadata_files.csv"
    cand_df.to_csv(cand_csv, index=False)

    rare_info = infer_rare_labels_from_rankings(results_dir)
    rare_json = out_dir / "rare_class_labels_candidate.json"
    rare_json.write_text(json.dumps(rare_info, indent=2), encoding="utf-8")

    md = []
    md.append("# Clean follow-up input scan\n")
    md.append("This scan searches candidate feature/metadata files and audits old ranking files only to identify available object-level metadata and the predefined rare-class set.\n")

    md.append("## Candidate feature/metadata files\n")
    if cand_df.empty:
        md.append("No candidate feature/metadata files were found.")
    else:
        view_cols = [
            "path", "extension", "n_columns_head", "id_candidates",
            "n_numeric_columns_head", "label_like_columns", "columns_preview"
        ]
        md.append(cand_df[view_cols].to_markdown(index=False))

    md.append("\n## Rare-label candidates inferred from old rankings\n")
    consensus = rare_info.get("consensus_rare_labels")
    if consensus is None:
        md.append("No consensus rare-label set could be inferred.")
    else:
        md.append(f"Consensus rare labels: `{consensus}`")
        md.append("")
        md.append("Save/use this set explicitly in the clean analysis only after checking that it matches the intended predefined rare-class taxonomy.")

    md.append("\n## Ranking files audited\n")
    ranking_rows = []
    for item in rare_info["ranking_files"]:
        ranking_rows.append({
            "path": item["path"],
            "n_rows_read": item["n_rows_read"],
            "has_rare_inference": "rare_labels_inferred" in item,
            "rare_labels_inferred": item.get("rare_labels_inferred", None),
            "columns_preview": ", ".join(item["columns"][:25]),
        })
    if ranking_rows:
        md.append(pd.DataFrame(ranking_rows).to_markdown(index=False))
    else:
        md.append("No ranking files were found.")

    md.append("\n## Output files\n")
    md.append(f"- `{cand_csv}`")
    md.append(f"- `{rare_json}`")

    summary = out_dir / "clean_followup_input_scan_summary.md"
    summary.write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Summary: {summary}")
    print(f"Candidate files: {cand_csv}")
    print(f"Rare-label candidates: {rare_json}")


if __name__ == "__main__":
    main()
