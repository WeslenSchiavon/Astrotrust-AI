#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scan_validation_inference_code_paths.py

Escaneia o código do projeto para descobrir onde as probabilidades de teste do ensemble/base models
são geradas e onde podemos inserir salvamento das probabilidades de validação.

Uso:
python .\experiments\scan_validation_inference_code_paths.py `
  --project-root . `
  --output-dir results\ensemble_validation_inference_code_scan
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PATTERNS = [
    "ensemble_hybrid_dominant",
    "test_probabilities",
    "validation",
    "val_",
    "valid",
    "np.save",
    ".npy",
    "predict_proba",
    "softmax",
    "hybrid_cnn_tabular",
    "lightgbm_baseline",
    "lightgbm_regularized",
    "xgboost_gpu_deeper",
    "members",
    "weights",
    "split_metadata",
    "train_test_split",
    "Stratified",
]


SCRIPT_EXTS = {".py"}
CONFIG_EXTS = {".csv", ".json", ".md", ".txt"}


def snippet_around(lines: list[str], idx: int, context: int = 3) -> str:
    start = max(0, idx - context)
    end = min(len(lines), idx + context + 1)
    out = []
    for j in range(start, end):
        marker = ">>" if j == idx else "  "
        out.append(f"{marker} L{j+1}: {lines[j].rstrip()}")
    return "\n".join(out)


def scan_file(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [{
            "file": str(path),
            "pattern": "READ_ERROR",
            "line": None,
            "snippet": repr(e),
        }]

    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        low = line.lower()
        matched = [p for p in PATTERNS if p.lower() in low]
        if matched:
            hits.append({
                "file": str(path),
                "pattern": ", ".join(matched),
                "line": i + 1,
                "snippet": snippet_around(lines, i, context=3),
            })
    return hits


def summarize_csv(path: Path) -> dict:
    info = {
        "path": str(path),
        "exists": path.exists(),
        "columns": "",
        "rows": None,
        "ensemble_hybrid_dominant_rows": "",
        "error": "",
    }
    try:
        df = pd.read_csv(path)
        info["columns"] = ",".join(map(str, df.columns.tolist()))
        info["rows"] = len(df)
        if "model" in df.columns:
            sub = df[df["model"].astype(str).str.contains("ensemble_hybrid_dominant", case=False, na=False)]
            if not sub.empty:
                info["ensemble_hybrid_dominant_rows"] = sub.to_string(index=False)
    except Exception as e:
        info["error"] = repr(e)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/ensemble_validation_inference_code_scan")
    args = parser.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    candidate_scripts = []
    for p in root.rglob("*.py"):
        s = str(p).lower()
        if any(k in s for k in [
            "ensemble",
            "hybrid",
            "temporal",
            "lightgbm",
            "xgboost",
            "inference",
            "train",
            "calibration",
        ]):
            candidate_scripts.append(p)

    all_hits = []
    for p in candidate_scripts:
        all_hits.extend(scan_file(p))

    hits_df = pd.DataFrame(all_hits)
    hits_csv = out / "validation_inference_code_hits.csv"
    hits_df.to_csv(hits_csv, index=False)

    # Rank files by number and diversity of hits.
    if not hits_df.empty:
        rank = (
            hits_df.groupby("file")
            .agg(n_hits=("line", "count"), patterns=("pattern", lambda x: " | ".join(sorted(set(map(str, x))))))
            .reset_index()
            .sort_values("n_hits", ascending=False)
        )
    else:
        rank = pd.DataFrame(columns=["file", "n_hits", "patterns"])
    rank.to_csv(out / "validation_inference_candidate_script_ranking.csv", index=False)

    # Inspect known ensemble comparison files for weights/members.
    known_csvs = [
        root / "results/hybrid_tabular_ensemble_250k/hybrid_tabular_ensemble_comparison_250k.csv",
        root / "results/final_publication/model_performance/best_ensemble_summary.csv",
        root / "results/final_probability_ensemble_search_250k/best_ensemble_summary.csv",
        root / "results/final_probability_ensemble_search_250k/ensemble_weight_search_results.csv",
    ]
    csv_infos = [summarize_csv(p) for p in known_csvs]
    pd.DataFrame(csv_infos).to_csv(out / "ensemble_weight_config_summary.csv", index=False)

    md = []
    md.append("# Validation inference code-path scan\n")
    md.append("This report identifies where the project likely generates ensemble/base-model test probabilities and where validation probability saving should be added.\n")
    md.append("## Candidate script ranking\n")
    if not rank.empty:
        md.append(rank.head(30).to_markdown(index=False))
    else:
        md.append("No candidate Python scripts found.\n")

    md.append("\n## Ensemble weight/config summary\n")
    for info in csv_infos:
        md.append(f"### `{info['path']}`")
        md.append(f"- exists: `{info['exists']}`")
        md.append(f"- rows: `{info['rows']}`")
        md.append(f"- columns: `{info['columns']}`")
        if info["ensemble_hybrid_dominant_rows"]:
            md.append("```text")
            md.append(info["ensemble_hybrid_dominant_rows"])
            md.append("```")
        if info["error"]:
            md.append(f"- error: `{info['error']}`")
        md.append("")

    md.append("\n## Most relevant snippets\n")
    if not hits_df.empty:
        # Prioritize scripts with ensemble_hybrid_dominant and np.save/test_probabilities.
        relevant = hits_df[
            hits_df["pattern"].astype(str).str.contains("ensemble_hybrid_dominant|test_probabilities|np.save|validation|predict_proba|softmax", case=False, na=False)
        ].copy()
        if relevant.empty:
            relevant = hits_df.copy()

        # Keep first several snippets from top-ranked files.
        top_files = rank.head(8)["file"].tolist()
        rel = relevant[relevant["file"].isin(top_files)].head(80)
        for _, r in rel.iterrows():
            md.append(f"### `{r['file']}` line {r['line']} pattern `{r['pattern']}`")
            md.append("```python")
            md.append(str(r["snippet"]))
            md.append("```")
    else:
        md.append("No snippets found.\n")

    md.append("\n## Next decision\n")
    md.append("- If the ensemble script already computes validation probabilities but does not save them, add `np.save(...validation_probabilities.npy, val_probs)` at that point.")
    md.append("- If it only computes test probabilities, modify the script to run the same inference function on the validation split.")
    md.append("- If the ensemble is just a weighted average of saved base-model probabilities, we must first generate validation probabilities for each base model, then reconstruct the ensemble validation probabilities with the exact same weights.")
    md.append("- After validation probabilities exist, run `run_publication_safe_ensemble_temperature_calibration.py` and fit temperature only on validation.\n")

    summary_path = out / "validation_inference_code_scan_summary.md"
    summary_path.write_text("\n".join(md), encoding="utf-8")

    schema = {
        "analysis": "validation inference code path scan",
        "n_candidate_scripts": len(candidate_scripts),
        "n_hits": int(len(hits_df)),
        "outputs": {
            "summary": str(summary_path),
            "hits_csv": str(hits_csv),
            "ranking_csv": str(out / "validation_inference_candidate_script_ranking.csv"),
            "weight_config_summary_csv": str(out / "ensemble_weight_config_summary.csv"),
        },
    }
    (out / "validation_inference_code_scan_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Done. Results written to: {out}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
