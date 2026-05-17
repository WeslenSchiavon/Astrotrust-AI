#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
prepare_exact_calibration_patch_context.py

Extrai trechos exatos dos scripts locais necessários para criar um patch seguro
que salve probabilidades de validação para a calibração final do ensemble.

Uso:
python .\experiments\prepare_exact_calibration_patch_context.py `
  --project-root . `
  --output-dir results\ensemble_temperature_patch_context
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGETS = [
    "experiments/run_hybrid_followup_policy_eval_250k.py",
    "experiments/run_hybrid_tabular_ensemble_250k_multiseed_ready.py",
    "experiments/run_hybrid_tabular_ensemble_scaling_with_md.py",
    "experiments/run_hybrid_tabular_ensemble_scaling.py",
]


KEYWORDS = [
    "def train_predict_proba",
    "def save_probabilities",
    "predict_proba",
    "train_test_split",
    "X_train",
    "X_val",
    "X_test",
    "val_idx",
    "test_idx",
    "probs_val_raw",
    "probs_test_raw",
    "torch.softmax",
    "np.save",
    "ensemble_hybrid_dominant",
    "hybrid_cnn_tabular",
    "lightgbm_baseline",
    "lightgbm_regularized",
    "xgboost_gpu_deeper",
    "ensemble_configs",
    "results_df",
    "comparison_path",
]


def collect_snippets(path: Path, context: int = 18) -> list[dict]:
    if not path.exists():
        return [{"path": str(path), "exists": False, "line": None, "keyword": "", "snippet": ""}]

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits = []
    used_ranges = []

    for i, line in enumerate(lines):
        low = line.lower()
        matched = [k for k in KEYWORDS if k.lower() in low]
        if not matched:
            continue

        start = max(0, i - context)
        end = min(len(lines), i + context + 1)

        # Merge near-duplicate ranges.
        if used_ranges and start <= used_ranges[-1][1] + 5:
            old_start, old_end, kws = used_ranges[-1]
            used_ranges[-1] = (old_start, max(old_end, end), kws + matched)
        else:
            used_ranges.append((start, end, matched))

    for start, end, kws in used_ranges:
        snippet_lines = []
        for j in range(start, end):
            snippet_lines.append(f"L{j+1}: {lines[j]}")
        hits.append({
            "path": str(path),
            "exists": True,
            "line_start": start + 1,
            "line_end": end,
            "keywords": sorted(set(kws)),
            "snippet": "\n".join(snippet_lines),
        })
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/ensemble_temperature_patch_context")
    args = parser.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_hits = []
    for rel in TARGETS:
        all_hits.extend(collect_snippets(root / rel))

    md = []
    md.append("# Exact calibration patch context\n")
    md.append("This file contains exact local-code snippets needed to create a safe patch for saving validation probabilities.\n")
    md.append("## Target files\n")
    for rel in TARGETS:
        p = root / rel
        md.append(f"- `{rel}` exists: `{p.exists()}`")
    md.append("")

    for hit in all_hits:
        md.append(f"## `{hit['path']}`")
        md.append(f"- exists: `{hit['exists']}`")
        if not hit["exists"]:
            continue
        md.append(f"- lines: `{hit['line_start']}-{hit['line_end']}`")
        md.append(f"- keywords: `{hit['keywords']}`")
        md.append("```python")
        md.append(hit["snippet"])
        md.append("```")
        md.append("")

    (out / "exact_calibration_patch_context.md").write_text("\n".join(md), encoding="utf-8")
    (out / "exact_calibration_patch_context.json").write_text(json.dumps(all_hits, indent=2), encoding="utf-8")

    print(f"Done. Results written to: {out}")
    print(f"Upload this file if you want an exact patch:")
    print(out / "exact_calibration_patch_context.md")


if __name__ == "__main__":
    main()
