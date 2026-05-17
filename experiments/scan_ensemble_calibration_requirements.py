#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scan_ensemble_calibration_requirements.py

Procura artefatos necessários para calibração publication-safe do ensemble final:
- probabilidades de validação;
- probabilidades de teste;
- arquivos de labels/splits;
- possíveis arquivos de pesos/configuração do ensemble;
- evidências de que só existem probabilidades de teste.

Uso:
python .\experiments\scan_ensemble_calibration_requirements.py `
  --project-root . `
  --output-dir results\ensemble_calibration_requirement_scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROB_EXTS = {".npy", ".npz"}
TABLE_EXTS = {".csv", ".parquet", ".pq", ".json", ".txt", ".md"}


def classify_path(p: Path) -> dict:
    s = str(p).lower()
    name = p.name.lower()

    is_prob = p.suffix.lower() in PROB_EXTS and ("prob" in name or "probab" in name)
    is_validation = any(k in s for k in ["validation", "val_", "_val", "\\val", "/val"])
    is_test = any(k in s for k in ["test", "heldout", "held_out"])
    is_ensemble = any(k in s for k in ["ensemble", "hybrid_dominant", "dominant"])
    is_hybrid = "hybrid_temporal_tabular" in s or "hybrid_cnn_tabular" in s
    is_lgbm = "lightgbm" in s or "lgbm" in s
    is_cnn = "temporal_cnn" in s or "cnn" in s
    is_weight = any(k in s for k in ["weight", "weights", "ensemble_comparison", "best_ensemble", "summary", "metadata", "config"])

    info = {
        "path": str(p),
        "name": p.name,
        "suffix": p.suffix.lower(),
        "size_bytes": p.stat().st_size if p.exists() else None,
        "is_probability_file": is_prob,
        "split_hint_validation": is_validation,
        "split_hint_test": is_test,
        "model_hint_ensemble": is_ensemble,
        "model_hint_hybrid": is_hybrid,
        "model_hint_lgbm": is_lgbm,
        "model_hint_cnn": is_cnn,
        "possible_weight_or_config": is_weight,
        "shape": "",
        "columns": "",
        "read_error": "",
    }

    try:
        if is_prob and p.suffix.lower() == ".npy":
            arr = np.load(p, mmap_mode="r")
            info["shape"] = str(tuple(arr.shape))
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p, nrows=5)
            info["columns"] = ",".join(map(str, df.columns.tolist()[:80]))
        elif p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
            info["shape"] = str(df.shape)
            info["columns"] = ",".join(map(str, df.columns.tolist()[:80]))
    except Exception as e:
        info["read_error"] = repr(e)

    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/ensemble_calibration_requirement_scan")
    args = parser.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        s = str(p).lower()
        name = p.name.lower()

        relevant = False
        if p.suffix.lower() in PROB_EXTS and ("prob" in name or "probab" in name):
            relevant = True
        if p.suffix.lower() in TABLE_EXTS and any(k in s for k in [
            "split", "validation", "val", "ensemble", "probabil", "prediction",
            "weights", "weight", "comparison", "metadata", "temperature", "calibration"
        ]):
            relevant = True

        if relevant:
            paths.append(p)

    rows = [classify_path(p) for p in paths]
    df = pd.DataFrame(rows).sort_values(["is_probability_file", "split_hint_validation", "model_hint_ensemble", "path"], ascending=[False, False, False, True])
    df.to_csv(out / "ensemble_calibration_candidate_files.csv", index=False)

    prob = df[df["is_probability_file"] == True].copy()
    val_prob = prob[prob["split_hint_validation"] == True]
    test_prob = prob[prob["split_hint_test"] == True]
    ensemble_val = val_prob[val_prob["model_hint_ensemble"] == True]
    ensemble_test = test_prob[test_prob["model_hint_ensemble"] == True]

    possible_split_labels = df[
        df["columns"].astype(str).str.contains("object_id", case=False, na=False)
        & df["columns"].astype(str).str.contains("label", case=False, na=False)
        & df["columns"].astype(str).str.contains("split", case=False, na=False)
    ].copy()

    possible_weights = df[df["possible_weight_or_config"] == True].copy()

    summary = []
    summary.append("# Ensemble calibration requirement scan\n")
    summary.append("## Goal\n")
    summary.append("Find whether we can perform publication-safe temperature scaling for the final `ensemble_hybrid_dominant` using validation-set probabilities.\n")
    summary.append("## Summary\n")
    summary.append(f"- Candidate files scanned: `{len(df)}`")
    summary.append(f"- Probability files found: `{len(prob)}`")
    summary.append(f"- Validation probability files found: `{len(val_prob)}`")
    summary.append(f"- Test probability files found: `{len(test_prob)}`")
    summary.append(f"- Ensemble-like validation probability files found: `{len(ensemble_val)}`")
    summary.append(f"- Ensemble-like test probability files found: `{len(ensemble_test)}`")
    summary.append(f"- Split/label candidate files found: `{len(possible_split_labels)}`")
    summary.append(f"- Weight/config candidate files found: `{len(possible_weights)}`\n")

    summary.append("## Ensemble-like validation probability candidates\n")
    if len(ensemble_val):
        summary.append(ensemble_val[["path", "shape", "size_bytes"]].to_markdown(index=False))
    else:
        summary.append("None found. To fully satisfy the requirement, generate or reconstruct validation probabilities for the final ensemble.\n")

    summary.append("\n## Ensemble-like test probability candidates\n")
    if len(ensemble_test):
        summary.append(ensemble_test[["path", "shape", "size_bytes"]].to_markdown(index=False))
    else:
        summary.append("None found.\n")

    summary.append("\n## All validation probability candidates\n")
    if len(val_prob):
        summary.append(val_prob[["path", "shape", "model_hint_ensemble", "model_hint_hybrid", "model_hint_lgbm", "model_hint_cnn"]].to_markdown(index=False))
    else:
        summary.append("None found.\n")

    summary.append("\n## Split/label candidates\n")
    if len(possible_split_labels):
        summary.append(possible_split_labels[["path", "shape", "columns"]].head(30).to_markdown(index=False))
    else:
        summary.append("None found.\n")

    summary.append("\n## Weight/config candidates\n")
    if len(possible_weights):
        summary.append(possible_weights[["path", "columns", "shape"]].head(50).to_markdown(index=False))
    else:
        summary.append("None found.\n")

    summary.append("\n## Decision rule\n")
    summary.append("- If an ensemble validation probability file exists, fit temperature on that validation file and apply it to the held-out ensemble test probabilities.")
    summary.append("- If no ensemble validation file exists but base-model validation probabilities exist, reconstruct the ensemble validation probabilities using the exact final ensemble weights, verify that the reconstructed test ensemble matches the official test ensemble probabilities, then fit temperature on validation.")
    summary.append("- If neither exists, rerun inference on the validation split for the final ensemble/base models, then perform the calibration analysis.")
    summary.append("- Do not fit temperature on the test set for a main manuscript claim.\n")

    (out / "ensemble_calibration_requirement_scan_summary.md").write_text("\n".join(summary), encoding="utf-8")

    schema = {
        "analysis": "ensemble calibration requirement scan",
        "project_root": str(root),
        "output_dir": str(out),
        "n_candidate_files": int(len(df)),
        "n_probability_files": int(len(prob)),
        "n_validation_probability_files": int(len(val_prob)),
        "n_ensemble_validation_probability_files": int(len(ensemble_val)),
        "decision_rule": "Fit temperature on validation probabilities only; apply to held-out test probabilities.",
    }
    (out / "ensemble_calibration_requirement_scan_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Done. Results written to: {out}")
    print(f"Summary: {out / 'ensemble_calibration_requirement_scan_summary.md'}")


if __name__ == "__main__":
    main()
