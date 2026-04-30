from __future__ import annotations

from pathlib import Path
import json
import argparse

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "results" / "interface_case_studies"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "interface_case_studies_summary"


def safe_get(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def summarize_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))

    prediction = report.get("prediction", {})
    input_summary = report.get("input_summary", {})
    reliability = report.get("reliability", {})
    feature_builder = report.get("feature_builder") or {}

    input_quality = reliability.get("input_quality", {})
    feature_completeness = reliability.get("feature_completeness", {})
    model_domain = reliability.get("model_domain_reliability", {})

    top_classes = report.get("top_classes", [])
    top1 = top_classes[0] if len(top_classes) >= 1 else {}
    top2 = top_classes[1] if len(top_classes) >= 2 else {}
    top3 = top_classes[2] if len(top_classes) >= 3 else {}

    return {
        "case_name": path.stem,
        "file": path.name,
        "object_id": report.get("object_id"),
        "source_domain": report.get("source_domain"),
        "predicted_class": prediction.get("predicted_class_name"),
        "confidence": prediction.get("confidence"),
        "uncertainty": prediction.get("uncertainty_score"),
        "novelty": prediction.get("novelty_score"),
        "rarity": prediction.get("rarity_score"),
        "priority_score": prediction.get("priority_score"),
        "is_predicted_rare": prediction.get("is_predicted_rare"),
        "input_quality": input_quality.get("level"),
        "feature_completeness": feature_completeness.get("level"),
        "model_domain_reliability": model_domain.get("level") or reliability.get("level"),
        "n_rows": input_summary.get("n_rows"),
        "n_bands": input_summary.get("n_bands"),
        "time_span": input_summary.get("time_span"),
        "bands": ",".join(input_summary.get("bands", [])) if isinstance(input_summary.get("bands"), list) else input_summary.get("bands"),
        "n_expected_features": feature_completeness.get("n_expected_features") or feature_builder.get("n_expected_features"),
        "n_matched_features": feature_completeness.get("n_matched_features") or feature_builder.get("n_matched_features"),
        "n_missing_filled_zero": feature_completeness.get("n_missing_filled_zero") or feature_builder.get("n_missing_filled_zero"),
        "matched_fraction": feature_completeness.get("matched_fraction"),
        "top1_class": top1.get("class_name"),
        "top1_probability": top1.get("probability"),
        "top2_class": top2.get("class_name"),
        "top2_probability": top2.get("probability"),
        "top3_class": top3.get("class_name"),
        "top3_probability": top3.get("probability"),
        "input_reasons": " | ".join(input_quality.get("reasons", [])),
        "feature_reasons": " | ".join(feature_completeness.get("reasons", [])),
        "domain_reasons": " | ".join(model_domain.get("reasons", []) or reliability.get("reasons", [])),
    }


def make_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "case_name",
        "object_id",
        "source_domain",
        "predicted_class",
        "confidence",
        "novelty",
        "input_quality",
        "feature_completeness",
        "model_domain_reliability",
        "n_bands",
        "n_matched_features",
        "n_missing_filled_zero",
    ]

    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    for col in ["confidence", "novelty"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")

    return out.to_markdown(index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize AstroTrust-AI interface prediction JSON reports into CSV and Markdown tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing exported astrotrust_prediction_report_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where summary files will be saved.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Create it and copy the exported prediction_report JSON files there."
        )

    report_paths = sorted(input_dir.glob("*.json"))

    if not report_paths:
        raise FileNotFoundError(
            f"No JSON reports found in: {input_dir}\n"
            "Export reports from the dashboard and copy them into this folder."
        )

    rows = [summarize_report(path) for path in report_paths]
    df = pd.DataFrame(rows)

    csv_path = output_dir / "interface_case_studies_summary.csv"
    md_path = output_dir / "interface_case_studies_summary.md"

    df.to_csv(csv_path, index=False)

    markdown = "# AstroTrust-AI Interface Case Studies Summary\n\n"
    markdown += make_markdown_table(df)
    markdown += "\n\n## Interpretation notes\n\n"
    markdown += (
        "- `Input quality` describes whether the uploaded light curve is technically usable.\n"
        "- `Feature completeness` describes how much of the model feature vector was observed or derived.\n"
        "- `Model-domain reliability` describes whether the prediction should be scientifically trusted under the model's training domain.\n"
        "- High confidence with high novelty should be treated as possible overconfident out-of-domain behavior.\n"
    )

    md_path.write_text(markdown, encoding="utf-8")

    print("[OK] Saved:")
    print(f"- {csv_path}")
    print(f"- {md_path}")
    print("\nSummary:")
    print(df[[
        "object_id",
        "source_domain",
        "predicted_class",
        "confidence",
        "novelty",
        "input_quality",
        "feature_completeness",
        "model_domain_reliability",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
