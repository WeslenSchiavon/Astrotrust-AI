#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
update_final_publication_with_topk_family_bootstrap.py

Consolida o teste:
  Top-k and family-level bootstrap confidence intervals

em:
  results/final_publication/topk_family_bootstrap/

e atualiza:
  results/final_publication/final_publication_summary.md

Definição oficial usada:
- fine top-k: verifica se a classe fina verdadeira aparece nos top-k fine labels.
- family top-k: mapeia os top-k fine labels para famílias e verifica se a
  família verdadeira aparece nesse conjunto.
- NÃO soma probabilidades por família antes do top-k.

Uso:
python .\experiments\update_final_publication_with_topk_family_bootstrap.py `
  --topk-family-dir results\topk_family_bootstrap_ci_250k_final `
  --final-publication-dir results\final_publication
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


START = "<!-- BEGIN_TOPK_FAMILY_BOOTSTRAP_CI -->"
END = "<!-- END_TOPK_FAMILY_BOOTSTRAP_CI -->"


FILES_TO_COPY = [
    "topk_family_bootstrap_summary.md",
    "topk_family_bootstrap_ci.csv",
    "per_family_recall_bootstrap_ci.csv",
    "topk_family_object_level_indicators.csv",
    "topk_family_bootstrap_schema.json",
    "fig_topk_family_bootstrap_ci.png",
    "fig_per_family_recall_bootstrap_ci.png",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def fmt(x, digits=6):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


def get_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    sub = df[df["metric"] == metric]
    if sub.empty:
        raise ValueError(f"Metric not found: {metric}")
    return sub.iloc[0]


def ci_text(row: pd.Series) -> str:
    return f"[{fmt(row['ci_low_95'])}, {fmt(row['ci_high_95'])}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-family-dir", default="results/topk_family_bootstrap_ci_250k_final")
    parser.add_argument("--final-publication-dir", default="results/final_publication")
    args = parser.parse_args()

    src_dir = Path(args.topk_family_dir)
    final_dir = Path(args.final_publication_dir)
    target_dir = final_dir / "topk_family_bootstrap"
    final_summary = final_dir / "final_publication_summary.md"

    main_csv = src_dir / "topk_family_bootstrap_ci.csv"
    per_family_csv = src_dir / "per_family_recall_bootstrap_ci.csv"
    report_md = src_dir / "topk_family_bootstrap_summary.md"

    for p in [main_csv, per_family_csv, report_md, final_summary]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for name in FILES_TO_COPY:
        src = src_dir / name
        dst = target_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append((src, dst))
        else:
            missing.append(name)

    main = pd.read_csv(main_csv)
    per_family = pd.read_csv(per_family_csv)

    fine_acc = get_metric(main, "fine_accuracy")
    fine_top2 = get_metric(main, "fine_top2_accuracy")
    fine_top3 = get_metric(main, "fine_top3_accuracy")
    fine_top5 = get_metric(main, "fine_top5_accuracy")
    fam_acc = get_metric(main, "family_accuracy")
    fam_top2 = get_metric(main, "family_top2_family_accuracy")
    fam_top3 = get_metric(main, "family_top3_family_accuracy")
    fam_top5 = get_metric(main, "family_top5_family_accuracy")

    key_table = pd.DataFrame([
        {
            "metric": "fine_accuracy",
            "point_estimate": fmt(fine_acc["point_estimate"]),
            "95% CI": ci_text(fine_acc),
            "definition": "true fine class equals top-1 fine prediction",
        },
        {
            "metric": "fine_top2_accuracy",
            "point_estimate": fmt(fine_top2["point_estimate"]),
            "95% CI": ci_text(fine_top2),
            "definition": "true fine class appears among top-2 fine predictions",
        },
        {
            "metric": "fine_top3_accuracy",
            "point_estimate": fmt(fine_top3["point_estimate"]),
            "95% CI": ci_text(fine_top3),
            "definition": "true fine class appears among top-3 fine predictions",
        },
        {
            "metric": "fine_top5_accuracy",
            "point_estimate": fmt(fine_top5["point_estimate"]),
            "95% CI": ci_text(fine_top5),
            "definition": "true fine class appears among top-5 fine predictions",
        },
        {
            "metric": "family_accuracy",
            "point_estimate": fmt(fam_acc["point_estimate"]),
            "95% CI": ci_text(fam_acc),
            "definition": "top-1 fine prediction mapped to true family",
        },
        {
            "metric": "family_top2_family_accuracy",
            "point_estimate": fmt(fam_top2["point_estimate"]),
            "95% CI": ci_text(fam_top2),
            "definition": "true family appears after mapping top-2 fine predictions to families",
        },
        {
            "metric": "family_top3_family_accuracy",
            "point_estimate": fmt(fam_top3["point_estimate"]),
            "95% CI": ci_text(fam_top3),
            "definition": "true family appears after mapping top-3 fine predictions to families",
        },
        {
            "metric": "family_top5_family_accuracy",
            "point_estimate": fmt(fam_top5["point_estimate"]),
            "95% CI": ci_text(fam_top5),
            "definition": "true family appears after mapping top-5 fine predictions to families",
        },
    ])

    per_family_short = per_family.copy()
    per_family_short["point_estimate"] = per_family_short["point_estimate"].map(lambda x: fmt(x))
    per_family_short["95% CI"] = per_family.apply(lambda r: f"[{fmt(r['ci_low_95'])}, {fmt(r['ci_high_95'])}]", axis=1)
    per_family_short = per_family_short[["family", "point_estimate", "95% CI", "support"]]

    copied_rows = "\n".join(f"| `{src}` | `{dst}` |" for src, dst in copied)
    missing_block = ""
    if missing:
        missing_block = "\n### Missing optional files\n\n" + "\n".join(f"- `{m}`" for m in missing) + "\n"

    new_section = f"""
{START}

## Top-k and family-level bootstrap confidence intervals

Updated from: `{report_md}`

This section consolidates bootstrap confidence intervals for fine-grained top-k and family-level performance. The outputs were copied to:

`{target_dir}`

### Definition used

Family-level top-k metrics are computed by mapping the fine-grained top-k predicted classes to their astronomical families and checking whether the true family appears in that mapped set. Probabilities are **not** summed by family before selecting top-k families.

This definition is consistent with the original top-k/family-level table in the final publication summary.

### Key top-k and family-level conclusion

Fine-grained top-1 accuracy is {fmt(fine_acc['point_estimate'])} with 95% CI {ci_text(fine_acc)}, while fine top-3 accuracy increases to {fmt(fine_top3['point_estimate'])} with 95% CI {ci_text(fine_top3)}, and fine top-5 accuracy increases to {fmt(fine_top5['point_estimate'])} with 95% CI {ci_text(fine_top5)}.

At the family level, mapping the fine-grained predictions to astronomical families gives family top-1 accuracy of {fmt(fam_acc['point_estimate'])} with 95% CI {ci_text(fam_acc)}. Family top-3 accuracy reaches {fmt(fam_top3['point_estimate'])} with 95% CI {ci_text(fam_top3)}, and family top-5 accuracy reaches {fmt(fam_top5['point_estimate'])} with 95% CI {ci_text(fam_top5)}.

These results support the use of AstroTrust-AI as a broker-like triage tool: even when the exact fine-grained subclass remains uncertain, the correct class is often present among the highest-probability fine candidates, and the correct astronomical family is frequently retained after mapping fine top-k candidates to families.

### Key bootstrap numbers

{key_table.to_markdown(index=False)}

### Full bootstrap table

{main.to_markdown(index=False)}

### Per-family top-1 recall from mapped fine top-1

{per_family_short.to_markdown(index=False)}

### Recommended manuscript wording

```latex
We quantified uncertainty in the top-$k$ and family-level metrics using non-parametric bootstrap resampling over the held-out test objects. Although fine-grained top-1 accuracy was \\textbf{{{fine_acc['point_estimate']:.4f}}} (95\\% CI [{fine_acc['ci_low_95']:.4f}, {fine_acc['ci_high_95']:.4f}]), the fine-grained top-3 and top-5 accuracies increased to \\textbf{{{fine_top3['point_estimate']:.4f}}} (95\\% CI [{fine_top3['ci_low_95']:.4f}, {fine_top3['ci_high_95']:.4f}]) and \\textbf{{{fine_top5['point_estimate']:.4f}}} (95\\% CI [{fine_top5['ci_low_95']:.4f}, {fine_top5['ci_high_95']:.4f}]), respectively. For family-level evaluation, we mapped the fine-grained top-$k$ predictions to their corresponding astronomical families. Under this definition, family top-1 accuracy reached \\textbf{{{fam_acc['point_estimate']:.4f}}} (95\\% CI [{fam_acc['ci_low_95']:.4f}, {fam_acc['ci_high_95']:.4f}]), while family top-3 and top-5 accuracies reached \\textbf{{{fam_top3['point_estimate']:.4f}}} (95\\% CI [{fam_top3['ci_low_95']:.4f}, {fam_top3['ci_high_95']:.4f}]) and \\textbf{{{fam_top5['point_estimate']:.4f}}} (95\\% CI [{fam_top5['ci_low_95']:.4f}, {fam_top5['ci_high_95']:.4f}]). These results support the use of AstroTrust-AI as a broker-like triage tool: even when the exact fine-grained subclass is uncertain, the correct class often remains within the high-probability candidate set or within the correct astrophysical family.
```

### Copied top-k family bootstrap files

| source | destination |
|:--|:--|
{copied_rows}
{missing_block}
{END}
""".strip() + "\n"

    text = read_text(final_summary)
    pattern = rf"{re.escape(START)}.*?{re.escape(END)}"
    if re.search(pattern, text, flags=re.S):
        updated = re.sub(pattern, new_section, text, flags=re.S)
    else:
        updated = text.rstrip() + "\n\n" + new_section

    write_text(final_summary, updated)

    print(f"Copied {len(copied)} files to: {target_dir}")
    if missing:
        print("Missing optional files:")
        for m in missing:
            print(f"  - {m}")
    print(f"Updated: {final_summary}")


if __name__ == "__main__":
    main()
