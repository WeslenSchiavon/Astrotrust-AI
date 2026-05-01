from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS = (
    ROOT_DIR
    / "results"
    / "hybrid_temporal_tabular_cnn_250k"
    / "hybrid_temporal_tabular_cnn_test_predictions.csv"
)

DEFAULT_PROBABILITIES = (
    ROOT_DIR
    / "results"
    / "hybrid_temporal_tabular_cnn_250k"
    / "hybrid_temporal_tabular_cnn_test_probabilities.npy"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "error_analysis_hierarchy_topk"


TRUE_CANDIDATES = [
    "true_label",
    "y_true",
    "label",
    "target",
    "true_class",
    "true_class_id",
    "class_id",
]

PRED_CANDIDATES = [
    "predicted_label",
    "y_pred",
    "prediction",
    "pred_label",
    "predicted_class",
    "pred_class_id",
]

TRUE_NAME_CANDIDATES = [
    "true_class_name",
    "true_name",
    "class_name",
    "target_name",
]

PRED_NAME_CANDIDATES = [
    "predicted_class_name",
    "pred_name",
    "prediction_name",
    "predicted_name",
]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_label_names(pred_df: pd.DataFrame, label_metadata_path: Path | None = None) -> dict[int, str]:
    """Try to recover integer-label to class-name mapping."""
    mapping: dict[int, str] = {}

    # 1. Try explicit metadata JSON if provided.
    if label_metadata_path is not None and label_metadata_path.exists():
        try:
            meta = json.loads(label_metadata_path.read_text(encoding="utf-8"))

            for key in ["label_to_class", "label_to_name", "class_names", "classes"]:
                if key in meta:
                    value = meta[key]

                    if isinstance(value, dict):
                        for k, v in value.items():
                            try:
                                mapping[int(k)] = str(v)
                            except Exception:
                                pass

                    elif isinstance(value, list):
                        for idx, name in enumerate(value):
                            mapping[int(idx)] = str(name)

                    if mapping:
                        return mapping
        except Exception:
            pass

    # 2. Try common artifact from inference export.
    default_meta = (
        ROOT_DIR
        / "results"
        / "hybrid_inference_artifacts_250k"
        / "label_and_rarity_metadata.json"
    )

    if default_meta.exists():
        try:
            meta = json.loads(default_meta.read_text(encoding="utf-8"))

            for key in ["label_to_class", "label_to_name", "class_names", "classes"]:
                if key in meta:
                    value = meta[key]

                    if isinstance(value, dict):
                        for k, v in value.items():
                            try:
                                mapping[int(k)] = str(v)
                            except Exception:
                                pass

                    elif isinstance(value, list):
                        for idx, name in enumerate(value):
                            mapping[int(idx)] = str(name)

                    if mapping:
                        return mapping
        except Exception:
            pass

    # 3. Try name columns inside predictions file.
    true_col = find_column(pred_df, TRUE_CANDIDATES)
    true_name_col = find_column(pred_df, TRUE_NAME_CANDIDATES)

    if true_col is not None and true_name_col is not None:
        tmp = pred_df[[true_col, true_name_col]].dropna().drop_duplicates()

        for _, row in tmp.iterrows():
            try:
                mapping[int(row[true_col])] = str(row[true_name_col])
            except Exception:
                pass

    pred_col = find_column(pred_df, PRED_CANDIDATES)
    pred_name_col = find_column(pred_df, PRED_NAME_CANDIDATES)

    if pred_col is not None and pred_name_col is not None:
        tmp = pred_df[[pred_col, pred_name_col]].dropna().drop_duplicates()

        for _, row in tmp.iterrows():
            try:
                mapping[int(row[pred_col])] = str(row[pred_name_col])
            except Exception:
                pass

    return mapping


def normalize_class_name(value, label_names: dict[int, str]) -> str:
    if pd.isna(value):
        return "unknown"

    try:
        label = int(value)
        return label_names.get(label, str(label))
    except Exception:
        return str(value)


def map_to_family(class_name: str) -> str:
    """Astronomically motivated coarse grouping for triage analysis."""
    name = str(class_name).strip()
    low = name.lower()

    if low in ["clagn"] or "agn" in low:
        return "AGN"

    if "cepheid" in low or low == "rrl" or "d-sct" in low or "dsct" in low:
        return "Periodic variable"

    if low == "eb" or "eclips" in low:
        return "Eclipsing binary"

    if "mdwarf" in low or "flare" in low:
        return "Stellar flare"

    if "dwarf-nova" in low or "dwarf nova" in low:
        return "Cataclysmic variable"

    if "ulens" in low or "microlens" in low:
        return "Microlensing"

    if low.startswith("kn") or "kilonova" in low:
        return "Kilonova"

    if "slsn" in low:
        return "Superluminous SN"

    if "tde" in low:
        return "TDE"

    if "ilot" in low:
        return "ILOT"

    if "pisn" in low:
        return "PISN"

    if "snia" in low or "sn ia" in low:
        return "SN Ia"

    if "snii" in low or "sn ii" in low:
        return "SN II"

    if "sniib" in low or "sn iib" in low:
        return "SN II"

    if "sniin" in low or "sn iin" in low:
        return "SN II"

    if "snib" in low or "snic" in low or "sn ib" in low or "sn ic" in low:
        return "SN Ib/c"

    if "cart" in low:
        return "CART"

    return "Other / unknown"


def topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    topk = np.argsort(probabilities, axis=1)[:, -k:]
    hits = [yt in row for yt, row in zip(y_true, topk)]
    return float(np.mean(hits))


def entropy(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.maximum(row_sums, eps)

    k = probabilities.shape[1]
    ent = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return ent / np.log(k)


def save_confusion_heatmap(cm: np.ndarray, labels: list[str], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze current AstroTrust-AI errors, top-k accuracy, and hierarchical family performance."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--label-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-confusions", type=int, default=50)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

    pred_df = pd.read_csv(args.predictions)

    true_col = find_column(pred_df, TRUE_CANDIDATES)
    pred_col = find_column(pred_df, PRED_CANDIDATES)
    true_name_col = find_column(pred_df, TRUE_NAME_CANDIDATES)
    pred_name_col = find_column(pred_df, PRED_NAME_CANDIDATES)

    if true_col is None or pred_col is None:
        raise ValueError(
            "Could not detect true/predicted label columns.\n"
            f"Columns available: {list(pred_df.columns)}\n"
            f"True candidates: {TRUE_CANDIDATES}\n"
            f"Pred candidates: {PRED_CANDIDATES}"
        )

    label_names = load_label_names(pred_df, args.label_metadata)

    y_true = pred_df[true_col].astype(int).to_numpy()
    y_pred = pred_df[pred_col].astype(int).to_numpy()

    if true_name_col is not None:
        true_names = pred_df[true_name_col].astype(str).to_numpy()
    else:
        true_names = np.array([normalize_class_name(v, label_names) for v in y_true])

    if pred_name_col is not None:
        pred_names = pred_df[pred_name_col].astype(str).to_numpy()
    else:
        pred_names = np.array([normalize_class_name(v, label_names) for v in y_pred])

    labels_sorted = sorted(set(true_names).union(set(pred_names)))

    # Fine-grained metrics.
    fine_metrics = {
        "task": "32-class fine labels",
        "accuracy": accuracy_score(true_names, pred_names),
        "balanced_accuracy": balanced_accuracy_score(true_names, pred_names),
        "macro_f1": f1_score(true_names, pred_names, average="macro"),
        "weighted_f1": f1_score(true_names, pred_names, average="weighted"),
    }

    # Family-level metrics.
    true_family = np.array([map_to_family(v) for v in true_names])
    pred_family = np.array([map_to_family(v) for v in pred_names])
    family_labels = sorted(set(true_family).union(set(pred_family)))

    family_metrics = {
        "task": "coarse astronomical families",
        "accuracy": accuracy_score(true_family, pred_family),
        "balanced_accuracy": balanced_accuracy_score(true_family, pred_family),
        "macro_f1": f1_score(true_family, pred_family, average="macro"),
        "weighted_f1": f1_score(true_family, pred_family, average="weighted"),
    }

    metrics_rows = [fine_metrics, family_metrics]

    # Top-k accuracy if probabilities are available.
    if args.probabilities.exists():
        probabilities = np.load(args.probabilities)

        if len(probabilities) != len(y_true):
            print(
                f"[WARN] Probability rows ({len(probabilities)}) do not match predictions ({len(y_true)}). Skipping top-k."
            )
        else:
            for k in [2, 3, 5]:
                fine_metrics[f"top{k}_accuracy"] = topk_accuracy(y_true, probabilities, k)

            pred_entropy = entropy(probabilities)
            pred_df["prediction_entropy"] = pred_entropy
            pred_df["top1_confidence_from_probs"] = probabilities.max(axis=1)

            # Family top-k: hit if any top-k fine class maps to the true family.
            inv_label_names = label_names

            for k in [2, 3, 5]:
                topk = np.argsort(probabilities, axis=1)[:, -k:]
                hits = []

                for i, row in enumerate(topk):
                    pred_fams = {
                        map_to_family(inv_label_names.get(int(label), str(int(label))))
                        for label in row
                    }
                    hits.append(true_family[i] in pred_fams)

                family_metrics[f"top{k}_family_accuracy"] = float(np.mean(hits))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "error_analysis_metrics_summary.csv", index=False)

    # Per-class report.
    class_report = classification_report(
        true_names,
        pred_names,
        output_dict=True,
        zero_division=0,
    )
    class_report_df = pd.DataFrame(class_report).T.reset_index().rename(columns={"index": "class_name"})
    class_report_df.to_csv(output_dir / "per_class_classification_report.csv", index=False)

    family_report = classification_report(
        true_family,
        pred_family,
        output_dict=True,
        zero_division=0,
    )
    family_report_df = pd.DataFrame(family_report).T.reset_index().rename(columns={"index": "family"})
    family_report_df.to_csv(output_dir / "per_family_classification_report.csv", index=False)

    # Confusion matrix and most confused pairs.
    cm = confusion_matrix(true_names, pred_names, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_df.to_csv(output_dir / "confusion_matrix_32_classes.csv")
    save_confusion_heatmap(
        cm,
        labels_sorted,
        output_dir / "confusion_matrix_32_classes.png",
        "AstroTrust-AI 32-class confusion matrix",
    )

    rows = []
    for i, true_label in enumerate(labels_sorted):
        for j, pred_label in enumerate(labels_sorted):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                true_total = int(cm[i, :].sum())
                rows.append(
                    {
                        "true_class": true_label,
                        "predicted_class": pred_label,
                        "count": count,
                        "true_class_total": true_total,
                        "fraction_of_true_class": count / max(true_total, 1),
                        "same_family": map_to_family(true_label) == map_to_family(pred_label),
                        "true_family": map_to_family(true_label),
                        "predicted_family": map_to_family(pred_label),
                    }
                )

    confusions_df = pd.DataFrame(rows).sort_values(
        ["count", "fraction_of_true_class"],
        ascending=[False, False],
    )
    confusions_df.to_csv(output_dir / "most_confused_class_pairs.csv", index=False)
    confusions_df.head(args.top_confusions).to_csv(
        output_dir / "top_confused_class_pairs.csv",
        index=False,
    )

    family_cm = confusion_matrix(true_family, pred_family, labels=family_labels)
    family_cm_df = pd.DataFrame(family_cm, index=family_labels, columns=family_labels)
    family_cm_df.to_csv(output_dir / "confusion_matrix_families.csv")
    save_confusion_heatmap(
        family_cm,
        family_labels,
        output_dir / "confusion_matrix_families.png",
        "AstroTrust-AI family-level confusion matrix",
    )

    # Enriched predictions file for downstream analysis.
    enriched = pred_df.copy()
    enriched["true_class_name_resolved"] = true_names
    enriched["predicted_class_name_resolved"] = pred_names
    enriched["true_family"] = true_family
    enriched["predicted_family"] = pred_family
    enriched["family_correct"] = true_family == pred_family
    enriched["fine_correct"] = true_names == pred_names
    enriched.to_csv(output_dir / "test_predictions_with_families.csv", index=False)



    total = len(enriched)
    fine_errors = enriched[~enriched["fine_correct"]].copy()
    n_fine_errors = len(fine_errors)

    same_family_errors = fine_errors[
        fine_errors["true_family"] == fine_errors["predicted_family"]
    ]

    cross_family_errors = fine_errors[
        fine_errors["true_family"] != fine_errors["predicted_family"]
    ]

    error_decomposition = {
        "n_total": int(total),
        "n_fine_correct": int(enriched["fine_correct"].sum()),
        "n_fine_errors": int(n_fine_errors),
        "n_same_family_errors": int(len(same_family_errors)),
        "n_cross_family_errors": int(len(cross_family_errors)),
        "fraction_errors_same_family": float(len(same_family_errors) / max(n_fine_errors, 1)),
        "fraction_errors_cross_family": float(len(cross_family_errors) / max(n_fine_errors, 1)),
        "family_accuracy": float(enriched["family_correct"].mean()),
        "fine_accuracy": float(enriched["fine_correct"].mean()),
    }

    pd.DataFrame([error_decomposition]).to_csv(
        output_dir / "error_decomposition_summary.csv",
        index=False,
    )

    print("\nError decomposition:")
    for k, v in error_decomposition.items():
        print(f"{k}: {v}")



    # Markdown summary.
    md = "# AstroTrust-AI Error, Hierarchy, and Top-k Analysis\n\n"
    md += "## Metrics summary\n\n"
    md += metrics_df.to_markdown(index=False)
    md += "\n\n## Top confused fine-grained class pairs\n\n"
    md += confusions_df.head(20).to_markdown(index=False)
    md += "\n\n## Interpretation guide\n\n"
    md += (
        "- If family-level accuracy is much higher than 32-class accuracy, most errors are between fine subclasses.\n"
        "- If confused pairs are within the same family, the model may still be useful for broker-level triage.\n"
        "- Top-3/top-5 accuracy is relevant because follow-up decisions can use ranked hypotheses, not only top-1 labels.\n"
        "- This analysis should guide whether the next improvement should be hierarchical classification, stronger ensembling, or more training data.\n"
    )

    (output_dir / "error_analysis_summary.md").write_text(md, encoding="utf-8")

    print("[OK] Saved analysis to:")
    print(output_dir)
    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False))
    print("\nTop confused pairs:")
    print(confusions_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
