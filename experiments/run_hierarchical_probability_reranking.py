from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


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

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "hierarchical_probability_reranking"

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
    mapping: dict[int, str] = {}

    candidate_paths = []

    if label_metadata_path is not None:
        candidate_paths.append(label_metadata_path)

    candidate_paths.append(
        ROOT_DIR
        / "results"
        / "hybrid_inference_artifacts_250k"
        / "label_and_rarity_metadata.json"
    )

    for path in candidate_paths:
        if path is None or not path.exists():
            continue

        try:
            meta = json.loads(path.read_text(encoding="utf-8"))

            for key in ["label_to_class", "label_to_name", "class_names", "classes"]:
                if key not in meta:
                    continue

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


def map_to_family(class_name: str) -> str:
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


def sanitize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / np.maximum(row_sums, eps)
    return p


def topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    topk = np.argsort(probabilities, axis=1)[:, -k:]
    hits = [yt in row for yt, row in zip(y_true, topk)]
    return float(np.mean(hits))


def topk_family_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_to_family: dict[int, str],
    k: int,
) -> float:
    topk = np.argsort(probabilities, axis=1)[:, -k:]
    hits = []

    for yt, row in zip(y_true, topk):
        true_family = label_to_family[int(yt)]
        pred_families = {label_to_family[int(label)] for label in row}
        hits.append(true_family in pred_families)

    return float(np.mean(hits))


def adjust_probabilities_by_family(
    probabilities: np.ndarray,
    label_to_family: dict[int, str],
    alpha: float,
) -> np.ndarray:
    """Re-rank fine classes by the total probability mass of their family.

    adjusted_p(class) = p(class) * p(family(class)) ** alpha

    alpha=0 returns the original probabilities.
    """
    eps = 1e-12
    p = sanitize_probabilities(probabilities)
    n_samples, n_classes = p.shape

    family_mass_by_class = np.zeros_like(p)

    families = sorted(set(label_to_family.values()))
    family_to_labels = {
        fam: [label for label, label_family in label_to_family.items() if label_family == fam]
        for fam in families
    }

    for fam, labels in family_to_labels.items():
        labels = [label for label in labels if 0 <= label < n_classes]
        if not labels:
            continue
        fam_mass = p[:, labels].sum(axis=1, keepdims=True)
        family_mass_by_class[:, labels] = fam_mass

    adjusted = p * np.power(np.maximum(family_mass_by_class, eps), alpha)
    adjusted = sanitize_probabilities(adjusted)
    return adjusted


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    label_names: dict[int, str],
    label_to_family: dict[int, str],
) -> dict:
    y_pred = np.argmax(probabilities, axis=1)

    true_names = np.array([label_names.get(int(v), str(int(v))) for v in y_true])
    pred_names = np.array([label_names.get(int(v), str(int(v))) for v in y_pred])

    true_family = np.array([label_to_family[int(v)] for v in y_true])
    pred_family = np.array([label_to_family[int(v)] for v in y_pred])

    metrics = {
        "fine_accuracy": accuracy_score(true_names, pred_names),
        "fine_balanced_accuracy": balanced_accuracy_score(true_names, pred_names),
        "fine_macro_f1": f1_score(true_names, pred_names, average="macro"),
        "fine_weighted_f1": f1_score(true_names, pred_names, average="weighted"),
        "family_accuracy": accuracy_score(true_family, pred_family),
        "family_balanced_accuracy": balanced_accuracy_score(true_family, pred_family),
        "family_macro_f1": f1_score(true_family, pred_family, average="macro"),
        "family_weighted_f1": f1_score(true_family, pred_family, average="weighted"),
        "top2_accuracy": topk_accuracy(y_true, probabilities, 2),
        "top3_accuracy": topk_accuracy(y_true, probabilities, 3),
        "top5_accuracy": topk_accuracy(y_true, probabilities, 5),
        "top2_family_accuracy": topk_family_accuracy(y_true, probabilities, label_to_family, 2),
        "top3_family_accuracy": topk_family_accuracy(y_true, probabilities, label_to_family, 3),
        "top5_family_accuracy": topk_family_accuracy(y_true, probabilities, label_to_family, 5),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Explore family-aware probability re-ranking without retraining."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--label-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        help="Family-mass exponents to evaluate. alpha=0 is the original model.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)
    if not args.probabilities.exists():
        raise FileNotFoundError(args.probabilities)

    pred_df = pd.read_csv(args.predictions)
    probs = sanitize_probabilities(np.load(args.probabilities))

    true_col = find_column(pred_df, TRUE_CANDIDATES)

    if true_col is None:
        raise ValueError(f"Could not find true label column. Columns: {list(pred_df.columns)}")

    y_true = pred_df[true_col].astype(int).to_numpy()

    label_names = load_label_names(pred_df, args.label_metadata)

    if not label_names:
        n_classes = probs.shape[1]
        label_names = {i: str(i) for i in range(n_classes)}

    label_to_family = {
        int(label): map_to_family(name)
        for label, name in label_names.items()
    }

    # Ensure every probability column has a family.
    for label in range(probs.shape[1]):
        label_to_family.setdefault(label, map_to_family(label_names.get(label, str(label))))
        label_names.setdefault(label, str(label))

    rows = []
    adjusted_probs_by_alpha = {}

    for alpha in args.alphas:
        adjusted = adjust_probabilities_by_family(
            probabilities=probs,
            label_to_family=label_to_family,
            alpha=float(alpha),
        )
        adjusted_probs_by_alpha[float(alpha)] = adjusted

        metrics = evaluate_probabilities(
            y_true=y_true,
            probabilities=adjusted,
            label_names=label_names,
            label_to_family=label_to_family,
        )
        metrics["alpha"] = float(alpha)
        rows.append(metrics)

    grid = pd.DataFrame(rows)
    grid = grid[["alpha"] + [c for c in grid.columns if c != "alpha"]]
    grid.to_csv(args.output_dir / "hierarchical_reranking_grid.csv", index=False)

    best_fine = grid.sort_values(["fine_macro_f1", "fine_accuracy"], ascending=False).iloc[0]
    best_family = grid.sort_values(["family_macro_f1", "family_accuracy"], ascending=False).iloc[0]

    best_summary = pd.DataFrame(
        [
            {"selection": "best_fine_macro_f1", **best_fine.to_dict()},
            {"selection": "best_family_macro_f1", **best_family.to_dict()},
        ]
    )
    best_summary.to_csv(args.output_dir / "hierarchical_reranking_best_summary.csv", index=False)

    # Save predictions from best fine setting.
    best_alpha = float(best_fine["alpha"])
    best_probs = adjusted_probs_by_alpha[best_alpha]
    best_pred = np.argmax(best_probs, axis=1)

    out_pred = pred_df.copy()
    out_pred["hier_alpha"] = best_alpha
    out_pred["hier_predicted_label"] = best_pred
    out_pred["hier_predicted_class_name"] = [label_names.get(int(v), str(int(v))) for v in best_pred]
    out_pred["hier_predicted_family"] = [label_to_family[int(v)] for v in best_pred]
    out_pred["hier_confidence"] = best_probs.max(axis=1)
    out_pred.to_csv(args.output_dir / "hierarchical_reranked_predictions_best_fine.csv", index=False)

    md = "# Hierarchical Probability Re-ranking Analysis\n\n"
    md += "This analysis re-ranks fine-class probabilities using astronomical family probability mass.\n\n"
    md += "Formula: `adjusted_p(class) = p(class) * p(family(class))^alpha`.\n\n"
    md += "## Grid results\n\n"
    md += grid.to_markdown(index=False)
    md += "\n\n## Best settings\n\n"
    md += best_summary.to_markdown(index=False)
    md += "\n\n## Interpretation\n\n"
    md += (
        "- If alpha > 0 improves fine accuracy or Macro-F1, family-level consistency helps classification.\n"
        "- If alpha = 0 remains best, the current probability ranking is already near-optimal under this simple re-ranking.\n"
        "- This is a no-retraining test; a true hierarchical model may still improve beyond this result.\n"
    )
    (args.output_dir / "hierarchical_reranking_summary.md").write_text(md, encoding="utf-8")

    print("[OK] Saved:")
    print(f"- {args.output_dir / 'hierarchical_reranking_grid.csv'}")
    print(f"- {args.output_dir / 'hierarchical_reranking_best_summary.csv'}")
    print(f"- {args.output_dir / 'hierarchical_reranked_predictions_best_fine.csv'}")
    print(f"- {args.output_dir / 'hierarchical_reranking_summary.md'}")

    print("\nGrid results:")
    print(grid.to_string(index=False))

    print("\nBest fine Macro-F1 setting:")
    print(best_fine.to_string())

    print("\nBest family Macro-F1 setting:")
    print(best_family.to_string())


if __name__ == "__main__":
    main()
