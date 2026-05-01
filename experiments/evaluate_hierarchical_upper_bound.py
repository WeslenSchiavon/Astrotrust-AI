from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS = (
    ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_250k" / "hybrid_temporal_tabular_cnn_test_predictions.csv"
)
DEFAULT_PROBABILITIES = (
    ROOT_DIR / "results" / "hybrid_temporal_tabular_cnn_250k" / "hybrid_temporal_tabular_cnn_test_probabilities.npy"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "hierarchical_upper_bound"

TRUE_CANDIDATES = ["true_label", "y_true", "label", "target", "true_class", "true_class_id", "class_id"]
PRED_CANDIDATES = ["predicted_label", "y_pred", "prediction", "pred_label", "predicted_class", "pred_class_id"]
TRUE_NAME_CANDIDATES = ["true_class_name", "true_name", "class_name", "target_name"]
PRED_NAME_CANDIDATES = ["predicted_class_name", "pred_name", "prediction_name", "predicted_name"]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def load_label_names(pred_df: pd.DataFrame, label_metadata_path: Path | None = None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    paths = []
    if label_metadata_path is not None:
        paths.append(label_metadata_path)
    paths.append(ROOT_DIR / "results" / "hybrid_inference_artifacts_250k" / "label_and_rarity_metadata.json")

    for path in paths:
        if not path.exists():
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
    if true_col and true_name_col:
        for _, row in pred_df[[true_col, true_name_col]].dropna().drop_duplicates().iterrows():
            try:
                mapping[int(row[true_col])] = str(row[true_name_col])
            except Exception:
                pass

    pred_col = find_column(pred_df, PRED_CANDIDATES)
    pred_name_col = find_column(pred_df, PRED_NAME_CANDIDATES)
    if pred_col and pred_name_col:
        for _, row in pred_df[[pred_col, pred_name_col]].dropna().drop_duplicates().iterrows():
            try:
                mapping[int(row[pred_col])] = str(row[pred_name_col])
            except Exception:
                pass

    return mapping


def map_to_family(class_name: str) -> str:
    low = str(class_name).strip().lower()

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
    if "snii" in low or "sn ii" in low or "sniib" in low or "sniin" in low:
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
    p = p / np.maximum(p.sum(axis=1, keepdims=True), eps)
    return p


def compute_family_mass(probabilities: np.ndarray, label_to_family: dict[int, str]):
    families = sorted(set(label_to_family.values()))
    family_to_idx = {fam: i for i, fam in enumerate(families)}
    mass = np.zeros((probabilities.shape[0], len(families)), dtype=float)

    for label in range(probabilities.shape[1]):
        fam = label_to_family[label]
        mass[:, family_to_idx[fam]] += probabilities[:, label]

    return mass, families, family_to_idx


def predict_family_mass_then_subclass(probabilities: np.ndarray, label_to_family: dict[int, str]) -> np.ndarray:
    """Hard hierarchical no-retraining baseline.

    Step 1: pick the family with highest total probability mass.
    Step 2: inside that family, pick the fine class with highest probability.
    """
    mass, families, _ = compute_family_mass(probabilities, label_to_family)
    pred_family_idx = np.argmax(mass, axis=1)
    pred_labels = []

    for i, fam_idx in enumerate(pred_family_idx):
        fam = families[int(fam_idx)]
        labels = [label for label, f in label_to_family.items() if f == fam and label < probabilities.shape[1]]
        best_label = max(labels, key=lambda label: probabilities[i, label])
        pred_labels.append(best_label)

    return np.array(pred_labels, dtype=int)


def predict_oracle_family_then_subclass(y_true: np.ndarray, probabilities: np.ndarray, label_to_family: dict[int, str]) -> np.ndarray:
    """Upper bound if the first-stage family classifier were perfect."""
    pred_labels = []

    for i, yt in enumerate(y_true):
        fam = label_to_family[int(yt)]
        labels = [label for label, f in label_to_family.items() if f == fam and label < probabilities.shape[1]]
        best_label = max(labels, key=lambda label: probabilities[i, label])
        pred_labels.append(best_label)

    return np.array(pred_labels, dtype=int)


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, label_names: dict[int, str], label_to_family: dict[int, str], name: str) -> dict:
    true_names = np.array([label_names.get(int(v), str(int(v))) for v in y_true])
    pred_names = np.array([label_names.get(int(v), str(int(v))) for v in y_pred])
    true_fam = np.array([label_to_family[int(v)] for v in y_true])
    pred_fam = np.array([label_to_family[int(v)] for v in y_pred])

    return {
        "method": name,
        "fine_accuracy": accuracy_score(true_names, pred_names),
        "fine_balanced_accuracy": balanced_accuracy_score(true_names, pred_names),
        "fine_macro_f1": f1_score(true_names, pred_names, average="macro"),
        "fine_weighted_f1": f1_score(true_names, pred_names, average="weighted"),
        "family_accuracy": accuracy_score(true_fam, pred_fam),
        "family_balanced_accuracy": balanced_accuracy_score(true_fam, pred_fam),
        "family_macro_f1": f1_score(true_fam, pred_fam, average="macro"),
        "family_weighted_f1": f1_score(true_fam, pred_fam, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical upper bound and hard family-mass baseline.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--label-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(args.predictions)
    probs = sanitize_probabilities(np.load(args.probabilities))

    true_col = find_column(pred_df, TRUE_CANDIDATES)
    pred_col = find_column(pred_df, PRED_CANDIDATES)
    if true_col is None:
        raise ValueError(f"Could not detect true label column. Columns: {list(pred_df.columns)}")

    y_true = pred_df[true_col].astype(int).to_numpy()
    y_original = pred_df[pred_col].astype(int).to_numpy() if pred_col is not None else np.argmax(probs, axis=1)

    label_names = load_label_names(pred_df, args.label_metadata)
    if not label_names:
        label_names = {i: str(i) for i in range(probs.shape[1])}

    for label in range(probs.shape[1]):
        label_names.setdefault(label, str(label))

    label_to_family = {label: map_to_family(name) for label, name in label_names.items()}

    y_family_mass = predict_family_mass_then_subclass(probs, label_to_family)
    y_oracle_family = predict_oracle_family_then_subclass(y_true, probs, label_to_family)

    rows = [
        evaluate_prediction(y_true, y_original, label_names, label_to_family, "original_top1"),
        evaluate_prediction(y_true, y_family_mass, label_names, label_to_family, "family_mass_then_subclass"),
        evaluate_prediction(y_true, y_oracle_family, label_names, label_to_family, "oracle_true_family_then_subclass"),
    ]

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "hierarchical_upper_bound_summary.csv", index=False)

    out_pred = pred_df.copy()
    out_pred["family_mass_predicted_label"] = y_family_mass
    out_pred["family_mass_predicted_class"] = [label_names[int(v)] for v in y_family_mass]
    out_pred["family_mass_predicted_family"] = [label_to_family[int(v)] for v in y_family_mass]
    out_pred["oracle_family_predicted_label"] = y_oracle_family
    out_pred["oracle_family_predicted_class"] = [label_names[int(v)] for v in y_oracle_family]
    out_pred["oracle_family_predicted_family"] = [label_to_family[int(v)] for v in y_oracle_family]
    out_pred.to_csv(args.output_dir / "hierarchical_upper_bound_predictions.csv", index=False)

    md = "# Hierarchical Upper Bound Analysis\n\n"
    md += df.to_markdown(index=False)
    md += "\n\n## Interpretation\n\n"
    md += (
        "- `family_mass_then_subclass` is a no-retraining hard hierarchical baseline.\n"
        "- `oracle_true_family_then_subclass` is an upper bound assuming a perfect first-stage family classifier.\n"
        "- If the oracle result is much higher than the original result, a trained hierarchical classifier may be worthwhile.\n"
    )
    (args.output_dir / "hierarchical_upper_bound_summary.md").write_text(md, encoding="utf-8")

    print("[OK] Saved:")
    print(f"- {args.output_dir / 'hierarchical_upper_bound_summary.csv'}")
    print(f"- {args.output_dir / 'hierarchical_upper_bound_predictions.csv'}")
    print(f"- {args.output_dir / 'hierarchical_upper_bound_summary.md'}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
