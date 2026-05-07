from __future__ import annotations

from pathlib import Path
import argparse
import json
import warnings
import random

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False
    LGBMClassifier = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HGB = True
except Exception:
    HAS_HGB = False
    HistGradientBoostingClassifier = None


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_TENSOR_DATASET = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elasticc2_large"
    / "temporal_tensor_v1_250000obj_64bins.npz"
)

DEFAULT_SPLIT_METADATA = (
    ROOT_DIR
    / "results"
    / "hybrid_inference_artifacts_250k"
    / "split_metadata.csv"
)

DEFAULT_LABEL_METADATA = (
    ROOT_DIR
    / "results"
    / "hybrid_inference_artifacts_250k"
    / "label_and_rarity_metadata.json"
)

DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "hierarchical_lightgbm_250k"

RANDOM_STATE = 42
DEFAULT_SPLIT_SEED = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def load_label_names(path: Path, n_classes: int) -> dict[int, str]:
    if not path.exists():
        return {i: str(i) for i in range(n_classes)}

    meta = json.loads(path.read_text(encoding="utf-8"))

    for key in ["label_to_class", "label_to_name", "class_names", "classes"]:
        if key not in meta:
            continue

        value = meta[key]

        if isinstance(value, dict):
            mapping = {}
            for k, v in value.items():
                try:
                    mapping[int(k)] = str(v)
                except Exception:
                    pass
            if mapping:
                for i in range(n_classes):
                    mapping.setdefault(i, str(i))
                return mapping

        if isinstance(value, list):
            mapping = {i: str(v) for i, v in enumerate(value)}
            for i in range(n_classes):
                mapping.setdefault(i, str(i))
            return mapping

    return {i: str(i) for i in range(n_classes)}


def find_array_key(data, candidates: list[str]):
    keys = list(data.keys())
    lower_map = {k.lower(): k for k in keys}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_dataset(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    print("NPZ keys:", list(data.keys()))

    x_key = find_array_key(data, ["X_tab", "x_tab", "tabular", "tabular_features"])
    y_key = find_array_key(data, ["y", "labels", "target", "targets"])
    obj_key = find_array_key(data, ["object_ids", "object_id", "object_ids_str", "ids"])

    if x_key is None or y_key is None:
        raise ValueError(
            f"Could not find X_tab/y in {npz_path}. Available keys: {list(data.keys())}"
        )

    X = np.asarray(data[x_key], dtype=np.float32)
    y = np.asarray(data[y_key]).astype(int)

    if obj_key is not None:
        object_ids = np.asarray(data[obj_key]).astype(str)
    else:
        object_ids = np.arange(len(y)).astype(str)

    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10).astype(np.float32)

    return X, y, object_ids, data


def load_splits(data, split_metadata_path: Path, n_samples: int):
    """Recover train/val/test split from NPZ or split_metadata.csv.

    The export script generated a split_metadata.csv in most runs. This loader is intentionally
    permissive because column names may evolve during experiments.
    """
    for train_key, val_key, test_key in [
        ("train_idx", "val_idx", "test_idx"),
        ("train_indices", "val_indices", "test_indices"),
        ("idx_train", "idx_val", "idx_test"),
    ]:
        if train_key in data and val_key in data and test_key in data:
            return (
                np.asarray(data[train_key]).astype(int),
                np.asarray(data[val_key]).astype(int),
                np.asarray(data[test_key]).astype(int),
                "npz_indices",
            )

    if split_metadata_path.exists():
        split_df = pd.read_csv(split_metadata_path)
        lower = {str(c).lower(): c for c in split_df.columns}

        split_col = None
        for c in ["split", "split_name", "dataset_split", "partition"]:
            if c in lower:
                split_col = lower[c]
                break

        idx_col = None
        for c in ["index", "idx", "row_index", "sample_index", "array_index"]:
            if c in lower:
                idx_col = lower[c]
                break

        if split_col is not None:
            tmp = split_df.copy()

            if idx_col is None:
                tmp["_idx"] = np.arange(len(tmp))
                idx_col = "_idx"

            split_values = tmp[split_col].astype(str).str.lower()
            train_idx = tmp.loc[split_values.str.contains("train"), idx_col].astype(int).to_numpy()
            val_idx = tmp.loc[split_values.str.contains("val"), idx_col].astype(int).to_numpy()
            test_idx = tmp.loc[split_values.str.contains("test"), idx_col].astype(int).to_numpy()

            if len(train_idx) and len(val_idx) and len(test_idx):
                return train_idx, val_idx, test_idx, "split_metadata_csv"

    # Last resort: deterministic split matching the known proportions approximately.
    warnings.warn(
        "Could not recover original split. Falling back to deterministic random split. "
        "Use this only for exploratory runs."
    )
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(0.6375 * n_samples)
    n_val = int(0.1125 * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx, "fallback_random"


def make_model(kind: str, n_classes: int, class_weight=None):
    if kind == "lightgbm":
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm is not installed. Install with: pip install lightgbm")

        return LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            num_class=n_classes if n_classes > 2 else None,
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=96,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if kind == "hgb":
        if not HAS_HGB:
            raise RuntimeError("HistGradientBoostingClassifier is unavailable.")

        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=400,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model kind: {kind}")


def predict_proba_safe(model, X, n_classes: int):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if isinstance(proba, list):
            proba = np.column_stack([p[:, 1] for p in proba])

        proba = np.asarray(proba, dtype=float)

        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])

        if proba.shape[1] != n_classes:
            fixed = np.zeros((len(X), n_classes), dtype=float)
            classes = getattr(model, "classes_", np.arange(proba.shape[1]))
            for j, cls in enumerate(classes):
                if int(cls) < n_classes:
                    fixed[:, int(cls)] = proba[:, j]
            proba = fixed

        row_sum = proba.sum(axis=1, keepdims=True)
        proba = proba / np.maximum(row_sum, 1e-12)
        return proba

    pred = model.predict(X).astype(int)
    proba = np.zeros((len(X), n_classes), dtype=float)
    proba[np.arange(len(X)), pred] = 1.0
    return proba


def evaluate(y_true, y_pred, label_names, label_to_family, name):
    y_true_names = np.array([label_names[int(v)] for v in y_true])
    y_pred_names = np.array([label_names[int(v)] for v in y_pred])
    y_true_family = np.array([label_to_family[int(v)] for v in y_true])
    y_pred_family = np.array([label_to_family[int(v)] for v in y_pred])

    return {
        "model": name,
        "fine_accuracy": accuracy_score(y_true_names, y_pred_names),
        "fine_balanced_accuracy": balanced_accuracy_score(y_true_names, y_pred_names),
        "fine_macro_f1": f1_score(y_true_names, y_pred_names, average="macro"),
        "fine_weighted_f1": f1_score(y_true_names, y_pred_names, average="weighted"),
        "family_accuracy": accuracy_score(y_true_family, y_pred_family),
        "family_balanced_accuracy": balanced_accuracy_score(y_true_family, y_pred_family),
        "family_macro_f1": f1_score(y_true_family, y_pred_family, average="macro"),
        "family_weighted_f1": f1_score(y_true_family, y_pred_family, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a hierarchical family→subclass tabular model on AstroTrust-AI 250k features.")
    parser.add_argument("--tensor-dataset", type=Path, default=DEFAULT_TENSOR_DATASET)
    parser.add_argument("--split-metadata", type=Path, default=DEFAULT_SPLIT_METADATA)
    parser.add_argument("--label-metadata", type=Path, default=DEFAULT_LABEL_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Training random seed for tabular models.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED, help="Fallback split seed; ignored when split metadata is available.")
    parser.add_argument("--model-kind", choices=["lightgbm", "hgb"], default="lightgbm")
    parser.add_argument("--train-family-on-trainval", action="store_true", help="Train final family/subclass models on train+val before testing.")
    args = parser.parse_args()

    global RANDOM_STATE
    RANDOM_STATE = int(args.seed)
    set_global_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training seed: {args.seed}")
    print(f"Split seed:    {args.split_seed}")
    print(f"Output dir:    {args.output_dir}")

    X, y, object_ids, npz_data = load_dataset(args.tensor_dataset)
    n_classes = int(np.max(y)) + 1
    label_names = load_label_names(args.label_metadata, n_classes=n_classes)
    label_to_family = {label: map_to_family(name) for label, name in label_names.items()}

    family_names = np.array([label_to_family[int(v)] for v in y])
    family_encoder = LabelEncoder()
    y_family = family_encoder.fit_transform(family_names)

    train_idx, val_idx, test_idx, split_source = load_splits(npz_data, args.split_metadata, len(y))

    print("Dataset summary:")
    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"Fine classes: {n_classes}")
    print(f"Families: {len(family_encoder.classes_)} -> {list(family_encoder.classes_)}")
    print(f"Split source: {split_source}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    if args.train_family_on_trainval:
        fit_idx = np.concatenate([train_idx, val_idx])
    else:
        fit_idx = train_idx

    # Baseline single-stage tabular model for fair comparison of the same features/model kind.
    print("\nTraining single-stage fine classifier...")
    single_model = make_model(args.model_kind, n_classes=n_classes, class_weight="balanced" if args.model_kind == "lightgbm" else None)
    single_model.fit(X[fit_idx], y[fit_idx])
    single_proba = predict_proba_safe(single_model, X[test_idx], n_classes=n_classes)
    single_pred = np.argmax(single_proba, axis=1)

    # Family classifier.
    print("\nTraining family classifier...")
    family_model = make_model(args.model_kind, n_classes=len(family_encoder.classes_), class_weight="balanced" if args.model_kind == "lightgbm" else None)
    family_model.fit(X[fit_idx], y_family[fit_idx])
    family_proba = predict_proba_safe(family_model, X[test_idx], n_classes=len(family_encoder.classes_))
    family_pred_encoded = np.argmax(family_proba, axis=1)
    family_pred_names = family_encoder.inverse_transform(family_pred_encoded)

    # Subclass classifiers per family.
    print("\nTraining subclass classifiers per family...")
    family_to_model = {}
    family_to_label_encoder = {}
    family_to_labels = {}

    for fam in family_encoder.classes_:
        fam_idx = np.where(family_names[fit_idx] == fam)[0]
        real_idx = fit_idx[fam_idx]
        labels_in_family = sorted(np.unique(y[real_idx]).astype(int).tolist())
        family_to_labels[fam] = labels_in_family

        print(f"  Family {fam}: {len(real_idx)} samples, {len(labels_in_family)} subclasses")

        if len(labels_in_family) <= 1:
            family_to_model[fam] = None
            family_to_label_encoder[fam] = None
            continue

        le = LabelEncoder()
        y_sub = le.fit_transform(y[real_idx])
        model = make_model(args.model_kind, n_classes=len(labels_in_family), class_weight="balanced" if args.model_kind == "lightgbm" else None)
        model.fit(X[real_idx], y_sub)
        family_to_model[fam] = model
        family_to_label_encoder[fam] = le

    # Hierarchical prediction.
    print("\nPredicting hierarchical outputs...")
    hier_pred = []
    hier_conf = []

    for i, global_idx in enumerate(test_idx):
        fam = family_pred_names[i]
        labels_in_family = family_to_labels[fam]
        model = family_to_model[fam]
        le = family_to_label_encoder[fam]

        if model is None or le is None:
            pred_label = labels_in_family[0]
            conf = float(family_proba[i, family_pred_encoded[i]])
        else:
            sub_proba = predict_proba_safe(model, X[global_idx:global_idx + 1], n_classes=len(labels_in_family))[0]
            sub_pred_encoded = int(np.argmax(sub_proba))
            pred_label = int(le.inverse_transform([sub_pred_encoded])[0])
            conf = float(family_proba[i, family_pred_encoded[i]] * sub_proba[sub_pred_encoded])

        hier_pred.append(pred_label)
        hier_conf.append(conf)

    hier_pred = np.asarray(hier_pred, dtype=int)
    hier_conf = np.asarray(hier_conf, dtype=float)

    # Oracle family constrained with trained subclass models.
    oracle_hier_pred = []
    oracle_hier_conf = []

    true_test_family = np.array([label_to_family[int(v)] for v in y[test_idx]])

    for i, global_idx in enumerate(test_idx):
        fam = true_test_family[i]
        labels_in_family = family_to_labels[fam]
        model = family_to_model[fam]
        le = family_to_label_encoder[fam]

        if model is None or le is None:
            pred_label = labels_in_family[0]
            conf = 1.0
        else:
            sub_proba = predict_proba_safe(model, X[global_idx:global_idx + 1], n_classes=len(labels_in_family))[0]
            sub_pred_encoded = int(np.argmax(sub_proba))
            pred_label = int(le.inverse_transform([sub_pred_encoded])[0])
            conf = float(sub_proba[sub_pred_encoded])

        oracle_hier_pred.append(pred_label)
        oracle_hier_conf.append(conf)

    oracle_hier_pred = np.asarray(oracle_hier_pred, dtype=int)
    oracle_hier_conf = np.asarray(oracle_hier_conf, dtype=float)

    rows = [
        evaluate(y[test_idx], single_pred, label_names, label_to_family, f"single_stage_{args.model_kind}"),
        evaluate(y[test_idx], hier_pred, label_names, label_to_family, f"hierarchical_{args.model_kind}"),
        evaluate(y[test_idx], oracle_hier_pred, label_names, label_to_family, f"oracle_family_hierarchical_{args.model_kind}"),
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "hierarchical_lightgbm_summary.csv", index=False)

    predictions = pd.DataFrame({
        "object_id": object_ids[test_idx],
        "true_label": y[test_idx],
        "true_class_name": [label_names[int(v)] for v in y[test_idx]],
        "true_family": [label_to_family[int(v)] for v in y[test_idx]],
        "single_pred_label": single_pred,
        "single_pred_class_name": [label_names[int(v)] for v in single_pred],
        "single_pred_family": [label_to_family[int(v)] for v in single_pred],
        "hier_pred_label": hier_pred,
        "hier_pred_class_name": [label_names[int(v)] for v in hier_pred],
        "hier_pred_family": [label_to_family[int(v)] for v in hier_pred],
        "hier_confidence": hier_conf,
        "oracle_hier_pred_label": oracle_hier_pred,
        "oracle_hier_pred_class_name": [label_names[int(v)] for v in oracle_hier_pred],
        "oracle_hier_pred_family": [label_to_family[int(v)] for v in oracle_hier_pred],
        "oracle_hier_confidence": oracle_hier_conf,
    })
    predictions["single_correct"] = predictions["true_label"] == predictions["single_pred_label"]
    predictions["hier_correct"] = predictions["true_label"] == predictions["hier_pred_label"]
    predictions["oracle_hier_correct"] = predictions["true_label"] == predictions["oracle_hier_pred_label"]
    predictions.to_csv(args.output_dir / "hierarchical_lightgbm_test_predictions.csv", index=False)

    family_model_metrics = {
        "family_classifier_accuracy": accuracy_score(y_family[test_idx], family_pred_encoded),
        "family_classifier_balanced_accuracy": balanced_accuracy_score(y_family[test_idx], family_pred_encoded),
        "family_classifier_macro_f1": f1_score(y_family[test_idx], family_pred_encoded, average="macro"),
        "family_classifier_weighted_f1": f1_score(y_family[test_idx], family_pred_encoded, average="weighted"),
    }
    pd.DataFrame([family_model_metrics]).to_csv(args.output_dir / "family_classifier_metrics.csv", index=False)

    # Per-family subclass performance.
    per_family_rows = []
    for fam in family_encoder.classes_:
        mask = predictions["true_family"] == fam
        if mask.sum() == 0:
            continue
        per_family_rows.append({
            "family": fam,
            "n_test": int(mask.sum()),
            "single_fine_accuracy": float(predictions.loc[mask, "single_correct"].mean()),
            "hier_fine_accuracy": float(predictions.loc[mask, "hier_correct"].mean()),
            "oracle_hier_fine_accuracy": float(predictions.loc[mask, "oracle_hier_correct"].mean()),
        })
    pd.DataFrame(per_family_rows).to_csv(args.output_dir / "per_family_hierarchical_performance.csv", index=False)

    md = "# Hierarchical LightGBM 250k Summary\n\n"
    md += "## Overall metrics\n\n"
    md += summary.to_markdown(index=False)
    md += "\n\n## Family classifier metrics\n\n"
    md += pd.DataFrame([family_model_metrics]).to_markdown(index=False)
    md += "\n\n## Interpretation\n\n"
    md += (
        "- `single_stage` is a tabular model trained directly on 32 fine classes.\n"
        "- `hierarchical` first predicts astronomical family and then predicts a subclass within that family.\n"
        "- `oracle_family_hierarchical` shows the upper bound if the family stage were perfect, using the trained subclass models.\n"
        "- Compare these results to the neural hybrid model and to the oracle upper bound analysis.\n"
    )
    (args.output_dir / "hierarchical_lightgbm_summary.md").write_text(md, encoding="utf-8")

    print("\nFamily classifier metrics:")
    print(pd.DataFrame([family_model_metrics]).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"- {args.output_dir / 'hierarchical_lightgbm_summary.csv'}")
    print(f"- {args.output_dir / 'hierarchical_lightgbm_test_predictions.csv'}")
    print(f"- {args.output_dir / 'family_classifier_metrics.csv'}")
    print(f"- {args.output_dir / 'per_family_hierarchical_performance.csv'}")
    print(f"- {args.output_dir / 'hierarchical_lightgbm_summary.md'}")


if __name__ == "__main__":
    main()
