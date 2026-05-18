from pathlib import Path
import argparse
import json
import pickle
import re
import sys

import pandas as pd


ID_COLUMNS = [
    "fine_label_id",
    "model_fine_label_id",
    "class_id",
    "target_id",
    "true_target",
    "target",
    "y_true",
    "true_label",
    "label",
]

NAME_COLUMNS = [
    "fine_label_name",
    "class_name",
    "target_name",
    "true_target_name",
    "true_class",
    "label_name",
    "class",
    "type",
]


def normalize_label_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.replace("-", "")
    s = s.replace(".", "")
    return s


FAMILY_BY_NORMALIZED_NAME = {
    # Type Ia supernovae
    "snia": "Type Ia supernova",
    "snia91bg": "Type Ia supernova",
    "sniax": "Type Ia supernova",
    "sniax": "Type Ia supernova",
    "sniax": "Type Ia supernova",

    # Core-collapse supernovae
    "snii": "Core-collapse supernova",
    "sniin": "Core-collapse supernova",
    "sniib": "Core-collapse supernova",
    "snib": "Core-collapse supernova",
    "snic": "Core-collapse supernova",
    "snicbl": "Core-collapse supernova",

    # Superluminous / pair-instability
    "slsni": "Superluminous supernova",
    "slsnii": "Superluminous supernova",
    "pisn": "Pair-instability supernova",

    # Kilonovae
    "kn": "Kilonova",
    "knb19": "Kilonova",
    "knk17": "Kilonova",

    # TDE
    "tde": "Tidal disruption event",

    # Microlensing
    "ulenssingle": "Microlensing",
    "ulensbinary": "Microlensing",
    "ulensstring": "Microlensing",

    # AGN
    "agn": "Active galactic nucleus",

    # Cataclysmic variables
    "dwarfnova": "Cataclysmic variable",
    "dwarfnovae": "Cataclysmic variable",

    # Variable stars
    "cepheid": "Variable star",
    "rrl": "Variable star",
    "rrlyrae": "Variable star",
    "rrlyr": "Variable star",
    "dscut": "Variable star",
    "dscuti": "Variable star",
    "deltascuti": "Variable star",
    "eb": "Variable star",
    "eclipsingbinary": "Variable star",
    "mdwarf": "Variable star",
    "mdwarfflare": "Variable star",

    # Other transients
    "ilot": "Other luminous transient",
    "cart": "Other rapid transient",
}


def infer_family(name: str) -> str:
    key = normalize_label_name(name)

    if key in FAMILY_BY_NORMALIZED_NAME:
        return FAMILY_BY_NORMALIZED_NAME[key]

    if key.startswith("snia"):
        return "Type Ia supernova"

    if key.startswith(("snii", "snib", "snic")):
        return "Core-collapse supernova"

    if key.startswith("slsn"):
        return "Superluminous supernova"

    if key.startswith("kn"):
        return "Kilonova"

    if key.startswith("ulens"):
        return "Microlensing"

    return "UNKNOWN_REVIEW"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported table format: {path}")


def load_label_encoder_classes(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            return data

        for key in ["classes", "classes_", "class_names", "label_names", "fine_label_names"]:
            if key in data:
                return data[key]

        if isinstance(data, dict):
            # Case: {"0": "SNIa", "1": "SNII", ...}
            if all(str(k).isdigit() for k in data.keys()):
                return [data[str(i)] for i in sorted(map(int, data.keys()))]

    if suffix in [".pkl", ".pickle", ".joblib"]:
        try:
            import joblib
            obj = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f)

        if hasattr(obj, "classes_"):
            return list(obj.classes_)

        if isinstance(obj, dict):
            for key in ["classes", "classes_", "class_names", "label_names", "fine_label_names"]:
                if key in obj:
                    return obj[key]

    return None


def find_first_existing_column(df: pd.DataFrame, candidates):
    lower_to_original = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

    return None


def extract_mapping_from_predictions(predictions_path: Path, classes=None) -> pd.DataFrame:
    df = read_table(predictions_path)

    id_col = find_first_existing_column(df, ID_COLUMNS)
    name_col = find_first_existing_column(df, NAME_COLUMNS)

    rows = []

    if id_col is not None and name_col is not None:
        tmp = df[[id_col, name_col]].dropna().drop_duplicates()
        tmp = tmp.rename(columns={id_col: "model_fine_label_id", name_col: "fine_label_name"})
        tmp["model_fine_label_id"] = tmp["model_fine_label_id"].astype(int)
        return tmp.sort_values("model_fine_label_id").reset_index(drop=True)

    if classes is not None:
        return pd.DataFrame(
            {
                "model_fine_label_id": list(range(len(classes))),
                "fine_label_name": [str(c) for c in classes],
            }
        )

    # Try probability columns: prob_SNIa, p_SNIa, classprob_SNIa
    prob_cols = [
        c for c in df.columns
        if c.lower().startswith(("prob_", "proba_", "p_"))
    ]

    class_names = []
    for c in prob_cols:
        name = re.sub(r"^(prob_|proba_|p_)", "", c, flags=re.IGNORECASE)
        if name and not name.isdigit():
            class_names.append(name)

    if class_names:
        return pd.DataFrame(
            {
                "model_fine_label_id": list(range(len(class_names))),
                "fine_label_name": class_names,
            }
        )

    raise RuntimeError(
        "Could not infer label mapping from prediction file. "
        "Pass --label-encoder pointing to the LabelEncoder/class_names file."
    )


def add_dataset_target_id_if_available(mapping: pd.DataFrame, truth_path: Path | None) -> pd.DataFrame:
    if truth_path is None:
        mapping["dataset_target_id"] = ""
        return mapping

    truth = read_table(truth_path)

    target_id_col = find_first_existing_column(
        truth,
        ["true_target", "target", "target_id", "class_id", "dataset_target_id"],
    )

    name_col = find_first_existing_column(
        truth,
        ["class_name", "target_name", "true_target_name", "true_class", "fine_label_name", "type"],
    )

    if target_id_col is None or name_col is None:
        mapping["dataset_target_id"] = ""
        return mapping

    aux = truth[[target_id_col, name_col]].dropna().drop_duplicates()
    aux = aux.rename(
        columns={
            target_id_col: "dataset_target_id",
            name_col: "fine_label_name",
        }
    )

    merged = mapping.merge(aux, on="fine_label_name", how="left")
    return merged


def latex_escape(s: str) -> str:
    s = str(s)
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s


def write_markdown(df: pd.DataFrame, path: Path):
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_rows(df: pd.DataFrame, path: Path):
    cols = list(df.columns)
    lines = []

    for _, row in df.iterrows():
        vals = [latex_escape(row[c]) for c in cols]
        lines.append(" & ".join(vals) + r" \\")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--label-encoder", type=str, default=None)
    parser.add_argument("--truth-file", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/final_publication/taxonomy")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = None
    if args.label_encoder:
        classes = load_label_encoder_classes(Path(args.label_encoder))
        if classes is None:
            raise RuntimeError(f"Could not read classes from {args.label_encoder}")

    mapping = extract_mapping_from_predictions(predictions_path, classes=classes)

    truth_path = Path(args.truth_file) if args.truth_file else None
    mapping = add_dataset_target_id_if_available(mapping, truth_path)

    mapping["astronomical_family"] = mapping["fine_label_name"].apply(infer_family)

    # Put columns in a clean order
    preferred_cols = [
        "model_fine_label_id",
        "dataset_target_id",
        "fine_label_name",
        "astronomical_family",
    ]
    mapping = mapping[[c for c in preferred_cols if c in mapping.columns]]

    mapping = mapping.sort_values("model_fine_label_id").reset_index(drop=True)

    csv_path = out_dir / "fine_label_taxonomy_mapping.csv"
    md_path = out_dir / "fine_label_taxonomy_mapping.md"
    tex_rows_path = out_dir / "fine_label_taxonomy_rows.tex"
    unknown_path = out_dir / "unknown_family_labels.txt"

    mapping.to_csv(csv_path, index=False)
    write_markdown(mapping, md_path)
    write_latex_rows(mapping, tex_rows_path)

    unknown = mapping[mapping["astronomical_family"] == "UNKNOWN_REVIEW"]
    if len(unknown) > 0:
        unknown_path.write_text(
            "\n".join(unknown["fine_label_name"].astype(str).tolist()),
            encoding="utf-8",
        )
    else:
        unknown_path.write_text("No unknown families.\n", encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {tex_rows_path}")
    print(f"Saved: {unknown_path}")

    if len(unknown) > 0:
        print("\nWARNING: Some labels need manual family review:")
        print(unknown[["model_fine_label_id", "fine_label_name"]].to_string(index=False))


if __name__ == "__main__":
    main()