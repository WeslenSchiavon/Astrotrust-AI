from pathlib import Path
import argparse
import pandas as pd


def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )


def map_to_family(class_name: str) -> str:
    name = normalize_name(class_name)

    if "clagn" in name or name == "agn":
        return "Active galactic nucleus"

    if "cepheid" in name or name == "rrl" or "rrlyr" in name or "eb" == name:
        return "Variable star"

    if "mdwarf" in name or "m-dwarf" in name:
        return "Variable star"

    if "d-sct" in name or "dsct" in name or "delta-scuti" in name:
        return "Variable star"

    if "dwarf-nova" in name or "dwarfnova" in name:
        return "Cataclysmic variable"

    if name.startswith("kn") or "kilonova" in name:
        return "Kilonova"

    if "ulens" in name or "microlens" in name:
        return "Microlensing"

    if "tde" in name:
        return "Tidal disruption event"

    if "snia" in name or "sn-ia" in name:
        return "Type Ia supernova"

    if "snii" in name or "sn-ii" in name:
        return "Core-collapse supernova"

    if "sniib" in name or "sn-iib" in name:
        return "Core-collapse supernova"

    if "sniin" in name or "sn-iin" in name:
        return "Core-collapse supernova"

    if "snib" in name or "snic" in name or "sn-ib" in name or "sn-ic" in name:
        return "Core-collapse supernova"

    if "slsn" in name:
        return "Superluminous supernova"

    if "pisn" in name:
        return "Pair-instability supernova"

    if "ilot" in name:
        return "Other luminous transient"

    if "cart" in name:
        return "Other rapid transient"

    return "UNKNOWN_REVIEW"


def latex_escape(value) -> str:
    s = str(value)
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
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
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['fine_label_id'])} & "
            f"{latex_escape(row['fine_label_name'])} & "
            f"{latex_escape(row['astronomical_family'])} \\\\"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-counts",
        required=True,
        help="CSV with label and class_name columns.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/final_publication/taxonomy",
    )
    args = parser.parse_args()

    class_counts_path = Path(args.class_counts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(class_counts_path)

    required = {"label", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {class_counts_path}: {missing}")

    mapping = (
        df[["label", "class_name"]]
        .drop_duplicates()
        .copy()
    )

    mapping["label"] = pd.to_numeric(mapping["label"], errors="raise").astype(int)
    mapping = mapping.sort_values("label").reset_index(drop=True)

    mapping = mapping.rename(
        columns={
            "label": "fine_label_id",
            "class_name": "fine_label_name",
        }
    )

    mapping["astronomical_family"] = mapping["fine_label_name"].apply(map_to_family)

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
        print("\nWARNING: labels requiring manual review:")
        print(unknown.to_string(index=False))
    else:
        unknown_path.write_text("No unknown families.\n", encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {tex_rows_path}")
    print(f"Saved: {unknown_path}")

    print("\nTaxonomy preview:")
    print(mapping.to_string(index=False))

if __name__ == "__main__":
    main()