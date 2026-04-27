from pathlib import Path
import argparse

import pandas as pd
from astropy.table import Table

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_ROOT = ROOT_DIR / Path("data/raw/elasticc2_train_02")
OUTPUT_ROOT = ROOT_DIR / Path("data/processed/elasticc2_large")


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return value


def decode_dataframe_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(decode_value)

    return df


def class_name_from_dir(path: Path) -> str:
    prefix = "ELASTICC2_TRAIN_02_"
    name = path.name

    if name.startswith(prefix):
        return name.replace(prefix, "", 1)

    return name


def read_head_inventory(head_path: Path, class_name: str, label: int) -> pd.DataFrame:
    table = Table.read(head_path)
    df = table.to_pandas()
    df = decode_dataframe_strings(df)

    required = ["SNID", "NOBS", "PTROBS_MIN", "PTROBS_MAX"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{head_path} is missing columns: {missing}")

    rows = pd.DataFrame({
        "SNID": df["SNID"],
        "class_name": class_name,
        "label": label,
        "NOBS": df["NOBS"],
        "PTROBS_MIN": df["PTROBS_MIN"],
        "PTROBS_MAX": df["PTROBS_MAX"],
        "head_path": str(head_path),
        "phot_path": str(head_path).replace("_HEAD.FITS.gz", "_PHOT.FITS.gz"),
    })

    optional_cols = [
        "SNTYPE",
        "SIM_TYPE_INDEX",
        "SIM_TYPE_NAME",
        "SIM_MODEL_NAME",
        "RA",
        "DEC",
        "MWEBV",
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SNSEP",
    ]

    for col in optional_cols:
        if col in df.columns:
            rows[col] = df[col]

    rows = decode_dataframe_strings(rows)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(
        [p for p in args.raw_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )

    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {args.raw_root}")

    class_names = [class_name_from_dir(p) for p in class_dirs]
    label_map = {name: i for i, name in enumerate(sorted(class_names))}

    all_parts = []

    print(f"Class directories found: {len(class_dirs)}")

    for class_dir in class_dirs:
        class_name = class_name_from_dir(class_dir)
        label = label_map[class_name]

        head_files = sorted(class_dir.glob("*_HEAD.FITS.gz"))

        print("\n" + "=" * 80)
        print(f"Class: {class_name}")
        print(f"Label: {label}")
        print(f"HEAD files: {len(head_files)}")

        if not head_files:
            print(f"[WARN] No HEAD files found for {class_name}")
            continue

        class_parts = []

        for i, head_path in enumerate(head_files, start=1):
            inv = read_head_inventory(
                head_path=head_path,
                class_name=class_name,
                label=label,
            )
            class_parts.append(inv)

            if i % 10 == 0 or i == len(head_files):
                current = sum(len(x) for x in class_parts)
                print(f"  processed {i}/{len(head_files)} HEAD files | objects so far: {current}")

        class_df = pd.concat(class_parts, ignore_index=True)
        all_parts.append(class_df)

        print(f"Class total objects: {len(class_df)}")

    inventory = pd.concat(all_parts, ignore_index=True)

    inventory["global_object_id"] = range(1, len(inventory) + 1)

    class_counts = (
        inventory
        .groupby(["label", "class_name"], as_index=False)
        .agg(
            n_objects=("SNID", "count"),
            n_head_files=("head_path", "nunique"),
            mean_nobs=("NOBS", "mean"),
            median_nobs=("NOBS", "median"),
        )
        .sort_values(["n_objects", "class_name"], ascending=[True, True])
    )

    total_objects = len(inventory)
    class_counts["fraction"] = class_counts["n_objects"] / total_objects
    class_counts["rarity_rank"] = range(1, len(class_counts) + 1)

    output_inventory = args.output_root / "full_inventory.csv"
    output_counts = args.output_root / "full_class_counts.csv"

    inventory.to_csv(output_inventory, index=False)
    class_counts.to_csv(output_counts, index=False)

    print("\n" + "=" * 80)
    print("FULL INVENTORY SUMMARY")
    print("=" * 80)
    print(f"Total objects: {len(inventory)}")
    print(f"Classes: {inventory['class_name'].nunique()}")
    print(f"Saved inventory: {output_inventory}")
    print(f"Saved class counts: {output_counts}")

    print("\nClass counts sorted by rarity:")
    print(class_counts)


if __name__ == "__main__":
    main()