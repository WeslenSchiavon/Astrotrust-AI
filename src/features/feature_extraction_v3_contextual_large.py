from pathlib import Path
import argparse

import sys
import numpy as np
import pandas as pd

# Adiciona a pasta src ao path automaticamente
ROOT_DIR = Path(__file__).resolve().parents[2]  # vai até astrotrust-ai
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.features.feature_extraction_v2 import build_feature_table


INPUT_ROOT = ROOT_DIR / Path("data/processed/elasticc2_large")
OUTPUT_ROOT = ROOT_DIR / Path("data/processed/elasticc2_large")


def add_context_engineered_features(context: pd.DataFrame) -> pd.DataFrame:
    context = context.copy()

    color_pairs = [
        ("u", "g"),
        ("g", "r"),
        ("r", "i"),
        ("i", "z"),
        ("z", "y"),
        ("g", "i"),
        ("r", "z"),
    ]

    for b1, b2 in color_pairs:
        c1 = f"hostgal_mag_{b1}"
        c2 = f"hostgal_mag_{b2}"

        if c1 in context.columns and c2 in context.columns:
            context[f"hostgal_color_{b1}_{b2}"] = context[c1] - context[c2]

    if "hostgal_zphot" in context.columns and "hostgal_zphot_err" in context.columns:
        denom = context["hostgal_zphot"].abs().replace(0, np.nan)
        context["hostgal_zphot_relative_err"] = context["hostgal_zphot_err"] / denom

    if "hostgal_zphot_q100" in context.columns and "hostgal_zphot_q000" in context.columns:
        context["hostgal_zphot_q100_q000_width"] = (
            context["hostgal_zphot_q100"] - context["hostgal_zphot_q000"]
        )

    if "hostgal_zphot_q090" in context.columns and "hostgal_zphot_q010" in context.columns:
        context["hostgal_zphot_q090_q010_width"] = (
            context["hostgal_zphot_q090"] - context["hostgal_zphot_q010"]
        )

    if "hostgal_zphot_q080" in context.columns and "hostgal_zphot_q020" in context.columns:
        context["hostgal_zphot_q080_q020_width"] = (
            context["hostgal_zphot_q080"] - context["hostgal_zphot_q020"]
        )

    return context


def add_position_to_lightcurves(lightcurves: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    position_cols = ["object_id", "ra", "decl"]
    missing = [c for c in position_cols if c not in context.columns]
    if missing:
        raise ValueError(f"Context missing required position columns: {missing}")

    position = context[position_cols].drop_duplicates("object_id")

    lightcurves = lightcurves.merge(position, on="object_id", how="left")
    lightcurves = lightcurves.rename(columns={"decl": "dec"})

    lightcurves["ra"] = pd.to_numeric(lightcurves["ra"], errors="coerce").fillna(0.0)
    lightcurves["dec"] = pd.to_numeric(lightcurves["dec"], errors="coerce").fillna(0.0)

    return lightcurves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)

    args = parser.parse_args()

    suffix = f"{args.n_objects}obj"

    lightcurves_path = args.input_root / f"forced_lightcurves_{suffix}.parquet"
    context_path = args.input_root / f"object_context_{suffix}.parquet"
    output_path = args.output_root / f"features_v3_contextual_{suffix}.parquet"

    print(f"Loading light curves: {lightcurves_path}")
    lightcurves = pd.read_parquet(lightcurves_path)

    print(f"Loading context: {context_path}")
    context = pd.read_parquet(context_path)

    print("Adding ra/dec to light curves...")
    lightcurves = add_position_to_lightcurves(lightcurves, context)

    print("Extracting temporal light-curve features...")
    temporal_features = build_feature_table(lightcurves)

    print(f"Temporal features: {temporal_features.shape}")

    print("Engineering contextual features...")
    context = add_context_engineered_features(context)

    context_features = context.drop(columns=["label"], errors="ignore")

    features = temporal_features.merge(
        context_features,
        on="object_id",
        how="left",
    )

    for col in features.columns:
        if col in ["object_id", "label"]:
            continue
        features[col] = pd.to_numeric(features[col], errors="coerce")

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)

    args.output_root.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Objects: {len(features)}")
    print(f"Features: {features.shape[1]}")
    print("\nClass distribution:")
    print(features["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()