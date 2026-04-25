from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Adiciona a pasta src ao path automaticamente
ROOT_DIR = Path(__file__).resolve().parents[2]  # vai até astrotrust-ai
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))


from src.features.feature_extraction_v2 import build_feature_table


FORCED_LIGHTCURVES_PATH = ROOT_DIR / Path("data/processed/elasticc2/forced_lightcurves_1000obj.parquet")
CONTEXT_PATH = ROOT_DIR / Path("data/processed/elasticc2/object_context_1000obj.parquet")
OUTPUT_PATH = ROOT_DIR / Path("data/processed/elasticc2/features_v3_contextual_1000obj.parquet")


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


def add_position_to_forced_lightcurves(
    forced: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds ra/dec to forced light curves using object-level context.
    The forced-source table itself does not contain ra/dec, but the v2 feature
    extractor expects these columns.
    """
    position = context[["object_id", "ra", "decl"]].drop_duplicates("object_id")

    forced = forced.merge(
        position,
        on="object_id",
        how="left",
    )

    forced = forced.rename(columns={"decl": "dec"})

    forced["ra"] = pd.to_numeric(forced["ra"], errors="coerce").fillna(0.0)
    forced["dec"] = pd.to_numeric(forced["dec"], errors="coerce").fillna(0.0)

    return forced


def main():
    print(f"Loading forced light curves: {FORCED_LIGHTCURVES_PATH}")
    forced = pd.read_parquet(FORCED_LIGHTCURVES_PATH)

    print(f"Loading context: {CONTEXT_PATH}")
    context = pd.read_parquet(CONTEXT_PATH)

    print("Adding ra/dec to forced light curves...")
    forced = add_position_to_forced_lightcurves(forced, context)

    print("Extracting temporal features from forced light curves...")
    temporal_features = build_feature_table(forced)

    print(f"Temporal features: {temporal_features.shape}")

    context = add_context_engineered_features(context)

    # Avoid duplicated label; temporal_features already contains label.
    context_features = context.drop(columns=["label"])

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

    print(f"Final features: {features.shape}")
    print(features.head())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()