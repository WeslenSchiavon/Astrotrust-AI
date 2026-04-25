from pathlib import Path
import os

import pandas as pd
from sqlalchemy import create_engine, text


DB_URL = "postgresql+psycopg2://astrotrust:astrotrust@localhost:5432/tom_desc"

CURRENT_DIR = Path("C:/Users/wesle/Desktop/Astrolara/MeuProjeto/astrotrust-ai")

OUTPUT_DIR = CURRENT_DIR / Path("data/processed/elasticc2")
OUTPUT_LIGHTCURVES =  OUTPUT_DIR / "lightcurves_1000obj.parquet"
OUTPUT_METADATA = OUTPUT_DIR / "metadata_1000obj.parquet"
OUTPUT_CLASS_COUNTS = OUTPUT_DIR / "class_counts_1000obj.csv"


def load_lightcurves(engine) -> pd.DataFrame:
    query = text("""
        SELECT
            s.diaobject_id AS object_id,
            s.diasource_id AS source_id,
            s.midpointtai AS mjd,
            s.filtername AS band,
            s.psflux AS flux,
            s.psfluxerr AS flux_err,
            s.snr AS snr,
            s.ra AS ra,
            s.decl AS dec,
            t.gentype AS label,
            t.zcmb AS zcmb,
            t.zhelio AS zhelio,
            t.mwebv AS mwebv,
            t.peakmjd AS peakmjd,
            t.mjd_detect_first AS mjd_detect_first,
            t.mjd_detect_last AS mjd_detect_last,
            t.nobs AS nobs
        FROM public.temp_copy_elasticc2_ppdbdiasource s
        INNER JOIN public.temp_copy_elasticc2_diaobjecttruth t
            ON s.diaobject_id = t.diaobject_id
        ORDER BY s.diaobject_id, s.midpointtai;
    """)

    return pd.read_sql(query, engine)


def build_metadata(lightcurves: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "object_id",
        "label",
        "ra",
        "dec",
        "zcmb",
        "zhelio",
        "mwebv",
        "peakmjd",
        "mjd_detect_first",
        "mjd_detect_last",
        "nobs",
    ]

    metadata = (
        lightcurves[metadata_cols]
        .drop_duplicates(subset=["object_id"])
        .sort_values("object_id")
        .reset_index(drop=True)
    )

    return metadata


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DB_URL)

    print("Loading joined light-curve data from PostgreSQL...")
    lightcurves = load_lightcurves(engine)

    print(f"Rows: {len(lightcurves)}")
    print(f"Objects: {lightcurves['object_id'].nunique()}")
    print(f"Classes: {lightcurves['label'].nunique()}")

    print("\nPreview:")
    print(lightcurves.head())

    print("\nClass distribution:")
    class_counts = (
        lightcurves[["object_id", "label"]]
        .drop_duplicates()
        .groupby("label")
        .size()
        .reset_index(name="n_objects")
        .sort_values("n_objects", ascending=False)
    )
    print(class_counts)

    metadata = build_metadata(lightcurves)

    lightcurves.to_parquet(OUTPUT_LIGHTCURVES, index=False)
    metadata.to_parquet(OUTPUT_METADATA, index=False)
    class_counts.to_csv(OUTPUT_CLASS_COUNTS, index=False)

    print("\nSaved files:")
    print(f"- {OUTPUT_LIGHTCURVES}")
    print(f"- {OUTPUT_METADATA}")
    print(f"- {OUTPUT_CLASS_COUNTS}")


if __name__ == "__main__":
    main()