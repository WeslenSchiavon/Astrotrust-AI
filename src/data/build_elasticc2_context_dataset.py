from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


DB_URL = "postgresql+psycopg2://astrotrust:astrotrust@localhost:5432/tom_desc"

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT_DIR /  Path("data/processed/elasticc2")
OUTPUT_FORCED = OUTPUT_DIR / "forced_lightcurves_1000obj.parquet"
OUTPUT_CONTEXT = OUTPUT_DIR / "object_context_1000obj.parquet"


def load_forced_lightcurves(engine) -> pd.DataFrame:
    query = text("""
        SELECT
            f.diaobject_id AS object_id,
            f.diaforcedsource_id AS source_id,
            f.midpointtai AS mjd,
            f.filtername AS band,
            f.psflux AS flux,
            f.psfluxerr AS flux_err,
            CASE
                WHEN f.psfluxerr IS NULL OR f.psfluxerr = 0 THEN 0
                ELSE f.psflux / f.psfluxerr
            END AS snr,
            t.gentype AS label
        FROM public.temp_copy_elasticc2_ppdbdiaforcedsource f
        INNER JOIN public.temp_copy_elasticc2_diaobjecttruth t
            ON f.diaobject_id = t.diaobject_id
        ORDER BY f.diaobject_id, f.midpointtai;
    """)

    return pd.read_sql(query, engine)


def load_object_context(engine) -> pd.DataFrame:
    query = text("""
        SELECT
            o.diaobject_id AS object_id,
            o.ra,
            o.decl,
            o.mwebv,
            o.mwebv_err,
            o.hostgal_ellipticity,
            o.hostgal_sqradius,
            o.hostgal_zphot,
            o.hostgal_zphot_err,
            o.hostgal_zphot_q000,
            o.hostgal_zphot_q010,
            o.hostgal_zphot_q020,
            o.hostgal_zphot_q030,
            o.hostgal_zphot_q040,
            o.hostgal_zphot_q050,
            o.hostgal_zphot_q060,
            o.hostgal_zphot_q070,
            o.hostgal_zphot_q080,
            o.hostgal_zphot_q090,
            o.hostgal_zphot_q100,
            o.hostgal_mag_u,
            o.hostgal_mag_g,
            o.hostgal_mag_r,
            o.hostgal_mag_i,
            o.hostgal_mag_z,
            o.hostgal_mag_y,
            o.hostgal_magerr_u,
            o.hostgal_magerr_g,
            o.hostgal_magerr_r,
            o.hostgal_magerr_i,
            o.hostgal_magerr_z,
            o.hostgal_magerr_y,
            o.hostgal_snsep,
            o.isddf,
            t.gentype AS label
        FROM public.temp_copy_elasticc2_ppdbdiaobject o
        INNER JOIN public.temp_copy_elasticc2_diaobjecttruth t
            ON o.diaobject_id = t.diaobject_id
        ORDER BY o.diaobject_id;
    """)

    return pd.read_sql(query, engine)


def clean_context_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert boolean to numeric.
    if "isddf" in df.columns:
        df["isddf"] = df["isddf"].astype(int)

    # ELAsTiCC/TOM-style sentinel values for missing catalog information.
    sentinel_values = [-9999, -999, -99, -9, 999]

    for col in df.columns:
        if col in ["object_id", "label"]:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(sentinel_values, np.nan)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DB_URL)

    print("Loading forced light curves...")
    forced = load_forced_lightcurves(engine)

    print(f"Forced rows: {len(forced)}")
    print(f"Forced objects: {forced['object_id'].nunique()}")
    print(f"Forced classes: {forced['label'].nunique()}")
    print(forced.head())

    print("\nLoading object context...")
    context = load_object_context(engine)
    context = clean_context_values(context)

    print(f"Context rows: {len(context)}")
    print(f"Context objects: {context['object_id'].nunique()}")
    print(f"Context classes: {context['label'].nunique()}")
    print(context.head())

    forced.to_parquet(OUTPUT_FORCED, index=False)
    context.to_parquet(OUTPUT_CONTEXT, index=False)

    print("\nSaved:")
    print(f"- {OUTPUT_FORCED}")
    print(f"- {OUTPUT_CONTEXT}")


if __name__ == "__main__":
    main()