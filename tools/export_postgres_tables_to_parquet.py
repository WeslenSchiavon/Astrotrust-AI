from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


DB_URL = "postgresql+psycopg2://astrotrust:astrotrust@localhost:5432/tom_desc"
OUTPUT_DIR = Path("C:/Users/wesle/Desktop/Astrolara/MeuProjeto/astrotrust-ai/data/processed/elasticc2/tables")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        tables = pd.read_sql(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """),
            conn,
        )

        print("\nTables found:")
        print(tables)

        for table_name in tables["table_name"]:
            print(f"\nExporting {table_name}...")

            query = text(f'SELECT * FROM public."{table_name}"')
            df = pd.read_sql(query, conn)

            output_path = OUTPUT_DIR / f"{table_name}.parquet"
            df.to_parquet(output_path, index=False)

            print(f"[OK] {table_name}: {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()