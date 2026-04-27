from pathlib import Path

from astropy.table import Table

ROOT_DIR = Path(__file__).resolve().parents[1]

ROOT = ROOT_DIR / Path("data/raw/elasticc2_train_02")


def find_first_pair():
    head_files = sorted(ROOT.rglob("*_HEAD.FITS.gz"))

    if not head_files:
        raise FileNotFoundError("No *_HEAD.FITS.gz files found.")

    head_path = head_files[0]
    phot_path = Path(str(head_path).replace("_HEAD.FITS.gz", "_PHOT.FITS.gz"))

    if not phot_path.exists():
        raise FileNotFoundError(f"Matching PHOT file not found for: {head_path}")

    return head_path, phot_path


def print_table_info(name: str, path: Path, max_rows: int = 5):
    print("\n" + "=" * 100)
    print(f"{name}: {path}")
    print("=" * 100)

    table = Table.read(path)

    print(f"Rows: {len(table)}")
    print(f"Columns ({len(table.colnames)}):")
    for col in table.colnames:
        print(f"  - {col}: {table[col].dtype}")

    print("\nPreview:")
    print(table[:max_rows])

    return table


def main():
    head_path, phot_path = find_first_pair()

    head = print_table_info("HEAD", head_path)
    phot = print_table_info("PHOT", phot_path)

    print("\n" + "=" * 100)
    print("Useful candidate columns")
    print("=" * 100)

    for col in [
        "SNID",
        "SNTYPE",
        "SIM_TYPE_INDEX",
        "RA",
        "DECL",
        "MWEBV",
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "PTROBS_MIN",
        "PTROBS_MAX",
    ]:
        if col in head.colnames:
            print(f"HEAD has {col}")

    for col in [
        "MJD",
        "BAND",
        "FLT",
        "FLUXCAL",
        "FLUXCALERR",
        "PHOTFLAG",
    ]:
        if col in phot.colnames:
            print(f"PHOT has {col}")


if __name__ == "__main__":
    main()