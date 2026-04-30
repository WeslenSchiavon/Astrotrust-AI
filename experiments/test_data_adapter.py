from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "app"))

from astrotrust_data_adapter import read_and_normalize_from_path, summarize_objects

csv_path = Path(r"C:\Users\wesle\Desktop\Nova pasta (3)\nph_light_curves.csv")

raw_df, lc, report = read_and_normalize_from_path(csv_path)

print("=== Adapter report ===")
print(report.to_dict())

print("\n=== Normalized light curve ===")
print(lc.head())
print(lc.columns)
print(lc.shape)

print("\n=== Object summary ===")
print(summarize_objects(lc).head(10))