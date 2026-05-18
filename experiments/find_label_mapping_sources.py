from pathlib import Path
import json
import pickle
import re

import pandas as pd


ROOT = Path(".")
OUT_DIR = Path("results/final_publication/taxonomy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_CLASS_PATTERNS = [
    "SNIa", "SNIax", "SNII", "SNIb", "SNIc", "KN_K17", "CART",
    "uLens", "dwarf-nova", "AGN", "TDE", "Cepheid", "RR",
]


def looks_like_class_name(value):
    if pd.isna(value):
        return False

    s = str(value)
    return any(p.lower() in s.lower() for p in KNOWN_CLASS_PATTERNS)


def inspect_csv_or_parquet(path):
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=5000)
        elif path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
            df = df.head(5000)
        else:
            return []

        candidates = []

        for col in df.columns:
            name = col.lower()

            if any(k in name for k in ["label", "class", "target", "type", "name"]):
                sample = df[col].dropna().astype(str).head(100).tolist()
                n_unique = df[col].dropna().nunique()

                has_known_name = any(looks_like_class_name(v) for v in sample)

                if has_known_name or n_unique <= 40:
                    candidates.append(
                        {
                            "file": str(path),
                            "column": col,
                            "n_unique_sample": int(n_unique),
                            "sample_values": sample[:15],
                        }
                    )

        return candidates

    except Exception:
        return []


def inspect_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))

        text = json.dumps(data)[:5000]

        if any(p.lower() in text.lower() for p in KNOWN_CLASS_PATTERNS):
            return [{
                "file": str(path),
                "type": "json_with_known_class_names",
                "preview": text[:1000],
            }]

        if isinstance(data, dict):
            keys = list(data.keys())
            if any(k.lower() in ["classes", "class_names", "label_names", "target_names", "mapping"] for k in keys):
                return [{
                    "file": str(path),
                    "type": "json_possible_mapping",
                    "keys": keys[:30],
                    "preview": text[:1000],
                }]

        if isinstance(data, list) and len(data) <= 100:
            return [{
                "file": str(path),
                "type": "json_list_possible_classes",
                "preview": data[:40],
            }]

    except Exception:
        return []

    return []


def inspect_pickle(path):
    try:
        try:
            import joblib
            obj = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f)

        if hasattr(obj, "classes_"):
            return [{
                "file": str(path),
                "type": "pickle_label_encoder_classes_",
                "classes": [str(x) for x in list(obj.classes_)],
            }]

        if isinstance(obj, dict):
            hits = {}
            for k, v in obj.items():
                if any(x in str(k).lower() for x in ["class", "label", "target", "mapping"]):
                    hits[str(k)] = str(v)[:500]

            if hits:
                return [{
                    "file": str(path),
                    "type": "pickle_dict_possible_mapping",
                    "keys": list(obj.keys())[:30],
                    "hits": hits,
                }]

    except Exception:
        return []

    return []


def inspect_py_or_txt(path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")

        if any(p.lower() in text.lower() for p in KNOWN_CLASS_PATTERNS):
            lines = []
            for i, line in enumerate(text.splitlines(), start=1):
                if any(p.lower() in line.lower() for p in KNOWN_CLASS_PATTERNS):
                    lines.append(f"L{i}: {line.strip()}")

            return [{
                "file": str(path),
                "type": "text_or_python_contains_class_names",
                "matches": lines[:40],
            }]

    except Exception:
        return []

    return []


def main():
    all_candidates = []

    suffixes = [".csv", ".parquet", ".pq", ".json", ".pkl", ".pickle", ".joblib", ".py", ".txt", ".md"]

    files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]

    for path in files:
        # evitar varrer ambientes virtuais e caches
        if any(part.lower() in [".git", "__pycache__", ".venv", "venv", "env", "site-packages"] for part in path.parts):
            continue

        suffix = path.suffix.lower()

        if suffix in [".csv", ".parquet", ".pq"]:
            all_candidates.extend(inspect_csv_or_parquet(path))
        elif suffix == ".json":
            all_candidates.extend(inspect_json(path))
        elif suffix in [".pkl", ".pickle", ".joblib"]:
            all_candidates.extend(inspect_pickle(path))
        elif suffix in [".py", ".txt", ".md"]:
            all_candidates.extend(inspect_py_or_txt(path))

    out_md = OUT_DIR / "label_mapping_source_candidates.md"

    lines = ["# Candidate files for fine-label mapping\n"]

    if not all_candidates:
        lines.append("No candidate mapping files found.\n")
    else:
        for idx, cand in enumerate(all_candidates, start=1):
            lines.append(f"\n## Candidate {idx}\n")
            for k, v in cand.items():
                lines.append(f"- **{k}**: `{v}`\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Found {len(all_candidates)} candidate entries.")
    print(f"Saved: {out_md}")

    for cand in all_candidates[:30]:
        print("\n---")
        print(cand.get("file"))
        print(cand.get("type", "table"))
        if "column" in cand:
            print("column:", cand["column"])
            print("sample:", cand["sample_values"])
        elif "classes" in cand:
            print("classes:", cand["classes"])
        elif "matches" in cand:
            print("\n".join(cand["matches"][:8]))


if __name__ == "__main__":
    main()