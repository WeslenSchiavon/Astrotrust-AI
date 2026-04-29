from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data" / "processed" / "elasticc2_large"

FOLLOWUP_DIR = RESULTS_DIR / "hybrid_followup_policy_eval_250k"
SUMMARY_DIR = RESULTS_DIR / "final_ai_summary_hybrid"

FORCED_LIGHTCURVES_PATH = DATA_DIR / "forced_lightcurves_250000obj.parquet"
OUTPUT_DIR = DATA_DIR / "dashboard_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    scoring_path = FOLLOWUP_DIR / "hybrid_test_scoring_table.csv"

    if not scoring_path.exists():
        raise FileNotFoundError(scoring_path)

    print(f"Reading scoring table: {scoring_path}")
    scoring = pd.read_csv(scoring_path)

    object_ids = scoring["object_id"].drop_duplicates().astype(int).tolist()

    print(f"Test objects available for dashboard: {len(object_ids)}")

    # Save compact scoring table.
    out_scoring = OUTPUT_DIR / "dashboard_scoring_table.csv"
    scoring.to_csv(out_scoring, index=False)
    print(f"[OK] Saved {out_scoring}")

    # Copy rankings if available.
    ranking_files = [
        "hybrid_test_ranking_novelty_rarity.csv",
        "hybrid_test_ranking_rarity_only.csv",
        "hybrid_test_ranking_previous_discovery.csv",
        "hybrid_test_ranking_fixed_discovery.csv",
    ]

    for filename in ranking_files:
        src = FOLLOWUP_DIR / filename
        if src.exists():
            df = pd.read_csv(src)
            dst = OUTPUT_DIR / filename
            df.to_csv(dst, index=False)
            print(f"[OK] Saved {dst}")
        else:
            print(f"[WARN] Missing ranking file: {src}")

    # Copy summary files.
    summary_files = [
        "final_model_performance_summary.csv",
        "final_hybrid_calibration_summary.csv",
        "final_followup_policy_summary.csv",
        "final_selected_followup_policies.csv",
    ]

    for filename in summary_files:
        src = SUMMARY_DIR / filename
        if src.exists():
            df = pd.read_csv(src)
            dst = OUTPUT_DIR / filename
            df.to_csv(dst, index=False)
            print(f"[OK] Saved {dst}")
        else:
            print(f"[WARN] Missing summary file: {src}")

    # Copy class counts.
    class_counts_path = DATA_DIR / "full_class_counts.csv"
    if class_counts_path.exists():
        class_counts = pd.read_csv(class_counts_path)
        dst = OUTPUT_DIR / "full_class_counts.csv"
        class_counts.to_csv(dst, index=False)
        print(f"[OK] Saved {dst}")
    else:
        print(f"[WARN] Missing class counts file: {class_counts_path}")

    # Build light-curve subset for only test objects.
    # This avoids loading the full large Parquet file inside the dashboard.
    out_lc = OUTPUT_DIR / "dashboard_test_lightcurves.parquet"

    if out_lc.exists():
        print(f"[OK] Light-curve dashboard cache already exists: {out_lc}")
        return

    if not FORCED_LIGHTCURVES_PATH.exists():
        print(f"[WARN] Missing forced light curves: {FORCED_LIGHTCURVES_PATH}")
        return

    print(f"Reading full light curves: {FORCED_LIGHTCURVES_PATH}")
    print("This may take a while, but only needs to be done once.")

    lc = pd.read_parquet(FORCED_LIGHTCURVES_PATH)

    if "object_id" not in lc.columns:
        raise ValueError("Expected column object_id in forced light curves.")

    lc["object_id"] = lc["object_id"].astype(int)

    print("Filtering light curves to test objects...")
    lc_test = lc[lc["object_id"].isin(object_ids)].copy()

    print(f"Rows in test light-curve subset: {len(lc_test)}")
    lc_test.to_parquet(out_lc, index=False)

    print(f"[OK] Saved {out_lc}")


if __name__ == "__main__":
    main()