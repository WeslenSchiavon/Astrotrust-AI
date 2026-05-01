from __future__ import annotations

from pathlib import Path
import argparse
import json
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "final_probability_ensemble_search_250k"
DEFAULT_LABEL_METADATA = ROOT_DIR / "results" / "hybrid_inference_artifacts_250k" / "label_and_rarity_metadata.json"

TRUE_CANDIDATES = [
    "true_label", "y_true", "label", "target", "true_class", "true_class_id", "class_id"
]
PRED_CANDIDATES = [
    "predicted_label", "y_pred", "prediction", "pred_label", "predicted_class", "pred_class_id"
]

EXCLUDE_PROBABILITY_PATTERNS = [
    "family_test_probabilities",      # family-level, not 32-class probabilities
    "hierarchical",                  # may not be directly comparable
]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def sanitize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    row_sum = p.sum(axis=1, keepdims=True)
    p = p / np.maximum(row_sum, eps)
    return p


def load_label_names(path: Path, n_classes: int) -> dict[int, str]:
    if not path.exists():
        return {i: str(i) for i in range(n_classes)}

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {i: str(i) for i in range(n_classes)}

    for key in ["label_to_class", "label_to_name", "class_names", "classes"]:
        if key not in meta:
            continue

        value = meta[key]
        if isinstance(value, dict):
            mapping = {}
            for k, v in value.items():
                try:
                    mapping[int(k)] = str(v)
                except Exception:
                    pass
            if mapping:
                for i in range(n_classes):
                    mapping.setdefault(i, str(i))
                return mapping

        if isinstance(value, list):
            mapping = {i: str(v) for i, v in enumerate(value)}
            for i in range(n_classes):
                mapping.setdefault(i, str(i))
            return mapping

    return {i: str(i) for i in range(n_classes)}


def topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    topk = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if not mask.any():
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    p = sanitize_probabilities(probabilities)
    y_pred = np.argmax(p, axis=1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "top2_accuracy": topk_accuracy(y_true, p, 2),
        "top3_accuracy": topk_accuracy(y_true, p, 3),
        "top5_accuracy": topk_accuracy(y_true, p, 5),
        "mean_confidence": float(p.max(axis=1).mean()),
        "ece": expected_calibration_error(y_true, p),
    }


def find_prediction_csv(prob_path: Path) -> Path | None:
    folder = prob_path.parent
    candidates = sorted(folder.glob("*predictions*.csv"))

    if not candidates:
        candidates = sorted(folder.glob("*.csv"))

    if not candidates:
        return None

    # Prefer files with true-label columns and same broad prefix.
    scored = []
    prob_stem = prob_path.stem.lower().replace("probabilities", "").replace("probs", "")

    for path in candidates:
        try:
            head = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        has_true = find_column(head, TRUE_CANDIDATES) is not None
        name = path.stem.lower()
        prefix_score = 1 if any(tok and tok in name for tok in prob_stem.split("_")[:3]) else 0
        scored.append((int(has_true), prefix_score, -len(name), path))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][-1]


def infer_split_name(path: Path) -> str:
    low = str(path).lower()
    if "val" in low and "test" not in low:
        return "val"
    if "validation" in low and "test" not in low:
        return "val"
    if "test" in low:
        return "test"
    return "unknown"


def should_exclude_probability(path: Path) -> bool:
    low = str(path).lower()
    return any(pattern.lower() in low for pattern in EXCLUDE_PROBABILITY_PATTERNS)


def load_candidate(prob_path: Path, n_classes_expected: int | None = None) -> dict | None:
    if should_exclude_probability(prob_path):
        return None

    try:
        probabilities = sanitize_probabilities(np.load(prob_path))
    except Exception as exc:
        print(f"[SKIP] Could not load probabilities {prob_path}: {exc}")
        return None

    if probabilities.ndim != 2:
        print(f"[SKIP] Probability file is not 2D: {prob_path}")
        return None

    if n_classes_expected is not None and probabilities.shape[1] != n_classes_expected:
        print(
            f"[SKIP] Class count mismatch: {prob_path} has {probabilities.shape[1]} columns, "
            f"expected {n_classes_expected}."
        )
        return None

    pred_csv = find_prediction_csv(prob_path)
    if pred_csv is None:
        print(f"[SKIP] No prediction CSV found for {prob_path}")
        return None

    try:
        pred_df = pd.read_csv(pred_csv)
    except Exception as exc:
        print(f"[SKIP] Could not read prediction CSV {pred_csv}: {exc}")
        return None

    true_col = find_column(pred_df, TRUE_CANDIDATES)
    if true_col is None:
        print(f"[SKIP] No true-label column in {pred_csv}")
        return None

    if len(pred_df) != len(probabilities):
        print(
            f"[SKIP] Row mismatch for {prob_path}: probabilities={len(probabilities)}, "
            f"predictions={len(pred_df)}"
        )
        return None

    y_true = pred_df[true_col].astype(int).to_numpy()
    model_name = prob_path.parent.name + "__" + prob_path.stem

    return {
        "model_name": model_name,
        "prob_path": str(prob_path),
        "pred_csv": str(pred_csv),
        "split": infer_split_name(prob_path),
        "probabilities": probabilities,
        "y_true": y_true,
    }


def discover_candidates(results_dir: Path, n_classes_expected: int | None = 32) -> list[dict]:
    prob_paths = sorted(results_dir.glob("**/*probabilities*.npy"))

    candidates = []
    for path in prob_paths:
        cand = load_candidate(path, n_classes_expected=n_classes_expected)
        if cand is not None:
            candidates.append(cand)

    return candidates


def align_candidates(candidates: list[dict], split: str | None = None) -> list[dict]:
    if split is not None:
        candidates = [c for c in candidates if c["split"] == split]

    if not candidates:
        return []

    # Use the most common y_true signature and shape.
    signatures = {}
    for c in candidates:
        sig = (len(c["y_true"]), int(c["probabilities"].shape[1]), hash(c["y_true"].tobytes()))
        signatures.setdefault(sig, []).append(c)

    best_sig = max(signatures, key=lambda k: len(signatures[k]))
    aligned = signatures[best_sig]
    return aligned


def ensemble_probabilities(candidates: list[dict], weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.maximum(weights.sum(), 1e-12)
    probs = np.zeros_like(candidates[0]["probabilities"], dtype=float)

    for w, c in zip(weights, candidates):
        probs += float(w) * c["probabilities"]

    return sanitize_probabilities(probs)


def random_weight_search(
    candidates: list[dict],
    y_true: np.ndarray,
    n_trials: int,
    top_k_models: int,
    seed: int,
    optimize_metric: str,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    # Restrict to strongest individual models for tractability.
    individual_rows = []
    for i, c in enumerate(candidates):
        metrics = evaluate_probabilities(y_true, c["probabilities"])
        individual_rows.append({"idx": i, "model_name": c["model_name"], **metrics})

    individual_df = pd.DataFrame(individual_rows).sort_values(optimize_metric, ascending=False)
    selected_indices = individual_df.head(min(top_k_models, len(individual_df)))["idx"].to_numpy(dtype=int)
    selected = [candidates[i] for i in selected_indices]

    rows = []
    best_score = -np.inf
    best_weights = None
    best_metrics = None

    # Always test uniform weights.
    trial_weights = [np.ones(len(selected)) / len(selected)]

    # Also include each single model as a degenerate ensemble.
    for i in range(len(selected)):
        w = np.zeros(len(selected))
        w[i] = 1.0
        trial_weights.append(w)

    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(len(selected)))
        trial_weights.append(w)

    for trial_id, weights in enumerate(trial_weights):
        if trial_id % 100 == 0:
            print(f"  ensemble trial {trial_id}/{len(trial_weights)}", flush=True)

        probs = ensemble_probabilities(selected, weights)
        metrics = evaluate_probabilities(y_true, probs)
        score = metrics[optimize_metric]

        row = {
            "trial_id": trial_id,
            "n_models": len(selected),
            "weights_json": json.dumps({selected[i]["model_name"]: float(weights[i]) for i in range(len(selected))}),
            **metrics,
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best_weights = weights.copy()
            best_metrics = metrics.copy()

    search_df = pd.DataFrame(rows).sort_values(optimize_metric, ascending=False).reset_index(drop=True)
    best_info = {
        "selected_model_names": [c["model_name"] for c in selected],
        "selected_prob_paths": [c["prob_path"] for c in selected],
        "weights": {selected[i]["model_name"]: float(best_weights[i]) for i in range(len(selected))},
        "metrics": best_metrics,
    }

    return search_df, best_weights, best_info


def build_paired_val_test(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    val = align_candidates(candidates, split="val")
    test = align_candidates(candidates, split="test")

    if not val or not test:
        return [], []

    val_by_prefix = {}
    for c in val:
        prefix = c["model_name"].replace("val", "").replace("validation", "")
        val_by_prefix[prefix] = c

    paired_val = []
    paired_test = []

    for t in test:
        t_prefix = t["model_name"].replace("test", "")
        best_key = None
        for key in val_by_prefix:
            if key and key in t_prefix or t_prefix and t_prefix in key:
                best_key = key
                break

        if best_key is not None:
            paired_val.append(val_by_prefix[best_key])
            paired_test.append(t)

    return paired_val, paired_test


def main():
    parser = argparse.ArgumentParser(description="Final probability ensemble search for AstroTrust-AI 250k outputs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-metadata", type=Path, default=DEFAULT_LABEL_METADATA)
    parser.add_argument("--n-classes", type=int, default=32)
    parser.add_argument("--n-trials", type=int, default=5000)
    parser.add_argument("--top-k-models", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize-metric", default="macro_f1", choices=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"])
    parser.add_argument(
        "--allow-test-tuning",
        action="store_true",
        help=(
            "If no validation probability files are found, tune weights on test predictions. "
            "This is exploratory only and should not be reported as an unbiased final result."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering candidate probability files...")
    candidates = discover_candidates(args.results_dir, n_classes_expected=args.n_classes)

    if not candidates:
        raise FileNotFoundError(f"No compatible probability files found under {args.results_dir}")

    candidate_rows = []
    for c in candidates:
        metrics = evaluate_probabilities(c["y_true"], c["probabilities"])
        candidate_rows.append({
            "model_name": c["model_name"],
            "split": c["split"],
            "prob_path": c["prob_path"],
            "pred_csv": c["pred_csv"],
            **metrics,
        })

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["split", args.optimize_metric], ascending=[True, False])
    candidate_df.to_csv(args.output_dir / "candidate_probability_files_metrics.csv", index=False)

    print("\nCandidate probability files:")
    print(candidate_df[["model_name", "split", "accuracy", "macro_f1", "top3_accuracy", "ece"]].to_string(index=False))

    # Prefer validation tuning if paired val/test outputs exist.
    paired_val, paired_test = build_paired_val_test(candidates)

    tuning_mode = None
    tune_candidates = None
    eval_candidates = None

    if paired_val and paired_test:
        tuning_mode = "validation_tuned_test_evaluated"
        tune_candidates = paired_val
        eval_candidates = paired_test
        print("\nUsing validation probabilities for weight tuning and test probabilities for final evaluation.")
    else:
        test_candidates = align_candidates(candidates, split="test")
        if not test_candidates:
            test_candidates = align_candidates(candidates, split=None)

        if not args.allow_test_tuning:
            print("\n[STOP] No paired validation/test probability files were found.")
            print("To run an exploratory test-tuned ensemble, rerun with: --allow-test-tuning")
            print("This exploratory result should not be reported as an unbiased final test result.")
            return

        tuning_mode = "test_tuned_exploratory"
        tune_candidates = test_candidates
        eval_candidates = test_candidates
        warnings.warn(
            "Tuning ensemble weights on the test set. This is exploratory only and should not be used as a final unbiased result."
        )

    y_tune = tune_candidates[0]["y_true"]
    y_eval = eval_candidates[0]["y_true"]

    print(f"\nTuning mode: {tuning_mode}")
    print(f"Tuning candidates: {len(tune_candidates)}")
    print(f"Evaluation candidates: {len(eval_candidates)}")

    search_df, best_weights, best_info_tune = random_weight_search(
        candidates=tune_candidates,
        y_true=y_tune,
        n_trials=args.n_trials,
        top_k_models=args.top_k_models,
        seed=args.seed,
        optimize_metric=args.optimize_metric,
    )
    search_df.to_csv(args.output_dir / "ensemble_weight_search_results.csv", index=False)

    selected_names = best_info_tune["selected_model_names"]

    # Apply weights to evaluation candidates with matching order by model name where possible.
    eval_by_name = {c["model_name"]: c for c in eval_candidates}
    selected_eval = []
    selected_eval_weights = []

    for name, weight in best_info_tune["weights"].items():
        if name in eval_by_name:
            selected_eval.append(eval_by_name[name])
            selected_eval_weights.append(weight)
        else:
            # Fallback: match by folder prefix/stem simplification.
            simplified = name.replace("val", "").replace("validation", "")
            match = None
            for c in eval_candidates:
                c_simplified = c["model_name"].replace("test", "")
                if simplified in c_simplified or c_simplified in simplified:
                    match = c
                    break
            if match is not None:
                selected_eval.append(match)
                selected_eval_weights.append(weight)

    if not selected_eval:
        # In test-tuned mode, this should not happen.
        selected_eval = tune_candidates
        selected_eval_weights = list(best_info_tune["weights"].values())

    selected_eval_weights = np.asarray(selected_eval_weights, dtype=float)
    selected_eval_weights = selected_eval_weights / np.maximum(selected_eval_weights.sum(), 1e-12)

    eval_probs = ensemble_probabilities(selected_eval, selected_eval_weights)
    eval_metrics = evaluate_probabilities(y_eval, eval_probs)
    eval_pred = np.argmax(eval_probs, axis=1)

    label_names = load_label_names(args.label_metadata, n_classes=args.n_classes)

    summary = {
        "tuning_mode": tuning_mode,
        "optimize_metric": args.optimize_metric,
        "n_trials": args.n_trials,
        "top_k_models": args.top_k_models,
        "is_test_tuned_exploratory": tuning_mode == "test_tuned_exploratory",
        "selected_models": [c["model_name"] for c in selected_eval],
        "weights": {selected_eval[i]["model_name"]: float(selected_eval_weights[i]) for i in range(len(selected_eval))},
        "tuning_metrics": best_info_tune["metrics"],
        "evaluation_metrics": eval_metrics,
    }

    (args.output_dir / "best_ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame([
        {
            "tuning_mode": tuning_mode,
            "is_test_tuned_exploratory": tuning_mode == "test_tuned_exploratory",
            "optimize_metric": args.optimize_metric,
            "n_trials": args.n_trials,
            "top_k_models": args.top_k_models,
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
        }
    ])
    summary_df.to_csv(args.output_dir / "best_ensemble_summary.csv", index=False)

    predictions = pd.DataFrame({
        "true_label": y_eval,
        "true_class_name": [label_names.get(int(v), str(int(v))) for v in y_eval],
        "ensemble_predicted_label": eval_pred,
        "ensemble_predicted_class_name": [label_names.get(int(v), str(int(v))) for v in eval_pred],
        "ensemble_confidence": eval_probs.max(axis=1),
        "ensemble_correct": eval_pred == y_eval,
    })
    predictions.to_csv(args.output_dir / "best_ensemble_predictions.csv", index=False)
    np.save(args.output_dir / "best_ensemble_probabilities.npy", eval_probs)

    md = "# Final Probability Ensemble Search\n\n"
    md += f"Tuning mode: `{tuning_mode}`\n\n"
    if tuning_mode == "test_tuned_exploratory":
        md += "**Warning:** weights were tuned on the test set. Treat as exploratory upper-bound only.\n\n"
    md += "## Evaluation metrics\n\n"
    md += summary_df.to_markdown(index=False)
    md += "\n\n## Selected weights\n\n"
    md += pd.DataFrame([
        {"model_name": name, "weight": weight}
        for name, weight in summary["weights"].items()
    ]).to_markdown(index=False)
    md += "\n\n## Candidate files\n\n"
    md += candidate_df[["model_name", "split", "accuracy", "macro_f1", "top3_accuracy", "ece"]].to_markdown(index=False)
    (args.output_dir / "final_probability_ensemble_summary.md").write_text(md, encoding="utf-8")

    print("\nBest ensemble evaluation metrics:")
    for k, v in eval_metrics.items():
        print(f"{k}: {v:.6f}")

    print("\nSelected weights:")
    for name, weight in summary["weights"].items():
        print(f"{weight:.4f}  {name}")

    print("\nSaved:")
    print(f"- {args.output_dir / 'candidate_probability_files_metrics.csv'}")
    print(f"- {args.output_dir / 'ensemble_weight_search_results.csv'}")
    print(f"- {args.output_dir / 'best_ensemble_summary.csv'}")
    print(f"- {args.output_dir / 'best_ensemble_summary.json'}")
    print(f"- {args.output_dir / 'best_ensemble_predictions.csv'}")
    print(f"- {args.output_dir / 'best_ensemble_probabilities.npy'}")
    print(f"- {args.output_dir / 'final_probability_ensemble_summary.md'}")


if __name__ == "__main__":
    main()
