#!/usr/bin/env python3
"""Recommend a binary decision threshold from saved CV predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep classification thresholds using per-fold validation predictions and "
            "recommend a threshold that prioritizes sensitivity."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory that contains fold_*/val_predictions_best.csv",
    )
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        default=None,
        help="Optional single CSV with columns y_true,y_prob (overrides --outputs-dir scan).",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.05,
        help="Minimum threshold for sweep (inclusive).",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.95,
        help="Maximum threshold for sweep (inclusive).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold step size for sweep.",
    )
    parser.add_argument(
        "--min-sensitivity",
        type=float,
        default=0.90,
        help="Target minimum sensitivity. Recommender picks highest specificity meeting this target.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/threshold_sweep.csv"),
        help="Where to save threshold sweep table.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/threshold_recommendation.json"),
        help="Where to save recommendation summary JSON.",
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "auc": auc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def collect_prediction_files(outputs_dir: Path) -> List[Path]:
    files = sorted(outputs_dir.glob("fold_*/val_predictions_best.csv"))
    if not files:
        raise FileNotFoundError(
            f"No fold prediction files found in {outputs_dir}. "
            "Expected files like outputs/fold_1/val_predictions_best.csv"
        )
    return files


def load_predictions(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, int]:
    if args.prediction_csv is not None:
        df = pd.read_csv(args.prediction_csv)
        required_cols = {"y_true", "y_prob"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{args.prediction_csv} must contain columns: {required_cols}")
        y_true = df["y_true"].to_numpy(dtype=np.int64)
        y_prob = df["y_prob"].to_numpy(dtype=np.float64)
        return y_true, y_prob, 1

    files = collect_prediction_files(args.outputs_dir)
    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)
    required_cols = {"y_true", "y_prob"}
    if not required_cols.issubset(merged.columns):
        raise ValueError(
            f"Fold prediction CSVs must contain columns {required_cols}. "
            "Please rerun training with the latest template."
        )
    y_true = merged["y_true"].to_numpy(dtype=np.int64)
    y_prob = merged["y_prob"].to_numpy(dtype=np.float64)
    return y_true, y_prob, len(files)


def recommend_threshold(results: pd.DataFrame, min_sensitivity: float) -> Dict[str, object]:
    # Primary strategy: among thresholds meeting sensitivity target, maximize specificity.
    candidates = results[results["recall_sensitivity"] >= min_sensitivity]
    if not candidates.empty:
        sorted_candidates = candidates.sort_values(
            by=["specificity", "f1", "threshold"], ascending=[False, False, True]
        )
        chosen = sorted_candidates.iloc[0]
        strategy = (
            "highest_specificity_with_sensitivity_constraint"
        )
    else:
        # Fallback: maximize sensitivity, then maximize specificity, then use lower threshold.
        sorted_all = results.sort_values(
            by=["recall_sensitivity", "specificity", "threshold"], ascending=[False, False, True]
        )
        chosen = sorted_all.iloc[0]
        strategy = "max_sensitivity_fallback"

    record = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in chosen.to_dict().items()}
    return {"strategy": strategy, "min_sensitivity_target": float(min_sensitivity), "recommended": record}


def main() -> None:
    args = parse_args()
    if not (0.0 < args.step <= 1.0):
        raise ValueError("--step must be in (0, 1].")
    if args.min_threshold >= args.max_threshold:
        raise ValueError("--min-threshold must be < --max-threshold.")
    if not (0.0 <= args.min_sensitivity <= 1.0):
        raise ValueError("--min-sensitivity must be in [0, 1].")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    y_true, y_prob, source_count = load_predictions(args)
    if len(y_true) == 0:
        raise ValueError("No predictions loaded.")

    thresholds = np.arange(args.min_threshold, args.max_threshold + (args.step / 2.0), args.step)
    rows = [compute_metrics(y_true, y_prob, float(t)) for t in thresholds]
    results = pd.DataFrame(rows)
    results.to_csv(args.out_csv, index=False)

    recommendation = recommend_threshold(results, min_sensitivity=args.min_sensitivity)
    recommendation["num_samples"] = int(len(y_true))
    recommendation["num_prediction_sources"] = int(source_count)

    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)

    rec = recommendation["recommended"]
    print("===== Threshold recommendation =====")
    print(f"Strategy: {recommendation['strategy']}")
    print(f"Samples: {recommendation['num_samples']}")
    print(f"Recommended threshold: {rec['threshold']:.3f}")
    print(
        "Metrics @ threshold: "
        f"sensitivity={rec['recall_sensitivity']:.4f}, "
        f"specificity={rec['specificity']:.4f}, "
        f"f1={rec['f1']:.4f}, "
        f"accuracy={rec['accuracy']:.4f}, "
        f"auc={rec['auc']:.4f}"
    )
    print(f"Saved sweep table: {args.out_csv}")
    print(f"Saved recommendation: {args.out_json}")


if __name__ == "__main__":
    main()
