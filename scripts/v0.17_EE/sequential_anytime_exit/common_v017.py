from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
V016_COMMON_DIR = PROJECT_ROOT / "scripts" / "v0.16_EE" / "multiobjective_per_label_margin"
for path in (PROJECT_ROOT, V016_COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v016 import (  # noqa: E402,F401
    ParentMetricContext,
    collect_outputs,
    jsonable,
    load_checkpoint,
    load_feature,
    load_json,
    load_labels,
    load_run_config,
    load_thresholds_by_exit,
    multilabel_metrics,
    parse_tap_blocks,
    resolve_model_cfg,
    robust_upper_bound,
    save_json,
    synchronize,
    threshold_mapping,
)


def sequential_constraint_violation(
    *,
    macro_upper: float,
    micro_upper: float,
    exact_upper: float,
    hamming_upper: float,
    total_early_fraction: float,
    exit1_fraction: float,
    max_macro_drop: float,
    max_micro_drop: float,
    max_exact_drop: float,
    max_hamming_increase: float,
    min_total_early_fraction: float,
    min_exit1_fraction: float,
) -> float:
    terms = [
        max(0.0, macro_upper - max_macro_drop) / max(max_macro_drop, 1e-9),
        max(0.0, micro_upper - max_micro_drop) / max(max_micro_drop, 1e-9),
        max(0.0, exact_upper - max_exact_drop) / max(max_exact_drop, 1e-9),
        max(0.0, hamming_upper - max_hamming_increase)
        / max(max_hamming_increase, 1e-9),
        max(0.0, min_total_early_fraction - total_early_fraction)
        / max(min_total_early_fraction, 1e-9),
        max(0.0, min_exit1_fraction - exit1_fraction)
        / max(min_exit1_fraction, 1e-9),
    ]
    return float(np.sum(terms))


def evaluate_sequential_candidate(
    *,
    strategy: str,
    parameters: dict[str, Any],
    selected_probabilities: np.ndarray,
    selected_exit: np.ndarray,
    y_true: np.ndarray,
    thresholds_by_exit: Sequence[np.ndarray],
    parent_context: ParentMetricContext,
    flops_by_exit: dict[str, float],
    max_macro_drop: float,
    max_micro_drop: float,
    max_exact_drop: float,
    max_hamming_increase: float,
    min_total_early_fraction: float,
    min_exit1_fraction: float,
) -> dict[str, Any]:
    probabilities = np.asarray(selected_probabilities, dtype=np.float32)
    exits = np.asarray(selected_exit, dtype=np.int16).reshape(-1)
    truth = np.asarray(y_true, dtype=np.int8)
    if probabilities.shape != truth.shape or len(exits) != len(truth):
        raise ValueError("Sequential candidate arrays are not aligned.")
    num_exits = len(thresholds_by_exit)
    if set(np.unique(exits).tolist()) - set(range(1, num_exits + 1)):
        raise ValueError("selected_exit contains an unavailable exit number.")

    predictions = np.zeros_like(truth, dtype=np.int8)
    exit_counts: dict[int, int] = {}
    for exit_no in range(1, num_exits + 1):
        mask = exits == exit_no
        exit_counts[exit_no] = int(mask.sum())
        if np.any(mask):
            thresholds = np.asarray(thresholds_by_exit[exit_no - 1], dtype=np.float32)
            predictions[mask] = (
                probabilities[mask] >= thresholds.reshape(1, -1)
            ).astype(np.int8)

    segment_metrics = multilabel_metrics(truth, predictions)
    parent_predictions = parent_context.predictions(probabilities)
    parent_metrics = parent_context.metrics_from_predictions(parent_predictions)
    fold_metrics = parent_context.fold_metrics_from_predictions(parent_predictions)

    fold_macro: list[float] = []
    fold_micro: list[float] = []
    fold_exact: list[float] = []
    fold_hamming: list[float] = []
    for reference, candidate in zip(
        parent_context.reference_fold_metrics, fold_metrics, strict=True
    ):
        fold_macro.append(float(reference["macro_f1"] - candidate["macro_f1"]))
        fold_micro.append(float(reference["micro_f1"] - candidate["micro_f1"]))
        fold_exact.append(float(reference["exact_match"] - candidate["exact_match"]))
        fold_hamming.append(
            float(candidate["hamming_loss"] - reference["hamming_loss"])
        )
    macro_stats = robust_upper_bound(fold_macro)
    micro_stats = robust_upper_bound(fold_micro)
    exact_stats = robust_upper_bound(fold_exact)
    hamming_stats = robust_upper_bound(fold_hamming)

    reference = parent_context.reference_metrics
    macro_drop = float(reference["macro_f1"] - parent_metrics["macro_f1"])
    micro_drop = float(reference["micro_f1"] - parent_metrics["micro_f1"])
    exact_drop = float(reference["exact_match"] - parent_metrics["exact_match"])
    hamming_increase = float(
        parent_metrics["hamming_loss"] - reference["hamming_loss"]
    )

    fractions = {
        exit_no: float(exit_counts[exit_no] / max(len(exits), 1))
        for exit_no in range(1, num_exits + 1)
    }
    total_early_fraction = float(1.0 - fractions[num_exits])
    exit1_fraction = float(fractions[1])
    average_exit_depth = float(np.mean(exits))
    average_flops = float(
        sum(fractions[e] * float(flops_by_exit[f"exit{e}"]) for e in fractions)
    )
    final_flops = float(flops_by_exit[f"exit{num_exits}"])
    flops_saved = float(100.0 * (1.0 - average_flops / max(final_flops, 1.0)))

    macro_upper = max(macro_drop, macro_stats["upper"])
    micro_upper = max(micro_drop, micro_stats["upper"])
    exact_upper = max(exact_drop, exact_stats["upper"])
    hamming_upper = max(hamming_increase, hamming_stats["upper"])
    violation = sequential_constraint_violation(
        macro_upper=macro_upper,
        micro_upper=micro_upper,
        exact_upper=exact_upper,
        hamming_upper=hamming_upper,
        total_early_fraction=total_early_fraction,
        exit1_fraction=exit1_fraction,
        max_macro_drop=max_macro_drop,
        max_micro_drop=max_micro_drop,
        max_exact_drop=max_exact_drop,
        max_hamming_increase=max_hamming_increase,
        min_total_early_fraction=min_total_early_fraction,
        min_exit1_fraction=min_exit1_fraction,
    )

    row: dict[str, Any] = {
        "strategy": strategy,
        "parameters_json": json.dumps(jsonable(parameters), sort_keys=True),
        "num_exits": int(num_exits),
        "total_early_fraction": total_early_fraction,
        "exit1_fraction": exit1_fraction,
        "average_exit_depth": average_exit_depth,
        "estimated_flops_saved_pct": flops_saved,
        "parent_macro_f1": float(parent_metrics["macro_f1"]),
        "parent_micro_f1": float(parent_metrics["micro_f1"]),
        "parent_samples_f1": float(parent_metrics["samples_f1"]),
        "parent_exact_match": float(parent_metrics["exact_match"]),
        "parent_hamming_loss": float(parent_metrics["hamming_loss"]),
        "macro_f1_drop": macro_drop,
        "micro_f1_drop": micro_drop,
        "exact_match_drop": exact_drop,
        "hamming_loss_increase": hamming_increase,
        "fold_macro_drop_upper": float(macro_stats["upper"]),
        "fold_micro_drop_upper": float(micro_stats["upper"]),
        "fold_exact_drop_upper": float(exact_stats["upper"]),
        "fold_hamming_increase_upper": float(hamming_stats["upper"]),
        "robust_macro_drop": macro_upper,
        "robust_micro_drop": micro_upper,
        "robust_exact_drop": exact_upper,
        "robust_hamming_increase": hamming_upper,
        "constraint_violation": violation,
        "quality_constraints_met": bool(violation <= 1e-12),
        "objective_compute": -flops_saved,
        "objective_macro": macro_upper,
        "objective_micro": micro_upper,
        "objective_exact": exact_upper,
        "objective_hamming": hamming_upper,
        **{f"segment_{key}": value for key, value in segment_metrics.items()},
    }
    for exit_no in range(1, num_exits + 1):
        row[f"exit{exit_no}_samples"] = exit_counts[exit_no]
        row[f"exit{exit_no}_fraction"] = fractions[exit_no]
    return row


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = [
        "objective_compute",
        "objective_macro",
        "objective_micro",
        "objective_exact",
        "objective_hamming",
    ]
    return frame[columns].to_numpy(dtype=np.float64)


def validate_fair_comparison(
    *,
    labels_3: Sequence[str],
    labels_5: Sequence[str],
    manifest_3: Path,
    manifest_5: Path,
    features_root_3: Path,
    features_root_5: Path,
) -> dict[str, Any]:
    same_labels = list(labels_3) == list(labels_5)
    same_manifest = manifest_3.resolve() == manifest_5.resolve()
    same_features = features_root_3.resolve() == features_root_5.resolve()
    valid = bool(same_labels and same_manifest and same_features)
    return {
        "fair_comparison_valid": valid,
        "same_label_order": same_labels,
        "same_manifest": same_manifest,
        "same_features_root": same_features,
        "requirement": (
            "3-exit and 5-exit checkpoints must be evaluated on the identical "
            "manifest, feature cache, label order, LATS configuration, threshold "
            "mode, optimisation budget, constraints and timing protocol."
        ),
    }
