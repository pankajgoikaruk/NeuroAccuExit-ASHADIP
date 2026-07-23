from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
V014_COMMON_DIR = PROJECT_ROOT / "scripts" / "v0.14_EE" / "parent_aware_gate"
for path in (PROJECT_ROOT, V014_COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v014 import (  # noqa: E402,F401
    collect_outputs,
    jsonable,
    load_checkpoint,
    load_feature,
    load_json,
    load_labels,
    load_run_config,
    load_thresholds_by_exit,
    multilabel_metrics,
    parse_float_list,
    parse_tap_blocks,
    resolve_model_cfg,
    robust_drop_statistics,
    save_json,
    synchronize,
    threshold_mapping,
)
from policies.parent_aware_adaptive_gate import ConstantProbabilityModel  # noqa: E402
from policies.whole_parent_selective_exit import (  # noqa: E402
    expand_parent_label_features,
    fit_empirical_risk_calibrators,
    predict_empirical_unsafe_probabilities,
    predict_shared_unsafe_probabilities,
    wilson_upper_bound,
)


def parse_optional_float_list(value: str) -> list[float | None]:
    result: list[float | None] = []
    for item in str(value).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "off"}:
            result.append(None)
        else:
            result.append(float(token))
    if not result:
        raise ValueError("Expected at least one optional float value.")
    unique: list[float | None] = []
    for item in result:
        if item not in unique:
            unique.append(item)
    return unique


def fit_shared_parent_gate(
    *,
    parent_features: np.ndarray,
    unsafe_targets: np.ndarray,
    seed: int,
) -> Any:
    features = np.asarray(parent_features, dtype=np.float32)
    targets = np.asarray(unsafe_targets, dtype=np.int8)
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("Parent features and targets must be aligned matrices.")
    expanded, _, _ = expand_parent_label_features(features, targets.shape[1])
    flat_targets = targets.reshape(-1)
    unique = np.unique(flat_targets)
    if len(unique) == 1:
        return ConstantProbabilityModel(float(unique[0]))
    model = Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    model.fit(expanded, flat_targets)
    return model


def parent_oof_probabilities(
    *,
    parent_features: np.ndarray,
    raw_nonparametric_risk: np.ndarray,
    unsafe_targets: np.ndarray,
    n_splits: int,
    seed: int,
    empirical_bins: int,
    minimum_positive_examples: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    features = np.asarray(parent_features, dtype=np.float32)
    raw = np.asarray(raw_nonparametric_risk, dtype=np.float32)
    targets = np.asarray(unsafe_targets, dtype=np.int8)
    if features.ndim != 2 or raw.shape != targets.shape:
        raise ValueError("OOF parent features, risks and targets are misaligned.")
    folds = min(int(n_splits), len(features))
    if folds < 2:
        raise ValueError("At least two parents are required for OOF prediction.")

    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
    shared_oof = np.zeros_like(targets, dtype=np.float32)
    empirical_oof = np.zeros_like(targets, dtype=np.float32)
    fold_index = np.full(len(features), -1, dtype=np.int16)
    records: list[dict[str, Any]] = []

    for fold_no, (train_idx, valid_idx) in enumerate(splitter.split(features), start=1):
        shared_model = fit_shared_parent_gate(
            parent_features=features[train_idx],
            unsafe_targets=targets[train_idx],
            seed=int(seed) + fold_no * 100,
        )
        shared_oof[valid_idx] = predict_shared_unsafe_probabilities(
            shared_model,
            features[valid_idx],
            targets.shape[1],
        )
        calibrators, positive_counts, used_pooled = fit_empirical_risk_calibrators(
            raw_scores=raw[train_idx],
            unsafe_targets=targets[train_idx],
            num_bins=int(empirical_bins),
            minimum_positive_examples=int(minimum_positive_examples),
        )
        empirical_oof[valid_idx] = predict_empirical_unsafe_probabilities(
            calibrators,
            raw[valid_idx],
        )
        fold_index[valid_idx] = fold_no
        records.append(
            {
                "fold": fold_no,
                "train_parents": int(len(train_idx)),
                "validation_parents": int(len(valid_idx)),
                "training_unsafe_label_events": int(targets[train_idx].sum()),
                "validation_unsafe_label_events": int(targets[valid_idx].sum()),
                "empirical_positive_counts": positive_counts.tolist(),
                "empirical_used_pooled": used_pooled.tolist(),
            }
        )
    if np.any(fold_index < 0):
        raise RuntimeError("OOF prediction left unassigned parents.")
    return (
        {
            "nonparametric_parent_risk": empirical_oof,
            "shared_logistic_parent_gate": shared_oof,
        },
        fold_index,
        records,
    )


def whole_parent_candidate_result(
    *,
    strategy: str,
    parameters: dict[str, Any],
    stop_parent_mask: np.ndarray,
    parent_truth: np.ndarray,
    source_parent_predictions: np.ndarray,
    deeper_parent_predictions: np.ndarray,
    unsafe_targets: np.ndarray,
    row_to_parent: np.ndarray,
    y_true_segments: np.ndarray,
    source_segment_probabilities: np.ndarray,
    deeper_segment_probabilities: np.ndarray,
    source_segment_predictions: np.ndarray,
    deeper_segment_predictions: np.ndarray,
    source_flops: float,
    deeper_flops: float,
    reference_parent_metrics: dict[str, float],
    max_macro_f1_drop: float,
    max_micro_f1_drop: float,
    max_exact_match_drop: float,
    max_overall_harm_fraction: float,
    min_parent_stop_fraction: float,
) -> dict[str, Any]:
    stop = np.asarray(stop_parent_mask, dtype=bool).reshape(-1)
    truth = np.asarray(parent_truth, dtype=np.int8)
    source_parent = np.asarray(source_parent_predictions, dtype=np.int8)
    deeper_parent = np.asarray(deeper_parent_predictions, dtype=np.int8)
    unsafe = np.asarray(unsafe_targets, dtype=np.int8)
    if len(stop) != len(truth):
        raise ValueError("Parent stop mask length does not match parent truth.")

    selected_parent = np.where(
        stop.reshape(-1, 1), source_parent, deeper_parent
    )
    parent_metrics = multilabel_metrics(truth, selected_parent)

    row_stop = stop[np.asarray(row_to_parent, dtype=np.int64)]
    selected_segment_predictions = np.where(
        row_stop.reshape(-1, 1),
        source_segment_predictions,
        deeper_segment_predictions,
    )
    segment_metrics = multilabel_metrics(
        y_true_segments,
        selected_segment_predictions,
    )

    parent_stop_fraction = float(stop.mean())
    segment_stop_fraction = float(row_stop.mean())
    average_flops = float(
        segment_stop_fraction * float(source_flops)
        + (1.0 - segment_stop_fraction) * float(deeper_flops)
    )
    flops_saved_pct = float(
        100.0 * (1.0 - average_flops / max(float(deeper_flops), 1.0))
    )

    any_unsafe = np.any(unsafe == 1, axis=1)
    harmful_stopped = stop & any_unsafe
    harmful_count = int(harmful_stopped.sum())
    stopped_count = int(stop.sum())
    overall_harm_fraction = float(harmful_count / max(len(stop), 1))
    conditional_harm_rate = float(harmful_count / max(stopped_count, 1))
    conditional_harm_ucb = wilson_upper_bound(harmful_count, stopped_count)
    overall_harm_ucb = wilson_upper_bound(harmful_count, len(stop))

    macro_drop = float(
        reference_parent_metrics["macro_f1"] - parent_metrics["macro_f1"]
    )
    micro_drop = float(
        reference_parent_metrics["micro_f1"] - parent_metrics["micro_f1"]
    )
    exact_drop = float(
        reference_parent_metrics["exact_match"] - parent_metrics["exact_match"]
    )
    base_constraint = bool(
        macro_drop <= float(max_macro_f1_drop) + 1e-12
        and micro_drop <= float(max_micro_f1_drop) + 1e-12
        and exact_drop <= float(max_exact_match_drop) + 1e-12
        and overall_harm_fraction <= float(max_overall_harm_fraction) + 1e-12
        and parent_stop_fraction >= float(min_parent_stop_fraction)
    )

    return {
        "strategy": strategy,
        "parameters_json": json.dumps(jsonable(parameters), sort_keys=True),
        "parent_stop_count": stopped_count,
        "parent_continue_count": int(len(stop) - stopped_count),
        "parent_stop_fraction": parent_stop_fraction,
        "segment_stop_fraction": segment_stop_fraction,
        "average_exit_depth": float(
            2.0 * segment_stop_fraction + 3.0 * (1.0 - segment_stop_fraction)
        ),
        "estimated_flops_saved_pct": flops_saved_pct,
        "harmful_stopped_parents": harmful_count,
        "overall_harm_fraction": overall_harm_fraction,
        "conditional_harm_rate": conditional_harm_rate,
        "overall_harm_fraction_upper_confidence": overall_harm_ucb,
        "conditional_harm_rate_upper_confidence": conditional_harm_ucb,
        "parent_macro_f1": float(parent_metrics["macro_f1"]),
        "parent_micro_f1": float(parent_metrics["micro_f1"]),
        "parent_samples_f1": float(parent_metrics["samples_f1"]),
        "parent_exact_match": float(parent_metrics["exact_match"]),
        "parent_hamming_loss": float(parent_metrics["hamming_loss"]),
        "macro_f1_drop": macro_drop,
        "micro_f1_drop": micro_drop,
        "exact_match_drop": exact_drop,
        "base_risk_constraint_met": base_constraint,
        **{f"segment_{key}": value for key, value in segment_metrics.items()},
    }


def select_risk_controlled_candidate(
    strategy_df: pd.DataFrame,
) -> tuple[pd.Series, str]:
    feasible = strategy_df[
        strategy_df["robust_risk_constraint_met"] == True  # noqa: E712
    ].copy()
    if not feasible.empty:
        selected = feasible.sort_values(
            [
                "estimated_flops_saved_pct",
                "parent_macro_f1",
                "parent_micro_f1",
                "parent_exact_match",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        return selected, "robust_risk_constraint_met"
    selected = strategy_df.sort_values(
        [
            "fold_macro_f1_drop_upper_confidence",
            "overall_harm_fraction",
            "parent_macro_f1",
            "estimated_flops_saved_pct",
        ],
        ascending=[True, True, False, False],
    ).iloc[0]
    return selected, "fallback_best_risk_control"
