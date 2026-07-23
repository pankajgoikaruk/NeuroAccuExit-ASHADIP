from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
V013_COMMON_DIR = (
    PROJECT_ROOT / "scripts" / "v0.13_EE" / "matched_policy_comparison"
)
for path in (PROJECT_ROOT, V013_COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v013 import (  # noqa: E402,F401
    collect_outputs,
    jsonable,
    load_checkpoint,
    load_feature,
    load_json,
    load_labels,
    load_lats_module,
    load_run_config,
    load_thresholds_by_exit,
    multilabel_metrics,
    parent_level_metrics,
    parse_float_list,
    parse_tap_blocks,
    resolve_model_cfg,
    save_json,
    synchronize,
    threshold_mapping,
)
from policies.parent_aware_adaptive_gate import (  # noqa: E402
    ConstantProbabilityModel,
    predict_multilabel_unsafe_probabilities,
)


def fit_label_gate_models(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
) -> list[Any]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int8)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("features and targets must be aligned 2-D matrices.")

    models: list[Any] = []
    for label_idx in range(y.shape[1]):
        label_target = y[:, label_idx]
        unique = np.unique(label_target)
        if len(unique) == 1:
            models.append(ConstantProbabilityModel(float(unique[0])))
            continue
        model = Pipeline(
            steps=[
                ("standardize", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=int(seed) + label_idx,
                    ),
                ),
            ]
        )
        model.fit(x, label_target)
        models.append(model)
    return models


def grouped_oof_gate_probabilities(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[Any],
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int8)
    group_array = np.asarray([str(item) for item in groups], dtype=object)
    unique_groups = np.unique(group_array)
    folds = min(int(n_splits), len(unique_groups))
    if folds < 2:
        raise ValueError("At least two parent groups are required for OOF training.")

    splitter = GroupKFold(n_splits=folds)
    oof = np.zeros_like(y, dtype=np.float32)
    fold_index = np.full(len(x), -1, dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    for fold_no, (train_idx, valid_idx) in enumerate(
        splitter.split(x, y, groups=group_array),
        start=1,
    ):
        models = fit_label_gate_models(
            features=x[train_idx],
            targets=y[train_idx],
            seed=int(seed) + fold_no * 100,
        )
        oof[valid_idx] = predict_multilabel_unsafe_probabilities(
            models,
            x[valid_idx],
        )
        fold_index[valid_idx] = fold_no
        fold_records.append(
            {
                "fold": fold_no,
                "train_segments": int(len(train_idx)),
                "validation_segments": int(len(valid_idx)),
                "train_parents": int(len(np.unique(group_array[train_idx]))),
                "validation_parents": int(len(np.unique(group_array[valid_idx]))),
            }
        )
    if np.any(fold_index < 0):
        raise RuntimeError("OOF generation left unassigned samples.")
    return oof, fold_index, fold_records


def adaptive_candidate_result(
    *,
    strategy: str,
    parameters: dict[str, Any],
    stop_mask: np.ndarray,
    y_true: np.ndarray,
    source_probabilities: np.ndarray,
    deeper_probabilities: np.ndarray,
    source_predictions: np.ndarray,
    deeper_predictions: np.ndarray,
    metadata_df: pd.DataFrame,
    labels: list[str],
    lats_config_json: Path | None,
    parent_id_col: str,
    lats_module,
    reference_macro_f1: float,
    max_macro_f1_drop: float,
    min_source_fraction: float,
    source_exit_no: int,
    source_flops: float,
    deeper_flops: float,
) -> dict[str, Any]:
    mask = np.asarray(stop_mask, dtype=bool).reshape(-1)
    selected_probabilities = np.where(
        mask.reshape(-1, 1), source_probabilities, deeper_probabilities
    )
    selected_predictions = np.where(
        mask.reshape(-1, 1), source_predictions, deeper_predictions
    )
    segment_metrics = multilabel_metrics(y_true, selected_predictions)
    parent_metrics = None
    if lats_module is not None and lats_config_json is not None:
        parent_metrics = parent_level_metrics(
            metadata_df=metadata_df,
            labels=labels,
            probabilities=selected_probabilities,
            lats_config_json=lats_config_json,
            parent_id_col=parent_id_col,
            lats_module=lats_module,
        )
    selection_macro_f1 = (
        float(parent_metrics["macro_f1"])
        if parent_metrics is not None
        else float(segment_metrics["macro_f1"])
    )
    source_count = int(mask.sum())
    source_fraction = float(source_count / max(len(mask), 1))
    average_exit_depth = float(
        float(source_exit_no) * source_fraction
        + 3.0 * (1.0 - source_fraction)
    )
    average_flops = float(
        source_fraction * float(source_flops)
        + (1.0 - source_fraction) * float(deeper_flops)
    )
    flops_saved_pct = float(
        100.0 * (1.0 - average_flops / max(float(deeper_flops), 1.0))
    )
    macro_f1_drop = float(reference_macro_f1 - selection_macro_f1)
    row: dict[str, Any] = {
        "strategy": strategy,
        "parameters_json": json.dumps(jsonable(parameters), sort_keys=True),
        "source_exit": int(source_exit_no),
        "source_exit_samples": source_count,
        "deeper_exit_samples": int(len(mask) - source_count),
        "source_exit_fraction": source_fraction,
        "avg_exit_depth": average_exit_depth,
        "estimated_flops_saved_pct": flops_saved_pct,
        "selection_macro_f1": selection_macro_f1,
        "reference_deeper_macro_f1": float(reference_macro_f1),
        "macro_f1_drop": macro_f1_drop,
        "base_quality_constraint_met": bool(
            macro_f1_drop <= float(max_macro_f1_drop) + 1e-12
            and source_fraction >= float(min_source_fraction)
        ),
        **{f"segment_{key}": value for key, value in segment_metrics.items()},
    }
    if parent_metrics is not None:
        row.update({f"parent_{key}": value for key, value in parent_metrics.items()})
    return row


def robust_drop_statistics(
    drops: Sequence[float],
    *,
    one_sided_z: float = 1.645,
) -> dict[str, float]:
    values = np.asarray(list(drops), dtype=np.float64)
    if len(values) == 0:
        raise ValueError("At least one fold drop is required.")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    standard_error = float(std / np.sqrt(len(values)))
    upper = float(mean + float(one_sided_z) * standard_error)
    return {
        "fold_macro_f1_drop_mean": mean,
        "fold_macro_f1_drop_std": std,
        "fold_macro_f1_drop_standard_error": standard_error,
        "fold_macro_f1_drop_upper_confidence": upper,
        "fold_macro_f1_drop_max": float(np.max(values)),
    }


def select_robust_candidate(
    strategy_df: pd.DataFrame,
) -> tuple[pd.Series, str]:
    feasible = strategy_df[
        strategy_df["robust_quality_constraint_met"] == True  # noqa: E712
    ].copy()
    if not feasible.empty:
        selected = feasible.sort_values(
            [
                "estimated_flops_saved_pct",
                "selection_macro_f1",
                "parent_micro_f1",
            ],
            ascending=[False, False, False],
        ).iloc[0]
        return selected, "robust_quality_constraint_met"
    selected = strategy_df.sort_values(
        [
            "fold_macro_f1_drop_upper_confidence",
            "selection_macro_f1",
            "estimated_flops_saved_pct",
        ],
        ascending=[True, False, False],
    ).iloc[0]
    return selected, "fallback_best_robust_quality"
