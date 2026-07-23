from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
V012_COMMON_DIR = (
    PROJECT_ROOT / "scripts" / "v0.12_EE" / "label_aware_policy"
)
for path in (PROJECT_ROOT, V012_COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v012 import (  # noqa: E402,F401
    collect_outputs,
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


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def candidate_result(
    *,
    strategy: str,
    parameters: dict[str, Any],
    stop_mask: np.ndarray,
    y_true: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit2_predictions: np.ndarray,
    exit3_predictions: np.ndarray,
    metadata_df: pd.DataFrame,
    labels: list[str],
    lats_config_json: Path | None,
    parent_id_col: str,
    lats_module,
    reference_macro_f1: float,
    max_macro_f1_drop: float,
    min_exit2_fraction: float,
    exit2_flops: float,
    exit3_flops: float,
) -> dict[str, Any]:
    mask = np.asarray(stop_mask, dtype=bool).reshape(-1)
    if len(mask) != len(y_true):
        raise ValueError("stop_mask length does not match candidate data.")

    selected_probabilities = np.where(
        mask.reshape(-1, 1),
        exit2_probabilities,
        exit3_probabilities,
    )
    selected_predictions = np.where(
        mask.reshape(-1, 1),
        exit2_predictions,
        exit3_predictions,
    )
    segment_metrics = multilabel_metrics(
        y_true,
        selected_predictions,
    )

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
    exit2_count = int(mask.sum())
    exit2_fraction = float(exit2_count / max(len(mask), 1))
    avg_exit_depth = float(
        2.0 * exit2_fraction
        + 3.0 * (1.0 - exit2_fraction)
    )
    average_flops = float(
        exit2_fraction * exit2_flops
        + (1.0 - exit2_fraction) * exit3_flops
    )
    flops_saved_pct = float(
        100.0
        * (1.0 - average_flops / max(exit3_flops, 1.0))
    )
    macro_f1_drop = float(
        reference_macro_f1 - selection_macro_f1
    )
    constraint_met = bool(
        macro_f1_drop <= float(max_macro_f1_drop) + 1e-12
        and exit2_fraction >= float(min_exit2_fraction)
    )

    row: dict[str, Any] = {
        "strategy": strategy,
        "parameters_json": json.dumps(
            jsonable(parameters),
            sort_keys=True,
        ),
        "exit2_samples": exit2_count,
        "exit3_samples": int(len(mask) - exit2_count),
        "exit2_fraction": exit2_fraction,
        "avg_exit_depth": avg_exit_depth,
        "estimated_flops_saved_pct": flops_saved_pct,
        "selection_macro_f1": selection_macro_f1,
        "reference_exit3_macro_f1": float(reference_macro_f1),
        "macro_f1_drop": macro_f1_drop,
        "quality_constraint_met": constraint_met,
        **{
            f"segment_{key}": value
            for key, value in segment_metrics.items()
        },
    }
    if parent_metrics is not None:
        row.update(
            {
                f"parent_{key}": value
                for key, value in parent_metrics.items()
            }
        )
    return row


def select_candidate(
    strategy_df: pd.DataFrame,
) -> tuple[pd.Series, str]:
    feasible = strategy_df[
        strategy_df["quality_constraint_met"] == True  # noqa: E712
    ].copy()
    if not feasible.empty:
        selected = feasible.sort_values(
            [
                "estimated_flops_saved_pct",
                "selection_macro_f1",
            ],
            ascending=[False, False],
        ).iloc[0]
        return selected, "quality_constraint_met"

    selected = strategy_df.sort_values(
        ["selection_macro_f1", "estimated_flops_saved_pct"],
        ascending=[False, False],
    ).iloc[0]
    return (
        selected,
        "fallback_best_quality_constraint_not_met",
    )
