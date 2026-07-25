from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
V013_COMMON_DIR = PROJECT_ROOT / "scripts" / "v0.13_EE" / "matched_policy_comparison"
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
    load_run_config,
    load_thresholds_by_exit,
    multilabel_metrics,
    parse_float_list,
    parse_tap_blocks,
    resolve_model_cfg,
    save_json,
    synchronize,
    threshold_mapping,
)
from policies.parent_aware_adaptive_gate import (  # noqa: E402
    LATSLabelRule,
    parse_lats_rules,
)


def robust_upper_bound(values: Sequence[float], one_sided_z: float = 1.645) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        raise ValueError("At least one value is required.")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    standard_error = float(std / np.sqrt(len(array)))
    return {
        "mean": mean,
        "std": std,
        "standard_error": standard_error,
        "upper": float(mean + float(one_sided_z) * standard_error),
        "maximum": float(np.max(array)),
    }


def constraint_violation(
    *,
    macro_drop_upper: float,
    micro_drop_upper: float,
    exact_drop_upper: float,
    hamming_increase_upper: float,
    exit2_fraction: float,
    max_macro_drop: float,
    max_micro_drop: float,
    max_exact_drop: float,
    max_hamming_increase: float,
    min_exit2_fraction: float,
) -> float:
    components = [
        max(0.0, float(macro_drop_upper) - float(max_macro_drop))
        / max(float(max_macro_drop), 1e-9),
        max(0.0, float(micro_drop_upper) - float(max_micro_drop))
        / max(float(max_micro_drop), 1e-9),
        max(0.0, float(exact_drop_upper) - float(max_exact_drop))
        / max(float(max_exact_drop), 1e-9),
        max(0.0, float(hamming_increase_upper) - float(max_hamming_increase))
        / max(float(max_hamming_increase), 1e-9),
        max(0.0, float(min_exit2_fraction) - float(exit2_fraction))
        / max(float(min_exit2_fraction), 1e-9),
    ]
    return float(np.sum(components))


@dataclass
class ParentMetricContext:
    labels: list[str]
    rules: tuple[LATSLabelRule, ...]
    parent_ids: np.ndarray
    parent_truth: np.ndarray
    row_to_parent: np.ndarray
    parent_rows: np.ndarray
    parent_sizes: np.ndarray
    fold_index: np.ndarray
    reference_metrics: dict[str, float]
    reference_fold_metrics: list[dict[str, float]]

    @classmethod
    def build(
        cls,
        *,
        metadata_df: pd.DataFrame,
        labels: list[str],
        parent_id_col: str,
        lats_config_json: Path,
        reference_probabilities: np.ndarray,
        cv_folds: int,
    ) -> "ParentMetricContext":
        frame = metadata_df.reset_index(drop=True)
        ids = frame[parent_id_col].astype(str).to_numpy()
        unique_ids, row_to_parent = np.unique(ids, return_inverse=True)
        num_parents = len(unique_ids)
        sizes = np.bincount(row_to_parent, minlength=num_parents).astype(np.int64)
        max_size = int(sizes.max())
        parent_rows = np.full((num_parents, max_size), -1, dtype=np.int64)
        offsets = np.zeros(num_parents, dtype=np.int64)
        for row_idx, parent_idx in enumerate(row_to_parent):
            parent_rows[parent_idx, offsets[parent_idx]] = row_idx
            offsets[parent_idx] += 1

        truth_segments = frame[labels].astype(int).to_numpy(dtype=np.int8)
        parent_truth = np.zeros((num_parents, len(labels)), dtype=np.int8)
        for parent_idx in range(num_parents):
            rows = parent_rows[parent_idx, : sizes[parent_idx]]
            values = truth_segments[rows]
            if not np.all(values == values[0].reshape(1, -1)):
                raise ValueError(
                    f"Ground-truth labels differ within parent {unique_ids[parent_idx]!r}."
                )
            parent_truth[parent_idx] = values[0]

        rules = parse_lats_rules(load_json(lats_config_json), labels)
        folds = min(max(2, int(cv_folds)), num_parents)
        splitter = GroupKFold(n_splits=folds)
        fold_index = np.full(num_parents, -1, dtype=np.int16)
        dummy = np.zeros((num_parents, 1), dtype=np.float32)
        for fold_no, (_, valid_idx) in enumerate(
            splitter.split(dummy, parent_truth, groups=unique_ids),
            start=1,
        ):
            fold_index[valid_idx] = fold_no
        if np.any(fold_index < 0):
            raise RuntimeError("Parent fold assignment left unassigned parents.")

        placeholder = cls(
            labels=labels,
            rules=rules,
            parent_ids=unique_ids,
            parent_truth=parent_truth,
            row_to_parent=row_to_parent.astype(np.int64),
            parent_rows=parent_rows,
            parent_sizes=sizes,
            fold_index=fold_index,
            reference_metrics={},
            reference_fold_metrics=[],
        )
        reference_metrics = placeholder.metrics(reference_probabilities)
        reference_predictions = placeholder.predictions(reference_probabilities)
        reference_fold_metrics = []
        for fold_no in sorted(np.unique(fold_index).tolist()):
            mask = fold_index == fold_no
            reference_fold_metrics.append(
                multilabel_metrics(parent_truth[mask], reference_predictions[mask])
            )
        placeholder.reference_metrics = {
            key: float(value) for key, value in reference_metrics.items()
        }
        placeholder.reference_fold_metrics = [
            {key: float(value) for key, value in item.items()}
            for item in reference_fold_metrics
        ]
        return placeholder

    def aggregate(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] != len(self.labels):
            raise ValueError("Probability matrix has unexpected shape.")
        padded = np.full(
            (len(self.parent_ids), self.parent_rows.shape[1], len(self.labels)),
            np.nan,
            dtype=np.float32,
        )
        for parent_idx in range(len(self.parent_ids)):
            rows = self.parent_rows[parent_idx, : self.parent_sizes[parent_idx]]
            padded[parent_idx, : len(rows)] = probs[rows]

        scores = np.zeros((len(self.parent_ids), len(self.labels)), dtype=np.float32)
        for label_idx, rule in enumerate(self.rules):
            values = padded[:, :, label_idx]
            method = rule.aggregation
            if method == "mean":
                scores[:, label_idx] = np.nanmean(values, axis=1)
            elif method == "max":
                scores[:, label_idx] = np.nanmax(values, axis=1)
            elif method == "p75":
                scores[:, label_idx] = np.nanquantile(values, 0.75, axis=1)
            elif method.startswith("top") and method.endswith("mean"):
                digits = method[3:-4]
                if not digits.isdigit():
                    raise ValueError(f"Unsupported top-k aggregation: {method}")
                k_requested = int(digits)
                for parent_idx in range(len(self.parent_ids)):
                    count = int(self.parent_sizes[parent_idx])
                    k = max(1, min(k_requested, count))
                    row = values[parent_idx, :count]
                    scores[parent_idx, label_idx] = float(
                        np.mean(np.partition(row, count - k)[count - k :])
                    )
            elif method == "noisy_or":
                neutral = np.where(np.isnan(values), 0.0, values)
                scores[:, label_idx] = 1.0 - np.prod(1.0 - neutral, axis=1)
            else:
                raise ValueError(f"Unsupported LATS aggregation: {method}")
        return scores

    def predictions(self, probabilities: np.ndarray) -> np.ndarray:
        scores = self.aggregate(probabilities)
        thresholds = np.asarray(
            [rule.threshold for rule in self.rules], dtype=np.float32
        ).reshape(1, -1)
        return (scores >= thresholds).astype(np.int8)

    def metrics_from_predictions(self, predictions: np.ndarray) -> dict[str, float]:
        pred = np.asarray(predictions, dtype=np.int8)
        if pred.shape != self.parent_truth.shape:
            raise ValueError("Parent prediction matrix has unexpected shape.")
        return {
            key: float(value)
            for key, value in multilabel_metrics(self.parent_truth, pred).items()
        }

    def fold_metrics_from_predictions(
        self, predictions: np.ndarray
    ) -> list[dict[str, float]]:
        pred = np.asarray(predictions, dtype=np.int8)
        if pred.shape != self.parent_truth.shape:
            raise ValueError("Parent prediction matrix has unexpected shape.")
        result: list[dict[str, float]] = []
        for fold_no in sorted(np.unique(self.fold_index).tolist()):
            mask = self.fold_index == fold_no
            result.append(
                {
                    key: float(value)
                    for key, value in multilabel_metrics(
                        self.parent_truth[mask], pred[mask]
                    ).items()
                }
            )
        return result

    def metrics(self, probabilities: np.ndarray) -> dict[str, float]:
        return self.metrics_from_predictions(self.predictions(probabilities))

    def fold_metrics(self, probabilities: np.ndarray) -> list[dict[str, float]]:
        return self.fold_metrics_from_predictions(self.predictions(probabilities))


def evaluate_margin_candidate(
    *,
    strategy: str,
    parameters: dict[str, Any],
    stop_mask: np.ndarray,
    y_true: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit2_predictions: np.ndarray,
    exit3_predictions: np.ndarray,
    parent_context: ParentMetricContext,
    exit2_flops: float,
    exit3_flops: float,
    max_macro_drop: float,
    max_micro_drop: float,
    max_exact_drop: float,
    max_hamming_increase: float,
    min_exit2_fraction: float,
) -> dict[str, Any]:
    mask = np.asarray(stop_mask, dtype=bool).reshape(-1)
    if len(mask) != len(y_true):
        raise ValueError("stop_mask length does not match candidate data.")
    selected_probabilities = np.where(
        mask.reshape(-1, 1), exit2_probabilities, exit3_probabilities
    )
    selected_predictions = np.where(
        mask.reshape(-1, 1), exit2_predictions, exit3_predictions
    )
    segment_metrics = multilabel_metrics(y_true, selected_predictions)
    parent_predictions = parent_context.predictions(selected_probabilities)
    parent_metrics = parent_context.metrics_from_predictions(parent_predictions)
    fold_metrics = parent_context.fold_metrics_from_predictions(parent_predictions)

    fold_macro_drops: list[float] = []
    fold_micro_drops: list[float] = []
    fold_exact_drops: list[float] = []
    fold_hamming_increases: list[float] = []
    for reference, candidate in zip(
        parent_context.reference_fold_metrics, fold_metrics, strict=True
    ):
        fold_macro_drops.append(
            float(reference["macro_f1"] - candidate["macro_f1"])
        )
        fold_micro_drops.append(
            float(reference["micro_f1"] - candidate["micro_f1"])
        )
        fold_exact_drops.append(
            float(reference["exact_match"] - candidate["exact_match"])
        )
        fold_hamming_increases.append(
            float(candidate["hamming_loss"] - reference["hamming_loss"])
        )

    macro_stats = robust_upper_bound(fold_macro_drops)
    micro_stats = robust_upper_bound(fold_micro_drops)
    exact_stats = robust_upper_bound(fold_exact_drops)
    hamming_stats = robust_upper_bound(fold_hamming_increases)

    reference = parent_context.reference_metrics
    macro_drop = float(reference["macro_f1"] - parent_metrics["macro_f1"])
    micro_drop = float(reference["micro_f1"] - parent_metrics["micro_f1"])
    exact_drop = float(reference["exact_match"] - parent_metrics["exact_match"])
    hamming_increase = float(
        parent_metrics["hamming_loss"] - reference["hamming_loss"]
    )
    exit2_count = int(mask.sum())
    exit2_fraction = float(exit2_count / max(len(mask), 1))
    average_flops = float(
        exit2_fraction * float(exit2_flops)
        + (1.0 - exit2_fraction) * float(exit3_flops)
    )
    flops_saved = float(
        100.0 * (1.0 - average_flops / max(float(exit3_flops), 1.0))
    )

    violation = constraint_violation(
        macro_drop_upper=max(macro_drop, macro_stats["upper"]),
        micro_drop_upper=max(micro_drop, micro_stats["upper"]),
        exact_drop_upper=max(exact_drop, exact_stats["upper"]),
        hamming_increase_upper=max(hamming_increase, hamming_stats["upper"]),
        exit2_fraction=exit2_fraction,
        max_macro_drop=max_macro_drop,
        max_micro_drop=max_micro_drop,
        max_exact_drop=max_exact_drop,
        max_hamming_increase=max_hamming_increase,
        min_exit2_fraction=min_exit2_fraction,
    )
    feasible = violation <= 1e-12
    objectives = np.asarray(
        [
            -flops_saved,
            max(macro_drop, macro_stats["upper"]),
            max(micro_drop, micro_stats["upper"]),
            max(exact_drop, exact_stats["upper"]),
            max(hamming_increase, hamming_stats["upper"]),
        ],
        dtype=np.float64,
    )

    return {
        "strategy": strategy,
        "parameters_json": json.dumps(jsonable(parameters), sort_keys=True),
        "exit2_samples": exit2_count,
        "exit3_samples": int(len(mask) - exit2_count),
        "exit2_fraction": exit2_fraction,
        "average_exit_depth": float(2.0 * exit2_fraction + 3.0 * (1.0 - exit2_fraction)),
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
        "fold_macro_drop_mean": macro_stats["mean"],
        "fold_macro_drop_upper": macro_stats["upper"],
        "fold_micro_drop_mean": micro_stats["mean"],
        "fold_micro_drop_upper": micro_stats["upper"],
        "fold_exact_drop_mean": exact_stats["mean"],
        "fold_exact_drop_upper": exact_stats["upper"],
        "fold_hamming_increase_mean": hamming_stats["mean"],
        "fold_hamming_increase_upper": hamming_stats["upper"],
        "constraint_violation": violation,
        "quality_constraints_met": feasible,
        "objective_compute": float(objectives[0]),
        "objective_macro": float(objectives[1]),
        "objective_micro": float(objectives[2]),
        "objective_exact": float(objectives[3]),
        "objective_hamming": float(objectives[4]),
        **{f"segment_{key}": value for key, value in segment_metrics.items()},
    }


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[
        [
            "objective_compute",
            "objective_macro",
            "objective_micro",
            "objective_exact",
            "objective_hamming",
        ]
    ].to_numpy(dtype=np.float64)
