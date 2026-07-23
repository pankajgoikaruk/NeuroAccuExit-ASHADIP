# policies/whole_parent_selective_exit.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from policies.parent_aware_adaptive_gate import (
    LATSLabelRule,
    aggregate_parent_probabilities,
    parent_predictions,
)


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, labels], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return array


def _vector(
    value: np.ndarray | Sequence[float],
    size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def parent_truth_matrix(
    y_true: np.ndarray,
    parent_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int8)
    if truth.ndim != 2:
        raise ValueError("y_true must have shape [segments, labels].")
    ids = np.asarray([str(item) for item in parent_ids], dtype=object)
    if len(ids) != len(truth):
        raise ValueError("parent_ids length does not match y_true rows.")

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    parent_truth = np.zeros((len(unique_ids), truth.shape[1]), dtype=np.int8)
    for parent_index, parent_id in enumerate(unique_ids):
        rows = np.flatnonzero(inverse == parent_index)
        first = truth[rows[0]]
        if not np.all(truth[rows] == first.reshape(1, -1)):
            raise ValueError(
                f"Ground-truth labels differ within parent {parent_id!r}."
            )
        parent_truth[parent_index] = first
    return unique_ids, parent_truth, inverse.astype(np.int64)


def _parent_segment_statistics(
    probabilities: np.ndarray,
    inverse: np.ndarray,
    parent_count: int,
) -> dict[str, np.ndarray]:
    probs = _matrix(probabilities, "probabilities")
    mean = np.zeros((parent_count, probs.shape[1]), dtype=np.float32)
    maximum = np.zeros_like(mean)
    minimum = np.zeros_like(mean)
    std = np.zeros_like(mean)
    positive_fraction = np.zeros_like(mean)
    parent_size = np.zeros(parent_count, dtype=np.float32)

    for parent_index in range(parent_count):
        rows = probs[inverse == parent_index]
        mean[parent_index] = rows.mean(axis=0)
        maximum[parent_index] = rows.max(axis=0)
        minimum[parent_index] = rows.min(axis=0)
        std[parent_index] = rows.std(axis=0)
        positive_fraction[parent_index] = (rows >= 0.5).mean(axis=0)
        parent_size[parent_index] = float(len(rows))

    return {
        "mean": mean,
        "max": maximum,
        "min": minimum,
        "std": std,
        "range": maximum - minimum,
        "positive_fraction": positive_fraction,
        "parent_size": parent_size,
    }


def build_whole_parent_features(
    *,
    current_probabilities: np.ndarray,
    previous_probabilities: np.ndarray,
    parent_ids: Sequence[Any],
    rules: Sequence[LATSLabelRule],
    margin_scale: float = 0.25,
    delta_scale: float = 0.50,
    dispersion_scale: float = 0.25,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Create one feature row for each complete parent.

    All features are available after Exit 2 and before Blocks 4-5 execute.
    """

    current = _matrix(current_probabilities, "current_probabilities")
    previous = _matrix(previous_probabilities, "previous_probabilities")
    if current.shape != previous.shape:
        raise ValueError("Current and previous probabilities must share shape.")
    if len(rules) != current.shape[1]:
        raise ValueError("LATS rule count does not match probability columns.")
    if margin_scale <= 0.0 or delta_scale <= 0.0 or dispersion_scale <= 0.0:
        raise ValueError("Feature scales must be greater than zero.")

    ids = np.asarray([str(item) for item in parent_ids], dtype=object)
    if len(ids) != len(current):
        raise ValueError("parent_ids length does not match probability rows.")

    unique_ids, current_scores, inverse = aggregate_parent_probabilities(
        probabilities=current,
        parent_ids=ids,
        rules=rules,
    )
    previous_ids, previous_scores, previous_inverse = aggregate_parent_probabilities(
        probabilities=previous,
        parent_ids=ids,
        rules=rules,
    )
    if not np.array_equal(unique_ids, previous_ids) or not np.array_equal(
        inverse, previous_inverse
    ):
        raise RuntimeError("Current and previous parent aggregation misaligned.")

    thresholds = np.asarray(
        [float(rule.threshold) for rule in rules],
        dtype=np.float32,
    )
    current_pred = parent_predictions(current_scores, rules)
    previous_pred = parent_predictions(previous_scores, rules)
    signed_margin = current_scores - thresholds.reshape(1, -1)
    absolute_margin = np.abs(signed_margin)
    score_delta = np.abs(current_scores - previous_scores)
    label_disagreement = (current_pred != previous_pred).astype(np.float32)

    stats = _parent_segment_statistics(current, inverse, len(unique_ids))
    clipped = np.clip(current, 1e-7, 1.0 - 1e-7)
    segment_entropy = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )
    parent_entropy_mean = np.zeros(len(unique_ids), dtype=np.float32)
    for parent_index in range(len(unique_ids)):
        parent_entropy_mean[parent_index] = float(
            segment_entropy[inverse == parent_index].mean()
        )

    per_label_parts = [
        current_scores,
        previous_scores,
        score_delta,
        signed_margin,
        absolute_margin,
        stats["mean"],
        stats["max"],
        stats["min"],
        stats["std"],
        stats["range"],
        stats["positive_fraction"],
        label_disagreement,
    ]
    per_label_prefixes = [
        "exit2_lats_score",
        "exit1_lats_score",
        "lats_score_delta",
        "signed_lats_margin",
        "absolute_lats_margin",
        "segment_mean",
        "segment_max",
        "segment_min",
        "segment_std",
        "segment_range",
        "segment_positive_fraction",
        "exit1_exit2_parent_disagreement",
    ]
    feature_names: list[str] = []
    for prefix in per_label_prefixes:
        feature_names.extend(
            [f"{prefix}_{label_idx}" for label_idx in range(current.shape[1])]
        )

    scalar = np.column_stack(
        [
            absolute_margin.min(axis=1),
            absolute_margin.mean(axis=1),
            score_delta.max(axis=1),
            score_delta.mean(axis=1),
            stats["std"].max(axis=1),
            stats["std"].mean(axis=1),
            label_disagreement.sum(axis=1),
            current_pred.sum(axis=1),
            (current_pred.sum(axis=1) > 0).astype(np.float32),
            stats["parent_size"],
            parent_entropy_mean,
        ]
    ).astype(np.float32)
    scalar_names = [
        "minimum_parent_lats_margin",
        "mean_parent_lats_margin",
        "maximum_parent_score_delta",
        "mean_parent_score_delta",
        "maximum_segment_dispersion",
        "mean_segment_dispersion",
        "parent_label_disagreement_count",
        "predicted_label_count",
        "non_empty",
        "parent_segment_count",
        "mean_segment_binary_entropy",
    ]
    features = np.concatenate([*per_label_parts, scalar], axis=1).astype(
        np.float32
    )

    margin_uncertainty = 1.0 - np.clip(
        absolute_margin / float(margin_scale), 0.0, 1.0
    )
    normalized_delta = np.clip(
        score_delta / float(delta_scale), 0.0, 1.0
    )
    normalized_dispersion = np.clip(
        stats["std"] / float(dispersion_scale), 0.0, 1.0
    )
    raw_risk = (
        0.45 * margin_uncertainty
        + 0.25 * normalized_delta
        + 0.20 * normalized_dispersion
        + 0.10 * label_disagreement
    ).astype(np.float32)

    diagnostics = {
        "parent_ids": unique_ids,
        "row_to_parent": inverse,
        "current_scores": current_scores.astype(np.float32),
        "previous_scores": previous_scores.astype(np.float32),
        "current_predictions": current_pred.astype(np.int8),
        "previous_predictions": previous_pred.astype(np.int8),
        "absolute_lats_margin": absolute_margin.astype(np.float32),
        "signed_lats_margin": signed_margin.astype(np.float32),
        "lats_score_delta": score_delta.astype(np.float32),
        "segment_std": stats["std"].astype(np.float32),
        "non_empty": (current_pred.sum(axis=1) > 0),
        "raw_nonparametric_risk": raw_risk,
    }
    return features, feature_names + scalar_names, diagnostics


def whole_parent_unsafe_targets(
    *,
    y_true: np.ndarray,
    source_probabilities: np.ndarray,
    deeper_probabilities: np.ndarray,
    parent_ids: Sequence[Any],
    rules: Sequence[LATSLabelRule],
) -> dict[str, np.ndarray]:
    """Compare the complete all-Exit2 parent against the all-Exit3 parent."""

    source = _matrix(source_probabilities, "source_probabilities")
    deeper = _matrix(deeper_probabilities, "deeper_probabilities")
    if source.shape != deeper.shape:
        raise ValueError("Source and deeper probabilities must share shape.")

    truth_ids, parent_truth, truth_inverse = parent_truth_matrix(
        y_true, parent_ids
    )
    source_ids, source_scores, source_inverse = aggregate_parent_probabilities(
        probabilities=source, parent_ids=parent_ids, rules=rules
    )
    deeper_ids, deeper_scores, deeper_inverse = aggregate_parent_probabilities(
        probabilities=deeper, parent_ids=parent_ids, rules=rules
    )
    if not (
        np.array_equal(truth_ids, source_ids)
        and np.array_equal(source_ids, deeper_ids)
        and np.array_equal(truth_inverse, source_inverse)
        and np.array_equal(source_inverse, deeper_inverse)
    ):
        raise RuntimeError("Parent aggregation or truth alignment failed.")

    source_pred = parent_predictions(source_scores, rules)
    deeper_pred = parent_predictions(deeper_scores, rules)
    unsafe = (
        (deeper_pred == parent_truth) & (source_pred != parent_truth)
    ).astype(np.int8)
    beneficial = (
        (source_pred == parent_truth) & (deeper_pred != parent_truth)
    ).astype(np.int8)

    source_error_count = np.sum(source_pred != parent_truth, axis=1).astype(
        np.int16
    )
    deeper_error_count = np.sum(deeper_pred != parent_truth, axis=1).astype(
        np.int16
    )
    return {
        "parent_ids": truth_ids,
        "row_to_parent": truth_inverse,
        "parent_truth": parent_truth,
        "source_scores": source_scores.astype(np.float32),
        "deeper_scores": deeper_scores.astype(np.float32),
        "source_predictions": source_pred.astype(np.int8),
        "deeper_predictions": deeper_pred.astype(np.int8),
        "unsafe_targets": unsafe,
        "beneficial_targets": beneficial,
        "source_error_count": source_error_count,
        "deeper_error_count": deeper_error_count,
        "net_error_increase": (source_error_count - deeper_error_count).astype(
            np.int16
        ),
        "any_unsafe": np.any(unsafe == 1, axis=1),
    }


def expand_parent_label_features(
    parent_features: np.ndarray,
    num_labels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(parent_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("parent_features must be a 2-D matrix.")
    if num_labels < 1:
        raise ValueError("num_labels must be positive.")

    parent_index = np.repeat(
        np.arange(len(features), dtype=np.int64), num_labels
    )
    label_index = np.tile(
        np.arange(num_labels, dtype=np.int64), len(features)
    )
    repeated = features[parent_index]
    one_hot = np.eye(num_labels, dtype=np.float32)[label_index]
    expanded = np.concatenate([repeated, one_hot], axis=1).astype(np.float32)
    return expanded, parent_index, label_index


def predict_shared_unsafe_probabilities(
    model: Any,
    parent_features: np.ndarray,
    num_labels: int,
) -> np.ndarray:
    expanded, parent_index, label_index = expand_parent_label_features(
        parent_features, num_labels
    )
    probabilities = np.asarray(model.predict_proba(expanded), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Shared gate must return two-class probabilities.")
    result = np.zeros((len(parent_features), num_labels), dtype=np.float32)
    result[parent_index, label_index] = probabilities[:, 1].astype(np.float32)
    return result


@dataclass(frozen=True)
class EmpiricalRiskCalibrator:
    bin_edges: tuple[float, ...]
    bin_probabilities: tuple[float, ...]

    def predict(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float32)
        edges = np.asarray(self.bin_edges, dtype=np.float32)
        probabilities = np.asarray(self.bin_probabilities, dtype=np.float32)
        indices = np.searchsorted(edges, values, side="right") - 1
        indices = np.clip(indices, 0, len(probabilities) - 1)
        return probabilities[indices]


def _fit_single_empirical_calibrator(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    num_bins: int,
    alpha: float,
    beta: float,
) -> EmpiricalRiskCalibrator:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(targets, dtype=np.int8).reshape(-1)
    if len(values) != len(labels) or len(values) == 0:
        raise ValueError("Calibration scores and targets must align.")
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")

    quantiles = np.linspace(0.0, 1.0, int(num_bins) + 1)
    edges = np.unique(np.quantile(values, quantiles).astype(np.float32))
    if len(edges) < 2:
        epsilon = np.float32(1e-6)
        edges = np.asarray(
            [float(values[0] - epsilon), float(values[0] + epsilon)],
            dtype=np.float32,
        )
    edges[0] = min(float(edges[0]), float(values.min()) - 1e-6)
    edges[-1] = max(float(edges[-1]), float(values.max()) + 1e-6)

    bin_ids = np.searchsorted(edges, values, side="right") - 1
    bin_count = len(edges) - 1
    bin_ids = np.clip(bin_ids, 0, bin_count - 1)
    estimates = np.zeros(bin_count, dtype=np.float32)
    for bin_index in range(bin_count):
        mask = bin_ids == bin_index
        positives = float(labels[mask].sum())
        count = float(mask.sum())
        estimates[bin_index] = float(
            (positives + float(alpha))
            / (count + float(alpha) + float(beta))
        )
    estimates = np.maximum.accumulate(estimates)
    return EmpiricalRiskCalibrator(
        bin_edges=tuple(float(item) for item in edges),
        bin_probabilities=tuple(float(item) for item in estimates),
    )


def fit_empirical_risk_calibrators(
    *,
    raw_scores: np.ndarray,
    unsafe_targets: np.ndarray,
    num_bins: int = 5,
    minimum_positive_examples: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[list[EmpiricalRiskCalibrator], np.ndarray, np.ndarray]:
    scores = _matrix(raw_scores, "raw_scores")
    targets = np.asarray(unsafe_targets, dtype=np.int8)
    if scores.shape != targets.shape:
        raise ValueError("raw_scores and unsafe_targets must share shape.")
    if minimum_positive_examples < 1:
        raise ValueError("minimum_positive_examples must be positive.")

    pooled = _fit_single_empirical_calibrator(
        scores.reshape(-1),
        targets.reshape(-1),
        num_bins=num_bins,
        alpha=alpha,
        beta=beta,
    )
    calibrators: list[EmpiricalRiskCalibrator] = []
    positive_counts = targets.sum(axis=0).astype(np.int64)
    used_pooled = np.zeros(scores.shape[1], dtype=bool)
    for label_index in range(scores.shape[1]):
        if positive_counts[label_index] < int(minimum_positive_examples):
            calibrators.append(pooled)
            used_pooled[label_index] = True
        else:
            calibrators.append(
                _fit_single_empirical_calibrator(
                    scores[:, label_index],
                    targets[:, label_index],
                    num_bins=num_bins,
                    alpha=alpha,
                    beta=beta,
                )
            )
    return calibrators, positive_counts, used_pooled


def predict_empirical_unsafe_probabilities(
    calibrators: Sequence[EmpiricalRiskCalibrator],
    raw_scores: np.ndarray,
) -> np.ndarray:
    scores = _matrix(raw_scores, "raw_scores")
    if len(calibrators) != scores.shape[1]:
        raise ValueError("Calibrator count does not match label count.")
    return np.column_stack(
        [
            calibrator.predict(scores[:, label_index])
            for label_index, calibrator in enumerate(calibrators)
        ]
    ).astype(np.float32)


def derive_label_unsafe_thresholds(
    *,
    unsafe_targets: np.ndarray,
    unsafe_probabilities: np.ndarray,
    target_recall: float,
    minimum_positive_examples: int = 3,
    fallback_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < float(target_recall) <= 1.0:
        raise ValueError("target_recall must be in (0, 1].")
    targets = np.asarray(unsafe_targets, dtype=np.int8)
    probabilities = _matrix(unsafe_probabilities, "unsafe_probabilities")
    if targets.shape != probabilities.shape:
        raise ValueError("Targets and probabilities must share shape.")

    pooled_scores = probabilities[targets == 1]
    if fallback_threshold is None:
        fallback = (
            float(
                np.quantile(
                    pooled_scores,
                    max(0.0, 1.0 - float(target_recall)),
                )
            )
            if len(pooled_scores) > 0
            else 0.5
        )
    else:
        fallback = float(fallback_threshold)

    thresholds = np.full(
        probabilities.shape[1], fallback, dtype=np.float32
    )
    counts = targets.sum(axis=0).astype(np.int64)
    used_fallback = np.ones(probabilities.shape[1], dtype=bool)
    quantile = max(0.0, 1.0 - float(target_recall))
    for label_index in range(probabilities.shape[1]):
        values = probabilities[
            targets[:, label_index] == 1, label_index
        ]
        if len(values) >= int(minimum_positive_examples):
            thresholds[label_index] = float(np.quantile(values, quantile))
            used_fallback[label_index] = False

    return (
        np.clip(thresholds, 0.0, 1.0).astype(np.float32),
        counts,
        used_fallback,
    )


def whole_parent_stop_mask(
    *,
    unsafe_probabilities: np.ndarray,
    label_thresholds: np.ndarray | Sequence[float],
    expected_harm_threshold: float | None,
    non_empty: np.ndarray | None = None,
    label_weights: np.ndarray | Sequence[float] | None = None,
    allow_empty_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = _matrix(unsafe_probabilities, "unsafe_probabilities")
    thresholds = _vector(
        label_thresholds, probabilities.shape[1], "label_thresholds"
    )
    if label_weights is None:
        weights = np.ones(probabilities.shape[1], dtype=np.float32)
    else:
        weights = _vector(
            label_weights, probabilities.shape[1], "label_weights"
        )
        if np.any(weights < 0.0):
            raise ValueError("label_weights must be non-negative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("At least one label weight must be positive.")
    weights = weights / total

    triggered = probabilities >= thresholds.reshape(1, -1)
    expected_harm = probabilities @ weights
    continue_mask = np.any(triggered, axis=1)
    if expected_harm_threshold is not None:
        continue_mask |= expected_harm >= float(expected_harm_threshold)
    stop_mask = ~continue_mask
    if not allow_empty_stop and non_empty is not None:
        stop_mask &= np.asarray(non_empty, dtype=bool).reshape(-1)
    highest_risk = np.argmax(
        probabilities / np.maximum(thresholds.reshape(1, -1), 1e-6),
        axis=1,
    ).astype(np.int64)
    return stop_mask, expected_harm.astype(np.float32), highest_risk


def wilson_upper_bound(
    errors: int,
    total: int,
    *,
    z: float = 1.645,
) -> float:
    if total <= 0:
        return 0.0
    count = float(errors)
    n = float(total)
    proportion = count / n
    denominator = 1.0 + (z * z) / n
    centre = proportion + (z * z) / (2.0 * n)
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / n
        + (z * z) / (4.0 * n * n)
    )
    return float((centre + radius) / denominator)
