from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, labels], got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return arr


def _vector(value: np.ndarray | Sequence[float], size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def label_predictions(
    probabilities: np.ndarray,
    thresholds: np.ndarray | Sequence[float],
) -> np.ndarray:
    probs = _matrix(probabilities, "probabilities")
    thr = _vector(thresholds, probs.shape[1], "thresholds")
    return (probs >= thr.reshape(1, -1)).astype(np.int8)


@dataclass(frozen=True)
class LATSLabelRule:
    aggregation: str
    threshold: float


@dataclass(frozen=True)
class ConstantProbabilityModel:
    """Tiny joblib-safe estimator for labels with one training target class."""

    probability: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        count = len(np.asarray(features))
        positive = np.full(count, float(self.probability), dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def parse_lats_rules(
    payload: dict[str, Any],
    labels: Sequence[str],
) -> tuple[LATSLabelRule, ...]:
    config = payload.get("config", payload)
    rules: list[LATSLabelRule] = []
    for label in labels:
        if label not in config:
            raise KeyError(f"LATS configuration is missing label {label!r}.")
        item = config[label]
        aggregation = str(item["aggregation"]).strip().lower()
        threshold = float(item["threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid LATS threshold for {label}: {threshold}")
        rules.append(LATSLabelRule(aggregation=aggregation, threshold=threshold))
    return tuple(rules)


def aggregate_1d(values: np.ndarray, aggregation: str) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(data) == 0:
        raise ValueError("Cannot aggregate an empty parent.")
    method = str(aggregation).strip().lower()
    if method == "mean":
        return float(np.mean(data))
    if method == "max":
        return float(np.max(data))
    if method == "p75":
        return float(np.quantile(data, 0.75))
    if method.startswith("top") and method.endswith("mean"):
        digits = method[3:-4]
        if not digits.isdigit():
            raise ValueError(f"Unsupported top-k aggregation: {aggregation}")
        k = max(1, min(int(digits), len(data)))
        return float(np.mean(np.sort(data)[-k:]))
    if method == "noisy_or":
        return float(1.0 - np.prod(1.0 - np.clip(data, 0.0, 1.0)))
    raise ValueError(f"Unsupported LATS aggregation: {aggregation}")


def aggregate_parent_probabilities(
    *,
    probabilities: np.ndarray,
    parent_ids: Sequence[Any],
    rules: Sequence[LATSLabelRule],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = _matrix(probabilities, "probabilities")
    ids = np.asarray([str(item) for item in parent_ids], dtype=object)
    if len(ids) != len(probs):
        raise ValueError("parent_ids length does not match probability rows.")
    if len(rules) != probs.shape[1]:
        raise ValueError("LATS rule count does not match probability columns.")

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    scores = np.zeros((len(unique_ids), probs.shape[1]), dtype=np.float32)
    for parent_index in range(len(unique_ids)):
        parent_mask = inverse == parent_index
        parent_probs = probs[parent_mask]
        for label_index, rule in enumerate(rules):
            scores[parent_index, label_index] = aggregate_1d(
                parent_probs[:, label_index],
                rule.aggregation,
            )
    return unique_ids, scores, inverse.astype(np.int64)


def parent_predictions(
    scores: np.ndarray,
    rules: Sequence[LATSLabelRule],
) -> np.ndarray:
    matrix = _matrix(scores, "scores")
    thresholds = np.asarray([rule.threshold for rule in rules], dtype=np.float32)
    return label_predictions(matrix, thresholds)


def build_parent_aware_features(
    *,
    current_probabilities: np.ndarray,
    parent_ids: Sequence[Any],
    current_thresholds: np.ndarray | Sequence[float],
    rules: Sequence[LATSLabelRule],
    previous_probabilities: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    current = _matrix(current_probabilities, "current_probabilities")
    num_labels = current.shape[1]
    thresholds = _vector(current_thresholds, num_labels, "current_thresholds")
    previous = None
    if previous_probabilities is not None:
        previous = _matrix(previous_probabilities, "previous_probabilities")
        if previous.shape != current.shape:
            raise ValueError("Previous and current probabilities must share shape.")

    _, parent_scores, row_to_parent = aggregate_parent_probabilities(
        probabilities=current,
        parent_ids=parent_ids,
        rules=rules,
    )
    row_parent_scores = parent_scores[row_to_parent]
    lats_thresholds = np.asarray([rule.threshold for rule in rules], dtype=np.float32)
    decision_margin = np.abs(current - thresholds.reshape(1, -1))
    parent_lats_margin = np.abs(
        row_parent_scores - lats_thresholds.reshape(1, -1)
    )
    segment_parent_deviation = np.abs(current - row_parent_scores)
    current_pred = label_predictions(current, thresholds)
    confidence = np.maximum(current, 1.0 - current)
    clipped = np.clip(current, 1e-7, 1.0 - 1e-7)
    entropy = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )

    parts: list[np.ndarray] = [current]
    names: list[str] = [f"current_prob_{idx}" for idx in range(num_labels)]
    if previous is not None:
        delta = np.abs(current - previous)
        parts.extend([previous, delta])
        names.extend([f"previous_prob_{idx}" for idx in range(num_labels)])
        names.extend([f"inter_exit_delta_{idx}" for idx in range(num_labels)])
    else:
        delta = np.zeros_like(current, dtype=np.float32)

    parts.extend(
        [
            decision_margin,
            row_parent_scores,
            parent_lats_margin,
            segment_parent_deviation,
        ]
    )
    names.extend([f"decision_margin_{idx}" for idx in range(num_labels)])
    names.extend([f"parent_lats_score_{idx}" for idx in range(num_labels)])
    names.extend([f"parent_lats_margin_{idx}" for idx in range(num_labels)])
    names.extend([f"segment_parent_deviation_{idx}" for idx in range(num_labels)])

    ids = np.asarray([str(item) for item in parent_ids], dtype=object)
    unique, counts = np.unique(ids, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    parent_size = np.asarray([count_map[item] for item in ids], dtype=np.float32)
    scalar = np.column_stack(
        [
            confidence.mean(axis=1),
            confidence.min(axis=1),
            entropy.mean(axis=1),
            decision_margin.min(axis=1),
            parent_lats_margin.min(axis=1),
            segment_parent_deviation.max(axis=1),
            delta.max(axis=1),
            (current_pred.sum(axis=1) > 0).astype(np.float32),
            current_pred.sum(axis=1).astype(np.float32),
            parent_size,
        ]
    ).astype(np.float32)
    scalar_names = [
        "mean_binary_confidence",
        "minimum_binary_confidence",
        "mean_binary_entropy",
        "minimum_decision_margin",
        "minimum_parent_lats_margin",
        "maximum_segment_parent_deviation",
        "maximum_inter_exit_delta",
        "non_empty",
        "predicted_label_count",
        "parent_segment_count",
    ]
    features = np.concatenate([*parts, scalar], axis=1).astype(np.float32)
    diagnostics = {
        "parent_scores": row_parent_scores.astype(np.float32),
        "parent_lats_margin": parent_lats_margin.astype(np.float32),
        "decision_margin": decision_margin.astype(np.float32),
        "inter_exit_delta": delta.astype(np.float32),
        "non_empty": (current_pred.sum(axis=1) > 0),
        "predicted_label_count": current_pred.sum(axis=1).astype(np.float32),
    }
    return features, names + scalar_names, diagnostics


def counterfactual_parent_unsafe_targets(
    *,
    y_true: np.ndarray,
    source_probabilities: np.ndarray,
    deeper_probabilities: np.ndarray,
    parent_ids: Sequence[Any],
    rules: Sequence[LATSLabelRule],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-label unsafe targets for replacing one deeper segment by source.

    A label is unsafe for a segment when the all-deeper parent prediction is
    correct for that label but substituting the source-exit probability for the
    segment makes the parent prediction wrong.
    """

    truth = np.asarray(y_true, dtype=np.int8)
    source = _matrix(source_probabilities, "source_probabilities")
    deeper = _matrix(deeper_probabilities, "deeper_probabilities")
    if truth.shape != source.shape or source.shape != deeper.shape:
        raise ValueError("Truth and source/deeper probabilities must share shape.")

    ids = np.asarray([str(item) for item in parent_ids], dtype=object)
    if len(ids) != len(source):
        raise ValueError("parent_ids length does not match samples.")

    unique_ids = np.unique(ids)
    unsafe = np.zeros_like(truth, dtype=np.int8)
    baseline_parent_pred_by_row = np.zeros_like(truth, dtype=np.int8)
    counterfactual_parent_pred_by_row = np.zeros_like(truth, dtype=np.int8)

    for parent_id in unique_ids:
        row_indices = np.flatnonzero(ids == parent_id)
        parent_truth = truth[row_indices[0]]
        if not np.all(truth[row_indices] == parent_truth.reshape(1, -1)):
            raise ValueError(
                f"Ground-truth labels differ within parent {parent_id!r}."
            )
        parent_deeper = deeper[row_indices]
        baseline_scores = np.asarray(
            [
                aggregate_1d(parent_deeper[:, label_idx], rules[label_idx].aggregation)
                for label_idx in range(deeper.shape[1])
            ],
            dtype=np.float32,
        )
        baseline_pred = (
            baseline_scores
            >= np.asarray([rule.threshold for rule in rules], dtype=np.float32)
        ).astype(np.int8)

        for local_index, row_index in enumerate(row_indices):
            hybrid = parent_deeper.copy()
            hybrid[local_index] = source[row_index]
            hybrid_scores = np.asarray(
                [
                    aggregate_1d(hybrid[:, label_idx], rules[label_idx].aggregation)
                    for label_idx in range(deeper.shape[1])
                ],
                dtype=np.float32,
            )
            hybrid_pred = (
                hybrid_scores
                >= np.asarray([rule.threshold for rule in rules], dtype=np.float32)
            ).astype(np.int8)
            unsafe[row_index] = (
                (baseline_pred == parent_truth)
                & (hybrid_pred != parent_truth)
            ).astype(np.int8)
            baseline_parent_pred_by_row[row_index] = baseline_pred
            counterfactual_parent_pred_by_row[row_index] = hybrid_pred

    return unsafe, baseline_parent_pred_by_row, counterfactual_parent_pred_by_row


def derive_label_probability_thresholds(
    *,
    unsafe_targets: np.ndarray,
    unsafe_probabilities: np.ndarray,
    target_recall: float,
    minimum_positive_examples: int = 3,
    fallback_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < float(target_recall) <= 1.0:
        raise ValueError("target_recall must be in (0, 1].")
    if minimum_positive_examples < 1:
        raise ValueError("minimum_positive_examples must be at least 1.")
    targets = np.asarray(unsafe_targets, dtype=np.int8)
    probs = _matrix(unsafe_probabilities, "unsafe_probabilities")
    if targets.shape != probs.shape:
        raise ValueError("Unsafe targets and probabilities must share shape.")

    positive_scores = probs[targets == 1]
    if fallback_threshold is None:
        if len(positive_scores) > 0:
            fallback = float(
                np.quantile(positive_scores, max(0.0, 1.0 - float(target_recall)))
            )
        else:
            fallback = 0.5
    else:
        fallback = float(fallback_threshold)

    thresholds = np.full(probs.shape[1], fallback, dtype=np.float32)
    counts = targets.sum(axis=0).astype(np.int64)
    used_fallback = np.ones(probs.shape[1], dtype=bool)
    quantile = max(0.0, 1.0 - float(target_recall))
    for label_idx in range(probs.shape[1]):
        label_scores = probs[targets[:, label_idx] == 1, label_idx]
        if len(label_scores) >= int(minimum_positive_examples):
            thresholds[label_idx] = float(np.quantile(label_scores, quantile))
            used_fallback[label_idx] = False
    thresholds = np.clip(thresholds, 0.0, 1.0).astype(np.float32)
    return thresholds, counts, used_fallback


def adaptive_label_stop_mask(
    *,
    unsafe_probabilities: np.ndarray,
    label_thresholds: np.ndarray | Sequence[float],
    expected_harm_threshold: float | None = None,
    label_weights: np.ndarray | Sequence[float] | None = None,
    non_empty: np.ndarray | None = None,
    allow_empty_stop: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = _matrix(unsafe_probabilities, "unsafe_probabilities")
    thresholds = _vector(label_thresholds, probs.shape[1], "label_thresholds")
    triggered = probs >= thresholds.reshape(1, -1)

    if label_weights is None:
        weights = np.ones(probs.shape[1], dtype=np.float32) / float(probs.shape[1])
    else:
        weights = _vector(label_weights, probs.shape[1], "label_weights")
        if np.any(weights < 0.0):
            raise ValueError("label_weights must be non-negative.")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("At least one label weight must be positive.")
        weights = weights / total
    expected_harm = probs @ weights

    continue_mask = np.any(triggered, axis=1)
    if expected_harm_threshold is not None:
        continue_mask |= expected_harm >= float(expected_harm_threshold)
    stop_mask = ~continue_mask
    if not allow_empty_stop and non_empty is not None:
        stop_mask &= np.asarray(non_empty, dtype=bool).reshape(-1)
    highest_risk_label = np.argmax(
        probs / np.maximum(thresholds.reshape(1, -1), 1e-6),
        axis=1,
    ).astype(np.int64)
    return stop_mask, expected_harm.astype(np.float32), highest_risk_label


def predict_multilabel_unsafe_probabilities(
    models: Sequence[Any],
    features: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    columns: list[np.ndarray] = []
    for model in models:
        probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Each gate model must return two-class probabilities.")
        columns.append(probabilities[:, 1].astype(np.float32))
    return np.column_stack(columns).astype(np.float32)
