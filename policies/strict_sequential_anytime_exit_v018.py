from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, labels], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _vector(value: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int8).reshape(-1)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else float(2 * tp / denominator)


@dataclass(frozen=True)
class StrictSequentialStageConfig:
    mean_confidence_threshold: float
    max_probability_delta: float
    max_label_risk: float
    risk_score_threshold: float
    risk_margin_multiplier: float
    risk_uncertainty_band: float
    exit1_confidence_boost: float
    per_label_margins: tuple[float, ...]
    require_previous_label_stability: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrictSequentialPolicyConfig:
    num_exits: int
    stages: tuple[StrictSequentialStageConfig, ...]
    allow_empty_stop: bool = False

    def __post_init__(self) -> None:
        if int(self.num_exits) < 2:
            raise ValueError("num_exits must be at least two.")
        if len(self.stages) != int(self.num_exits) - 1:
            raise ValueError("One strict stage config is required per non-final exit.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_exits": int(self.num_exits),
            "allow_empty_stop": bool(self.allow_empty_stop),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def label_predictions(probabilities: np.ndarray, thresholds: Sequence[float] | np.ndarray) -> np.ndarray:
    probs = _matrix(probabilities, "probabilities")
    threshold_array = _vector(thresholds, probs.shape[1], "thresholds")
    return (probs >= threshold_array.reshape(1, -1)).astype(np.int8)


def genes_per_stage(num_labels: int) -> int:
    return int(num_labels) + 7


def total_genes(num_exits: int, num_labels: int) -> int:
    return (int(num_exits) - 1) * genes_per_stage(num_labels)


def make_strict_bounds(
    *,
    num_exits: int,
    num_labels: int,
    confidence_bounds: tuple[float, float] = (0.55, 0.995),
    delta_bounds: tuple[float, float] = (0.0, 1.0),
    risk_bounds: tuple[float, float] = (0.0, 1.0),
    risk_score_bounds: tuple[float, float] = (0.35, 0.95),
    risk_multiplier_bounds: tuple[float, float] = (1.0, 3.0),
    risk_band_bounds: tuple[float, float] = (0.01, 0.25),
    exit1_boost_bounds: tuple[float, float] = (0.0, 0.20),
    margin_bounds: tuple[float, float] = (0.0, 0.50),
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        confidence_bounds,
        delta_bounds,
        risk_bounds,
        risk_score_bounds,
        risk_multiplier_bounds,
        risk_band_bounds,
        exit1_boost_bounds,
        margin_bounds,
    ]
    if any(float(low) >= float(high) for low, high in pairs):
        raise ValueError("Each strict optimisation lower bound must be below its upper bound.")
    lower_stage = [
        confidence_bounds[0], delta_bounds[0], risk_bounds[0],
        risk_score_bounds[0], risk_multiplier_bounds[0], risk_band_bounds[0],
        exit1_boost_bounds[0], *([margin_bounds[0]] * int(num_labels)),
    ]
    upper_stage = [
        confidence_bounds[1], delta_bounds[1], risk_bounds[1],
        risk_score_bounds[1], risk_multiplier_bounds[1], risk_band_bounds[1],
        exit1_boost_bounds[1], *([margin_bounds[1]] * int(num_labels)),
    ]
    return (
        np.tile(np.asarray(lower_stage, dtype=np.float64), int(num_exits) - 1),
        np.tile(np.asarray(upper_stage, dtype=np.float64), int(num_exits) - 1),
    )


def decode_strict_genes(
    genes: Sequence[float] | np.ndarray,
    *,
    num_exits: int,
    num_labels: int,
    allow_empty_stop: bool = False,
) -> StrictSequentialPolicyConfig:
    values = _vector(genes, total_genes(num_exits, num_labels), "genes")
    width = genes_per_stage(num_labels)
    stages: list[StrictSequentialStageConfig] = []
    for stage_index in range(int(num_exits) - 1):
        block = values[stage_index * width : (stage_index + 1) * width]
        stages.append(
            StrictSequentialStageConfig(
                mean_confidence_threshold=float(block[0]),
                max_probability_delta=float(block[1]),
                max_label_risk=float(block[2]),
                risk_score_threshold=float(block[3]),
                risk_margin_multiplier=float(block[4]),
                risk_uncertainty_band=float(block[5]),
                exit1_confidence_boost=float(block[6]),
                per_label_margins=tuple(float(item) for item in block[7:]),
                require_previous_label_stability=stage_index > 0,
            )
        )
    return StrictSequentialPolicyConfig(
        num_exits=int(num_exits), stages=tuple(stages), allow_empty_stop=bool(allow_empty_stop)
    )


def encode_strict_config(config: StrictSequentialPolicyConfig) -> np.ndarray:
    values: list[float] = []
    for stage in config.stages:
        values.extend([
            float(stage.mean_confidence_threshold),
            float(stage.max_probability_delta),
            float(stage.max_label_risk),
            float(stage.risk_score_threshold),
            float(stage.risk_margin_multiplier),
            float(stage.risk_uncertainty_band),
            float(stage.exit1_confidence_boost),
            *[float(value) for value in stage.per_label_margins],
        ])
    return np.asarray(values, dtype=np.float64)


def derive_strict_continuation_profile(
    *,
    y_true: np.ndarray,
    exit_probabilities: Sequence[np.ndarray],
    thresholds_by_exit: Sequence[Sequence[float] | np.ndarray],
    minimum_score: float = 0.05,
) -> dict[str, np.ndarray]:
    probabilities = [
        _matrix(item, f"exit{index + 1}_probabilities")
        for index, item in enumerate(exit_probabilities)
    ]
    if len(probabilities) < 2:
        raise ValueError("At least two exits are required.")
    shape = probabilities[0].shape
    if any(item.shape != shape for item in probabilities):
        raise ValueError("All exit probability matrices must share shape.")
    truth = np.asarray(y_true, dtype=np.int8)
    if truth.shape != shape:
        raise ValueError("y_true and exit probabilities must share shape.")
    if len(thresholds_by_exit) != len(probabilities):
        raise ValueError("A threshold vector is required for every exit.")

    predictions = [
        label_predictions(probabilities[index], thresholds_by_exit[index])
        for index in range(len(probabilities))
    ]
    final_prediction = predictions[-1]
    stages = len(probabilities) - 1
    labels = shape[1]
    correction_counts = np.zeros((stages, labels), dtype=np.int64)
    error_counts = np.zeros((stages, labels), dtype=np.int64)
    correction_rates = np.zeros((stages, labels), dtype=np.float32)
    f1_gains = np.zeros((stages, labels), dtype=np.float32)
    risk_scores = np.zeros((stages, labels), dtype=np.float32)

    for exit_index in range(stages):
        current = predictions[exit_index]
        current_errors = current != truth
        corrected = current_errors & (final_prediction == truth)
        error_counts[exit_index] = current_errors.sum(axis=0)
        correction_counts[exit_index] = corrected.sum(axis=0)
        correction_rates[exit_index] = correction_counts[exit_index] / np.maximum(
            error_counts[exit_index], 1
        )
        for label_index in range(labels):
            f1_gains[exit_index, label_index] = max(
                0.0,
                _binary_f1(truth[:, label_index], final_prediction[:, label_index])
                - _binary_f1(truth[:, label_index], current[:, label_index]),
            )
        correction_scale = max(float(correction_rates[exit_index].max()), 1e-12)
        f1_scale = max(float(f1_gains[exit_index].max()), 1e-12)
        correction_norm = correction_rates[exit_index] / correction_scale
        f1_norm = f1_gains[exit_index] / f1_scale
        risk_scores[exit_index] = np.maximum(
            float(minimum_score), 0.65 * correction_norm + 0.35 * f1_norm
        )

    return {
        "risk_scores": risk_scores,
        "correction_counts": correction_counts,
        "error_counts": error_counts,
        "correction_rates": correction_rates,
        "f1_gains": f1_gains,
    }


def stage_diagnostics(
    *,
    current_probabilities: np.ndarray,
    current_thresholds: Sequence[float] | np.ndarray,
    previous_probabilities: np.ndarray | None = None,
    previous_thresholds: Sequence[float] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    current = _matrix(current_probabilities, "current_probabilities")
    thresholds = _vector(current_thresholds, current.shape[1], "current_thresholds")
    prediction = label_predictions(current, thresholds)
    margin = np.abs(current - thresholds.reshape(1, -1))
    confidence = np.maximum(current, 1.0 - current)
    if previous_probabilities is None:
        delta = np.zeros_like(current, dtype=np.float32)
        stable = np.ones(len(current), dtype=bool)
    else:
        previous = _matrix(previous_probabilities, "previous_probabilities")
        if previous.shape != current.shape:
            raise ValueError("Previous and current probabilities must share shape.")
        if previous_thresholds is None:
            raise ValueError("previous_thresholds are required with previous probabilities.")
        previous_prediction = label_predictions(previous, previous_thresholds)
        delta = np.abs(current - previous)
        stable = np.all(previous_prediction == prediction, axis=1)
    return {
        "prediction": prediction,
        "non_empty": prediction.sum(axis=1) > 0,
        "mean_binary_confidence": confidence.mean(axis=1).astype(np.float32),
        "decision_margin": margin.astype(np.float32),
        "maximum_probability_delta": delta.max(axis=1).astype(np.float32),
        "label_set_stability": stable,
    }


def strict_stage_stop_mask(
    *,
    diagnostic: dict[str, np.ndarray],
    stage: StrictSequentialStageConfig,
    risk_scores: Sequence[float] | np.ndarray,
    exit_index: int,
    allow_empty_stop: bool,
) -> np.ndarray:
    margin = _matrix(diagnostic["decision_margin"], "decision_margin")
    label_count = margin.shape[1]
    required = _vector(stage.per_label_margins, label_count, "per_label_margins")
    scores = _vector(risk_scores, label_count, "risk_scores")
    confidence_threshold = float(stage.mean_confidence_threshold)
    if int(exit_index) == 0:
        confidence_threshold = min(
            0.999, confidence_threshold + float(stage.exit1_confidence_boost)
        )

    stop = np.ones(len(margin), dtype=bool)
    if stage.require_previous_label_stability and int(exit_index) > 0:
        stop &= np.asarray(diagnostic["label_set_stability"], dtype=bool)
    if not allow_empty_stop:
        stop &= np.asarray(diagnostic["non_empty"], dtype=bool)
    stop &= np.asarray(diagnostic["mean_binary_confidence"], dtype=float) >= confidence_threshold
    stop &= np.asarray(diagnostic["maximum_probability_delta"], dtype=float) <= float(
        stage.max_probability_delta
    )
    stop &= np.all(margin >= required.reshape(1, -1), axis=1)

    high_risk = scores >= float(stage.risk_score_threshold)
    if np.any(high_risk):
        strict_required = np.maximum(
            required * float(stage.risk_margin_multiplier),
            float(stage.risk_uncertainty_band),
        )
        risky_uncertainty = np.any(
            high_risk.reshape(1, -1) & (margin < strict_required.reshape(1, -1)),
            axis=1,
        )
        stop &= ~risky_uncertainty
        weighted_uncertainty = scores.reshape(1, -1) * (
            1.0 - np.clip(margin / np.maximum(strict_required.reshape(1, -1), 1e-6), 0.0, 1.0)
        )
        stop &= weighted_uncertainty.max(axis=1) <= float(stage.max_label_risk)
    return stop


def strict_sequential_select(
    *,
    exit_probabilities: Sequence[np.ndarray],
    thresholds_by_exit: Sequence[Sequence[float] | np.ndarray],
    risk_scores_by_exit: np.ndarray,
    config: StrictSequentialPolicyConfig,
    minimum_exit: int = 1,
) -> dict[str, Any]:
    probabilities = [
        _matrix(item, f"exit{index + 1}_probabilities")
        for index, item in enumerate(exit_probabilities)
    ]
    if len(probabilities) != int(config.num_exits):
        raise ValueError("Policy exit count does not match supplied probabilities.")
    shape = probabilities[0].shape
    if any(item.shape != shape for item in probabilities):
        raise ValueError("All exit probability matrices must share shape.")
    if len(thresholds_by_exit) != len(probabilities):
        raise ValueError("A threshold vector is required for every exit.")
    risk_scores = np.asarray(risk_scores_by_exit, dtype=np.float32)
    if risk_scores.shape != (len(probabilities) - 1, shape[1]):
        raise ValueError("risk_scores_by_exit has unexpected shape.")

    selected_probabilities = probabilities[-1].copy()
    selected_exit = np.full(shape[0], len(probabilities), dtype=np.int8)
    alive = np.ones(shape[0], dtype=bool)
    previous: np.ndarray | None = None
    stop_masks: list[np.ndarray] = []

    for exit_index in range(len(probabilities) - 1):
        diagnostic = stage_diagnostics(
            current_probabilities=probabilities[exit_index],
            current_thresholds=thresholds_by_exit[exit_index],
            previous_probabilities=previous,
            previous_thresholds=None if exit_index == 0 else thresholds_by_exit[exit_index - 1],
        )
        if exit_index + 1 < int(minimum_exit):
            stop = np.zeros(shape[0], dtype=bool)
        else:
            stop = alive & strict_stage_stop_mask(
                diagnostic=diagnostic,
                stage=config.stages[exit_index],
                risk_scores=risk_scores[exit_index],
                exit_index=exit_index,
                allow_empty_stop=config.allow_empty_stop,
            )
        stop_masks.append(stop.copy())
        if np.any(stop):
            selected_probabilities[stop] = probabilities[exit_index][stop]
            selected_exit[stop] = exit_index + 1
            alive[stop] = False
        previous = probabilities[exit_index]

    return {
        "selected_probabilities": selected_probabilities,
        "selected_exit": selected_exit,
        "stage_stop_masks": stop_masks,
        "remaining_to_final": alive,
    }


__all__ = [
    "StrictSequentialStageConfig",
    "StrictSequentialPolicyConfig",
    "label_predictions",
    "genes_per_stage",
    "total_genes",
    "make_strict_bounds",
    "decode_strict_genes",
    "encode_strict_config",
    "derive_strict_continuation_profile",
    "stage_diagnostics",
    "strict_stage_stop_mask",
    "strict_sequential_select",
]
