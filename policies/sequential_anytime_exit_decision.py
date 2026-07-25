from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from policies.sequential_anytime_exit_types import SequentialPolicyConfig

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


def label_predictions(
    probabilities: np.ndarray, thresholds: Sequence[float] | np.ndarray
) -> np.ndarray:
    probs = _matrix(probabilities, "probabilities")
    threshold_array = _vector(thresholds, probs.shape[1], "thresholds")
    return (probs >= threshold_array.reshape(1, -1)).astype(np.int8)


def derive_validation_risk_weights(
    *,
    y_true: np.ndarray,
    exit_probabilities: Sequence[np.ndarray],
    thresholds_by_exit: Sequence[Sequence[float] | np.ndarray],
    minimum_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = [_matrix(item, f"exit{idx + 1}_probabilities") for idx, item in enumerate(exit_probabilities)]
    if len(probabilities) < 2:
        raise ValueError("At least two exits are required.")
    shape = probabilities[0].shape
    if any(item.shape != shape for item in probabilities):
        raise ValueError("All exits must share the same probability shape.")
    truth = np.asarray(y_true, dtype=np.int8)
    if truth.shape != shape:
        raise ValueError("y_true and exit probabilities must share shape.")
    if len(thresholds_by_exit) != len(probabilities):
        raise ValueError("A threshold vector is required for every exit.")

    predictions = [
        label_predictions(probabilities[idx], thresholds_by_exit[idx])
        for idx in range(len(probabilities))
    ]
    final_prediction = predictions[-1]
    counts = np.zeros((len(probabilities) - 1, shape[1]), dtype=np.int64)
    weights = np.zeros_like(counts, dtype=np.float32)
    for exit_index in range(len(probabilities) - 1):
        corrected = (predictions[exit_index] != truth) & (final_prediction == truth)
        counts[exit_index] = corrected.sum(axis=0)
        maximum = max(int(counts[exit_index].max()), 1)
        weights[exit_index] = np.maximum(
            float(minimum_weight), counts[exit_index].astype(np.float32) / float(maximum)
        )
    return weights, counts


def stage_diagnostics(
    *,
    current_probabilities: np.ndarray,
    current_thresholds: Sequence[float] | np.ndarray,
    risk_weights: Sequence[float] | np.ndarray,
    previous_probabilities: np.ndarray | None = None,
    previous_thresholds: Sequence[float] | np.ndarray | None = None,
    risk_margin_scale: float = 0.25,
) -> dict[str, np.ndarray]:
    current = _matrix(current_probabilities, "current_probabilities")
    thresholds = _vector(current_thresholds, current.shape[1], "current_thresholds")
    weights = _vector(risk_weights, current.shape[1], "risk_weights")
    if np.any(weights < 0.0):
        raise ValueError("risk_weights must be non-negative.")
    if float(risk_margin_scale) <= 0.0:
        raise ValueError("risk_margin_scale must be positive.")

    current_prediction = label_predictions(current, thresholds)
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
        stable = np.all(previous_prediction == current_prediction, axis=1)

    margin_uncertainty = 1.0 - np.clip(
        margin / float(risk_margin_scale), 0.0, 1.0
    )
    per_label_risk = weights.reshape(1, -1) * (
        0.60 * margin_uncertainty + 0.40 * delta
    )
    return {
        "prediction": current_prediction,
        "non_empty": current_prediction.sum(axis=1) > 0,
        "mean_binary_confidence": confidence.mean(axis=1).astype(np.float32),
        "decision_margin": margin.astype(np.float32),
        "maximum_probability_delta": delta.max(axis=1).astype(np.float32),
        "label_set_stability": stable,
        "per_label_risk": per_label_risk.astype(np.float32),
        "maximum_label_risk": per_label_risk.max(axis=1).astype(np.float32),
    }


def sequential_select(
    *,
    exit_probabilities: Sequence[np.ndarray],
    thresholds_by_exit: Sequence[Sequence[float] | np.ndarray],
    risk_weights_by_exit: np.ndarray,
    config: SequentialPolicyConfig,
    minimum_exit: int = 1,
) -> dict[str, Any]:
    probabilities = [_matrix(item, f"exit{idx + 1}_probabilities") for idx, item in enumerate(exit_probabilities)]
    if len(probabilities) != int(config.num_exits):
        raise ValueError("Policy exit count does not match supplied probabilities.")
    if len(thresholds_by_exit) != len(probabilities):
        raise ValueError("A threshold vector is required for every exit.")
    shape = probabilities[0].shape
    if any(item.shape != shape for item in probabilities):
        raise ValueError("All exit probability matrices must share shape.")
    risk_weights = np.asarray(risk_weights_by_exit, dtype=np.float32)
    if risk_weights.shape != (len(probabilities) - 1, shape[1]):
        raise ValueError("risk_weights_by_exit has unexpected shape.")
    if int(minimum_exit) < 1 or int(minimum_exit) > len(probabilities):
        raise ValueError("minimum_exit is outside the available exit range.")

    selected_probabilities = probabilities[-1].copy()
    selected_exit = np.full(shape[0], len(probabilities), dtype=np.int8)
    alive = np.ones(shape[0], dtype=bool)
    diagnostics: list[dict[str, np.ndarray]] = []
    stop_masks: list[np.ndarray] = []

    for exit_index in range(len(probabilities) - 1):
        current_exit = exit_index + 1
        diagnostic = stage_diagnostics(
            current_probabilities=probabilities[exit_index],
            current_thresholds=thresholds_by_exit[exit_index],
            risk_weights=risk_weights[exit_index],
            previous_probabilities=(None if exit_index == 0 else probabilities[exit_index - 1]),
            previous_thresholds=(None if exit_index == 0 else thresholds_by_exit[exit_index - 1]),
        )
        diagnostics.append(diagnostic)
        stage = config.stages[exit_index]
        stop = alive.copy()
        if current_exit < int(minimum_exit):
            stop[:] = False
        else:
            if stage.require_previous_label_stability and exit_index > 0:
                stop &= diagnostic["label_set_stability"]
            if not config.allow_empty_stop:
                stop &= diagnostic["non_empty"]
            stop &= diagnostic["mean_binary_confidence"] >= float(
                stage.mean_confidence_threshold
            )
            stop &= diagnostic["maximum_probability_delta"] <= float(
                stage.max_probability_delta
            )
            required = _vector(
                stage.per_label_margins, shape[1], "per_label_margins"
            )
            stop &= np.all(
                diagnostic["decision_margin"] >= required.reshape(1, -1), axis=1
            )
            stop &= diagnostic["maximum_label_risk"] <= float(stage.max_label_risk)
        stop_masks.append(stop.copy())
        if np.any(stop):
            selected_probabilities[stop] = probabilities[exit_index][stop]
            selected_exit[stop] = current_exit
            alive[stop] = False

    return {
        "selected_probabilities": selected_probabilities,
        "selected_exit": selected_exit,
        "stage_stop_masks": stop_masks,
        "stage_diagnostics": diagnostics,
        "remaining_to_final": alive,
    }



__all__ = ["label_predictions", "derive_validation_risk_weights", "stage_diagnostics", "sequential_select"]
