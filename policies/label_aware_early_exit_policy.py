# policies/label_aware_early_exit_policy.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


def _as_probability_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, labels], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return array


def _as_threshold_vector(
    value: np.ndarray | Sequence[float],
    *,
    num_labels: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (num_labels,):
        raise ValueError(
            f"{name} must contain {num_labels} values, got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain thresholds in [0, 1].")
    return array


def _as_risk_vector(
    value: np.ndarray | Sequence[float],
    *,
    num_labels: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (num_labels,):
        raise ValueError(
            f"risk_weights must contain {num_labels} values, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("risk_weights contains NaN or infinite values.")
    if np.any(array < 0.0):
        raise ValueError("risk_weights must be non-negative.")
    return array


def label_predictions(
    probabilities: np.ndarray,
    thresholds: np.ndarray | Sequence[float],
) -> np.ndarray:
    probs = _as_probability_matrix(probabilities, "probabilities")
    threshold_vector = _as_threshold_vector(
        thresholds,
        num_labels=probs.shape[1],
        name="thresholds",
    )
    return (probs >= threshold_vector.reshape(1, -1)).astype(np.int8)


def binary_f1_per_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(y_pred, dtype=np.int8)
    if truth.ndim != 2 or pred.ndim != 2 or truth.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same [samples, labels] shape; "
            f"got {truth.shape} and {pred.shape}."
        )

    tp = np.sum((truth == 1) & (pred == 1), axis=0, dtype=np.float64)
    fp = np.sum((truth == 0) & (pred == 1), axis=0, dtype=np.float64)
    fn = np.sum((truth == 1) & (pred == 0), axis=0, dtype=np.float64)
    denominator = 2.0 * tp + fp + fn
    return np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0.0,
    ).astype(np.float32)


@dataclass(frozen=True)
class LabelRiskProfile:
    """Validation-derived estimate of which labels benefit from deeper inference."""

    labels: tuple[str, ...]
    exit2_f1: tuple[float, ...]
    exit3_f1: tuple[float, ...]
    improvement: tuple[float, ...]
    risk_weights: tuple[float, ...]
    minimum_improvement: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelRiskProfile":
        return cls(
            labels=tuple(str(item) for item in payload["labels"]),
            exit2_f1=tuple(float(item) for item in payload["exit2_f1"]),
            exit3_f1=tuple(float(item) for item in payload["exit3_f1"]),
            improvement=tuple(float(item) for item in payload["improvement"]),
            risk_weights=tuple(float(item) for item in payload["risk_weights"]),
            minimum_improvement=float(payload["minimum_improvement"]),
        )


def derive_label_risk_profile(
    *,
    labels: Sequence[str],
    y_true: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit2_thresholds: np.ndarray | Sequence[float],
    exit3_thresholds: np.ndarray | Sequence[float],
    minimum_improvement: float = 0.0,
) -> LabelRiskProfile:
    """Derive label-risk weights from validation-only Exit-2 to Exit-3 gains.

    A label receives a larger weight when its validation F1 improves more at
    Exit 3 than at Exit 2. Labels whose improvement is below
    ``minimum_improvement`` receive zero weight. The largest retained
    improvement is normalized to 1.0.
    """

    label_names = tuple(str(label) for label in labels)
    truth = np.asarray(y_true, dtype=np.int8)
    exit2_probs = _as_probability_matrix(exit2_probabilities, "exit2_probabilities")
    exit3_probs = _as_probability_matrix(exit3_probabilities, "exit3_probabilities")

    if exit2_probs.shape != exit3_probs.shape:
        raise ValueError(
            "Exit-2 and Exit-3 probability matrices must have identical shape; "
            f"got {exit2_probs.shape} and {exit3_probs.shape}."
        )
    if truth.shape != exit2_probs.shape:
        raise ValueError(
            "y_true must match probability shape; "
            f"got {truth.shape} and {exit2_probs.shape}."
        )
    if len(label_names) != exit2_probs.shape[1]:
        raise ValueError(
            f"labels contains {len(label_names)} names but probabilities have "
            f"{exit2_probs.shape[1]} columns."
        )
    if minimum_improvement < 0.0:
        raise ValueError("minimum_improvement must be non-negative.")

    exit2_pred = label_predictions(exit2_probs, exit2_thresholds)
    exit3_pred = label_predictions(exit3_probs, exit3_thresholds)
    exit2_f1 = binary_f1_per_label(truth, exit2_pred)
    exit3_f1 = binary_f1_per_label(truth, exit3_pred)

    improvement = np.maximum(exit3_f1 - exit2_f1, 0.0)
    retained = np.where(
        improvement >= float(minimum_improvement),
        improvement,
        0.0,
    )
    max_gain = float(np.max(retained)) if retained.size else 0.0
    if max_gain > 0.0:
        risk_weights = retained / max_gain
    else:
        risk_weights = np.zeros_like(retained, dtype=np.float32)

    return LabelRiskProfile(
        labels=label_names,
        exit2_f1=tuple(float(item) for item in exit2_f1),
        exit3_f1=tuple(float(item) for item in exit3_f1),
        improvement=tuple(float(item) for item in improvement),
        risk_weights=tuple(float(item) for item in risk_weights),
        minimum_improvement=float(minimum_improvement),
    )


@dataclass(frozen=True)
class LabelAwarePolicyConfig:
    """Frozen rule thresholds for the label-aware Exit-2 stopping policy."""

    mean_confidence_threshold: float = 0.55
    global_margin_threshold: float = 0.0
    max_probability_delta: float = 1.0
    label_risk_threshold: float = 1.0
    margin_scale: float = 0.25
    margin_weight: float = 0.5
    delta_weight: float = 0.5
    require_label_set_agreement: bool = True
    allow_empty_stop: bool = False

    def __post_init__(self) -> None:
        bounded = {
            "mean_confidence_threshold": self.mean_confidence_threshold,
            "global_margin_threshold": self.global_margin_threshold,
            "max_probability_delta": self.max_probability_delta,
            "label_risk_threshold": self.label_risk_threshold,
            "margin_weight": self.margin_weight,
            "delta_weight": self.delta_weight,
        }
        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.margin_scale <= 0.0:
            raise ValueError("margin_scale must be greater than zero.")
        if self.margin_weight + self.delta_weight <= 0.0:
            raise ValueError("At least one risk component weight must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelAwarePolicyConfig":
        return cls(**payload)


def compute_label_aware_diagnostics(
    *,
    exit1_probabilities: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit1_thresholds: np.ndarray | Sequence[float],
    exit2_thresholds: np.ndarray | Sequence[float],
    risk_weights: np.ndarray | Sequence[float],
    margin_scale: float,
    margin_weight: float = 0.5,
    delta_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute sample-wise stopping diagnostics using validation-derived label risk."""

    exit1_probs = _as_probability_matrix(exit1_probabilities, "exit1_probabilities")
    exit2_probs = _as_probability_matrix(exit2_probabilities, "exit2_probabilities")
    if exit1_probs.shape != exit2_probs.shape:
        raise ValueError(
            "Exit-1 and Exit-2 probability matrices must have identical shape; "
            f"got {exit1_probs.shape} and {exit2_probs.shape}."
        )
    if margin_scale <= 0.0:
        raise ValueError("margin_scale must be greater than zero.")
    if margin_weight < 0.0 or delta_weight < 0.0:
        raise ValueError("Risk component weights must be non-negative.")
    component_total = float(margin_weight + delta_weight)
    if component_total <= 0.0:
        raise ValueError("At least one risk component weight must be positive.")

    num_labels = exit1_probs.shape[1]
    threshold1 = _as_threshold_vector(
        exit1_thresholds,
        num_labels=num_labels,
        name="exit1_thresholds",
    )
    threshold2 = _as_threshold_vector(
        exit2_thresholds,
        num_labels=num_labels,
        name="exit2_thresholds",
    )
    weights = _as_risk_vector(risk_weights, num_labels=num_labels)

    exit1_pred = label_predictions(exit1_probs, threshold1)
    exit2_pred = label_predictions(exit2_probs, threshold2)
    absolute_delta = np.abs(exit2_probs - exit1_probs)
    decision_margin = np.abs(exit2_probs - threshold2.reshape(1, -1))

    margin_uncertainty = 1.0 - np.clip(
        decision_margin / float(margin_scale),
        0.0,
        1.0,
    )
    normalized_margin_weight = float(margin_weight) / component_total
    normalized_delta_weight = float(delta_weight) / component_total
    per_label_risk = weights.reshape(1, -1) * (
        normalized_margin_weight * margin_uncertainty
        + normalized_delta_weight * absolute_delta
    )

    return {
        "exit1_pred": exit1_pred,
        "exit2_pred": exit2_pred,
        "label_set_agreement": np.all(exit1_pred == exit2_pred, axis=1),
        "non_empty": exit2_pred.sum(axis=1) > 0,
        "mean_binary_confidence": np.maximum(
            exit2_probs,
            1.0 - exit2_probs,
        ).mean(axis=1),
        "minimum_decision_margin": np.min(decision_margin, axis=1),
        "maximum_probability_delta": np.max(absolute_delta, axis=1),
        "maximum_label_risk": np.max(per_label_risk, axis=1),
        "per_label_risk": per_label_risk,
        "decision_margin": decision_margin,
        "absolute_probability_delta": absolute_delta,
    }


def label_aware_stop_mask(
    diagnostics: dict[str, np.ndarray],
    config: LabelAwarePolicyConfig,
) -> np.ndarray:
    """Return True for samples that may safely stop at Exit 2."""

    required = {
        "label_set_agreement",
        "non_empty",
        "mean_binary_confidence",
        "minimum_decision_margin",
        "maximum_probability_delta",
        "maximum_label_risk",
    }
    missing = sorted(required.difference(diagnostics))
    if missing:
        raise KeyError(f"Diagnostics are missing required keys: {missing}")

    sample_count = len(np.asarray(diagnostics["label_set_agreement"]))
    mask = np.ones(sample_count, dtype=bool)

    if config.require_label_set_agreement:
        mask &= np.asarray(diagnostics["label_set_agreement"], dtype=bool)
    if not config.allow_empty_stop:
        mask &= np.asarray(diagnostics["non_empty"], dtype=bool)

    mask &= (
        np.asarray(diagnostics["mean_binary_confidence"], dtype=np.float32)
        >= float(config.mean_confidence_threshold)
    )
    mask &= (
        np.asarray(diagnostics["minimum_decision_margin"], dtype=np.float32)
        >= float(config.global_margin_threshold)
    )
    mask &= (
        np.asarray(diagnostics["maximum_probability_delta"], dtype=np.float32)
        <= float(config.max_probability_delta)
    )
    mask &= (
        np.asarray(diagnostics["maximum_label_risk"], dtype=np.float32)
        <= float(config.label_risk_threshold)
    )
    return mask
