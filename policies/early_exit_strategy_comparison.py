from __future__ import annotations

from dataclasses import asdict, dataclass
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
    thresholds: Sequence[float] | np.ndarray,
) -> np.ndarray:
    probs = _matrix(probabilities, "probabilities")
    thr = _vector(thresholds, probs.shape[1], "thresholds")
    return (probs >= thr.reshape(1, -1)).astype(np.int8)


def split_parent_ids(
    parent_ids: Sequence[Any],
    *,
    derivation_fraction: float = 0.70,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < float(derivation_fraction) < 1.0:
        raise ValueError("derivation_fraction must be strictly between 0 and 1.")
    ids = np.asarray([str(item) for item in parent_ids], dtype=object)
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("At least two unique parents are required.")
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * float(derivation_fraction)))
    cut = min(max(cut, 1), len(shuffled) - 1)
    derivation_parents = set(shuffled[:cut].tolist())
    derivation_mask = np.asarray(
        [item in derivation_parents for item in ids],
        dtype=bool,
    )
    selection_mask = ~derivation_mask
    if not derivation_mask.any() or not selection_mask.any():
        raise RuntimeError("Parent split produced an empty subset.")
    return derivation_mask, selection_mask


def compute_common_diagnostics(
    *,
    exit1_probabilities: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit1_thresholds: Sequence[float] | np.ndarray,
    exit2_thresholds: Sequence[float] | np.ndarray,
    risk_weights: Sequence[float] | np.ndarray | None = None,
    risk_margin_scale: float = 0.25,
    risk_margin_weight: float = 0.5,
    risk_delta_weight: float = 0.5,
) -> dict[str, np.ndarray]:
    p1 = _matrix(exit1_probabilities, "exit1_probabilities")
    p2 = _matrix(exit2_probabilities, "exit2_probabilities")
    if p1.shape != p2.shape:
        raise ValueError(
            f"Exit probability shapes differ: {p1.shape} vs {p2.shape}."
        )
    if risk_margin_scale <= 0.0:
        raise ValueError("risk_margin_scale must be greater than zero.")
    if risk_margin_weight < 0.0 or risk_delta_weight < 0.0:
        raise ValueError("Risk component weights must be non-negative.")
    component_total = float(risk_margin_weight + risk_delta_weight)
    if component_total <= 0.0:
        raise ValueError("At least one risk component weight must be positive.")

    t1 = _vector(exit1_thresholds, p1.shape[1], "exit1_thresholds")
    t2 = _vector(exit2_thresholds, p2.shape[1], "exit2_thresholds")
    pred1 = label_predictions(p1, t1)
    pred2 = label_predictions(p2, t2)
    delta = np.abs(p2 - p1)
    margin = np.abs(p2 - t2.reshape(1, -1))
    confidence = np.maximum(p2, 1.0 - p2)

    if risk_weights is None:
        weights = np.zeros(p1.shape[1], dtype=np.float32)
    else:
        weights = _vector(risk_weights, p1.shape[1], "risk_weights")
        if np.any(weights < 0.0):
            raise ValueError("risk_weights must be non-negative.")

    margin_uncertainty = 1.0 - np.clip(
        margin / float(risk_margin_scale),
        0.0,
        1.0,
    )
    mw = float(risk_margin_weight) / component_total
    dw = float(risk_delta_weight) / component_total
    per_label_risk = weights.reshape(1, -1) * (
        mw * margin_uncertainty + dw * delta
    )

    return {
        "exit1_pred": pred1,
        "exit2_pred": pred2,
        "label_set_agreement": np.all(pred1 == pred2, axis=1),
        "non_empty": pred2.sum(axis=1) > 0,
        "mean_binary_confidence": confidence.mean(axis=1),
        "minimum_decision_margin": margin.min(axis=1),
        "maximum_probability_delta": delta.max(axis=1),
        "predicted_label_count": pred2.sum(axis=1).astype(np.float32),
        "decision_margin": margin,
        "absolute_probability_delta": delta,
        "per_label_risk": per_label_risk,
        "maximum_label_risk": per_label_risk.max(axis=1),
        "highest_risk_label_index": per_label_risk.argmax(axis=1).astype(
            np.int64
        ),
    }


@dataclass(frozen=True)
class GlobalRuleConfig:
    mean_confidence_threshold: float
    global_margin_threshold: float
    max_probability_delta: float = 1.0
    require_label_set_agreement: bool = True
    allow_empty_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def global_rule_stop_mask(
    diagnostics: dict[str, np.ndarray],
    config: GlobalRuleConfig,
) -> np.ndarray:
    mask = np.ones(len(diagnostics["label_set_agreement"]), dtype=bool)
    if config.require_label_set_agreement:
        mask &= np.asarray(diagnostics["label_set_agreement"], dtype=bool)
    if not config.allow_empty_stop:
        mask &= np.asarray(diagnostics["non_empty"], dtype=bool)
    mask &= (
        np.asarray(diagnostics["mean_binary_confidence"])
        >= float(config.mean_confidence_threshold)
    )
    mask &= (
        np.asarray(diagnostics["minimum_decision_margin"])
        >= float(config.global_margin_threshold)
    )
    mask &= (
        np.asarray(diagnostics["maximum_probability_delta"])
        <= float(config.max_probability_delta)
    )
    return mask


@dataclass(frozen=True)
class LabelRiskRuleConfig(GlobalRuleConfig):
    label_risk_threshold: float = 1.0


def label_risk_stop_mask(
    diagnostics: dict[str, np.ndarray],
    config: LabelRiskRuleConfig,
) -> np.ndarray:
    mask = global_rule_stop_mask(diagnostics, config)
    mask &= (
        np.asarray(diagnostics["maximum_label_risk"])
        <= float(config.label_risk_threshold)
    )
    return mask


@dataclass(frozen=True)
class PerLabelMarginConfig:
    mean_confidence_threshold: float
    per_label_margins: tuple[float, ...]
    require_label_set_agreement: bool = True
    allow_empty_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def per_label_margin_stop_mask(
    diagnostics: dict[str, np.ndarray],
    config: PerLabelMarginConfig,
) -> np.ndarray:
    margins = np.asarray(diagnostics["decision_margin"], dtype=np.float32)
    required = _vector(
        config.per_label_margins,
        margins.shape[1],
        "per_label_margins",
    )
    mask = np.ones(len(margins), dtype=bool)
    if config.require_label_set_agreement:
        mask &= np.asarray(diagnostics["label_set_agreement"], dtype=bool)
    if not config.allow_empty_stop:
        mask &= np.asarray(diagnostics["non_empty"], dtype=bool)
    mask &= (
        np.asarray(diagnostics["mean_binary_confidence"])
        >= float(config.mean_confidence_threshold)
    )
    mask &= np.all(margins >= required.reshape(1, -1), axis=1)
    return mask


def derive_per_label_margin_thresholds(
    *,
    y_true: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit2_thresholds: Sequence[float] | np.ndarray,
    exit3_thresholds: Sequence[float] | np.ndarray,
    capture_fraction: float,
    minimum_corrected_examples: int = 3,
    maximum_margin: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= float(capture_fraction) <= 1.0:
        raise ValueError("capture_fraction must be in [0, 1].")
    if minimum_corrected_examples < 1:
        raise ValueError("minimum_corrected_examples must be at least 1.")
    p2 = _matrix(exit2_probabilities, "exit2_probabilities")
    p3 = _matrix(exit3_probabilities, "exit3_probabilities")
    truth = np.asarray(y_true, dtype=np.int8)
    if p2.shape != p3.shape or truth.shape != p2.shape:
        raise ValueError(
            "Ground truth and Exit-2/Exit-3 probabilities must share shape."
        )
    t2 = _vector(exit2_thresholds, p2.shape[1], "exit2_thresholds")
    pred2 = label_predictions(p2, t2)
    pred3 = label_predictions(p3, exit3_thresholds)
    corrected = (pred2 != truth) & (pred3 == truth)
    margins = np.abs(p2 - t2.reshape(1, -1))
    result = np.zeros(p2.shape[1], dtype=np.float32)
    counts = corrected.sum(axis=0).astype(np.int64)
    for label_idx in range(p2.shape[1]):
        values = margins[corrected[:, label_idx], label_idx]
        if len(values) >= int(minimum_corrected_examples):
            result[label_idx] = float(
                np.quantile(values, float(capture_fraction))
            )
    result = np.clip(result, 0.0, float(maximum_margin)).astype(np.float32)
    return result, counts


def build_gate_features(
    *,
    exit1_probabilities: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit1_thresholds: Sequence[float] | np.ndarray,
    exit2_thresholds: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    p1 = _matrix(exit1_probabilities, "exit1_probabilities")
    p2 = _matrix(exit2_probabilities, "exit2_probabilities")
    if p1.shape != p2.shape:
        raise ValueError("Exit-1 and Exit-2 probabilities must share shape.")
    t1 = _vector(exit1_thresholds, p1.shape[1], "exit1_thresholds")
    t2 = _vector(exit2_thresholds, p2.shape[1], "exit2_thresholds")
    delta = np.abs(p2 - p1)
    margin = np.abs(p2 - t2.reshape(1, -1))
    pred1 = label_predictions(p1, t1)
    pred2 = label_predictions(p2, t2)
    confidence = np.maximum(p2, 1.0 - p2)
    clipped = np.clip(p2, 1e-7, 1.0 - 1e-7)
    entropy = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )

    parts = [p1, p2, delta, margin]
    names = (
        [f"exit1_prob_{idx}" for idx in range(p1.shape[1])]
        + [f"exit2_prob_{idx}" for idx in range(p1.shape[1])]
        + [f"abs_delta_{idx}" for idx in range(p1.shape[1])]
        + [f"margin_{idx}" for idx in range(p1.shape[1])]
    )
    scalar = np.column_stack(
        [
            confidence.mean(axis=1),
            confidence.min(axis=1),
            entropy.mean(axis=1),
            margin.min(axis=1),
            delta.max(axis=1),
            np.all(pred1 == pred2, axis=1).astype(np.float32),
            (pred2.sum(axis=1) > 0).astype(np.float32),
            pred2.sum(axis=1).astype(np.float32),
        ]
    ).astype(np.float32)
    scalar_names = [
        "mean_binary_confidence",
        "minimum_binary_confidence",
        "mean_binary_entropy",
        "minimum_decision_margin",
        "maximum_probability_delta",
        "label_set_agreement",
        "non_empty",
        "predicted_label_count",
    ]
    features = np.concatenate([*parts, scalar], axis=1).astype(np.float32)
    return features, names + scalar_names


def gate_safe_targets(
    *,
    y_true: np.ndarray,
    exit2_predictions: np.ndarray,
    exit3_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int8)
    p2 = np.asarray(exit2_predictions, dtype=np.int8)
    p3 = np.asarray(exit3_predictions, dtype=np.int8)
    if truth.shape != p2.shape or truth.shape != p3.shape:
        raise ValueError("Ground truth and predictions must share shape.")
    errors2 = np.sum(p2 != truth, axis=1).astype(np.int16)
    errors3 = np.sum(p3 != truth, axis=1).astype(np.int16)
    improvement = errors2 - errors3
    safe = (improvement <= 0).astype(np.int8)
    return safe, improvement


def logistic_gate_stop_mask(
    *,
    safe_probabilities: np.ndarray,
    threshold: float,
    diagnostics: dict[str, np.ndarray],
    allow_empty_stop: bool = False,
) -> np.ndarray:
    probs = np.asarray(safe_probabilities, dtype=np.float32).reshape(-1)
    if len(probs) != len(diagnostics["non_empty"]):
        raise ValueError(
            "Gate probabilities and diagnostics have different lengths."
        )
    mask = probs >= float(threshold)
    if not allow_empty_stop:
        mask &= np.asarray(diagnostics["non_empty"], dtype=bool)
    return mask


def continuation_reasons(
    *,
    method: str,
    diagnostics: dict[str, np.ndarray],
    config: dict[str, Any],
    stop_mask: np.ndarray,
    gate_safe_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    n = len(stop_mask)
    reasons = np.full(n, "stopped_at_exit2", dtype=object)
    for idx in np.flatnonzero(~np.asarray(stop_mask, dtype=bool)):
        failed: list[str] = []
        if (
            not bool(config.get("allow_empty_stop", False))
            and not bool(diagnostics["non_empty"][idx])
        ):
            failed.append("empty_prediction")
        if (
            bool(config.get("require_label_set_agreement", False))
            and not bool(diagnostics["label_set_agreement"][idx])
        ):
            failed.append("label_set_disagreement")
        if (
            "mean_confidence_threshold" in config
            and float(diagnostics["mean_binary_confidence"][idx])
            < float(config["mean_confidence_threshold"])
        ):
            failed.append("low_mean_confidence")
        if (
            "global_margin_threshold" in config
            and float(diagnostics["minimum_decision_margin"][idx])
            < float(config["global_margin_threshold"])
        ):
            failed.append("low_global_margin")
        if (
            "max_probability_delta" in config
            and float(diagnostics["maximum_probability_delta"][idx])
            > float(config["max_probability_delta"])
        ):
            failed.append("large_probability_change")
        if (
            "label_risk_threshold" in config
            and float(diagnostics["maximum_label_risk"][idx])
            > float(config["label_risk_threshold"])
        ):
            failed.append("high_label_risk")
        if "per_label_margins" in config:
            required = np.asarray(
                config["per_label_margins"],
                dtype=np.float32,
            )
            if np.any(
                np.asarray(diagnostics["decision_margin"])[idx] < required
            ):
                failed.append("per_label_margin_not_met")
        if method == "logistic_gate" and gate_safe_probabilities is not None:
            if float(gate_safe_probabilities[idx]) < float(
                config["gate_probability_threshold"]
            ):
                failed.append("gate_predicts_exit3_useful")
        reasons[idx] = (
            ";".join(failed) if failed else "policy_forced_continue"
        )
    return reasons
