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


def label_predictions(probabilities: np.ndarray, thresholds: Sequence[float] | np.ndarray) -> np.ndarray:
    probs = _matrix(probabilities, "probabilities")
    values = _vector(thresholds, probs.shape[1], "thresholds")
    return (probs >= values.reshape(1, -1)).astype(np.int8)


@dataclass(frozen=True)
class TargetedExit3ToExit5Config:
    mean_confidence_threshold: float
    max_probability_delta: float
    max_label_risk: float
    risk_score_threshold: float
    risk_margin_multiplier: float
    risk_uncertainty_band: float
    per_label_margins: tuple[float, ...]
    require_exit2_label_stability: bool = True
    allow_empty_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gene_count(num_labels: int) -> int:
    return int(num_labels) + 6


def make_targeted_bounds(
    *,
    num_labels: int,
    confidence_bounds: tuple[float, float] = (0.65, 0.995),
    delta_bounds: tuple[float, float] = (0.0, 0.50),
    max_label_risk_bounds: tuple[float, float] = (0.0, 0.50),
    risk_score_bounds: tuple[float, float] = (0.25, 0.95),
    risk_multiplier_bounds: tuple[float, float] = (1.0, 4.0),
    risk_band_bounds: tuple[float, float] = (0.01, 0.30),
    margin_bounds: tuple[float, float] = (0.0, 0.50),
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        confidence_bounds,
        delta_bounds,
        max_label_risk_bounds,
        risk_score_bounds,
        risk_multiplier_bounds,
        risk_band_bounds,
        margin_bounds,
    ]
    if any(float(low) >= float(high) for low, high in pairs):
        raise ValueError("Each targeted optimisation lower bound must be below its upper bound.")
    lower = np.asarray(
        [
            confidence_bounds[0],
            delta_bounds[0],
            max_label_risk_bounds[0],
            risk_score_bounds[0],
            risk_multiplier_bounds[0],
            risk_band_bounds[0],
            *([margin_bounds[0]] * int(num_labels)),
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            confidence_bounds[1],
            delta_bounds[1],
            max_label_risk_bounds[1],
            risk_score_bounds[1],
            risk_multiplier_bounds[1],
            risk_band_bounds[1],
            *([margin_bounds[1]] * int(num_labels)),
        ],
        dtype=np.float64,
    )
    return lower, upper


def decode_targeted_genes(
    genes: Sequence[float] | np.ndarray,
    *,
    num_labels: int,
) -> TargetedExit3ToExit5Config:
    values = _vector(genes, gene_count(num_labels), "genes")
    return TargetedExit3ToExit5Config(
        mean_confidence_threshold=float(values[0]),
        max_probability_delta=float(values[1]),
        max_label_risk=float(values[2]),
        risk_score_threshold=float(values[3]),
        risk_margin_multiplier=float(values[4]),
        risk_uncertainty_band=float(values[5]),
        per_label_margins=tuple(float(item) for item in values[6:]),
    )


def encode_targeted_config(config: TargetedExit3ToExit5Config) -> np.ndarray:
    return np.asarray(
        [
            config.mean_confidence_threshold,
            config.max_probability_delta,
            config.max_label_risk,
            config.risk_score_threshold,
            config.risk_margin_multiplier,
            config.risk_uncertainty_band,
            *config.per_label_margins,
        ],
        dtype=np.float64,
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = max(float(np.max(array)), 1e-12)
    return (array / maximum).astype(np.float32)


def derive_grouped_continuation_risk(
    *,
    y_true: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit5_probabilities: np.ndarray,
    exit3_thresholds: Sequence[float] | np.ndarray,
    exit5_thresholds: Sequence[float] | np.ndarray,
    group_ids: Sequence[str] | np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
    minimum_score: float = 0.05,
) -> dict[str, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int8)
    exit3 = _matrix(exit3_probabilities, "exit3_probabilities")
    exit5 = _matrix(exit5_probabilities, "exit5_probabilities")
    if truth.shape != exit3.shape or exit3.shape != exit5.shape:
        raise ValueError("Truth, Exit 3 and Exit 5 arrays must share shape.")
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    if len(groups) != len(truth):
        raise ValueError("group_ids must align with the segment rows.")
    pred3 = label_predictions(exit3, exit3_thresholds)
    pred5 = label_predictions(exit5, exit5_thresholds)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two parent groups are required.")
    fold_count = min(max(2, int(cv_folds)), len(unique))
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, fold_count)
    labels = truth.shape[1]
    fold_scores = np.zeros((fold_count, labels), dtype=np.float32)
    fold_correction_rates = np.zeros_like(fold_scores)
    fold_f1_gains = np.zeros_like(fold_scores)
    fold_fn_correction = np.zeros_like(fold_scores)
    fold_fp_correction = np.zeros_like(fold_scores)

    for fold_index, fold_groups in enumerate(folds):
        mask = np.isin(groups, fold_groups)
        y = truth[mask]
        p3 = pred3[mask]
        p5 = pred5[mask]
        errors3 = p3 != y
        corrected = errors3 & (p5 == y)
        correction_rate = corrected.sum(axis=0) / np.maximum(errors3.sum(axis=0), 1)
        fn3 = (p3 == 0) & (y == 1)
        fp3 = (p3 == 1) & (y == 0)
        fn_corrected = fn3 & (p5 == 1)
        fp_corrected = fp3 & (p5 == 0)
        fn_rate = fn_corrected.sum(axis=0) / np.maximum(fn3.sum(axis=0), 1)
        fp_rate = fp_corrected.sum(axis=0) / np.maximum(fp3.sum(axis=0), 1)
        f1_gain = np.asarray(
            [
                max(0.0, _binary_f1(y[:, label], p5[:, label]) - _binary_f1(y[:, label], p3[:, label]))
                for label in range(labels)
            ],
            dtype=np.float32,
        )
        fold_correction_rates[fold_index] = correction_rate
        fold_f1_gains[fold_index] = f1_gain
        fold_fn_correction[fold_index] = fn_rate
        fold_fp_correction[fold_index] = fp_rate
        fold_scores[fold_index] = (
            0.35 * _normalise(correction_rate)
            + 0.30 * _normalise(f1_gain)
            + 0.20 * _normalise(fn_rate)
            + 0.15 * _normalise(fp_rate)
        )

    support = truth.sum(axis=0).astype(np.float32)
    rarity = 1.0 - support / max(float(np.max(support)), 1.0)
    risk = np.maximum(float(minimum_score), np.max(fold_scores, axis=0) + 0.10 * rarity)
    risk = np.clip(risk, float(minimum_score), 1.0).astype(np.float32)
    return {
        "risk_scores": risk,
        "fold_scores": fold_scores,
        "fold_correction_rates": fold_correction_rates,
        "fold_f1_gains": fold_f1_gains,
        "fold_fn_correction_rates": fold_fn_correction,
        "fold_fp_correction_rates": fold_fp_correction,
        "label_support": support,
        "rarity_scores": rarity.astype(np.float32),
    }


def targeted_diagnostics(
    *,
    exit2_probabilities: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit2_thresholds: Sequence[float] | np.ndarray,
    exit3_thresholds: Sequence[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    exit2 = _matrix(exit2_probabilities, "exit2_probabilities")
    exit3 = _matrix(exit3_probabilities, "exit3_probabilities")
    if exit2.shape != exit3.shape:
        raise ValueError("Exit 2 and Exit 3 probabilities must share shape.")
    pred2 = label_predictions(exit2, exit2_thresholds)
    pred3 = label_predictions(exit3, exit3_thresholds)
    thresholds3 = _vector(exit3_thresholds, exit3.shape[1], "exit3_thresholds")
    margin = np.abs(exit3 - thresholds3.reshape(1, -1))
    confidence = np.maximum(exit3, 1.0 - exit3)
    return {
        "prediction": pred3,
        "non_empty": pred3.sum(axis=1) > 0,
        "mean_binary_confidence": confidence.mean(axis=1).astype(np.float32),
        "decision_margin": margin.astype(np.float32),
        "maximum_probability_delta": np.abs(exit3 - exit2).max(axis=1).astype(np.float32),
        "label_set_stability": np.all(pred2 == pred3, axis=1),
    }


def targeted_stop_mask(
    *,
    diagnostics: dict[str, np.ndarray],
    risk_scores: Sequence[float] | np.ndarray,
    config: TargetedExit3ToExit5Config,
) -> np.ndarray:
    margin = _matrix(diagnostics["decision_margin"], "decision_margin")
    labels = margin.shape[1]
    required = _vector(config.per_label_margins, labels, "per_label_margins")
    scores = _vector(risk_scores, labels, "risk_scores")
    prediction = np.asarray(diagnostics["prediction"], dtype=np.int8)
    stop = np.ones(len(margin), dtype=bool)
    if config.require_exit2_label_stability:
        stop &= np.asarray(diagnostics["label_set_stability"], dtype=bool)
    if not config.allow_empty_stop:
        stop &= np.asarray(diagnostics["non_empty"], dtype=bool)
    stop &= np.asarray(diagnostics["mean_binary_confidence"], dtype=float) >= float(
        config.mean_confidence_threshold
    )
    stop &= np.asarray(diagnostics["maximum_probability_delta"], dtype=float) <= float(
        config.max_probability_delta
    )
    stop &= np.all(margin >= required.reshape(1, -1), axis=1)

    high_risk = scores >= float(config.risk_score_threshold)
    if np.any(high_risk):
        strict_required = np.maximum(
            required * float(config.risk_margin_multiplier),
            float(config.risk_uncertainty_band),
        )
        sample_required = strict_required.reshape(1, -1) * (
            1.0 + 0.25 * prediction.astype(np.float64)
        )
        risky_uncertainty = np.any(
            high_risk.reshape(1, -1) & (margin < sample_required), axis=1
        )
        stop &= ~risky_uncertainty
        weighted_uncertainty = scores.reshape(1, -1) * (
            1.0 - np.clip(margin / np.maximum(sample_required, 1e-6), 0.0, 1.0)
        )
        stop &= weighted_uncertainty.max(axis=1) <= float(config.max_label_risk)
    return stop


def targeted_select(
    *,
    exit2_probabilities: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit5_probabilities: np.ndarray,
    exit2_thresholds: Sequence[float] | np.ndarray,
    exit3_thresholds: Sequence[float] | np.ndarray,
    risk_scores: Sequence[float] | np.ndarray,
    config: TargetedExit3ToExit5Config,
) -> dict[str, np.ndarray]:
    exit3 = _matrix(exit3_probabilities, "exit3_probabilities")
    exit5 = _matrix(exit5_probabilities, "exit5_probabilities")
    if exit3.shape != exit5.shape:
        raise ValueError("Exit 3 and Exit 5 probabilities must share shape.")
    diagnostics = targeted_diagnostics(
        exit2_probabilities=exit2_probabilities,
        exit3_probabilities=exit3,
        exit2_thresholds=exit2_thresholds,
        exit3_thresholds=exit3_thresholds,
    )
    stop = targeted_stop_mask(diagnostics=diagnostics, risk_scores=risk_scores, config=config)
    selected = exit5.copy()
    selected[stop] = exit3[stop]
    selected_exit = np.where(stop, 3, 5).astype(np.int8)
    return {
        "selected_probabilities": selected,
        "selected_exit": selected_exit,
        "exit3_stop_mask": stop,
        "diagnostics": diagnostics,
    }


__all__ = [
    "TargetedExit3ToExit5Config",
    "gene_count",
    "make_targeted_bounds",
    "decode_targeted_genes",
    "encode_targeted_config",
    "derive_grouped_continuation_risk",
    "targeted_diagnostics",
    "targeted_stop_mask",
    "targeted_select",
]
