from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from policies.strict_sequential_anytime_exit_v018 import (
    StrictSequentialPolicyConfig,
    StrictSequentialStageConfig,
    stage_diagnostics,
    strict_stage_stop_mask,
)


@dataclass(frozen=True)
class Exit3ToExit5Config:
    mean_confidence_threshold: float
    max_probability_delta: float
    max_label_risk: float
    risk_score_threshold: float
    risk_margin_multiplier: float
    risk_uncertainty_band: float
    per_label_margins: tuple[float, ...]
    require_exit2_exit3_stability: bool = True
    allow_empty_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gene_count(num_labels: int) -> int:
    return int(num_labels) + 6


def decode_genes(genes: Sequence[float] | np.ndarray, num_labels: int) -> Exit3ToExit5Config:
    values = np.asarray(genes, dtype=np.float64).reshape(-1)
    expected = gene_count(num_labels)
    if values.shape != (expected,):
        raise ValueError(f"Expected {expected} genes, got {values.shape}.")
    return Exit3ToExit5Config(
        mean_confidence_threshold=float(values[0]),
        max_probability_delta=float(values[1]),
        max_label_risk=float(values[2]),
        risk_score_threshold=float(values[3]),
        risk_margin_multiplier=float(values[4]),
        risk_uncertainty_band=float(values[5]),
        per_label_margins=tuple(float(v) for v in values[6:]),
    )


def bounds(num_labels: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([0.60, 0.00, 0.00, 0.30, 1.00, 0.01, *([0.00] * num_labels)], dtype=np.float64)
    upper = np.asarray([0.995, 0.60, 1.00, 0.98, 4.00, 0.30, *([0.60] * num_labels)], dtype=np.float64)
    return lower, upper


def select_exit3_or_exit5(
    *,
    exit2_probabilities: np.ndarray,
    exit3_probabilities: np.ndarray,
    exit5_probabilities: np.ndarray,
    thresholds_by_exit: Sequence[np.ndarray],
    exit3_risk_scores: np.ndarray,
    config: Exit3ToExit5Config,
) -> dict[str, np.ndarray]:
    if len(thresholds_by_exit) != 5:
        raise ValueError("Five threshold vectors are required.")
    diagnostic = stage_diagnostics(
        current_probabilities=exit3_probabilities,
        current_thresholds=thresholds_by_exit[2],
        previous_probabilities=exit2_probabilities,
        previous_thresholds=thresholds_by_exit[1],
    )
    stage = StrictSequentialStageConfig(
        mean_confidence_threshold=config.mean_confidence_threshold,
        max_probability_delta=config.max_probability_delta,
        max_label_risk=config.max_label_risk,
        risk_score_threshold=config.risk_score_threshold,
        risk_margin_multiplier=config.risk_margin_multiplier,
        risk_uncertainty_band=config.risk_uncertainty_band,
        exit1_confidence_boost=0.0,
        per_label_margins=config.per_label_margins,
        require_previous_label_stability=config.require_exit2_exit3_stability,
    )
    stop = strict_stage_stop_mask(
        diagnostic=diagnostic,
        stage=stage,
        risk_scores=exit3_risk_scores,
        exit_index=2,
        allow_empty_stop=config.allow_empty_stop,
    )
    selected_exit = np.where(stop, 3, 5).astype(np.int8)
    selected_probabilities = np.where(stop[:, None], exit3_probabilities, exit5_probabilities).astype(np.float32)
    return {
        "selected_exit": selected_exit,
        "selected_probabilities": selected_probabilities,
        "stop_at_exit3": stop,
    }


def to_v018_compatible_policy(config: Exit3ToExit5Config) -> StrictSequentialPolicyConfig:
    disabled = StrictSequentialStageConfig(
        mean_confidence_threshold=1.10,
        max_probability_delta=0.0,
        max_label_risk=0.0,
        risk_score_threshold=0.0,
        risk_margin_multiplier=4.0,
        risk_uncertainty_band=1.0,
        exit1_confidence_boost=0.0,
        per_label_margins=tuple(1.0 for _ in config.per_label_margins),
        require_previous_label_stability=True,
    )
    active = StrictSequentialStageConfig(
        mean_confidence_threshold=config.mean_confidence_threshold,
        max_probability_delta=config.max_probability_delta,
        max_label_risk=config.max_label_risk,
        risk_score_threshold=config.risk_score_threshold,
        risk_margin_multiplier=config.risk_margin_multiplier,
        risk_uncertainty_band=config.risk_uncertainty_band,
        exit1_confidence_boost=0.0,
        per_label_margins=config.per_label_margins,
        require_previous_label_stability=config.require_exit2_exit3_stability,
    )
    return StrictSequentialPolicyConfig(num_exits=5, stages=(disabled, disabled, active, disabled), allow_empty_stop=False)
