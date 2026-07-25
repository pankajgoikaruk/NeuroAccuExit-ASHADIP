from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


def _vector(value: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array

@dataclass(frozen=True)
class SequentialStageConfig:
    mean_confidence_threshold: float
    max_probability_delta: float
    max_label_risk: float
    per_label_margins: tuple[float, ...]
    require_previous_label_stability: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequentialPolicyConfig:
    num_exits: int
    stages: tuple[SequentialStageConfig, ...]
    allow_empty_stop: bool = False

    def __post_init__(self) -> None:
        if int(self.num_exits) < 2:
            raise ValueError("num_exits must be at least two.")
        if len(self.stages) != int(self.num_exits) - 1:
            raise ValueError("One stage config is required for every non-final exit.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_exits": int(self.num_exits),
            "allow_empty_stop": bool(self.allow_empty_stop),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def genes_per_stage(num_labels: int) -> int:
    return int(num_labels) + 3


def total_genes(num_exits: int, num_labels: int) -> int:
    return (int(num_exits) - 1) * genes_per_stage(num_labels)


def make_sequential_bounds(
    *,
    num_exits: int,
    num_labels: int,
    confidence_bounds: tuple[float, float] = (0.50, 0.99),
    delta_bounds: tuple[float, float] = (0.00, 1.00),
    risk_bounds: tuple[float, float] = (0.00, 1.00),
    margin_bounds: tuple[float, float] = (0.00, 0.50),
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [confidence_bounds, delta_bounds, risk_bounds, margin_bounds]
    if any(float(lo) >= float(hi) for lo, hi in pairs):
        raise ValueError("Each lower optimisation bound must be below its upper bound.")
    lower_stage = [confidence_bounds[0], delta_bounds[0], risk_bounds[0]] + [
        margin_bounds[0]
    ] * int(num_labels)
    upper_stage = [confidence_bounds[1], delta_bounds[1], risk_bounds[1]] + [
        margin_bounds[1]
    ] * int(num_labels)
    return (
        np.tile(np.asarray(lower_stage, dtype=np.float64), int(num_exits) - 1),
        np.tile(np.asarray(upper_stage, dtype=np.float64), int(num_exits) - 1),
    )


def decode_sequential_genes(
    genes: Sequence[float] | np.ndarray,
    *,
    num_exits: int,
    num_labels: int,
    allow_empty_stop: bool = False,
) -> SequentialPolicyConfig:
    values = _vector(genes, total_genes(num_exits, num_labels), "genes")
    width = genes_per_stage(num_labels)
    stages: list[SequentialStageConfig] = []
    for stage_index in range(int(num_exits) - 1):
        offset = stage_index * width
        stage_values = values[offset : offset + width]
        stages.append(
            SequentialStageConfig(
                mean_confidence_threshold=float(stage_values[0]),
                max_probability_delta=float(stage_values[1]),
                max_label_risk=float(stage_values[2]),
                per_label_margins=tuple(float(item) for item in stage_values[3:]),
                require_previous_label_stability=stage_index > 0,
            )
        )
    return SequentialPolicyConfig(
        num_exits=int(num_exits), stages=tuple(stages), allow_empty_stop=allow_empty_stop
    )


def encode_sequential_config(config: SequentialPolicyConfig) -> np.ndarray:
    values: list[float] = []
    for stage in config.stages:
        values.extend(
            [
                float(stage.mean_confidence_threshold),
                float(stage.max_probability_delta),
                float(stage.max_label_risk),
                *[float(item) for item in stage.per_label_margins],
            ]
        )
    return np.asarray(values, dtype=np.float64)



__all__ = ["SequentialStageConfig", "SequentialPolicyConfig", "genes_per_stage", "total_genes", "make_sequential_bounds", "decode_sequential_genes", "encode_sequential_config"]
