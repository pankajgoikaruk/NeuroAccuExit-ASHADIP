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
    return arr


def _vector(value: np.ndarray | Sequence[float], size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


@dataclass(frozen=True)
class MultiObjectiveMarginConfig:
    mean_confidence_threshold: float
    max_probability_delta: float
    per_label_margins: tuple[float, ...]
    require_label_set_agreement: bool = True
    allow_empty_stop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decode_genes(
    genes: np.ndarray | Sequence[float],
    *,
    num_labels: int,
    require_label_set_agreement: bool = True,
    allow_empty_stop: bool = False,
) -> MultiObjectiveMarginConfig:
    values = _vector(genes, num_labels + 2, "genes")
    return MultiObjectiveMarginConfig(
        mean_confidence_threshold=float(values[0]),
        max_probability_delta=float(values[1]),
        per_label_margins=tuple(float(item) for item in values[2:]),
        require_label_set_agreement=bool(require_label_set_agreement),
        allow_empty_stop=bool(allow_empty_stop),
    )


def encode_config(config: MultiObjectiveMarginConfig) -> np.ndarray:
    return np.asarray(
        [
            float(config.mean_confidence_threshold),
            float(config.max_probability_delta),
            *[float(value) for value in config.per_label_margins],
        ],
        dtype=np.float64,
    )


def multiobjective_margin_stop_mask(
    diagnostics: dict[str, np.ndarray],
    config: MultiObjectiveMarginConfig,
) -> np.ndarray:
    margins = _matrix(diagnostics["decision_margin"], "decision_margin")
    required = _vector(
        config.per_label_margins,
        margins.shape[1],
        "per_label_margins",
    )
    count = len(margins)
    mask = np.ones(count, dtype=bool)
    if config.require_label_set_agreement:
        mask &= np.asarray(diagnostics["label_set_agreement"], dtype=bool).reshape(-1)
    if not config.allow_empty_stop:
        mask &= np.asarray(diagnostics["non_empty"], dtype=bool).reshape(-1)
    mask &= (
        np.asarray(diagnostics["mean_binary_confidence"], dtype=np.float64).reshape(-1)
        >= float(config.mean_confidence_threshold)
    )
    mask &= (
        np.asarray(diagnostics["maximum_probability_delta"], dtype=np.float64).reshape(-1)
        <= float(config.max_probability_delta)
    )
    mask &= np.all(margins >= required.reshape(1, -1), axis=1)
    return mask


def make_bounds(
    num_labels: int,
    *,
    confidence_bounds: tuple[float, float] = (0.50, 0.99),
    delta_bounds: tuple[float, float] = (0.01, 1.00),
    margin_bounds: tuple[float, float] = (0.00, 0.50),
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [confidence_bounds[0], delta_bounds[0], *([margin_bounds[0]] * num_labels)],
        dtype=np.float64,
    )
    upper = np.asarray(
        [confidence_bounds[1], delta_bounds[1], *([margin_bounds[1]] * num_labels)],
        dtype=np.float64,
    )
    if np.any(lower >= upper):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    return lower, upper


def clip_genes(
    genes: np.ndarray | Sequence[float],
    lower: np.ndarray | Sequence[float],
    upper: np.ndarray | Sequence[float],
) -> np.ndarray:
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    values = _vector(genes, len(lo), "genes")
    if hi.shape != lo.shape:
        raise ValueError("lower and upper bounds must share shape.")
    return np.clip(values, lo, hi)


def random_population(
    *,
    size: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    seeds: Sequence[np.ndarray | Sequence[float]] | None = None,
) -> np.ndarray:
    if size < 2:
        raise ValueError("Population size must be at least 2.")
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    population: list[np.ndarray] = []
    for seed in seeds or []:
        population.append(clip_genes(seed, lo, hi))
        if len(population) >= size:
            break
    while len(population) < size:
        population.append(rng.uniform(lo, hi))
    return np.vstack(population).astype(np.float64)


def constraint_dominates(
    objectives_a: np.ndarray | Sequence[float],
    violation_a: float,
    objectives_b: np.ndarray | Sequence[float],
    violation_b: float,
) -> bool:
    """Return True when A dominates B under Deb's constraint rules.

    All objectives are minimised. A feasible point dominates every infeasible
    point. Between infeasible points, the smaller total violation dominates.
    Between feasible points, ordinary Pareto dominance is used.
    """

    a = np.asarray(objectives_a, dtype=np.float64).reshape(-1)
    b = np.asarray(objectives_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Objective vectors must share shape.")
    va = max(0.0, float(violation_a))
    vb = max(0.0, float(violation_b))
    a_feasible = va <= 1e-12
    b_feasible = vb <= 1e-12
    if a_feasible and not b_feasible:
        return True
    if b_feasible and not a_feasible:
        return False
    if not a_feasible and not b_feasible:
        return va < vb - 1e-12
    return bool(np.all(a <= b + 1e-12) and np.any(a < b - 1e-12))


def fast_non_dominated_sort(
    objectives: np.ndarray,
    violations: np.ndarray | Sequence[float],
) -> tuple[list[list[int]], np.ndarray]:
    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("objectives must be a 2-D matrix.")
    violation = np.asarray(violations, dtype=np.float64).reshape(-1)
    if len(violation) != len(values):
        raise ValueError("violations length must match objectives rows.")

    dominates: list[list[int]] = [[] for _ in range(len(values))]
    dominated_count = np.zeros(len(values), dtype=np.int64)
    first: list[int] = []
    for p in range(len(values)):
        for q in range(len(values)):
            if p == q:
                continue
            if constraint_dominates(values[p], violation[p], values[q], violation[q]):
                dominates[p].append(q)
            elif constraint_dominates(values[q], violation[q], values[p], violation[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            first.append(p)

    fronts: list[list[int]] = []
    rank = np.full(len(values), -1, dtype=np.int64)
    current = first
    level = 0
    while current:
        fronts.append(current)
        next_front: list[int] = []
        for p in current:
            rank[p] = level
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        current = next_front
        level += 1
    if np.any(rank < 0):
        raise RuntimeError("Non-dominated sorting left unranked points.")
    return fronts, rank


def crowding_distance(
    objectives: np.ndarray,
    front: Sequence[int],
) -> np.ndarray:
    values = np.asarray(objectives, dtype=np.float64)
    indices = np.asarray(list(front), dtype=np.int64)
    distances = np.zeros(len(values), dtype=np.float64)
    if len(indices) == 0:
        return distances
    if len(indices) <= 2:
        distances[indices] = np.inf
        return distances

    for objective_idx in range(values.shape[1]):
        ordered = indices[np.argsort(values[indices, objective_idx], kind="mergesort")]
        distances[ordered[0]] = np.inf
        distances[ordered[-1]] = np.inf
        low = float(values[ordered[0], objective_idx])
        high = float(values[ordered[-1], objective_idx])
        if high - low <= 1e-12:
            continue
        for position in range(1, len(ordered) - 1):
            if np.isinf(distances[ordered[position]]):
                continue
            previous_value = values[ordered[position - 1], objective_idx]
            next_value = values[ordered[position + 1], objective_idx]
            distances[ordered[position]] += float(
                (next_value - previous_value) / (high - low)
            )
    return distances


def rank_and_crowding(
    objectives: np.ndarray,
    violations: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    fronts, rank = fast_non_dominated_sort(objectives, violations)
    crowding = np.zeros(len(objectives), dtype=np.float64)
    for front in fronts:
        distances = crowding_distance(objectives, front)
        crowding[np.asarray(front, dtype=np.int64)] = distances[
            np.asarray(front, dtype=np.int64)
        ]
    return rank, crowding, fronts


def tournament_select(
    *,
    rank: np.ndarray,
    crowding: np.ndarray,
    violations: np.ndarray,
    rng: np.random.Generator,
) -> int:
    a, b = rng.integers(0, len(rank), size=2)
    va = max(0.0, float(violations[a]))
    vb = max(0.0, float(violations[b]))
    if va <= 1e-12 < vb:
        return int(a)
    if vb <= 1e-12 < va:
        return int(b)
    if va > 1e-12 and vb > 1e-12 and abs(va - vb) > 1e-12:
        return int(a if va < vb else b)
    if rank[a] != rank[b]:
        return int(a if rank[a] < rank[b] else b)
    if crowding[a] != crowding[b]:
        return int(a if crowding[a] > crowding[b] else b)
    return int(a if rng.random() < 0.5 else b)


def blend_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(parent_a, dtype=np.float64).reshape(-1)
    b = np.asarray(parent_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Parents must share shape.")
    gamma = rng.uniform(-float(alpha), 1.0 + float(alpha), size=len(a))
    child_a = gamma * a + (1.0 - gamma) * b
    child_b = gamma * b + (1.0 - gamma) * a
    return (
        clip_genes(child_a, lower, upper),
        clip_genes(child_b, lower, upper),
    )


def mutate_gaussian(
    genes: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    scale: float = 0.08,
) -> np.ndarray:
    values = np.asarray(genes, dtype=np.float64).copy().reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    if values.shape != lo.shape or lo.shape != hi.shape:
        raise ValueError("genes and bounds must share shape.")
    mutation_mask = rng.random(len(values)) < float(probability)
    if not np.any(mutation_mask):
        mutation_mask[int(rng.integers(0, len(values)))] = True
    sigma = (hi - lo) * float(scale)
    values[mutation_mask] += rng.normal(0.0, sigma[mutation_mask])
    return clip_genes(values, lo, hi)


def make_offspring(
    *,
    population: np.ndarray,
    objectives: np.ndarray,
    violations: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    crossover_probability: float,
    mutation_probability: float,
    mutation_scale: float,
) -> np.ndarray:
    rank, crowding, _ = rank_and_crowding(objectives, violations)
    children: list[np.ndarray] = []
    while len(children) < len(population):
        idx_a = tournament_select(
            rank=rank,
            crowding=crowding,
            violations=violations,
            rng=rng,
        )
        idx_b = tournament_select(
            rank=rank,
            crowding=crowding,
            violations=violations,
            rng=rng,
        )
        parent_a = population[idx_a]
        parent_b = population[idx_b]
        if rng.random() < float(crossover_probability):
            child_a, child_b = blend_crossover(
                parent_a,
                parent_b,
                lower=lower,
                upper=upper,
                rng=rng,
            )
        else:
            child_a, child_b = parent_a.copy(), parent_b.copy()
        child_a = mutate_gaussian(
            child_a,
            lower=lower,
            upper=upper,
            rng=rng,
            probability=mutation_probability,
            scale=mutation_scale,
        )
        child_b = mutate_gaussian(
            child_b,
            lower=lower,
            upper=upper,
            rng=rng,
            probability=mutation_probability,
            scale=mutation_scale,
        )
        children.extend([child_a, child_b])
    return np.vstack(children[: len(population)]).astype(np.float64)


def environmental_select(
    *,
    population: np.ndarray,
    objectives: np.ndarray,
    violations: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fronts, _ = fast_non_dominated_sort(objectives, violations)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
            continue
        distance = crowding_distance(objectives, front)
        ordered = sorted(
            front,
            key=lambda idx: (
                -float(distance[idx]),
                float(violations[idx]),
            ),
        )
        selected.extend(ordered[: size - len(selected)])
        break
    chosen = np.asarray(selected, dtype=np.int64)
    return population[chosen], objectives[chosen], violations[chosen]


def pareto_front_mask(
    objectives: np.ndarray,
    violations: np.ndarray | Sequence[float],
) -> np.ndarray:
    fronts, _ = fast_non_dominated_sort(objectives, violations)
    mask = np.zeros(len(objectives), dtype=bool)
    if fronts:
        mask[np.asarray(fronts[0], dtype=np.int64)] = True
    return mask
