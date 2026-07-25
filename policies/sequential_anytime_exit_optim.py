from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np



def constraint_dominates(
    objectives_a: np.ndarray, violation_a: float, objectives_b: np.ndarray, violation_b: float
) -> bool:
    a = np.asarray(objectives_a, dtype=np.float64)
    b = np.asarray(objectives_b, dtype=np.float64)
    va, vb = max(0.0, float(violation_a)), max(0.0, float(violation_b))
    if va <= 1e-12 and vb > 1e-12:
        return True
    if va > 1e-12 and vb <= 1e-12:
        return False
    if va > 1e-12 and vb > 1e-12:
        return va < vb - 1e-12
    return bool(np.all(a <= b + 1e-12) and np.any(a < b - 1e-12))


def non_dominated_sort(objectives: np.ndarray, violations: np.ndarray) -> list[list[int]]:
    matrix = np.asarray(objectives, dtype=np.float64)
    violation = np.asarray(violations, dtype=np.float64).reshape(-1)
    dominates: list[list[int]] = [[] for _ in range(len(matrix))]
    dominated_count = np.zeros(len(matrix), dtype=np.int64)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if constraint_dominates(matrix[i], violation[i], matrix[j], violation[j]):
                dominates[i].append(j); dominated_count[j] += 1
            elif constraint_dominates(matrix[j], violation[j], matrix[i], violation[i]):
                dominates[j].append(i); dominated_count[i] += 1
    fronts: list[list[int]] = [np.flatnonzero(dominated_count == 0).tolist()]
    while fronts[-1]:
        next_front: list[int] = []
        for i in fronts[-1]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts


def crowding_distance(objectives: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    idx = np.asarray(list(indices), dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(idx) <= 2:
        return np.full(len(idx), np.inf, dtype=np.float64)
    values = np.asarray(objectives, dtype=np.float64)[idx]
    distance = np.zeros(len(idx), dtype=np.float64)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column])
        distance[order[0]] = distance[order[-1]] = np.inf
        span = values[order[-1], column] - values[order[0], column]
        if span <= 1e-12:
            continue
        for pos in range(1, len(order) - 1):
            if np.isfinite(distance[order[pos]]):
                distance[order[pos]] += (
                    values[order[pos + 1], column] - values[order[pos - 1], column]
                ) / span
    return distance


def environmental_select(
    *, population: np.ndarray, objectives: np.ndarray, violations: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fronts = non_dominated_sort(objectives, violations)
    selected: list[int] = []
    rank = np.full(len(population), len(fronts), dtype=np.int64)
    crowd = np.zeros(len(population), dtype=np.float64)
    for front_rank, front in enumerate(fronts):
        for i in front:
            rank[i] = front_rank
        distances = crowding_distance(objectives, front)
        for local, i in enumerate(front):
            crowd[i] = distances[local]
        remaining = int(size) - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
        else:
            order = np.argsort(-distances, kind='stable')
            selected.extend([front[int(i)] for i in order[:remaining]])
            break
    chosen = np.asarray(selected, dtype=np.int64)
    return np.asarray(population)[chosen], rank[chosen], crowd[chosen]


def random_population(
    *, size: int, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator,
    seeds: Sequence[np.ndarray | Sequence[float]] | None = None
) -> np.ndarray:
    result: list[np.ndarray] = []
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    for seed in seeds or []:
        result.append(np.clip(np.asarray(seed, dtype=np.float64), lo, hi))
        if len(result) >= int(size):
            break
    while len(result) < int(size):
        result.append(rng.uniform(lo, hi))
    return np.vstack(result)


def make_offspring(
    *, population: np.ndarray, objectives: np.ndarray, violations: np.ndarray,
    lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator,
    crossover_probability: float = 0.9, mutation_probability: float = 0.2,
    mutation_scale: float = 0.08
) -> np.ndarray:
    pop = np.asarray(population, dtype=np.float64)
    fronts = non_dominated_sort(objectives, violations)
    rank = np.full(len(pop), len(fronts), dtype=np.int64)
    crowd = np.zeros(len(pop), dtype=np.float64)
    for r, front in enumerate(fronts):
        distances = crowding_distance(objectives, front)
        for local, i in enumerate(front):
            rank[i] = r; crowd[i] = distances[local]
    def pick() -> int:
        a, b = rng.integers(0, len(pop), size=2)
        if rank[a] != rank[b]:
            return int(a if rank[a] < rank[b] else b)
        return int(a if crowd[a] >= crowd[b] else b)
    children: list[np.ndarray] = []
    span = np.asarray(upper) - np.asarray(lower)
    while len(children) < len(pop):
        p1, p2 = pop[pick()].copy(), pop[pick()].copy()
        if rng.random() < float(crossover_probability):
            alpha = rng.random(len(p1))
            c1 = alpha * p1 + (1.0 - alpha) * p2
            c2 = alpha * p2 + (1.0 - alpha) * p1
        else:
            c1, c2 = p1, p2
        for child in (c1, c2):
            mask = rng.random(len(child)) < float(mutation_probability)
            if np.any(mask):
                child[mask] += rng.normal(0.0, float(mutation_scale), size=mask.sum()) * span[mask]
            children.append(np.clip(child, lower, upper))
            if len(children) >= len(pop):
                break
    return np.vstack(children)


def pareto_front_mask(objectives: np.ndarray, violations: np.ndarray) -> np.ndarray:
    fronts = non_dominated_sort(objectives, violations)
    mask = np.zeros(len(objectives), dtype=bool)
    if fronts:
        mask[fronts[0]] = True
    return mask



__all__ = [
    "constraint_dominates", "non_dominated_sort", "crowding_distance",
    "environmental_select", "random_population", "make_offspring",
    "pareto_front_mask",
]
