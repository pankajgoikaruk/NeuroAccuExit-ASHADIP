from __future__ import annotations

import unittest

import numpy as np

from policies.multiobjective_per_label_margin import (
    MultiObjectiveMarginConfig,
    constraint_dominates,
    environmental_select,
    fast_non_dominated_sort,
    make_bounds,
    multiobjective_margin_stop_mask,
    mutate_gaussian,
    pareto_front_mask,
)


class MultiObjectivePerLabelMarginTest(unittest.TestCase):
    def test_stop_mask_uses_each_label_margin(self) -> None:
        diagnostics = {
            "decision_margin": np.asarray([[0.20, 0.04], [0.20, 0.20]], dtype=np.float32),
            "label_set_agreement": np.asarray([True, True]),
            "non_empty": np.asarray([True, True]),
            "mean_binary_confidence": np.asarray([0.90, 0.90], dtype=np.float32),
            "maximum_probability_delta": np.asarray([0.05, 0.05], dtype=np.float32),
        }
        config = MultiObjectiveMarginConfig(
            mean_confidence_threshold=0.80,
            max_probability_delta=0.10,
            per_label_margins=(0.10, 0.10),
        )
        np.testing.assert_array_equal(
            multiobjective_margin_stop_mask(diagnostics, config),
            np.asarray([False, True]),
        )

    def test_feasible_point_dominates_infeasible_point(self) -> None:
        self.assertTrue(
            constraint_dominates(
                np.asarray([-1.0, 0.01]),
                0.0,
                np.asarray([-5.0, 0.00]),
                0.2,
            )
        )

    def test_pareto_front_keeps_tradeoff_points(self) -> None:
        objectives = np.asarray(
            [
                [-5.0, 0.02],
                [-3.0, 0.01],
                [-1.0, 0.03],
            ],
            dtype=np.float64,
        )
        violations = np.zeros(3, dtype=np.float64)
        mask = pareto_front_mask(objectives, violations)
        np.testing.assert_array_equal(mask, np.asarray([True, True, False]))

    def test_non_dominated_sort_prefers_lower_violation(self) -> None:
        objectives = np.asarray([[-9.0, 0.1], [-1.0, 0.0]], dtype=np.float64)
        violations = np.asarray([0.2, 0.1], dtype=np.float64)
        fronts, rank = fast_non_dominated_sort(objectives, violations)
        self.assertEqual(fronts[0], [1])
        self.assertEqual(int(rank[1]), 0)

    def test_mutation_respects_bounds(self) -> None:
        lower, upper = make_bounds(3)
        rng = np.random.default_rng(7)
        genes = np.asarray([0.5, 1.0, 0.0, 0.5, 0.25], dtype=np.float64)
        mutated = mutate_gaussian(
            genes,
            lower=lower,
            upper=upper,
            rng=rng,
            probability=1.0,
            scale=10.0,
        )
        self.assertTrue(np.all(mutated >= lower))
        self.assertTrue(np.all(mutated <= upper))

    def test_environmental_selection_returns_requested_size(self) -> None:
        population = np.arange(24, dtype=np.float64).reshape(6, 4)
        objectives = np.asarray(
            [
                [-6.0, 0.06],
                [-5.0, 0.05],
                [-4.0, 0.04],
                [-3.0, 0.03],
                [-2.0, 0.02],
                [-1.0, 0.01],
            ]
        )
        violations = np.zeros(6, dtype=np.float64)
        selected, selected_objectives, selected_violations = environmental_select(
            population=population,
            objectives=objectives,
            violations=violations,
            size=4,
        )
        self.assertEqual(selected.shape, (4, 4))
        self.assertEqual(selected_objectives.shape, (4, 2))
        self.assertEqual(selected_violations.shape, (4,))


if __name__ == "__main__":
    unittest.main()
