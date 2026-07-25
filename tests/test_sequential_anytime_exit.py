from __future__ import annotations

import unittest

import numpy as np

from policies.sequential_anytime_exit import (
    SequentialPolicyConfig,
    SequentialStageConfig,
    decode_sequential_genes,
    derive_validation_risk_weights,
    make_sequential_bounds,
    sequential_select,
)


class SequentialAnytimeExitTest(unittest.TestCase):
    def test_three_exit_uses_exit1_exit2_and_exit3(self) -> None:
        probabilities = [
            np.asarray([[0.99], [0.60], [0.51]], dtype=np.float32),
            np.asarray([[0.98], [0.95], [0.52]], dtype=np.float32),
            np.asarray([[0.97], [0.96], [0.90]], dtype=np.float32),
        ]
        config = SequentialPolicyConfig(
            num_exits=3,
            stages=(
                SequentialStageConfig(0.90, 1.0, 1.0, (0.20,), False),
                SequentialStageConfig(0.90, 0.40, 1.0, (0.20,), True),
            ),
        )
        result = sequential_select(
            exit_probabilities=probabilities,
            thresholds_by_exit=[np.asarray([0.5])] * 3,
            risk_weights_by_exit=np.zeros((2, 1), dtype=np.float32),
            config=config,
        )
        self.assertEqual(result["selected_exit"].tolist(), [1, 2, 3])

    def test_five_exit_routes_sequentially(self) -> None:
        probabilities = [
            np.asarray([[0.99], [0.60], [0.60], [0.60], [0.60]], dtype=np.float32),
            np.asarray([[0.99], [0.98], [0.60], [0.60], [0.60]], dtype=np.float32),
            np.asarray([[0.99], [0.98], [0.97], [0.60], [0.60]], dtype=np.float32),
            np.asarray([[0.99], [0.98], [0.97], [0.96], [0.60]], dtype=np.float32),
            np.asarray([[0.99], [0.98], [0.97], [0.96], [0.95]], dtype=np.float32),
        ]
        stages = tuple(
            SequentialStageConfig(0.90, 1.0, 1.0, (0.20,), index > 0)
            for index in range(4)
        )
        result = sequential_select(
            exit_probabilities=probabilities,
            thresholds_by_exit=[np.asarray([0.5])] * 5,
            risk_weights_by_exit=np.zeros((4, 1), dtype=np.float32),
            config=SequentialPolicyConfig(num_exits=5, stages=stages),
        )
        self.assertEqual(result["selected_exit"].tolist(), [1, 2, 3, 4, 5])

    def test_later_exit_requires_label_stability(self) -> None:
        probabilities = [
            np.asarray([[0.90]], dtype=np.float32),
            np.asarray([[0.10]], dtype=np.float32),
            np.asarray([[0.95]], dtype=np.float32),
        ]
        config = SequentialPolicyConfig(
            num_exits=3,
            stages=(
                SequentialStageConfig(0.99, 1.0, 1.0, (0.0,), False),
                SequentialStageConfig(0.50, 1.0, 1.0, (0.0,), True),
            ),
        )
        result = sequential_select(
            exit_probabilities=probabilities,
            thresholds_by_exit=[np.asarray([0.5])] * 3,
            risk_weights_by_exit=np.zeros((2, 1), dtype=np.float32),
            config=config,
        )
        self.assertEqual(int(result["selected_exit"][0]), 3)

    def test_minimum_exit_can_disable_exit1_ablation(self) -> None:
        probabilities = [
            np.asarray([[0.99]], dtype=np.float32),
            np.asarray([[0.99]], dtype=np.float32),
            np.asarray([[0.99]], dtype=np.float32),
        ]
        stages = (
            SequentialStageConfig(0.5, 1.0, 1.0, (0.0,), False),
            SequentialStageConfig(0.5, 1.0, 1.0, (0.0,), True),
        )
        result = sequential_select(
            exit_probabilities=probabilities,
            thresholds_by_exit=[np.asarray([0.5])] * 3,
            risk_weights_by_exit=np.zeros((2, 1), dtype=np.float32),
            config=SequentialPolicyConfig(num_exits=3, stages=stages),
            minimum_exit=2,
        )
        self.assertEqual(int(result["selected_exit"][0]), 2)

    def test_risk_weights_identify_final_exit_corrections(self) -> None:
        truth = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
        exits = [
            np.asarray([[0.1, 0.9], [0.1, 0.9]], dtype=np.float32),
            np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        ]
        weights, counts = derive_validation_risk_weights(
            y_true=truth,
            exit_probabilities=exits,
            thresholds_by_exit=[np.asarray([0.5, 0.5])] * 2,
        )
        self.assertEqual(counts.tolist(), [[1, 1]])
        self.assertTrue(np.allclose(weights, np.ones((1, 2))))

    def test_gene_shape_for_five_exit(self) -> None:
        lower, upper = make_sequential_bounds(num_exits=5, num_labels=3)
        self.assertEqual(lower.shape, (24,))
        config = decode_sequential_genes(
            (lower + upper) / 2.0, num_exits=5, num_labels=3
        )
        self.assertEqual(len(config.stages), 4)
        self.assertEqual(len(config.stages[0].per_label_margins), 3)


if __name__ == "__main__":
    unittest.main()
