from __future__ import annotations

import unittest

import numpy as np

from policies.parent_aware_adaptive_gate import (
    LATSLabelRule,
    adaptive_label_stop_mask,
    aggregate_1d,
    build_parent_aware_features,
    counterfactual_parent_unsafe_targets,
    derive_label_probability_thresholds,
)


class ParentAwareAdaptiveGateTest(unittest.TestCase):
    def test_lats_aggregations(self) -> None:
        values = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        self.assertAlmostEqual(aggregate_1d(values, "mean"), 0.5, places=6)
        self.assertAlmostEqual(aggregate_1d(values, "top3mean"), 1.9 / 3.0, places=6)
        self.assertAlmostEqual(aggregate_1d(values, "p75"), 0.825, places=6)
        self.assertAlmostEqual(
            aggregate_1d(np.asarray([0.2, 0.5]), "noisy_or"),
            0.6,
            places=6,
        )

    def test_counterfactual_parent_target_detects_harm(self) -> None:
        rules = (LATSLabelRule("mean", 0.5),)
        truth = np.asarray([[1], [1]], dtype=np.int8)
        source = np.asarray([[0.0], [0.9]], dtype=np.float32)
        deeper = np.asarray([[0.9], [0.9]], dtype=np.float32)
        unsafe, baseline, counterfactual = counterfactual_parent_unsafe_targets(
            y_true=truth,
            source_probabilities=source,
            deeper_probabilities=deeper,
            parent_ids=["p", "p"],
            rules=rules,
        )
        self.assertEqual(int(unsafe[0, 0]), 1)
        self.assertEqual(int(unsafe[1, 0]), 0)
        self.assertEqual(int(baseline[0, 0]), 1)
        self.assertEqual(int(counterfactual[0, 0]), 0)

    def test_parent_aware_feature_width(self) -> None:
        rules = (
            LATSLabelRule("mean", 0.5),
            LATSLabelRule("max", 0.7),
        )
        current = np.asarray(
            [[0.2, 0.8], [0.7, 0.4], [0.9, 0.3]], dtype=np.float32
        )
        previous = np.asarray(
            [[0.1, 0.7], [0.6, 0.5], [0.8, 0.2]], dtype=np.float32
        )
        features, names, diagnostics = build_parent_aware_features(
            current_probabilities=current,
            previous_probabilities=previous,
            parent_ids=["a", "a", "b"],
            current_thresholds=[0.5, 0.5],
            rules=rules,
        )
        self.assertEqual(features.shape, (3, 24))
        self.assertEqual(len(names), 24)
        self.assertEqual(diagnostics["parent_scores"].shape, (3, 2))

    def test_thresholds_are_label_specific(self) -> None:
        targets = np.asarray(
            [[1, 0], [1, 1], [1, 1], [0, 1]], dtype=np.int8
        )
        probabilities = np.asarray(
            [[0.2, 0.1], [0.4, 0.8], [0.6, 0.9], [0.1, 0.7]],
            dtype=np.float32,
        )
        thresholds, counts, fallback = derive_label_probability_thresholds(
            unsafe_targets=targets,
            unsafe_probabilities=probabilities,
            target_recall=2.0 / 3.0,
            minimum_positive_examples=2,
        )
        self.assertEqual(counts.tolist(), [3, 3])
        self.assertFalse(bool(fallback[0]))
        self.assertFalse(bool(fallback[1]))
        self.assertNotAlmostEqual(float(thresholds[0]), float(thresholds[1]))

    def test_adaptive_stop_checks_each_label(self) -> None:
        probabilities = np.asarray(
            [[0.2, 0.7], [0.4, 0.2]], dtype=np.float32
        )
        stop, harm, highest = adaptive_label_stop_mask(
            unsafe_probabilities=probabilities,
            label_thresholds=[0.5, 0.6],
            non_empty=np.asarray([True, True]),
        )
        self.assertEqual(stop.tolist(), [False, True])
        self.assertEqual(highest.tolist(), [1, 0])
        self.assertEqual(harm.shape, (2,))


if __name__ == "__main__":
    unittest.main()
