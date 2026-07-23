from __future__ import annotations

import unittest

import numpy as np

from policies.parent_aware_adaptive_gate import LATSLabelRule
from policies.whole_parent_selective_exit import (
    build_whole_parent_features,
    expand_parent_label_features,
    fit_empirical_risk_calibrators,
    predict_empirical_unsafe_probabilities,
    whole_parent_stop_mask,
    whole_parent_unsafe_targets,
    wilson_upper_bound,
)


class WholeParentSelectiveExitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = (
            LATSLabelRule("mean", 0.5),
            LATSLabelRule("max", 0.6),
        )
        self.parent_ids = np.asarray(["a", "a", "b", "b"], dtype=object)

    def test_features_have_one_row_per_parent(self) -> None:
        p1 = np.asarray([[0.2, 0.1], [0.3, 0.2], [0.7, 0.4], [0.8, 0.5]])
        p2 = np.asarray([[0.3, 0.2], [0.4, 0.3], [0.8, 0.5], [0.9, 0.7]])
        features, names, diagnostics = build_whole_parent_features(
            current_probabilities=p2,
            previous_probabilities=p1,
            parent_ids=self.parent_ids,
            rules=self.rules,
        )
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], len(names))
        np.testing.assert_array_equal(diagnostics["parent_ids"], ["a", "b"])
        np.testing.assert_array_equal(diagnostics["row_to_parent"], [0, 0, 1, 1])

    def test_complete_parent_target_detects_joint_harm(self) -> None:
        truth = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8)
        p2 = np.asarray([[0.4, 0.1], [0.4, 0.2], [0.2, 0.55], [0.2, 0.50]])
        p3 = np.asarray([[0.8, 0.1], [0.8, 0.2], [0.2, 0.8], [0.2, 0.7]])
        result = whole_parent_unsafe_targets(
            y_true=truth,
            source_probabilities=p2,
            deeper_probabilities=p3,
            parent_ids=self.parent_ids,
            rules=self.rules,
        )
        self.assertEqual(result["unsafe_targets"][0, 0], 1)
        self.assertEqual(result["unsafe_targets"][1, 1], 1)
        self.assertTrue(result["any_unsafe"].all())

    def test_parent_label_expansion_width(self) -> None:
        features = np.ones((3, 7), dtype=np.float32)
        expanded, parent_index, label_index = expand_parent_label_features(features, 4)
        self.assertEqual(expanded.shape, (12, 11))
        self.assertEqual(len(parent_index), 12)
        self.assertEqual(len(label_index), 12)

    def test_empirical_calibration_is_monotone(self) -> None:
        raw = np.asarray(
            [[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]], dtype=np.float32
        )
        target = np.asarray([[0], [0], [0], [1], [1], [1]], dtype=np.int8)
        calibrators, _, _ = fit_empirical_risk_calibrators(
            raw_scores=raw,
            unsafe_targets=target,
            num_bins=3,
            minimum_positive_examples=1,
        )
        predicted = predict_empirical_unsafe_probabilities(calibrators, raw)[:, 0]
        self.assertTrue(np.all(np.diff(predicted) >= -1e-7))
        self.assertGreater(predicted[-1], predicted[0])

    def test_stop_is_one_decision_per_parent(self) -> None:
        probabilities = np.asarray([[0.1, 0.2], [0.1, 0.8]], dtype=np.float32)
        stop, expected, highest = whole_parent_stop_mask(
            unsafe_probabilities=probabilities,
            label_thresholds=np.asarray([0.5, 0.5]),
            expected_harm_threshold=None,
            non_empty=np.asarray([True, True]),
        )
        np.testing.assert_array_equal(stop, [True, False])
        self.assertEqual(expected.shape, (2,))
        self.assertEqual(highest[1], 1)

    def test_wilson_bound_exceeds_observed_rate(self) -> None:
        bound = wilson_upper_bound(2, 100)
        self.assertGreater(bound, 0.02)
        self.assertLess(bound, 0.10)


if __name__ == "__main__":
    unittest.main()
