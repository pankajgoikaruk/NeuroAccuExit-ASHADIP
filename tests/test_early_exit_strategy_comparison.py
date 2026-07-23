from __future__ import annotations

import unittest

import numpy as np

from policies.early_exit_strategy_comparison import (
    GlobalRuleConfig,
    build_gate_features,
    compute_common_diagnostics,
    continuation_reasons,
    derive_per_label_margin_thresholds,
    gate_safe_targets,
    global_rule_stop_mask,
    split_parent_ids,
)


class EarlyExitStrategyComparisonTest(unittest.TestCase):
    def test_parent_split_is_disjoint_and_grouped(self) -> None:
        parents = ["a", "a", "b", "b", "c", "c", "d", "d"]
        derivation, selection = split_parent_ids(
            parents,
            derivation_fraction=0.5,
            seed=7,
        )
        self.assertTrue(np.all(derivation ^ selection))
        for parent in set(parents):
            idx = [i for i, value in enumerate(parents) if value == parent]
            self.assertTrue(
                np.all(derivation[idx]) or np.all(selection[idx])
            )

    def test_gate_features_have_expected_width(self) -> None:
        p1 = np.full((3, 2), 0.2, dtype=np.float32)
        p2 = np.full((3, 2), 0.8, dtype=np.float32)
        features, names = build_gate_features(
            exit1_probabilities=p1,
            exit2_probabilities=p2,
            exit1_thresholds=np.asarray([0.5, 0.5]),
            exit2_thresholds=np.asarray([0.5, 0.5]),
        )
        self.assertEqual(features.shape, (3, 16))
        self.assertEqual(len(names), 16)

    def test_gate_target_marks_positive_exit3_gain_unsafe(self) -> None:
        truth = np.asarray([[1, 0], [1, 1]], dtype=np.int8)
        pred2 = np.asarray([[0, 0], [1, 1]], dtype=np.int8)
        pred3 = np.asarray([[1, 0], [1, 0]], dtype=np.int8)
        safe, improvement = gate_safe_targets(
            y_true=truth,
            exit2_predictions=pred2,
            exit3_predictions=pred3,
        )
        np.testing.assert_array_equal(
            improvement,
            np.asarray([1, -1]),
        )
        np.testing.assert_array_equal(safe, np.asarray([0, 1]))

    def test_per_label_margin_uses_corrected_examples(self) -> None:
        truth = np.asarray([[1], [1], [1], [0]], dtype=np.int8)
        p2 = np.asarray(
            [[0.49], [0.45], [0.40], [0.10]],
            dtype=np.float32,
        )
        p3 = np.asarray(
            [[0.90], [0.90], [0.90], [0.10]],
            dtype=np.float32,
        )
        margins, counts = derive_per_label_margin_thresholds(
            y_true=truth,
            exit2_probabilities=p2,
            exit3_probabilities=p3,
            exit2_thresholds=np.asarray([0.5]),
            exit3_thresholds=np.asarray([0.5]),
            capture_fraction=0.5,
            minimum_corrected_examples=3,
        )
        self.assertEqual(int(counts[0]), 3)
        self.assertAlmostEqual(float(margins[0]), 0.05, places=5)

    def test_global_reason_reports_failed_conditions(self) -> None:
        p1 = np.asarray([[0.9, 0.4]], dtype=np.float32)
        p2 = np.asarray([[0.8, 0.6]], dtype=np.float32)
        diagnostics = compute_common_diagnostics(
            exit1_probabilities=p1,
            exit2_probabilities=p2,
            exit1_thresholds=np.asarray([0.5, 0.5]),
            exit2_thresholds=np.asarray([0.5, 0.5]),
        )
        config = GlobalRuleConfig(0.9, 0.2, 0.1)
        stop = global_rule_stop_mask(diagnostics, config)
        reasons = continuation_reasons(
            method="global_conf_margin_delta",
            diagnostics=diagnostics,
            config=config.to_dict(),
            stop_mask=stop,
        )
        self.assertIn("label_set_disagreement", reasons[0])
        self.assertIn("low_mean_confidence", reasons[0])
        self.assertIn("large_probability_change", reasons[0])


if __name__ == "__main__":
    unittest.main()
