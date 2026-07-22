# tests/test_label_aware_early_exit_policy.py

from __future__ import annotations

import unittest

import numpy as np

from policies.label_aware_early_exit_policy import (
    LabelAwarePolicyConfig,
    compute_label_aware_diagnostics,
    derive_label_risk_profile,
    label_aware_stop_mask,
)


class LabelAwareEarlyExitPolicyTest(unittest.TestCase):
    def test_validation_gain_creates_label_risk_weight(self) -> None:
        labels = ("easy", "difficult")
        y_true = np.asarray(
            [
                [1, 1],
                [1, 0],
                [0, 1],
                [0, 0],
            ],
            dtype=np.int8,
        )
        exit2_probs = np.asarray(
            [
                [0.9, 0.4],
                [0.8, 0.6],
                [0.1, 0.4],
                [0.2, 0.6],
            ],
            dtype=np.float32,
        )
        exit3_probs = np.asarray(
            [
                [0.9, 0.9],
                [0.8, 0.1],
                [0.1, 0.8],
                [0.2, 0.1],
            ],
            dtype=np.float32,
        )

        profile = derive_label_risk_profile(
            labels=labels,
            y_true=y_true,
            exit2_probabilities=exit2_probs,
            exit3_probabilities=exit3_probs,
            exit2_thresholds=np.asarray([0.5, 0.5]),
            exit3_thresholds=np.asarray([0.5, 0.5]),
            minimum_improvement=0.01,
        )

        self.assertAlmostEqual(profile.risk_weights[0], 0.0)
        self.assertAlmostEqual(profile.risk_weights[1], 1.0)
        self.assertGreater(profile.improvement[1], profile.improvement[0])

    def test_risky_label_near_threshold_blocks_exit(self) -> None:
        exit1_probs = np.asarray([[0.90, 0.51]], dtype=np.float32)
        exit2_probs = np.asarray([[0.92, 0.52]], dtype=np.float32)
        thresholds = np.asarray([0.5, 0.5], dtype=np.float32)

        diagnostics = compute_label_aware_diagnostics(
            exit1_probabilities=exit1_probs,
            exit2_probabilities=exit2_probs,
            exit1_thresholds=thresholds,
            exit2_thresholds=thresholds,
            risk_weights=np.asarray([0.0, 1.0], dtype=np.float32),
            margin_scale=0.25,
        )
        config = LabelAwarePolicyConfig(
            mean_confidence_threshold=0.5,
            global_margin_threshold=0.0,
            max_probability_delta=1.0,
            label_risk_threshold=0.4,
        )

        self.assertTrue(diagnostics["label_set_agreement"][0])
        self.assertFalse(label_aware_stop_mask(diagnostics, config)[0])

    def test_easy_label_uncertainty_is_not_overweighted(self) -> None:
        exit1_probs = np.asarray([[0.51, 0.90]], dtype=np.float32)
        exit2_probs = np.asarray([[0.52, 0.92]], dtype=np.float32)
        thresholds = np.asarray([0.5, 0.5], dtype=np.float32)

        diagnostics = compute_label_aware_diagnostics(
            exit1_probabilities=exit1_probs,
            exit2_probabilities=exit2_probs,
            exit1_thresholds=thresholds,
            exit2_thresholds=thresholds,
            risk_weights=np.asarray([0.0, 1.0], dtype=np.float32),
            margin_scale=0.25,
        )
        config = LabelAwarePolicyConfig(
            mean_confidence_threshold=0.5,
            global_margin_threshold=0.0,
            max_probability_delta=1.0,
            label_risk_threshold=0.4,
        )

        self.assertTrue(label_aware_stop_mask(diagnostics, config)[0])

    def test_large_inter_exit_change_blocks_exit(self) -> None:
        exit1_probs = np.asarray([[0.55, 0.90]], dtype=np.float32)
        exit2_probs = np.asarray([[0.90, 0.92]], dtype=np.float32)
        thresholds = np.asarray([0.5, 0.5], dtype=np.float32)

        diagnostics = compute_label_aware_diagnostics(
            exit1_probabilities=exit1_probs,
            exit2_probabilities=exit2_probs,
            exit1_thresholds=thresholds,
            exit2_thresholds=thresholds,
            risk_weights=np.asarray([1.0, 1.0], dtype=np.float32),
            margin_scale=0.25,
        )
        config = LabelAwarePolicyConfig(
            mean_confidence_threshold=0.5,
            global_margin_threshold=0.0,
            max_probability_delta=0.20,
            label_risk_threshold=1.0,
        )

        self.assertFalse(label_aware_stop_mask(diagnostics, config)[0])

    def test_label_set_disagreement_blocks_exit(self) -> None:
        exit1_probs = np.asarray([[0.49, 0.90]], dtype=np.float32)
        exit2_probs = np.asarray([[0.51, 0.92]], dtype=np.float32)
        thresholds = np.asarray([0.5, 0.5], dtype=np.float32)

        diagnostics = compute_label_aware_diagnostics(
            exit1_probabilities=exit1_probs,
            exit2_probabilities=exit2_probs,
            exit1_thresholds=thresholds,
            exit2_thresholds=thresholds,
            risk_weights=np.asarray([0.0, 0.0], dtype=np.float32),
            margin_scale=0.25,
        )
        config = LabelAwarePolicyConfig(
            mean_confidence_threshold=0.5,
            global_margin_threshold=0.0,
            max_probability_delta=1.0,
            label_risk_threshold=1.0,
            require_label_set_agreement=True,
        )

        self.assertFalse(label_aware_stop_mask(diagnostics, config)[0])

    def test_invalid_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_label_aware_diagnostics(
                exit1_probabilities=np.zeros((2, 2), dtype=np.float32),
                exit2_probabilities=np.zeros((2, 3), dtype=np.float32),
                exit1_thresholds=np.asarray([0.5, 0.5]),
                exit2_thresholds=np.asarray([0.5, 0.5, 0.5]),
                risk_weights=np.asarray([1.0, 1.0, 1.0]),
                margin_scale=0.25,
            )


if __name__ == "__main__":
    unittest.main()
