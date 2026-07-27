from __future__ import annotations

import unittest

import numpy as np

from policies.targeted_exit3_to_exit5_v019 import (
    TargetedExit3ToExit5Config,
    decode_targeted_genes,
    derive_grouped_continuation_risk,
    make_targeted_bounds,
    targeted_select,
    targeted_stop_mask,
)


class TargetedExit3ToExit5Test(unittest.TestCase):
    def config(self) -> TargetedExit3ToExit5Config:
        return TargetedExit3ToExit5Config(
            mean_confidence_threshold=0.90,
            max_probability_delta=0.20,
            max_label_risk=0.50,
            risk_score_threshold=0.60,
            risk_margin_multiplier=2.0,
            risk_uncertainty_band=0.10,
            per_label_margins=(0.20, 0.20),
        )

    def test_selected_exits_are_only_three_or_five(self) -> None:
        exit2 = np.asarray([[0.90, 0.10], [0.51, 0.49]], np.float32)
        exit3 = np.asarray([[0.95, 0.05], [0.55, 0.45]], np.float32)
        exit5 = np.asarray([[0.96, 0.04], [0.90, 0.10]], np.float32)
        result = targeted_select(
            exit2_probabilities=exit2,
            exit3_probabilities=exit3,
            exit5_probabilities=exit5,
            exit2_thresholds=np.asarray([0.5, 0.5]),
            exit3_thresholds=np.asarray([0.5, 0.5]),
            risk_scores=np.asarray([0.1, 0.1]),
            config=self.config(),
        )
        self.assertEqual(result["selected_exit"].tolist(), [3, 5])

    def test_exit2_to_exit3_instability_blocks_stop(self) -> None:
        diagnostics = {
            "prediction": np.asarray([[1, 0]], np.int8),
            "non_empty": np.asarray([True]),
            "mean_binary_confidence": np.asarray([0.95]),
            "decision_margin": np.asarray([[0.45, 0.45]], np.float32),
            "maximum_probability_delta": np.asarray([0.10]),
            "label_set_stability": np.asarray([False]),
        }
        stop = targeted_stop_mask(
            diagnostics=diagnostics,
            risk_scores=np.asarray([0.1, 0.1]),
            config=self.config(),
        )
        self.assertFalse(bool(stop[0]))

    def test_high_risk_positive_requires_stricter_margin(self) -> None:
        diagnostics = {
            "prediction": np.asarray([[1, 0]], np.int8),
            "non_empty": np.asarray([True]),
            "mean_binary_confidence": np.asarray([0.95]),
            "decision_margin": np.asarray([[0.30, 0.45]], np.float32),
            "maximum_probability_delta": np.asarray([0.10]),
            "label_set_stability": np.asarray([True]),
        }
        stop = targeted_stop_mask(
            diagnostics=diagnostics,
            risk_scores=np.asarray([1.0, 0.1]),
            config=self.config(),
        )
        self.assertFalse(bool(stop[0]))

    def test_low_risk_confident_sample_can_stop(self) -> None:
        diagnostics = {
            "prediction": np.asarray([[1, 0]], np.int8),
            "non_empty": np.asarray([True]),
            "mean_binary_confidence": np.asarray([0.95]),
            "decision_margin": np.asarray([[0.45, 0.45]], np.float32),
            "maximum_probability_delta": np.asarray([0.10]),
            "label_set_stability": np.asarray([True]),
        }
        stop = targeted_stop_mask(
            diagnostics=diagnostics,
            risk_scores=np.asarray([0.1, 0.1]),
            config=self.config(),
        )
        self.assertTrue(bool(stop[0]))

    def test_grouped_risk_profile_has_valid_shape(self) -> None:
        y = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], np.int8)
        exit3 = np.asarray([[0.4, 0.1], [0.9, 0.1], [0.1, 0.4], [0.1, 0.9]], np.float32)
        exit5 = np.asarray([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]], np.float32)
        profile = derive_grouped_continuation_risk(
            y_true=y,
            exit3_probabilities=exit3,
            exit5_probabilities=exit5,
            exit3_thresholds=np.asarray([0.5, 0.5]),
            exit5_thresholds=np.asarray([0.5, 0.5]),
            group_ids=np.asarray(["a", "b", "c", "d"]),
            cv_folds=2,
        )
        self.assertEqual(profile["risk_scores"].shape, (2,))
        self.assertTrue(np.all(profile["risk_scores"] >= 0.05))
        self.assertTrue(np.all(profile["risk_scores"] <= 1.0))

    def test_gene_bounds_and_decoding(self) -> None:
        lower, upper = make_targeted_bounds(num_labels=3)
        self.assertEqual(lower.shape, (9,))
        config = decode_targeted_genes((lower + upper) / 2.0, num_labels=3)
        self.assertEqual(len(config.per_label_margins), 3)


if __name__ == "__main__":
    unittest.main()
