from __future__ import annotations

import unittest
import numpy as np

from policies.strict_sequential_anytime_exit_v018 import (
    StrictSequentialPolicyConfig,
    StrictSequentialStageConfig,
    decode_strict_genes,
    derive_strict_continuation_profile,
    make_strict_bounds,
    strict_sequential_select,
)


def stage(conf=.80, boost=0.0, risk_threshold=.70, multiplier=2.0, band=.10, labels=1, stability=False):
    return StrictSequentialStageConfig(
        mean_confidence_threshold=conf,
        max_probability_delta=1.0,
        max_label_risk=1.0,
        risk_score_threshold=risk_threshold,
        risk_margin_multiplier=multiplier,
        risk_uncertainty_band=band,
        exit1_confidence_boost=boost,
        per_label_margins=tuple([0.05] * labels),
        require_previous_label_stability=stability,
    )


class StrictSequentialV018Test(unittest.TestCase):
    def test_exit1_boost_blocks_borderline_easy_sample(self):
        probs=[np.asarray([[0.90]],np.float32),np.asarray([[0.95]],np.float32),np.asarray([[0.96]],np.float32)]
        config=StrictSequentialPolicyConfig(3,(stage(conf=.80,boost=.15),stage(stability=True)))
        result=strict_sequential_select(exit_probabilities=probs,thresholds_by_exit=[np.asarray([.5])]*3,risk_scores_by_exit=np.zeros((2,1),np.float32),config=config)
        self.assertEqual(int(result["selected_exit"][0]),2)

    def test_high_risk_uncertainty_forces_continuation(self):
        probs=[np.asarray([[0.62]],np.float32),np.asarray([[0.90]],np.float32),np.asarray([[0.95]],np.float32)]
        config=StrictSequentialPolicyConfig(3,(stage(conf=.50,risk_threshold=.5,band=.20),stage(conf=.99,stability=True)))
        risk=np.asarray([[1.0],[0.0]],np.float32)
        result=strict_sequential_select(exit_probabilities=probs,thresholds_by_exit=[np.asarray([.5])]*3,risk_scores_by_exit=risk,config=config)
        self.assertEqual(int(result["selected_exit"][0]),3)

    def test_confident_high_risk_label_can_exit(self):
        probs=[np.asarray([[0.99]],np.float32),np.asarray([[0.99]],np.float32),np.asarray([[0.99]],np.float32)]
        config=StrictSequentialPolicyConfig(3,(stage(conf=.80,risk_threshold=.5,band=.20),stage(stability=True)))
        risk=np.ones((2,1),np.float32)
        result=strict_sequential_select(exit_probabilities=probs,thresholds_by_exit=[np.asarray([.5])]*3,risk_scores_by_exit=risk,config=config)
        self.assertEqual(int(result["selected_exit"][0]),1)

    def test_validation_profile_ranks_corrected_label_higher(self):
        truth=np.asarray([[1,0],[1,0],[0,1],[0,1]],np.int8)
        exit1=np.asarray([[.1,.1],[.1,.1],[.1,.9],[.1,.9]],np.float32)
        final=np.asarray([[.9,.1],[.9,.1],[.1,.9],[.1,.9]],np.float32)
        profile=derive_strict_continuation_profile(y_true=truth,exit_probabilities=[exit1,final],thresholds_by_exit=[np.asarray([.5,.5])]*2)
        self.assertGreater(float(profile["risk_scores"][0,0]),float(profile["risk_scores"][0,1]))
        self.assertEqual(int(profile["correction_counts"][0,0]),2)

    def test_five_exit_gene_dimensions(self):
        lower,upper=make_strict_bounds(num_exits=5,num_labels=10)
        self.assertEqual(lower.shape,(68,))
        config=decode_strict_genes((lower+upper)/2,num_exits=5,num_labels=10)
        self.assertEqual(len(config.stages),4)
        self.assertEqual(len(config.stages[0].per_label_margins),10)

    def test_minimum_exit_disables_exit1_only(self):
        probs=[np.asarray([[.99]],np.float32),np.asarray([[.99]],np.float32),np.asarray([[.99]],np.float32)]
        config=StrictSequentialPolicyConfig(3,(stage(),stage(stability=True)))
        result=strict_sequential_select(exit_probabilities=probs,thresholds_by_exit=[np.asarray([.5])]*3,risk_scores_by_exit=np.zeros((2,1),np.float32),config=config,minimum_exit=2)
        self.assertEqual(int(result["selected_exit"][0]),2)

if __name__=="__main__":
    unittest.main()
