import unittest
import numpy as np

from policies.exit3_to_exit5_v019 import Exit3ToExit5Config, decode_genes, select_exit3_or_exit5, to_v018_compatible_policy


class Exit3ToExit5V019Test(unittest.TestCase):
    def test_only_exit3_or_exit5_are_selected(self):
        cfg = Exit3ToExit5Config(0.5, 1.0, 1.0, 1.1, 1.0, 0.0, (0.0, 0.0))
        e2 = np.array([[0.9, 0.1], [0.6, 0.4]], dtype=np.float32)
        e3 = np.array([[0.95, 0.05], [0.51, 0.49]], dtype=np.float32)
        e5 = np.array([[0.96, 0.04], [0.2, 0.8]], dtype=np.float32)
        out = select_exit3_or_exit5(
            exit2_probabilities=e2, exit3_probabilities=e3, exit5_probabilities=e5,
            thresholds_by_exit=[np.array([0.5, 0.5])] * 5,
            exit3_risk_scores=np.array([0.1, 0.1]), config=cfg,
        )
        self.assertTrue(set(out["selected_exit"].tolist()).issubset({3, 5}))

    def test_v018_adapter_disables_other_stops(self):
        cfg = Exit3ToExit5Config(0.8, 0.2, 0.4, 0.7, 2.0, 0.1, (0.2, 0.2))
        adapted = to_v018_compatible_policy(cfg)
        self.assertEqual(adapted.num_exits, 5)
        self.assertGreater(adapted.stages[0].mean_confidence_threshold, 1.0)
        self.assertEqual(adapted.stages[2].mean_confidence_threshold, 0.8)
        self.assertGreater(adapted.stages[3].mean_confidence_threshold, 1.0)

    def test_gene_width(self):
        cfg = decode_genes(np.zeros(16), 10)
        self.assertEqual(len(cfg.per_label_margins), 10)


if __name__ == "__main__":
    unittest.main()
