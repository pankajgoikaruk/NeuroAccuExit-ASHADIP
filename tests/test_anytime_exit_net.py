# tests/test_anytime_exit_net.py

from __future__ import annotations

import unittest

import torch

from adapters.audio_adapter import TinyAudioCNN
from models.anytime_exit_net import AnytimeExitNet
from models.exit_net import ExitNet


class AnytimeExitNetEquivalenceTest(unittest.TestCase):
    def _build_model(
        self,
        *,
        tap_blocks: tuple[int, ...],
        hint_dim: int = 0,
    ) -> ExitNet:
        torch.manual_seed(17)

        backbone = TinyAudioCNN(
            n_mels=64,
            tap_blocks=tap_blocks,
        )
        model = ExitNet(
            backbone=backbone,
            tap_dims=backbone.tap_dims,
            final_dim=backbone.final_dim,
            num_classes=10,
            hint_dim=hint_dim,
            hint_source="probs",
            hint_detach=True,
            hint_use_stats=True,
            hint_activation="sigmoid",
        )
        model.eval()
        return model

    def _assert_staged_matches_full(
        self,
        *,
        tap_blocks: tuple[int, ...],
        hint_dim: int = 0,
    ) -> None:
        model = self._build_model(
            tap_blocks=tap_blocks,
            hint_dim=hint_dim,
        )
        anytime = AnytimeExitNet(model)
        anytime.eval()

        torch.manual_seed(23)
        x = torch.randn(3, 1, 64, 100)

        with torch.no_grad():
            full_outputs = model(x)
            staged_outputs = anytime.forward_all_staged(x)

        self.assertEqual(len(full_outputs), len(staged_outputs))
        for exit_no, (full_logits, staged_logits) in enumerate(
            zip(full_outputs, staged_outputs),
            start=1,
        ):
            torch.testing.assert_close(
                staged_logits,
                full_logits,
                rtol=1e-6,
                atol=1e-7,
                msg=f"Staged logits differ at Exit {exit_no}",
            )

    def test_three_exit_no_hint_equivalence(self) -> None:
        self._assert_staged_matches_full(
            tap_blocks=(1, 3),
            hint_dim=0,
        )

    def test_five_exit_no_hint_equivalence(self) -> None:
        self._assert_staged_matches_full(
            tap_blocks=(1, 2, 3, 4),
            hint_dim=0,
        )

    def test_three_exit_hint_equivalence(self) -> None:
        # Hint-pass is not selected for the current research baseline, but the
        # staged wrapper remains backward-compatible with existing hint models.
        self._assert_staged_matches_full(
            tap_blocks=(1, 3),
            hint_dim=8,
        )

    def test_three_exit_state_progression(self) -> None:
        model = self._build_model(tap_blocks=(1, 3), hint_dim=0)
        anytime = AnytimeExitNet(model)
        anytime.eval()

        x = torch.randn(2, 1, 64, 100)

        with torch.no_grad():
            _, state1 = anytime.start(x)
            _, state2 = anytime.continue_from(state1)
            _, state3 = anytime.continue_from(state2)

        self.assertEqual(state1.block_index, 1)
        self.assertEqual(state1.completed_exit_number, 1)
        self.assertFalse(state1.finished)

        self.assertEqual(state2.block_index, 3)
        self.assertEqual(state2.completed_exit_number, 2)
        self.assertFalse(state2.finished)

        self.assertEqual(state3.block_index, 5)
        self.assertEqual(state3.completed_exit_number, 3)
        self.assertTrue(state3.finished)

        with self.assertRaises(RuntimeError):
            anytime.continue_from(state3)


if __name__ == "__main__":
    unittest.main()
