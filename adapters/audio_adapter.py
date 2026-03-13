# adapters/audio_adapter.py
from __future__ import annotations

from typing import List, Sequence
import torch
import torch.nn as nn


class TinyAudioCNN(nn.Module):
    """
    5-block TinyAudioCNN backbone with configurable tap points.

    Backward compatible default:
        tap_blocks=(1, 3) -> taps after block1 (C=16) and block3 (C=32)
        final_feat after block5 (C=64)
        => total exits = 2 taps + 1 final = 3 exits (same as v0.4.2)

    Example for 5 exits total (4 early + final):
        tap_blocks=(1, 2, 3, 4) -> tap_dims=[16, 24, 32, 48], final_dim=64
        => total exits = 4 taps + 1 final = 5 exits
    """

    # Output channels after each block (must match the conv out_channels below)
    _BLOCK_CHANNELS = [16, 24, 32, 48, 64]

    def __init__(self, n_mels: int = 64, tap_blocks: Sequence[int] = (1, 3)):
        super().__init__()
        self.n_mels = int(n_mels)

        # -----------------------
        # Block 1 (cheap + early pooling)
        # -----------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # -----------------------
        # Block 2 (cheap + pooling)
        # -----------------------
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # -----------------------
        # Block 3 (mid-level features)
        # -----------------------
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # -----------------------
        # Block 4 (deeper features)
        # -----------------------
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # -----------------------
        # Block 5 (final embedding + global pooling)
        # -----------------------
        self.block5 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B,64,1,1)
        )

        # Store blocks in order for iteration
        self._blocks: List[nn.Module] = [self.block1, self.block2, self.block3, self.block4, self.block5]

        # -----------------------
        # Tap configuration (where to place early exits)
        # We allow taps only after blocks 1..4.
        # Block5 is reserved for final_feat.
        # -----------------------
        tb = [int(b) for b in tap_blocks]
        if len(tb) == 0:
            raise ValueError("tap_blocks must contain at least one tap block (e.g., (1,3) or (1,2,3,4)).")
        if any(b < 1 or b > 4 for b in tb):
            raise ValueError(f"tap_blocks must be in [1..4]. Got: {tap_blocks}")

        self.tap_blocks = sorted(set(tb))
        self.tap_dims = [self._BLOCK_CHANNELS[b - 1] for b in self.tap_blocks]
        self.final_dim = self._BLOCK_CHANNELS[-1]

    @staticmethod
    def _tap_pool(feat_map: torch.Tensor) -> torch.Tensor:
        """
        Convert (B,C,H,W) feature map into (B,C) vector:
          max over time (W), then mean over freq (H).
        """
        return torch.amax(feat_map, dim=-1).mean(-1)

    def forward(self, x: torch.Tensor):
        taps: List[torch.Tensor] = []

        f = x
        for i, blk in enumerate(self._blocks, start=1):
            f = blk(f)
            if i in self.tap_blocks:
                taps.append(self._tap_pool(f))

        # final block output is (B,64,1,1) because of AdaptiveAvgPool2d
        final_feat = f.view(f.size(0), -1)  # (B,64)
        return final_feat, taps