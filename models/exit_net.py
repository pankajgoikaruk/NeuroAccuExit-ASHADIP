# models/exit_net.py
from __future__ import annotations

from typing import Sequence, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ExitNet(nn.Module):
    """
    Generic multi-exit wrapper (K-exit, C-class).

    Requirements:
      - backbone.forward(x) returns:
          final_feat: Tensor (B, final_dim)
          taps: Sequence[Tensor], each tap is (B, tap_dim_i)

    Output:
      - logits_list: List[Tensor] length K = len(taps) + 1
          [exit1, exit2, ..., exit_{K-1}, final]
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int,                       # REQUIRED (C-class generic)
        tap_dims: Optional[Sequence[int]] = None,
        final_dim: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone

        # --- num_classes must be explicit (no binary hardcode) ---
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2. Got: {self.num_classes}")

        # --- infer tap_dims/final_dim from backbone when not provided ---
        if tap_dims is None:
            if hasattr(backbone, "tap_dims"):
                tap_dims = getattr(backbone, "tap_dims")
            else:
                raise ValueError(
                    "tap_dims was None and backbone has no attribute 'tap_dims'. "
                    "Pass tap_dims explicitly or add backbone.tap_dims."
                )
        if final_dim is None:
            if hasattr(backbone, "final_dim"):
                final_dim = int(getattr(backbone, "final_dim"))
            else:
                raise ValueError(
                    "final_dim was None and backbone has no attribute 'final_dim'. "
                    "Pass final_dim explicitly or add backbone.final_dim."
                )

        self.tap_dims = [int(d) for d in tap_dims]
        self.final_dim = int(final_dim)

        # --- basic validation ---
        if len(self.tap_dims) == 0:
            raise ValueError("tap_dims is empty. Provide at least one tap dim for early exits.")
        if any(d <= 0 for d in self.tap_dims):
            raise ValueError(f"All tap_dims must be positive. Got: {self.tap_dims}")
        if self.final_dim <= 0:
            raise ValueError(f"final_dim must be positive. Got: {self.final_dim}")

        # --- heads ---
        self.exit_heads = nn.ModuleList([nn.Linear(d, self.num_classes) for d in self.tap_dims])
        self.final_head = nn.Linear(self.final_dim, self.num_classes)

    @property
    def num_exits(self) -> int:
        return len(self.exit_heads) + 1

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        final_feat, taps = self.backbone(x)

        if not isinstance(taps, (list, tuple)):
            raise RuntimeError(
                f"Backbone must return taps as a list/tuple, got {type(taps)}."
            )

        if len(taps) != len(self.exit_heads):
            raise RuntimeError(
                f"Backbone returned {len(taps)} taps but ExitNet was built for {len(self.exit_heads)} taps "
                f"(tap_dims={self.tap_dims})."
            )

        logits: List[torch.Tensor] = []
        for head, t in zip(self.exit_heads, taps):
            logits.append(head(t))
        logits.append(self.final_head(final_feat))
        return logits