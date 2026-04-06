# models/exit_net.py

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class ExitNet(nn.Module):
    """
    Generic K-exit / C-class wrapper.

    Expected backbone contract:
        backbone(x) -> (final_feat, taps)

    where:
        final_feat: Tensor of shape (B, final_dim)
        taps: list/tuple of Tensors, each of shape (B, tap_dim_i)

    Output:
        [logits_exit1, logits_exit2, ..., logits_exit_{K-1}, logits_final]

    Notes:
    - K = len(taps) + 1
    - Supports both:
        * old hard-coded style by explicitly passing tap_dims/final_dim
        * new generic style by reading backbone.tap_dims / backbone.final_dim
    """

    def __init__(
        self,
        backbone: nn.Module,
        tap_dims: Optional[Sequence[int]] = None,
        final_dim: Optional[int] = None,
        num_classes: int = 2,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_classes = int(num_classes)

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")

        # Prefer dynamic metadata from backbone if available
        if tap_dims is None:
            if hasattr(backbone, "tap_dims"):
                tap_dims = getattr(backbone, "tap_dims")
            else:
                raise ValueError(
                    "tap_dims not provided and backbone has no attribute 'tap_dims'."
                )

        if final_dim is None:
            if hasattr(backbone, "final_dim"):
                final_dim = getattr(backbone, "final_dim")
            else:
                raise ValueError(
                    "final_dim not provided and backbone has no attribute 'final_dim'."
                )

        self.tap_dims = [int(d) for d in tap_dims]
        self.final_dim = int(final_dim)

        if len(self.tap_dims) == 0:
            raise ValueError("tap_dims must contain at least one tap dimension.")
        if any(d <= 0 for d in self.tap_dims):
            raise ValueError(f"All tap_dims must be positive, got {self.tap_dims}")
        if self.final_dim <= 0:
            raise ValueError(f"final_dim must be positive, got {self.final_dim}")

        # Early-exit heads
        self.exit_heads = nn.ModuleList(
            [nn.Linear(dim, self.num_classes) for dim in self.tap_dims]
        )

        # Final head
        self.final_head = nn.Linear(self.final_dim, self.num_classes)

        # Optional backward-compatible aliases
        for i, head in enumerate(self.exit_heads, start=1):
            setattr(self, f"exit{i}", head)
        self.final = self.final_head

    @property
    def num_exits(self) -> int:
        return len(self.exit_heads) + 1

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        final_feat, taps = self.backbone(x)

        if not isinstance(taps, (list, tuple)):
            raise RuntimeError(
                f"Backbone must return taps as list/tuple, got {type(taps)}"
            )

        if len(taps) != len(self.exit_heads):
            raise RuntimeError(
                f"Backbone returned {len(taps)} taps, but ExitNet was built for "
                f"{len(self.exit_heads)} taps (tap_dims={self.tap_dims})."
            )

        logits: List[torch.Tensor] = []

        for head, tap in zip(self.exit_heads, taps):
            logits.append(head(tap))

        logits.append(self.final_head(final_feat))
        return logits