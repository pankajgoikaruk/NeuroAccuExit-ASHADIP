# models/exit_net.py
from __future__ import annotations

from typing import Sequence, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitNet(nn.Module):
    """
    Generic multi-exit wrapper (K-exit, C-class) with optional local
    exit-to-exit message passing.

    Requirements:
      - backbone.forward(x) returns:
          final_feat: Tensor (B, final_dim)
          taps: Sequence[Tensor], each tap is (B, tap_dim_i)

    Output:
      - logits_list: List[Tensor] length K = len(taps) + 1
          [exit1, exit2, ..., exit_{K-1}, final]

    Optional hint path:
      - After each exit i, project its logits/probabilities (plus optional
        uncertainty statistics) into a small fixed hint vector h_i.
      - Later exit i+1 consumes only the previous hint h_i (local chain).
      - This keeps the design TinyML-friendly and multiclass-compatible.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_classes: int,
        tap_dims: Optional[Sequence[int]] = None,
        final_dim: Optional[int] = None,
        hint_dim: int = 0,
        hint_source: str = "probs",
        hint_detach: bool = True,
        hint_use_stats: bool = True,
    ):
        super().__init__()
        self.backbone = backbone

        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2. Got: {self.num_classes}")

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

        if len(self.tap_dims) == 0:
            raise ValueError("tap_dims is empty. Provide at least one tap dim for early exits.")
        if any(d <= 0 for d in self.tap_dims):
            raise ValueError(f"All tap_dims must be positive. Got: {self.tap_dims}")
        if self.final_dim <= 0:
            raise ValueError(f"final_dim must be positive. Got: {self.final_dim}")

        self.hint_dim = int(hint_dim)
        self.hint_source = str(hint_source).lower().strip()
        self.hint_detach = bool(hint_detach)
        self.hint_use_stats = bool(hint_use_stats)
        self.use_exit_hints = self.hint_dim > 0

        if self.hint_source not in {"probs", "logits"}:
            raise ValueError(f"hint_source must be 'probs' or 'logits'. Got: {hint_source}")

        summary_dim = self.num_classes + (3 if self.hint_use_stats else 0)
        self.hint_summary_dim = summary_dim

        # Heads use local previous hint only: e2<-h1, e3<-h2, ..., final<-h_{K-1}
        self.exit_heads = nn.ModuleList()
        for i, d in enumerate(self.tap_dims):
            in_dim = int(d) + (self.hint_dim if self.use_exit_hints and i > 0 else 0)
            self.exit_heads.append(nn.Linear(in_dim, self.num_classes))

        final_in_dim = self.final_dim + (self.hint_dim if self.use_exit_hints else 0)
        self.final_head = nn.Linear(final_in_dim, self.num_classes)

        if self.use_exit_hints:
            self.hint_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hint_summary_dim, self.hint_dim),
                    nn.ReLU(),
                )
                for _ in range(len(self.tap_dims))
            ])
        else:
            self.hint_projections = nn.ModuleList()

    @property
    def num_exits(self) -> int:
        return len(self.exit_heads) + 1

    def _make_hint(self, logits: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        src = logits
        if self.hint_detach:
            src = src.detach()

        if self.hint_source == "probs":
            base = F.softmax(src, dim=1)
        else:
            base = src

        if self.hint_use_stats:
            probs = F.softmax(src, dim=1)
            conf = probs.max(dim=1, keepdim=True).values
            top2 = probs.topk(k=min(2, probs.size(1)), dim=1).values
            if top2.size(1) == 1:
                margin = top2[:, :1]
            else:
                margin = top2[:, :1] - top2[:, 1:2]
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1, keepdim=True)
            summary = torch.cat([base, conf, margin, entropy], dim=1)
        else:
            summary = base

        return proj(summary)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        final_feat, taps = self.backbone(x)

        if not isinstance(taps, (list, tuple)):
            raise RuntimeError(f"Backbone must return taps as a list/tuple, got {type(taps)}.")
        if len(taps) != len(self.exit_heads):
            raise RuntimeError(
                f"Backbone returned {len(taps)} taps but ExitNet was built for {len(self.exit_heads)} taps "
                f"(tap_dims={self.tap_dims})."
            )

        logits: List[torch.Tensor] = []
        prev_hint: Optional[torch.Tensor] = None

        for i, (head, t) in enumerate(zip(self.exit_heads, taps)):
            head_in = t
            if self.use_exit_hints and prev_hint is not None:
                head_in = torch.cat([head_in, prev_hint], dim=1)

            lg = head(head_in)
            logits.append(lg)

            if self.use_exit_hints:
                prev_hint = self._make_hint(lg, self.hint_projections[i])

        final_in = final_feat
        if self.use_exit_hints and prev_hint is not None:
            final_in = torch.cat([final_in, prev_hint], dim=1)
        logits.append(self.final_head(final_in))
        return logits