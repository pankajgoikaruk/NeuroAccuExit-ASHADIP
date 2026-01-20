import torch
import torch.nn as nn


class TinyAudioCNN(nn.Module):
    """
    TinyAudioCNN backbone extended from 3 → 5 convolutional blocks.

    Goal (Option A: clean comparison):
      - Keep ONLY 3 exits, but move the final exit deeper.
      - Attach exits at depths: 1 / 3 / 5
        * Exit1: after block1  (very early, cheap)
        * Exit2: after block3  (mid-depth)
        * Exit3: after block5  (final, strongest)

    Input:
      x: (B, 1, M, T)  where:
        B = batch size
        M = n_mels (e.g., 64)
        T = frames (time)

    Output format (unchanged from your old 3-block design):
      return final_feat, taps
        final_feat: (B, 64)       -> used for final head
        taps: [t1, t2]
          t1: (B, 16)             -> exit1 head input
          t2: (B, 32)             -> exit2 head input

    Why this design is convenient:
      - ExitNet does NOT need to change if we keep dims (16, 32, 64).
      - Most of your pipeline remains identical, so comparisons are clean.
    """

    def __init__(self, n_mels: int = 64):
        super().__init__()

        # Channel schedule (TinyML-friendly gradual growth)
        # 1 -> 16 -> 24 -> 32 -> 48 -> 64
        #
        # We are NOT adding extra downsampling yet (that comes later),
        # so we keep pooling exactly like before in blocks 1–2 only.

        # -----------------------
        # Block 1 (cheap + early pooling)
        # -----------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # preserves spatial size
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),             # halves (M,T)
        )

        # -----------------------
        # Block 2 (cheap + pooling)
        # -----------------------
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),             # halves again
        )

        # -----------------------
        # Block 3 (mid-depth, NO pooling yet)
        # Exit2 tap comes from AFTER this block.
        # -----------------------
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # -----------------------
        # Block 4 (deeper features, still NO pooling)
        # -----------------------
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # -----------------------
        # Block 5 (final block)
        # Add global pooling to produce a fixed-size embedding (B,64).
        # -----------------------
        self.block5 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                 # -> (B,64,1,1)
        )

    @staticmethod
    def _tap_pool(feat_map: torch.Tensor) -> torch.Tensor:
        """
        Convert a 2D feature map (B,C,H,W) into a compact vector (B,C)
        in a way consistent with your previous code.

        Strategy (cheap + stable):
          1) max over time axis (W)
          2) mean over frequency axis (H)

        This roughly means:
          "for each channel, take strongest evidence over time,
           then average across frequency bins"
        """
        # feat_map: (B, C, H, W)
        return torch.amax(feat_map, dim=-1).mean(-1)  # -> (B, C)

    def forward(self, x: torch.Tensor):
        """
        Forward pass, returning:
          final_feat: (B, 64)
          taps: [t1 (B,16), t2 (B,32)]
        """

        # After block1:
        # f1: (B,16,M/2,T/2)
        f1 = self.block1(x)

        # After block2:
        # f2: (B,24,M/4,T/4)
        f2 = self.block2(f1)

        # After block3:
        # f3: (B,32,M/4,T/4)  <-- Exit2 tap source
        f3 = self.block3(f2)

        # After block4:
        # f4: (B,48,M/4,T/4)
        f4 = self.block4(f3)

        # After block5:
        # f5: (B,64,1,1)
        f5 = self.block5(f4)

        # Exit taps (Option A: exits at 1/3/5)
        t1 = self._tap_pool(f1)              # (B,16) after block1
        t2 = self._tap_pool(f3)              # (B,32) after block3
        final_feat = f5.view(f5.size(0), -1) # (B,64) after block5

        return final_feat, [t1, t2]
