# agentic_preprocessing\tools\progress_tool.py

"""
Small dependency-free CLI progress bar for Agentic AI preprocessing.

Why not tqdm?
- Keeps the project lightweight.
- Avoids adding new dependencies.
- Works directly in PowerShell.
"""

from __future__ import annotations

import sys
import time
from typing import Optional


class ProgressBar:
    def __init__(
        self,
        total: int,
        label: str = "Processing",
        width: int = 32,
        enabled: bool = True,
        update_every: int = 1,
    ) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.width = max(int(width), 10)
        self.enabled = enabled
        self.update_every = max(int(update_every), 1)

        self.current = 0
        self.start_time = time.time()
        self.last_render_len = 0

    def _format_time(self, seconds: float) -> str:
        seconds = max(float(seconds), 0.0)

        if seconds < 60:
            return f"{seconds:5.1f}s"

        minutes = int(seconds // 60)
        seconds = int(seconds % 60)

        if minutes < 60:
            return f"{minutes:02d}m{seconds:02d}s"

        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

    def _render(self, postfix: str = "") -> None:
        if not self.enabled:
            return

        elapsed = time.time() - self.start_time

        if self.total <= 0:
            message = f"{self.label}: {self.current} items | elapsed {self._format_time(elapsed)}"
            if postfix:
                message += f" | {postfix}"

            self._write_line(message)
            return

        fraction = min(max(self.current / self.total, 0.0), 1.0)
        filled = int(self.width * fraction)
        bar = "█" * filled + "░" * (self.width - filled)
        percent = fraction * 100.0

        if self.current > 0:
            rate = self.current / max(elapsed, 1e-9)
            remaining = (self.total - self.current) / max(rate, 1e-9)
        else:
            rate = 0.0
            remaining = 0.0

        message = (
            f"{self.label}: |{bar}| "
            f"{self.current}/{self.total} "
            f"({percent:6.2f}%) "
            f"elapsed {self._format_time(elapsed)} "
            f"eta {self._format_time(remaining)} "
            f"{rate:6.1f}/s"
        )

        if postfix:
            message += f" | {postfix}"

        self._write_line(message)

    def _write_line(self, message: str) -> None:
        padding = max(self.last_render_len - len(message), 0)
        sys.stdout.write("\r" + message + (" " * padding))
        sys.stdout.flush()
        self.last_render_len = len(message)

    def update(self, step: int = 1, postfix: str = "") -> None:
        self.current += int(step)

        should_render = (
            self.current == 1
            or self.current >= self.total
            or self.current % self.update_every == 0
        )

        if should_render:
            self._render(postfix=postfix)

    def finish(self, postfix: str = "done") -> None:
        if self.total > 0:
            self.current = self.total

        self._render(postfix=postfix)

        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()