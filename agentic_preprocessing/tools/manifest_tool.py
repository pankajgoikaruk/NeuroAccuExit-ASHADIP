# agentic_preprocessing\tools\manifest_tool.py

"""
Manifest utilities.

Planned responsibilities:
- read/write agentic decision manifests
- create accepted-only manifests
- create manually-approved manifests
- preserve source_file, parent_clip_id, segment_id, and class traceability

This is intentionally minimal in V0.1.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def read_csv_manifest(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))