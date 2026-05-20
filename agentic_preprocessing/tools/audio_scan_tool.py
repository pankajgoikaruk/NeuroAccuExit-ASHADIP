# agentic_preprocessing\tools\audio_scan_tool.py

"""
Audio dataset scanning utilities for Agentic AI preprocessing.

This module only discovers files and class-folder issues.
It does not modify, move, or delete files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_AUDIO_EXTENSIONS = [".wav"]


@dataclass
class AudioFileRecord:
    source_file: str
    file_name: str
    class_label: str
    relative_path: str


def parse_classes(classes: str | Sequence[str]) -> List[str]:
    if isinstance(classes, str):
        return [x.strip() for x in classes.split(",") if x.strip()]
    return [str(x).strip() for x in classes if str(x).strip()]


def discover_audio_files(
    raw_root: Path,
    class_names: Iterable[str],
    audio_extensions: Iterable[str] = DEFAULT_AUDIO_EXTENSIONS,
) -> Tuple[List[AudioFileRecord], List[Dict[str, str]]]:
    """
    Discover audio files for target classes.

    Returns:
    - file records
    - class-level issues such as missing folders or no files
    """
    raw_root = Path(raw_root)
    extensions = {ext.lower() for ext in audio_extensions}

    records: List[AudioFileRecord] = []
    issues: List[Dict[str, str]] = []

    for class_name in class_names:
        class_dir = raw_root / class_name

        if not class_dir.exists():
            issues.append(
                {
                    "class_label": class_name,
                    "issue": "class_folder_missing",
                    "path": str(class_dir),
                }
            )
            continue

        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not files:
            issues.append(
                {
                    "class_label": class_name,
                    "issue": "no_audio_files_found",
                    "path": str(class_dir),
                }
            )
            continue

        for path in files:
            try:
                rel_path = str(path.relative_to(raw_root))
            except ValueError:
                rel_path = str(path)

            records.append(
                AudioFileRecord(
                    source_file=str(path),
                    file_name=path.name,
                    class_label=class_name,
                    relative_path=rel_path,
                )
            )

    return records, issues