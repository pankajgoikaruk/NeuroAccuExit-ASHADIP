# agentic_preprocessing\agents\dataset_auditor_agent.py

"""
Dataset Auditor Agent.

This is the first research-grade agentic preprocessing component.

Responsibilities:
- scan raw class folders
- compute basic audio quality indicators
- detect exact duplicate candidates
- assign accepted / needs_review / rejected / blocked decisions
- write dataset audit reports
- preserve all raw data without deletion or movement
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentic_preprocessing.tools.audio_quality_tool import (
    AGENT_VERSION,
    analyse_audio_file,
    sha256_file,
)
from agentic_preprocessing.tools.audio_scan_tool import discover_audio_files, parse_classes
from agentic_preprocessing.tools.report_tool import write_report_bundle


class DatasetAuditorAgent:
    """
    Non-destructive dataset audit agent.
    """

    def __init__(
        self,
        raw_root: str | Path,
        out_dir: str | Path,
        classes: str | Iterable[str],
        policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.out_dir = Path(out_dir)
        self.classes = parse_classes(classes)
        self.policy = policy or {}

    def _build_class_issue_row(self, class_label: str, issue: str, path: str) -> Dict[str, Any]:
        return {
            "source_file": "",
            "relative_path": path,
            "file_name": "",
            "class_label": class_label,
            "decision": "blocked",
            "reason_codes": [issue],
            "safe_to_train": False,
            "readable": False,
            "error": issue,
            "duration_sec": "",
            "sample_rate": "",
            "channels": "",
            "sample_width": "",
            "n_frames": "",
            "rms_db": "",
            "peak_db": "",
            "silence_ratio": "",
            "clipping_ratio": "",
            "speech_activity_ratio": "",
            "sha256": "",
            "duplicate_group": "",
            "duplicate_count": "",
            "background_music_score": "",
            "background_noise_score": "",
            "wrong_speaker_suspicion": "",
            "agent_version": AGENT_VERSION,
        }

    def _compute_duplicate_groups(self, file_records: List[Any]) -> Dict[str, Dict[str, Any]]:
        """
        Detect exact duplicate candidates using SHA256.

        Returns mapping:
        source_file -> duplicate metadata
        """
        hash_to_files: Dict[str, List[str]] = defaultdict(list)
        file_to_hash: Dict[str, str] = {}

        for record in file_records:
            path = Path(record.source_file)
            try:
                file_hash = sha256_file(path)
            except Exception:
                file_hash = ""

            file_to_hash[record.source_file] = file_hash

            if file_hash:
                hash_to_files[file_hash].append(record.source_file)

        duplicate_meta: Dict[str, Dict[str, Any]] = {}
        duplicate_index = 1

        for file_hash, files in hash_to_files.items():
            if len(files) <= 1:
                for f in files:
                    duplicate_meta[f] = {
                        "sha256": file_hash,
                        "duplicate_group": "",
                        "duplicate_count": 1,
                        "is_duplicate_candidate": False,
                    }
                continue

            group_name = f"dup_exact_{duplicate_index:04d}"
            duplicate_index += 1

            for f in files:
                duplicate_meta[f] = {
                    "sha256": file_hash,
                    "duplicate_group": group_name,
                    "duplicate_count": len(files),
                    "is_duplicate_candidate": True,
                }

        # Files that failed hash generation
        for source_file, file_hash in file_to_hash.items():
            if source_file not in duplicate_meta:
                duplicate_meta[source_file] = {
                    "sha256": file_hash,
                    "duplicate_group": "",
                    "duplicate_count": "",
                    "is_duplicate_candidate": False,
                }

        return duplicate_meta

    def run(self) -> Dict[str, str]:
        file_records, class_issues = discover_audio_files(
            raw_root=self.raw_root,
            class_names=self.classes,
            audio_extensions=self.policy.get("audio_extensions", [".wav"]),
        )

        rows: List[Dict[str, Any]] = []

        for issue in class_issues:
            rows.append(
                self._build_class_issue_row(
                    class_label=issue["class_label"],
                    issue=issue["issue"],
                    path=issue["path"],
                )
            )

        duplicate_meta = self._compute_duplicate_groups(file_records)

        for record in file_records:
            source_path = Path(record.source_file)
            meta = duplicate_meta.get(record.source_file, {})

            extra_reasons = []
            if meta.get("is_duplicate_candidate", False):
                extra_reasons.append("exact_duplicate_candidate")

            stats = analyse_audio_file(
                source_path,
                policy=self.policy,
                extra_reasons=extra_reasons,
            )

            row: Dict[str, Any] = {
                "source_file": record.source_file,
                "relative_path": record.relative_path,
                "file_name": record.file_name,
                "class_label": record.class_label,
                "decision": stats.get("decision", "blocked"),
                "reason_codes": stats.get("reason_codes", []),
                "safe_to_train": bool(stats.get("safe_to_train", False)),
                "readable": bool(stats.get("readable", False)),
                "error": stats.get("error", ""),
                "duration_sec": stats.get("duration_sec", ""),
                "sample_rate": stats.get("sample_rate", ""),
                "channels": stats.get("channels", ""),
                "sample_width": stats.get("sample_width", ""),
                "n_frames": stats.get("n_frames", ""),
                "rms_db": stats.get("rms_db", ""),
                "peak_db": stats.get("peak_db", ""),
                "silence_ratio": stats.get("silence_ratio", ""),
                "clipping_ratio": stats.get("clipping_ratio", ""),
                "speech_activity_ratio": stats.get("speech_activity_ratio", ""),
                "sha256": meta.get("sha256", ""),
                "duplicate_group": meta.get("duplicate_group", ""),
                "duplicate_count": meta.get("duplicate_count", ""),
                "background_music_score": "",
                "background_noise_score": "",
                "wrong_speaker_suspicion": "",
                "agent_version": AGENT_VERSION,
            }

            rows.append(row)

        return write_report_bundle(
            rows=rows,
            out_dir=self.out_dir,
            report_prefix="dataset_audit_agent_report",
        )