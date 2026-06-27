# agentic_preprocessing\agents\dataset_auditor_agent.py


"""
Dataset Auditor Agent.

Version 0.3 responsibilities:
- scan raw class folders
- compute basic audio quality indicators
- detect exact duplicate candidates
- separate preprocessing reasons from quality reasons
- add warning codes for borderline samples
- assign accepted / needs_review / rejected / blocked decisions
- write dataset audit reports and split manifests
- preserve all raw data without deletion, movement, or copying
- show CLI progress so long scans do not look stuck
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
from agentic_preprocessing.tools.progress_tool import ProgressBar
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
        show_progress: bool = True,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.out_dir = Path(out_dir)
        self.classes = parse_classes(classes)
        self.policy = policy or {}
        self.show_progress = show_progress

    def _build_class_issue_row(self, class_label: str, issue: str, path: str) -> Dict[str, Any]:
        return {
            "source_file": "",
            "relative_path": path,
            "file_name": "",
            "class_label": class_label,
            "decision": "blocked",
            "safe_after_preprocessing": False,
            "requires_preprocessing": False,
            "preprocessing_actions": [],
            "raw_reason_codes": [issue],
            "preprocessing_reason_codes": [],
            "quality_reason_codes": [],
            "warning_codes": [],
            "decision_reason_codes": [issue],
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
        """
        hash_to_files: Dict[str, List[str]] = defaultdict(list)
        file_to_hash: Dict[str, str] = {}

        progress = ProgressBar(
            total=len(file_records),
            label="Hashing files for duplicate detection",
            enabled=self.show_progress,
            update_every=5,
        )

        for record in file_records:
            path = Path(record.source_file)

            try:
                file_hash = sha256_file(path)
            except Exception:
                file_hash = ""

            file_to_hash[record.source_file] = file_hash

            if file_hash:
                hash_to_files[file_hash].append(record.source_file)

            progress.update(postfix=record.file_name)

        progress.finish(postfix="duplicate hash pass complete")

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
        print("")
        print("Starting DatasetAuditorAgent")
        print(f"Raw root: {self.raw_root}")
        print(f"Output dir: {self.out_dir}")
        print(f"Classes: {', '.join(self.classes)}")
        print("Mode: audit-only, non-destructive")
        print("Routed folders: disabled in V0.3")
        print("")

        print("Discovering audio files...")
        file_records, class_issues = discover_audio_files(
            raw_root=self.raw_root,
            class_names=self.classes,
            audio_extensions=self.policy.get("audio_extensions", [".wav"]),
        )

        print(f"Discovered audio files: {len(file_records)}")
        print(f"Class-level issues: {len(class_issues)}")
        print("")

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

        progress = ProgressBar(
            total=len(file_records),
            label="Auditing audio quality",
            enabled=self.show_progress,
            update_every=5,
        )

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
                "safe_after_preprocessing": bool(stats.get("safe_after_preprocessing", False)),
                "requires_preprocessing": bool(stats.get("requires_preprocessing", False)),
                "preprocessing_actions": stats.get("preprocessing_actions", []),
                "raw_reason_codes": stats.get("raw_reason_codes", []),
                "preprocessing_reason_codes": stats.get("preprocessing_reason_codes", []),
                "quality_reason_codes": stats.get("quality_reason_codes", []),
                "warning_codes": stats.get("warning_codes", []),
                "decision_reason_codes": stats.get("decision_reason_codes", []),
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
            progress.update(postfix=f"{record.class_label}/{record.file_name}")

        progress.finish(postfix="audio audit complete")

        print("")
        print("Writing report bundle and split manifests...")
        outputs = write_report_bundle(
            rows=rows,
            out_dir=self.out_dir,
            report_prefix="dataset_audit_agent_report",
        )

        print("Report writing complete.")
        print("")

        return outputs