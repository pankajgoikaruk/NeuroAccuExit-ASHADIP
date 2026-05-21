# agentic_preprocessing/agents/dataset_builder_agent.py


"""
DatasetBuilderAgent

Builds raw5_agentic_cleaned from accepted_manifest.csv.

Safety:
- Does not modify raw audio.
- Writes cleaned copies only.
- Creates build manifest and summary.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from agentic_preprocessing.tools.audio_transform_tool import transform_to_clean_wav


class DatasetBuilderAgent:
    def __init__(
        self,
        accepted_manifest: str | Path,
        out_root: str | Path,
        raw_root: str | Path | None = None,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        apply: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.accepted_manifest = Path(accepted_manifest)
        self.out_root = Path(out_root)
        self.raw_root = Path(raw_root) if raw_root else None
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.apply = apply
        self.overwrite = overwrite


    def _first_value(self, row: Dict[str, str], keys: list[str]) -> str:
        for key in keys:
            value = row.get(key, "")
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""



    def _read_manifest(self) -> List[Dict[str, str]]:
        if not self.accepted_manifest.exists():
            raise FileNotFoundError(f"Accepted manifest not found: {self.accepted_manifest}")

        with self.accepted_manifest.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))



    def _resolve_source_path(self, row: Dict[str, str]) -> Path:
        candidate_paths = [
            self._first_value(row, [
                "source_path",
                "audit__source_path",
                "audit__file_path",
                "audit__path",
                "audit__audio_path",
                "audit__filepath",
                "file_path",
                "path",
                "audio_path",
                "filepath",
            ])
        ]

        for candidate in candidate_paths:
            if candidate:
                p = Path(candidate)
                if p.is_file():
                    return p

        class_name = self._first_value(row, [
            "class_name",
            "audit__class_name",
            "audit__label",
            "audit__class",
            "audit__folder",
            "audit__class_dir",
        ])

        source_file = self._first_value(row, [
            "source_file",
            "audit__source_file",
            "audit__filename",
            "audit__file",
            "audit__old_name",
            "filename",
            "file",
            "old_name",
        ])

        if self.raw_root and class_name and source_file:
            p = self.raw_root / class_name / source_file
            if p.is_file():
                return p

        rel_path = self._first_value(row, [
            "rel_path",
            "audit__rel_path",
            "audit__relative_path",
        ])

        if self.raw_root and rel_path:
            p = self.raw_root / rel_path
            if p.is_file():
                return p

        return Path("")

    def _target_path(self, row: Dict[str, str], source_path: Path) -> Path:
        class_name = self._first_value(row, [
            "class_name",
            "class_label",
            "audit__class_name",
            "audit__class_label",
            "audit__label",
            "audit__class",
            "audit__folder",
            "audit__class_dir",
        ])

        if not class_name and source_path and source_path.parent:
            class_name = source_path.parent.name

        if not class_name:
            class_name = "unknown"

        source_file = self._first_value(row, [
            "source_file",
            "file_name",
            "audit__source_file",
            "audit__file_name",
            "audit__filename",
            "audit__file",
            "audit__old_name",
            "filename",
            "file",
            "old_name",
        ])

        if not source_file and source_path and source_path.name:
            source_file = source_path.name

        output_name = Path(source_file).with_suffix(".wav").name if source_file else "missing_source.wav"

        return self.out_root / class_name / output_name

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run(self) -> Dict[str, Any]:
        rows = self._read_manifest()

        print("")
        print("Starting DatasetBuilderAgent")
        print("----------------------------")
        print(f"Accepted manifest: {self.accepted_manifest}")
        print(f"Output root:       {self.out_root}")
        print(f"Mode:              {'APPLY / BUILD DATASET' if self.apply else 'DRY RUN ONLY'}")
        print(f"Target format:     {self.target_sample_rate} Hz, {self.target_channels} channel(s)")
        print("")

        build_rows: List[Dict[str, Any]] = []

        for i, row in enumerate(rows, start=1):
            source_path = self._resolve_source_path(row)
            target_path = self._target_path(row, source_path)

            status = "planned"
            error = ""
            transform_info: Dict[str, Any] = {}

            if not source_path.exists():
                status = "missing_source"
                error = f"source not found: {source_path}"

            elif target_path.exists() and not self.overwrite:
                status = "exists_skip"
                error = "target exists; use --overwrite to rebuild"

            if not source_path.is_file():
                status = "missing_source"
                error = f"source not found: {source_path}"

            elif self.apply:
                try:
                    transform_info = transform_to_clean_wav(
                        source_path=source_path,
                        target_path=target_path,
                        target_sample_rate=self.target_sample_rate,
                        target_channels=self.target_channels,
                    )
                    status = "built"
                except Exception as exc:
                    status = "failed"
                    error = str(exc)

            if i == 1 or i % 100 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] {status}: {source_path.name}")

            build_row = {
                "build_id": f"cleaned_{i:06d}",
                "status": status,
                "class_name": row.get("class_name", ""),
                "source_file": row.get("source_file", ""),
                "source_path": str(source_path),
                "cleaned_path": str(target_path),
                "cleaned_file": target_path.name,
                "error": error,
            }

            for key, value in transform_info.items():
                build_row[key] = value

            build_rows.append(build_row)

        build_manifest_path = self.out_root.parent / "raw5_agentic_cleaned_build_manifest.csv"
        summary_json_path = self.out_root.parent / "raw5_agentic_cleaned_build_summary.json"
        summary_md_path = self.out_root.parent / "raw5_agentic_cleaned_build_summary.md"

        status_counts = Counter(row["status"] for row in build_rows)
        class_counts = Counter(
            row["class_name"]
            for row in build_rows
            if row["status"] in {"built", "planned", "exists_skip"}
        )

        summary = {
            "agent_name": "DatasetBuilderAgent",
            "agent_version": "v0.1_cleaned_dataset_builder",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "apply" if self.apply else "dry_run",
            "inputs": {
                "accepted_manifest": str(self.accepted_manifest),
                "raw_root": str(self.raw_root) if self.raw_root else "",
            },
            "outputs": {
                "cleaned_dataset_root": str(self.out_root),
                "build_manifest": str(build_manifest_path),
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
            },
            "target_audio": {
                "sample_rate": self.target_sample_rate,
                "channels": self.target_channels,
                "format": "wav_pcm16",
            },
            "status_counts": dict(status_counts),
            "class_counts": dict(class_counts),
            "safety": {
                "raw_files_modified": False,
                "raw_files_deleted": False,
                "cleaned_copies_written": bool(self.apply),
            },
        }

        self._write_csv(build_manifest_path, build_rows)
        self._write_json(summary_json_path, summary)

        md_lines = [
            "# DatasetBuilderAgent Report",
            "",
            f"Mode: `{summary['mode']}`",
            "",
            "## Status Counts",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]

        for status, count in summary["status_counts"].items():
            md_lines.append(f"| {status} | {count} |")

        md_lines.extend(
            [
                "",
                "## Class Counts",
                "",
                "| Class | Count |",
                "|---|---:|",
            ]
        )

        for class_name, count in sorted(summary["class_counts"].items()):
            md_lines.append(f"| {class_name} | {count} |")

        md_lines.extend(
            [
                "",
                "## Safety",
                "",
                "- Raw audio files were not modified.",
                "- Cleaned files were written as separate 16 kHz mono WAV copies.",
            ]
        )

        summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

        print("")
        print("DatasetBuilderAgent summary")
        print("---------------------------")
        for status, count in status_counts.items():
            print(f"{status:14s}: {count}")

        print("")
        print(f"Build manifest: {build_manifest_path}")
        print(f"Summary JSON:   {summary_json_path}")
        print(f"Summary MD:     {summary_md_path}")

        return summary
