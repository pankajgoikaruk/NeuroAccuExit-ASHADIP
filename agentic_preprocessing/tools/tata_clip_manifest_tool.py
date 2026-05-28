# agentic_preprocessing/tools/tata_clip_manifest_tool.py

"""
TinyAudioTriageAgent clip-level manifest builder.

This script creates a 5-second clip-level multi-label manifest template.

It does NOT require manual timing.
It does NOT create segment-level labels yet.
It does NOT train TATA.

The folder name gives the default primary label.
You can then manually correct extra labels in the CSV.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


TARGET_SPEAKER_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
]


NON_TARGET_SPEECH_LABELS = [
    "other_speaker_present",
]


EVENT_BACKGROUND_LABELS = [
    "music_present",
    "applause_present",
    "laughter_present",
    "crowd_cheer_present",
    "silence_present",
]


ALL_TATA_LABELS = (
    TARGET_SPEAKER_LABELS
    + NON_TARGET_SPEECH_LABELS
    + EVENT_BACKGROUND_LABELS
)


EVENT_FOLDER_TO_LABEL = {
    "music": "music_present",
    "applause": "applause_present",
    "laughter": "laughter_present",
    "crowd_cheer": "crowd_cheer_present",
    "silence": "silence_present",
}


STANDARD_FIELDS = [
    "clip_id",
    "file_path",
    "rel_path",
    "file_name",
    "source_group",
    "source_subfolder",
    "primary_label",
    "labels",
    "num_active_labels",
    "split",
    "needs_manual_check",
    "review_status",
    "notes",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def collect_audio_files(root: Path) -> List[Path]:
    return sorted(
        [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda p: str(p).lower(),
    )


def infer_default_labels(audio_path: Path, root: Path) -> Dict[str, Any] | None:
    rel = audio_path.relative_to(root)
    parts = rel.parts

    if len(parts) < 3:
        return None

    source_group = parts[0]
    source_subfolder = parts[1]

    active_labels: List[str] = []
    primary_label = ""

    if source_group == "target_speaker":
        if source_subfolder not in TARGET_SPEAKER_LABELS:
            return None
        primary_label = source_subfolder
        active_labels = [source_subfolder]

    elif source_group == "other_speaker":
        primary_label = "other_speaker_present"
        active_labels = ["other_speaker_present"]

    elif source_group == "events":
        event_label = EVENT_FOLDER_TO_LABEL.get(source_subfolder)
        if event_label is None:
            return None
        primary_label = event_label
        active_labels = [event_label]

    else:
        return None

    active_labels = sorted(set(active_labels))

    row: Dict[str, Any] = {
        "clip_id": audio_path.stem,
        "file_path": str(audio_path),
        "rel_path": str(rel),
        "file_name": audio_path.name,
        "source_group": source_group,
        "source_subfolder": source_subfolder,
        "primary_label": primary_label,
        "labels": "|".join(active_labels),
        "num_active_labels": len(active_labels),
        "split": "",
        "needs_manual_check": 1,
        "review_status": "pending",
        "notes": "",
    }

    for label in ALL_TATA_LABELS:
        row[label] = 1 if label in active_labels else 0

    return row


def build_clip_manifest_rows(seed_root: Path) -> List[Dict[str, Any]]:
    if not seed_root.exists():
        raise FileNotFoundError(f"Seed root not found: {seed_root}")

    rows: List[Dict[str, Any]] = []

    for audio_path in collect_audio_files(seed_root):
        row = infer_default_labels(audio_path, seed_root)
        if row is not None:
            rows.append(row)

    return rows


def validate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts = Counter()
    primary_counts = Counter()
    source_group_counts = Counter()
    source_subfolder_counts = Counter()
    problems: List[str] = []

    for row in rows:
        primary_counts[row["primary_label"]] += 1
        source_group_counts[row["source_group"]] += 1
        source_subfolder_counts[row["source_subfolder"]] += 1

        active = []
        for label in ALL_TATA_LABELS:
            value = int(row.get(label, 0))
            if value == 1:
                active.append(label)
                label_counts[label] += 1

        if len(active) == 0:
            problems.append(f"{row['rel_path']}: no active labels")

        if int(row["num_active_labels"]) != len(active):
            problems.append(f"{row['rel_path']}: num_active_labels mismatch")

        label_text = "|".join(sorted(active))
        if row["labels"] != label_text:
            problems.append(f"{row['rel_path']}: labels column mismatch")

    return {
        "total_rows": len(rows),
        "num_labels": len(ALL_TATA_LABELS),
        "labels": ALL_TATA_LABELS,
        "label_counts": dict(label_counts),
        "primary_label_counts": dict(primary_counts),
        "source_group_counts": dict(source_group_counts),
        "source_subfolder_counts": dict(source_subfolder_counts),
        "problems": problems,
        "is_valid": len(problems) == 0,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = STANDARD_FIELDS + ALL_TATA_LABELS

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    validation = summary["validation"]

    lines: List[str] = []
    lines.append("# TATA Clip-Level Manifest Report")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This manifest is the 5-second clip-level multi-label annotation template "
        "for TinyAudioTriageAgent. It is created from folder structure and then "
        "manually corrected where clips contain extra labels."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows: `{validation['total_rows']}`")
    lines.append(f"- Number of labels: `{validation['num_labels']}`")
    lines.append(f"- Valid: `{validation['is_valid']}`")
    lines.append("")
    lines.append("## Label Counts")
    lines.append("")
    lines.append("| Label | Count |")
    lines.append("|---|---:|")

    for label in ALL_TATA_LABELS:
        lines.append(f"| `{label}` | {validation['label_counts'].get(label, 0)} |")

    lines.append("")
    lines.append("## Source Group Counts")
    lines.append("")
    lines.append("| Source group | Count |")
    lines.append("|---|---:|")

    for group, count in sorted(validation["source_group_counts"].items()):
        lines.append(f"| `{group}` | {count} |")

    lines.append("")
    lines.append("## Manual Review Instruction")
    lines.append("")
    lines.append("- Keep `primary_label` as the dominant folder/source label.")
    lines.append("- Add extra multi-hot labels where the 5-second clip contains mixed content.")
    lines.append("- Do not annotate exact timing.")
    lines.append("- Do not label 1-second segments manually.")
    lines.append("")
    lines.append("Example:")
    lines.append("")
    lines.append("| primary_label | applause_present | laughter_present | crowd_cheer_present |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| applause_present | 1 | 0 | 0 |")
    lines.append("| applause_present | 1 | 1 | 0 |")
    lines.append("| crowd_cheer_present | 1 | 0 | 1 |")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")

    for name, value in summary["output_files"].items():
        lines.append(f"- `{name}`: `{value}`")

    if validation["problems"]:
        lines.append("")
        lines.append("## Validation Problems")
        lines.append("")
        for problem in validation["problems"]:
            lines.append(f"- {problem}")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_tata_clip_manifest(seed_root: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_clip_manifest_rows(seed_root)
    validation = validate_rows(rows)

    manifest_path = out_dir / "tata_clip_level_manifest.csv"
    labels_path = out_dir / "tata_labels.json"
    summary_json_path = out_dir / "tata_clip_level_manifest_summary.json"
    summary_md_path = out_dir / "tata_clip_level_manifest_summary.md"

    write_csv(manifest_path, rows)

    labels_payload = {
        "task": "tiny_audio_triage",
        "labeling_level": "clip_level",
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "labels": ALL_TATA_LABELS,
        "target_speaker_labels": TARGET_SPEAKER_LABELS,
        "non_target_speech_labels": NON_TARGET_SPEECH_LABELS,
        "event_background_labels": EVENT_BACKGROUND_LABELS,
    }

    write_json(labels_path, labels_payload)

    summary = {
        "agent_name": "TinyAudioTriageAgent",
        "branch": "agentic_data_preprocessing_v0.5_tata_2",
        "stage": "clip_level_manifest_template",
        "generated_at": now_iso(),
        "inputs": {
            "seed_root": str(seed_root),
            "out_dir": str(out_dir),
        },
        "validation": validation,
        "output_files": {
            "manifest_csv": str(manifest_path),
            "labels_json": str(labels_path),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
        },
    }

    write_json(summary_json_path, summary)
    write_md(summary_md_path, summary)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    validation = summary["validation"]

    print("")
    print("=" * 90)
    print("TATA Clip-Level Manifest Summary")
    print("=" * 90)
    print(f"Total rows: {validation['total_rows']}")
    print(f"Labels:     {validation['num_labels']}")
    print(f"Valid:      {validation['is_valid']}")
    print("")
    print("Label counts:")

    for label in ALL_TATA_LABELS:
        print(f"  {label:24s}: {validation['label_counts'].get(label, 0)}")

    print("")
    print("Output files:")

    for name, value in summary["output_files"].items():
        print(f"  {name:16s}: {value}")

    print("=" * 90)
