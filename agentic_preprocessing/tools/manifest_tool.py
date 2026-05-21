# agentic_preprocessing\tools\manifest_tool.py

"""
Manifest utilities for agentic preprocessing.

This module converts DatasetAuditorAgent outputs into clean, traceable manifests.

Main outputs:
- accepted_manifest.csv
- needs_review_manifest.csv
- rejected_manifest.csv
- blocked_manifest.csv
- triage_seed_manifest.csv
- manifest_summary.json
- manifest_summary.md

Safety:
- Does not move, delete, or overwrite raw audio files.
- Only reads CSV/audio folders and writes manifest/report files.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


AUDIT_STANDARD_FIELDS = [
    "manifest_id",
    "dataset_source",
    "decision",
    "class_name",
    "source_file",
    "source_path",
    "rel_path",
    "safe_to_train",
    "requires_preprocessing",
    "preprocessing_actions",
    "reason_codes",
    "warning_codes",
]


TRIAGE_STANDARD_FIELDS = [
    "manifest_id",
    "dataset_source",
    "label_group",
    "label_type",
    "triage_label",
    "speaker_identity",
    "event_label",
    "class_or_event",
    "source_file",
    "source_path",
    "rel_path",
    "parent_clip_id",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], preferred_fields: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if preferred_fields is None:
        preferred_fields = []

    fieldnames: List[str] = []

    for field in preferred_fields:
        if field not in fieldnames:
            fieldnames.append(field)

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


def first_non_empty(row: Dict[str, Any], candidates: Iterable[str], default: str = "") -> str:
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def normalize_bool_text(value: Any) -> str:
    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return "true"

    if text in {"false", "0", "no", "n"}:
        return "false"

    return str(value).strip() if value is not None else ""


def normalize_decision(row: Dict[str, Any]) -> str:
    decision = first_non_empty(
        row,
        [
            "decision",
            "status",
            "agent_decision",
            "routing_decision",
            "final_decision",
        ],
        default="",
    ).lower()

    decision = decision.replace(" ", "_").replace("-", "_")

    if decision in {"accepted", "accept", "safe", "safe_to_train"}:
        return "accepted"

    if decision in {"needs_review", "review", "manual_review", "needs_manual_review"}:
        return "needs_review"

    if decision in {"rejected", "reject"}:
        return "rejected"

    if decision in {"blocked", "block"}:
        return "blocked"

    safe_to_train = normalize_bool_text(first_non_empty(row, ["safe_to_train"], default=""))

    if safe_to_train == "true":
        return "accepted"

    return "needs_review"


def infer_source_path(row: Dict[str, Any]) -> str:
    return first_non_empty(
        row,
        [
            "source_path",
            "file_path",
            "path",
            "audio_path",
            "filepath",
            "filename",
        ],
        default="",
    )


def infer_class_name(row: Dict[str, Any]) -> str:
    return first_non_empty(
        row,
        [
            "class_name",
            "label",
            "speaker",
            "class",
            "folder",
            "class_dir",
        ],
        default="unknown",
    )


def safe_rel_path(source_path: str, root_hint: str = "") -> str:
    if not source_path:
        return ""

    path = Path(source_path)

    if root_hint:
        try:
            return str(path.resolve().relative_to(Path(root_hint).resolve()))
        except Exception:
            pass

    return str(path)


def normalize_audit_row(
    row: Dict[str, Any],
    index: int,
    dataset_source: str = "dataset_audit",
) -> Dict[str, Any]:
    decision = normalize_decision(row)
    source_path = infer_source_path(row)
    source_file = Path(source_path).name if source_path else first_non_empty(row, ["source_file", "old_name", "file"], "")

    normalized: Dict[str, Any] = {
        "manifest_id": f"audit_{index:06d}",
        "dataset_source": dataset_source,
        "decision": decision,
        "class_name": infer_class_name(row),
        "source_file": source_file,
        "source_path": source_path,
        "rel_path": first_non_empty(row, ["rel_path", "relative_path"], default=safe_rel_path(source_path)),
        "safe_to_train": normalize_bool_text(first_non_empty(row, ["safe_to_train"], default="")),
        "requires_preprocessing": normalize_bool_text(first_non_empty(row, ["requires_preprocessing"], default="")),
        "preprocessing_actions": first_non_empty(row, ["preprocessing_actions"], default=""),
        "reason_codes": first_non_empty(row, ["reason_codes", "reasons", "issue_codes"], default=""),
        "warning_codes": first_non_empty(row, ["warning_codes", "warnings"], default=""),
    }

    # Preserve original audit columns for traceability.
    for key, value in row.items():
        audit_key = f"audit__{key}"
        if audit_key not in normalized:
            normalized[audit_key] = value

    return normalized


def split_audit_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    split = {
        "accepted": [],
        "needs_review": [],
        "rejected": [],
        "blocked": [],
    }

    for idx, row in enumerate(rows, start=1):
        normalized = normalize_audit_row(row, index=idx)
        decision = normalized["decision"]

        if decision not in split:
            decision = "needs_review"
            normalized["decision"] = decision

        split[decision].append(normalized)

    return split


def collect_audio_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []

    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def collect_triage_seed_manifest(root: Path) -> List[Dict[str, Any]]:
    """
    Expected structure:

    human_talk_triage_seed_dataset/
    ├─ target_speaker/
    │  ├─ Brene_Brown/
    │  └─ ...
    ├─ other_speaker/
    │  ├─ Brene_Brown_interviewer/
    │  └─ ...
    └─ events/
       ├─ silence/
       ├─ music/
       ├─ applause/
       └─ laughter/
    """
    rows: List[Dict[str, Any]] = []
    idx = 1

    layout = [
        ("target_speaker", "speaker", "target_speaker"),
        ("other_speaker", "speaker", "other_speaker"),
        ("events", "event", ""),
    ]

    for label_group, label_type, fixed_triage_label in layout:
        base = root / label_group

        if not base.exists():
            continue

        group_dirs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

        for group_dir in group_dirs:
            class_or_event = group_dir.name

            if label_group == "events":
                triage_label = class_or_event
                speaker_identity = ""
                event_label = class_or_event
            else:
                triage_label = fixed_triage_label
                speaker_identity = class_or_event
                event_label = ""

            for audio_path in collect_audio_files(group_dir):
                try:
                    rel_path = str(audio_path.resolve().relative_to(root.resolve()))
                except Exception:
                    rel_path = str(audio_path)

                rows.append(
                    {
                        "manifest_id": f"triage_{idx:06d}",
                        "dataset_source": "triage_seed",
                        "label_group": label_group,
                        "label_type": label_type,
                        "triage_label": triage_label,
                        "speaker_identity": speaker_identity,
                        "event_label": event_label,
                        "class_or_event": class_or_event,
                        "source_file": audio_path.name,
                        "source_path": str(audio_path),
                        "rel_path": rel_path,
                        "parent_clip_id": audio_path.stem,
                    }
                )
                idx += 1

    return rows


def summarize_split_manifests(split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    decision_counts = {name: len(rows) for name, rows in split.items()}

    class_counts: Dict[str, Dict[str, int]] = {}

    for decision, rows in split.items():
        counter = Counter(row.get("class_name", "unknown") for row in rows)
        for class_name, count in counter.items():
            class_counts.setdefault(class_name, {})
            class_counts[class_name][decision] = count

    reason_counts = Counter()

    for rows in split.values():
        for row in rows:
            reason_text = str(row.get("reason_codes", "")).strip()
            if not reason_text:
                continue

            parts = [
                p.strip()
                for p in reason_text.replace(",", "|").split("|")
                if p.strip()
            ]

            for part in parts:
                reason_counts[part] += 1

    preprocessing_counts = Counter()

    for rows in split.values():
        for row in rows:
            action_text = str(row.get("preprocessing_actions", "")).strip()
            if not action_text:
                continue

            parts = [
                p.strip()
                for p in action_text.replace(",", "|").split("|")
                if p.strip()
            ]

            for part in parts:
                preprocessing_counts[part] += 1

    return {
        "decision_counts": decision_counts,
        "class_counts": class_counts,
        "reason_counts": dict(reason_counts.most_common()),
        "preprocessing_action_counts": dict(preprocessing_counts.most_common()),
    }


def summarize_triage_seed(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_label_group = Counter(row.get("label_group", "unknown") for row in rows)
    by_triage_label = Counter(row.get("triage_label", "unknown") for row in rows)
    by_class_or_event = Counter(row.get("class_or_event", "unknown") for row in rows)

    return {
        "total": len(rows),
        "by_label_group": dict(by_label_group),
        "by_triage_label": dict(by_triage_label),
        "by_class_or_event": dict(by_class_or_event),
    }


def write_manifest_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    audit = summary.get("audit_manifests", {})
    triage = summary.get("triage_seed_manifest", {})

    lines: List[str] = []

    lines.append("# ManifestBuilderAgent Report")
    lines.append("")
    lines.append(f"Generated: `{summary.get('generated_at', '')}`")
    lines.append("")
    lines.append("## Audit Manifest Split")
    lines.append("")
    lines.append("| Decision | Count |")
    lines.append("|---|---:|")

    for decision, count in audit.get("decision_counts", {}).items():
        lines.append(f"| {decision} | {count} |")

    lines.append("")
    lines.append("## Class Counts")
    lines.append("")
    lines.append("| Class | Accepted | Needs Review | Rejected | Blocked |")
    lines.append("|---|---:|---:|---:|---:|")

    for class_name, counts in sorted(audit.get("class_counts", {}).items()):
        lines.append(
            f"| {class_name} | "
            f"{counts.get('accepted', 0)} | "
            f"{counts.get('needs_review', 0)} | "
            f"{counts.get('rejected', 0)} | "
            f"{counts.get('blocked', 0)} |"
        )

    lines.append("")
    lines.append("## Triage Seed Manifest")
    lines.append("")
    lines.append(f"Total triage seed files: `{triage.get('total', 0)}`")
    lines.append("")
    lines.append("| Triage Label | Count |")
    lines.append("|---|---:|")

    for label, count in sorted(triage.get("by_triage_label", {}).items()):
        lines.append(f"| {label} | {count} |")

    lines.append("")
    lines.append("## Output Files")
    lines.append("")

    for name, output_path in summary.get("output_files", {}).items():
        lines.append(f"- `{name}`: `{output_path}`")

    lines.append("")
    lines.append("## Safety Notes")
    lines.append("")
    lines.append("- No raw audio files were moved, deleted, or overwritten.")
    lines.append("- These manifests are routing files for later dataset-building steps.")
    lines.append("- `needs_review` files require manual approval before training.")
    lines.append("- `rejected` and `blocked` files remain traceable.")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_manifest_summary(summary: Dict[str, Any]) -> None:
    audit = summary.get("audit_manifests", {})
    triage = summary.get("triage_seed_manifest", {})

    print("")
    print("=" * 90)
    print("ManifestBuilderAgent Summary")
    print("=" * 90)

    print("")
    print("Audit split:")
    for decision, count in audit.get("decision_counts", {}).items():
        print(f"  {decision:12s}: {count}")

    print("")
    print("Triage seed:")
    print(f"  total       : {triage.get('total', 0)}")

    for label, count in sorted(triage.get("by_triage_label", {}).items()):
        print(f"  {label:12s}: {count}")

    print("")
    print("Output files:")
    for name, output_path in summary.get("output_files", {}).items():
        print(f"  {name:28s}: {output_path}")

    print("=" * 90)