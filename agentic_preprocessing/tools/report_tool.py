# agentic_preprocessing\tools\report_tool.py

"""
Report writing utilities for Agentic AI preprocessing.

Outputs:
- CSV decision manifest
- JSON full report
- Markdown summary report
- recommended_next_actions.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


REPORT_FIELDNAMES = [
    "source_file",
    "relative_path",
    "file_name",
    "class_label",
    "decision",
    "reason_codes",
    "safe_to_train",
    "readable",
    "error",
    "duration_sec",
    "sample_rate",
    "channels",
    "sample_width",
    "n_frames",
    "rms_db",
    "peak_db",
    "silence_ratio",
    "clipping_ratio",
    "speech_activity_ratio",
    "sha256",
    "duplicate_group",
    "duplicate_count",
    "background_music_score",
    "background_noise_score",
    "wrong_speaker_suspicion",
    "agent_version",
]


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return "|".join(str(x) for x in value)
    return value


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()

        for row in rows:
            normalized = {k: _format_value(row.get(k, "")) for k in REPORT_FIELDNAMES}
            writer.writerow(normalized)


def write_json(obj: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "agent_mode": "audit_only_non_destructive",
        "total_rows": len(rows),
        "total_audio_files": sum(1 for r in rows if r.get("source_file")),
        "safe_to_train_count": 0,
        "by_decision": {},
        "by_class": {},
        "duration_sec": {},
    }

    durations = []

    for row in rows:
        decision = str(row.get("decision", "unknown"))
        class_label = str(row.get("class_label", "unknown"))

        summary["by_decision"][decision] = summary["by_decision"].get(decision, 0) + 1

        if class_label not in summary["by_class"]:
            summary["by_class"][class_label] = {
                "total": 0,
                "accepted": 0,
                "needs_review": 0,
                "rejected": 0,
                "blocked": 0,
                "safe_to_train": 0,
            }

        summary["by_class"][class_label]["total"] += 1

        if decision in summary["by_class"][class_label]:
            summary["by_class"][class_label][decision] += 1

        if bool(row.get("safe_to_train", False)):
            summary["safe_to_train_count"] += 1
            summary["by_class"][class_label]["safe_to_train"] += 1

        duration = row.get("duration_sec")
        if isinstance(duration, (float, int)) and duration > 0:
            durations.append(float(duration))

    if durations:
        summary["duration_sec"] = {
            "min": round(min(durations), 4),
            "mean": round(mean(durations), 4),
            "max": round(max(durations), 4),
        }

    return summary


def build_recommended_actions(summary: Dict[str, Any]) -> Dict[str, Any]:
    by_decision = summary.get("by_decision", {})
    accepted = int(by_decision.get("accepted", 0))
    needs_review = int(by_decision.get("needs_review", 0))
    rejected = int(by_decision.get("rejected", 0))
    blocked = int(by_decision.get("blocked", 0))

    actions = [
        "Keep the raw dataset unchanged.",
        "Use accepted samples as the first safe subset for raw4_agentic_cleaned.",
    ]

    if needs_review > 0:
        actions.append("Manually inspect needs_review samples before including them in training.")

    if rejected > 0:
        actions.append("Exclude rejected samples from the first agentic-cleaned training stage, but preserve them for traceability.")

    if blocked > 0:
        actions.append("Do not train on blocked samples; inspect corrupted, unreadable, or unsupported files separately.")

    actions.extend(
        [
            "Do not claim full automatic cleaning yet; report this as non-destructive human-in-the-loop agentic preprocessing.",
            "Next version should add stronger speech-activity, background-noise, background-music, near-duplicate, and wrong-speaker checks.",
        ]
    )

    return {
        "agent_mode": "audit_only_non_destructive",
        "safe_to_build_cleaned_stage": accepted > 0,
        "accepted_count": accepted,
        "needs_review_count": needs_review,
        "rejected_count": rejected,
        "blocked_count": blocked,
        "recommended_actions": actions,
    }


def write_markdown_report(
    summary: Dict[str, Any],
    recommended_actions: Dict[str, Any],
    out_path: Path,
) -> None:
    lines: List[str] = []

    lines.append("# Dataset Audit Agent Report")
    lines.append("")
    lines.append("Mode: **audit-only, non-destructive**")
    lines.append("")
    lines.append("The agent did **not** delete, move, or modify raw audio files.")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append(f"- Total rows: `{summary.get('total_rows', 0)}`")
    lines.append(f"- Total audio files: `{summary.get('total_audio_files', 0)}`")
    lines.append(f"- Safe-to-train count: `{summary.get('safe_to_train_count', 0)}`")
    lines.append("")

    duration = summary.get("duration_sec", {})
    if duration:
        lines.append("## Duration Summary")
        lines.append("")
        lines.append(f"- Minimum duration: `{duration.get('min')}` sec")
        lines.append(f"- Mean duration: `{duration.get('mean')}` sec")
        lines.append(f"- Maximum duration: `{duration.get('max')}` sec")
        lines.append("")

    lines.append("## Decision Counts")
    lines.append("")
    lines.append("| Decision | Count |")
    lines.append("|---|---:|")
    for decision, count in summary.get("by_decision", {}).items():
        lines.append(f"| {decision} | {count} |")
    lines.append("")

    lines.append("## Class-wise Summary")
    lines.append("")
    lines.append("| Class | Total | Accepted | Needs Review | Rejected | Blocked | Safe to Train |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for class_label, item in summary.get("by_class", {}).items():
        lines.append(
            f"| {class_label} | "
            f"{item.get('total', 0)} | "
            f"{item.get('accepted', 0)} | "
            f"{item.get('needs_review', 0)} | "
            f"{item.get('rejected', 0)} | "
            f"{item.get('blocked', 0)} | "
            f"{item.get('safe_to_train', 0)} |"
        )

    lines.append("")
    lines.append("## Recommended Next Actions")
    lines.append("")

    for idx, action in enumerate(recommended_actions.get("recommended_actions", []), start=1):
        lines.append(f"{idx}. {action}")

    lines.append("")
    lines.append("## Research Interpretation")
    lines.append("")
    lines.append(
        "This report supports a human-in-the-loop agentic preprocessing workflow. "
        "The agent flags and routes samples using interpretable quality indicators, "
        "while preserving raw data and allowing human verification before training."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_bundle(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    report_prefix: str = "dataset_audit_agent_report",
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_rows(rows)
    recommended_actions = build_recommended_actions(summary)

    csv_path = out_dir / f"{report_prefix}.csv"
    json_path = out_dir / f"{report_prefix}.json"
    md_path = out_dir / f"{report_prefix}.md"
    actions_path = out_dir / "recommended_next_actions.json"

    write_csv(rows, csv_path)
    write_json({"summary": summary, "rows": rows}, json_path)
    write_markdown_report(summary, recommended_actions, md_path)
    write_json(recommended_actions, actions_path)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "recommended_actions": str(actions_path),
    }