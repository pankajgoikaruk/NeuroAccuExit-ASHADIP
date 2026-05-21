# agentic_preprocessing\tools\report_tool.py


"""
Report writing utilities for Agentic AI preprocessing.

Version 0.3 outputs:
- dataset_audit_agent_report.csv
- dataset_audit_agent_report.json
- dataset_audit_agent_report.md
- recommended_next_actions.json
- accepted_manifest.csv
- needs_review_manifest.csv
- rejected_manifest.csv
- blocked_manifest.csv
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


REPORT_FIELDNAMES = [
    "source_file",
    "relative_path",
    "file_name",
    "class_label",
    "decision",
    "safe_after_preprocessing",
    "requires_preprocessing",
    "preprocessing_actions",
    "raw_reason_codes",
    "preprocessing_reason_codes",
    "quality_reason_codes",
    "warning_codes",
    "decision_reason_codes",
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


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        if not value:
            return []
        return [x for x in value.split("|") if x]
    return [str(value)]


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
        "safe_after_preprocessing_count": 0,
        "requires_preprocessing_count": 0,
        "warning_count": 0,
        "by_decision": {},
        "by_class": {},
        "duration_sec": {},
        "reason_code_counts": {},
        "quality_reason_code_counts": {},
        "preprocessing_reason_code_counts": {},
        "warning_code_counts": {},
        "needs_review_reason_code_counts": {},
        "rejected_reason_code_counts": {},
    }

    durations = []

    reason_counter: Counter[str] = Counter()
    quality_counter: Counter[str] = Counter()
    preprocessing_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()
    needs_review_counter: Counter[str] = Counter()
    rejected_counter: Counter[str] = Counter()

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
                "safe_after_preprocessing": 0,
                "requires_preprocessing": 0,
                "warning_count": 0,
                "quality_score_percent": 0.0,
            }

        class_item = summary["by_class"][class_label]
        class_item["total"] += 1

        if decision in class_item:
            class_item[decision] += 1

        if bool(row.get("safe_after_preprocessing", False)):
            summary["safe_after_preprocessing_count"] += 1
            class_item["safe_after_preprocessing"] += 1

        if bool(row.get("requires_preprocessing", False)):
            summary["requires_preprocessing_count"] += 1
            class_item["requires_preprocessing"] += 1

        warning_codes = _as_list(row.get("warning_codes"))
        if warning_codes:
            summary["warning_count"] += 1
            class_item["warning_count"] += 1

        raw_codes = _as_list(row.get("raw_reason_codes"))
        quality_codes = _as_list(row.get("quality_reason_codes"))
        preprocessing_codes = _as_list(row.get("preprocessing_reason_codes"))
        decision_codes = _as_list(row.get("decision_reason_codes"))

        reason_counter.update(raw_codes)
        quality_counter.update(quality_codes)
        preprocessing_counter.update(preprocessing_codes)
        warning_counter.update(warning_codes)

        if decision == "needs_review":
            needs_review_counter.update(decision_codes or quality_codes or raw_codes)

        if decision == "rejected":
            rejected_counter.update(decision_codes or quality_codes or raw_codes)

        duration = row.get("duration_sec")
        if isinstance(duration, (float, int)) and duration > 0:
            durations.append(float(duration))

    for class_label, item in summary["by_class"].items():
        total = max(int(item.get("total", 0)), 1)
        accepted = int(item.get("accepted", 0))
        item["quality_score_percent"] = round((accepted / total) * 100.0, 2)

    if durations:
        summary["duration_sec"] = {
            "min": round(min(durations), 4),
            "mean": round(mean(durations), 4),
            "max": round(max(durations), 4),
        }

    summary["reason_code_counts"] = dict(reason_counter.most_common())
    summary["quality_reason_code_counts"] = dict(quality_counter.most_common())
    summary["preprocessing_reason_code_counts"] = dict(preprocessing_counter.most_common())
    summary["warning_code_counts"] = dict(warning_counter.most_common())
    summary["needs_review_reason_code_counts"] = dict(needs_review_counter.most_common())
    summary["rejected_reason_code_counts"] = dict(rejected_counter.most_common())

    return summary


def build_recommended_actions(summary: Dict[str, Any]) -> Dict[str, Any]:
    by_decision = summary.get("by_decision", {})
    accepted = int(by_decision.get("accepted", 0))
    needs_review = int(by_decision.get("needs_review", 0))
    rejected = int(by_decision.get("rejected", 0))
    blocked = int(by_decision.get("blocked", 0))
    requires_preprocessing = int(summary.get("requires_preprocessing_count", 0))

    actions = [
        "Keep the raw dataset unchanged.",
        "Use accepted samples as the first safe subset for the current agentic-cleaned dataset stage.",
    ]

    if requires_preprocessing > 0:
        actions.append(
            "Apply standard preprocessing to accepted samples before training: resample to 16 kHz and downmix to mono where required."
        )

    if needs_review > 0:
        actions.append(
            "Manually inspect needs_review_manifest.csv before deciding whether those samples can be included later."
        )

    if rejected > 0:
        actions.append(
            "Exclude rejected_manifest.csv samples from the first agentic-cleaned training stage, but preserve them for traceability."
        )

    if blocked > 0:
        actions.append(
            "Do not train on blocked_manifest.csv samples; inspect corrupted, missing, or unsupported files separately."
        )

    actions.extend(
        [
            "Do not claim full automatic cleaning yet; report this as non-destructive human-in-the-loop agentic preprocessing.",
            "Do not create routed accepted/rejected folders yet; keep V0.3 manifest-first.",
            "Next version should add stronger background-noise, background-music, near-duplicate, and wrong-speaker checks.",
        ]
    )

    return {
        "agent_mode": "audit_only_non_destructive",
        "safe_to_build_cleaned_stage": accepted > 0,
        "accepted_count": accepted,
        "needs_review_count": needs_review,
        "rejected_count": rejected,
        "blocked_count": blocked,
        "requires_preprocessing_count": requires_preprocessing,
        "recommended_actions": actions,
    }


def _write_reason_table(lines: List[str], title: str, counts: Dict[str, int]) -> None:
    lines.append(f"## {title}")
    lines.append("")

    if not counts:
        lines.append("No reason codes recorded.")
        lines.append("")
        return

    lines.append("| Reason code | Count |")
    lines.append("|---|---:|")

    for reason, count in counts.items():
        lines.append(f"| {reason} | {count} |")

    lines.append("")


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
    lines.append("The agent did **not** delete, move, copy, or modify raw audio files.")
    lines.append("This V0.3 report uses a **manifest-first** design.")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("")
    lines.append(f"- Total rows: `{summary.get('total_rows', 0)}`")
    lines.append(f"- Total audio files: `{summary.get('total_audio_files', 0)}`")
    lines.append(f"- Safe-after-preprocessing count: `{summary.get('safe_after_preprocessing_count', 0)}`")
    lines.append(f"- Requires preprocessing count: `{summary.get('requires_preprocessing_count', 0)}`")
    lines.append(f"- Files with borderline warnings: `{summary.get('warning_count', 0)}`")
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
    lines.append(
        "| Class | Total | Accepted | Needs Review | Rejected | Blocked | "
        "Safe After Preprocessing | Requires Preprocessing | Warnings | Quality Score |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for class_label, item in summary.get("by_class", {}).items():
        lines.append(
            f"| {class_label} | "
            f"{item.get('total', 0)} | "
            f"{item.get('accepted', 0)} | "
            f"{item.get('needs_review', 0)} | "
            f"{item.get('rejected', 0)} | "
            f"{item.get('blocked', 0)} | "
            f"{item.get('safe_after_preprocessing', 0)} | "
            f"{item.get('requires_preprocessing', 0)} | "
            f"{item.get('warning_count', 0)} | "
            f"{item.get('quality_score_percent', 0.0)}% |"
        )

    lines.append("")

    _write_reason_table(
        lines,
        "Preprocessing Reason Codes",
        summary.get("preprocessing_reason_code_counts", {}),
    )

    _write_reason_table(
        lines,
        "Quality Reason Codes",
        summary.get("quality_reason_code_counts", {}),
    )

    _write_reason_table(
        lines,
        "Borderline Warning Codes",
        summary.get("warning_code_counts", {}),
    )

    _write_reason_table(
        lines,
        "Needs-review Explanation",
        summary.get("needs_review_reason_code_counts", {}),
    )

    _write_reason_table(
        lines,
        "Rejected-sample Explanation",
        summary.get("rejected_reason_code_counts", {}),
    )

    lines.append("## Generated Manifest Files")
    lines.append("")
    lines.append("| Manifest | Purpose |")
    lines.append("|---|---|")
    lines.append("| `accepted_manifest.csv` | Use for first raw4_agentic_cleaned stage |")
    lines.append("| `needs_review_manifest.csv` | Human inspection candidates |")
    lines.append("| `rejected_manifest.csv` | Excluded from first cleaned training stage |")
    lines.append("| `blocked_manifest.csv` | Unreadable/missing/unsupported items, if any |")
    lines.append("")

    lines.append("## Recommended Next Actions")
    lines.append("")

    for idx, action in enumerate(recommended_actions.get("recommended_actions", []), start=1):
        lines.append(f"{idx}. {action}")

    lines.append("")
    lines.append("## Research Interpretation")
    lines.append("")
    lines.append(
        "This report supports a non-destructive, human-in-the-loop agentic preprocessing workflow. "
        "The agent separates standard preprocessing requirements from true quality concerns, "
        "flags borderline samples with warning codes, and creates traceable manifests for training, "
        "review, and exclusion decisions."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_split_manifests(rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, str]:
    manifest_paths: Dict[str, str] = {}

    mapping = {
        "accepted": "accepted_manifest.csv",
        "needs_review": "needs_review_manifest.csv",
        "rejected": "rejected_manifest.csv",
        "blocked": "blocked_manifest.csv",
    }

    for decision, filename in mapping.items():
        selected_rows = [row for row in rows if row.get("decision") == decision]
        out_path = out_dir / filename
        write_csv(selected_rows, out_path)
        manifest_paths[decision] = str(out_path)

    return manifest_paths


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

    manifest_paths = write_split_manifests(rows, out_dir)

    outputs = {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "recommended_actions": str(actions_path),
        "accepted_manifest": manifest_paths["accepted"],
        "needs_review_manifest": manifest_paths["needs_review"],
        "rejected_manifest": manifest_paths["rejected"],
        "blocked_manifest": manifest_paths["blocked"],
    }

    return outputs