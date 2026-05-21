# agentic_preprocessing/agents/manifest_builder_agent.py


"""
ManifestBuilderAgent

Purpose:
- Read DatasetAuditorAgent CSV output.
- Split files into accepted / needs_review / rejected / blocked manifests.
- Build TinyAudioTriageAgent seed manifest.
- Write JSON and Markdown summaries.

This agent is non-destructive:
- It does not move files.
- It does not delete files.
- It does not modify raw audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agentic_preprocessing.tools.manifest_tool import (
    AUDIT_STANDARD_FIELDS,
    TRIAGE_STANDARD_FIELDS,
    collect_triage_seed_manifest,
    print_manifest_summary,
    read_csv_rows,
    split_audit_rows,
    summarize_split_manifests,
    summarize_triage_seed,
    now_iso,
    write_csv_rows,
    write_json,
    write_manifest_summary_md,
)


class ManifestBuilderAgent:
    def __init__(
        self,
        audit_csv: str | Path,
        triage_seed_root: str | Path,
        out_dir: str | Path,
        show_summary: bool = True,
    ) -> None:
        self.audit_csv = Path(audit_csv)
        self.triage_seed_root = Path(triage_seed_root)
        self.out_dir = Path(out_dir)
        self.show_summary = show_summary

    def run(self) -> Dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        print("")
        print("Starting ManifestBuilderAgent")
        print("-----------------------------")
        print(f"Audit CSV:        {self.audit_csv}")
        print(f"Triage seed root: {self.triage_seed_root}")
        print(f"Output dir:       {self.out_dir}")
        print("")

        audit_rows = read_csv_rows(self.audit_csv)
        split = split_audit_rows(audit_rows)

        accepted_path = self.out_dir / "accepted_manifest.csv"
        needs_review_path = self.out_dir / "needs_review_manifest.csv"
        rejected_path = self.out_dir / "rejected_manifest.csv"
        blocked_path = self.out_dir / "blocked_manifest.csv"

        write_csv_rows(accepted_path, split["accepted"], preferred_fields=AUDIT_STANDARD_FIELDS)
        write_csv_rows(needs_review_path, split["needs_review"], preferred_fields=AUDIT_STANDARD_FIELDS)
        write_csv_rows(rejected_path, split["rejected"], preferred_fields=AUDIT_STANDARD_FIELDS)
        write_csv_rows(blocked_path, split["blocked"], preferred_fields=AUDIT_STANDARD_FIELDS)

        triage_rows = collect_triage_seed_manifest(self.triage_seed_root)
        triage_seed_path = self.out_dir / "triage_seed_manifest.csv"

        write_csv_rows(triage_seed_path, triage_rows, preferred_fields=TRIAGE_STANDARD_FIELDS)

        summary = {
            "agent_name": "ManifestBuilderAgent",
            "agent_version": "v0.1_manifest_builder",
            "generated_at": now_iso(),
            "inputs": {
                "audit_csv": str(self.audit_csv),
                "triage_seed_root": str(self.triage_seed_root),
                "out_dir": str(self.out_dir),
            },
            "audit_manifests": summarize_split_manifests(split),
            "triage_seed_manifest": summarize_triage_seed(triage_rows),
            "output_files": {
                "accepted_manifest": str(accepted_path),
                "needs_review_manifest": str(needs_review_path),
                "rejected_manifest": str(rejected_path),
                "blocked_manifest": str(blocked_path),
                "triage_seed_manifest": str(triage_seed_path),
                "manifest_summary_json": str(self.out_dir / "manifest_summary.json"),
                "manifest_summary_md": str(self.out_dir / "manifest_summary.md"),
            },
            "safety": {
                "raw_files_modified": False,
                "raw_files_deleted": False,
                "requires_manual_review_for_needs_review": True,
                "notes": [
                    "This agent only writes manifest and report files.",
                    "Accepted files are eligible for the first cleaned-stage experiment.",
                    "Needs-review files require manual approval before training.",
                    "Rejected and blocked files remain traceable.",
                ],
            },
        }

        summary_json_path = self.out_dir / "manifest_summary.json"
        summary_md_path = self.out_dir / "manifest_summary.md"

        write_json(summary_json_path, summary)
        write_manifest_summary_md(summary_md_path, summary)

        if self.show_summary:
            print_manifest_summary(summary)

        return summary