# agentic_preprocessing/run_manifest_builder.py


"""
CLI entry point for ManifestBuilderAgent.

Example:

python -m agentic_preprocessing.run_manifest_builder `
  --audit_csv human_talk_workspace\\agent_reports\\dataset_audit_agent_report.csv `
  --triage_seed_root human_talk_triage_seed_dataset `
  --out_dir human_talk_workspace\\agent_reports
"""

from __future__ import annotations

import argparse

from agentic_preprocessing.agents.manifest_builder_agent import ManifestBuilderAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training/review manifests from DatasetAuditorAgent output."
    )

    parser.add_argument(
        "--audit_csv",
        required=True,
        help="Path to dataset_audit_agent_report.csv",
    )

    parser.add_argument(
        "--triage_seed_root",
        required=True,
        help="Path to human_talk_triage_seed_dataset",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory where manifest files should be written.",
    )

    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable CLI summary printing.",
    )

    args = parser.parse_args()

    agent = ManifestBuilderAgent(
        audit_csv=args.audit_csv,
        triage_seed_root=args.triage_seed_root,
        out_dir=args.out_dir,
        show_summary=not args.no_summary,
    )

    agent.run()


if __name__ == "__main__":
    main()