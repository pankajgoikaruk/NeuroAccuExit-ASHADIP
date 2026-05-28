# agentic_preprocessing/run_tata_clip_manifest_builder.py

"""
CLI runner for TATA 5-second clip-level manifest builder.

Example:

python -m agentic_preprocessing.run_tata_clip_manifest_builder `
  --seed_root human_talk_tata_seed_dataset `
  --out_dir human_talk_workspace\tata_2\metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_preprocessing.tools.tata_clip_manifest_tool import (
    build_tata_clip_manifest,
    print_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TinyAudioTriageAgent clip-level multi-label manifest."
    )

    parser.add_argument(
        "--seed_root",
        required=True,
        help="Path to human_talk_tata_seed_dataset.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for TATA metadata files.",
    )

    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable CLI summary printing.",
    )

    args = parser.parse_args()

    summary = build_tata_clip_manifest(
        seed_root=Path(args.seed_root),
        out_dir=Path(args.out_dir),
    )

    if not args.no_summary:
        print_summary(summary)


if __name__ == "__main__":
    main()
