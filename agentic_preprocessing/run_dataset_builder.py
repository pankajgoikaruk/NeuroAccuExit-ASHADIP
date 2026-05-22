# agentic_preprocessing/run_dataset_builder.py


"""
CLI for DatasetBuilderAgent.

Dry run:

python -m agentic_preprocessing.run_dataset_builder `
  --accepted_manifest human_talk_workspace\\agent_reports\\accepted_manifest.csv `
  --raw_root human_talk_dataset `
  --out_root human_talk_workspace\\datasets\\raw5_agentic_cleaned

Apply:

python -m agentic_preprocessing.run_dataset_builder `
  --accepted_manifest human_talk_workspace\\agent_reports\\accepted_manifest.csv `
  --raw_root human_talk_dataset `
  --out_root human_talk_workspace\\datasets\\raw5_agentic_cleaned `
  --apply
"""

from __future__ import annotations

import argparse

from agentic_preprocessing.agents.dataset_builder_agent import DatasetBuilderAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned dataset from accepted_manifest.csv."
    )

    parser.add_argument("--accepted_manifest", required=True)
    parser.add_argument("--raw_root", default="")
    parser.add_argument("--out_root", required=True)

    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--target_channels", type=int, default=1)

    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    agent = DatasetBuilderAgent(
        accepted_manifest=args.accepted_manifest,
        raw_root=args.raw_root if args.raw_root else None,
        out_root=args.out_root,
        target_sample_rate=args.target_sample_rate,
        target_channels=args.target_channels,
        apply=args.apply,
        overwrite=args.overwrite,
    )

    agent.run()


if __name__ == "__main__":
    main()