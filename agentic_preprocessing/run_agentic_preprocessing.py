# agentic_preprocessing/run_agentic_preprocessing.py

"""
Entry point for Agentic AI Data Preprocessing.

Version 0.1:
- Runs Dataset Auditor Agent
- Scans raw4 speaker classes
- Produces non-destructive audit reports
- Writes outputs to human_talk_workspace/agent_reports

Recommended command:

python -m agentic_preprocessing.run_agentic_preprocessing `
  --raw_root human_talk_dataset `
  --out_dir human_talk_workspace/agent_reports `
  --classes "Brene_Brown,Eckhart_Tolle,Eric_Thomas,Gary_Vee"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from agentic_preprocessing.agents.dataset_auditor_agent import DatasetAuditorAgent


RAW4_CLASSES = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
]


def default_policy(
    expected_sample_rate: int = 16000,
    expected_duration_sec: float = 5.0,
) -> Dict[str, Any]:
    return {
        "audio_extensions": [".wav"],
        "expected_sample_rate": expected_sample_rate,
        "expected_duration_sec": expected_duration_sec,
        "duration_tolerance_sec": 1.0,
        "min_duration_sec": 1.0,
        "high_silence_ratio": 0.70,
        "very_low_rms_db": -45.0,
        "clipping_ratio_threshold": 0.01,
        "min_speech_activity_ratio": 0.25,
    }


def load_policy_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML policy when PyYAML is available.

    If PyYAML is unavailable, this safely returns an empty dict.
    The runner still works using default policy values.
    """
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    policy: Dict[str, Any] = {}

    audio = data.get("audio", {}) or {}
    thresholds = data.get("quality_thresholds", {}) or {}

    policy.update(audio)
    policy.update(thresholds)

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run non-destructive Agentic AI dataset preprocessing audit."
    )

    parser.add_argument(
        "--raw_root",
        default="human_talk_dataset",
        help="Raw dataset root folder.",
    )

    parser.add_argument(
        "--out_dir",
        default="human_talk_workspace/agent_reports",
        help="Output directory for generated agent reports.",
    )

    parser.add_argument(
        "--classes",
        default=",".join(RAW4_CLASSES),
        help="Comma-separated class names to audit.",
    )

    parser.add_argument(
        "--policy",
        default="agentic_preprocessing/policies/preprocessing_policy.yaml",
        help="Path to preprocessing policy YAML.",
    )

    parser.add_argument(
        "--expected_sample_rate",
        type=int,
        default=16000,
        help="Expected sample rate.",
    )

    parser.add_argument(
        "--expected_duration_sec",
        type=float,
        default=5.0,
        help="Expected parent clip duration in seconds.",
    )

    args = parser.parse_args()

    policy = default_policy(
        expected_sample_rate=args.expected_sample_rate,
        expected_duration_sec=args.expected_duration_sec,
    )

    policy_from_yaml = load_policy_yaml(Path(args.policy))
    policy.update(policy_from_yaml)

    agent = DatasetAuditorAgent(
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        classes=args.classes,
        policy=policy,
    )

    outputs = agent.run()

    print("")
    print("Agentic preprocessing audit complete.")
    print("Mode: audit-only, non-destructive")
    print("")
    print("Generated outputs:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")

    print("")
    print("Next recommended command:")
    print(f"notepad {outputs.get('markdown', '')}")
    print("")


if __name__ == "__main__":
    main()