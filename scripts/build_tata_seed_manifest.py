# scripts/build_tata_seed_manifest.py
#
# Build a true multi-label seed manifest for TinyAudioTriageAgent (TATA).
#
# Purpose:
#   - Convert human_talk_triage_seed_dataset into a multi-hot CSV manifest.
#   - Predict speaker identity labels, interviewer identity labels, and event/background labels.
#   - Support clean and mixed audio conditions without requiring one folder per combination.
#   - Produce labels.json and a summary file compatible with scripts/extract_multilabel_features.py
#     and training/train_multilabel.py.
#
# Example:
#   python -m scripts.build_tata_seed_manifest `
#     --seed_root human_talk_triage_seed_dataset `
#     --out_dir human_talk_workspace\tata_seed `
#     --seed 42
#
# Optional manual annotations:
#   python -m scripts.build_tata_seed_manifest `
#     --seed_root human_talk_triage_seed_dataset `
#     --annotations_csv human_talk_workspace\tata_seed\manual_tata_annotations.csv `
#     --out_dir human_talk_workspace\tata_seed

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}

SPEAKER_LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
]

INTERVIEWER_LABELS = [
    "Brene_Brown_interviewer",
    "Eckhart_Tolle_interviewer",
    "Eric_Thomas_interviewer",
    "Gary_Vee_interviewer",
    "Jay_Shetty_interviewer",
]

EVENT_LABELS = [
    "music_present",
    "applause_present",
    "laughter_present",
    "crowd_noise_present",
    "silence_present",
]

TATA_LABELS = SPEAKER_LABELS + INTERVIEWER_LABELS + EVENT_LABELS

LABEL_ALIASES: Dict[str, List[str]] = {
    "Brene_Brown": ["brene_brown", "brene", "brene-brown"],
    "Eckhart_Tolle": ["eckhart_tolle", "eckhart", "eckhart-tolle"],
    "Eric_Thomas": ["eric_thomas", "eric", "eric-thomas"],
    "Gary_Vee": ["gary_vee", "garyvee", "gary", "gary-vee"],
    "Jay_Shetty": ["jay_shetty", "jay", "jay-shetty"],
    "Brene_Brown_interviewer": [
        "brene_brown_interviewer",
        "brene_brown_interviwer",
        "brene_interviewer",
        "brene_interviwer",
        "brene_brown_other_speaker",
    ],
    "Eckhart_Tolle_interviewer": [
        "eckhart_tolle_interviewer",
        "eckhart_tolle_interviwer",
        "eckhart_interviewer",
        "eckhart_interviwer",
        "eckhart_tolle_other_speaker",
    ],
    "Eric_Thomas_interviewer": [
        "eric_thomas_interviewer",
        "eric_thomas_interviwer",
        "eric_interviewer",
        "eric_interviwer",
        "eric_thomas_other_speaker",
    ],
    "Gary_Vee_interviewer": [
        "gary_vee_interviewer",
        "gary_vee_interviwer",
        "gary_interviewer",
        "gary_interviwer",
        "gary_vee_other_speaker",
    ],
    "Jay_Shetty_interviewer": [
        "jay_shetty_interviewer",
        "jay_shetty_interviwer",
        "jay_interviewer",
        "jay_interviwer",
        "jay_shetty_other_speaker",
    ],
    "music_present": [
        "music",
        "background_music",
        "bg_music",
        "song",
        "instrumental",
    ],
    "applause_present": [
        "applause",
        "clap",
        "claps",
        "clapping",
    ],
    "laughter_present": [
        "laughter",
        "laugh",
        "laughing",
    ],
    "crowd_noise_present": [
        "crowd_noise",
        "crowd_moise",  # historical typo support
        "crowd",
        "audience_noise",
        "audience",
        "cheer",
        "cheering",
        "room_noise",
    ],
    "silence_present": [
        "silence",
        "silent",
        "low_speech",
        "no_speech",
        "very_low_signal",
    ],
}

SPLITS = ["train", "val", "test"]


def safe_name(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalise_key(text: str) -> str:
    text = str(text).replace("\\", "/").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "present", "positive"}


def split_label_text(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"[|,;+]", text)
    return [safe_name(p) for p in parts if str(p).strip()]


def discover_audio_files(seed_root: Path) -> List[Path]:
    if not seed_root.exists():
        raise FileNotFoundError(f"Seed root not found: {seed_root}")

    files = [
        p for p in seed_root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    return sorted(files, key=lambda p: str(p).replace("\\", "/").lower())


def label_dict_empty() -> Dict[str, int]:
    return {label: 0 for label in TATA_LABELS}


def _contains_alias(text: str, alias: str) -> bool:
    norm = "_" + normalise_key(text) + "_"
    alias_norm = "_" + normalise_key(alias) + "_"
    return alias_norm in norm


def apply_label_aliases(text: str, labels: Dict[str, int]) -> None:
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if _contains_alias(text, alias):
                labels[label] = 1
                break


def infer_labels_from_path(path: Path, seed_root: Path) -> Tuple[Dict[str, int], str, str]:
    labels = label_dict_empty()

    try:
        rel_path = path.resolve().relative_to(seed_root.resolve())
    except Exception:
        rel_path = path

    parts = [safe_name(p) for p in rel_path.parts]
    top_group = parts[0] if parts else "unknown"
    full_text = "_".join(parts + [safe_name(path.stem)])

    # Strong layout priors. These avoid accidental activation of speaker labels inside
    # interviewer labels and make event folders deterministic.
    if top_group == "target_speaker":
        for label in SPEAKER_LABELS:
            for alias in LABEL_ALIASES[label]:
                if _contains_alias(full_text, alias):
                    labels[label] = 1
                    break
    elif top_group == "other_speaker":
        for label in INTERVIEWER_LABELS:
            for alias in LABEL_ALIASES[label]:
                if _contains_alias(full_text, alias):
                    labels[label] = 1
                    break
    elif top_group == "events":
        for label in EVENT_LABELS:
            for alias in LABEL_ALIASES[label]:
                if _contains_alias(full_text, alias):
                    labels[label] = 1
                    break
    else:
        apply_label_aliases(full_text, labels)

    # Mixed conditions can still be represented by folder names, filenames, or annotations.
    # For example: Brene_Brown__music__applause.wav should set all three labels.
    for label in EVENT_LABELS:
        for alias in LABEL_ALIASES[label]:
            if _contains_alias(full_text, alias):
                labels[label] = 1
                break

    # If a target folder contains a mixed interviewer marker, allow it too.
    if top_group == "target_speaker":
        for label in INTERVIEWER_LABELS:
            for alias in LABEL_ALIASES[label]:
                if _contains_alias(full_text, alias):
                    labels[label] = 1
                    break

    # If an other_speaker folder contains explicit target speaker marker, allow it too.
    if top_group == "other_speaker":
        for label in SPEAKER_LABELS:
            for alias in LABEL_ALIASES[label]:
                if _contains_alias(full_text, alias):
                    labels[label] = 1
                    break

    active = [label for label in TATA_LABELS if labels[label] == 1]
    condition_name = "clean_unknown" if not active else "__".join(active)

    return labels, top_group, condition_name


def canonical_label_name(value: str) -> str:
    value_safe = safe_name(value)

    for label in TATA_LABELS:
        if value == label or value_safe == safe_name(label):
            return label

    if not value_safe.endswith("_present"):
        candidate = f"{value_safe}_present"
        for label in TATA_LABELS:
            if candidate == safe_name(label):
                return label

    for label, aliases in LABEL_ALIASES.items():
        if value_safe in {safe_name(a) for a in aliases}:
            return label

    return value


def build_annotation_lookup(path: Path | None) -> Dict[str, Dict[str, int]]:
    """
    Optional manual annotations CSV.

    Supported matching keys:
      - rel_path
      - source_file
      - source_path
      - abs_path

    Supported label formats:
      - binary columns named like Brene_Brown / music_present / etc.
      - a `labels` column containing labels separated by |, comma, semicolon, or plus.
    """
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotations_csv not found: {path}")

    lookup: Dict[str, Dict[str, int]] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = label_dict_empty()

            for label in TATA_LABELS:
                if label in row and str(row.get(label, "")).strip() != "":
                    labels[label] = 1 if truthy(row[label]) else 0

            for raw_label in split_label_text(row.get("labels", "")):
                label = canonical_label_name(raw_label)
                if label in labels:
                    labels[label] = 1

            keys = []
            for key in ["rel_path", "source_file", "source_path", "abs_path"]:
                value = str(row.get(key, "") or "").replace("\\", "/").strip()
                if value:
                    keys.append(value)
                    keys.append(Path(value).name)
                    keys.append(normalise_key(value))

            for key in keys:
                lookup[str(key)] = dict(labels)

    return lookup


def apply_annotations(
    path: Path,
    seed_root: Path,
    labels: Dict[str, int],
    annotation_lookup: Dict[str, Dict[str, int]],
    mode: str,
) -> Tuple[Dict[str, int], bool]:
    if not annotation_lookup:
        return labels, False

    try:
        rel_path = str(path.resolve().relative_to(seed_root.resolve())).replace("\\", "/")
    except Exception:
        rel_path = str(path).replace("\\", "/")

    candidates = [
        rel_path,
        path.name,
        str(path).replace("\\", "/"),
        str(path.resolve()).replace("\\", "/"),
        normalise_key(rel_path),
        normalise_key(path.name),
    ]

    ann = None
    for key in candidates:
        if key in annotation_lookup:
            ann = annotation_lookup[key]
            break

    if ann is None:
        return labels, False

    if mode == "override":
        return dict(ann), True

    merged = dict(labels)
    for label, value in ann.items():
        if int(value) == 1:
            merged[label] = 1

    return merged, True


def label_signature(labels: Dict[str, int]) -> str:
    active = [label for label in TATA_LABELS if int(labels[label]) == 1]
    return "__".join(active) if active else "unlabelled"


def decision_hint(labels: Dict[str, int]) -> Tuple[str, str]:
    active_speakers = [label for label in SPEAKER_LABELS if labels[label] == 1]
    active_interviewers = [label for label in INTERVIEWER_LABELS if labels[label] == 1]
    active_events = [label for label in EVENT_LABELS if labels[label] == 1]
    active_count = sum(int(labels[label]) for label in TATA_LABELS)

    if active_count == 0:
        return "needs_review", "no_positive_tata_label"

    if active_speakers and active_interviewers:
        return "needs_review", "speaker_and_interviewer_overlap"

    if active_speakers and active_events:
        if labels["silence_present"] == 1:
            return "needs_review", "speaker_with_silence_or_low_speech"
        return "accepted_with_warning", "speaker_with_background_event"

    if active_speakers:
        return "accepted", "speaker_identity_present"

    if active_interviewers:
        return "rejected", "interviewer_without_target_speaker"

    return "rejected", "event_or_silence_without_speaker_identity"


def assign_splits(rows: List[Dict[str, Any]], seed: int, ratios: Tuple[float, float, float]) -> None:
    train_ratio, val_ratio, test_ratio = ratios

    explicit = [row for row in rows if row.get("split") in SPLITS]
    if len(explicit) == len(rows):
        return

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["label_signature"])].append(row)

    rng = random.Random(seed)

    for _, items in sorted(groups.items()):
        rng.shuffle(items)
        n = len(items)

        if n == 1:
            counts = (1, 0, 0)
        elif n == 2:
            counts = (1, 1, 0)
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            n_test = n - n_train - n_val

            if n_test <= 0:
                n_test = 1
                if n_train > n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1

            counts = (n_train, n_val, n_test)

        split_values = (["train"] * counts[0]) + (["val"] * counts[1]) + (["test"] * counts[2])
        split_values = split_values[:n]

        for row, split in zip(items, split_values):
            row["split"] = split


def make_sample_id(row: Dict[str, Any], used: set[str]) -> str:
    rel = str(row["rel_path"])
    h = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    base = f"tata_{safe_name(row['split'])}_{safe_name(row['condition_name'])}_{h}"
    base = base[:180]

    if base not in used:
        used.add(base)
        return base

    i = 2
    while True:
        candidate = f"{base}_{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    preferred = [
        "sample_id",
        "abs_path",
        "rel_path",
        "source_file",
        "split",
        "primary_label",
        "labels",
        "num_labels",
        "is_clean_seed",
        "is_synthetic",
        "label_group",
        "condition_name",
        "label_signature",
        "is_mixed_clip",
        "decision_hint",
        "decision_reason",
        "annotation_applied",
    ] + TATA_LABELS

    for field in preferred:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TinyAudioTriageAgent multi-label seed manifest."
    )

    parser.add_argument("--seed_root", default="human_talk_triage_seed_dataset")
    parser.add_argument("--out_dir", default="human_talk_workspace/tata_seed")
    parser.add_argument("--annotations_csv", default="")
    parser.add_argument(
        "--annotation_mode",
        choices=["merge", "override"],
        default="merge",
        help="merge adds manual positive labels; override replaces inferred labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--dataset_name", default="tata_seed_v0_5_speaker_event_multilabel")

    args = parser.parse_args()

    seed_root = Path(args.seed_root)
    out_dir = Path(args.out_dir)
    annotations_csv = Path(args.annotations_csv) if str(args.annotations_csv).strip() else None

    ratio_sum = float(args.train_ratio + args.val_ratio + args.test_ratio)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0. Got {ratio_sum:.6f}: "
            f"train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}"
        )

    audio_files = discover_audio_files(seed_root)
    annotation_lookup = build_annotation_lookup(annotations_csv)

    rows: List[Dict[str, Any]] = []

    for path in audio_files:
        try:
            rel_path = str(path.resolve().relative_to(seed_root.resolve())).replace("\\", "/")
        except Exception:
            rel_path = str(path).replace("\\", "/")

        inferred_labels, label_group, condition_name = infer_labels_from_path(path, seed_root)
        labels, annotation_applied = apply_annotations(
            path=path,
            seed_root=seed_root,
            labels=inferred_labels,
            annotation_lookup=annotation_lookup,
            mode=args.annotation_mode,
        )

        signature = label_signature(labels)
        active_labels = [label for label in TATA_LABELS if labels[label] == 1]
        hint, reason = decision_hint(labels)

        split = ""
        rel_parts = [safe_name(part) for part in Path(rel_path).parts]
        for candidate in rel_parts:
            if candidate in SPLITS:
                split = candidate
                break

        rows.append(
            {
                "sample_id": "",
                "abs_path": str(path.resolve()),
                "rel_path": rel_path,
                "source_file": path.name,
                "split": split,
                "primary_label": active_labels[0] if active_labels else "unlabelled",
                "labels": "|".join(active_labels),
                "num_labels": len(active_labels),
                "is_clean_seed": 1,
                "is_synthetic": 0,
                "dataset_source": "tata_seed",
                "label_group": label_group,
                "condition_name": condition_name if active_labels else "unlabelled",
                "label_signature": signature,
                "is_mixed_clip": 1 if len(active_labels) > 1 else 0,
                "decision_hint": hint,
                "decision_reason": reason,
                "annotation_applied": 1 if annotation_applied else 0,
                **labels,
            }
        )

    assign_splits(
        rows=rows,
        seed=int(args.seed),
        ratios=(float(args.train_ratio), float(args.val_ratio), float(args.test_ratio)),
    )

    used_ids: set[str] = set()
    for row in rows:
        row["sample_id"] = make_sample_id(row, used_ids)

    split_order = {"train": 0, "val": 1, "test": 2}
    rows = sorted(
        rows,
        key=lambda r: (
            split_order.get(str(r.get("split", "")), 99),
            str(r.get("label_signature", "")),
            str(r.get("rel_path", "")),
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "tata_seed_manifest.csv"
    labels_json_path = out_dir / "labels.json"
    summary_path = out_dir / "tata_seed_summary.json"
    summary_md_path = out_dir / "tata_seed_summary.md"

    write_csv(manifest_path, rows)

    labels_payload = {
        "dataset_name": args.dataset_name,
        "task": "tiny_audio_triage_speaker_event_multilabel",
        "model_family": "TinyAudioTriageAgent",
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "description": (
            "True multi-label seed labels for speaker identity, interviewer identity, "
            "and audio-content triage before final multi-label early-exit training."
        ),
        "label_groups": {
            "speaker_identity": SPEAKER_LABELS,
            "interviewer_identity": INTERVIEWER_LABELS,
            "event_or_background": EVENT_LABELS,
        },
        "labels": TATA_LABELS,
    }
    write_json(labels_json_path, labels_payload)

    split_counts = Counter(row["split"] for row in rows)
    label_counts = {label: int(sum(int(row[label]) for row in rows)) for label in TATA_LABELS}
    signature_counts = Counter(row["label_signature"] for row in rows)
    decision_counts = Counter(row["decision_hint"] for row in rows)
    group_counts = Counter(row["label_group"] for row in rows)

    summary = {
        "dataset_name": args.dataset_name,
        "task": "tiny_audio_triage_speaker_event_multilabel",
        "seed_root": str(seed_root),
        "annotations_csv": str(annotations_csv) if annotations_csv else "",
        "annotation_mode": str(args.annotation_mode),
        "outputs": {
            "manifest": str(manifest_path),
            "labels_json": str(labels_json_path),
            "summary_json": str(summary_path),
            "summary_md": str(summary_md_path),
        },
        "rows": len(rows),
        "labels": TATA_LABELS,
        "label_groups": {
            "speaker_identity": SPEAKER_LABELS,
            "interviewer_identity": INTERVIEWER_LABELS,
            "event_or_background": EVENT_LABELS,
        },
        "split_counts": dict(split_counts),
        "label_positive_counts": label_counts,
        "label_signature_counts": dict(signature_counts.most_common()),
        "decision_hint_counts": dict(decision_counts),
        "label_group_counts": dict(group_counts),
        "mixed_clip_count": int(sum(int(row["is_mixed_clip"]) for row in rows)),
        "annotation_applied_count": int(sum(int(row["annotation_applied"]) for row in rows)),
        "safety": {
            "raw_files_modified": False,
            "raw_files_deleted": False,
            "dataset_files_copied": False,
            "manifest_only": True,
        },
    }
    write_json(summary_path, summary)

    md_lines = [
        "# TinyAudioTriageAgent Seed Manifest Summary",
        "",
        f"Dataset: `{args.dataset_name}`",
        f"Rows: `{len(rows)}`",
        "",
        "## Split counts",
        "",
        "| Split | Count |",
        "|---|---:|",
    ]
    for split in SPLITS:
        md_lines.append(f"| {split} | {split_counts.get(split, 0)} |")

    md_lines.extend(["", "## Label-positive counts", "", "| Label | Count |", "|---|---:|"])
    for label in TATA_LABELS:
        md_lines.append(f"| {label} | {label_counts[label]} |")

    md_lines.extend(["", "## Decision hints", "", "| Decision | Count |", "|---|---:|"])
    for decision, count in sorted(decision_counts.items()):
        md_lines.append(f"| {decision} | {count} |")

    md_lines.extend([
        "",
        "## Notes",
        "",
        "- This script writes manifests only; it does not move, delete, or copy audio files.",
        "- Mixed clips are represented by multiple binary label columns, not by separate combination folders.",
        "- Speaker identity, interviewer identity, and event/background tags are all direct output labels.",
        "- Common `interviwer` spelling mistakes are accepted as aliases, but canonical labels use `interviewer`.",
    ])
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\nTinyAudioTriageAgent seed manifest created")
    print("-" * 90)
    print(f"Seed root:    {seed_root}")
    print(f"Manifest:     {manifest_path}")
    print(f"Labels JSON:  {labels_json_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Rows:         {len(rows)}")
    print(f"Mixed clips:  {summary['mixed_clip_count']}")
    print("\nLabel-positive counts:")
    for label in TATA_LABELS:
        print(f"  {label}: {label_counts[label]}")
    print("-" * 90)


if __name__ == "__main__":
    main()
