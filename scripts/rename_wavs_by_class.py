# scripts/rename_wavs_by_class.py
# Rename audio files inside each class directory using:
#   class_name_0001.wav
#   class_name_0002.flac
#   class_name_0003.mp3
#
# Supports dry-run first and writes a CSV manifest for traceability.

from __future__ import annotations

import argparse
import csv
import re
import uuid
from pathlib import Path


AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}


def natural_key(path: Path):
    """
    Sort filenames naturally:
    file_2.wav before file_10.wav
    """
    text = path.name.lower()
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def safe_label_name(name: str) -> str:
    """
    Convert folder name into a safe filename prefix.
    Example:
      "car crash" -> "car_crash"
      "rain_ thunderstorm" -> "rain_thunderstorm"
    """
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def collect_class_dirs(root: Path) -> list[Path]:
    """
    Collect immediate class folders under root.
    Example:
      clean_seed/car_crash
      clean_seed/conversation
      clean_seed/fireworks
    """
    return sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )


def collect_audio_files(class_dir: Path) -> list[Path]:
    """
    Collect supported audio files directly inside a class folder.
    This does not search recursively.
    """
    return sorted(
        [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ],
        key=natural_key,
    )


def parse_existing_index(label: str, path: Path) -> int | None:
    """
    Detect whether a file is already named like:
      label_0001.wav
      label_12.flac

    Returns the number if matched, otherwise None.
    """
    pattern = rf"^{re.escape(label)}_(\d+)$"
    match = re.match(pattern, path.stem.lower())
    if not match:
        return None
    return int(match.group(1))


def make_index(i: int, digits: int) -> str:
    if digits > 0:
        return str(i).zfill(digits)
    return str(i)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename audio files inside each class directory using "
            "class_name_index while preserving original file extensions."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing class folders, e.g. multilabel_data/clean_seed",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index for each class. Default: 1",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Zero padding. Example: --digits 4 gives car_crash_0001.wav. Default: 4",
    )
    parser.add_argument(
        "--manifest",
        default="rename_manifest.csv",
        help="CSV file to store old/new filename mapping.",
    )
    parser.add_argument(
        "--skip_already_named",
        action="store_true",
        help=(
            "Skip files already named like class_0001.ext. "
            "Useful if you already renamed WAV files and now only want to rename remaining FLAC/MP3/etc."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, only dry-run is shown.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    class_dirs = collect_class_dirs(root)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {root}")

    manifest_path = root / args.manifest
    rows = []

    print(f"\nRoot: {root}")
    print(f"Mode: {'APPLY / RENAME FILES' if args.apply else 'DRY RUN ONLY'}")
    print(f"Supported audio extensions: {', '.join(sorted(AUDIO_EXTS))}")
    print(f"Skip already named files: {args.skip_already_named}")
    print("-" * 90)

    total_audio_files = 0
    total_planned = 0
    total_skipped = 0

    for class_dir in class_dirs:
        label = safe_label_name(class_dir.name)
        audio_files = collect_audio_files(class_dir)

        print(f"\nClass: {class_dir.name} -> prefix: {label}")
        print(f"Found audio files: {len(audio_files)}")

        total_audio_files += len(audio_files)

        if not audio_files:
            continue

        planned: list[tuple[Path, Path]] = []
        skipped: list[Path] = []

        # If skipping already named files, continue numbering after existing max index.
        existing_indices = []
        if args.skip_already_named:
            for p in audio_files:
                idx = parse_existing_index(label, p)
                if idx is not None:
                    existing_indices.append(idx)

            next_index = max(existing_indices, default=args.start - 1) + 1
        else:
            next_index = args.start

        for old_path in audio_files:
            existing_idx = parse_existing_index(label, old_path)

            if args.skip_already_named and existing_idx is not None:
                skipped.append(old_path)
                continue

            index = make_index(next_index, args.digits)

            # Preserve original extension correctly:
            # .wav stays .wav, .flac stays .flac, .mp3 stays .mp3
            new_name = f"{label}_{index}{old_path.suffix.lower()}"
            new_path = class_dir / new_name

            planned.append((old_path, new_path))
            next_index += 1

        print(f"Planned renames: {len(planned)}")
        print(f"Skipped already named: {len(skipped)}")

        total_planned += len(planned)
        total_skipped += len(skipped)

        if not planned:
            continue

        # Safety check 1: target names must be unique in the plan.
        target_names_lower = [new_path.name.lower() for _, new_path in planned]
        if len(target_names_lower) != len(set(target_names_lower)):
            raise RuntimeError(f"Duplicate target filenames detected in planned rename: {class_dir}")

        # Safety check 2: do not overwrite files that are not part of this rename plan.
        planned_old_paths = {old_path.resolve() for old_path, _ in planned}
        for old_path, new_path in planned:
            if new_path.exists() and new_path.resolve() not in planned_old_paths:
                raise RuntimeError(
                    "Target file already exists and is not part of this rename plan:\n"
                    f"  old: {old_path}\n"
                    f"  new: {new_path}\n"
                    "Use --skip_already_named or inspect the directory manually."
                )

        # Show first few examples
        for old_path, new_path in planned[:10]:
            print(f"  {old_path.name}  ->  {new_path.name}")

        if len(planned) > 10:
            print(f"  ... {len(planned) - 10} more")

        # Store manifest rows
        for old_path, new_path in planned:
            rows.append({
                "class_dir": class_dir.name,
                "old_path": str(old_path),
                "new_path": str(new_path),
                "old_name": old_path.name,
                "new_name": new_path.name,
                "old_ext": old_path.suffix.lower(),
                "new_ext": new_path.suffix.lower(),
                "action": "rename",
            })

        if args.apply:
            # Two-stage rename avoids collisions, for example:
            # file_a.wav -> car_crash_0001.wav
            # file_b.wav -> car_crash_0002.wav
            #
            # Even if target names overlap old names, temporary names keep it safe.
            temp_pairs: list[tuple[Path, Path]] = []

            for old_path, new_path in planned:
                temp_name = f".tmp_rename_{uuid.uuid4().hex}_{old_path.name}"
                temp_path = old_path.with_name(temp_name)
                old_path.rename(temp_path)
                temp_pairs.append((temp_path, new_path))

            for temp_path, new_path in temp_pairs:
                temp_path.rename(new_path)

    # Write manifest
    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class_dir",
                    "old_path",
                    "new_path",
                    "old_name",
                    "new_name",
                    "old_ext",
                    "new_ext",
                    "action",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "-" * 90)
    print(f"Total supported audio files found: {total_audio_files}")
    print(f"Total planned renames: {total_planned}")
    print(f"Total skipped already named: {total_skipped}")

    if rows:
        print(f"Manifest written to: {manifest_path}")
    else:
        print("No manifest written because no renames were planned.")

    if not args.apply:
        print("\nDry run completed. No files were renamed.")
        print("Run again with --apply to actually rename files.")
    else:
        print("\nRenaming completed successfully.")


if __name__ == "__main__":
    main()