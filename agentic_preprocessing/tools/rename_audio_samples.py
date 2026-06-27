"""
Safe audio sample renaming utility.

Supports:
1. TinyAudioTriageAgent seed dataset
2. Raw speaker folders inside human_talk_dataset

Safety:
- dry-run by default
- writes rename manifest
- writes undo PowerShell script
- uses two-phase rename to avoid filename collisions
"""

from __future__ import annotations

import argparse
import csv
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    name = re.sub(r"_+", "_", name)
    return name


def natural_sort_key(path: Path):
    text = path.name

    def convert(part):
        return int(part) if part.isdigit() else part.lower()

    return [convert(p) for p in re.split(r"(\d+)", text)]


def parse_classes(classes: str | None) -> List[str]:
    if not classes:
        return []
    return [x.strip() for x in classes.split(",") if x.strip()]


def audio_files_in_dir(folder: Path) -> List[Path]:
    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=natural_sort_key,
    )


def collect_triage_seed_renames(root: Path, zero_pad: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    layout = [
        ("target_speaker", "target_speaker"),
        ("other_speaker", "other_speaker"),
        ("events", "event"),
    ]

    for top_folder, label_type in layout:
        base = root / top_folder

        if not base.exists():
            continue

        for group_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            group_name = sanitize_name(group_dir.name)
            files = audio_files_in_dir(group_dir)

            for idx, old_path in enumerate(files, start=1):
                ext = old_path.suffix.lower()

                if label_type == "target_speaker":
                    new_name = f"{group_name}__target_speaker__{idx:0{zero_pad}d}{ext}"
                elif label_type == "other_speaker":
                    new_name = f"{group_name}__other_speaker__{idx:0{zero_pad}d}{ext}"
                else:
                    new_name = f"{group_name}__event__{idx:0{zero_pad}d}{ext}"

                new_path = old_path.with_name(new_name)

                rows.append(
                    {
                        "mode": "triage_seed",
                        "label_group": top_folder,
                        "label_type": label_type,
                        "class_or_event": group_name,
                        "index": str(idx),
                        "old_path": str(old_path),
                        "new_path": str(new_path),
                        "old_name": old_path.name,
                        "new_name": new_name,
                        "status": "unchanged" if old_path == new_path else "rename",
                    }
                )

    return rows


def collect_speaker_raw_renames(root: Path, classes: Iterable[str], zero_pad: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for class_name in classes:
        class_name = sanitize_name(class_name)
        class_dir = root / class_name

        if not class_dir.exists():
            rows.append(
                {
                    "mode": "speaker_raw",
                    "label_group": "raw_speaker",
                    "label_type": "speaker_class",
                    "class_or_event": class_name,
                    "index": "",
                    "old_path": str(class_dir),
                    "new_path": "",
                    "old_name": "",
                    "new_name": "",
                    "status": "class_folder_missing",
                }
            )
            continue

        files = audio_files_in_dir(class_dir)

        for idx, old_path in enumerate(files, start=1):
            ext = old_path.suffix.lower()
            new_name = f"{class_name}__{idx:0{zero_pad}d}{ext}"
            new_path = old_path.with_name(new_name)

            rows.append(
                {
                    "mode": "speaker_raw",
                    "label_group": "raw_speaker",
                    "label_type": "speaker_class",
                    "class_or_event": class_name,
                    "index": str(idx),
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                    "old_name": old_path.name,
                    "new_name": new_name,
                    "status": "unchanged" if old_path == new_path else "rename",
                }
            )

    return rows


def check_duplicate_targets(rows: List[Dict[str, str]]) -> None:
    target_counts: Dict[str, int] = {}

    for row in rows:
        if row["status"] in {"rename", "unchanged"} and row["new_path"]:
            target_counts[row["new_path"]] = target_counts.get(row["new_path"], 0) + 1

    duplicates = [path for path, count in target_counts.items() if count > 1]

    if duplicates:
        msg = "\n".join(duplicates[:20])
        raise RuntimeError(f"Duplicate target paths detected:\n{msg}")


def write_manifest(rows: List[Dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "label_group",
        "label_type",
        "class_or_event",
        "index",
        "old_path",
        "new_path",
        "old_name",
        "new_name",
        "status",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ps_quote(path: str) -> str:
    return "'" + path.replace("'", "''") + "'"


def write_undo_script(rows: List[Dict[str, str]], undo_path: Path) -> None:
    undo_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Auto-generated undo script for audio renaming")
    lines.append("# Run only if you need to revert the applied rename operation.")
    lines.append("")

    for row in rows:
        if row["status"] == "rename":
            old_path = row["old_path"]
            new_path = row["new_path"]
            lines.append(f"if (Test-Path {ps_quote(new_path)}) {{")
            lines.append(f"  Move-Item -LiteralPath {ps_quote(new_path)} -Destination {ps_quote(old_path)} -Force")
            lines.append("}")

    undo_path.write_text("\n".join(lines), encoding="utf-8")


def apply_renames(rows: List[Dict[str, str]]) -> None:
    rename_rows = [r for r in rows if r["status"] == "rename"]

    temp_pairs: List[Tuple[Path, Path, Path]] = []

    for row in rename_rows:
        old_path = Path(row["old_path"])
        new_path = Path(row["new_path"])

        if not old_path.exists():
            raise FileNotFoundError(f"Missing source file: {old_path}")

        temp_path = old_path.with_name(f".tmp_rename_{uuid.uuid4().hex}{old_path.suffix.lower()}")

        if temp_path.exists():
            raise FileExistsError(f"Temporary path already exists: {temp_path}")

        temp_pairs.append((old_path, temp_path, new_path))

    # Phase 1: old -> temp
    for old_path, temp_path, _ in temp_pairs:
        old_path.rename(temp_path)

    # Phase 2: temp -> final
    for _, temp_path, new_path in temp_pairs:
        if new_path.exists():
            raise FileExistsError(f"Target already exists after temp rename: {new_path}")
        temp_path.rename(new_path)


def shorten(text: str, width: int) -> str:
    text = str(text)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_table(rows: List[Dict[str, str]], columns: List[Tuple[str, str, int]]) -> None:
    """
    Print a small readable CLI table.

    columns:
      list of (field_name, display_name, width)
    """
    if not rows:
        print("  No rows to display.")
        return

    header = "  " + " | ".join(shorten(title, width).ljust(width) for _, title, width in columns)
    sep = "  " + "-+-".join("-" * width for _, _, width in columns)

    print(header)
    print(sep)

    for row in rows:
        line = "  " + " | ".join(
            shorten(row.get(field, ""), width).ljust(width)
            for field, _, width in columns
        )
        print(line)


def print_cli_preview(rows: List[Dict[str, str]], rows_per_group: int = 10) -> None:
    """
    Display rename preview clearly in CLI, grouped by class/event folder.
    """
    print("")
    print("=" * 110)
    print("CLI Rename Preview")
    print("=" * 110)

    grouped: Dict[str, List[Dict[str, str]]] = {}

    for row in rows:
        group = row.get("class_or_event", "unknown")
        grouped.setdefault(group, []).append(row)

    for group_name, group_rows in grouped.items():
        rename_count = sum(1 for r in group_rows if r.get("status") == "rename")
        unchanged_count = sum(1 for r in group_rows if r.get("status") == "unchanged")
        missing_count = sum(1 for r in group_rows if r.get("status") == "class_folder_missing")

        print("")
        print(f"Group: {group_name}")
        print(f"Rows: {len(group_rows)} | To rename: {rename_count} | Unchanged: {unchanged_count} | Missing: {missing_count}")
        print("-" * 110)

        preview_rows = group_rows[:rows_per_group]

        print_table(
            preview_rows,
            columns=[
                ("index", "Index", 6),
                ("label_group", "Group", 16),
                ("label_type", "Type", 16),
                ("old_name", "Old name", 32),
                ("new_name", "New name", 38),
                ("status", "Status", 12),
            ],
        )

        remaining = len(group_rows) - len(preview_rows)
        if remaining > 0:
            print(f"  ... {remaining} more rows in this group")

    print("")
    print("=" * 110)



def summarize(rows: List[Dict[str, str]]) -> None:
    total = len(rows)
    rename_count = sum(1 for r in rows if r["status"] == "rename")
    unchanged_count = sum(1 for r in rows if r["status"] == "unchanged")
    missing_count = sum(1 for r in rows if r["status"] == "class_folder_missing")

    print("")
    print("Rename summary")
    print("--------------")
    print(f"Total rows:      {total}")
    print(f"To rename:       {rename_count}")
    print(f"Unchanged:       {unchanged_count}")
    print(f"Missing folders: {missing_count}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely rename audio samples into sequential format.")

    parser.add_argument("--mode", choices=["triage_seed", "speaker_raw"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--classes", default="")
    parser.add_argument("--zero_pad", type=int, default=4)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--undo_script", required=True)
    parser.add_argument(
        "--preview_rows_per_group",
        type=int,
        default=10,
        help="Number of rename preview rows to show per class/event group in CLI.",
    )

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry_run", action="store_true")
    action.add_argument("--apply", action="store_true")

    args = parser.parse_args()

    root = Path(args.root)

    if args.mode == "triage_seed":
        rows = collect_triage_seed_renames(root=root, zero_pad=args.zero_pad)
    else:
        classes = parse_classes(args.classes)
        if not classes:
            raise ValueError("--classes is required for speaker_raw mode")
        rows = collect_speaker_raw_renames(root=root, classes=classes, zero_pad=args.zero_pad)

    check_duplicate_targets(rows)

    print_cli_preview(
        rows,
        rows_per_group=args.preview_rows_per_group,
    )

    write_manifest(rows, Path(args.manifest))
    write_undo_script(rows, Path(args.undo_script))

    summarize(rows)

    print(f"Manifest written:    {args.manifest}")
    print(f"Undo script written: {args.undo_script}")

    if args.apply:
        print("")
        print("Applying renames...")
        apply_renames(rows)
        print("Rename completed.")
    else:
        print("")
        print("Dry run only. No files were renamed.")
        print("Check the manifest first. Then rerun with --apply.")


if __name__ == "__main__":
    main()