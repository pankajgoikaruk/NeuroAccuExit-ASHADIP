# agentic_preprocessing/run_tata_segment_manifest_builder.py

"""
Build weak 1-second TATA segment manifest from reviewed 5-second clip-level manifest.

Input:
  human_talk_workspace/tata_2/metadata/tata_clip_level_manifest_training_ready.csv

Output:
  human_talk_workspace/tata_2/segments/
  human_talk_workspace/tata_2/metadata/tata_segment_manifest.csv

This script:
- reads reviewed clip-level multi-hot labels
- loads each audio file
- resamples to 16 kHz mono
- exports 1-second segment WAVs
- inherits parent clip labels for every segment
- creates a manifest compatible with scripts/extract_multilabel_features.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf


LABELS = [
    "Brene_Brown",
    "Eckhart_Tolle",
    "Eric_Thomas",
    "Gary_Vee",
    "Jay_Shetty",
    "Nick_Vujicic",
    "other_speaker_present",
    "music_present",
    "applause_present",
    "laughter_present",
    "crowd_cheer_present",
    "silence_present",
]


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        return y

    if y.ndim == 2:
        return y.mean(axis=1).astype(np.float32)

    raise ValueError(f"Unsupported audio shape: {y.shape}")


def read_audio_any(path: Path) -> tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_mono(y), int(sr)
    except Exception as sf_error:
        try:
            import librosa

            y, sr = librosa.load(str(path), sr=None, mono=True)
            return y.astype(np.float32), int(sr)
        except Exception as librosa_error:
            raise RuntimeError(
                f"Could not read audio: {path}\n"
                f"soundfile error: {sf_error}\n"
                f"librosa error: {librosa_error}"
            )


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        return resample_poly(y, up, down).astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr).astype(np.float32)


def normalise_audio(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)

    if len(y) == 0:
        return y

    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))

    if peak > 1e-8:
        y = y / max(peak, 1.0)

    return y.astype(np.float32)


def stable_split(key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF

    if value < train_ratio:
        return "train"

    if value < train_ratio + val_ratio:
        return "val"

    return "test"


def active_labels_from_row(row: pd.Series) -> list[str]:
    active = []

    for lab in LABELS:
        if int(row.get(lab, 0)) == 1:
            active.append(lab)

    return active


def crop_or_pad(segment: np.ndarray, target_len: int) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32)

    if len(segment) >= target_len:
        return segment[:target_len].astype(np.float32)

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(segment)] = segment
    return out


def build_segments(args: argparse.Namespace) -> dict[str, Any]:
    clip_manifest = Path(args.clip_manifest)
    out_dir = Path(args.out_dir)
    segment_root = out_dir / "segment_wavs"
    metadata_dir = out_dir / "metadata"

    segment_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(clip_manifest)

    required = {"clip_id", "file_path", "primary_label"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(f"Clip manifest missing required columns: {sorted(missing)}")

    missing_labels = [lab for lab in LABELS if lab not in df.columns]
    if missing_labels:
        raise RuntimeError(f"Missing label columns: {missing_labels}")

    if "exclude_from_tata_training" in df.columns:
        df = df[df["exclude_from_tata_training"].astype(str) != "1"].copy()

    segment_len = int(round(args.sample_rate * args.segment_sec))
    hop_len = int(round(args.sample_rate * args.hop_sec))

    out_rows = []
    errors = []

    print("")
    print("Building weak TATA 1-second segment manifest")
    print("-" * 90)
    print(f"Input clip manifest: {clip_manifest}")
    print(f"Rows:                {len(df)}")
    print(f"Output dir:          {out_dir}")
    print(f"Segment root:        {segment_root}")
    print(f"Sample rate:         {args.sample_rate}")
    print(f"Segment seconds:     {args.segment_sec}")
    print(f"Hop seconds:         {args.hop_sec}")
    print("-" * 90)

    for idx, row in df.reset_index(drop=True).iterrows():
        clip_id = safe_name(str(row["clip_id"]))
        audio_path = Path(str(row["file_path"]))

        if not audio_path.exists():
            errors.append({"clip_id": clip_id, "file_path": str(audio_path), "error": "missing_file"})
            continue

        active = active_labels_from_row(row)

        if not active:
            errors.append({"clip_id": clip_id, "file_path": str(audio_path), "error": "no_active_labels"})
            continue

        try:
            y, sr = read_audio_any(audio_path)
            y = resample_audio(y, sr, args.sample_rate)
            y = normalise_audio(y)
        except Exception as e:
            errors.append({"clip_id": clip_id, "file_path": str(audio_path), "error": str(e)})
            continue

        if len(y) == 0:
            errors.append({"clip_id": clip_id, "file_path": str(audio_path), "error": "empty_audio"})
            continue

        # For short files, still create one padded segment.
        if len(y) <= segment_len:
            starts = [0]
        else:
            starts = list(range(0, max(1, len(y) - segment_len + 1), hop_len))

            # Ensure final tail is represented if requested.
            final_start = max(0, len(y) - segment_len)
            if args.include_tail and final_start not in starts:
                starts.append(final_start)

        split = str(row.get("split", "")).strip()

        if split == "" or split.lower() == "nan":
            split = stable_split(
                key=str(row.get("clip_id", audio_path.stem)),
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )

        for seg_idx, start in enumerate(starts):
            end = start + segment_len
            segment = crop_or_pad(y[start:end], segment_len)

            seg_id = f"{clip_id}_seg{seg_idx:04d}"
            seg_rel = Path(split) / f"{seg_id}.wav"
            seg_path = segment_root / seg_rel
            seg_path.parent.mkdir(parents=True, exist_ok=True)

            sf.write(seg_path, segment, args.sample_rate)

            out_row = {
                "sample_id": seg_id,
                "parent_clip_id": clip_id,
                "abs_path": str(seg_path.resolve()),
                "segment_wav_relpath": str(seg_rel),
                "split": split,
                "start_sec": round(start / args.sample_rate, 4),
                "end_sec": round(min(end, len(y)) / args.sample_rate, 4),
                "segment_sec": args.segment_sec,
                "hop_sec": args.hop_sec,
                "labels": "|".join(active),
                "num_active_labels": len(active),
                "primary_label": row.get("primary_label", ""),
                "source_file": row.get("file_name", audio_path.name),
                "source_path": str(audio_path),
                "source_rel_path": row.get("rel_path", ""),
                "is_clean_seed": 1,
                "is_synthetic": 0,
                "labeling_level": "weak_segment_from_clip",
            }

            for lab in LABELS:
                out_row[lab] = int(row.get(lab, 0))

            out_rows.append(out_row)

        if args.progress_every and (idx + 1) % args.progress_every == 0:
            print(f"[segment_manifest] processed clips: {idx + 1}/{len(df)} | segments: {len(out_rows)}")

    out_manifest = metadata_dir / "tata_segment_manifest.csv"
    labels_json = metadata_dir / "tata_labels.json"
    summary_json = metadata_dir / "tata_segment_manifest_summary.json"
    errors_csv = metadata_dir / "tata_segment_manifest_errors.csv"

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_manifest, index=False)

    labels_payload = {
        "task": "tiny_audio_triage",
        "labeling_level": "weak_segment_from_clip",
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "labels": LABELS,
    }

    labels_json.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    if errors:
        pd.DataFrame(errors).to_csv(errors_csv, index=False)
    else:
        pd.DataFrame(columns=["clip_id", "file_path", "error"]).to_csv(errors_csv, index=False)

    summary = {
        "input_clip_manifest": str(clip_manifest),
        "output_manifest": str(out_manifest),
        "labels_json": str(labels_json),
        "errors_csv": str(errors_csv),
        "input_rows_after_filter": int(len(df)),
        "segments_created": int(len(out_df)),
        "errors": int(len(errors)),
        "split_counts": out_df["split"].value_counts().to_dict() if len(out_df) else {},
        "label_counts": {lab: int(out_df[lab].sum()) for lab in LABELS} if len(out_df) else {},
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("TATA segment manifest complete")
    print("-" * 90)
    print(f"Output manifest: {out_manifest}")
    print(f"Labels JSON:     {labels_json}")
    print(f"Summary JSON:    {summary_json}")
    print(f"Errors CSV:      {errors_csv}")
    print(f"Segments:        {len(out_df)}")
    print(f"Errors:          {len(errors)}")

    if len(out_df):
        print("")
        print("Split counts:")
        print(out_df["split"].value_counts().to_string())

        print("")
        print("Label-positive segment counts:")
        for lab in LABELS:
            print(f"  {lab:24s}: {int(out_df[lab].sum())}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build weak 1-second TATA segment manifest from clip-level manifest."
    )

    parser.add_argument("--clip_manifest", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--include_tail", action="store_true")

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--progress_every", type=int, default=100)

    args = parser.parse_args()
    build_segments(args)


if __name__ == "__main__":
    main()
