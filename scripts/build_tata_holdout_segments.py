# scripts\build_tata_holdout_segments.py

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

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
    "audience_reaction_present",
    "silence_present",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_bin(v) -> int:
    try:
        if pd.isna(v):
            return 0
        return 1 if int(float(v)) == 1 else 0
    except Exception:
        return 0


def active_text(row) -> str:
    return "|".join([lab for lab in LABELS if int(row[lab]) == 1])


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
    except Exception:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)

    try:
        from scipy.signal import resample_poly
        gcd = math.gcd(sr, target_sr)
        return resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
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


def crop_or_pad(segment: np.ndarray, target_len: int) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32)
    if len(segment) >= target_len:
        return segment[:target_len].astype(np.float32)

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(segment)] = segment
    return out


def main():
    parser = argparse.ArgumentParser(description="Build labelled 1-sec final-holdout segment manifest.")
    parser.add_argument("--holdout_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=1.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--include_tail", action="store_true")
    args = parser.parse_args()

    holdout_csv = Path(args.holdout_csv)
    out_dir = Path(args.out_dir)
    wav_root = out_dir / "segment_wavs"
    meta_dir = out_dir / "metadata"

    wav_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(holdout_csv)

    required = {"parent_clip_id", "source_path", "source_file"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    for lab in LABELS:
        if lab not in df.columns:
            raise RuntimeError(f"Missing label column: {lab}")
        df[lab] = df[lab].apply(to_bin)

    df["labels"] = df.apply(active_text, axis=1)
    df["num_active_labels"] = df[LABELS].sum(axis=1).astype(int)

    zero = df[df["num_active_labels"] == 0].copy()
    if len(zero):
        zero.to_csv(meta_dir / "zero_active_holdout_rows.csv", index=False)
        raise RuntimeError(f"Found {len(zero)} zero-active holdout rows. Fix before evaluation.")

    segment_len = int(round(args.sample_rate * args.segment_sec))
    hop_len = int(round(args.sample_rate * args.hop_sec))

    rows = []
    errors = []

    for idx, row in df.reset_index(drop=True).iterrows():
        parent_id = str(row["parent_clip_id"])
        audio_path = Path(str(row["source_path"]))

        if not audio_path.exists():
            errors.append({"parent_clip_id": parent_id, "source_path": str(audio_path), "error": "missing_file"})
            continue

        try:
            y, sr = read_audio_any(audio_path)
            y = resample_audio(y, sr, args.sample_rate)
            y = normalise_audio(y)
        except Exception as e:
            errors.append({"parent_clip_id": parent_id, "source_path": str(audio_path), "error": str(e)})
            continue

        if len(y) <= segment_len:
            starts = [0]
        else:
            starts = list(range(0, max(1, len(y) - segment_len + 1), hop_len))
            final_start = max(0, len(y) - segment_len)
            if args.include_tail and final_start not in starts:
                starts.append(final_start)

        for seg_idx, start in enumerate(starts):
            end = start + segment_len
            segment = crop_or_pad(y[start:end], segment_len)

            sample_id = f"{parent_id}_seg{seg_idx:04d}"
            rel = Path("holdout") / f"{sample_id}.wav"
            out_wav = wav_root / rel
            out_wav.parent.mkdir(parents=True, exist_ok=True)

            sf.write(out_wav, segment, args.sample_rate)

            out_row = {
                "sample_id": sample_id,
                "parent_clip_id": parent_id,
                "abs_path": str(out_wav.resolve()),
                "segment_wav_relpath": str(rel),
                "split": "test",
                "start_sec": round(start / args.sample_rate, 4),
                "end_sec": round(min(end, len(y)) / args.sample_rate, 4),
                "segment_sec": args.segment_sec,
                "hop_sec": args.hop_sec,
                "source_file": row.get("source_file", ""),
                "source_path": str(audio_path),
                "source_rel_path": row.get("source_rel_path", ""),
                "source_class_dir": row.get("source_class_dir", ""),
                "primary_label": row.get("primary_label", ""),
                "labels": row["labels"],
                "num_active_labels": int(row["num_active_labels"]),
                "labeling_level": "final_raw_holdout_ground_truth",
                "is_clean_seed": 0,
                "is_synthetic": 0,
            }

            for lab in LABELS:
                out_row[lab] = int(row[lab])

            rows.append(out_row)

    out_manifest = meta_dir / "final_holdout_segment_manifest.csv"
    out_errors = meta_dir / "final_holdout_segment_errors.csv"
    out_labels = meta_dir / "tata_v06_labels.json"
    out_summary = meta_dir / "final_holdout_segment_summary.json"

    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    pd.DataFrame(errors).to_csv(out_errors, index=False)

    out_labels.write_text(json.dumps({
        "task": "final_raw_holdout_ground_truth",
        "activation": "sigmoid",
        "loss": "BCEWithLogitsLoss",
        "labels": LABELS,
    }, indent=2), encoding="utf-8")

    summary = {
        "generated_at": now_iso(),
        "holdout_parent_rows": int(len(df)),
        "segment_rows": int(len(rows)),
        "errors": int(len(errors)),
        "label_counts_parent": {lab: int(df[lab].sum()) for lab in LABELS},
        "label_counts_segment": {lab: int(pd.DataFrame(rows)[lab].sum()) for lab in LABELS} if rows else {},
        "outputs": {
            "segment_manifest": str(out_manifest),
            "errors_csv": str(out_errors),
            "labels_json": str(out_labels),
        },
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Final holdout segment manifest created")
    print("-" * 90)
    print(f"Parent rows:  {len(df)}")
    print(f"Segment rows: {len(rows)}")
    print(f"Errors:       {len(errors)}")
    print(f"Manifest:     {out_manifest}")
    print(f"Errors CSV:   {out_errors}")


if __name__ == "__main__":
    main()