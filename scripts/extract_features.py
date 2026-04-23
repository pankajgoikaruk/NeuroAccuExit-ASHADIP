# scripts/extract_features.py

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from data.transforms_audio import to_logmel, cmvn_feat


def _pad_to_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] >= length:
        return x[:length]
    pad = length - x.shape[0]
    return np.pad(x, (0, pad), mode="constant")


def _to_mono_if_needed(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.shape[0] >= y.shape[1]:
        return y.mean(axis=1).astype(np.float32)
    return y.mean(axis=0).astype(np.float32)


def _make_short_feature_name(rel: str, segment_uid: str, start_i: int, dur_i: int) -> str:
    key = f"{rel}|{segment_uid}|{start_i}|{dur_i}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return f"seg_{h}.npy"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--win_ms", type=int, default=25)
    ap.add_argument("--hop_ms", type=int, default=10)
    ap.add_argument("--cmvn", action="store_true")
    ap.add_argument("--pad_short", action="store_true")
    ap.add_argument("--progress_every", type=int, default=500)
    args = ap.parse_args()

    cache = Path(args.cache)
    seg_path = cache / "segments.csv"
    if not seg_path.exists():
        raise SystemExit(f"segments.csv not found: {seg_path}")

    seg = pd.read_csv(seg_path)

    required_cols = ["wav_relpath", "label", "start", "duration", "split"]
    missing_cols = [c for c in required_cols if c not in seg.columns]
    if missing_cols:
        raise SystemExit(f"segments.csv missing required column(s): {missing_cols}")

    feat_root = cache / "features"
    feat_root.mkdir(parents=True, exist_ok=True)

    seg_work = seg.copy()
    if "segment_id" in seg_work.columns:
        seg_work["segment_uid"] = seg_work["segment_id"].astype(str)
        if seg_work["segment_uid"].duplicated().any():
            raise SystemExit("segments.csv has duplicate segment_id values.")
    else:
        seg_work["segment_uid"] = [f"row_{i:09d}" for i in range(len(seg_work))]

    seg_work["wav_relpath"] = (
        seg_work["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
    )

    seg_sorted = seg_work.sort_values(
        ["wav_relpath", "start", "duration", "segment_uid"]
    ).reset_index(drop=True)

    feat_rows = []
    current_rel = None
    current_y = None
    current_sr = None
    total = len(seg_sorted)

    for i, row in enumerate(seg_sorted.itertuples(index=False), start=1):
        rel = str(row.wav_relpath)
        wav_path = cache / "clean" / rel

        if args.progress_every and (i % args.progress_every == 0):
            print(f"[extract_features] processed {i}/{total} segments...")

        if rel != current_rel:
            y, sr = sf.read(wav_path, dtype="float32")
            y = _to_mono_if_needed(y)
            current_rel, current_y, current_sr = rel, y, sr

        start_s = float(row.start)
        dur_s = float(row.duration)
        start_i = int(round(start_s * current_sr))
        dur_i = int(round(dur_s * current_sr))

        clip = current_y[start_i:start_i + dur_i]

        if clip.shape[0] != dur_i:
            if args.pad_short:
                clip = _pad_to_length(clip, dur_i)
            else:
                feat_rows.append({
                    "segment_uid": row.segment_uid,
                    "feat_relpath": "",
                })
                continue

        S = to_logmel(
            clip, current_sr, args.n_mels, args.n_fft, args.win_ms, args.hop_ms
        )
        if args.cmvn:
            S = cmvn_feat(S)

        label_dir = Path(rel).parent.as_posix()
        fname = _make_short_feature_name(rel, str(row.segment_uid), start_i, dur_i)
        out_rel = f"{label_dir}/{fname}" if label_dir not in ("", ".") else fname
        out_path = feat_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "wb") as f:
            np.save(f, S)

        feat_rows.append({
            "segment_uid": row.segment_uid,
            "feat_relpath": out_rel.replace("\\", "/"),
        })

    feat_df = pd.DataFrame(feat_rows)
    if feat_df["segment_uid"].duplicated().any():
        dups = feat_df.loc[feat_df["segment_uid"].duplicated(), "segment_uid"].head(5).tolist()
        raise SystemExit(f"Duplicate segment_uid values in features output: {dups}")

    seg_out = seg_work.merge(feat_df, on="segment_uid", how="left", validate="one_to_one")

    if seg_out["feat_relpath"].isna().any():
        missing = seg_out.loc[seg_out["feat_relpath"].isna(), "wav_relpath"].head(5).tolist()
        raise SystemExit(f"Failed to map feat_relpath back to rows (examples): {missing}")

    nonempty = seg_out["feat_relpath"].astype(str)
    nonempty = nonempty[nonempty.str.len() > 0]
    if nonempty.duplicated().any():
        dups = nonempty[nonempty.duplicated()].head(5).tolist()
        raise SystemExit(f"feat_relpath has duplicates (examples): {dups}")

    seg_out = seg_out.drop(columns=["segment_uid"])
    seg_out.to_csv(seg_path, index=False)
    print("Saved features to", feat_root)
    print("Updated segments.csv feat_relpath with short unique filenames.")