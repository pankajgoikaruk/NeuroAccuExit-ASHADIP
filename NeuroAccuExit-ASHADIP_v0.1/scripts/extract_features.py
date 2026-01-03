# scripts/extract_features.py

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

from data.transforms_audio import to_logmel, cmvn_feat


def _pad_to_length(x: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros if shorter than required length."""
    if x.shape[0] >= length:
        return x[:length]
    pad = length - x.shape[0]
    return np.pad(x, (0, pad), mode="constant")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--win_ms", type=int, default=25)
    ap.add_argument("--hop_ms", type=int, default=10)
    ap.add_argument("--cmvn", action="store_true")
    ap.add_argument(
        "--pad_short",
        action="store_true",
        help="If a clip is shorter than expected, pad zeros (recommended).",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=500,
        help="Print progress every N segments (0 disables).",
    )
    args = ap.parse_args()

    cache = Path(args.cache)
    seg_path = cache / "segments.csv"
    if not seg_path.exists():
        raise SystemExit(f"segments.csv not found: {seg_path}")

    seg = pd.read_csv(seg_path)

    feat_root = cache / "features"
    feat_root.mkdir(parents=True, exist_ok=True)

    # ---- Path safety: normalize to POSIX separators in the dataframe ----
    if "wav_relpath" not in seg.columns:
        raise SystemExit("segments.csv missing required column: wav_relpath")

    seg2 = seg.copy()
    seg2["wav_relpath"] = (
        seg2["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
    )

    # ---- Speed: sort by wav_relpath so we read each wav only once ----
    # (Keeps logic simple and avoids unbounded caching)
    seg2 = seg2.sort_values(["wav_relpath", "start"]).reset_index(drop=True)

    feats = []

    current_rel = None
    current_y = None
    current_sr = None

    total = len(seg2)

    for i, row in enumerate(seg2.itertuples(index=False), start=1):
        rel = str(getattr(row, "wav_relpath"))
        wav_path = cache / "clean" / rel

        if args.progress_every and (i % args.progress_every == 0):
            print(f"[extract_features] processed {i}/{total} segments...")

        # Read audio only when wav changes
        if rel != current_rel:
            y, sr = sf.read(wav_path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            current_rel, current_y, current_sr = rel, y, sr

        start_s = float(getattr(row, "start"))
        dur_s = float(getattr(row, "duration"))

        # Convert to *sample indices* (more robust than encoding floats)
        start_i = int(round(start_s * current_sr))
        dur_i = int(round(dur_s * current_sr))

        clip = current_y[start_i : start_i + dur_i]

        # Handle rare short slice (rounding / end-of-file)
        if clip.shape[0] != dur_i:
            if args.pad_short:
                clip = _pad_to_length(clip, dur_i)
            else:
                feats.append("")  # keep alignment
                continue

        S = to_logmel(
            clip, current_sr, args.n_mels, args.n_fft, args.win_ms, args.hop_ms
        )
        if args.cmvn:
            S = cmvn_feat(S)

        # ---- Unique feat_relpath per segment (robust + no overwrite) ----
        # Keep class folder (male/ or female/) from wav_relpath
        p = Path(rel)  # rel is POSIX-style now
        stem = p.stem
        # encode *sample* start/dur => avoids float issues
        fname = f"{stem}_si{start_i:09d}_di{dur_i:09d}.npy"
        out_rel = str(p.parent / fname).replace("\\", "/")
        # ----------------------------------------------------------------

        out_path = feat_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, S)

        feats.append(out_rel)

    # Attach feat_relpath back
    seg2["feat_relpath"] = feats

    # Restore original row order so downstream behaviour stays stable
    # (We align by original content: add a temporary index before sort)
    # If segments.csv order doesn't matter for you, you can skip this.
    # Here we keep it stable by merging on key columns.
    key_cols = ["wav_relpath", "label", "start", "duration", "split"]
    for c in key_cols:
        if c not in seg.columns:
            raise SystemExit(f"segments.csv missing required column: {c}")

    # Normalize the original seg wav_relpath too for a consistent merge
    seg_norm = seg.copy()
    seg_norm["wav_relpath"] = (
        seg_norm["wav_relpath"].astype(str).str.replace("\\", "/", regex=False)
    )

    # Merge feat_relpath back onto the original seg ordering
    seg_out = seg_norm.merge(
        seg2[key_cols + ["feat_relpath"]],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    if seg_out["feat_relpath"].isna().any():
        missing = seg_out.loc[seg_out["feat_relpath"].isna(), "wav_relpath"].head(5).tolist()
        raise SystemExit(
            f"Failed to map feat_relpath back to segments.csv rows (examples wav_relpath): {missing}"
        )

    # Sanity: should be unique now (ignoring skipped empty paths)
    nonempty = seg_out["feat_relpath"].astype(str)
    nonempty = nonempty[nonempty.str.len() > 0]
    if nonempty.duplicated().any():
        dups = nonempty[nonempty.duplicated()].head(5).tolist()
        raise SystemExit(f"feat_relpath still has duplicates (examples): {dups}")

    seg_out.to_csv(seg_path, index=False)
    print("Saved features to", feat_root)
    print("Updated segments.csv feat_relpath with unique per-segment filenames.")
