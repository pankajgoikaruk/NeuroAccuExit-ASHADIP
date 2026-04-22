# scripts/prep_segments.py

import os
import re
import hashlib
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import yaml

from data.transforms_audio import bandpass


SUPPORTED_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".au", ".mp3", ".m4a"}


def rms_dbfs(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    return 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)


def detect_num_channels(y: np.ndarray) -> int:
    y = np.asarray(y)
    if y.ndim == 1:
        return 1
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return int(y.shape[0])   # channel-first
    return int(y.shape[1])       # channel-last


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    # librosa.load(..., mono=False) -> (channels, samples)
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return y.mean(axis=0).astype(np.float32)
    # soundfile.read stereo -> (samples, channels)
    return y.mean(axis=1).astype(np.float32)


def safe_read_audio(path, dtype="float32"):
    """
    Try soundfile first, then fall back to librosa/audioread.
    Returns (y, sr) or (None, None) if unreadable.
    """
    try:
        y, sr = sf.read(path, dtype=dtype)
        return y, sr
    except Exception:
        try:
            y, sr = librosa.load(path, sr=None, mono=False)
            y = np.asarray(y, dtype=np.float32)
            return y, sr
        except Exception as e:
            warnings.warn(f"Skipping unreadable file: {path} ({e})")
            return None, None


def load_yaml(path: str):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        warnings.warn(f"--config was provided but file not found: {p}. Ignoring.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def two_stage_parse():
    """
    Parse --config first, then load YAML defaults, then allow CLI overrides.
    """
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default="", help="Optional YAML config (e.g., configs/audio_moth.yaml).")
    cfg0, rest = p0.parse_known_args()

    cfg = load_yaml(cfg0.config)
    seed_default = int(cfg.get("seed", 42))

    split_cfg = cfg.get("split", {}) or {}
    train_default = float(split_cfg.get("train", 0.7))
    val_default = float(split_cfg.get("val", 0.15))
    test_default = float(split_cfg.get("test", 0.15))
    strat_default = bool(split_cfg.get("stratify", True))

    audio_cfg = cfg.get("audio", {}) or {}
    sr_default = int(audio_cfg.get("sample_rate", 16000))
    seg_default = float(audio_cfg.get("segment_sec", 1.0))
    hop_default = float(audio_cfg.get("segment_hop", 0.5))
    silence_default = float(audio_cfg.get("silence_dbfs", -40))
    bp_default = audio_cfg.get("bandpass", [100, 3000])
    if not (isinstance(bp_default, (list, tuple)) and len(bp_default) == 2):
        bp_default = [100, 3000]

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=cfg0.config)

    p.add_argument("--root", default="data")
    p.add_argument("--cache", default="data_cache")

    p.add_argument("--sr", type=int, default=sr_default)
    p.add_argument("--segment_sec", type=float, default=seg_default)
    p.add_argument("--hop", type=float, default=hop_default)
    p.add_argument("--silence_dbfs", type=float, default=silence_default)
    p.add_argument("--bandpass", nargs=2, type=float, default=bp_default)

    p.add_argument("--seed", type=int, default=seed_default)
    p.add_argument("--train_frac", type=float, default=train_default)
    p.add_argument("--val_frac", type=float, default=val_default)
    p.add_argument("--test_frac", type=float, default=test_default)

    p.add_argument("--labels", nargs="*", default=None,
                   help="Optional explicit label folder names. Default: auto-discover subfolders under --root.")
    p.add_argument("--min_keep_sec", type=float, default=0.25,
                   help="If a file is shorter than segment_sec but >= min_keep_sec, keep one padded segment.")
    p.add_argument("--inspect_only", action="store_true",
                   help="Only scan and summarize audio. Do not write cleaned audio or segments.")
    p.add_argument("--write_compat_manifest", action="store_true", default=True,
                   help="Also write moths_manifest.csv for compatibility with older scripts.")
    p.add_argument("--max_segments_per_file_gunshot", type=int, default=0,
                   help="Max kept segments per gunshot file. 0 = keep all.")
    p.add_argument("--max_segments_per_file_non_gunshot", type=int, default=5,
                   help="Max kept segments per non_gunshot file. 0 = keep all.")

    if strat_default:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=True)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")
    else:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=False)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")

    args = p.parse_args(rest)
    args._cfg = cfg
    return args


def list_label_dirs(root: Path, explicit_labels=None):
    if explicit_labels:
        labels = [str(x) for x in explicit_labels]
        missing = [lab for lab in labels if not (root / lab).is_dir()]
        if missing:
            raise SystemExit(f"These label folders were not found under {root}: {missing}")
        return labels

    labels = sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not labels:
        raise SystemExit(f"No class folders found under {root}")
    return labels


def iter_audio_files(label_dir: Path):
    files = []
    for p in label_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return sorted(files)


def slugify_relpath(relpath: str) -> str:
    stem = Path(relpath).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    h = hashlib.md5(relpath.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{h}.wav"


def print_inventory_summary(inv_df: pd.DataFrame):
    print("\n=== Audio inventory summary ===")
    print(f"Files found: {len(inv_df)}")
    print("Labels:", sorted(inv_df["label"].unique().tolist()))

    by_label = (
        inv_df.groupby("label")
        .agg(
            files=("label", "size"),
            min_sec=("duration_sec", "min"),
            median_sec=("duration_sec", "median"),
            mean_sec=("duration_sec", "mean"),
            max_sec=("duration_sec", "max"),
            unreadable=("read_ok", lambda s: int((~s).sum()))
        )
        .reset_index()
    )
    print(by_label.to_string(index=False))

    short_df = inv_df[(inv_df["read_ok"]) & (inv_df["duration_sec"] < 1.0)]
    if len(short_df) > 0:
        print("\nShorter than 1.0 sec:")
        print(
            short_df.groupby("label")
            .size()
            .rename("count")
            .reset_index()
            .to_string(index=False)
        )


def file_level_split(seg_df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float,
                     seed: int, stratify: bool):
    from sklearn.model_selection import train_test_split

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0; got {total:.6f} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("Split ratios must be > 0 for train/val/test.")

    file_df = seg_df[["wav_relpath", "label"]].drop_duplicates().reset_index(drop=True)

    nunq = file_df.groupby("wav_relpath")["label"].nunique()
    if nunq.max() != 1:
        bad = nunq[nunq > 1].index.tolist()[:10]
        raise ValueError(f"Some wav_relpath map to multiple labels (examples): {bad}")

    files = file_df["wav_relpath"].values
    labels = file_df["label"].values

    temp_frac = val_frac + test_frac
    try:
        if stratify:
            files_train, files_temp, y_train, y_temp = train_test_split(
                files, labels, test_size=temp_frac, stratify=labels, random_state=seed
            )
        else:
            files_train, files_temp, y_train, y_temp = train_test_split(
                files, labels, test_size=temp_frac, random_state=seed, shuffle=True
            )

        test_within_temp = test_frac / temp_frac
        if stratify:
            files_val, files_test = train_test_split(
                files_temp, test_size=test_within_temp, stratify=y_temp, random_state=seed
            )
        else:
            files_val, files_test = train_test_split(
                files_temp, test_size=test_within_temp, random_state=seed, shuffle=True
            )

    except ValueError as e:
        warnings.warn(f"Stratified file-level split failed ({e}). Falling back to non-stratified split.")
        files_train, files_temp = train_test_split(
            files, test_size=temp_frac, random_state=seed, shuffle=True
        )
        files_val, files_test = train_test_split(
            files_temp, test_size=(test_frac / temp_frac), random_state=seed, shuffle=True
        )

    split_map = {f: "train" for f in files_train}
    split_map.update({f: "val" for f in files_val})
    split_map.update({f: "test" for f in files_test})

    seg_df = seg_df.copy()
    seg_df["split"] = seg_df["wav_relpath"].map(split_map)

    if seg_df["split"].isna().any():
        missing = seg_df.loc[seg_df["split"].isna(), "wav_relpath"].unique()[:10]
        raise SystemExit(f"Some wav_relpath not assigned a split (examples): {missing}")

    seg_counts = seg_df["split"].value_counts().to_dict()
    file_counts = file_df.assign(split=file_df["wav_relpath"].map(split_map))["split"].value_counts().to_dict()
    return seg_df, seg_counts, file_counts


def select_evenly_spaced_starts(all_starts, max_keep: int):
    """
    Keep up to max_keep starts, evenly spaced across the file.
    If max_keep <= 0 or enough room already, keep all.
    """
    if max_keep <= 0 or len(all_starts) <= max_keep:
        return all_starts

    idx = np.linspace(0, len(all_starts) - 1, num=max_keep, dtype=int)
    idx = np.unique(idx)  # just in case
    return [all_starts[i] for i in idx]


def main():
    args = two_stage_parse()

    root = Path(args.root)
    cache = Path(args.cache)
    clean_root = cache / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    labels = list_label_dirs(root, args.labels)

    # ------------------------------------------------------------
    # Stage 1: inventory / summary
    # ------------------------------------------------------------
    inv_rows = []
    for label in labels:
        for src in iter_audio_files(root / label):
            rel = os.path.relpath(src, root)
            y, sr = safe_read_audio(src, dtype="float32")
            if y is None:
                inv_rows.append({
                    "label": label,
                    "orig_filepath": str(src),
                    "orig_relpath": rel,
                    "orig_ext": src.suffix.lower(),
                    "read_ok": False,
                    "orig_sr": np.nan,
                    "channels": np.nan,
                    "num_samples": np.nan,
                    "duration_sec": np.nan,
                })
                continue

            channels = detect_num_channels(y)
            y_mono = to_mono(y)
            duration_sec = float(len(y_mono) / sr)

            inv_rows.append({
                "label": label,
                "orig_filepath": str(src),
                "orig_relpath": rel,
                "orig_ext": src.suffix.lower(),
                "read_ok": True,
                "orig_sr": int(sr),
                "channels": int(channels),
                "num_samples": int(len(y_mono)),
                "duration_sec": duration_sec,
            })

    inv_df = pd.DataFrame(inv_rows)
    if len(inv_df) == 0:
        raise SystemExit(f"No audio files found under {root}")

    inv_df.to_csv(cache / "audio_inventory.csv", index=False)
    print_inventory_summary(inv_df)

    summary_by_label = (
        inv_df[inv_df["read_ok"]]
        .groupby("label")
        .agg(
            files=("label", "size"),
            min_sec=("duration_sec", "min"),
            median_sec=("duration_sec", "median"),
            mean_sec=("duration_sec", "mean"),
            max_sec=("duration_sec", "max"),
        )
        .reset_index()
    )
    summary_by_label.to_csv(cache / "audio_inventory_by_label.csv", index=False)

    if args.inspect_only:
        print("\ninspect_only=True -> wrote inventory CSVs only. Stopping before cleaning/segmentation.")
        return

    # ------------------------------------------------------------
    # Stage 2: clean/resample/normalize and write cache WAVs
    # ------------------------------------------------------------
    rows = []
    skipped = []

    for _, r in inv_df.iterrows():
        if not bool(r["read_ok"]):
            skipped.append(str(r["orig_filepath"]))
            continue

        src = Path(r["orig_filepath"])
        label = str(r["label"])
        orig_relpath = str(r["orig_relpath"])

        y, sr = safe_read_audio(src, dtype="float32")
        if y is None:
            skipped.append(str(src))
            continue

        y = to_mono(y)

        if sr != args.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
            sr = args.sr

        y = y.astype(np.float32)
        if y.size > 0:
            y = y - float(np.mean(y))

        if args.bandpass:
            y = bandpass(y, sr, float(args.bandpass[0]), float(args.bandpass[1]))

        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 0:
            y = 0.8913 * y / peak

        file_hash = hashlib.md5(orig_relpath.encode("utf-8")).hexdigest()[:12]
        out_name = f"{label}_{file_hash}.wav"
        out = clean_root / label / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, y, sr)

        clean_rel = os.path.relpath(out, clean_root)
        dur = float(len(y) / sr)

        rows.append({
            "filepath": str(out),
            "clean_relpath": clean_rel,
            "label": label,
            "duration": dur,
            "orig_filepath": str(src),
            "orig_relpath": orig_relpath,
        })

    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable files (showing up to 10):")
        for s in skipped[:10]:
            print(" -", s)
        if len(skipped) > 10:
            print(" ... (more skipped)")

    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        raise SystemExit("No valid audio files were cleaned successfully.")

    manifest.to_csv(cache / "manifest.csv", index=False)
    if args.write_compat_manifest:
        manifest.to_csv(cache / "moths_manifest.csv", index=False)

    # ------------------------------------------------------------
    # Stage 3: segmentation
    # ------------------------------------------------------------
    seg_rows = []
    dropped_too_short = []

    for _, r in manifest.iterrows():
        y, sr = sf.read(r["filepath"], dtype="float32")
        y = np.asarray(y, dtype=np.float32)

        win = int(args.segment_sec * sr)
        hop = int(args.hop * sr)
        n = int(len(y))
        dur_sec = float(n / sr)

        rel = r["clean_relpath"]
        label = r["label"]

        max_keep = (
            int(args.max_segments_per_file_non_gunshot)
            if label == "non_gunshot"
            else int(args.max_segments_per_file_gunshot)
        )

        # short file: keep one padded segment if long enough
        if n < win:
            if dur_sec < float(args.min_keep_sec):
                dropped_too_short.append((rel, dur_sec))
                continue

            if rms_dbfs(y) < args.silence_dbfs:
                continue

            seg_rows.append({
                "wav_relpath": rel,
                "label": label,
                "start": 0.0,
                "duration": float(args.segment_sec),
            })
            continue

        # all candidate starts
        all_starts = list(range(0, max(n - win + 1, 0), hop))

        # keep only non-silent candidates
        valid_starts = []
        for s in all_starts:
            seg = y[s:s + win]
            if rms_dbfs(seg) < args.silence_dbfs:
                continue
            valid_starts.append(s)

        # apply per-file cap
        kept_starts = select_evenly_spaced_starts(valid_starts, max_keep)

        for s in kept_starts:
            seg_rows.append({
                "wav_relpath": rel,
                "label": label,
                "start": float(s / sr),
                "duration": float(args.segment_sec),
            })

    if dropped_too_short:
        print(
            f"\nDropped {len(dropped_too_short)} files shorter than "
            f"min_keep_sec={args.min_keep_sec} sec (showing up to 10):"
        )
        for rel, dur in dropped_too_short[:10]:
            print(f" - {rel} ({dur:.3f}s)")
        if len(dropped_too_short) > 10:
            print(" ... (more dropped)")

    seg_df = pd.DataFrame(seg_rows)
    if len(seg_df) == 0:
        raise SystemExit("No segments above silence threshold; try raising --silence_dbfs (e.g., -55).")

    seg_df, seg_counts, file_counts = file_level_split(
        seg_df,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        stratify=bool(args.stratify),
    )

    seg_df.to_csv(cache / "segments.csv", index=False)

    print("\n=== Segmentation summary ===")
    print("Files:", file_counts)
    print("Segments:", seg_counts)

    print("\nSegments by label:")
    print(
        seg_df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False)
    )


if __name__ == "__main__":
    main()