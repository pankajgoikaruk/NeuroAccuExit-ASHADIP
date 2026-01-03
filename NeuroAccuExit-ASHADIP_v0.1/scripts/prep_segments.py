# scripts/prep_segments.py

import os, argparse, warnings
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import yaml

from data.transforms_audio import bandpass


def rms_dbfs(y):
    if y.size == 0:
        return -120.0
    return 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)


def safe_read_audio(path, dtype="float32"):
    """Read audio if it's a valid PCM/WAV. Returns (y, sr) or (None, None) if unreadable."""
    try:
        y, sr = sf.read(path, dtype=dtype)
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
    Parse --config first, then load YAML and use it to set defaults,
    while still allowing CLI args to override.
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

    # Full parser (YAML-backed defaults)
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=cfg0.config)

    p.add_argument("--root", default="data")
    p.add_argument("--cache", default="data_cache")

    p.add_argument("--sr", type=int, default=sr_default)
    p.add_argument("--segment_sec", type=float, default=seg_default)
    p.add_argument("--hop", type=float, default=hop_default)
    p.add_argument("--silence_dbfs", type=float, default=silence_default)
    p.add_argument("--bandpass", nargs=2, type=float, default=bp_default)

    # Split/seed (YAML defaults, CLI override)
    p.add_argument("--seed", type=int, default=seed_default)
    p.add_argument("--train_frac", type=float, default=train_default)
    p.add_argument("--val_frac", type=float, default=val_default)
    p.add_argument("--test_frac", type=float, default=test_default)

    # stratify flag with YAML default
    if strat_default:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=True)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")
    else:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=False)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")

    args = p.parse_args(rest)
    args._cfg = cfg  # keep if you want to debug/extend
    return args


def file_level_split(seg_df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float,
                     seed: int, stratify: bool):
    """
    Split by wav_relpath (file-level), then map split to all segments.
    Prevents leakage: a wav file can never appear in multiple splits.
    """
    from sklearn.model_selection import train_test_split

    # Validate ratios
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0; got {total:.6f} "
                         f"(train={train_frac}, val={val_frac}, test={test_frac})")
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("Split ratios must be > 0 for train/val/test.")

    # one row per file
    file_df = seg_df[["wav_relpath", "label"]].drop_duplicates().reset_index(drop=True)

    # Sanity: each wav has exactly one label
    nunq = file_df.groupby("wav_relpath")["label"].nunique()
    if nunq.max() != 1:
        bad = nunq[nunq > 1].index.tolist()[:10]
        raise ValueError(f"Some wav_relpath map to multiple labels (examples): {bad}")

    files = file_df["wav_relpath"].values
    labels = file_df["label"].values

    temp_frac = val_frac + test_frac
    # split train vs temp
    try:
        if stratify:
            files_train, files_temp, y_train, y_temp = train_test_split(
                files, labels, test_size=temp_frac, stratify=labels, random_state=seed
            )
        else:
            files_train, files_temp, y_train, y_temp = train_test_split(
                files, labels, test_size=temp_frac, random_state=seed, shuffle=True
            )

        # split temp -> val/test
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
        files_train, files_temp = train_test_split(files, test_size=temp_frac, random_state=seed, shuffle=True)
        files_val, files_test = train_test_split(files_temp, test_size=(test_frac / temp_frac),
                                                 random_state=seed, shuffle=True)

    split_map = {f: "train" for f in files_train}
    split_map.update({f: "val" for f in files_val})
    split_map.update({f: "test" for f in files_test})

    seg_df = seg_df.copy()
    seg_df["split"] = seg_df["wav_relpath"].map(split_map)

    if seg_df["split"].isna().any():
        missing = seg_df.loc[seg_df["split"].isna(), "wav_relpath"].unique()[:10]
        raise SystemExit(f"Some wav_relpath not assigned a split (examples): {missing}")

    # Debug prints
    seg_counts = seg_df["split"].value_counts().to_dict()
    file_counts = file_df.assign(split=file_df["wav_relpath"].map(split_map))["split"].value_counts().to_dict()
    return seg_df, seg_counts, file_counts


def main():
    args = two_stage_parse()

    root = Path(args.root)
    cache = Path(args.cache)
    (cache / "clean").mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for label in ["male", "female"]:
        files = [p for p in (root / label).rglob("*.wav") if not p.name.startswith("._")]
        for wav in sorted(files):
            y, sr = safe_read_audio(wav, dtype="float32")
            if y is None:
                skipped.append(str(wav))
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != args.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
                sr = args.sr

            y = y - y.mean()
            if args.bandpass:
                y = bandpass(y, sr, float(args.bandpass[0]), float(args.bandpass[1]))

            peak = float(np.max(np.abs(y)) + 1e-9)
            y = 0.8913 * y / peak

            out = cache / "clean" / label / wav.name
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, y, sr)
            dur = len(y) / sr
            rows.append({"filepath": str(out), "label": label, "duration": dur})

    if skipped:
        print(f"Skipped {len(skipped)} unreadable files (showing up to 10):")
        for s in skipped[:10]:
            print(" -", s)
        if len(skipped) > 10:
            print(" ... (more skipped)")

    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        raise SystemExit("No valid WAVs found. Check your paths or remove non-audio files (e.g., ._*.wav).")
    manifest.to_csv(cache / "moths_manifest.csv", index=False)

    # segmentation
    seg_rows = []
    for _, r in manifest.iterrows():
        y, sr = sf.read(r["filepath"], dtype="float32")
        win = int(args.segment_sec * sr)
        hop = int(args.hop * sr)
        for s in range(0, max(len(y) - win + 1, 0), hop):
            seg = y[s:s + win]
            if rms_dbfs(seg) < args.silence_dbfs:
                continue
            rel = os.path.relpath(r["filepath"], cache / "clean")
            seg_rows.append({
                "wav_relpath": rel,
                "label": r["label"],
                "start": s / sr,
                "duration": float(args.segment_sec),
            })

    seg_df = pd.DataFrame(seg_rows)
    if len(seg_df) == 0:
        raise SystemExit("No segments above silence threshold; try raising --silence_dbfs (e.g., -55).")

    # split (FILE-LEVEL)
    seg_df, seg_counts, file_counts = file_level_split(
        seg_df,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        stratify=bool(args.stratify),
    )

    seg_df.to_csv(cache / "segments.csv", index=False)
    print("Segments:", seg_counts)
    print("Files:", file_counts)


if __name__ == "__main__":
    main()
