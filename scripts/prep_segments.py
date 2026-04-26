# scripts/prep_segments.py

"""
Unified ASHADIP audio preprocessing and segmentation.

Design goals
------------
1. Auto-discover any class folders under --root.
2. Support mixed-length raw audio through input_mode=segment.
3. Support already clipped datasets through input_mode=ready.
4. Always keep parent-file traceability in segments.csv.
5. Export physical fixed-length segment WAVs for reuse and inspection.
6. Keep downstream training compatible by preserving wav_relpath as the parent clip key.

Important columns in segments.csv
---------------------------------
- wav_relpath: parent cleaned WAV relative to cache/clean; used for clip grouping.
- segment_wav_relpath: exported segment WAV relative to cache; used by extract_features.py.
- parent_start: start time inside the parent cleaned WAV.
- start: kept as 0.0 because features are extracted from the physical segment WAV.
- split_key: file/group key used for leakage-safe splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from data.transforms_audio import bandpass


SUPPORTED_EXTS = {
    ".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".au", ".mp3", ".m4a"
}


def rms_dbfs(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(y, dtype=np.float32) ** 2)) + 1e-9))


def safe_read_audio(path: Path, dtype: str = "float32") -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Read audio with soundfile first; fall back to librosa for compressed formats."""
    try:
        y, sr = sf.read(path, dtype=dtype)
        return y, int(sr)
    except Exception:
        try:
            y, sr = librosa.load(path, sr=None, mono=False)
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception as e:
            warnings.warn(f"Skipping unreadable file: {path} ({e})")
            return None, None


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    # librosa mono=False gives (channels, samples); soundfile gives (samples, channels)
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return y.mean(axis=0).astype(np.float32)
    return y.mean(axis=1).astype(np.float32)


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        warnings.warn(f"--config was provided but file not found: {p}. Ignoring.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_json_map(text: str) -> Dict[str, int]:
    """
    Robust parser for per-label segment caps.

    Accepts:
      - JSON: {"gun_shot":0,"rain":5}
      - PowerShell-escaped JSON: {\"gun_shot\":0,\"rain\":5}
      - CLI-safe syntax: gun_shot=0,rain=5 or gun_shot:0,rain:5
      - @path/to/file.json containing any of the above
    """
    if not text or not str(text).strip():
        return {}

    raw = str(text).strip()

    # File-based caps avoid shell quoting issues entirely.
    if raw.startswith("@"):
        cap_path = Path(raw[1:])
        if not cap_path.exists():
            raise argparse.ArgumentTypeError(
                f"Cap file not found for --max_segments_per_label_json: {cap_path}"
            )
        raw = cap_path.read_text(encoding="utf-8").strip()

    # Remove literal wrapper quotes if a shell left them in the string.
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1].strip()

    candidates = [raw]

    # Windows/PowerShell may pass JSON as {\"label\":5}; convert to normal JSON.
    if '\\"' in raw:
        candidates.append(raw.replace('\\"', '"'))

    # Try generic unicode unescape too, but keep it non-fatal.
    try:
        decoded = bytes(raw, "utf-8").decode("unicode_escape")
        if decoded not in candidates:
            candidates.append(decoded)
    except Exception:
        pass

    last_error = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            if not isinstance(data, dict):
                raise argparse.ArgumentTypeError(
                    "--max_segments_per_label_json must be a JSON object."
                )
            return {str(k): int(v) for k, v in data.items()}
        except Exception as e:
            last_error = e

    # Fallback parser for CLI-safe syntax: gun_shot=0,rain=5 or {gun_shot:0,rain:5}
    simple = raw.strip().strip("{}")
    if simple:
        out: Dict[str, int] = {}
        try:
            for item in simple.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    key, value = item.split("=", 1)
                elif ":" in item:
                    key, value = item.split(":", 1)
                else:
                    raise ValueError(f"missing ':' or '=' in item: {item}")
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")
                if not key:
                    raise ValueError(f"empty label name in item: {item}")
                out[key] = int(value)
            if out:
                return out
        except Exception as e:
            last_error = e

    raise argparse.ArgumentTypeError(
        "Invalid value for --max_segments_per_label_json. "
        "Use JSON like '{\"rain\":5}', CLI-safe syntax like 'rain=5,wind=5', "
        "or a file reference like '@configs/caps.json'. "
        f"Original error: {last_error}"
    )


def two_stage_parse():
    """Parse --config first, then use YAML defaults while allowing CLI overrides."""
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default="", help="Optional YAML config, e.g. configs/audio_moth.yaml")
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

    p.add_argument(
        "--input_mode",
        choices=["segment", "ready"],
        default="segment",
        help="segment: raw mixed-length audio. ready: already fixed-length clips.",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional explicit class folders. Default: auto-discover subfolders under --root.",
    )
    p.add_argument(
        "--min_keep_sec",
        type=float,
        default=0.25,
        help="Short files >= this duration are kept as one padded segment.",
    )
    p.add_argument(
        "--max_segments_per_file_default",
        type=int,
        default=0,
        help="Default max segments per parent file. 0 = keep all.",
    )
    p.add_argument(
        "--max_segments_per_label_json",
        type=str,
        default="",
        help='Optional JSON caps per label, e.g. "{""fireworks"":5,""gunshot"":0}". 0 = keep all.',
    )
    p.add_argument(
        "--split_unit",
        choices=["file", "group"],
        default="file",
        help="file: split by source file. group: split related files by a group id.",
    )
    p.add_argument(
        "--group_regex",
        type=str,
        default="",
        help="Optional regex for split_unit=group. Use first capture group or named group 'group'.",
    )
    p.add_argument(
        "--export_segment_wavs",
        dest="export_segment_wavs",
        action="store_true",
        default=True,
        help="Export physical fixed-length segment WAVs. Default: enabled.",
    )
    p.add_argument(
        "--no_export_segment_wavs",
        dest="export_segment_wavs",
        action="store_false",
        help="Disable segment WAV export. Not recommended for this project.",
    )
    p.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Delete existing clean/features/segment_wavs/CSV artifacts inside the cache before rebuilding.",
    )

    if strat_default:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=True)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")
    else:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=False)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")

    args = p.parse_args(rest)
    args.max_segments_per_label = _parse_json_map(args.max_segments_per_label_json)
    args._cfg = cfg
    return args


def list_label_dirs(root: Path, explicit_labels: Optional[Sequence[str]] = None) -> List[str]:
    if explicit_labels:
        labels = [str(x).strip() for x in explicit_labels if str(x).strip()]
        missing = [lab for lab in labels if not (root / lab).is_dir()]
        if missing:
            raise SystemExit(f"These label folders were not found under {root}: {missing}")
        return labels

    labels = sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not labels:
        raise SystemExit(f"No class folders found under {root}")
    return labels


def iter_audio_files(label_dir: Path) -> Iterable[Path]:
    for p in sorted(label_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("._"):
            continue
        if p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _safe_token(text: str, max_len: int = 32) -> str:
    """Create a short filesystem-safe token for Windows-friendly paths."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    if not token:
        token = "x"
    return token[:max_len]


def stable_wav_name(label: str, orig_relpath: str) -> str:
    """Short deterministic parent WAV name to avoid Windows path-length failures."""
    label_safe = _safe_token(label, max_len=24)
    h = hashlib.md5(orig_relpath.encode("utf-8")).hexdigest()[:16]
    return f"{label_safe}_parent_{h}.wav"

def get_group_key(orig_relpath: str, group_regex: str = "") -> str:
    rel = str(orig_relpath).replace("\\", "/")
    if group_regex:
        m = re.search(group_regex, rel)
        if m:
            if "group" in m.groupdict():
                return str(m.group("group"))
            if m.groups():
                return str(m.group(1))
            return str(m.group(0))
        warnings.warn(f"group_regex did not match {rel}; falling back to source file key.")
        return rel

    parts = Path(rel).parts
    # Expected: label/session/file.wav -> group=session. Otherwise fallback to file.
    if len(parts) >= 3:
        return str(parts[1])
    return rel


def clean_audio_file(src: Path, label: str, root: Path, clean_root: Path, args) -> Optional[dict]:
    orig_relpath = os.path.relpath(src, root).replace("\\", "/")
    y, sr = safe_read_audio(src, dtype="float32")
    if y is None or sr is None:
        return None

    y = to_mono(y)
    if sr != args.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
        sr = int(args.sr)

    y = np.asarray(y, dtype=np.float32)
    if y.size > 0:
        y = y - float(np.mean(y))

    if args.bandpass:
        y = bandpass(y, sr, float(args.bandpass[0]), float(args.bandpass[1]))

    peak = float(np.max(np.abs(y)) + 1e-9) if y.size > 0 else 1e-9
    if peak > 0:
        y = (0.8913 * y / peak).astype(np.float32)

    out = clean_root / "parents" / label / stable_wav_name(label, orig_relpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr)

    clean_relpath = os.path.relpath(out, clean_root).replace("\\", "/")
    return {
        "filepath": str(out),
        "clean_relpath": clean_relpath,
        "label": label,
        "duration": float(len(y) / sr),
        "orig_filepath": str(src),
        "orig_relpath": orig_relpath,
        "orig_ext": src.suffix.lower(),
        "orig_sr": int(sr),
    }


def select_evenly_spaced_starts(starts: List[int], max_keep: int) -> List[int]:
    if max_keep <= 0 or len(starts) <= max_keep:
        return starts
    idx = np.linspace(0, len(starts) - 1, num=max_keep, dtype=int)
    idx = np.unique(idx)
    return [starts[int(i)] for i in idx]


def label_cap(label: str, default_cap: int, cap_map: Dict[str, int]) -> int:
    return int(cap_map.get(str(label), default_cap))


def build_candidate_segments(manifest: pd.DataFrame, clean_root: Path, args) -> pd.DataFrame:
    seg_rows = []
    dropped_short = []

    for _, r in manifest.iterrows():
        parent_rel = str(r["clean_relpath"]).replace("\\", "/")
        parent_path = clean_root / parent_rel
        label = str(r["label"])
        orig_relpath = str(r["orig_relpath"]).replace("\\", "/")
        split_key = orig_relpath if args.split_unit == "file" else get_group_key(orig_relpath, args.group_regex)

        y, sr = sf.read(parent_path, dtype="float32")
        y = to_mono(y)
        win = int(round(float(args.segment_sec) * sr))
        hop = int(round(float(args.hop) * sr))
        n = int(len(y))
        dur_sec = float(n / sr) if sr else 0.0

        if args.input_mode == "ready":
            if dur_sec < float(args.min_keep_sec):
                dropped_short.append((parent_rel, dur_sec))
                continue
            if rms_dbfs(y) < float(args.silence_dbfs):
                continue
            starts = [0]
        else:
            if n < win:
                if dur_sec < float(args.min_keep_sec):
                    dropped_short.append((parent_rel, dur_sec))
                    continue
                if rms_dbfs(y) < float(args.silence_dbfs):
                    continue
                starts = [0]
            else:
                all_starts = list(range(0, max(n - win + 1, 0), max(hop, 1)))
                valid_starts = []
                for s in all_starts:
                    seg = y[s:s + win]
                    if rms_dbfs(seg) >= float(args.silence_dbfs):
                        valid_starts.append(s)
                max_keep = label_cap(label, args.max_segments_per_file_default, args.max_segments_per_label)
                starts = select_evenly_spaced_starts(valid_starts, max_keep)

        for s in starts:
            seg_rows.append({
                "wav_relpath": parent_rel,                  # parent key for clip grouping
                "clean_relpath": parent_rel,
                "orig_relpath": orig_relpath,
                "label": label,
                "parent_start": float(s / sr),
                "start": 0.0,                               # features come from exported segment WAV
                "duration": float(args.segment_sec),
                "split_key": split_key,
                "input_mode": str(args.input_mode),
            })

    if dropped_short:
        print(
            f"\nDropped {len(dropped_short)} files shorter than min_keep_sec={args.min_keep_sec} sec "
            f"(showing up to 10):"
        )
        for rel, dur in dropped_short[:10]:
            print(f" - {rel} ({dur:.3f}s)")
        if len(dropped_short) > 10:
            print(" ... (more dropped)")

    seg_df = pd.DataFrame(seg_rows)
    if len(seg_df) == 0:
        raise SystemExit("No segments above silence threshold; try raising --silence_dbfs, e.g. -55.")
    return seg_df


def split_by_key(
    seg_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify: bool,
    key_col: str = "split_key",
):
    """Split by parent file/group key, then map split to all child segments."""
    from sklearn.model_selection import train_test_split

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0; got {total:.6f} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("Split ratios must be > 0 for train/val/test.")

    if key_col not in seg_df.columns:
        raise ValueError(f"Missing split key column: {key_col}")

    key_df = seg_df[[key_col, "label"]].drop_duplicates().reset_index(drop=True)
    label_counts = key_df.groupby(key_col)["label"].nunique()
    if label_counts.max() > 1:
        warnings.warn(
            "Some split keys contain multiple labels. Stratification will use the first observed label per key."
        )

    key_df = key_df.groupby(key_col, as_index=False).agg(label=("label", "first"))
    keys = key_df[key_col].values
    labels = key_df["label"].values

    temp_frac = val_frac + test_frac
    try:
        if stratify:
            keys_train, keys_temp, y_train, y_temp = train_test_split(
                keys, labels, test_size=temp_frac, stratify=labels, random_state=seed
            )
        else:
            keys_train, keys_temp, y_train, y_temp = train_test_split(
                keys, labels, test_size=temp_frac, random_state=seed, shuffle=True
            )

        test_within_temp = test_frac / temp_frac
        if stratify:
            keys_val, keys_test = train_test_split(
                keys_temp, test_size=test_within_temp, stratify=y_temp, random_state=seed
            )
        else:
            keys_val, keys_test = train_test_split(
                keys_temp, test_size=test_within_temp, random_state=seed, shuffle=True
            )
    except ValueError as e:
        warnings.warn(f"Stratified split failed ({e}). Falling back to non-stratified split.")
        keys_train, keys_temp = train_test_split(keys, test_size=temp_frac, random_state=seed, shuffle=True)
        keys_val, keys_test = train_test_split(
            keys_temp, test_size=(test_frac / temp_frac), random_state=seed, shuffle=True
        )

    split_map = {k: "train" for k in keys_train}
    split_map.update({k: "val" for k in keys_val})
    split_map.update({k: "test" for k in keys_test})

    seg_df = seg_df.copy()
    seg_df["split"] = seg_df[key_col].map(split_map)
    if seg_df["split"].isna().any():
        missing = seg_df.loc[seg_df["split"].isna(), key_col].unique()[:10]
        raise SystemExit(f"Some split keys were not assigned a split: {missing}")

    seg_counts = seg_df["split"].value_counts().to_dict()
    key_counts = key_df.assign(split=key_df[key_col].map(split_map))["split"].value_counts().to_dict()
    return seg_df, seg_counts, key_counts


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)), mode="constant").astype(np.float32)


def export_segments(seg_df: pd.DataFrame, cache: Path, clean_root: Path, args) -> pd.DataFrame:
    """Write physical fixed-length segment WAVs and add segment_wav_relpath."""
    segment_root = cache / "segment_wavs"
    segment_root.mkdir(parents=True, exist_ok=True)

    out_rels = []
    current_parent_rel = None
    current_y = None
    current_sr = None

    for idx, row in seg_df.reset_index(drop=True).iterrows():
        parent_rel = str(row["clean_relpath"]).replace("\\", "/")
        if parent_rel != current_parent_rel:
            parent_path = clean_root / parent_rel
            y, sr = sf.read(parent_path, dtype="float32")
            current_y = to_mono(y)
            current_sr = int(sr)
            current_parent_rel = parent_rel

        start_i = int(round(float(row["parent_start"]) * current_sr))
        dur_i = int(round(float(row["duration"]) * current_sr))
        clip = current_y[start_i:start_i + dur_i]
        clip = pad_or_trim(clip, dur_i)

        label = str(row["label"])
        split = str(row["split"])
        src_key = f"{row['orig_relpath']}|{row['parent_start']:.6f}|{idx}"
        h = hashlib.md5(src_key.encode("utf-8")).hexdigest()[:16]
        label_safe = _safe_token(label, max_len=24)
        fname = f"{label_safe}_seg_{h}.wav"

        out = segment_root / split / label_safe / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.export_segment_wavs:
            sf.write(out, clip, current_sr)

        out_rels.append(os.path.relpath(out, cache).replace("\\", "/"))

    seg_df = seg_df.copy().reset_index(drop=True)
    seg_df["segment_wav_relpath"] = out_rels
    return seg_df


def print_inventory_summary(inv_df: pd.DataFrame):
    print("\n=== Audio inventory summary ===")
    print(f"Files found: {len(inv_df)}")
    print("Labels:", sorted(inv_df["label"].unique().tolist()))
    if len(inv_df) == 0:
        return
    summary = (
        inv_df.groupby("label")
        .agg(
            files=("label", "size"),
            min_sec=("duration", "min"),
            median_sec=("duration", "median"),
            mean_sec=("duration", "mean"),
            max_sec=("duration", "max"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


def remove_cache_artifacts(cache: Path):
    for name in ["clean", "features", "segment_wavs"]:
        p = cache / name
        if p.exists():
            shutil.rmtree(p)
    for name in [
        "segments.csv", "manifest.csv", "moths_manifest.csv", "audio_inventory.csv",
        "audio_inventory_by_label.csv"
    ]:
        p = cache / name
        if p.exists():
            p.unlink()


def main():
    args = two_stage_parse()

    root = Path(args.root)
    cache = Path(args.cache)
    clean_root = cache / "clean"

    if args.force_rebuild:
        remove_cache_artifacts(cache)

    clean_root.mkdir(parents=True, exist_ok=True)
    labels = list_label_dirs(root, args.labels)

    rows = []
    skipped = []
    for label in labels:
        for src in iter_audio_files(root / label):
            row = clean_audio_file(src, label, root, clean_root, args)
            if row is None:
                skipped.append(str(src))
                continue
            rows.append(row)

    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable files (showing up to 10):")
        for s in skipped[:10]:
            print(" -", s)
        if len(skipped) > 10:
            print(" ... (more skipped)")

    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        raise SystemExit("No valid audio files found. Check --root and class folders.")

    cache.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cache / "manifest.csv", index=False)
    # Backward-compatible filename for older scripts/docs.
    manifest.to_csv(cache / "moths_manifest.csv", index=False)
    manifest.to_csv(cache / "audio_inventory.csv", index=False)
    print_inventory_summary(manifest)

    seg_df = build_candidate_segments(manifest, clean_root, args)
    seg_df, seg_counts, key_counts = split_by_key(
        seg_df,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        stratify=bool(args.stratify),
        key_col="split_key",
    )

    seg_df = export_segments(seg_df, cache, clean_root, args)

    # Keep stable column order while preserving any future extra columns.
    preferred = [
        "orig_relpath", "clean_relpath", "wav_relpath", "segment_wav_relpath",
        "label", "split", "split_key", "parent_start", "start", "duration", "input_mode"
    ]
    other = [c for c in seg_df.columns if c not in preferred]
    seg_df = seg_df[preferred + other]
    seg_df.to_csv(cache / "segments.csv", index=False)

    print("\n=== Segmentation summary ===")
    print("Split keys:", key_counts)
    print("Segments:", seg_counts)
    print("\nSegments by split/label:")
    print(seg_df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False))
    print(f"\nSaved segments.csv: {cache / 'segments.csv'}")
    print(f"Saved physical segment WAVs under: {cache / 'segment_wavs'}")


if __name__ == "__main__":
    main()
