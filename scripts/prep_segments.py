# scripts/prep_segments.py

import os
import re
import json
import shutil
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
    return 20.0 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)


def detect_num_channels(y: np.ndarray) -> int:
    y = np.asarray(y)
    if y.ndim == 1:
        return 1
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return int(y.shape[0])
    return int(y.shape[1])


def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
        return y.mean(axis=0).astype(np.float32)
    return y.mean(axis=1).astype(np.float32)


def safe_read_audio(path, dtype="float32"):
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


def parse_json_arg(text: str, default=None):
    if not text:
        return {} if default is None else default
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise SystemExit("JSON option must decode to an object/dict.")
    return obj


def derive_group_id(relpath: str, group_mode: str = "none", group_regex: str = "") -> str:
    rel = str(relpath).replace("\\", "/")
    p = Path(rel)
    if group_mode == "none":
        return rel
    if group_mode == "parent":
        parent = p.parent.as_posix()
        return parent if parent and parent != "." else rel
    if group_mode == "stem":
        return p.stem
    if group_mode == "regex":
        if not group_regex:
            raise SystemExit("group_mode=regex requires --group_regex")
        m = re.search(group_regex, rel)
        if not m:
            return rel
        if m.groups():
            return m.group(1)
        return m.group(0)
    raise SystemExit(f"Unknown group_mode: {group_mode}")


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


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
            unreadable=("read_ok", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    print(by_label.to_string(index=False))


def resolve_label_cap(label: str, args) -> int:
    if label in args.max_segments_per_label:
        return int(args.max_segments_per_label[label])
    if label == "gunshot" and args.max_segments_per_file_gunshot is not None:
        return int(args.max_segments_per_file_gunshot)
    if label == "non_gunshot" and args.max_segments_per_file_non_gunshot is not None:
        return int(args.max_segments_per_file_non_gunshot)
    return int(args.max_segments_per_file_default)


def select_evenly_spaced_starts(all_starts, max_keep: int):
    if max_keep <= 0 or len(all_starts) <= max_keep:
        return list(all_starts)
    idx = np.linspace(0, len(all_starts) - 1, num=max_keep, dtype=int)
    idx = np.unique(idx)
    return [all_starts[i] for i in idx]


def split_by_key(df: pd.DataFrame, key_col: str, train_frac: float, val_frac: float, test_frac: float,
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

    unit_df = df[[key_col, "label"]].drop_duplicates().reset_index(drop=True)
    nunq = unit_df.groupby(key_col)["label"].nunique()
    if nunq.max() != 1:
        bad = nunq[nunq > 1].index.tolist()[:10]
        raise ValueError(f"Some split keys map to multiple labels (examples): {bad}")

    keys = unit_df[key_col].values
    labels = unit_df["label"].values
    temp_frac = val_frac + test_frac

    try:
        if stratify:
            keys_train, keys_temp, _, y_temp = train_test_split(
                keys, labels, test_size=temp_frac, stratify=labels, random_state=seed
            )
        else:
            keys_train, keys_temp, _, y_temp = train_test_split(
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

    out_df = df.copy()
    out_df["split"] = out_df[key_col].map(split_map)
    if out_df["split"].isna().any():
        missing = out_df.loc[out_df["split"].isna(), key_col].unique()[:10]
        raise SystemExit(f"Some split keys were not assigned a split (examples): {missing}")

    split_counts = out_df["split"].value_counts().to_dict()
    unit_counts = unit_df.assign(split=unit_df[key_col].map(split_map))["split"].value_counts().to_dict()
    return out_df, split_counts, unit_counts


def build_inventory(root: Path, labels, group_mode: str, group_regex: str):
    inv_rows = []
    for label in labels:
        for src in iter_audio_files(root / label):
            rel = os.path.relpath(src, root)
            group_id = derive_group_id(rel, group_mode, group_regex)
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
                    "group_id": group_id,
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
                "group_id": group_id,
            })
    inv_df = pd.DataFrame(inv_rows)
    if len(inv_df) == 0:
        raise SystemExit(f"No audio files found under {root}")
    return inv_df


def write_clean_manifest(inv_df: pd.DataFrame, cache: Path, sr_target: int, bandpass_range, write_compat_manifest: bool):
    clean_root = cache / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for _, r in inv_df.iterrows():
        if not bool(r["read_ok"]):
            skipped.append(str(r["orig_filepath"]))
            continue

        src = Path(r["orig_filepath"])
        label = str(r["label"])
        orig_relpath = str(r["orig_relpath"])
        group_id = str(r["group_id"])

        y, sr = safe_read_audio(src, dtype="float32")
        if y is None:
            skipped.append(str(src))
            continue

        y = to_mono(y)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        y = y.astype(np.float32)
        if y.size > 0:
            y = y - float(np.mean(y))
        if bandpass_range:
            y = bandpass(y, sr, float(bandpass_range[0]), float(bandpass_range[1]))
        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 0:
            y = 0.8913 * y / peak

        file_hash = hashlib.md5(orig_relpath.encode("utf-8")).hexdigest()[:12]
        out_name = f"{label}_{file_hash}.wav"
        out = clean_root / label / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, y, sr)

        rows.append({
            "filepath": str(out),
            "clean_relpath": os.path.relpath(out, clean_root),
            "label": label,
            "duration": float(len(y) / sr),
            "orig_filepath": str(src),
            "orig_relpath": orig_relpath,
            "group_id": group_id,
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
    if write_compat_manifest:
        manifest.to_csv(cache / "moths_manifest.csv", index=False)
    return manifest


def build_segments_segment_mode(manifest: pd.DataFrame, args):
    seg_rows = []
    rejected_rows = []

    for _, r in manifest.iterrows():
        y, sr = sf.read(r["filepath"], dtype="float32")
        y = np.asarray(y, dtype=np.float32)
        win = int(args.segment_sec * sr)
        hop = int(args.hop * sr)
        n = int(len(y))
        dur_sec = float(n / sr)

        rel = str(r["clean_relpath"])
        label = str(r["label"])
        orig_relpath = str(r["orig_relpath"])
        group_id = str(r["group_id"])
        max_keep = resolve_label_cap(label, args)
        split_key = group_id if args.split_unit == "group" else rel

        if n < win:
            clip_rms = rms_dbfs(y)
            if dur_sec < float(args.min_keep_sec):
                rejected_rows.append({
                    "wav_relpath": rel,
                    "orig_relpath": orig_relpath,
                    "label": label,
                    "group_id": group_id,
                    "reason": "too_short",
                    "duration_sec": dur_sec,
                    "rms_dbfs": clip_rms,
                })
                continue
            if clip_rms < args.silence_dbfs:
                rejected_rows.append({
                    "wav_relpath": rel,
                    "orig_relpath": orig_relpath,
                    "label": label,
                    "group_id": group_id,
                    "reason": "short_but_silent",
                    "duration_sec": dur_sec,
                    "rms_dbfs": clip_rms,
                })
                continue
            seg_rows.append({
                "wav_relpath": rel,
                "orig_relpath": orig_relpath,
                "split_key": split_key,
                "group_id": group_id,
                "label": label,
                "start": 0.0,
                "duration": float(args.segment_sec),
                "source_duration": dur_sec,
                "is_padded_short": True,
                "segment_rms_dbfs": clip_rms,
                "input_mode": args.input_mode,
            })
            continue

        all_starts = list(range(0, max(n - win + 1, 0), hop))
        valid_pairs = []
        for s in all_starts:
            seg = y[s:s + win]
            seg_rms = rms_dbfs(seg)
            if seg_rms < args.silence_dbfs:
                rejected_rows.append({
                    "wav_relpath": rel,
                    "orig_relpath": orig_relpath,
                    "label": label,
                    "group_id": group_id,
                    "reason": "silent_window",
                    "start": float(s / sr),
                    "duration_sec": float(args.segment_sec),
                    "rms_dbfs": seg_rms,
                })
                continue
            valid_pairs.append((s, seg_rms))

        kept_pairs = valid_pairs
        if max_keep > 0 and len(valid_pairs) > max_keep:
            keep_starts = set(select_evenly_spaced_starts([s for s, _ in valid_pairs], max_keep))
            kept_pairs = [(s, seg_rms) for s, seg_rms in valid_pairs if s in keep_starts]
            dropped_pairs = [(s, seg_rms) for s, seg_rms in valid_pairs if s not in keep_starts]
            for s, seg_rms in dropped_pairs:
                rejected_rows.append({
                    "wav_relpath": rel,
                    "orig_relpath": orig_relpath,
                    "label": label,
                    "group_id": group_id,
                    "reason": "cap_dropped",
                    "start": float(s / sr),
                    "duration_sec": float(args.segment_sec),
                    "rms_dbfs": seg_rms,
                })

        for s, seg_rms in kept_pairs:
            seg_rows.append({
                "wav_relpath": rel,
                "orig_relpath": orig_relpath,
                "split_key": split_key,
                "group_id": group_id,
                "label": label,
                "start": float(s / sr),
                "duration": float(args.segment_sec),
                "source_duration": dur_sec,
                "is_padded_short": False,
                "segment_rms_dbfs": seg_rms,
                "input_mode": args.input_mode,
            })

    return pd.DataFrame(seg_rows), pd.DataFrame(rejected_rows)


def build_segments_ready_mode(manifest: pd.DataFrame, args):
    seg_rows = []
    rejected_rows = []

    for _, r in manifest.iterrows():
        y, sr = sf.read(r["filepath"], dtype="float32")
        y = np.asarray(y, dtype=np.float32)
        dur_sec = float(len(y) / sr)
        clip_rms = rms_dbfs(y)

        rel = str(r["clean_relpath"])
        label = str(r["label"])
        orig_relpath = str(r["orig_relpath"])
        group_id = str(r["group_id"])
        split_key = group_id if args.split_unit == "group" else rel

        if dur_sec < float(args.min_keep_sec):
            rejected_rows.append({
                "wav_relpath": rel,
                "orig_relpath": orig_relpath,
                "label": label,
                "group_id": group_id,
                "reason": "ready_too_short",
                "duration_sec": dur_sec,
                "rms_dbfs": clip_rms,
            })
            continue
        if clip_rms < args.silence_dbfs:
            rejected_rows.append({
                "wav_relpath": rel,
                "orig_relpath": orig_relpath,
                "label": label,
                "group_id": group_id,
                "reason": "ready_silent",
                "duration_sec": dur_sec,
                "rms_dbfs": clip_rms,
            })
            continue

        if args.strict_ready_length:
            if abs(dur_sec - float(args.segment_sec)) > float(args.ready_length_tolerance_sec):
                rejected_rows.append({
                    "wav_relpath": rel,
                    "orig_relpath": orig_relpath,
                    "label": label,
                    "group_id": group_id,
                    "reason": "ready_length_mismatch",
                    "duration_sec": dur_sec,
                    "target_duration_sec": float(args.segment_sec),
                    "rms_dbfs": clip_rms,
                })
                continue

        seg_rows.append({
            "wav_relpath": rel,
            "orig_relpath": orig_relpath,
            "split_key": split_key,
            "group_id": group_id,
            "label": label,
            "start": 0.0,
            "duration": float(args.segment_sec),
            "source_duration": dur_sec,
            "is_padded_short": bool(dur_sec < float(args.segment_sec)),
            "segment_rms_dbfs": clip_rms,
            "input_mode": args.input_mode,
        })

    return pd.DataFrame(seg_rows), pd.DataFrame(rejected_rows)


def sanitize_export_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return text.strip("._") or "clip"


def export_segment_wavs(seg_df: pd.DataFrame, cache: Path, export_root: Path, sr_target: int):
    clean_root = cache / "clean"
    export_rows = []

    for idx, row in seg_df.reset_index(drop=True).iterrows():
        src = clean_root / str(row["wav_relpath"])
        if not src.exists():
            warnings.warn(f"Cannot export segment; source file missing: {src}")
            continue

        y, sr = sf.read(src, dtype="float32")
        y = np.asarray(y, dtype=np.float32)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        start_sample = int(round(float(row["start"]) * sr))
        win = int(round(float(row["duration"]) * sr))
        seg = y[start_sample:start_sample + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)), mode="constant")

        split = str(row["split"])
        label = str(row["label"])
        rel = str(row["wav_relpath"])
        stem = sanitize_export_name(Path(rel).stem)
        start_ms = int(round(float(row["start"]) * 1000.0))
        out_name = f"{stem}__{start_ms:08d}ms__{idx:06d}.wav"
        out = export_root / split / label / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, seg, sr)

        export_rows.append({
            "wav_relpath": rel,
            "label": label,
            "split": split,
            "start": float(row["start"]),
            "duration": float(row["duration"]),
            "export_path": str(out),
            "export_relpath": os.path.relpath(out, export_root),
        })

    export_df = pd.DataFrame(export_rows)
    export_df.to_csv(export_root / "export_manifest.csv", index=False)
    return export_df


def two_stage_parse():
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default="", help="Optional YAML config.")
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

    p.add_argument("--input_mode", choices=["segment", "ready"], default="segment",
                   help="segment = cut longer files into windows; ready = each file is already one clip.")
    p.add_argument("--strict_ready_length", action="store_true",
                   help="In ready mode, reject files whose duration differs too much from segment_sec.")
    p.add_argument("--ready_length_tolerance_sec", type=float, default=0.05,
                   help="Allowed duration mismatch in ready mode when --strict_ready_length is used.")

    p.add_argument("--export_segment_wavs", action="store_true",
                   help="Also export physical segment WAVs split/label wise.")
    p.add_argument("--export_root", type=str, default="",
                   help="Where to export segment WAVs. Default: <cache>/exported_segments")

    p.add_argument("--skip_if_segments_exist", action="store_true",
                   help="If cache/segments.csv exists, stop early and reuse it.")
    p.add_argument("--force_rebuild", action="store_true",
                   help="Delete old cache outputs before rebuilding.")

    p.add_argument("--max_segments_per_file_default", type=int, default=0,
                   help="Generic default cap for kept segments per file. 0 = keep all.")
    p.add_argument("--max_segments_per_label_json", type=str, default="",
                   help='JSON dict of per-label caps, e.g. {"non_gunshot": 5, "fireworks": 8}.')

    # Backward-compatible old options
    p.add_argument("--max_segments_per_file_gunshot", type=int, default=None,
                   help="Backward-compatible legacy option. Prefer --max_segments_per_label_json.")
    p.add_argument("--max_segments_per_file_non_gunshot", type=int, default=None,
                   help="Backward-compatible legacy option. Prefer --max_segments_per_label_json.")

    p.add_argument("--group_mode", choices=["none", "parent", "stem", "regex"], default="none",
                   help="How to derive split groups from orig_relpath. none=file-level; parent=folder-level; stem=filename stem; regex=use group_regex.")
    p.add_argument("--group_regex", type=str, default="",
                   help="Regex used when group_mode=regex. First capture group is preferred.")
    p.add_argument("--split_unit", choices=["file", "group"], default="file",
                   help="Split by cleaned file path or derived group id.")

    if strat_default:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=True)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")
    else:
        p.add_argument("--stratify", dest="stratify", action="store_true", default=False)
        p.add_argument("--no_stratify", dest="stratify", action="store_false")

    args = p.parse_args(rest)
    args._cfg = cfg
    args.max_segments_per_label = parse_json_arg(args.max_segments_per_label_json, default={})
    return args


def maybe_cleanup_cache(cache: Path, export_root: Path, force_rebuild: bool):
    if not force_rebuild:
        return
    for target in [cache / "audio_inventory.csv", cache / "audio_inventory_by_label.csv", cache / "manifest.csv",
                   cache / "moths_manifest.csv", cache / "segments.csv", cache / "segments_rejected.csv",
                   cache / "clean"]:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink(missing_ok=True)
    if export_root.exists():
        shutil.rmtree(export_root, ignore_errors=True)


def main():
    args = two_stage_parse()

    root = Path(args.root)
    cache = Path(args.cache)
    export_root = Path(args.export_root) if args.export_root else (cache / "exported_segments")
    cache.mkdir(parents=True, exist_ok=True)

    maybe_cleanup_cache(cache, export_root, args.force_rebuild)

    segments_csv = cache / "segments.csv"
    if args.skip_if_segments_exist and segments_csv.exists() and not args.force_rebuild:
        print(f"segments.csv already exists at {segments_csv}. Skipping because --skip_if_segments_exist was set.")
        return

    labels = list_label_dirs(root, args.labels)

    inv_df = build_inventory(root, labels, args.group_mode, args.group_regex)
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

    manifest = write_clean_manifest(
        inv_df=inv_df,
        cache=cache,
        sr_target=int(args.sr),
        bandpass_range=args.bandpass,
        write_compat_manifest=bool(args.write_compat_manifest),
    )

    if args.input_mode == "segment":
        seg_df, rej_df = build_segments_segment_mode(manifest, args)
    else:
        seg_df, rej_df = build_segments_ready_mode(manifest, args)

    if len(seg_df) == 0:
        raise SystemExit("No valid segments were produced. Check silence threshold, labels, or ready-mode length assumptions.")

    key_col = "split_key"
    seg_df, seg_counts, unit_counts = split_by_key(
        seg_df,
        key_col=key_col,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        stratify=bool(args.stratify),
    )

    # Keep core downstream columns first for compatibility.
    preferred = [
        "wav_relpath", "label", "start", "duration", "split",
        "orig_relpath", "group_id", "source_duration", "is_padded_short",
        "segment_rms_dbfs", "input_mode", "split_key",
    ]
    seg_cols = [c for c in preferred if c in seg_df.columns] + [c for c in seg_df.columns if c not in preferred]
    seg_df = seg_df[seg_cols]

    seg_df.to_csv(cache / "segments.csv", index=False)
    if len(rej_df) == 0:
        rej_df = pd.DataFrame(columns=["wav_relpath", "orig_relpath", "label", "group_id", "reason"])
    rej_df.to_csv(cache / "segments_rejected.csv", index=False)

    print("\n=== Segmentation summary ===")
    print(f"Input mode: {args.input_mode}")
    print(f"Split unit: {args.split_unit}")
    print("Units:", unit_counts)
    print("Segments:", seg_counts)
    print("\nSegments by label:")
    print(seg_df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False))
    if len(rej_df) > 0:
        print("\nRejected segment reasons:")
        print(rej_df.groupby("reason").size().rename("count").reset_index().to_string(index=False))

    if args.export_segment_wavs:
        export_root.mkdir(parents=True, exist_ok=True)
        export_df = export_segment_wavs(seg_df, cache=cache, export_root=export_root, sr_target=int(args.sr))
        print(f"\nExported {len(export_df)} segment WAVs to: {export_root}")


if __name__ == "__main__":
    main()