# scripts/profile_latency.py

from __future__ import annotations

import os
import json
import csv
import argparse
import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data.datasets import make_loaders
from adapters.audio_adapter import TinyAudioCNN
from models.exit_net import ExitNet
from utils.profiling import measure_latency_ms, estimate_flops_tiny_audiocnn


def load_json_safepath(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _parse_tap_blocks(value) -> Optional[tuple]:
    """
    Accept:
      - None
      - "1,2,3,4"
      - [1,2,3,4]
      - (1,2,3,4)
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)

    value = str(value).strip()
    if value == "":
        return None

    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _build_backbone(n_mels: int, tap_blocks=None):
    """
    Compatible with both:
    - old style: TinyAudioCNN(n_mels=64)
    - new style: TinyAudioCNN(n_mels=64, tap_blocks=(1,2,3,4))
    """
    sig = inspect.signature(TinyAudioCNN.__init__)
    kwargs = {}

    if "n_mels" in sig.parameters:
        kwargs["n_mels"] = int(n_mels)

    if tap_blocks is not None and "tap_blocks" in sig.parameters:
        kwargs["tap_blocks"] = tap_blocks

    return TinyAudioCNN(**kwargs)


def _infer_tap_blocks_from_run(run_dir: Path) -> Optional[tuple]:
    candidates = [
        run_dir / "metrics.json",
        run_dir / "report.json",
        run_dir / "temperature.json",
        run_dir / "thresholds.json",
        run_dir / "summary.json",
        run_dir / "meta.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        obj = load_json_safepath(path, {})
        if not isinstance(obj, dict):
            continue

        tb = None
        if isinstance(obj.get("meta"), dict):
            tb = obj["meta"].get("tap_blocks")
        if tb is None and isinstance(obj.get("policy_summary"), dict):
            tb = obj["policy_summary"].get("tap_blocks")
        if tb is None:
            tb = obj.get("tap_blocks")

        tb = _parse_tap_blocks(tb)
        if tb is not None:
            return tb

    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            tb = _parse_tap_blocks((cfg.get("model") or {}).get("tap_blocks"))
            if tb is not None:
                return tb
        except Exception:
            pass

    return None


def _infer_n_mels_from_run(run_dir: Path, default=64) -> int:
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return int((cfg.get("features") or {}).get("n_mels", default))
        except Exception:
            pass
    return int(default)


def load_first_test_batch(segments_csv, features_root, batch_size=16, num_workers=2):
    """
    Build loaders and return the first batch from the TEST loader.
    This batch is used to measure latency.
    """
    _, _, dl_te, label2id = make_loaders(
        segments_csv,
        features_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    x, y = next(iter(dl_te))
    return x, y, label2id


def infer_feature_shape(segments_csv, features_root):
    """
    Load one feature file from the TEST split to infer (n_mels, frames).
    """
    import pandas as pd

    seg = pd.read_csv(segments_csv)
    test_rows = seg[seg["split"] == "test"]
    if test_rows.empty:
        raise SystemExit("No TEST rows found in segments.csv")

    feat_rel = str(test_rows.iloc[0]["feat_relpath"]).replace("\\", "/")
    feat_path = os.path.join(features_root, feat_rel)
    S = np.load(feat_path)  # (n_mels, T)
    n_mels, frames = S.shape
    return int(n_mels), int(frames)


def _safe_estimate_flops(n_mels, frames, num_classes, tap_blocks, num_exits):
    """
    Best-effort FLOPs estimation.
    If estimate_flops_tiny_audiocnn is still 3-exit-specific, do not crash.
    """
    try:
        sig = inspect.signature(estimate_flops_tiny_audiocnn)
        kwargs = {
            "n_mels": int(n_mels),
            "frames": int(frames),
            "num_classes": int(num_classes),
        }
        if "tap_blocks" in sig.parameters and tap_blocks is not None:
            kwargs["tap_blocks"] = tap_blocks

        fl = estimate_flops_tiny_audiocnn(**kwargs)

        if not isinstance(fl, dict):
            return {
                "flops_raw": fl,
                "warning": "estimate_flops_tiny_audiocnn did not return a dict.",
            }

        needed = [f"exit{i+1}" for i in range(num_exits)]
        if not all(k in fl for k in needed):
            return {
                "flops_raw": fl,
                "warning": "FLOPs helper does not expose all exits for current K.",
            }

        return {
            "flops_raw": fl,
            "warning": None,
        }

    except Exception as e:
        return {
            "flops_raw": None,
            "warning": f"FLOPs estimation failed: {type(e).__name__}: {e}",
        }


def _append_on_device_row_safely(csv_path: Path, row: dict):
    """
    If an old on_device_summary.csv already exists with a 3-exit schema,
    write to a sibling *_kexit.csv instead of corrupting the old schema.
    """
    csv_path.parent.mkdir(exist_ok=True)

    desired_header = list(row.keys())

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=desired_header)
            w.writeheader()
            w.writerow(row)
        print(f"[profile_latency] Appended row to {csv_path}")
        return

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
    except Exception:
        existing_header = None

    if existing_header == desired_header:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=desired_header)
            w.writerow(row)
        print(f"[profile_latency] Appended row to {csv_path}")
        return

    alt_csv = csv_path.with_name(csv_path.stem + "_kexit" + csv_path.suffix)
    write_header = not alt_csv.exists()

    with open(alt_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=desired_header)
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"[profile_latency] Existing CSV schema differs; appended row to {alt_csv} instead of {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Path to a single run directory, e.g. runs/variant/variant_001")
    ap.add_argument("--segments_csv", default="data_cache/segments.csv")
    ap.add_argument("--features_root", default="data_cache/features")
    ap.add_argument("--variant", default="V0",
                    help="Variant name label to store (e.g., V0, V1, V2).")
    ap.add_argument("--device", default="auto",
                    help="cpu | cuda | auto (default: auto picks cuda if available)")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Batch size used for latency measurement.")
    ap.add_argument("--n_warm", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=20)

    # New generic K-exit args
    ap.add_argument("--tap_blocks", type=str, default=None,
                    help='Example: "1,3" for 3 exits or "1,2,3,4" for 5 exits.')
    ap.add_argument("--n_mels", type=int, default=None)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tap_blocks = _parse_tap_blocks(args.tap_blocks)
    if tap_blocks is None:
        tap_blocks = _infer_tap_blocks_from_run(run_dir)

    n_mels_model = int(args.n_mels) if args.n_mels is not None else _infer_n_mels_from_run(run_dir, default=64)

    # --- Load first test batch ---
    x, y, label2id = load_first_test_batch(
        args.segments_csv,
        args.features_root,
        batch_size=args.batch_size,
        num_workers=2,
    )
    batch_size = x.size(0)
    num_classes = len(label2id)

    # --- Build and load model ---
    ckpt_path = run_dir / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    backbone = _build_backbone(n_mels=n_mels_model, tap_blocks=tap_blocks)

    model_kwargs = {
        "backbone": backbone,
        "num_classes": num_classes,
    }

    # Backward compatibility for old backbone versions
    if not hasattr(backbone, "tap_dims"):
        model_kwargs["tap_dims"] = (16, 32)
    if not hasattr(backbone, "final_dim"):
        model_kwargs["final_dim"] = 64

    model = ExitNet(**model_kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    num_exits = model.num_exits
    effective_tap_blocks = tap_blocks
    if effective_tap_blocks is None and hasattr(backbone, "tap_blocks"):
        try:
            effective_tap_blocks = tuple(int(v) for v in backbone.tap_blocks)
        except Exception:
            effective_tap_blocks = None

    # --- Measure latency for full forward (all exits produced) ---
    latency_full_ms = measure_latency_ms(
        model,
        x,
        n_warm=args.n_warm,
        n_iter=args.n_iter,
        device=device,
    )

    # --- Estimate FLOPs per exit to approximate per-exit latency ---
    feat_n_mels, frames = infer_feature_shape(args.segments_csv, args.features_root)
    flops_info = _safe_estimate_flops(
        n_mels=feat_n_mels,
        frames=frames,
        num_classes=num_classes,
        tap_blocks=effective_tap_blocks,
        num_exits=num_exits,
    )

    flops = flops_info.get("flops_raw")
    flops_warning = flops_info.get("warning")

    latency_ms = {}
    full_forward_ms = float(latency_full_ms)

    if isinstance(flops, dict) and f"exit{num_exits}" in flops:
        fl_full = float(flops[f"exit{num_exits}"])
        for k in range(1, num_exits + 1):
            fk = float(flops[f"exit{k}"])
            latency_ms[f"exit{k}"] = full_forward_ms * (fk / max(fl_full, 1e-12))
    else:
        # Fall back to storing only final/full latency as a guaranteed value
        latency_ms[f"exit{num_exits}"] = full_forward_ms

    # --- Load summary.json (for MFLOPs + compute saving) if available ---
    summary_path = run_dir / "summary.json"
    expected_mflops = None
    full_mflops = None
    compute_saving_pct = None

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        policy = summary.get("policy_summary", {})
        expected_mflops = policy.get("expected_mflops", None)
        full_mflops = policy.get("full_mflops", None)
        compute_saving_pct = policy.get("compute_saving_pct", None)

    # --- Build profiling dict ---
    profiling = {
        "variant": args.variant,
        "run_id": run_dir.name,
        "device": device,
        "batch_size": int(batch_size),
        "num_exits": int(num_exits),
        "tap_blocks": list(effective_tap_blocks) if effective_tap_blocks is not None else None,
        "n_mels": int(feat_n_mels),
        "frames": int(frames),
        "full_forward_latency_ms": full_forward_ms,
        "latency_ms": latency_ms,
        "flops": flops,
        "flops_warning": flops_warning,
        "expected_mflops": expected_mflops,
        "full_mflops": full_mflops,
        "compute_saving_pct": compute_saving_pct,
    }

    # Save per-run profiling.json
    out_json = run_dir / "profiling.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profiling, f, indent=2)
    print(f"[profile_latency] Saved {out_json}")

    # --- Append row to analysis/on_device_summary.csv ---
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    csv_path = analysis_dir / "on_device_summary.csv"

    row = {
        "variant": profiling["variant"],
        "run_id": profiling["run_id"],
        "device": profiling["device"],
        "batch_size": profiling["batch_size"],
        "num_exits": profiling["num_exits"],
        "tap_blocks_json": json.dumps(profiling["tap_blocks"]),
        "n_mels": profiling["n_mels"],
        "frames": profiling["frames"],
        "full_forward_latency_ms": profiling["full_forward_latency_ms"],
        "latency_ms_json": json.dumps(profiling["latency_ms"]),
        "flops_json": json.dumps(profiling["flops"]),
        "expected_mflops": profiling["expected_mflops"],
        "full_mflops": profiling["full_mflops"],
        "compute_saving_pct": profiling["compute_saving_pct"],
        "flops_warning": profiling["flops_warning"],
    }

    _append_on_device_row_safely(csv_path, row)


if __name__ == "__main__":
    main()