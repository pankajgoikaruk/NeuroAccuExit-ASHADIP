from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.anytime_exit_net import AnytimeExitNet
from utils.model_factory import build_audio_exit_net


def parse_tap_blocks(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def load_run_config(run_dir: Path) -> dict[str, Any]:
    json_path = run_dir / "config_used.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    yaml_path = run_dir / "config_used.yaml"
    if yaml_path.exists():
        try:
            from utils.config import load_config
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Could not import utils.config.load_config") from exc
        return load_config(yaml_path) or {}

    raise FileNotFoundError(
        f"No config_used.json or config_used.yaml found in run directory: {run_dir}"
    )


def resolve_model_settings(
    cfg: dict[str, Any],
    *,
    labels_json: Path | None,
    tap_blocks_override: str | None,
    n_mels_override: int | None,
) -> tuple[list[str], tuple[int, ...], int, dict[str, Any]]:
    labels = cfg.get("labels")
    if not labels and labels_json is not None:
        with labels_json.open("r", encoding="utf-8") as f:
            labels = json.load(f).get("labels")
    if not labels:
        raise RuntimeError("Could not resolve label order from run config or --labels_json.")
    labels = [str(x) for x in labels]

    model_cfg = cfg.get("model") or {}
    if not model_cfg and isinstance(cfg.get("exit_hint"), dict):
        model_cfg = {"exit_hint": cfg["exit_hint"]}
    if "exit_hint" not in model_cfg:
        model_cfg = dict(model_cfg)
        model_cfg["exit_hint"] = {"enable": False}

    raw_taps = (
        tap_blocks_override
        or cfg.get("tap_blocks")
        or model_cfg.get("tap_blocks")
        or "1,3"
    )
    tap_blocks = parse_tap_blocks(raw_taps)

    n_mels = int(
        n_mels_override
        if n_mels_override is not None
        else cfg.get("n_mels", (cfg.get("features") or {}).get("n_mels", 64))
    )
    return labels, tap_blocks, n_mels, model_cfg


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def load_real_batch(
    manifest_path: Path,
    features_root: Path,
    *,
    sample_count: int,
) -> torch.Tensor:
    df = pd.read_csv(manifest_path, low_memory=False)
    if "feat_relpath" not in df.columns:
        raise ValueError(f"Manifest lacks feat_relpath: {manifest_path}")
    if len(df) == 0:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    tensors = []
    for _, row in df.head(int(sample_count)).iterrows():
        rel = Path(str(row["feat_relpath"]).replace("\\", "/"))
        path = features_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"Expected [n_mels, T], got {arr.shape}: {path}")
        tensors.append(torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0))

    shapes = {tuple(x.shape) for x in tensors}
    if len(shapes) != 1:
        raise RuntimeError(f"Selected features have inconsistent shapes: {sorted(shapes)}")
    return torch.cat(tensors, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the v0.11 staged forward path reproduces the original "
            "full-forward logits for a trained checkpoint."
        )
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--labels_json", type=Path, default=None)
    parser.add_argument("--holdout_manifest", type=Path, default=None)
    parser.add_argument("--features_root", type=Path, default=None)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--tap_blocks", default=None)
    parser.add_argument("--n_mels", type=int, default=None)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint or (run_dir / "ckpt" / "best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = load_run_config(run_dir)
    labels, tap_blocks, n_mels, model_cfg = resolve_model_settings(
        cfg,
        labels_json=args.labels_json,
        tap_blocks_override=args.tap_blocks,
        n_mels_override=args.n_mels,
    )

    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=model_cfg,
    ).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()

    anytime_model = AnytimeExitNet(model).to(args.device)
    anytime_model.eval()

    if args.holdout_manifest is not None or args.features_root is not None:
        if args.holdout_manifest is None or args.features_root is None:
            raise ValueError(
                "--holdout_manifest and --features_root must be supplied together."
            )
        x = load_real_batch(
            args.holdout_manifest,
            args.features_root,
            sample_count=args.sample_count,
        )
        input_source = "real_holdout_features"
    else:
        torch.manual_seed(int(args.seed))
        x = torch.randn(
            int(args.sample_count),
            1,
            int(n_mels),
            int(args.frames),
        )
        input_source = "synthetic"

    x = x.to(args.device)
    with torch.no_grad():
        full_outputs = model(x)
        staged_outputs = anytime_model.forward_all_staged(x)

    if len(full_outputs) != len(staged_outputs):
        raise RuntimeError(
            f"Exit count mismatch: full={len(full_outputs)}, staged={len(staged_outputs)}"
        )

    exit_results = []
    passed = True
    for exit_no, (full_logits, staged_logits) in enumerate(
        zip(full_outputs, staged_outputs), start=1
    ):
        logit_diff = (full_logits - staged_logits).abs()
        prob_diff = (torch.sigmoid(full_logits) - torch.sigmoid(staged_logits)).abs()
        close = bool(
            torch.allclose(
                full_logits,
                staged_logits,
                rtol=float(args.rtol),
                atol=float(args.atol),
            )
        )
        passed = passed and close
        exit_results.append(
            {
                "exit": exit_no,
                "passed": close,
                "max_abs_logit_diff": float(logit_diff.max().item()),
                "mean_abs_logit_diff": float(logit_diff.mean().item()),
                "max_abs_probability_diff": float(prob_diff.max().item()),
            }
        )

    result = {
        "status": "PASS" if passed else "FAIL",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "input_source": input_source,
        "input_shape": list(x.shape),
        "labels": labels,
        "tap_blocks": list(tap_blocks),
        "num_exits": len(full_outputs),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "exit_results": exit_results,
    }

    out_json = args.out_json or (
        run_dir / "v0.11_EE" / "checkpoint_staged_equivalence.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved: {out_json}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
