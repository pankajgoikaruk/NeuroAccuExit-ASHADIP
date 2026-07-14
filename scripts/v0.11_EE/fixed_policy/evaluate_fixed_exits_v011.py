from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.anytime_exit_net import AnytimeExitNet
from utils.model_factory import build_audio_exit_net


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_tap_blocks(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def load_run_config(run_dir: Path) -> dict[str, Any]:
    json_path = run_dir / "config_used.json"
    if json_path.exists():
        return load_json(json_path)

    yaml_path = run_dir / "config_used.yaml"
    if yaml_path.exists():
        from utils.config import load_config

        return load_config(yaml_path) or {}

    raise FileNotFoundError(
        f"No config_used.json or config_used.yaml found in run directory: {run_dir}"
    )


def resolve_settings(
    cfg: dict[str, Any],
    *,
    labels_json: Path,
    tap_blocks_override: str | None,
    n_mels_override: int | None,
) -> tuple[list[str], tuple[int, ...], int, dict[str, Any]]:
    schema = load_json(labels_json)
    labels = [str(x) for x in schema.get("labels", [])]
    if not labels:
        labels = [str(x) for x in cfg.get("labels", [])]
    if not labels:
        raise RuntimeError("Could not resolve the label list.")

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


def load_feature(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected feature shape [n_mels, T], got {arr.shape}: {path}")
    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)


def segment_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_pred = (y_prob >= float(threshold)).astype(int)
    result = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
        "n_segments": int(y_true.shape[0]),
        "threshold": float(threshold),
    }

    per_label = []
    for idx, label in enumerate(labels):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        per_label.append(
            {
                "label": label,
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "support": int(yt.sum()),
                "predicted_positive": int(yp.sum()),
            }
        )
    return result, per_label


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Always Exit 1, Always Exit 2, and Always Exit 3 using "
            "the frozen v0.10 no-hint checkpoint."
        )
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--holdout_manifest", required=True, type=Path)
    parser.add_argument("--features_root", required=True, type=Path)
    parser.add_argument("--labels_json", required=True, type=Path)
    parser.add_argument("--lats_config_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument("--segment_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tap_blocks", default=None)
    parser.add_argument("--n_mels", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip_parent_eval",
        action="store_true",
        help="Export probabilities and segment metrics without frozen LATS parent evaluation.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint or (run_dir / "ckpt" / "best.pt")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for required_path in (
        checkpoint,
        args.holdout_manifest,
        args.features_root,
        args.labels_json,
        args.lats_config_json,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    cfg = load_run_config(run_dir)
    labels, tap_blocks, n_mels, model_cfg = resolve_settings(
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

    df = pd.read_csv(args.holdout_manifest, low_memory=False)
    required_columns = ["feat_relpath", args.parent_id_col, *labels]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"Holdout manifest is missing columns: {missing_columns}")

    y_true = df[labels].astype(int).to_numpy()
    probs_parts: list[list[np.ndarray]] | None = None

    for start in range(0, len(df), int(args.batch_size)):
        batch = df.iloc[start : start + int(args.batch_size)]
        tensors = []
        for _, row in batch.iterrows():
            rel = Path(str(row["feat_relpath"]).replace("\\", "/"))
            tensors.append(load_feature(args.features_root / rel))

        shapes = {tuple(x.shape) for x in tensors}
        if len(shapes) != 1:
            raise RuntimeError(
                f"Batch beginning at row {start} contains inconsistent feature shapes: {shapes}"
            )

        x = torch.cat(tensors, dim=0).to(args.device)
        logits_by_exit = anytime_model.forward_all_staged(x)
        probs_by_exit = [torch.sigmoid(logits).cpu().numpy() for logits in logits_by_exit]

        if probs_parts is None:
            probs_parts = [[] for _ in probs_by_exit]
        for exit_idx, probs in enumerate(probs_by_exit):
            probs_parts[exit_idx].append(probs)

    if probs_parts is None:
        raise RuntimeError("No holdout probabilities were generated.")

    probabilities = [np.concatenate(parts, axis=0) for parts in probs_parts]
    if any(prob.shape[0] != len(df) for prob in probabilities):
        raise RuntimeError("Probability row count does not match the holdout manifest.")

    export_columns = [args.parent_id_col, "feat_relpath", *labels]
    segment_export = df[export_columns].copy()
    for exit_no, probs in enumerate(probabilities, start=1):
        for label_idx, label in enumerate(labels):
            segment_export[f"exit{exit_no}_prob_{label}"] = probs[:, label_idx]

    segment_csv = out_dir / "v011_fixed_exit_segment_probabilities.csv"
    segment_export.to_csv(segment_csv, index=False)

    segment_rows = []
    segment_details: dict[str, Any] = {}
    for exit_no, probs in enumerate(probabilities, start=1):
        metrics, per_label = segment_metrics(
            y_true,
            probs,
            labels,
            threshold=float(args.segment_threshold),
        )
        row = {"exit": exit_no, **metrics}
        segment_rows.append(row)
        segment_details[f"exit_{exit_no}"] = {
            "metrics": metrics,
            "per_label": per_label,
        }
        pd.DataFrame(per_label).to_csv(
            out_dir / f"v011_exit{exit_no}_segment_per_label.csv",
            index=False,
        )

    segment_summary = pd.DataFrame(segment_rows)
    segment_summary.to_csv(out_dir / "v011_fixed_exit_segment_summary.csv", index=False)
    save_json(
        {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "holdout_manifest": str(args.holdout_manifest),
            "features_root": str(args.features_root),
            "labels": labels,
            "tap_blocks": list(tap_blocks),
            "num_exits": len(probabilities),
            "segment_threshold": float(args.segment_threshold),
            "fixed_exit_results": segment_details,
        },
        out_dir / "v011_fixed_exit_segment_results.json",
    )

    parent_rows = []
    if not args.skip_parent_eval:
        evaluator = PROJECT_ROOT / "scripts" / "v0.10" / "evaluate_frozen_lats_config_v010.py"
        if not evaluator.exists():
            raise FileNotFoundError(f"Frozen LATS evaluator not found: {evaluator}")

        for exit_no in range(1, len(probabilities) + 1):
            exit_out = out_dir / f"exit{exit_no}_parent_frozen_lats_transfer"
            command = [
                sys.executable,
                str(evaluator),
                "--segment-pred-csv",
                str(segment_csv),
                "--labels-json",
                str(args.labels_json),
                "--config-json",
                str(args.lats_config_json),
                "--out-dir",
                str(exit_out),
                "--parent-id-col",
                str(args.parent_id_col),
                "--prob-prefix",
                f"exit{exit_no}_prob_",
                "--model-name",
                f"v0.11_EE_always_exit_{exit_no}",
            ]
            subprocess.run(command, check=True)

            summary_path = exit_out / "v010_frozen_lats_eval.csv"
            summary = pd.read_csv(summary_path).iloc[0].to_dict()
            summary["exit"] = exit_no
            summary["evaluation_role"] = "frozen_lats_v2_policy_transfer"
            parent_rows.append(summary)

        parent_summary = pd.DataFrame(parent_rows).sort_values("exit")
        parent_summary.to_csv(
            out_dir / "v011_fixed_exit_parent_frozen_lats_summary.csv",
            index=False,
        )

    run_manifest = {
        "experiment": "v0.11_EE_fixed_exit_audit",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "holdout_manifest": str(args.holdout_manifest),
        "features_root": str(args.features_root),
        "labels_json": str(args.labels_json),
        "lats_config_json": str(args.lats_config_json),
        "parent_id_col": args.parent_id_col,
        "tap_blocks": list(tap_blocks),
        "num_exits": len(probabilities),
        "n_segments": int(len(df)),
        "n_parents": int(df[args.parent_id_col].astype(str).nunique()),
        "segment_probability_csv": str(segment_csv),
        "segment_summary_csv": str(out_dir / "v011_fixed_exit_segment_summary.csv"),
        "parent_summary_csv": (
            None
            if args.skip_parent_eval
            else str(out_dir / "v011_fixed_exit_parent_frozen_lats_summary.csv")
        ),
        "warning": (
            "Exit 1 and Exit 2 parent results transfer the frozen Exit 3 LATS-v2 "
            "policy without exit-specific recalibration."
        ),
    }
    save_json(run_manifest, out_dir / "v011_fixed_exit_run_manifest.json")

    print("\nV0.11 fixed-exit evaluation complete")
    print("-" * 100)
    print(segment_summary.to_string(index=False))
    if parent_rows:
        parent_display = pd.DataFrame(parent_rows).sort_values("exit")
        display_cols = [
            "exit",
            "macro_f1",
            "micro_f1",
            "samples_f1",
            "exact_match",
            "hamming_loss",
            "avg_pred_labels",
        ]
        print("\nParent-level frozen LATS-v2 transfer")
        print(parent_display[display_cols].to_string(index=False))
    print(f"\nSaved outputs: {out_dir}")


if __name__ == "__main__":
    main()
