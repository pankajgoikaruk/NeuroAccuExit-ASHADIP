#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a frozen v0.11 policy with genuine staged runtime compute skipping.

Every sample executes Blocks 1-3 to reach Exit 2. Samples accepted by the
frozen policy stop at Exit 2. Only rejected samples are sliced from the staged
state and passed through Blocks 4-5 to Exit 3.

No training or holdout retuning occurs here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, jaccard_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.anytime_exit_net import AnytimeExitNet, AnytimeExitState
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        raise TypeError(f"Cannot serialize {type(value)}")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=convert)


def parse_tap_blocks(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


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


def load_labels(labels_json: Path, cfg: dict[str, Any]) -> list[str]:
    payload = load_json(labels_json)
    if isinstance(payload, list):
        labels = payload
    else:
        labels = payload.get("labels") or payload.get("classes") or cfg.get("labels")

    if not labels:
        raise RuntimeError(f"Could not resolve labels from {labels_json}")
    return [str(label) for label in labels]


def resolve_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(cfg.get("model") or {})
    if "exit_hint" not in model_cfg:
        exit_hint = cfg.get("exit_hint")
        model_cfg["exit_hint"] = (
            exit_hint if isinstance(exit_hint, dict) else {"enable": False}
        )
    return model_cfg


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def load_feature(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    array = np.load(path).astype(np.float32)
    if array.ndim != 2:
        raise RuntimeError(f"Expected [n_mels, T], got {array.shape}: {path}")
    return torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)


def thresholds_from_policy(
    policy: dict[str, Any], labels: list[str], exit_no: int
) -> np.ndarray:
    mapping = policy["thresholds_by_exit"][f"exit{exit_no}"]
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(
            f"Frozen policy Exit {exit_no} thresholds are missing labels: {missing}"
        )
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def label_predictions(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (
        probabilities >= np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    ).astype(int)


def stop_diagnostics(
    exit1_probs: np.ndarray,
    exit2_probs: np.ndarray,
    exit1_thresholds: np.ndarray,
    exit2_thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    exit1_pred = label_predictions(exit1_probs, exit1_thresholds)
    exit2_pred = label_predictions(exit2_probs, exit2_thresholds)
    return {
        "exit1_pred": exit1_pred,
        "exit2_pred": exit2_pred,
        "agreement": np.all(exit1_pred == exit2_pred, axis=1),
        "non_empty": exit2_pred.sum(axis=1) > 0,
        "mean_binary_confidence": np.maximum(exit2_probs, 1.0 - exit2_probs).mean(axis=1),
        "min_decision_margin": np.min(
            np.abs(exit2_probs - exit2_thresholds.reshape(1, -1)), axis=1
        ),
    }


def frozen_stop_mask(
    diagnostics: dict[str, np.ndarray], stop_rule: dict[str, Any]
) -> np.ndarray:
    mask = np.ones(len(diagnostics["agreement"]), dtype=bool)
    if bool(stop_rule["require_exit1_exit2_agreement"]):
        mask &= diagnostics["agreement"]
    if not bool(stop_rule["allow_empty_stop"]):
        mask &= diagnostics["non_empty"]
    mask &= diagnostics["mean_binary_confidence"] >= float(
        stop_rule["confidence_threshold"]
    )
    mask &= diagnostics["min_decision_margin"] >= float(
        stop_rule["margin_threshold"]
    )
    return mask


def subset_state(state: AnytimeExitState, indices: torch.Tensor) -> AnytimeExitState:
    previous_hint = (
        None
        if state.prev_hint is None
        else state.prev_hint.index_select(0, indices)
    )
    return AnytimeExitState(
        feature_map=state.feature_map.index_select(0, indices),
        block_index=int(state.block_index),
        next_exit_index=int(state.next_exit_index),
        prev_hint=previous_hint,
        finished=bool(state.finished),
    )


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_score": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "avg_true_labels": float(y_true.sum(axis=1).mean()),
        "avg_pred_labels": float(y_pred.sum(axis=1).mean()),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run genuine staged Dynamic Early-Exit on corrected holdout."
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy_json", required=True, type=Path)
    parser.add_argument("--holdout_manifest", required=True, type=Path)
    parser.add_argument("--features_root", required=True, type=Path)
    parser.add_argument("--labels_json", required=True, type=Path)
    parser.add_argument("--lats_config_json", required=True, type=Path)
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--skip_parent_eval", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else run_dir / "ckpt" / "best.pt"
    )

    required = [
        run_dir,
        checkpoint,
        args.policy_json,
        args.holdout_manifest,
        args.features_root,
        args.labels_json,
        args.lats_config_json,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    cfg = load_run_config(run_dir)
    labels = load_labels(args.labels_json.resolve(), cfg)
    policy = load_json(args.policy_json.resolve())
    if list(policy.get("labels", [])) != labels:
        raise RuntimeError("Label order mismatch between frozen policy and labels JSON.")

    architecture = policy["architecture"]
    if int(architecture["num_exits"]) != 3:
        raise RuntimeError("This evaluator requires a frozen three-exit policy.")
    if bool(architecture.get("exit1_stopping_enabled", False)):
        raise RuntimeError("This v0.11 evaluator supports Exit 2 or Exit 3 only.")

    tap_blocks = parse_tap_blocks(architecture["tap_blocks"])
    n_mels = int(architecture["n_mels"])
    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=resolve_model_cfg(cfg),
    ).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()

    anytime_model = AnytimeExitNet(model).to(args.device)
    anytime_model.eval()

    threshold1 = thresholds_from_policy(policy, labels, 1)
    threshold2 = thresholds_from_policy(policy, labels, 2)
    threshold3 = thresholds_from_policy(policy, labels, 3)
    stop_rule = policy["stop_rule"]

    frame = pd.read_csv(args.holdout_manifest, low_memory=False)
    required_columns = ["feat_relpath", args.parent_id_col, *labels]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Holdout manifest is missing columns: {missing}")

    n_samples = len(frame)
    n_labels = len(labels)
    selected_probs = np.zeros((n_samples, n_labels), dtype=np.float32)
    selected_pred = np.zeros((n_samples, n_labels), dtype=np.int8)
    selected_exit = np.full(n_samples, 3, dtype=np.int8)
    agreement_values = np.zeros(n_samples, dtype=bool)
    confidence_values = np.zeros(n_samples, dtype=np.float32)
    margin_values = np.zeros(n_samples, dtype=np.float32)
    non_empty_values = np.zeros(n_samples, dtype=bool)

    stage12_seconds = 0.0
    stage3_seconds = 0.0
    frames_observed: int | None = None

    for start in range(0, n_samples, int(args.batch_size)):
        batch = frame.iloc[start : start + int(args.batch_size)]
        tensors = []
        for _, row in batch.iterrows():
            relative = Path(str(row["feat_relpath"]).replace("\\", "/"))
            tensors.append(load_feature(args.features_root.resolve() / relative))

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise RuntimeError(
                f"Batch beginning at row {start} has inconsistent shapes: {sorted(shapes)}"
            )

        x = torch.cat(tensors, dim=0).to(args.device)
        if frames_observed is None:
            frames_observed = int(x.shape[-1])

        synchronize(args.device)
        t0 = time.perf_counter()
        exit1_logits, state1 = anytime_model.start(x)
        exit2_logits, state2 = anytime_model.continue_from(state1)
        synchronize(args.device)
        stage12_seconds += time.perf_counter() - t0

        exit1_probs = torch.sigmoid(exit1_logits).detach().cpu().numpy()
        exit2_probs = torch.sigmoid(exit2_logits).detach().cpu().numpy()
        diagnostics = stop_diagnostics(
            exit1_probs,
            exit2_probs,
            threshold1,
            threshold2,
        )
        stop_mask = frozen_stop_mask(diagnostics, stop_rule)
        continue_mask = ~stop_mask

        local_count = len(batch)
        global_indices = np.arange(start, start + local_count)
        agreement_values[global_indices] = diagnostics["agreement"]
        confidence_values[global_indices] = diagnostics["mean_binary_confidence"]
        margin_values[global_indices] = diagnostics["min_decision_margin"]
        non_empty_values[global_indices] = diagnostics["non_empty"]

        stopped_global = global_indices[stop_mask]
        if len(stopped_global):
            selected_probs[stopped_global] = exit2_probs[stop_mask]
            selected_pred[stopped_global] = diagnostics["exit2_pred"][stop_mask]
            selected_exit[stopped_global] = 2

        continuing_local = np.flatnonzero(continue_mask)
        continuing_global = global_indices[continue_mask]
        if len(continuing_local):
            index_tensor = torch.as_tensor(
                continuing_local,
                dtype=torch.long,
                device=state2.feature_map.device,
            )
            continuing_state = subset_state(state2, index_tensor)

            synchronize(args.device)
            t1 = time.perf_counter()
            exit3_logits, final_state = anytime_model.continue_from(continuing_state)
            synchronize(args.device)
            stage3_seconds += time.perf_counter() - t1

            if not final_state.finished:
                raise RuntimeError("Expected Exit 3 to finish the staged state.")

            exit3_probs = torch.sigmoid(exit3_logits).detach().cpu().numpy()
            selected_probs[continuing_global] = exit3_probs
            selected_pred[continuing_global] = label_predictions(exit3_probs, threshold3)
            selected_exit[continuing_global] = 3

    if frames_observed is None:
        raise RuntimeError("No holdout samples were processed.")

    y_true = frame[labels].astype(int).to_numpy()
    segment_result = multilabel_metrics(y_true, selected_pred)
    exit2_count = int(np.sum(selected_exit == 2))
    exit3_count = int(np.sum(selected_exit == 3))
    exit2_fraction = float(exit2_count / max(n_samples, 1))
    average_exit_depth = float(np.mean(selected_exit))

    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames_observed,
        num_classes=n_labels,
        tap_blocks=tap_blocks,
    )
    exit2_flops = float(flops["exit2"])
    exit3_flops = float(flops["exit3"])
    policy_total_flops = float(exit2_count * exit2_flops + exit3_count * exit3_flops)
    full_total_flops = float(n_samples * exit3_flops)
    estimated_flops_saved_pct = float(
        100 * (1 - policy_total_flops / max(full_total_flops, 1))
    )

    model_seconds = float(stage12_seconds + stage3_seconds)
    runtime = {
        "stage1_to_exit2_seconds": stage12_seconds,
        "exit3_continuation_seconds": stage3_seconds,
        "total_model_inference_seconds": model_seconds,
        "model_latency_per_segment_ms": float(1000 * model_seconds / max(n_samples, 1)),
        "model_throughput_segments_per_second": float(
            n_samples / max(model_seconds, 1e-12)
        ),
        "timing_scope": "Model inference only; file loading and CSV writing excluded.",
        "device": args.device,
    }

    output_frame = frame[[args.parent_id_col, "feat_relpath", *labels]].copy()
    for idx, label in enumerate(labels):
        output_frame[f"dynamic_prob_{label}"] = selected_probs[:, idx]
        output_frame[f"dynamic_pred_{label}"] = selected_pred[:, idx]
    output_frame["selected_exit"] = selected_exit
    output_frame["stop_reason"] = np.where(
        selected_exit == 2, "reliable_early_exit", "final_exit"
    )
    output_frame["exit1_exit2_label_set_agreement"] = agreement_values
    output_frame["exit2_non_empty"] = non_empty_values
    output_frame["exit2_mean_binary_confidence"] = confidence_values
    output_frame["exit2_min_decision_margin"] = margin_values

    segment_csv = out_dir / "v011_dynamic_segment_predictions.csv"
    output_frame.to_csv(segment_csv, index=False)

    parent_summary = None
    if not args.skip_parent_eval:
        evaluator = PROJECT_ROOT / "scripts" / "v0.10" / "evaluate_frozen_lats_config_v010.py"
        if not evaluator.exists():
            raise FileNotFoundError(f"Frozen LATS evaluator not found: {evaluator}")
        parent_out = out_dir / "parent_frozen_lats_v2"
        command = [
            sys.executable,
            str(evaluator),
            "--segment-pred-csv",
            str(segment_csv),
            "--labels-json",
            str(args.labels_json.resolve()),
            "--config-json",
            str(args.lats_config_json.resolve()),
            "--out-dir",
            str(parent_out),
            "--parent-id-col",
            str(args.parent_id_col),
            "--prob-prefix",
            "dynamic_prob_",
            "--model-name",
            "v0.11_EE_genuine_dynamic_exit2_or_exit3",
        ]
        subprocess.run(command, check=True)
        parent_csv = parent_out / "v010_frozen_lats_eval.csv"
        parent_summary = pd.read_csv(parent_csv).iloc[0].to_dict()

    summary = {
        "experiment": "v0.11_EE_genuine_dynamic_exit2_or_exit3",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "frozen_policy_json": str(args.policy_json.resolve()),
        "holdout_manifest": str(args.holdout_manifest.resolve()),
        "features_root": str(args.features_root.resolve()),
        "n_segments": n_samples,
        "exit_distribution": {
            "exit2_samples": exit2_count,
            "exit3_samples": exit3_count,
            "exit2_fraction": exit2_fraction,
            "exit3_fraction": float(exit3_count / max(n_samples, 1)),
            "average_exit_depth": average_exit_depth,
        },
        "compute": {
            "frames_observed": frames_observed,
            "estimated_flops_by_exit": {k: float(v) for k, v in flops.items()},
            "policy_total_flops": policy_total_flops,
            "always_exit3_total_flops": full_total_flops,
            "estimated_flops_saved_pct": estimated_flops_saved_pct,
            "genuine_skipping_statement": "Only Exit-3 samples executed Blocks 4-5.",
        },
        "runtime": runtime,
        "segment_metrics": segment_result,
        "parent_frozen_lats_v2_metrics": parent_summary,
        "stop_rule": policy["stop_rule"],
        "stop_reasons": {
            "exit2": "reliable_early_exit",
            "exit3": "final_exit",
        },
        "important_note": "The frozen validation policy was used without holdout retuning.",
    }

    summary_json = out_dir / "v011_dynamic_runtime_summary.json"
    save_json(summary, summary_json)

    summary_row = {
        "n_segments": n_samples,
        "exit2_samples": exit2_count,
        "exit3_samples": exit3_count,
        "exit2_fraction": exit2_fraction,
        "average_exit_depth": average_exit_depth,
        "estimated_flops_saved_pct": estimated_flops_saved_pct,
        "model_latency_per_segment_ms": runtime["model_latency_per_segment_ms"],
        **{f"segment_{k}": v for k, v in segment_result.items()},
    }
    if parent_summary:
        for key, value in parent_summary.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                summary_row[f"parent_{key}"] = value
    summary_csv = out_dir / "v011_dynamic_runtime_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    print("\nV0.11 genuine Dynamic Early-Exit evaluation complete")
    print("-" * 100)
    print(f"Segments:                     {n_samples}")
    print(f"Stopped at Exit 2:            {exit2_count} ({exit2_fraction:.2%})")
    print(f"Continued to Exit 3:          {exit3_count}")
    print(f"Average exit depth:           {average_exit_depth:.4f}")
    print(f"Estimated FLOPs saved:        {estimated_flops_saved_pct:.2f}%")
    print(f"Model latency / segment:      {runtime['model_latency_per_segment_ms']:.4f} ms")
    print(f"Segment Macro-F1:             {segment_result['macro_f1']:.6f}")
    if parent_summary:
        print(f"Parent frozen-LATS Macro-F1:  {float(parent_summary['macro_f1']):.6f}")
    print(f"Predictions:                  {segment_csv}")
    print(f"Summary JSON:                 {summary_json}")
    print(f"Summary CSV:                  {summary_csv}")


if __name__ == "__main__":
    main()
