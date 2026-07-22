#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the frozen v0.12 label-aware policy with genuine staged computation.

Every sample executes through Exit 2. Only samples rejected by the frozen
validation-derived policy execute Blocks 4-5 and Exit 3.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v012 import (
    load_checkpoint,
    load_feature,
    load_json,
    load_labels,
    load_run_config,
    multilabel_metrics,
    parse_tap_blocks,
    resolve_model_cfg,
    save_json,
    synchronize,
)
from models.anytime_exit_net import AnytimeExitNet, AnytimeExitState
from policies.label_aware_early_exit_policy import (
    LabelAwarePolicyConfig,
    LabelRiskProfile,
    compute_label_aware_diagnostics,
    label_aware_stop_mask,
    label_predictions,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


def thresholds_from_policy(
    policy: dict,
    labels: list[str],
    exit_no: int,
) -> np.ndarray:
    mapping = policy["thresholds_by_exit"][f"exit{exit_no}"]
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(
            f"Frozen policy Exit {exit_no} thresholds miss labels: {missing}"
        )
    return np.asarray(
        [float(mapping[label]) for label in labels],
        dtype=np.float32,
    )


def subset_state(
    state: AnytimeExitState,
    indices: torch.Tensor,
) -> AnytimeExitState:
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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run genuine staged v0.12 label-aware Dynamic Early-Exit."
        )
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
        if args.checkpoint is not None
        else run_dir / "ckpt" / "best.pt"
    )

    required_paths = [
        run_dir,
        checkpoint,
        args.policy_json.resolve(),
        args.holdout_manifest.resolve(),
        args.features_root.resolve(),
        args.labels_json.resolve(),
        args.lats_config_json.resolve(),
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    cfg = load_run_config(run_dir)
    labels = load_labels(args.labels_json.resolve(), cfg)
    policy = load_json(args.policy_json.resolve())

    if list(policy.get("labels", [])) != labels:
        raise RuntimeError(
            "Label order mismatch between frozen policy and labels JSON."
        )
    if policy.get("experiment") != (
        "v0.12_EE_validation_derived_label_aware_policy"
    ):
        raise RuntimeError(
            "The supplied policy JSON is not a v0.12 label-aware policy."
        )

    architecture = policy["architecture"]
    if int(architecture["num_exits"]) != 3:
        raise RuntimeError("This evaluator requires a three-exit policy.")
    if bool(architecture.get("exit1_stopping_enabled", False)):
        raise RuntimeError(
            "The first v0.12 label-aware policy supports Exit 2 or Exit 3 only."
        )

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

    risk_profile = LabelRiskProfile.from_dict(
        policy["label_risk_profile"]
    )
    if list(risk_profile.labels) != labels:
        raise RuntimeError(
            "Label order mismatch in frozen label-risk profile."
        )
    risk_weights = np.asarray(
        risk_profile.risk_weights,
        dtype=np.float32,
    )

    stop_rule = dict(policy["stop_rule"])
    stop_rule.pop("policy_type", None)
    stop_rule.pop("stop_reason_exit2", None)
    stop_rule.pop("stop_reason_exit3", None)
    policy_config = LabelAwarePolicyConfig.from_dict(stop_rule)

    frame = pd.read_csv(args.holdout_manifest.resolve(), low_memory=False)
    required_columns = ["feat_relpath", args.parent_id_col, *labels]
    missing = [
        column for column in required_columns
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"Holdout manifest is missing columns: {missing}"
        )

    n_samples = len(frame)
    n_labels = len(labels)
    selected_probs = np.zeros((n_samples, n_labels), dtype=np.float32)
    selected_pred = np.zeros((n_samples, n_labels), dtype=np.int8)
    selected_exit = np.full(n_samples, 3, dtype=np.int8)

    agreement_values = np.zeros(n_samples, dtype=bool)
    non_empty_values = np.zeros(n_samples, dtype=bool)
    confidence_values = np.zeros(n_samples, dtype=np.float32)
    margin_values = np.zeros(n_samples, dtype=np.float32)
    max_delta_values = np.zeros(n_samples, dtype=np.float32)
    max_risk_values = np.zeros(n_samples, dtype=np.float32)

    stage12_seconds = 0.0
    stage3_seconds = 0.0
    frames_observed: int | None = None

    for start in range(0, n_samples, int(args.batch_size)):
        batch = frame.iloc[start : start + int(args.batch_size)]
        tensors: list[torch.Tensor] = []
        for _, row in batch.iterrows():
            relative = Path(
                str(row["feat_relpath"]).replace("\\", "/")
            )
            tensors.append(
                load_feature(args.features_root.resolve() / relative)
            )

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise RuntimeError(
                f"Batch beginning at row {start} has inconsistent feature "
                f"shapes: {sorted(shapes)}"
            )

        x = torch.cat(tensors, dim=0).to(args.device)
        if frames_observed is None:
            frames_observed = int(x.shape[-1])

        synchronize(args.device)
        stage12_start = time.perf_counter()
        exit1_logits, state1 = anytime_model.start(x)
        exit2_logits, state2 = anytime_model.continue_from(state1)
        synchronize(args.device)
        stage12_seconds += time.perf_counter() - stage12_start

        exit1_probs = torch.sigmoid(
            exit1_logits
        ).detach().cpu().numpy()
        exit2_probs = torch.sigmoid(
            exit2_logits
        ).detach().cpu().numpy()

        diagnostics = compute_label_aware_diagnostics(
            exit1_probabilities=exit1_probs,
            exit2_probabilities=exit2_probs,
            exit1_thresholds=threshold1,
            exit2_thresholds=threshold2,
            risk_weights=risk_weights,
            margin_scale=policy_config.margin_scale,
            margin_weight=policy_config.margin_weight,
            delta_weight=policy_config.delta_weight,
        )
        stop_mask = label_aware_stop_mask(
            diagnostics,
            policy_config,
        )
        continue_mask = ~stop_mask

        local_count = len(batch)
        global_indices = np.arange(start, start + local_count)

        agreement_values[global_indices] = diagnostics[
            "label_set_agreement"
        ]
        non_empty_values[global_indices] = diagnostics["non_empty"]
        confidence_values[global_indices] = diagnostics[
            "mean_binary_confidence"
        ]
        margin_values[global_indices] = diagnostics[
            "minimum_decision_margin"
        ]
        max_delta_values[global_indices] = diagnostics[
            "maximum_probability_delta"
        ]
        max_risk_values[global_indices] = diagnostics[
            "maximum_label_risk"
        ]

        stopped_global = global_indices[stop_mask]
        if len(stopped_global) > 0:
            selected_probs[stopped_global] = exit2_probs[stop_mask]
            selected_pred[stopped_global] = diagnostics[
                "exit2_pred"
            ][stop_mask]
            selected_exit[stopped_global] = 2

        continuing_local = np.flatnonzero(continue_mask)
        continuing_global = global_indices[continue_mask]
        if len(continuing_local) > 0:
            index_tensor = torch.as_tensor(
                continuing_local,
                dtype=torch.long,
                device=state2.feature_map.device,
            )
            continuing_state = subset_state(state2, index_tensor)

            synchronize(args.device)
            stage3_start = time.perf_counter()
            exit3_logits, final_state = anytime_model.continue_from(
                continuing_state
            )
            synchronize(args.device)
            stage3_seconds += time.perf_counter() - stage3_start

            if not final_state.finished:
                raise RuntimeError(
                    "Expected Exit 3 to complete the staged state."
                )

            exit3_probs = torch.sigmoid(
                exit3_logits
            ).detach().cpu().numpy()
            selected_probs[continuing_global] = exit3_probs
            selected_pred[continuing_global] = label_predictions(
                exit3_probs,
                threshold3,
            )
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
    policy_total_flops = float(
        exit2_count * exit2_flops
        + exit3_count * exit3_flops
    )
    full_total_flops = float(n_samples * exit3_flops)
    estimated_flops_saved_pct = float(
        100.0
        * (
            1.0
            - policy_total_flops
            / max(full_total_flops, 1.0)
        )
    )

    model_seconds = float(stage12_seconds + stage3_seconds)
    runtime = {
        "stage1_to_exit2_seconds": stage12_seconds,
        "exit3_continuation_seconds": stage3_seconds,
        "total_model_inference_seconds": model_seconds,
        "model_latency_per_segment_ms": float(
            1000.0 * model_seconds / max(n_samples, 1)
        ),
        "model_throughput_segments_per_second": float(
            n_samples / max(model_seconds, 1e-12)
        ),
        "timing_scope": (
            "Model inference only; feature loading and CSV writing excluded."
        ),
        "device": args.device,
    }

    output_frame = frame[
        [args.parent_id_col, "feat_relpath", *labels]
    ].copy()
    for label_idx, label in enumerate(labels):
        output_frame[f"dynamic_prob_{label}"] = selected_probs[:, label_idx]
        output_frame[f"dynamic_pred_{label}"] = selected_pred[:, label_idx]

    output_frame["selected_exit"] = selected_exit
    output_frame["stop_reason"] = np.where(
        selected_exit == 2,
        "label_aware_reliable_early_exit",
        "final_exit",
    )
    output_frame["exit1_exit2_label_set_agreement"] = agreement_values
    output_frame["exit2_non_empty"] = non_empty_values
    output_frame["exit2_mean_binary_confidence"] = confidence_values
    output_frame["exit2_min_decision_margin"] = margin_values
    output_frame["exit1_exit2_max_probability_delta"] = max_delta_values
    output_frame["exit2_maximum_label_risk"] = max_risk_values

    segment_csv = (
        out_dir / "v012_label_aware_segment_predictions.csv"
    )
    output_frame.to_csv(segment_csv, index=False)

    parent_summary = None
    if not args.skip_parent_eval:
        evaluator = (
            PROJECT_ROOT
            / "scripts"
            / "v0.10"
            / "evaluate_frozen_lats_config_v010.py"
        )
        if not evaluator.exists():
            raise FileNotFoundError(
                f"Frozen LATS evaluator not found: {evaluator}"
            )

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
            "v0.12_EE_validation_derived_label_aware_policy",
        ]
        subprocess.run(command, check=True)
        parent_csv = parent_out / "v010_frozen_lats_eval.csv"
        parent_summary = pd.read_csv(parent_csv).iloc[0].to_dict()

    summary = {
        "experiment": "v0.12_EE_validation_derived_label_aware_policy",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "frozen_policy_json": str(args.policy_json.resolve()),
        "holdout_manifest": str(args.holdout_manifest.resolve()),
        "features_root": str(args.features_root.resolve()),
        "labels_json": str(args.labels_json.resolve()),
        "lats_config_json": str(args.lats_config_json.resolve()),
        "n_segments": n_samples,
        "exit_distribution": {
            "exit2_samples": exit2_count,
            "exit3_samples": exit3_count,
            "exit2_fraction": exit2_fraction,
            "exit3_fraction": float(
                exit3_count / max(n_samples, 1)
            ),
            "average_exit_depth": average_exit_depth,
        },
        "compute": {
            "frames_observed": frames_observed,
            "estimated_flops_by_exit": {
                key: float(value)
                for key, value in flops.items()
            },
            "policy_total_flops": policy_total_flops,
            "always_exit3_total_flops": full_total_flops,
            "estimated_flops_saved_pct": estimated_flops_saved_pct,
            "genuine_skipping_statement": (
                "Only samples assigned to Exit 3 executed Blocks 4-5."
            ),
        },
        "runtime": runtime,
        "segment_metrics": segment_result,
        "parent_frozen_lats_v2_metrics": parent_summary,
        "label_risk_profile": risk_profile.to_dict(),
        "stop_rule": policy["stop_rule"],
        "important_note": (
            "The corrected holdout was evaluated with the frozen "
            "validation-derived label-aware policy; no holdout retuning "
            "occurred."
        ),
    }

    summary_json = (
        out_dir / "v012_label_aware_runtime_summary.json"
    )
    save_json(summary, summary_json)

    summary_row = {
        "n_segments": n_samples,
        "exit2_samples": exit2_count,
        "exit3_samples": exit3_count,
        "exit2_fraction": exit2_fraction,
        "average_exit_depth": average_exit_depth,
        "estimated_flops_saved_pct": estimated_flops_saved_pct,
        "model_latency_per_segment_ms": runtime[
            "model_latency_per_segment_ms"
        ],
        **{
            f"segment_{key}": value
            for key, value in segment_result.items()
        },
    }
    if parent_summary is not None:
        for key, value in parent_summary.items():
            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                summary_row[f"parent_{key}"] = value

    summary_csv = (
        out_dir / "v012_label_aware_runtime_summary.csv"
    )
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    print("\nV0.12 genuine label-aware Dynamic Early-Exit complete")
    print("-" * 100)
    print(f"Segments:                     {n_samples}")
    print(
        f"Stopped at Exit 2:            "
        f"{exit2_count} ({exit2_fraction:.2%})"
    )
    print(f"Continued to Exit 3:          {exit3_count}")
    print(f"Average exit depth:           {average_exit_depth:.4f}")
    print(
        f"Estimated FLOPs saved:        "
        f"{estimated_flops_saved_pct:.2f}%"
    )
    print(
        "Model latency / segment:     "
        f"{runtime['model_latency_per_segment_ms']:.4f} ms"
    )
    print(
        f"Segment Macro-F1:             "
        f"{segment_result['macro_f1']:.6f}"
    )
    if parent_summary is not None:
        print(
            "Parent frozen-LATS Macro-F1: "
            f"{float(parent_summary['macro_f1']):.6f}"
        )
    print(f"Predictions:                  {segment_csv}")
    print(f"Summary JSON:                 {summary_json}")
    print(f"Summary CSV:                  {summary_csv}")


if __name__ == "__main__":
    main()
