#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate frozen v0.14 parent-aware gates with genuine staged execution."""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v014 import (
    jsonable,
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
from policies.parent_aware_adaptive_gate import (
    adaptive_label_stop_mask,
    build_parent_aware_features,
    counterfactual_parent_unsafe_targets,
    label_predictions,
    parse_lats_rules,
    predict_multilabel_unsafe_probabilities,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


def subset_state(state: AnytimeExitState, indices: torch.Tensor) -> AnytimeExitState:
    previous_hint = (
        None if state.prev_hint is None else state.prev_hint.index_select(0, indices)
    )
    return AnytimeExitState(
        feature_map=state.feature_map.index_select(0, indices),
        block_index=int(state.block_index),
        next_exit_index=int(state.next_exit_index),
        prev_hint=previous_hint,
        finished=bool(state.finished),
    )


def thresholds_from_policy(policy: dict[str, Any], labels: list[str], exit_no: int) -> np.ndarray:
    mapping = policy["thresholds_by_exit"][f"exit{exit_no}"]
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def parent_complete_batches(
    frame: pd.DataFrame,
    *,
    parent_id_col: str,
    batch_size: int,
) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    current: list[int] = []
    for _, group in frame.groupby(parent_id_col, sort=False):
        indices = group.index.to_list()
        if current and len(current) + len(indices) > int(batch_size):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
        current.extend(indices)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def load_all_features(frame: pd.DataFrame, features_root: Path) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for relative_value in frame["feat_relpath"].astype(str):
        relative = Path(relative_value.replace("\\", "/"))
        tensors.append(load_feature(features_root / relative))
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise RuntimeError(f"Holdout features have inconsistent shapes: {sorted(shapes)}")
    return tensors


def run_always_exit3(
    *,
    anytime_model: AnytimeExitNet,
    tensors: list[torch.Tensor],
    batches: list[np.ndarray],
    device: str,
    collect: bool,
    num_labels: int,
) -> tuple[dict[str, np.ndarray] | None, float]:
    outputs = None
    if collect:
        outputs = {
            "p1": np.zeros((len(tensors), num_labels), dtype=np.float32),
            "p2": np.zeros((len(tensors), num_labels), dtype=np.float32),
            "p3": np.zeros((len(tensors), num_labels), dtype=np.float32),
        }
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for indices in batches:
            x = torch.cat([tensors[int(idx)] for idx in indices], dim=0).to(device)
            logits1, state1 = anytime_model.start(x)
            logits2, state2 = anytime_model.continue_from(state1)
            logits3, state3 = anytime_model.continue_from(state2)
            if not state3.finished:
                raise RuntimeError("Always-Exit3 staged state did not finish.")
            if outputs is not None:
                outputs["p1"][indices] = torch.sigmoid(logits1).cpu().numpy()
                outputs["p2"][indices] = torch.sigmoid(logits2).cpu().numpy()
                outputs["p3"][indices] = torch.sigmoid(logits3).cpu().numpy()
    synchronize(device)
    return outputs, float(time.perf_counter() - started)


def run_adaptive_transition(
    *,
    anytime_model: AnytimeExitNet,
    tensors: list[torch.Tensor],
    batches: list[np.ndarray],
    frame: pd.DataFrame,
    parent_id_col: str,
    transition_policy: dict[str, Any],
    gate_bundle: dict[str, Any],
    rules,
    exit_thresholds: list[np.ndarray],
    device: str,
    collect: bool,
    labels: list[str],
) -> tuple[dict[str, np.ndarray] | None, dict[str, float]]:
    source_exit = int(transition_policy["source_exit"])
    if source_exit not in (1, 2):
        raise ValueError(f"Unsupported source exit: {source_exit}")
    parameters = transition_policy["parameters"]
    label_thresholds = np.asarray(
        [float(parameters["label_probability_thresholds"][label]) for label in labels],
        dtype=np.float32,
    )
    expected_harm_threshold = parameters.get("expected_harm_threshold")
    models = gate_bundle["models"]
    num_labels = len(labels)

    output = None
    if collect:
        output = {
            "selected_probabilities": np.zeros(
                (len(tensors), num_labels), dtype=np.float32
            ),
            "selected_exit": np.full(len(tensors), 3, dtype=np.int8),
            "unsafe_probabilities": np.zeros(
                (len(tensors), num_labels), dtype=np.float32
            ),
            "expected_harm": np.zeros(len(tensors), dtype=np.float32),
            "highest_risk_label": np.zeros(len(tensors), dtype=np.int64),
            "stop_mask": np.zeros(len(tensors), dtype=bool),
        }

    model_seconds = 0.0
    policy_seconds = 0.0
    with torch.no_grad():
        for indices in batches:
            x = torch.cat([tensors[int(idx)] for idx in indices], dim=0).to(device)
            parent_ids = frame.loc[indices, parent_id_col].astype(str).to_numpy()

            synchronize(device)
            model_start = time.perf_counter()
            logits1, state1 = anytime_model.start(x)
            p1 = torch.sigmoid(logits1).cpu().numpy().astype(np.float32)
            if source_exit == 2:
                logits2, source_state = anytime_model.continue_from(state1)
                source_probs = torch.sigmoid(logits2).cpu().numpy().astype(np.float32)
                previous_probs = p1
            else:
                source_state = state1
                source_probs = p1
                previous_probs = None
            synchronize(device)
            model_seconds += time.perf_counter() - model_start

            policy_start = time.perf_counter()
            features, _, diagnostics = build_parent_aware_features(
                current_probabilities=source_probs,
                previous_probabilities=previous_probs,
                parent_ids=parent_ids,
                current_thresholds=exit_thresholds[source_exit - 1],
                rules=rules,
            )
            unsafe_probs = predict_multilabel_unsafe_probabilities(models, features)
            stop_mask, expected_harm, highest_risk = adaptive_label_stop_mask(
                unsafe_probabilities=unsafe_probs,
                label_thresholds=label_thresholds,
                expected_harm_threshold=expected_harm_threshold,
                non_empty=diagnostics["non_empty"],
                allow_empty_stop=False,
            )
            policy_seconds += time.perf_counter() - policy_start

            continuing_local = np.flatnonzero(~stop_mask)
            if output is not None:
                output["unsafe_probabilities"][indices] = unsafe_probs
                output["expected_harm"][indices] = expected_harm
                output["highest_risk_label"][indices] = highest_risk
                output["stop_mask"][indices] = stop_mask
                stopped_global = indices[stop_mask]
                output["selected_probabilities"][stopped_global] = source_probs[stop_mask]
                output["selected_exit"][stopped_global] = source_exit

            if len(continuing_local) > 0:
                local_tensor = torch.as_tensor(
                    continuing_local,
                    dtype=torch.long,
                    device=source_state.feature_map.device,
                )
                continuing_state = subset_state(source_state, local_tensor)
                synchronize(device)
                continuation_start = time.perf_counter()
                if source_exit == 1:
                    _, continuing_state = anytime_model.continue_from(continuing_state)
                logits3, final_state = anytime_model.continue_from(continuing_state)
                synchronize(device)
                model_seconds += time.perf_counter() - continuation_start
                if not final_state.finished:
                    raise RuntimeError("Adaptive continuation did not reach Exit 3.")
                if output is not None:
                    continuing_global = indices[~stop_mask]
                    output["selected_probabilities"][continuing_global] = (
                        torch.sigmoid(logits3).cpu().numpy().astype(np.float32)
                    )
                    output["selected_exit"][continuing_global] = 3

    return output, {
        "model_seconds": float(model_seconds),
        "policy_seconds": float(policy_seconds),
        "total_seconds": float(model_seconds + policy_seconds),
    }


def evaluate_parent_lats(
    *,
    segment_csv: Path,
    labels_json: Path,
    lats_config_json: Path,
    out_dir: Path,
    parent_id_col: str,
    model_name: str,
) -> dict[str, Any]:
    evaluator = (
        PROJECT_ROOT / "scripts" / "v0.10" / "evaluate_frozen_lats_config_v010.py"
    )
    command = [
        sys.executable,
        str(evaluator),
        "--segment-pred-csv",
        str(segment_csv),
        "--labels-json",
        str(labels_json),
        "--config-json",
        str(lats_config_json),
        "--out-dir",
        str(out_dir),
        "--parent-id-col",
        parent_id_col,
        "--prob-prefix",
        "dynamic_prob_",
        "--model-name",
        model_name,
    ]
    subprocess.run(command, check=True)
    return pd.read_csv(out_dir / "v010_frozen_lats_eval.csv").iloc[0].to_dict()


def quartiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_seconds": float(np.median(array)),
        "q1_seconds": float(np.quantile(array, 0.25)),
        "q3_seconds": float(np.quantile(array, 0.75)),
        "iqr_seconds": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "mean_seconds": float(np.mean(array)),
        "std_seconds": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "repeats": int(len(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate parent-aware adaptive Exit-1 and Exit-2 gates."
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
    parser.add_argument("--timing_repeats", type=int, default=10)
    parser.add_argument("--timing_seed", type=int, default=42)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else run_dir / "ckpt" / "best.pt"
    required = [
        checkpoint,
        args.policy_json.resolve(),
        args.holdout_manifest.resolve(),
        args.features_root.resolve(),
        args.labels_json.resolve(),
        args.lats_config_json.resolve(),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    cfg = load_run_config(run_dir)
    labels = load_labels(args.labels_json.resolve(), cfg)
    policy = load_json(args.policy_json.resolve())
    if policy.get("experiment") != "v0.14_EE_parent_aware_cross_validated_adaptive_gate":
        raise RuntimeError("The supplied policy is not a v0.14 parent-aware gate.")
    if list(policy.get("labels", [])) != labels:
        raise RuntimeError("Frozen policy and label schema order differ.")

    tap_blocks = parse_tap_blocks(policy["architecture"]["tap_blocks"])
    n_mels = int(policy["architecture"]["n_mels"])
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

    exit_thresholds = [
        thresholds_from_policy(policy, labels, exit_no) for exit_no in (1, 2, 3)
    ]
    rules = parse_lats_rules(load_json(args.lats_config_json.resolve()), labels)
    frame = pd.read_csv(args.holdout_manifest.resolve(), low_memory=False).reset_index(drop=True)
    missing = [
        column
        for column in ["feat_relpath", args.parent_id_col, *labels]
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(f"Holdout manifest is missing columns: {missing}")
    tensors = load_all_features(frame, args.features_root.resolve())
    batches = parent_complete_batches(
        frame,
        parent_id_col=args.parent_id_col,
        batch_size=args.batch_size,
    )
    frames_observed = int(tensors[0].shape[-1])
    y_true = frame[labels].astype(int).to_numpy()
    parent_ids = frame[args.parent_id_col].astype(str).to_numpy()

    warm_indices = batches[0]
    with torch.no_grad():
        warm_x = torch.cat([tensors[int(idx)] for idx in warm_indices], dim=0).to(args.device)
        _, warm_state1 = anytime_model.start(warm_x)
        _, warm_state2 = anytime_model.continue_from(warm_state1)
        _, _ = anytime_model.continue_from(warm_state2)
    synchronize(args.device)

    always_outputs, _ = run_always_exit3(
        anytime_model=anytime_model,
        tensors=tensors,
        batches=batches,
        device=args.device,
        collect=True,
        num_labels=len(labels),
    )
    assert always_outputs is not None

    methods = ["always_exit3", *policy["selected_policies"].keys()]
    timing_values: dict[str, list[float]] = {method: [] for method in methods}
    rng = random.Random(int(args.timing_seed))
    for _ in range(int(args.timing_repeats)):
        order = methods.copy()
        rng.shuffle(order)
        for method in order:
            if method == "always_exit3":
                _, seconds = run_always_exit3(
                    anytime_model=anytime_model,
                    tensors=tensors,
                    batches=batches,
                    device=args.device,
                    collect=False,
                    num_labels=len(labels),
                )
            else:
                transition_policy = policy["selected_policies"][method]
                bundle = joblib.load(
                    args.policy_json.resolve().parent
                    / transition_policy["gate_model_filename"]
                )
                _, timing = run_adaptive_transition(
                    anytime_model=anytime_model,
                    tensors=tensors,
                    batches=batches,
                    frame=frame,
                    parent_id_col=args.parent_id_col,
                    transition_policy=transition_policy,
                    gate_bundle=bundle,
                    rules=rules,
                    exit_thresholds=exit_thresholds,
                    device=args.device,
                    collect=False,
                    labels=labels,
                )
                seconds = timing["total_seconds"]
            timing_values[method].append(float(seconds))

    timing_summary = {method: quartiles(values) for method, values in timing_values.items()}
    baseline_median = timing_summary["always_exit3"]["median_seconds"]

    comparison_rows: list[dict[str, Any]] = []
    all_method_summaries: dict[str, Any] = {}
    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames_observed,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )

    method_outputs: dict[str, dict[str, np.ndarray]] = {
        "always_exit3": {
            "selected_probabilities": always_outputs["p3"],
            "selected_exit": np.full(len(frame), 3, dtype=np.int8),
        }
    }
    for method, transition_policy in policy["selected_policies"].items():
        bundle = joblib.load(
            args.policy_json.resolve().parent
            / transition_policy["gate_model_filename"]
        )
        output, single_timing = run_adaptive_transition(
            anytime_model=anytime_model,
            tensors=tensors,
            batches=batches,
            frame=frame,
            parent_id_col=args.parent_id_col,
            transition_policy=transition_policy,
            gate_bundle=bundle,
            rules=rules,
            exit_thresholds=exit_thresholds,
            device=args.device,
            collect=True,
            labels=labels,
        )
        assert output is not None
        output["single_pass_model_seconds"] = np.asarray(
            [single_timing["model_seconds"]], dtype=np.float64
        )
        output["single_pass_policy_seconds"] = np.asarray(
            [single_timing["policy_seconds"]], dtype=np.float64
        )
        method_outputs[method] = output

    for method, output in method_outputs.items():
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        selected_probs = output["selected_probabilities"]
        selected_exit = output["selected_exit"]
        selected_pred = np.zeros_like(y_true, dtype=np.int8)
        for exit_no in (1, 2, 3):
            mask = selected_exit == exit_no
            if np.any(mask):
                selected_pred[mask] = label_predictions(
                    selected_probs[mask], exit_thresholds[exit_no - 1]
                )
        segment_metrics = multilabel_metrics(y_true, selected_pred)

        segment_frame = frame[[args.parent_id_col, "feat_relpath", *labels]].copy()
        for label_idx, label in enumerate(labels):
            segment_frame[f"dynamic_prob_{label}"] = selected_probs[:, label_idx]
            segment_frame[f"dynamic_pred_{label}"] = selected_pred[:, label_idx]
        segment_frame["selected_exit"] = selected_exit

        if method != "always_exit3":
            transition_policy = policy["selected_policies"][method]
            source_exit = int(transition_policy["source_exit"])
            source_probs = always_outputs[f"p{source_exit}"]
            unsafe_true, _, _ = counterfactual_parent_unsafe_targets(
                y_true=y_true,
                source_probabilities=source_probs,
                deeper_probabilities=always_outputs["p3"],
                parent_ids=parent_ids,
                rules=rules,
            )
            unsafe_probs = output["unsafe_probabilities"]
            label_threshold_map = transition_policy["parameters"][
                "label_probability_thresholds"
            ]
            label_thresholds = np.asarray(
                [float(label_threshold_map[label]) for label in labels],
                dtype=np.float32,
            )
            triggered = unsafe_probs >= label_thresholds.reshape(1, -1)
            for label_idx, label in enumerate(labels):
                segment_frame[f"unsafe_probability_{label}"] = unsafe_probs[:, label_idx]
                segment_frame[f"unsafe_threshold_{label}"] = label_thresholds[label_idx]
                segment_frame[f"parent_counterfactual_unsafe_{label}"] = unsafe_true[:, label_idx]
            highest = output["highest_risk_label"]
            segment_frame["highest_risk_label"] = [labels[int(idx)] for idx in highest]
            segment_frame["expected_parent_harm"] = output["expected_harm"]
            reasons: list[str] = []
            for row_idx in range(len(frame)):
                if bool(output["stop_mask"][row_idx]):
                    reasons.append(f"stopped_at_exit{source_exit}")
                    continue
                active = [
                    labels[label_idx]
                    for label_idx in np.flatnonzero(triggered[row_idx])
                ]
                reason_parts = []
                if active:
                    reason_parts.append("unsafe_labels=" + "|".join(active))
                harm_threshold = transition_policy["parameters"].get(
                    "expected_harm_threshold"
                )
                if (
                    harm_threshold is not None
                    and float(output["expected_harm"][row_idx])
                    >= float(harm_threshold)
                ):
                    reason_parts.append("expected_harm_threshold")
                reasons.append(";".join(reason_parts) or "adaptive_gate_continue")
            segment_frame["continuation_reason"] = reasons
        else:
            segment_frame["continuation_reason"] = "always_exit3"

        segment_csv = method_dir / "segment_predictions.csv"
        segment_frame.to_csv(segment_csv, index=False)
        parent_metrics = evaluate_parent_lats(
            segment_csv=segment_csv,
            labels_json=args.labels_json.resolve(),
            lats_config_json=args.lats_config_json.resolve(),
            out_dir=method_dir / "parent_frozen_lats_v2",
            parent_id_col=args.parent_id_col,
            model_name=method,
        )

        source_exit = 3 if method == "always_exit3" else int(
            policy["selected_policies"][method]["source_exit"]
        )
        source_count = int(np.sum(selected_exit == source_exit)) if source_exit < 3 else 0
        source_fraction = float(source_count / max(len(frame), 1))
        average_flops = float(
            source_fraction * float(flops[f"exit{source_exit}"])
            + (1.0 - source_fraction) * float(flops["exit3"])
        ) if source_exit < 3 else float(flops["exit3"])
        flops_saved = float(
            100.0 * (1.0 - average_flops / max(float(flops["exit3"]), 1.0))
        )
        timing = timing_summary[method]
        speedup = float(baseline_median / max(timing["median_seconds"], 1e-12))
        row = {
            "method": method,
            "source_exit": source_exit,
            "source_exit_fraction": source_fraction,
            "average_exit_depth": float(np.mean(selected_exit)),
            "estimated_flops_saved_pct": flops_saved,
            "latency_median_per_segment_ms": float(
                1000.0 * timing["median_seconds"] / len(frame)
            ),
            "latency_iqr_per_segment_ms": float(
                1000.0 * timing["iqr_seconds"] / len(frame)
            ),
            "measured_speedup_vs_always_exit3": speedup,
            **{f"segment_{key}": value for key, value in segment_metrics.items()},
            **{
                f"parent_{key}": value
                for key, value in parent_metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            },
        }
        comparison_rows.append(row)
        method_summary = {
            **row,
            "timing": timing,
            "selected_policy": (
                None if method == "always_exit3" else policy["selected_policies"][method]
            ),
            "genuine_skipping_statement": (
                "Stopped samples did not execute deeper blocks during adaptive timing."
            ),
            "audit_note": (
                "All-Exit3 probabilities were generated separately for diagnostic "
                "counterfactual labels and were not available to the runtime gate."
            ),
        }
        save_json(method_summary, method_dir / "runtime_summary.json")
        all_method_summaries[method] = method_summary

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = out_dir / "v014_parent_aware_holdout_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    save_json(
        {
            "experiment": "v0.14_EE_parent_aware_cross_validated_adaptive_gate",
            "comparison": [jsonable(row) for row in comparison_rows],
            "methods": all_method_summaries,
            "important_note": (
                "The holdout used frozen OOF-selected label-specific thresholds. "
                "No threshold or gate parameter was changed after holdout access."
            ),
        },
        out_dir / "v014_parent_aware_holdout_comparison.json",
    )

    display_columns = [
        "method",
        "source_exit_fraction",
        "estimated_flops_saved_pct",
        "latency_median_per_segment_ms",
        "measured_speedup_vs_always_exit3",
        "segment_macro_f1",
        "parent_macro_f1",
        "parent_micro_f1",
        "parent_exact_match",
        "parent_hamming_loss",
    ]
    print("\nV0.14 parent-aware holdout comparison complete")
    print("-" * 150)
    print(comparison[display_columns].to_string(index=False))
    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
