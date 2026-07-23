#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate frozen v0.15 whole-parent selective risk control."""

from __future__ import annotations

import argparse
import random
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
V014_DIR = PROJECT_ROOT / "scripts" / "v0.14_EE" / "parent_aware_gate"
for path in (SCRIPT_DIR, PROJECT_ROOT, V014_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v015 import (
    jsonable,
    load_checkpoint,
    load_json,
    load_labels,
    load_run_config,
    multilabel_metrics,
    parse_tap_blocks,
    resolve_model_cfg,
    save_json,
    synchronize,
)
from evaluate_parent_aware_gate_v014 import (
    evaluate_parent_lats,
    load_all_features,
    parent_complete_batches,
    quartiles,
    run_always_exit3,
    subset_state,
    thresholds_from_policy,
)
from models.anytime_exit_net import AnytimeExitNet
from policies.parent_aware_adaptive_gate import (
    label_predictions,
    parse_lats_rules,
)
from policies.whole_parent_selective_exit import (
    build_whole_parent_features,
    predict_empirical_unsafe_probabilities,
    predict_shared_unsafe_probabilities,
    whole_parent_stop_mask,
    whole_parent_unsafe_targets,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


def _predict_unsafe_probabilities(
    *,
    bundle: dict[str, Any],
    parent_features: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    num_labels: int,
) -> np.ndarray:
    bundle_type = str(bundle["type"])
    if bundle_type == "shared_logistic_parent_gate":
        return predict_shared_unsafe_probabilities(
            bundle["model"],
            parent_features,
            num_labels,
        )
    if bundle_type == "nonparametric_parent_risk":
        return predict_empirical_unsafe_probabilities(
            bundle["calibrators"],
            diagnostics["raw_nonparametric_risk"],
        )
    raise ValueError(f"Unsupported v0.15 bundle type: {bundle_type}")


def run_whole_parent_transition(
    *,
    anytime_model: AnytimeExitNet,
    tensors: list[torch.Tensor],
    batches: list[np.ndarray],
    frame: pd.DataFrame,
    parent_id_col: str,
    selected_policy: dict[str, Any],
    bundle: dict[str, Any],
    rules,
    device: str,
    collect: bool,
    labels: list[str],
) -> tuple[dict[str, np.ndarray] | None, dict[str, float]]:
    parameters = selected_policy["parameters"]
    label_thresholds = np.asarray(
        [
            float(parameters["label_unsafe_probability_thresholds"][label])
            for label in labels
        ],
        dtype=np.float32,
    )
    expected_harm_threshold = parameters.get("expected_harm_threshold")
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
            "parent_stop_mask_by_row": np.zeros(len(tensors), dtype=bool),
        }

    model_seconds = 0.0
    policy_seconds = 0.0
    with torch.no_grad():
        for indices in batches:
            x = torch.cat(
                [tensors[int(index)] for index in indices],
                dim=0,
            ).to(device)
            batch_parent_ids = frame.loc[
                indices,
                parent_id_col,
            ].astype(str).to_numpy()

            synchronize(device)
            source_started = time.perf_counter()
            logits1, state1 = anytime_model.start(x)
            logits2, state2 = anytime_model.continue_from(state1)
            p1 = torch.sigmoid(logits1).cpu().numpy().astype(np.float32)
            p2 = torch.sigmoid(logits2).cpu().numpy().astype(np.float32)
            synchronize(device)
            model_seconds += time.perf_counter() - source_started

            policy_started = time.perf_counter()
            parent_features, _, diagnostics = build_whole_parent_features(
                current_probabilities=p2,
                previous_probabilities=p1,
                parent_ids=batch_parent_ids,
                rules=rules,
            )
            unsafe_probabilities = _predict_unsafe_probabilities(
                bundle=bundle,
                parent_features=parent_features,
                diagnostics=diagnostics,
                num_labels=num_labels,
            )
            parent_stop, expected_harm, highest_risk = whole_parent_stop_mask(
                unsafe_probabilities=unsafe_probabilities,
                label_thresholds=label_thresholds,
                expected_harm_threshold=expected_harm_threshold,
                non_empty=diagnostics["non_empty"],
                allow_empty_stop=False,
            )
            row_to_parent = diagnostics["row_to_parent"]
            row_stop = parent_stop[row_to_parent]
            policy_seconds += time.perf_counter() - policy_started

            if output is not None:
                output["unsafe_probabilities"][indices] = unsafe_probabilities[
                    row_to_parent
                ]
                output["expected_harm"][indices] = expected_harm[row_to_parent]
                output["highest_risk_label"][indices] = highest_risk[row_to_parent]
                output["parent_stop_mask_by_row"][indices] = row_stop
                stopped_global = indices[row_stop]
                output["selected_probabilities"][stopped_global] = p2[row_stop]
                output["selected_exit"][stopped_global] = 2

            continuing_local = np.flatnonzero(~row_stop)
            if len(continuing_local) > 0:
                local_tensor = torch.as_tensor(
                    continuing_local,
                    dtype=torch.long,
                    device=state2.feature_map.device,
                )
                continuing_state = subset_state(state2, local_tensor)
                synchronize(device)
                continuation_started = time.perf_counter()
                logits3, final_state = anytime_model.continue_from(
                    continuing_state
                )
                synchronize(device)
                model_seconds += time.perf_counter() - continuation_started
                if not final_state.finished:
                    raise RuntimeError(
                        "Whole-parent continuation did not reach Exit 3."
                    )
                if output is not None:
                    continuing_global = indices[~row_stop]
                    output["selected_probabilities"][continuing_global] = (
                        torch.sigmoid(logits3).cpu().numpy().astype(np.float32)
                    )
                    output["selected_exit"][continuing_global] = 3

    return output, {
        "model_seconds": float(model_seconds),
        "policy_seconds": float(policy_seconds),
        "total_seconds": float(model_seconds + policy_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen whole-parent Exit-2/Exit-3 policies."
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
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else run_dir / "ckpt" / "best.pt"
    )
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
    if policy.get("experiment") != "v0.15_EE_whole_parent_selective_risk_control":
        raise RuntimeError("The supplied policy is not a v0.15 whole-parent policy.")
    if list(policy.get("labels", [])) != labels:
        raise RuntimeError("Frozen policy and label schema order differ.")

    architecture = policy["architecture"]
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

    exit_thresholds = [
        thresholds_from_policy(policy, labels, exit_no)
        for exit_no in (1, 2, 3)
    ]
    rules = parse_lats_rules(
        load_json(args.lats_config_json.resolve()),
        labels,
    )
    frame = pd.read_csv(
        args.holdout_manifest.resolve(),
        low_memory=False,
    ).reset_index(drop=True)
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
        warm_x = torch.cat(
            [tensors[int(index)] for index in warm_indices],
            dim=0,
        ).to(args.device)
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

    selected_policies = policy["selected_policies"]
    bundles = {
        method: joblib.load(
            args.policy_json.resolve().parent
            / selected_policy["bundle_filename"]
        )
        for method, selected_policy in selected_policies.items()
    }
    methods = ["always_exit3", *selected_policies.keys()]
    timing_values: dict[str, list[float]] = {
        method: [] for method in methods
    }
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
                _, timing = run_whole_parent_transition(
                    anytime_model=anytime_model,
                    tensors=tensors,
                    batches=batches,
                    frame=frame,
                    parent_id_col=args.parent_id_col,
                    selected_policy=selected_policies[method],
                    bundle=bundles[method],
                    rules=rules,
                    device=args.device,
                    collect=False,
                    labels=labels,
                )
                seconds = timing["total_seconds"]
            timing_values[method].append(float(seconds))

    timing_summary = {
        method: quartiles(values)
        for method, values in timing_values.items()
    }
    baseline_median = timing_summary["always_exit3"]["median_seconds"]

    method_outputs: dict[str, dict[str, np.ndarray]] = {
        "always_exit3": {
            "selected_probabilities": always_outputs["p3"],
            "selected_exit": np.full(len(frame), 3, dtype=np.int8),
        }
    }
    for method, selected_policy in selected_policies.items():
        output, single_timing = run_whole_parent_transition(
            anytime_model=anytime_model,
            tensors=tensors,
            batches=batches,
            frame=frame,
            parent_id_col=args.parent_id_col,
            selected_policy=selected_policy,
            bundle=bundles[method],
            rules=rules,
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

    target_bundle = whole_parent_unsafe_targets(
        y_true=y_true,
        source_probabilities=always_outputs["p2"],
        deeper_probabilities=always_outputs["p3"],
        parent_ids=parent_ids,
        rules=rules,
    )
    row_to_parent = target_bundle["row_to_parent"]
    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames_observed,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )

    comparison_rows: list[dict[str, Any]] = []
    all_method_summaries: dict[str, Any] = {}
    for method, output in method_outputs.items():
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        selected_probabilities = output["selected_probabilities"]
        selected_exit = output["selected_exit"]
        selected_predictions = np.zeros_like(y_true, dtype=np.int8)
        for exit_no in (2, 3):
            mask = selected_exit == exit_no
            if np.any(mask):
                selected_predictions[mask] = label_predictions(
                    selected_probabilities[mask],
                    exit_thresholds[exit_no - 1],
                )
        segment_metrics = multilabel_metrics(y_true, selected_predictions)

        segment_frame = frame[
            [args.parent_id_col, "feat_relpath", *labels]
        ].copy()
        for label_index, label in enumerate(labels):
            segment_frame[f"dynamic_prob_{label}"] = selected_probabilities[
                :, label_index
            ]
            segment_frame[f"dynamic_pred_{label}"] = selected_predictions[
                :, label_index
            ]
        segment_frame["selected_exit"] = selected_exit

        parent_diagnostic = pd.DataFrame(
            {
                args.parent_id_col: target_bundle["parent_ids"],
                "any_true_exit2_harm": target_bundle["any_unsafe"],
                "exit2_parent_error_count": target_bundle[
                    "source_error_count"
                ],
                "exit3_parent_error_count": target_bundle[
                    "deeper_error_count"
                ],
                "net_error_increase_exit2": target_bundle[
                    "net_error_increase"
                ],
            }
        )

        if method != "always_exit3":
            selected_policy = selected_policies[method]
            parameters = selected_policy["parameters"]
            label_thresholds = np.asarray(
                [
                    float(
                        parameters[
                            "label_unsafe_probability_thresholds"
                        ][label]
                    )
                    for label in labels
                ],
                dtype=np.float32,
            )
            unsafe_by_row = output["unsafe_probabilities"]
            unsafe_by_parent = np.zeros(
                (len(target_bundle["parent_ids"]), len(labels)),
                dtype=np.float32,
            )
            expected_by_parent = np.zeros(
                len(target_bundle["parent_ids"]), dtype=np.float32
            )
            highest_by_parent = np.zeros(
                len(target_bundle["parent_ids"]), dtype=np.int64
            )
            parent_stop = np.zeros(
                len(target_bundle["parent_ids"]), dtype=bool
            )
            for parent_index in range(len(target_bundle["parent_ids"])):
                rows = np.flatnonzero(row_to_parent == parent_index)
                unsafe_by_parent[parent_index] = unsafe_by_row[rows[0]]
                expected_by_parent[parent_index] = output["expected_harm"][rows[0]]
                highest_by_parent[parent_index] = output[
                    "highest_risk_label"
                ][rows[0]]
                parent_stop[parent_index] = bool(
                    output["parent_stop_mask_by_row"][rows[0]]
                )

            triggered = unsafe_by_parent >= label_thresholds.reshape(1, -1)
            parent_diagnostic["stopped_at_exit2"] = parent_stop
            parent_diagnostic["expected_parent_harm"] = expected_by_parent
            parent_diagnostic["highest_risk_label"] = [
                labels[int(index)] for index in highest_by_parent
            ]
            reasons: list[str] = []
            for parent_index in range(len(parent_stop)):
                if parent_stop[parent_index]:
                    reasons.append("whole_parent_stopped_at_exit2")
                    continue
                active = [
                    labels[label_index]
                    for label_index in np.flatnonzero(triggered[parent_index])
                ]
                parts: list[str] = []
                if active:
                    parts.append("unsafe_labels=" + "|".join(active))
                harm_threshold = parameters.get("expected_harm_threshold")
                if (
                    harm_threshold is not None
                    and float(expected_by_parent[parent_index])
                    >= float(harm_threshold)
                ):
                    parts.append("expected_harm_threshold")
                reasons.append(";".join(parts) or "whole_parent_continue")
            parent_diagnostic["decision_reason"] = reasons

            for label_index, label in enumerate(labels):
                parent_diagnostic[
                    f"unsafe_probability_{label}"
                ] = unsafe_by_parent[:, label_index]
                parent_diagnostic[
                    f"unsafe_threshold_{label}"
                ] = label_thresholds[label_index]
                parent_diagnostic[
                    f"true_unsafe_{label}"
                ] = target_bundle["unsafe_targets"][:, label_index]
                segment_frame[
                    f"unsafe_probability_{label}"
                ] = unsafe_by_row[:, label_index]
            segment_frame["whole_parent_stopped_at_exit2"] = output[
                "parent_stop_mask_by_row"
            ]
            segment_frame["expected_parent_harm"] = output[
                "expected_harm"
            ]
            segment_frame["highest_risk_label"] = [
                labels[int(index)]
                for index in output["highest_risk_label"]
            ]
        else:
            parent_diagnostic["stopped_at_exit2"] = False
            parent_diagnostic["decision_reason"] = "always_exit3"
            segment_frame["whole_parent_stopped_at_exit2"] = False

        segment_csv = method_dir / "segment_predictions.csv"
        parent_diagnostic_csv = method_dir / "parent_decisions.csv"
        segment_frame.to_csv(segment_csv, index=False)
        parent_diagnostic.to_csv(parent_diagnostic_csv, index=False)
        parent_metrics = evaluate_parent_lats(
            segment_csv=segment_csv,
            labels_json=args.labels_json.resolve(),
            lats_config_json=args.lats_config_json.resolve(),
            out_dir=method_dir / "parent_frozen_lats_v2",
            parent_id_col=args.parent_id_col,
            model_name=method,
        )

        segment_stop_fraction = float(np.mean(selected_exit == 2))
        if method == "always_exit3":
            parent_stop_fraction = 0.0
        else:
            parent_stop_fraction = float(
                parent_diagnostic["stopped_at_exit2"].mean()
            )
        average_flops = float(
            segment_stop_fraction * float(flops["exit2"])
            + (1.0 - segment_stop_fraction) * float(flops["exit3"])
        )
        flops_saved = float(
            100.0
            * (1.0 - average_flops / max(float(flops["exit3"]), 1.0))
        )
        timing = timing_summary[method]
        speedup = float(
            baseline_median / max(timing["median_seconds"], 1e-12)
        )
        row = {
            "method": method,
            "deployment_eligible": bool(
                method == "always_exit3"
                or selected_policies[method]["deployment_eligible"]
            ),
            "parent_stop_fraction": parent_stop_fraction,
            "segment_stop_fraction": segment_stop_fraction,
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
                None if method == "always_exit3" else selected_policies[method]
            ),
            "genuine_skipping_statement": (
                "When a parent stopped, none of its segments executed Blocks 4-5."
            ),
            "audit_note": (
                "All-Exit3 probabilities were produced separately for final audit "
                "and were unavailable to the runtime decision."
            ),
        }
        save_json(method_summary, method_dir / "runtime_summary.json")
        all_method_summaries[method] = method_summary

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = out_dir / "v015_whole_parent_holdout_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    save_json(
        {
            "experiment": "v0.15_EE_whole_parent_selective_risk_control",
            "comparison": [jsonable(row) for row in comparison_rows],
            "methods": all_method_summaries,
            "important_note": (
                "The holdout used frozen validation-only OOF-selected policies; "
                "no model, threshold or risk budget was changed after holdout access."
            ),
        },
        out_dir / "v015_whole_parent_holdout_comparison.json",
    )

    display_columns = [
        "method",
        "deployment_eligible",
        "parent_stop_fraction",
        "segment_stop_fraction",
        "estimated_flops_saved_pct",
        "latency_median_per_segment_ms",
        "measured_speedup_vs_always_exit3",
        "parent_macro_f1",
        "parent_micro_f1",
        "parent_exact_match",
        "parent_hamming_loss",
    ]
    print("\nV0.15 whole-parent holdout comparison complete")
    print("-" * 158)
    print(comparison[display_columns].to_string(index=False))
    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
