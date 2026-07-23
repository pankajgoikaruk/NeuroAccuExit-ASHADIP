#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate five frozen Early-Exit strategies with genuine staged inference."""

from __future__ import annotations

import argparse
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

from common_v013 import (
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
from policies.early_exit_strategy_comparison import (
    GlobalRuleConfig,
    LabelRiskRuleConfig,
    PerLabelMarginConfig,
    build_gate_features,
    compute_common_diagnostics,
    continuation_reasons,
    global_rule_stop_mask,
    label_predictions,
    label_risk_stop_mask,
    logistic_gate_stop_mask,
    per_label_margin_stop_mask,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


METHODS = (
    "always_exit3",
    "global_conf_margin",
    "global_conf_margin_delta",
    "label_risk",
    "per_label_margin",
    "logistic_gate",
)


def thresholds_from_policy(
    policy: dict[str, Any],
    labels: list[str],
    exit_no: int,
) -> np.ndarray:
    mapping = policy["thresholds_by_exit"][f"exit{exit_no}"]
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(
            f"Exit {exit_no} thresholds miss labels: {missing}"
        )
    return np.asarray(
        [float(mapping[label]) for label in labels],
        dtype=np.float32,
    )


def subset_state(
    state: AnytimeExitState,
    indices: torch.Tensor,
) -> AnytimeExitState:
    hint = (
        None
        if state.prev_hint is None
        else state.prev_hint.index_select(0, indices)
    )
    return AnytimeExitState(
        feature_map=state.feature_map.index_select(0, indices),
        block_index=int(state.block_index),
        next_exit_index=int(state.next_exit_index),
        prev_hint=hint,
        finished=bool(state.finished),
    )


def sanitize_method(method: str) -> str:
    return method.replace("/", "_").replace(" ", "_")


def corrections_and_regressions(
    *,
    truth: np.ndarray,
    exit2_pred: np.ndarray,
    exit3_pred: np.ndarray,
    labels: list[str],
) -> tuple[list[str], list[str]]:
    corrections: list[str] = []
    regressions: list[str] = []
    for idx, label in enumerate(labels):
        if (
            exit2_pred[idx] != truth[idx]
            and exit3_pred[idx] == truth[idx]
        ):
            corrections.append(label)
        elif (
            exit2_pred[idx] == truth[idx]
            and exit3_pred[idx] != truth[idx]
        ):
            regressions.append(label)
    return corrections, regressions


def policy_stop_mask(
    *,
    method: str,
    diagnostics: dict[str, np.ndarray],
    parameters: dict[str, Any],
    gate_model,
    exit1_probabilities: np.ndarray,
    exit2_probabilities: np.ndarray,
    exit1_thresholds: np.ndarray,
    exit2_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    if method == "always_exit3":
        return (
            np.zeros(len(exit2_probabilities), dtype=bool),
            None,
        )
    if method in {
        "global_conf_margin",
        "global_conf_margin_delta",
    }:
        config = GlobalRuleConfig(**parameters)
        return global_rule_stop_mask(diagnostics, config), None
    if method == "label_risk":
        config = LabelRiskRuleConfig(**parameters)
        return label_risk_stop_mask(diagnostics, config), None
    if method == "per_label_margin":
        clean = dict(parameters)
        clean.pop("capture_fraction", None)
        clean.pop("corrected_example_counts", None)
        clean["per_label_margins"] = tuple(
            float(value)
            for value in clean["per_label_margins"]
        )
        config = PerLabelMarginConfig(**clean)
        return per_label_margin_stop_mask(diagnostics, config), None
    if method == "logistic_gate":
        if gate_model is None:
            raise RuntimeError("Logistic gate model was not loaded.")
        features, _ = build_gate_features(
            exit1_probabilities=exit1_probabilities,
            exit2_probabilities=exit2_probabilities,
            exit1_thresholds=exit1_thresholds,
            exit2_thresholds=exit2_thresholds,
        )
        safe_probability = gate_model.predict_proba(
            features
        )[:, 1].astype(np.float32)
        mask = logistic_gate_stop_mask(
            safe_probabilities=safe_probability,
            threshold=float(
                parameters["gate_probability_threshold"]
            ),
            diagnostics=diagnostics,
            allow_empty_stop=bool(
                parameters.get("allow_empty_stop", False)
            ),
        )
        return mask, safe_probability
    raise ValueError(f"Unsupported method: {method}")


def warmup(
    *,
    anytime_model: AnytimeExitNet,
    frame: pd.DataFrame,
    features_root: Path,
    batch_size: int,
    device: str,
) -> None:
    batch = frame.iloc[: min(len(frame), batch_size)]
    tensors: list[torch.Tensor] = []
    for _, row in batch.iterrows():
        relative = Path(
            str(row["feat_relpath"]).replace("\\", "/")
        )
        tensors.append(load_feature(features_root / relative))
    if not tensors:
        return
    x = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        _, state1 = anytime_model.start(x)
        _, state2 = anytime_model.continue_from(state1)
        _, _ = anytime_model.continue_from(state2)
    synchronize(device)


@torch.no_grad()
def evaluate_method(
    *,
    method: str,
    parameters: dict[str, Any],
    frame: pd.DataFrame,
    labels: list[str],
    anytime_model: AnytimeExitNet,
    features_root: Path,
    threshold1: np.ndarray,
    threshold2: np.ndarray,
    threshold3: np.ndarray,
    risk_weights: np.ndarray,
    risk_margin_scale: float,
    risk_margin_weight: float,
    risk_delta_weight: float,
    gate_model,
    batch_size: int,
    device: str,
    parent_id_col: str,
    out_dir: Path,
    lats_config_json: Path,
    labels_json: Path,
    run_dir: Path,
    checkpoint: Path,
    skip_parent_eval: bool,
) -> dict[str, Any]:
    n_samples = len(frame)
    n_labels = len(labels)
    selected_probs = np.zeros(
        (n_samples, n_labels),
        dtype=np.float32,
    )
    selected_pred = np.zeros(
        (n_samples, n_labels),
        dtype=np.int8,
    )
    selected_exit = np.full(n_samples, 3, dtype=np.int8)

    agreement = np.zeros(n_samples, dtype=bool)
    non_empty = np.zeros(n_samples, dtype=bool)
    mean_confidence = np.zeros(n_samples, dtype=np.float32)
    min_margin = np.zeros(n_samples, dtype=np.float32)
    max_delta = np.zeros(n_samples, dtype=np.float32)
    max_risk = np.zeros(n_samples, dtype=np.float32)
    highest_risk_idx = np.zeros(n_samples, dtype=np.int64)
    gate_safe_probability = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )
    continuation_reason = np.full(
        n_samples,
        "stopped_at_exit2",
        dtype=object,
    )
    exit3_error_improvement = np.full(
        n_samples,
        np.nan,
        dtype=np.float32,
    )
    exit3_corrections = np.full(n_samples, "", dtype=object)
    exit3_regressions = np.full(n_samples, "", dtype=object)
    per_label_risk = np.zeros(
        (n_samples, n_labels),
        dtype=np.float32,
    )

    stage12_seconds = 0.0
    stage3_seconds = 0.0
    policy_seconds = 0.0
    frames_observed: int | None = None
    y_true_all = frame[labels].astype(int).to_numpy(
        dtype=np.int8
    )

    for start in range(0, n_samples, int(batch_size)):
        batch = frame.iloc[
            start : start + int(batch_size)
        ]
        tensors: list[torch.Tensor] = []
        for _, row in batch.iterrows():
            relative = Path(
                str(row["feat_relpath"]).replace("\\", "/")
            )
            tensors.append(
                load_feature(features_root / relative)
            )
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise RuntimeError(
                f"Inconsistent feature shapes in batch {start}: "
                f"{sorted(shapes)}"
            )
        x = torch.cat(tensors, dim=0).to(device)
        if frames_observed is None:
            frames_observed = int(x.shape[-1])

        synchronize(device)
        stage12_start = time.perf_counter()
        exit1_logits, state1 = anytime_model.start(x)
        exit2_logits, state2 = anytime_model.continue_from(
            state1
        )
        synchronize(device)
        stage12_seconds += (
            time.perf_counter() - stage12_start
        )

        p1 = torch.sigmoid(exit1_logits).cpu().numpy()
        p2 = torch.sigmoid(exit2_logits).cpu().numpy()

        if method == "always_exit3":
            diagnostics = compute_common_diagnostics(
                exit1_probabilities=p1,
                exit2_probabilities=p2,
                exit1_thresholds=threshold1,
                exit2_thresholds=threshold2,
                risk_weights=risk_weights,
                risk_margin_scale=risk_margin_scale,
                risk_margin_weight=risk_margin_weight,
                risk_delta_weight=risk_delta_weight,
            )
            stop_mask, gate_probs = policy_stop_mask(
                method=method,
                diagnostics=diagnostics,
                parameters=parameters,
                gate_model=gate_model,
                exit1_probabilities=p1,
                exit2_probabilities=p2,
                exit1_thresholds=threshold1,
                exit2_thresholds=threshold2,
            )
        else:
            policy_start = time.perf_counter()
            diagnostics = compute_common_diagnostics(
                exit1_probabilities=p1,
                exit2_probabilities=p2,
                exit1_thresholds=threshold1,
                exit2_thresholds=threshold2,
                risk_weights=risk_weights,
                risk_margin_scale=risk_margin_scale,
                risk_margin_weight=risk_margin_weight,
                risk_delta_weight=risk_delta_weight,
            )
            stop_mask, gate_probs = policy_stop_mask(
                method=method,
                diagnostics=diagnostics,
                parameters=parameters,
                gate_model=gate_model,
                exit1_probabilities=p1,
                exit2_probabilities=p2,
                exit1_thresholds=threshold1,
                exit2_thresholds=threshold2,
            )
            policy_seconds += (
                time.perf_counter() - policy_start
            )

        reasons = continuation_reasons(
            method=method,
            diagnostics=diagnostics,
            config=parameters,
            stop_mask=stop_mask,
            gate_safe_probabilities=gate_probs,
        )

        local_count = len(batch)
        global_indices = np.arange(
            start,
            start + local_count,
        )
        agreement[global_indices] = diagnostics[
            "label_set_agreement"
        ]
        non_empty[global_indices] = diagnostics["non_empty"]
        mean_confidence[global_indices] = diagnostics[
            "mean_binary_confidence"
        ]
        min_margin[global_indices] = diagnostics[
            "minimum_decision_margin"
        ]
        max_delta[global_indices] = diagnostics[
            "maximum_probability_delta"
        ]
        max_risk[global_indices] = diagnostics[
            "maximum_label_risk"
        ]
        highest_risk_idx[global_indices] = diagnostics[
            "highest_risk_label_index"
        ]
        per_label_risk[global_indices] = diagnostics[
            "per_label_risk"
        ]
        continuation_reason[global_indices] = reasons
        if gate_probs is not None:
            gate_safe_probability[global_indices] = gate_probs

        stopped_global = global_indices[stop_mask]
        if len(stopped_global):
            selected_probs[stopped_global] = p2[stop_mask]
            selected_pred[stopped_global] = diagnostics[
                "exit2_pred"
            ][stop_mask]
            selected_exit[stopped_global] = 2

        continue_mask = ~stop_mask
        continuing_local = np.flatnonzero(continue_mask)
        continuing_global = global_indices[continue_mask]
        if len(continuing_local):
            index_tensor = torch.as_tensor(
                continuing_local,
                dtype=torch.long,
                device=state2.feature_map.device,
            )
            continuing_state = subset_state(
                state2,
                index_tensor,
            )
            synchronize(device)
            stage3_start = time.perf_counter()
            exit3_logits, final_state = (
                anytime_model.continue_from(continuing_state)
            )
            synchronize(device)
            stage3_seconds += (
                time.perf_counter() - stage3_start
            )
            if not final_state.finished:
                raise RuntimeError(
                    "Exit 3 did not finish the staged state."
                )

            p3 = torch.sigmoid(exit3_logits).cpu().numpy()
            pred3 = label_predictions(p3, threshold3)
            pred2_cont = diagnostics["exit2_pred"][
                continue_mask
            ]
            truth_cont = y_true_all[continuing_global]
            selected_probs[continuing_global] = p3
            selected_pred[continuing_global] = pred3
            selected_exit[continuing_global] = 3
            improvement = (
                np.sum(pred2_cont != truth_cont, axis=1)
                - np.sum(pred3 != truth_cont, axis=1)
            )
            exit3_error_improvement[
                continuing_global
            ] = improvement.astype(np.float32)
            for local_idx, global_idx in enumerate(
                continuing_global
            ):
                corrected, regressed = (
                    corrections_and_regressions(
                        truth=truth_cont[local_idx],
                        exit2_pred=pred2_cont[local_idx],
                        exit3_pred=pred3[local_idx],
                        labels=labels,
                    )
                )
                exit3_corrections[global_idx] = ";".join(
                    corrected
                )
                exit3_regressions[global_idx] = ";".join(
                    regressed
                )

    if frames_observed is None:
        raise RuntimeError("No holdout samples were processed.")

    segment_metrics = multilabel_metrics(
        y_true_all,
        selected_pred,
    )
    exit2_count = int(np.sum(selected_exit == 2))
    exit3_count = int(np.sum(selected_exit == 3))
    exit2_fraction = float(
        exit2_count / max(n_samples, 1)
    )

    flops = estimate_flops_tiny_audiocnn(
        n_mels=int(frame.attrs.get("n_mels", 64)),
        frames=frames_observed,
        num_classes=n_labels,
        tap_blocks=anytime_model.tap_blocks,
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

    total_adaptive_seconds = float(
        stage12_seconds + stage3_seconds + policy_seconds
    )
    runtime = {
        "stage1_to_exit2_seconds": stage12_seconds,
        "exit3_continuation_seconds": stage3_seconds,
        "policy_decision_seconds": policy_seconds,
        "total_adaptive_inference_seconds": (
            total_adaptive_seconds
        ),
        "adaptive_latency_per_segment_ms": (
            1000.0
            * total_adaptive_seconds
            / max(n_samples, 1)
        ),
        "model_only_latency_per_segment_ms": (
            1000.0
            * (stage12_seconds + stage3_seconds)
            / max(n_samples, 1)
        ),
        "timing_scope": (
            "Model and policy decision; feature loading and file "
            "writing excluded."
        ),
        "device": device,
    }

    method_dir = out_dir / sanitize_method(method)
    method_dir.mkdir(parents=True, exist_ok=True)
    output_frame = frame[
        [parent_id_col, "feat_relpath", *labels]
    ].copy()
    for label_idx, label in enumerate(labels):
        output_frame[
            f"dynamic_prob_{label}"
        ] = selected_probs[:, label_idx]
        output_frame[
            f"dynamic_pred_{label}"
        ] = selected_pred[:, label_idx]
        output_frame[
            f"risk_{label}"
        ] = per_label_risk[:, label_idx]
    output_frame["selected_exit"] = selected_exit
    output_frame["stop_reason"] = np.where(
        selected_exit == 2,
        "reliable_early_exit",
        "final_exit",
    )
    output_frame[
        "continuation_reason"
    ] = continuation_reason
    output_frame["highest_risk_label"] = [
        labels[int(idx)]
        for idx in highest_risk_idx
    ]
    output_frame["maximum_label_risk"] = max_risk
    output_frame[
        "exit1_exit2_label_set_agreement"
    ] = agreement
    output_frame["exit2_non_empty"] = non_empty
    output_frame[
        "exit2_mean_binary_confidence"
    ] = mean_confidence
    output_frame[
        "exit2_min_decision_margin"
    ] = min_margin
    output_frame[
        "exit1_exit2_max_probability_delta"
    ] = max_delta
    output_frame[
        "gate_safe_probability"
    ] = gate_safe_probability
    output_frame[
        "exit3_binary_error_improvement"
    ] = exit3_error_improvement
    output_frame[
        "exit3_corrected_labels"
    ] = exit3_corrections
    output_frame[
        "exit3_regressed_labels"
    ] = exit3_regressions

    segment_csv = method_dir / "segment_predictions.csv"
    output_frame.to_csv(segment_csv, index=False)

    parent_summary = None
    if not skip_parent_eval:
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
        parent_out = method_dir / "parent_frozen_lats_v2"
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
            str(parent_out),
            "--parent-id-col",
            parent_id_col,
            "--prob-prefix",
            "dynamic_prob_",
            "--model-name",
            f"v0.13_{method}",
        ]
        subprocess.run(command, check=True)
        parent_summary = pd.read_csv(
            parent_out / "v010_frozen_lats_eval.csv"
        ).iloc[0].to_dict()

    summary = {
        "experiment": "v0.13_EE_matched_policy_comparison",
        "method": method,
        "parameters": parameters,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "n_segments": n_samples,
        "exit_distribution": {
            "exit2_samples": exit2_count,
            "exit3_samples": exit3_count,
            "exit2_fraction": exit2_fraction,
            "average_exit_depth": float(
                np.mean(selected_exit)
            ),
        },
        "compute": {
            "estimated_flops_by_exit": {
                key: float(value)
                for key, value in flops.items()
            },
            "estimated_flops_saved_pct": (
                estimated_flops_saved_pct
            ),
            "genuine_skipping_statement": (
                "Only samples assigned to Exit 3 executed Blocks 4-5."
            ),
        },
        "runtime": runtime,
        "segment_metrics": segment_metrics,
        "parent_frozen_lats_v2_metrics": parent_summary,
        "diagnostic_counts": {
            "continued_samples_with_positive_exit3_improvement": int(
                np.nansum(exit3_error_improvement > 0)
            ),
            "continued_samples_with_zero_exit3_improvement": int(
                np.nansum(exit3_error_improvement == 0)
            ),
            "continued_samples_with_negative_exit3_improvement": int(
                np.nansum(exit3_error_improvement < 0)
            ),
        },
        "important_note": (
            "The frozen validation-selected method was applied without "
            "corrected-holdout retuning."
        ),
    }
    save_json(
        summary,
        method_dir / "runtime_summary.json",
    )

    row: dict[str, Any] = {
        "method": method,
        "exit2_samples": exit2_count,
        "exit3_samples": exit3_count,
        "exit2_fraction": exit2_fraction,
        "average_exit_depth": float(np.mean(selected_exit)),
        "estimated_flops_saved_pct": (
            estimated_flops_saved_pct
        ),
        "adaptive_latency_per_segment_ms": runtime[
            "adaptive_latency_per_segment_ms"
        ],
        "model_only_latency_per_segment_ms": runtime[
            "model_only_latency_per_segment_ms"
        ],
        "policy_decision_seconds": policy_seconds,
        **{
            f"segment_{key}": value
            for key, value in segment_metrics.items()
        },
    }
    if parent_summary is not None:
        for key, value in parent_summary.items():
            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                row[f"parent_{key}"] = float(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate matched v0.13 Early-Exit strategies."
        )
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--comparison_json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--holdout_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--features_root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--labels_json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--lats_config_json",
        required=True,
        type=Path,
    )
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--skip_parent_eval", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else run_dir / "ckpt" / "best.pt"
    )
    comparison_path = args.comparison_json.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    required = [
        run_dir,
        checkpoint,
        comparison_path,
        args.holdout_manifest.resolve(),
        args.features_root.resolve(),
        args.labels_json.resolve(),
        args.lats_config_json.resolve(),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found: {path}"
            )

    cfg = load_run_config(run_dir)
    labels = load_labels(args.labels_json.resolve(), cfg)
    comparison = load_json(comparison_path)
    if comparison.get("experiment") != (
        "v0.13_EE_matched_policy_comparison"
    ):
        raise RuntimeError(
            "Supplied JSON is not a frozen v0.13 comparison."
        )
    if list(comparison.get("labels", [])) != labels:
        raise RuntimeError(
            "Label order mismatch between comparison and schema."
        )

    architecture = comparison["architecture"]
    tap_blocks = parse_tap_blocks(
        architecture["tap_blocks"]
    )
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

    threshold1 = thresholds_from_policy(
        comparison,
        labels,
        1,
    )
    threshold2 = thresholds_from_policy(
        comparison,
        labels,
        2,
    )
    threshold3 = thresholds_from_policy(
        comparison,
        labels,
        3,
    )
    risk_weights = np.asarray(
        comparison["label_risk_profile"]["risk_weights"],
        dtype=np.float32,
    )
    risk_definition = comparison["risk_definition"]

    frame = pd.read_csv(
        args.holdout_manifest.resolve(),
        low_memory=False,
    )
    required_columns = [
        "feat_relpath",
        args.parent_id_col,
        *labels,
    ]
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"Holdout manifest is missing columns: {missing}"
        )
    frame.attrs["n_mels"] = n_mels

    gate_model_path = (
        comparison_path.parent
        / comparison["logistic_gate"]["model_filename"]
    )
    if not gate_model_path.exists():
        raise FileNotFoundError(
            f"Frozen logistic gate not found: {gate_model_path}"
        )
    gate_model = joblib.load(gate_model_path)

    warmup(
        anytime_model=anytime_model,
        frame=frame,
        features_root=args.features_root.resolve(),
        batch_size=args.batch_size,
        device=args.device,
    )

    selected = comparison["selected_policies"]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        parameters = (
            {}
            if method == "always_exit3"
            else dict(selected[method]["parameters"])
        )
        print(f"\nEvaluating {method}...")
        rows.append(
            evaluate_method(
                method=method,
                parameters=parameters,
                frame=frame,
                labels=labels,
                anytime_model=anytime_model,
                features_root=args.features_root.resolve(),
                threshold1=threshold1,
                threshold2=threshold2,
                threshold3=threshold3,
                risk_weights=risk_weights,
                risk_margin_scale=float(
                    risk_definition["margin_scale"]
                ),
                risk_margin_weight=float(
                    risk_definition["margin_weight"]
                ),
                risk_delta_weight=float(
                    risk_definition["delta_weight"]
                ),
                gate_model=gate_model,
                batch_size=args.batch_size,
                device=args.device,
                parent_id_col=args.parent_id_col,
                out_dir=out_dir,
                lats_config_json=(
                    args.lats_config_json.resolve()
                ),
                labels_json=args.labels_json.resolve(),
                run_dir=run_dir,
                checkpoint=checkpoint,
                skip_parent_eval=args.skip_parent_eval,
            )
        )

    comparison_df = pd.DataFrame(rows)
    baseline_latency = float(
        comparison_df.loc[
            comparison_df["method"] == "always_exit3",
            "adaptive_latency_per_segment_ms",
        ].iloc[0]
    )
    comparison_df[
        "measured_speedup_vs_always_exit3"
    ] = (
        baseline_latency
        / comparison_df["adaptive_latency_per_segment_ms"]
    )
    comparison_df[
        "measured_latency_saved_pct"
    ] = 100.0 * (
        1.0
        - comparison_df["adaptive_latency_per_segment_ms"]
        / baseline_latency
    )
    comparison_csv = (
        out_dir / "v013_matched_holdout_comparison.csv"
    )
    comparison_df.to_csv(comparison_csv, index=False)
    save_json(
        {
            "experiment": (
                "v0.13_EE_matched_policy_comparison"
            ),
            "comparison_json": str(comparison_path),
            "rows": comparison_df.to_dict(
                orient="records"
            ),
            "important_note": (
                "All methods used the same checkpoint, holdout, "
                "device, batch size, staged wrapper, and timing scope."
            ),
        },
        out_dir / "v013_matched_holdout_comparison.json",
    )

    display_columns = [
        "method",
        "exit2_fraction",
        "estimated_flops_saved_pct",
        "adaptive_latency_per_segment_ms",
        "measured_speedup_vs_always_exit3",
        "segment_macro_f1",
    ]
    if "parent_macro_f1" in comparison_df.columns:
        display_columns += [
            "parent_macro_f1",
            "parent_exact_match",
            "parent_hamming_loss",
        ]
    print("\nV0.13 matched holdout comparison complete")
    print("-" * 130)
    print(
        comparison_df[display_columns].to_string(
            index=False
        )
    )
    print(f"\nSaved comparison: {comparison_csv}")


if __name__ == "__main__":
    main()
