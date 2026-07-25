#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate frozen v0.16 and v0.13 per-label margin policies with real skipping."""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v016 import (  # noqa: E402
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
from models.anytime_exit_net import AnytimeExitNet, AnytimeExitState  # noqa: E402
from policies.early_exit_strategy_comparison import (  # noqa: E402
    compute_common_diagnostics,
    label_predictions,
)
from policies.multiobjective_per_label_margin import (  # noqa: E402
    MultiObjectiveMarginConfig,
    multiobjective_margin_stop_mask,
)
from utils.model_factory import build_audio_exit_net  # noqa: E402
from utils.profiling import estimate_flops_tiny_audiocnn  # noqa: E402


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


def thresholds_from_policy(
    policy: dict[str, Any], labels: list[str], exit_no: int
) -> np.ndarray:
    mapping = policy["thresholds_by_exit"][f"exit{exit_no}"]
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def load_features(frame: pd.DataFrame, root: Path) -> list[torch.Tensor]:
    tensors = [
        load_feature(root / Path(value.replace("\\", "/")))
        for value in frame["feat_relpath"].astype(str)
    ]
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise RuntimeError(f"Holdout features have inconsistent shapes: {sorted(shapes)}")
    return tensors


def make_batches(length: int, batch_size: int) -> list[np.ndarray]:
    return [
        np.arange(start, min(start + batch_size, length), dtype=np.int64)
        for start in range(0, length, batch_size)
    ]


def config_from_parameters(parameters: dict[str, Any]) -> MultiObjectiveMarginConfig:
    return MultiObjectiveMarginConfig(
        mean_confidence_threshold=float(parameters["mean_confidence_threshold"]),
        max_probability_delta=float(parameters.get("max_probability_delta", 1.0)),
        per_label_margins=tuple(
            float(value) for value in parameters["per_label_margins"]
        ),
        require_label_set_agreement=bool(
            parameters.get("require_label_set_agreement", True)
        ),
        allow_empty_stop=bool(parameters.get("allow_empty_stop", False)),
    )


def run_always_exit3(
    model: AnytimeExitNet,
    tensors: list[torch.Tensor],
    batches: list[np.ndarray],
    device: str,
    num_labels: int,
    collect: bool,
) -> tuple[dict[str, np.ndarray] | None, float]:
    output = None
    if collect:
        output = {
            "selected_probabilities": np.zeros(
                (len(tensors), num_labels), dtype=np.float32
            ),
            "selected_exit": np.full(len(tensors), 3, dtype=np.int8),
        }
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for indices in batches:
            x = torch.cat([tensors[int(idx)] for idx in indices], dim=0).to(device)
            _, state1 = model.start(x)
            _, state2 = model.continue_from(state1)
            logits3, state3 = model.continue_from(state2)
            if not state3.finished:
                raise RuntimeError("Always-Exit3 execution did not finish.")
            if output is not None:
                output["selected_probabilities"][indices] = (
                    torch.sigmoid(logits3).cpu().numpy()
                )
    synchronize(device)
    return output, float(time.perf_counter() - started)


def run_adaptive(
    model: AnytimeExitNet,
    tensors: list[torch.Tensor],
    batches: list[np.ndarray],
    config: MultiObjectiveMarginConfig,
    thresholds: list[np.ndarray],
    device: str,
    num_labels: int,
    collect: bool,
) -> tuple[dict[str, np.ndarray] | None, dict[str, float]]:
    output = None
    if collect:
        output = {
            "selected_probabilities": np.zeros(
                (len(tensors), num_labels), dtype=np.float32
            ),
            "selected_exit": np.full(len(tensors), 3, dtype=np.int8),
            "mean_binary_confidence": np.zeros(len(tensors), dtype=np.float32),
            "maximum_probability_delta": np.zeros(len(tensors), dtype=np.float32),
            "minimum_margin_ratio": np.zeros(len(tensors), dtype=np.float32),
        }
    model_seconds = 0.0
    policy_seconds = 0.0
    required = np.asarray(config.per_label_margins, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        for indices in batches:
            x = torch.cat([tensors[int(idx)] for idx in indices], dim=0).to(device)
            synchronize(device)
            started = time.perf_counter()
            logits1, state1 = model.start(x)
            logits2, state2 = model.continue_from(state1)
            p1 = torch.sigmoid(logits1).cpu().numpy().astype(np.float32)
            p2 = torch.sigmoid(logits2).cpu().numpy().astype(np.float32)
            synchronize(device)
            model_seconds += time.perf_counter() - started

            started = time.perf_counter()
            diagnostics = compute_common_diagnostics(
                exit1_probabilities=p1,
                exit2_probabilities=p2,
                exit1_thresholds=thresholds[0],
                exit2_thresholds=thresholds[1],
            )
            stop = multiobjective_margin_stop_mask(diagnostics, config)
            policy_seconds += time.perf_counter() - started

            if output is not None:
                stopped = indices[stop]
                output["selected_probabilities"][stopped] = p2[stop]
                output["selected_exit"][stopped] = 2
                output["mean_binary_confidence"][indices] = diagnostics[
                    "mean_binary_confidence"
                ]
                output["maximum_probability_delta"][indices] = diagnostics[
                    "maximum_probability_delta"
                ]
                output["minimum_margin_ratio"][indices] = np.min(
                    diagnostics["decision_margin"] / np.maximum(required, 1e-6),
                    axis=1,
                )

            continuing = np.flatnonzero(~stop)
            if len(continuing):
                local = torch.as_tensor(
                    continuing, dtype=torch.long, device=state2.feature_map.device
                )
                continuing_state = subset_state(state2, local)
                synchronize(device)
                started = time.perf_counter()
                logits3, state3 = model.continue_from(continuing_state)
                synchronize(device)
                model_seconds += time.perf_counter() - started
                if not state3.finished:
                    raise RuntimeError("Adaptive execution did not reach Exit 3.")
                if output is not None:
                    global_indices = indices[~stop]
                    output["selected_probabilities"][global_indices] = (
                        torch.sigmoid(logits3).cpu().numpy().astype(np.float32)
                    )
                    output["selected_exit"][global_indices] = 3
    return output, {
        "model_seconds": float(model_seconds),
        "policy_seconds": float(policy_seconds),
        "total_seconds": float(model_seconds + policy_seconds),
    }


def evaluate_parent_lats(
    segment_csv: Path,
    labels_json: Path,
    lats_json: Path,
    out_dir: Path,
    parent_id_col: str,
    model_name: str,
) -> dict[str, Any]:
    script = PROJECT_ROOT / "scripts" / "v0.10" / "evaluate_frozen_lats_config_v010.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--segment-pred-csv",
            str(segment_csv),
            "--labels-json",
            str(labels_json),
            "--config-json",
            str(lats_json),
            "--out-dir",
            str(out_dir),
            "--parent-id-col",
            parent_id_col,
            "--prob-prefix",
            "dynamic_prob_",
            "--model-name",
            model_name,
        ],
        check=True,
    )
    return pd.read_csv(out_dir / "v010_frozen_lats_eval.csv").iloc[0].to_dict()


def timing_stats(values: list[float]) -> dict[str, float]:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy_json", required=True, type=Path)
    parser.add_argument("--v013_policy_json", type=Path, default=None)
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
    for path in (
        checkpoint,
        args.policy_json.resolve(),
        args.holdout_manifest.resolve(),
        args.features_root.resolve(),
        args.labels_json.resolve(),
        args.lats_config_json.resolve(),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    cfg = load_run_config(run_dir)
    labels = load_labels(args.labels_json.resolve(), cfg)
    policy = load_json(args.policy_json.resolve())
    if policy.get("experiment") != "v0.16_EE_multiobjective_per_label_margin":
        raise RuntimeError("The supplied policy is not a v0.16 policy.")
    if list(policy.get("labels", [])) != labels:
        raise RuntimeError("Frozen policy and label schema order differ.")

    tap_blocks = parse_tap_blocks(cfg.get("tap_blocks", "1,3"))
    n_mels = int(cfg.get("n_mels", 64))
    base_model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=resolve_model_cfg(cfg),
    ).to(args.device)
    load_checkpoint(base_model, checkpoint, args.device)
    base_model.eval()
    model = AnytimeExitNet(base_model).to(args.device)
    model.eval()

    thresholds = [
        thresholds_from_policy(policy, labels, exit_no) for exit_no in (1, 2, 3)
    ]
    frame = pd.read_csv(args.holdout_manifest.resolve(), low_memory=False).reset_index(
        drop=True
    )
    missing = [
        column
        for column in ["feat_relpath", args.parent_id_col, *labels]
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(f"Holdout manifest is missing columns: {missing}")
    tensors = load_features(frame, args.features_root.resolve())
    batches = make_batches(len(frame), int(args.batch_size))
    y_true = frame[labels].astype(int).to_numpy(dtype=np.int8)

    selected = policy["selected_policy"]
    methods: dict[str, dict[str, Any]] = {
        "v016_multiobjective_margin": {
            "config": config_from_parameters(selected["parameters"]),
            "deployment_eligible": bool(selected["deployment_eligible"]),
            "source": "v0.16 frozen Pareto selection",
        }
    }
    if args.v013_policy_json and args.v013_policy_json.resolve().exists():
        v013 = load_json(args.v013_policy_json.resolve())
        item = v013.get("selected_policies", {}).get("per_label_margin")
        if item:
            methods["v013_per_label_margin"] = {
                "config": config_from_parameters(item["parameters"]),
                "deployment_eligible": bool(
                    item.get("selection_metrics", {}).get(
                        "quality_constraint_met", True
                    )
                ),
                "source": "v0.13 frozen per-label margin baseline",
            }

    with torch.no_grad():
        warm = torch.cat([tensors[int(idx)] for idx in batches[0]], dim=0).to(
            args.device
        )
        _, state1 = model.start(warm)
        _, state2 = model.continue_from(state1)
        _, _ = model.continue_from(state2)
    synchronize(args.device)

    always_output, _ = run_always_exit3(
        model, tensors, batches, args.device, len(labels), True
    )
    assert always_output is not None
    names = ["always_exit3", *methods]
    timings: dict[str, list[float]] = {name: [] for name in names}
    rng = random.Random(int(args.timing_seed))
    for _ in range(int(args.timing_repeats)):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            if name == "always_exit3":
                _, seconds = run_always_exit3(
                    model, tensors, batches, args.device, len(labels), False
                )
            else:
                _, timing = run_adaptive(
                    model,
                    tensors,
                    batches,
                    methods[name]["config"],
                    thresholds,
                    args.device,
                    len(labels),
                    False,
                )
                seconds = timing["total_seconds"]
            timings[name].append(float(seconds))
    timing_summary = {name: timing_stats(values) for name, values in timings.items()}
    baseline_median = timing_summary["always_exit3"]["median_seconds"]

    outputs: dict[str, dict[str, np.ndarray]] = {
        "always_exit3": always_output
    }
    single_timings: dict[str, dict[str, float]] = {}
    for name, item in methods.items():
        output, timing = run_adaptive(
            model,
            tensors,
            batches,
            item["config"],
            thresholds,
            args.device,
            len(labels),
            True,
        )
        assert output is not None
        outputs[name] = output
        single_timings[name] = timing

    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=int(tensors[0].shape[-1]),
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for name, output in outputs.items():
        method_dir = out_dir / name
        method_dir.mkdir(parents=True, exist_ok=True)
        probabilities = output["selected_probabilities"]
        selected_exit = output["selected_exit"]
        predictions = np.zeros_like(y_true, dtype=np.int8)
        for exit_no in (2, 3):
            mask = selected_exit == exit_no
            if np.any(mask):
                predictions[mask] = label_predictions(
                    probabilities[mask], thresholds[exit_no - 1]
                )
        segment_metrics = multilabel_metrics(y_true, predictions)

        segment_frame = frame[[args.parent_id_col, "feat_relpath", *labels]].copy()
        for label_idx, label in enumerate(labels):
            segment_frame[f"dynamic_prob_{label}"] = probabilities[:, label_idx]
            segment_frame[f"dynamic_pred_{label}"] = predictions[:, label_idx]
        segment_frame["selected_exit"] = selected_exit
        if name == "always_exit3":
            segment_frame["continuation_reason"] = "always_exit3"
        else:
            config = methods[name]["config"]
            segment_frame["mean_binary_confidence"] = output[
                "mean_binary_confidence"
            ]
            segment_frame["maximum_probability_delta"] = output[
                "maximum_probability_delta"
            ]
            segment_frame["minimum_margin_ratio"] = output[
                "minimum_margin_ratio"
            ]
            segment_frame["continuation_reason"] = np.where(
                selected_exit == 2,
                "stopped_at_exit2",
                "per_label_margin_policy_continue",
            )
            for label_idx, label in enumerate(labels):
                segment_frame[f"required_margin_{label}"] = float(
                    config.per_label_margins[label_idx]
                )

        segment_csv = method_dir / "segment_predictions.csv"
        segment_frame.to_csv(segment_csv, index=False)
        parent_metrics = evaluate_parent_lats(
            segment_csv,
            args.labels_json.resolve(),
            args.lats_config_json.resolve(),
            method_dir / "parent_frozen_lats_v2",
            args.parent_id_col,
            name,
        )
        exit2_fraction = float(np.mean(selected_exit == 2))
        average_flops = float(
            exit2_fraction * float(flops["exit2"])
            + (1.0 - exit2_fraction) * float(flops["exit3"])
        )
        saved = float(
            100.0 * (1.0 - average_flops / max(float(flops["exit3"]), 1.0))
        )
        timing = timing_summary[name]
        row = {
            "method": name,
            "deployment_eligible": bool(
                name == "always_exit3" or methods[name]["deployment_eligible"]
            ),
            "exit2_fraction": exit2_fraction,
            "average_exit_depth": float(np.mean(selected_exit)),
            "estimated_flops_saved_pct": saved,
            "latency_median_per_segment_ms": float(
                1000.0 * timing["median_seconds"] / len(frame)
            ),
            "latency_iqr_per_segment_ms": float(
                1000.0 * timing["iqr_seconds"] / len(frame)
            ),
            "measured_speedup_vs_always_exit3": float(
                baseline_median / max(timing["median_seconds"], 1e-12)
            ),
            **{f"segment_{key}": value for key, value in segment_metrics.items()},
            **{
                f"parent_{key}": value
                for key, value in parent_metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            },
        }
        rows.append(row)
        summary = {
            **row,
            "timing": timing,
            "policy": None
            if name == "always_exit3"
            else {
                "source": methods[name]["source"],
                "parameters": methods[name]["config"].to_dict(),
                "single_pass_timing": single_timings[name],
            },
            "genuine_skipping_statement": (
                "Samples stopped at Exit 2 did not execute the final blocks."
            ),
        }
        save_json(summary, method_dir / "runtime_summary.json")
        summaries[name] = summary

    comparison = pd.DataFrame(rows)
    comparison_path = out_dir / "v016_multiobjective_holdout_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    save_json(
        {
            "experiment": "v0.16_EE_multiobjective_per_label_margin",
            "comparison": [jsonable(row) for row in rows],
            "methods": summaries,
            "important_note": (
                "The holdout used frozen validation-only thresholds; no policy "
                "parameter or constraint changed after holdout access."
            ),
        },
        out_dir / "v016_multiobjective_holdout_comparison.json",
    )
    columns = [
        "method",
        "deployment_eligible",
        "exit2_fraction",
        "estimated_flops_saved_pct",
        "latency_median_per_segment_ms",
        "measured_speedup_vs_always_exit3",
        "parent_macro_f1",
        "parent_micro_f1",
        "parent_exact_match",
        "parent_hamming_loss",
    ]
    print("\nV0.16 multi-objective holdout comparison complete")
    print("-" * 154)
    print(comparison[columns].to_string(index=False))
    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
