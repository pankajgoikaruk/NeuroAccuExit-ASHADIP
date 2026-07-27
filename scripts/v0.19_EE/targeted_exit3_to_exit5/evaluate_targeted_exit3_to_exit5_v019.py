#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
V017_DIR = PROJECT_ROOT / "scripts" / "v0.17_EE" / "sequential_anytime_exit"
for path in (PROJECT_ROOT, V017_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v017 import jsonable, load_checkpoint, load_json, load_labels, load_run_config, multilabel_metrics, parse_tap_blocks, resolve_model_cfg, save_json, synchronize
from eval_v017_config import load_features, make_batches, subset_state, thresholds_from_policy
from eval_v017_reporting import evaluate_parent_lats, timing_stats
from models.anytime_exit_net import AnytimeExitNet
from policies.targeted_exit3_to_exit5_v019 import TargetedExit3ToExit5Config, targeted_diagnostics, targeted_stop_mask
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the frozen final v0.19 Exit-3-to-Exit-5 policy.")
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--policy_json", required=True, type=Path)
    p.add_argument("--holdout_manifest", required=True, type=Path)
    p.add_argument("--features_root", required=True, type=Path)
    p.add_argument("--labels_json", required=True, type=Path)
    p.add_argument("--lats_config_json", required=True, type=Path)
    p.add_argument("--parent_id_col", default="parent_clip_id")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--timing_repeats", type=int, default=30)
    p.add_argument("--timing_seed", type=int, default=42)
    p.add_argument("--torch_threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", required=True, type=Path)
    return p


def config_from_payload(payload: dict[str, Any]) -> TargetedExit3ToExit5Config:
    return TargetedExit3ToExit5Config(
        mean_confidence_threshold=float(payload["mean_confidence_threshold"]),
        max_probability_delta=float(payload["max_probability_delta"]),
        max_label_risk=float(payload["max_label_risk"]),
        risk_score_threshold=float(payload["risk_score_threshold"]),
        risk_margin_multiplier=float(payload["risk_margin_multiplier"]),
        risk_uncertainty_band=float(payload["risk_uncertainty_band"]),
        per_label_margins=tuple(float(value) for value in payload["per_label_margins"]),
        require_exit2_label_stability=bool(payload.get("require_exit2_label_stability", True)),
        allow_empty_stop=bool(payload.get("allow_empty_stop", False)),
    )


def ablation_config(config: TargetedExit3ToExit5Config, name: str) -> TargetedExit3ToExit5Config:
    if name == "no_risk":
        return replace(config, max_label_risk=1.0, risk_score_threshold=1.1, risk_margin_multiplier=1.0, risk_uncertainty_band=0.0)
    if name == "no_stability":
        return replace(config, require_exit2_label_stability=False)
    if name == "no_label_margins":
        return replace(config, per_label_margins=tuple(0.0 for _ in config.per_label_margins))
    if name == "confidence_only":
        return replace(
            config,
            max_probability_delta=1.0,
            max_label_risk=1.0,
            risk_score_threshold=1.1,
            risk_margin_multiplier=1.0,
            risk_uncertainty_band=0.0,
            per_label_margins=tuple(0.0 for _ in config.per_label_margins),
            require_exit2_label_stability=False,
        )
    raise ValueError(f"Unknown ablation: {name}")


def prepare(args: argparse.Namespace) -> SimpleNamespace:
    torch.set_num_threads(max(1, int(args.torch_threads)))
    run = args.run_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else run / "ckpt" / "best.pt"
    required = [checkpoint, args.policy_json.resolve(), args.holdout_manifest.resolve(), args.features_root.resolve(), args.labels_json.resolve(), args.lats_config_json.resolve()]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    cfg = load_run_config(run)
    labels = load_labels(args.labels_json.resolve(), cfg)
    policy = load_json(args.policy_json.resolve())
    if policy.get("experiment") != "v0.19_EE_final_targeted_exit3_to_exit5":
        raise RuntimeError("Expected a frozen v0.19 policy.")
    if policy.get("labels") != labels:
        raise RuntimeError("Frozen policy label order is incompatible with the checkpoint.")
    taps = parse_tap_blocks(cfg.get("tap_blocks", "1,2,3,4"))
    if list(taps) != [1, 2, 3, 4]:
        raise RuntimeError("v0.19 requires the fair five-exit model with taps 1,2,3,4.")
    n_mels = int(cfg.get("n_mels", 64))
    model = build_audio_exit_net(num_classes=len(labels), n_mels=n_mels, tap_blocks=taps, model_cfg=resolve_model_cfg(cfg)).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()
    anytime = AnytimeExitNet(model).to(args.device)
    anytime.eval()
    frame = pd.read_csv(args.holdout_manifest.resolve(), low_memory=False).reset_index(drop=True)
    tensors = load_features(frame, args.features_root.resolve())
    batches = make_batches(len(frame), int(args.batch_size))
    thresholds = thresholds_from_policy(policy, labels)
    config = config_from_payload(policy["selected_policy"]["parameters"])
    risk_scores = np.asarray(policy["grouped_risk_profile"]["risk_scores"], dtype=np.float32)
    y = frame[labels].astype(int).to_numpy(np.int8)
    methods = {
        "full_targeted": config,
        "no_risk": ablation_config(config, "no_risk"),
        "no_stability": ablation_config(config, "no_stability"),
        "no_label_margins": ablation_config(config, "no_label_margins"),
        "confidence_only": ablation_config(config, "confidence_only"),
    }
    return SimpleNamespace(args=args, run=run, out=out, checkpoint=checkpoint, cfg=cfg, labels=labels, policy=policy, taps=taps, n_mels=n_mels, model=anytime, frame=frame, tensors=tensors, batches=batches, thresholds=thresholds, config=config, risk_scores=risk_scores, y=y, methods=methods)


def run_always_final(*, p: SimpleNamespace, collect: bool) -> tuple[dict[str, np.ndarray] | None, float]:
    output = None if not collect else {
        "selected_probabilities": np.zeros((len(p.tensors), len(p.labels)), np.float32),
        "selected_exit": np.full(len(p.tensors), 5, np.int8),
    }
    synchronize(p.args.device)
    started = time.perf_counter()
    with torch.no_grad():
        for indices in p.batches:
            x = torch.cat([p.tensors[int(index)] for index in indices], dim=0).to(p.args.device)
            logits, state = p.model.start(x)
            while not state.finished:
                logits, state = p.model.continue_from(state)
            if output is not None:
                output["selected_probabilities"][indices] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    synchronize(p.args.device)
    return output, float(time.perf_counter() - started)


def run_targeted(*, p: SimpleNamespace, config: TargetedExit3ToExit5Config, collect: bool) -> tuple[dict[str, np.ndarray] | None, dict[str, float]]:
    output = None if not collect else {
        "selected_probabilities": np.zeros((len(p.tensors), len(p.labels)), np.float32),
        "selected_exit": np.full(len(p.tensors), 5, np.int8),
    }
    model_seconds = 0.0
    policy_seconds = 0.0
    with torch.no_grad():
        for global_indices in p.batches:
            x = torch.cat([p.tensors[int(index)] for index in global_indices], dim=0).to(p.args.device)
            active = np.asarray(global_indices, np.int64)
            synchronize(p.args.device)
            started = time.perf_counter()
            _, state = p.model.start(x)
            logits2, state = p.model.continue_from(state)
            logits3, state = p.model.continue_from(state)
            synchronize(p.args.device)
            model_seconds += time.perf_counter() - started
            probabilities2 = torch.sigmoid(logits2).cpu().numpy().astype(np.float32)
            probabilities3 = torch.sigmoid(logits3).cpu().numpy().astype(np.float32)
            policy_started = time.perf_counter()
            diagnostics = targeted_diagnostics(exit2_probabilities=probabilities2, exit3_probabilities=probabilities3, exit2_thresholds=p.thresholds[1], exit3_thresholds=p.thresholds[2])
            stop = targeted_stop_mask(diagnostics=diagnostics, risk_scores=p.risk_scores, config=config)
            policy_seconds += time.perf_counter() - policy_started
            if output is not None:
                output["selected_probabilities"][active[stop]] = probabilities3[stop]
                output["selected_exit"][active[stop]] = 3
            continuing = np.flatnonzero(~stop)
            if len(continuing) == 0:
                continue
            state = subset_state(state, torch.as_tensor(continuing, dtype=torch.long, device=state.feature_map.device))
            active = active[continuing]
            synchronize(p.args.device)
            started = time.perf_counter()
            _, state = p.model.continue_from(state)
            logits5, state = p.model.continue_from(state)
            synchronize(p.args.device)
            model_seconds += time.perf_counter() - started
            if output is not None:
                output["selected_probabilities"][active] = torch.sigmoid(logits5).cpu().numpy().astype(np.float32)
                output["selected_exit"][active] = 5
    return output, {"model_seconds": float(model_seconds), "policy_seconds": float(policy_seconds), "total_seconds": float(model_seconds + policy_seconds)}


def execute(p: SimpleNamespace) -> SimpleNamespace:
    with torch.no_grad():
        warm = torch.cat([p.tensors[int(index)] for index in p.batches[0]], dim=0).to(p.args.device)
        _, state = p.model.start(warm)
        while not state.finished:
            _, state = p.model.continue_from(state)
    synchronize(p.args.device)
    always_output, _ = run_always_final(p=p, collect=True)
    names = ["always_final", *p.methods]
    timings = {name: [] for name in names}
    rng = random.Random(int(p.args.timing_seed))
    for _ in range(int(p.args.timing_repeats)):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            if name == "always_final":
                _, seconds = run_always_final(p=p, collect=False)
            else:
                _, timing = run_targeted(p=p, config=p.methods[name], collect=False)
                seconds = timing["total_seconds"]
            timings[name].append(float(seconds))
    outputs = {"always_final": always_output}
    single: dict[str, dict[str, float]] = {}
    for name, config in p.methods.items():
        output, timing = run_targeted(p=p, config=config, collect=True)
        outputs[name] = output
        single[name] = timing
    return SimpleNamespace(outputs=outputs, single=single, timing={name: timing_stats(values) for name, values in timings.items()})


def save_results(p: SimpleNamespace, execution: SimpleNamespace) -> None:
    baseline_seconds = execution.timing["always_final"]["median_seconds"]
    flops = estimate_flops_tiny_audiocnn(n_mels=p.n_mels, frames=int(p.tensors[0].shape[-1]), num_classes=len(p.labels), tap_blocks=p.taps)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    baseline_parent: dict[str, Any] | None = None
    for name, output in execution.outputs.items():
        method_dir = p.out / name
        method_dir.mkdir(parents=True, exist_ok=True)
        probabilities = output["selected_probabilities"]
        selected_exit = output["selected_exit"]
        predictions = np.zeros_like(p.y, np.int8)
        for exit_no in (3, 5):
            mask = selected_exit == exit_no
            if np.any(mask):
                predictions[mask] = (probabilities[mask] >= p.thresholds[exit_no - 1].reshape(1, -1)).astype(np.int8)
        segment_metrics = multilabel_metrics(p.y, predictions)
        frame = p.frame[[p.args.parent_id_col, "feat_relpath", *p.labels]].copy()
        for index, label in enumerate(p.labels):
            frame[f"dynamic_prob_{label}"] = probabilities[:, index]
            frame[f"dynamic_pred_{label}"] = predictions[:, index]
        frame["selected_exit"] = selected_exit
        frame["continuation_reason"] = np.where(selected_exit == 3, "stopped_at_exit3", "continued_to_exit5")
        segment_csv = method_dir / "segment_predictions.csv"
        frame.to_csv(segment_csv, index=False)
        parent_metrics = evaluate_parent_lats(segment_csv=segment_csv, labels_json=p.args.labels_json.resolve(), lats_config_json=p.args.lats_config_json.resolve(), out_dir=method_dir / "parent_frozen_lats_v2", parent_id_col=p.args.parent_id_col, model_name=f"v019_{name}")
        if name == "always_final":
            baseline_parent = parent_metrics
        exit3_fraction = float(np.mean(selected_exit == 3))
        exit5_fraction = 1.0 - exit3_fraction
        average_flops = exit3_fraction * float(flops["exit3"]) + exit5_fraction * float(flops["exit5"])
        saved = 100.0 * (1.0 - average_flops / max(float(flops["exit5"]), 1.0))
        timing = execution.timing[name]
        row: dict[str, Any] = {
            "method": name,
            "decision_route": "Exit 3 -> Exit 5",
            "validation_eligible": bool(name == "always_final" or p.policy["selected_policy"]["validation_eligible"]),
            "exit3_fraction": exit3_fraction,
            "exit5_fraction": exit5_fraction,
            "average_exit_depth": float(np.mean(selected_exit)),
            "estimated_flops_saved_pct": float(saved),
            "latency_median_per_segment_ms": 1000.0 * timing["median_seconds"] / len(p.frame),
            "latency_iqr_per_segment_ms": 1000.0 * timing["iqr_seconds"] / len(p.frame),
            "measured_speedup_vs_always_final": baseline_seconds / max(timing["median_seconds"], 1e-12),
            **{f"segment_{key}": value for key, value in segment_metrics.items()},
            **{f"parent_{key}": value for key, value in parent_metrics.items() if isinstance(value, (int, float, np.integer, np.floating))},
        }
        rows.append(row)
        summaries[name] = {
            **row,
            "timing": timing,
            "single_pass_timing": execution.single.get(name),
            "policy": None if name == "always_final" else p.methods[name].to_dict(),
            "genuine_skipping_statement": "Samples accepted at Exit 3 did not execute blocks and heads required for Exits 4 and 5.",
        }
    if baseline_parent is None:
        raise RuntimeError("Always-final parent metrics were not produced.")
    constraints = p.policy["deployment_constraints"]
    for row in rows:
        row["parent_macro_f1_drop_vs_final"] = float(baseline_parent["macro_f1"] - row["parent_macro_f1"])
        row["parent_micro_f1_drop_vs_final"] = float(baseline_parent["micro_f1"] - row["parent_micro_f1"])
        row["parent_exact_drop_vs_final"] = float(baseline_parent["exact_match"] - row["parent_exact_match"])
        row["parent_hamming_increase_vs_final"] = float(row["parent_hamming_loss"] - baseline_parent["hamming_loss"])
        row["holdout_constraints_met"] = bool(
            row["parent_macro_f1_drop_vs_final"] <= float(constraints["max_parent_macro_f1_drop"])
            and row["parent_micro_f1_drop_vs_final"] <= float(constraints["max_parent_micro_f1_drop"])
            and row["parent_exact_drop_vs_final"] <= float(constraints["max_parent_exact_match_drop"])
            and row["parent_hamming_increase_vs_final"] <= float(constraints["max_parent_hamming_increase"])
        )
    comparison = pd.DataFrame(rows)
    comparison_path = p.out / "v019_targeted_holdout_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    save_json(
        {
            "experiment": "v0.19_EE_final_targeted_exit3_to_exit5",
            "comparison": [jsonable(row) for row in rows],
            "methods": summaries,
            "important_note": "Validation eligibility and corrected-holdout constraint compliance are separate fields.",
        },
        p.out / "v019_targeted_holdout_comparison.json",
    )
    columns = ["method", "validation_eligible", "holdout_constraints_met", "exit3_fraction", "estimated_flops_saved_pct", "measured_speedup_vs_always_final", "parent_macro_f1", "parent_micro_f1", "parent_exact_match", "parent_hamming_loss"]
    print("\nV0.19 final targeted holdout comparison complete")
    print("-" * 170)
    print(comparison[columns].to_string(index=False))
    print(f"Saved comparison: {comparison_path}")


def main() -> None:
    problem = prepare(parser().parse_args())
    save_results(problem, execute(problem))


if __name__ == "__main__":
    main()
