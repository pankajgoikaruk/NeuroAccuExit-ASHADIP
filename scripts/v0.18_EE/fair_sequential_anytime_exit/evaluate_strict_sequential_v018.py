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

from common_v017 import (
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
from eval_v017_config import load_features, make_batches, subset_state, thresholds_from_policy
from eval_v017_reporting import evaluate_parent_lats, timing_stats
from models.anytime_exit_net import AnytimeExitNet
from policies.strict_sequential_anytime_exit_v018 import (
    StrictSequentialPolicyConfig,
    StrictSequentialStageConfig,
    stage_diagnostics,
    strict_stage_stop_mask,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a frozen v0.18 strict sequential policy.")
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


def config_from_payload(payload: dict[str, Any]) -> StrictSequentialPolicyConfig:
    stages = tuple(
        StrictSequentialStageConfig(
            mean_confidence_threshold=float(item["mean_confidence_threshold"]),
            max_probability_delta=float(item["max_probability_delta"]),
            max_label_risk=float(item["max_label_risk"]),
            risk_score_threshold=float(item["risk_score_threshold"]),
            risk_margin_multiplier=float(item["risk_margin_multiplier"]),
            risk_uncertainty_band=float(item["risk_uncertainty_band"]),
            exit1_confidence_boost=float(item["exit1_confidence_boost"]),
            per_label_margins=tuple(float(value) for value in item["per_label_margins"]),
            require_previous_label_stability=bool(item.get("require_previous_label_stability", index > 0)),
        )
        for index, item in enumerate(payload["stages"])
    )
    return StrictSequentialPolicyConfig(
        num_exits=int(payload["num_exits"]),
        stages=stages,
        allow_empty_stop=bool(payload.get("allow_empty_stop", False)),
    )


def ablation_config(config: StrictSequentialPolicyConfig, name: str) -> tuple[StrictSequentialPolicyConfig, int]:
    minimum_exit = 1
    stages: list[StrictSequentialStageConfig] = []
    for stage in config.stages:
        item = stage
        if name == "no_stability":
            item = replace(item, require_previous_label_stability=False)
        elif name == "no_risk_veto":
            item = replace(
                item,
                max_label_risk=1.0,
                risk_score_threshold=1.1,
                risk_margin_multiplier=1.0,
                risk_uncertainty_band=0.0,
                exit1_confidence_boost=0.0,
            )
        elif name == "no_label_margins":
            item = replace(item, per_label_margins=tuple(0.0 for _ in item.per_label_margins))
        elif name == "confidence_only":
            item = replace(
                item,
                max_probability_delta=1.0,
                max_label_risk=1.0,
                risk_score_threshold=1.1,
                risk_margin_multiplier=1.0,
                risk_uncertainty_band=0.0,
                exit1_confidence_boost=0.0,
                per_label_margins=tuple(0.0 for _ in item.per_label_margins),
                require_previous_label_stability=False,
            )
        stages.append(item)
    if name == "no_exit1":
        minimum_exit = 2
    return StrictSequentialPolicyConfig(config.num_exits, tuple(stages), config.allow_empty_stop), minimum_exit


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
    if policy.get("experiment") != "v0.18_EE_strict_fair_sequential_anytime_exit":
        raise RuntimeError("Expected a v0.18 frozen strict sequential policy.")
    if policy.get("labels") != labels:
        raise RuntimeError("Frozen policy label order is incompatible with the checkpoint.")
    taps = parse_tap_blocks(cfg.get("tap_blocks", "1,3"))
    exits = len(taps) + 1
    if exits != int(policy["architecture"]["num_exits"]):
        raise RuntimeError("Frozen policy exit count is incompatible with the checkpoint.")
    n_mels = int(cfg.get("n_mels", 64))
    model = build_audio_exit_net(
        num_classes=len(labels), n_mels=n_mels, tap_blocks=taps, model_cfg=resolve_model_cfg(cfg)
    ).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()
    anytime = AnytimeExitNet(model).to(args.device)
    anytime.eval()

    frame = pd.read_csv(args.holdout_manifest.resolve(), low_memory=False).reset_index(drop=True)
    tensors = load_features(frame, args.features_root.resolve())
    batches = make_batches(len(frame), int(args.batch_size))
    thresholds = thresholds_from_policy(policy, labels)
    config = config_from_payload(policy["selected_policy"]["parameters"])
    risk_scores = np.asarray(policy["strict_risk_design"]["risk_scores"], dtype=np.float32)
    y = frame[labels].astype(int).to_numpy(np.int8)
    methods: dict[str, dict[str, Any]] = {}
    for name in ["full_strict", "no_exit1", "no_risk_veto", "no_stability", "no_label_margins", "confidence_only"]:
        if name == "full_strict":
            method_config, minimum_exit = config, 1
        else:
            method_config, minimum_exit = ablation_config(config, name)
        methods[name] = {
            "config": method_config,
            "minimum_exit": minimum_exit,
            "validation_eligible": bool(policy["selected_policy"]["deployment_eligible"] if name == "full_strict" else False),
        }
    return SimpleNamespace(
        args=args, run=run, out=out, checkpoint=checkpoint, cfg=cfg, labels=labels,
        policy=policy, taps=taps, exits=exits, n_mels=n_mels, model=anytime,
        frame=frame, tensors=tensors, batches=batches, thresholds=thresholds,
        risk_scores=risk_scores, methods=methods, y=y,
    )


def run_always_final(*, p: SimpleNamespace, collect: bool) -> tuple[dict[str, np.ndarray] | None, float]:
    output = None if not collect else {
        "selected_probabilities": np.zeros((len(p.tensors), len(p.labels)), np.float32),
        "selected_exit": np.full(len(p.tensors), p.exits, np.int8),
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


def run_strict(*, p: SimpleNamespace, config: StrictSequentialPolicyConfig, minimum_exit: int, collect: bool) -> tuple[dict[str, np.ndarray] | None, dict[str, float]]:
    output = None if not collect else {
        "selected_probabilities": np.zeros((len(p.tensors), len(p.labels)), np.float32),
        "selected_exit": np.full(len(p.tensors), p.exits, np.int8),
    }
    model_seconds = 0.0
    policy_seconds = 0.0
    with torch.no_grad():
        for global_indices in p.batches:
            x = torch.cat([p.tensors[int(index)] for index in global_indices], dim=0).to(p.args.device)
            active = np.asarray(global_indices, np.int64)
            previous = None
            synchronize(p.args.device)
            started = time.perf_counter()
            logits, state = p.model.start(x)
            synchronize(p.args.device)
            model_seconds += time.perf_counter() - started
            for exit_index in range(p.exits):
                exit_no = exit_index + 1
                probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                if exit_no == p.exits:
                    if output is not None:
                        output["selected_probabilities"][active] = probabilities
                        output["selected_exit"][active] = exit_no
                    break
                policy_started = time.perf_counter()
                diagnostic = stage_diagnostics(
                    current_probabilities=probabilities,
                    current_thresholds=p.thresholds[exit_index],
                    previous_probabilities=previous,
                    previous_thresholds=None if exit_index == 0 else p.thresholds[exit_index - 1],
                )
                if exit_no < int(minimum_exit):
                    stop = np.zeros(len(active), dtype=bool)
                else:
                    stop = strict_stage_stop_mask(
                        diagnostic=diagnostic,
                        stage=config.stages[exit_index],
                        risk_scores=p.risk_scores[exit_index],
                        exit_index=exit_index,
                        allow_empty_stop=config.allow_empty_stop,
                    )
                policy_seconds += time.perf_counter() - policy_started
                if output is not None:
                    output["selected_probabilities"][active[stop]] = probabilities[stop]
                    output["selected_exit"][active[stop]] = exit_no
                continuing = np.flatnonzero(~stop)
                if len(continuing) == 0:
                    break
                state = subset_state(
                    state,
                    torch.as_tensor(continuing, dtype=torch.long, device=state.feature_map.device),
                )
                active = active[continuing]
                previous = probabilities[continuing]
                synchronize(p.args.device)
                started = time.perf_counter()
                logits, state = p.model.continue_from(state)
                synchronize(p.args.device)
                model_seconds += time.perf_counter() - started
    return output, {
        "model_seconds": float(model_seconds),
        "policy_seconds": float(policy_seconds),
        "total_seconds": float(model_seconds + policy_seconds),
    }


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
                item = p.methods[name]
                _, timing = run_strict(p=p, config=item["config"], minimum_exit=item["minimum_exit"], collect=False)
                seconds = timing["total_seconds"]
            timings[name].append(float(seconds))

    outputs = {"always_final": always_output}
    single: dict[str, dict[str, float]] = {}
    for name, item in p.methods.items():
        output, timing = run_strict(p=p, config=item["config"], minimum_exit=item["minimum_exit"], collect=True)
        outputs[name] = output
        single[name] = timing
    return SimpleNamespace(
        outputs=outputs,
        single=single,
        timing={name: timing_stats(values) for name, values in timings.items()},
    )


def save_results(p: SimpleNamespace, execution: SimpleNamespace) -> None:
    args = p.args
    baseline_seconds = execution.timing["always_final"]["median_seconds"]
    flops = estimate_flops_tiny_audiocnn(
        n_mels=p.n_mels,
        frames=int(p.tensors[0].shape[-1]),
        num_classes=len(p.labels),
        tap_blocks=p.taps,
    )
    rows: list[dict[str, Any]] = []
    baseline_parent: dict[str, Any] | None = None
    for name, output in execution.outputs.items():
        method_dir = p.out / name
        method_dir.mkdir(parents=True, exist_ok=True)
        probabilities = output["selected_probabilities"]
        selected_exit = output["selected_exit"]
        predictions = np.zeros_like(p.y, np.int8)
        fractions: dict[int, float] = {}
        for exit_no in range(1, p.exits + 1):
            mask = selected_exit == exit_no
            fractions[exit_no] = float(np.mean(mask))
            if np.any(mask):
                predictions[mask] = (
                    probabilities[mask] >= p.thresholds[exit_no - 1].reshape(1, -1)
                ).astype(np.int8)
        segment_metrics = multilabel_metrics(p.y, predictions)
        frame = p.frame[[args.parent_id_col, "feat_relpath", *p.labels]].copy()
        for index, label in enumerate(p.labels):
            frame[f"dynamic_prob_{label}"] = probabilities[:, index]
            frame[f"dynamic_pred_{label}"] = predictions[:, index]
        frame["selected_exit"] = selected_exit
        frame["continuation_reason"] = np.where(
            selected_exit == p.exits,
            "reached_final_exit",
            np.asarray([f"stopped_at_exit{value}" for value in selected_exit]),
        )
        segment_csv = method_dir / "segment_predictions.csv"
        frame.to_csv(segment_csv, index=False)
        parent_metrics = evaluate_parent_lats(
            segment_csv=segment_csv,
            labels_json=args.labels_json.resolve(),
            lats_config_json=args.lats_config_json.resolve(),
            out_dir=method_dir / "parent_frozen_lats_v2",
            parent_id_col=args.parent_id_col,
            model_name=f"v018_{p.exits}exit_{name}",
        )
        if name == "always_final":
            baseline_parent = parent_metrics
        average_flops = sum(fractions[exit_no] * float(flops[f"exit{exit_no}"]) for exit_no in fractions)
        saved = 100.0 * (1.0 - average_flops / max(float(flops[f"exit{p.exits}"]), 1.0))
        timing = execution.timing[name]
        row: dict[str, Any] = {
            "architecture": f"{p.exits}-exit",
            "method": name,
            "validation_eligible": bool(name == "always_final" or p.methods[name]["validation_eligible"]),
            "total_early_fraction": 1.0 - fractions[p.exits],
            "average_exit_depth": float(np.mean(selected_exit)),
            "estimated_flops_saved_pct": float(saved),
            "latency_median_per_segment_ms": 1000.0 * timing["median_seconds"] / len(p.frame),
            "latency_iqr_per_segment_ms": 1000.0 * timing["iqr_seconds"] / len(p.frame),
            "measured_speedup_vs_always_final": baseline_seconds / max(timing["median_seconds"], 1e-12),
            **{f"segment_{key}": value for key, value in segment_metrics.items()},
            **{f"parent_{key}": value for key, value in parent_metrics.items() if isinstance(value, (int, float, np.integer, np.floating))},
        }
        for exit_no in range(1, p.exits + 1):
            row[f"exit{exit_no}_fraction"] = fractions[exit_no]
        rows.append(row)

    if baseline_parent is None:
        raise RuntimeError("Always-final parent metrics were not produced.")
    constraints = p.policy["selection_constraints"]
    for row in rows:
        row["parent_macro_f1_drop_vs_own_final"] = float(baseline_parent["macro_f1"] - row["parent_macro_f1"])
        row["parent_micro_f1_drop_vs_own_final"] = float(baseline_parent["micro_f1"] - row["parent_micro_f1"])
        row["parent_exact_drop_vs_own_final"] = float(baseline_parent["exact_match"] - row["parent_exact_match"])
        row["parent_hamming_increase_vs_own_final"] = float(row["parent_hamming_loss"] - baseline_parent["hamming_loss"])
        row["holdout_constraints_met"] = bool(
            row["parent_macro_f1_drop_vs_own_final"] <= float(constraints["max_parent_macro_f1_drop"]) and
            row["parent_micro_f1_drop_vs_own_final"] <= float(constraints["max_parent_micro_f1_drop"]) and
            row["parent_exact_drop_vs_own_final"] <= float(constraints["max_parent_exact_match_drop"]) and
            row["parent_hamming_increase_vs_own_final"] <= float(constraints["max_parent_hamming_increase"])
        )
        method_dir = p.out / str(row["method"])
        summary = {
            **row,
            "timing": execution.timing[str(row["method"])],
            "single_pass_timing": execution.single.get(str(row["method"])),
            "policy": None if row["method"] == "always_final" else {
                "parameters": p.methods[str(row["method"])]["config"].to_dict(),
                "minimum_exit": p.methods[str(row["method"])]["minimum_exit"],
            },
            "genuine_skipping_statement": "Each sample stopped at its selected exit and did not execute later blocks.",
        }
        save_json(summary, method_dir / "runtime_summary.json")

    comparison = pd.DataFrame(rows)
    comparison_path = p.out / f"v018_{p.exits}exit_holdout_comparison.csv"
    ablation_path = p.out / f"v018_{p.exits}exit_ablation_table.csv"
    comparison.to_csv(comparison_path, index=False)
    comparison[comparison["method"] != "always_final"].to_csv(ablation_path, index=False)
    save_json(
        {
            "experiment": "v0.18_EE_strict_fair_sequential_anytime_exit",
            "architecture": f"{p.exits}-exit",
            "comparison": [jsonable(row) for row in rows],
            "important_note": "Validation eligibility and corrected-holdout constraint compliance are reported separately.",
        },
        p.out / f"v018_{p.exits}exit_holdout_comparison.json",
    )
    columns = [
        "architecture", "method", "validation_eligible", "holdout_constraints_met",
        *[f"exit{exit_no}_fraction" for exit_no in range(1, p.exits + 1)],
        "estimated_flops_saved_pct", "measured_speedup_vs_always_final",
        "parent_macro_f1", "parent_micro_f1", "parent_exact_match", "parent_hamming_loss",
    ]
    print(f"\nV0.18 strict sequential {p.exits}-exit holdout comparison complete")
    print("-" * 180)
    print(comparison[columns].to_string(index=False))
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved ablation: {ablation_path}")


def main() -> None:
    problem = prepare(parser().parse_args())
    save_results(problem, execute(problem))


if __name__ == "__main__":
    main()
