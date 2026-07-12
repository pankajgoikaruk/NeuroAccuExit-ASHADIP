#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tune and freeze the v0.11 dynamic Early-Exit policy on validation data.

The first policy allows stopping at Exit 2 only. Every sample reaches Exit 2;
a sample stops there when Exit 1 and Exit 2 agree, Exit 2 is non-empty, and
Exit 2 satisfies frozen confidence and decision-margin thresholds. All other
samples continue to Exit 3.

No model weights are changed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, jaccard_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets_multilabel import make_multilabel_loaders
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


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result:
        raise ValueError("Expected at least one comma-separated float.")
    return sorted(set(result))


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


def threshold_vector(mapping: dict[str, Any], labels: list[str]) -> np.ndarray:
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise RuntimeError(f"Threshold mapping is missing labels: {missing}")
    return np.asarray([float(mapping[label]) for label in labels], dtype=np.float32)


def load_thresholds_by_exit(
    run_dir: Path,
    labels: list[str],
    num_exits: int,
    threshold_mode: str,
    fixed_threshold: float,
) -> list[np.ndarray]:
    if threshold_mode == "fixed_0p5":
        base = np.full(len(labels), float(fixed_threshold), dtype=np.float32)
        return [base.copy() for _ in range(num_exits)]

    comparison_path = run_dir / "threshold_tuning" / "threshold_comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError(
            "Per-exit threshold file not found. Expected:\n"
            f"  {comparison_path}\n"
            "Run scripts/tune_multilabel_thresholds.py first, or use "
            "--threshold_mode fixed_0p5."
        )

    payload = load_json(comparison_path)
    exits = payload.get("exits", [])
    if len(exits) < num_exits:
        raise RuntimeError(
            f"Threshold file contains {len(exits)} exits; model has {num_exits}."
        )

    if threshold_mode == "tuned_per_exit":
        return [
            threshold_vector(exits[idx]["tuned_thresholds"], labels)
            for idx in range(num_exits)
        ]

    if threshold_mode == "final_exit_tuned":
        final_vector = threshold_vector(
            exits[num_exits - 1]["tuned_thresholds"], labels
        )
        return [final_vector.copy() for _ in range(num_exits)]

    raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")


@torch.no_grad()
def collect_outputs(model, loader, device: str) -> tuple[np.ndarray, list[np.ndarray], int]:
    model.eval()
    target_parts: list[np.ndarray] = []
    probability_parts: list[list[np.ndarray]] | None = None
    frames: int | None = None

    for x, y in loader:
        if frames is None:
            frames = int(x.shape[-1])
        logits_by_exit = model(x.to(device))
        probabilities = [
            torch.sigmoid(logits).detach().cpu().numpy()
            for logits in logits_by_exit
        ]
        if probability_parts is None:
            probability_parts = [[] for _ in probabilities]
        target_parts.append(y.detach().cpu().numpy())
        for idx, probs in enumerate(probabilities):
            probability_parts[idx].append(probs)

    if not target_parts or probability_parts is None or frames is None:
        raise RuntimeError("Validation loader produced no samples.")

    return (
        np.concatenate(target_parts, axis=0).astype(int),
        [np.concatenate(parts, axis=0).astype(np.float32) for parts in probability_parts],
        frames,
    )


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


def candidate_stop_mask(
    diagnostics: dict[str, np.ndarray],
    confidence_threshold: float,
    margin_threshold: float,
    require_agreement: bool,
    allow_empty_stop: bool,
) -> np.ndarray:
    mask = np.ones(len(diagnostics["agreement"]), dtype=bool)
    if require_agreement:
        mask &= diagnostics["agreement"]
    if not allow_empty_stop:
        mask &= diagnostics["non_empty"]
    mask &= diagnostics["mean_binary_confidence"] >= float(confidence_threshold)
    mask &= diagnostics["min_decision_margin"] >= float(margin_threshold)
    return mask


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


def load_lats_module():
    evaluator_path = PROJECT_ROOT / "scripts" / "v0.10" / "evaluate_frozen_lats_config_v010.py"
    if not evaluator_path.exists():
        raise FileNotFoundError(f"Frozen LATS evaluator not found: {evaluator_path}")
    spec = importlib.util.spec_from_file_location("v010_frozen_lats_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator module: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_level_metrics(
    metadata_df: pd.DataFrame,
    labels: list[str],
    probabilities: np.ndarray,
    lats_config_json: Path,
    parent_id_col: str,
    lats_module,
) -> dict[str, float]:
    if parent_id_col not in metadata_df.columns:
        raise RuntimeError(f"Validation manifest lacks {parent_id_col!r}.")
    frame = metadata_df[[parent_id_col, *labels]].copy()
    prefix = "dynamic_prob_"
    for idx, label in enumerate(labels):
        frame[f"{prefix}{label}"] = probabilities[:, idx]
    cfg = lats_module.load_frozen_config(lats_config_json, labels, 0.5)
    truth_df, _, pred_df = lats_module.make_parent_tables(
        df=frame,
        labels=labels,
        cfg=cfg,
        parent_id_col=parent_id_col,
        prob_prefix=prefix,
    )
    return lats_module.compute_metrics(
        truth_df[labels].to_numpy(dtype=int),
        pred_df[labels].to_numpy(dtype=int),
    )


def threshold_mapping(labels: list[str], thresholds: np.ndarray) -> dict[str, float]:
    return {label: float(thresholds[idx]) for idx, label in enumerate(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune and freeze a genuine Exit-2/Exit-3 runtime policy."
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--features_root", type=Path, default=None)
    parser.add_argument("--labels_json", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--lats_config_json", type=Path, default=None)
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument(
        "--threshold_mode",
        choices=["tuned_per_exit", "final_exit_tuned", "fixed_0p5"],
        default="tuned_per_exit",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument(
        "--confidence_grid",
        default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--margin_grid", default="0.00,0.02,0.05,0.08,0.10,0.15")
    parser.add_argument("--disable_agreement", action="store_true")
    parser.add_argument("--allow_empty_stop", action="store_true")
    parser.add_argument("--max_macro_f1_drop", type=float, default=0.02)
    parser.add_argument("--min_exit2_fraction", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    if args.split != "val":
        print(f"[WARNING] Policy tuning should normally use val, not {args.split}.")

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(run_dir)

    manifest = args.manifest.resolve() if args.manifest else Path(cfg["manifest"]).resolve()
    features_root = (
        args.features_root.resolve()
        if args.features_root
        else Path(cfg["features_root"]).resolve()
    )
    labels_json = (
        args.labels_json.resolve()
        if args.labels_json
        else Path(cfg["labels_json"]).resolve()
    )
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else run_dir / "ckpt" / "best.pt"
    )

    required = [manifest, features_root, labels_json, checkpoint]
    if args.lats_config_json:
        required.append(args.lats_config_json)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    labels = load_labels(labels_json, cfg)
    tap_blocks = parse_tap_blocks(cfg.get("tap_blocks", "1,3"))
    n_mels = int(cfg.get("n_mels", 64))
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    seed = int(cfg.get("seed", 42))

    train_loader, val_loader, test_loader, loaded_labels = make_multilabel_loaders(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        seed=seed,
        label_balance_power=0.0,
        synthetic_balance_power=0.0,
    )
    del train_loader
    if list(loaded_labels) != labels:
        raise RuntimeError("Label order mismatch between config and dataset loader.")

    loader = val_loader if args.split == "val" else test_loader
    metadata_df = loader.dataset.df.reset_index(drop=True)

    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=resolve_model_cfg(cfg),
    ).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()

    y_true, probabilities, frames = collect_outputs(model, loader, args.device)
    if len(probabilities) != 3:
        raise RuntimeError(f"Expected 3 exits, got {len(probabilities)}.")
    if len(metadata_df) != len(y_true):
        raise RuntimeError("Metadata rows do not match validation samples.")

    thresholds = load_thresholds_by_exit(
        run_dir,
        labels,
        3,
        args.threshold_mode,
        args.fixed_threshold,
    )
    exit1_probs, exit2_probs, exit3_probs = probabilities
    exit2_pred = label_predictions(exit2_probs, thresholds[1])
    exit3_pred = label_predictions(exit3_probs, thresholds[2])
    diagnostics = stop_diagnostics(
        exit1_probs,
        exit2_probs,
        thresholds[0],
        thresholds[1],
    )

    lats_module = None
    final_parent_metrics = None
    if args.lats_config_json:
        lats_module = load_lats_module()
        final_parent_metrics = parent_level_metrics(
            metadata_df,
            labels,
            exit3_probs,
            args.lats_config_json.resolve(),
            args.parent_id_col,
            lats_module,
        )

    final_segment_metrics = multilabel_metrics(y_true, exit3_pred)
    reference_macro_f1 = (
        float(final_parent_metrics["macro_f1"])
        if final_parent_metrics is not None
        else final_segment_metrics["macro_f1"]
    )

    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )
    exit2_flops = float(flops["exit2"])
    exit3_flops = float(flops["exit3"])
    require_agreement = not args.disable_agreement

    rows: list[dict[str, Any]] = []
    for confidence_threshold in parse_float_list(args.confidence_grid):
        for margin_threshold in parse_float_list(args.margin_grid):
            stop_mask = candidate_stop_mask(
                diagnostics,
                confidence_threshold,
                margin_threshold,
                require_agreement,
                args.allow_empty_stop,
            )
            selected_probs = np.where(stop_mask[:, None], exit2_probs, exit3_probs)
            selected_pred = np.where(stop_mask[:, None], exit2_pred, exit3_pred)
            segment_result = multilabel_metrics(y_true, selected_pred)
            parent_result = None
            if lats_module is not None and args.lats_config_json:
                parent_result = parent_level_metrics(
                    metadata_df,
                    labels,
                    selected_probs,
                    args.lats_config_json.resolve(),
                    args.parent_id_col,
                    lats_module,
                )

            exit2_count = int(stop_mask.sum())
            exit2_fraction = float(exit2_count / max(len(stop_mask), 1))
            avg_exit_depth = float(2 * exit2_fraction + 3 * (1 - exit2_fraction))
            average_flops = exit2_fraction * exit2_flops + (1 - exit2_fraction) * exit3_flops
            flops_saved_pct = float(100 * (1 - average_flops / max(exit3_flops, 1)))
            selection_macro_f1 = (
                float(parent_result["macro_f1"])
                if parent_result is not None
                else float(segment_result["macro_f1"])
            )
            macro_f1_drop = float(reference_macro_f1 - selection_macro_f1)
            constraint_met = bool(
                macro_f1_drop <= args.max_macro_f1_drop
                and exit2_fraction >= args.min_exit2_fraction
            )

            row = {
                "confidence_threshold": confidence_threshold,
                "margin_threshold": margin_threshold,
                "require_exit1_exit2_agreement": require_agreement,
                "allow_empty_stop": bool(args.allow_empty_stop),
                "exit2_samples": exit2_count,
                "exit3_samples": int(len(stop_mask) - exit2_count),
                "exit2_fraction": exit2_fraction,
                "avg_exit_depth": avg_exit_depth,
                "estimated_flops_saved_pct": flops_saved_pct,
                "selection_macro_f1": selection_macro_f1,
                "reference_exit3_macro_f1": reference_macro_f1,
                "macro_f1_drop": macro_f1_drop,
                "quality_constraint_met": constraint_met,
                **{f"segment_{k}": v for k, v in segment_result.items()},
            }
            if parent_result is not None:
                row.update({f"parent_{k}": v for k, v in parent_result.items()})
            rows.append(row)

    sweep_df = pd.DataFrame(rows)
    feasible = sweep_df[sweep_df["quality_constraint_met"] == True].copy()  # noqa: E712
    if not feasible.empty:
        selected_row = feasible.sort_values(
            ["estimated_flops_saved_pct", "selection_macro_f1"],
            ascending=[False, False],
        ).iloc[0]
        selection_status = "quality_constraint_met"
    else:
        selected_row = sweep_df.sort_values(
            ["selection_macro_f1", "estimated_flops_saved_pct"],
            ascending=[False, False],
        ).iloc[0]
        selection_status = "fallback_best_quality_constraint_not_met"

    sweep_path = out_dir / "v011_dynamic_policy_validation_sweep.csv"
    sweep_df.sort_values(
        ["quality_constraint_met", "estimated_flops_saved_pct", "selection_macro_f1"],
        ascending=[False, False, False],
    ).to_csv(sweep_path, index=False)

    selected_candidate = {}
    for key, value in selected_row.items():
        if isinstance(value, (bool, np.bool_)):
            selected_candidate[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            selected_candidate[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            selected_candidate[key] = float(value)
        else:
            selected_candidate[key] = value

    policy = {
        "schema_version": 1,
        "experiment": "v0.11_EE_genuine_dynamic_exit2_or_exit3",
        "selection_status": selection_status,
        "selected_on_split": args.split,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "validation_manifest": str(manifest),
        "validation_features_root": str(features_root),
        "labels_json": str(labels_json),
        "lats_config_json": str(args.lats_config_json.resolve()) if args.lats_config_json else None,
        "labels": labels,
        "architecture": {
            "model": "ExitNet/TinyAudioCNN",
            "tap_blocks": list(tap_blocks),
            "num_exits": 3,
            "n_mels": n_mels,
            "frames_observed": frames,
            "exit1_stopping_enabled": False,
            "eligible_early_exit": 2,
            "final_exit": 3,
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {
            f"exit{idx + 1}": threshold_mapping(labels, threshold)
            for idx, threshold in enumerate(thresholds)
        },
        "stop_rule": {
            "policy_type": "exit1_exit2_label_set_agreement_plus_exit2_reliability",
            "require_exit1_exit2_agreement": require_agreement,
            "allow_empty_stop": bool(args.allow_empty_stop),
            "confidence_measure": "mean(max(p, 1-p)) across labels at Exit 2",
            "confidence_threshold": float(selected_row["confidence_threshold"]),
            "margin_measure": "minimum absolute distance from Exit 2 per-label threshold",
            "margin_threshold": float(selected_row["margin_threshold"]),
            "stop_reason_exit2": "reliable_early_exit",
            "stop_reason_exit3": "final_exit",
        },
        "selection_constraints": {
            "max_absolute_macro_f1_drop": float(args.max_macro_f1_drop),
            "minimum_exit2_fraction": float(args.min_exit2_fraction),
            "reference_level": "parent_frozen_lats_v2" if final_parent_metrics else "segment",
        },
        "reference_always_exit3": {
            "segment_metrics": final_segment_metrics,
            "parent_metrics": final_parent_metrics,
            "estimated_flops": exit3_flops,
        },
        "selected_candidate": selected_candidate,
        "estimated_flops_by_exit": {k: float(v) for k, v in flops.items()},
        "important_note": "Frozen on validation data; do not retune after holdout evaluation.",
    }

    policy_path = out_dir / "frozen_dynamic_policy_v011.json"
    save_json(policy, policy_path)
    save_json(
        {
            "policy_path": str(policy_path),
            "sweep_path": str(sweep_path),
            "selection_status": selection_status,
            "selected_candidate": selected_candidate,
            "reference_exit3_segment_metrics": final_segment_metrics,
            "reference_exit3_parent_metrics": final_parent_metrics,
            "n_validation_samples": int(len(y_true)),
        },
        out_dir / "v011_dynamic_policy_validation_summary.json",
    )

    print("\nV0.11 dynamic policy tuning complete")
    print("-" * 96)
    print(f"Validation samples: {len(y_true)}")
    print(f"Selection status:   {selection_status}")
    print(f"Frozen policy:      {policy_path}")
    print(f"Sweep table:        {sweep_path}")
    print("")
    cols = [
        "confidence_threshold",
        "margin_threshold",
        "exit2_fraction",
        "selection_macro_f1",
        "macro_f1_drop",
        "estimated_flops_saved_pct",
        "quality_constraint_met",
    ]
    print(pd.DataFrame([selected_row])[cols].to_string(index=False))


if __name__ == "__main__":
    main()
