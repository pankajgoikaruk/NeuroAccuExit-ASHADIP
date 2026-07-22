#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tune and freeze the v0.12 validation-derived label-aware Early-Exit policy.

All samples reach Exit 2. The tuner derives per-label risk weights exclusively
from validation Exit-2 to Exit-3 F1 improvement, then searches transparent rule
thresholds controlling:

- Exit-1/Exit-2 label-set agreement;
- Exit-2 mean binary confidence;
- global decision margin;
- maximum inter-exit probability change; and
- maximum validation-weighted label risk.

The corrected holdout is not used in policy derivation or selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v012 import (
    collect_outputs,
    load_checkpoint,
    load_labels,
    load_lats_module,
    load_run_config,
    load_thresholds_by_exit,
    multilabel_metrics,
    parent_level_metrics,
    parse_float_list,
    parse_tap_blocks,
    resolve_model_cfg,
    save_json,
    threshold_mapping,
)
from data.datasets_multilabel import make_multilabel_loaders
from policies.label_aware_early_exit_policy import (
    LabelAwarePolicyConfig,
    compute_label_aware_diagnostics,
    derive_label_risk_profile,
    label_aware_stop_mask,
    label_predictions,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a validation-derived label-aware Exit-2/Exit-3 policy."
        )
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
        default="fixed_0p5",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument(
        "--confidence_grid",
        default="0.55,0.65,0.75,0.85,0.95",
    )
    parser.add_argument(
        "--margin_grid",
        default="0.00,0.02,0.05",
    )
    parser.add_argument(
        "--delta_grid",
        default="0.05,0.10,0.20,1.00",
    )
    parser.add_argument(
        "--risk_grid",
        default="0.10,0.25,0.50,0.75,1.00",
    )
    parser.add_argument("--minimum_label_improvement", type=float, default=0.02)
    parser.add_argument("--margin_scale", type=float, default=0.25)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--delta_weight", type=float, default=0.5)
    parser.add_argument("--disable_agreement", action="store_true")
    parser.add_argument("--allow_empty_stop", action="store_true")
    parser.add_argument("--max_macro_f1_drop", type=float, default=0.01)
    parser.add_argument("--min_exit2_fraction", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    if args.split != "val":
        print(
            "[WARNING] Label-aware policy derivation should normally use "
            f"--split val, not {args.split}."
        )

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(run_dir)

    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else Path(cfg["manifest"]).resolve()
    )
    features_root = (
        args.features_root.resolve()
        if args.features_root is not None
        else Path(cfg["features_root"]).resolve()
    )
    labels_json = (
        args.labels_json.resolve()
        if args.labels_json is not None
        else Path(cfg["labels_json"]).resolve()
    )
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else run_dir / "ckpt" / "best.pt"
    )

    required_paths = [manifest, features_root, labels_json, checkpoint]
    if args.lats_config_json is not None:
        required_paths.append(args.lats_config_json.resolve())
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

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
        raise RuntimeError(
            "Label order mismatch between label schema and dataset loader."
        )

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

    y_true, probabilities, frames = collect_outputs(
        model,
        loader,
        args.device,
    )
    if len(probabilities) != 3:
        raise RuntimeError(
            f"Expected the canonical three-exit model, got {len(probabilities)} exits."
        )
    if len(metadata_df) != len(y_true):
        raise RuntimeError(
            "Metadata rows do not match validation prediction rows."
        )

    thresholds = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=3,
        threshold_mode=args.threshold_mode,
        fixed_threshold=float(args.fixed_threshold),
    )
    exit1_probs, exit2_probs, exit3_probs = probabilities
    exit2_pred = label_predictions(exit2_probs, thresholds[1])
    exit3_pred = label_predictions(exit3_probs, thresholds[2])

    risk_profile = derive_label_risk_profile(
        labels=labels,
        y_true=y_true,
        exit2_probabilities=exit2_probs,
        exit3_probabilities=exit3_probs,
        exit2_thresholds=thresholds[1],
        exit3_thresholds=thresholds[2],
        minimum_improvement=float(args.minimum_label_improvement),
    )
    risk_weights = np.asarray(risk_profile.risk_weights, dtype=np.float32)

    lats_module = None
    final_parent_metrics = None
    if args.lats_config_json is not None:
        lats_module = load_lats_module()
        final_parent_metrics = parent_level_metrics(
            metadata_df=metadata_df,
            labels=labels,
            probabilities=exit3_probs,
            lats_config_json=args.lats_config_json.resolve(),
            parent_id_col=args.parent_id_col,
            lats_module=lats_module,
        )

    final_segment_metrics = multilabel_metrics(y_true, exit3_pred)
    reference_macro_f1 = (
        float(final_parent_metrics["macro_f1"])
        if final_parent_metrics is not None
        else float(final_segment_metrics["macro_f1"])
    )

    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )
    exit2_flops = float(flops["exit2"])
    exit3_flops = float(flops["exit3"])
    require_agreement = not bool(args.disable_agreement)

    diagnostics = compute_label_aware_diagnostics(
        exit1_probabilities=exit1_probs,
        exit2_probabilities=exit2_probs,
        exit1_thresholds=thresholds[0],
        exit2_thresholds=thresholds[1],
        risk_weights=risk_weights,
        margin_scale=float(args.margin_scale),
        margin_weight=float(args.margin_weight),
        delta_weight=float(args.delta_weight),
    )

    rows: list[dict[str, Any]] = []
    for confidence_threshold in parse_float_list(args.confidence_grid):
        for margin_threshold in parse_float_list(args.margin_grid):
            for max_probability_delta in parse_float_list(args.delta_grid):
                for label_risk_threshold in parse_float_list(args.risk_grid):
                    config = LabelAwarePolicyConfig(
                        mean_confidence_threshold=confidence_threshold,
                        global_margin_threshold=margin_threshold,
                        max_probability_delta=max_probability_delta,
                        label_risk_threshold=label_risk_threshold,
                        margin_scale=float(args.margin_scale),
                        margin_weight=float(args.margin_weight),
                        delta_weight=float(args.delta_weight),
                        require_label_set_agreement=require_agreement,
                        allow_empty_stop=bool(args.allow_empty_stop),
                    )
                    stop_mask = label_aware_stop_mask(diagnostics, config)

                    selected_probs = np.where(
                        stop_mask.reshape(-1, 1),
                        exit2_probs,
                        exit3_probs,
                    )
                    selected_pred = np.where(
                        stop_mask.reshape(-1, 1),
                        exit2_pred,
                        exit3_pred,
                    )
                    segment_result = multilabel_metrics(y_true, selected_pred)

                    parent_result = None
                    if lats_module is not None and args.lats_config_json is not None:
                        parent_result = parent_level_metrics(
                            metadata_df=metadata_df,
                            labels=labels,
                            probabilities=selected_probs,
                            lats_config_json=args.lats_config_json.resolve(),
                            parent_id_col=args.parent_id_col,
                            lats_module=lats_module,
                        )

                    exit2_count = int(stop_mask.sum())
                    exit2_fraction = float(
                        exit2_count / max(len(stop_mask), 1)
                    )
                    avg_exit_depth = float(
                        2.0 * exit2_fraction
                        + 3.0 * (1.0 - exit2_fraction)
                    )
                    average_flops = float(
                        exit2_fraction * exit2_flops
                        + (1.0 - exit2_fraction) * exit3_flops
                    )
                    flops_saved_pct = float(
                        100.0
                        * (1.0 - average_flops / max(exit3_flops, 1.0))
                    )
                    selection_macro_f1 = (
                        float(parent_result["macro_f1"])
                        if parent_result is not None
                        else float(segment_result["macro_f1"])
                    )
                    macro_f1_drop = float(
                        reference_macro_f1 - selection_macro_f1
                    )
                    quality_constraint_met = bool(
                        macro_f1_drop <= float(args.max_macro_f1_drop)
                        and exit2_fraction >= float(args.min_exit2_fraction)
                    )

                    row: dict[str, Any] = {
                        "mean_confidence_threshold": confidence_threshold,
                        "global_margin_threshold": margin_threshold,
                        "max_probability_delta": max_probability_delta,
                        "label_risk_threshold": label_risk_threshold,
                        "require_label_set_agreement": require_agreement,
                        "allow_empty_stop": bool(args.allow_empty_stop),
                        "exit2_samples": exit2_count,
                        "exit3_samples": int(len(stop_mask) - exit2_count),
                        "exit2_fraction": exit2_fraction,
                        "avg_exit_depth": avg_exit_depth,
                        "estimated_flops_saved_pct": flops_saved_pct,
                        "selection_macro_f1": selection_macro_f1,
                        "reference_exit3_macro_f1": reference_macro_f1,
                        "macro_f1_drop": macro_f1_drop,
                        "quality_constraint_met": quality_constraint_met,
                        **{
                            f"segment_{key}": value
                            for key, value in segment_result.items()
                        },
                    }
                    if parent_result is not None:
                        row.update(
                            {
                                f"parent_{key}": value
                                for key, value in parent_result.items()
                            }
                        )
                    rows.append(row)

    sweep_df = pd.DataFrame(rows)
    feasible = sweep_df[
        sweep_df["quality_constraint_met"] == True  # noqa: E712
    ].copy()

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

    selected_config = LabelAwarePolicyConfig(
        mean_confidence_threshold=float(
            selected_row["mean_confidence_threshold"]
        ),
        global_margin_threshold=float(
            selected_row["global_margin_threshold"]
        ),
        max_probability_delta=float(
            selected_row["max_probability_delta"]
        ),
        label_risk_threshold=float(
            selected_row["label_risk_threshold"]
        ),
        margin_scale=float(args.margin_scale),
        margin_weight=float(args.margin_weight),
        delta_weight=float(args.delta_weight),
        require_label_set_agreement=require_agreement,
        allow_empty_stop=bool(args.allow_empty_stop),
    )

    sweep_path = out_dir / "v012_label_aware_validation_sweep.csv"
    sweep_df.sort_values(
        [
            "quality_constraint_met",
            "estimated_flops_saved_pct",
            "selection_macro_f1",
        ],
        ascending=[False, False, False],
    ).to_csv(sweep_path, index=False)

    selected_candidate: dict[str, Any] = {}
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
        "experiment": "v0.12_EE_validation_derived_label_aware_policy",
        "selection_status": selection_status,
        "selected_on_split": args.split,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "validation_manifest": str(manifest),
        "validation_features_root": str(features_root),
        "labels_json": str(labels_json),
        "lats_config_json": (
            str(args.lats_config_json.resolve())
            if args.lats_config_json is not None
            else None
        ),
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
        "label_risk_profile": risk_profile.to_dict(),
        "stop_rule": {
            "policy_type": (
                "validation_derived_label_aware_margin_delta_risk"
            ),
            **selected_config.to_dict(),
            "stop_reason_exit2": "label_aware_reliable_early_exit",
            "stop_reason_exit3": "final_exit",
        },
        "selection_constraints": {
            "max_absolute_macro_f1_drop": float(args.max_macro_f1_drop),
            "minimum_exit2_fraction": float(args.min_exit2_fraction),
            "reference_level": (
                "parent_frozen_lats_v2"
                if final_parent_metrics is not None
                else "segment"
            ),
        },
        "reference_always_exit3": {
            "segment_metrics": final_segment_metrics,
            "parent_metrics": final_parent_metrics,
            "estimated_flops": exit3_flops,
        },
        "selected_candidate": selected_candidate,
        "estimated_flops_by_exit": {
            key: float(value) for key, value in flops.items()
        },
        "important_note": (
            "Label risk and rule thresholds were derived only from validation "
            "data. Do not alter them after corrected-holdout evaluation."
        ),
    }

    policy_path = out_dir / "frozen_label_aware_policy_v012.json"
    save_json(policy, policy_path)
    save_json(
        {
            "policy_path": str(policy_path),
            "sweep_path": str(sweep_path),
            "selection_status": selection_status,
            "selected_candidate": selected_candidate,
            "label_risk_profile": risk_profile.to_dict(),
            "reference_exit3_segment_metrics": final_segment_metrics,
            "reference_exit3_parent_metrics": final_parent_metrics,
            "n_validation_samples": int(len(y_true)),
        },
        out_dir / "v012_label_aware_validation_summary.json",
    )

    profile_df = pd.DataFrame(
        {
            "label": labels,
            "exit2_f1": risk_profile.exit2_f1,
            "exit3_f1": risk_profile.exit3_f1,
            "exit3_minus_exit2_f1": risk_profile.improvement,
            "risk_weight": risk_profile.risk_weights,
        }
    ).sort_values("risk_weight", ascending=False)
    profile_path = out_dir / "v012_validation_label_risk_profile.csv"
    profile_df.to_csv(profile_path, index=False)

    print("\nV0.12 label-aware policy tuning complete")
    print("-" * 100)
    print(f"Validation samples: {len(y_true)}")
    print(f"Selection status:   {selection_status}")
    print(f"Frozen policy:      {policy_path}")
    print(f"Sweep table:        {sweep_path}")
    print(f"Risk profile:       {profile_path}")
    print("")
    display_columns = [
        "mean_confidence_threshold",
        "global_margin_threshold",
        "max_probability_delta",
        "label_risk_threshold",
        "exit2_fraction",
        "selection_macro_f1",
        "macro_f1_drop",
        "estimated_flops_saved_pct",
        "quality_constraint_met",
    ]
    print(
        pd.DataFrame([selected_row])[display_columns].to_string(index=False)
    )
    print("\nValidation-derived label risks:")
    print(profile_df.to_string(index=False))


if __name__ == "__main__":
    main()
