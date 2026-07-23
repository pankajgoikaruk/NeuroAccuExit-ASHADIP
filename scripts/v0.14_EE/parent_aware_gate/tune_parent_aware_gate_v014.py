#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tune parent-aware label-adaptive gates with parent-grouped OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v014 import (
    adaptive_candidate_result,
    collect_outputs,
    fit_label_gate_models,
    grouped_oof_gate_probabilities,
    jsonable,
    load_checkpoint,
    load_json,
    load_labels,
    load_lats_module,
    load_run_config,
    load_thresholds_by_exit,
    multilabel_metrics,
    parent_level_metrics,
    parse_float_list,
    parse_tap_blocks,
    resolve_model_cfg,
    robust_drop_statistics,
    save_json,
    select_robust_candidate,
    threshold_mapping,
)
from data.datasets_multilabel import make_multilabel_loaders
from policies.parent_aware_adaptive_gate import (
    adaptive_label_stop_mask,
    build_parent_aware_features,
    counterfactual_parent_unsafe_targets,
    derive_label_probability_thresholds,
    label_predictions,
    parse_lats_rules,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


TRANSITIONS = {
    "parent_gate_exit2_to_exit3": {"source_exit": 2, "deeper_exit": 3},
    "parent_gate_exit1_to_exit3_ablation": {"source_exit": 1, "deeper_exit": 3},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune parent-aware, per-label adaptive gates using grouped OOF validation."
        )
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--features_root", type=Path, default=None)
    parser.add_argument("--labels_json", type=Path, default=None)
    parser.add_argument("--lats_config_json", required=True, type=Path)
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument(
        "--threshold_mode",
        choices=["tuned_per_exit", "final_exit_tuned", "fixed_0p5"],
        default="fixed_0p5",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unsafe_recall_grid", default="0.80,0.90,0.95,0.98"
    )
    parser.add_argument(
        "--threshold_scale_grid", default="0.75,1.00,1.25"
    )
    parser.add_argument(
        "--expected_harm_grid", default="0.10,0.20,0.30,0.50,1.01"
    )
    parser.add_argument("--minimum_unsafe_examples", type=int, default=3)
    parser.add_argument("--max_macro_f1_drop", type=float, default=0.01)
    parser.add_argument("--min_source_fraction", type=float, default=0.01)
    parser.add_argument("--one_sided_z", type=float, default=1.645)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

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
    lats_config_json = args.lats_config_json.resolve()
    for path in (manifest, features_root, labels_json, checkpoint, lats_config_json):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    labels = load_labels(labels_json, cfg)
    tap_blocks = parse_tap_blocks(cfg.get("tap_blocks", "1,3"))
    n_mels = int(cfg.get("n_mels", 64))
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    loader_seed = int(cfg.get("seed", 42))

    train_loader, val_loader, test_loader, loaded_labels = make_multilabel_loaders(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=labels_json,
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        seed=loader_seed,
        label_balance_power=0.0,
        synthetic_balance_power=0.0,
    )
    del train_loader, test_loader
    if list(loaded_labels) != labels:
        raise RuntimeError("Label order mismatch between schema and loader.")

    metadata_df = val_loader.dataset.df.reset_index(drop=True)
    if args.parent_id_col not in metadata_df.columns:
        raise RuntimeError(f"Validation manifest lacks {args.parent_id_col!r}.")
    parent_ids = metadata_df[args.parent_id_col].astype(str).to_numpy()

    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=resolve_model_cfg(cfg),
    ).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()
    y_true, probabilities, frames = collect_outputs(model, val_loader, args.device)
    if len(probabilities) != 3:
        raise RuntimeError(f"Expected 3 exits, got {len(probabilities)}.")
    p1, p2, p3 = probabilities
    thresholds = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=3,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
    )
    predictions = [
        label_predictions(probabilities[idx], thresholds[idx]) for idx in range(3)
    ]

    lats_payload = load_json(lats_config_json)
    rules = parse_lats_rules(lats_payload, labels)
    lats_module = load_lats_module()
    reference_segment = multilabel_metrics(y_true, predictions[2])
    reference_parent = parent_level_metrics(
        metadata_df=metadata_df,
        labels=labels,
        probabilities=p3,
        lats_config_json=lats_config_json,
        parent_id_col=args.parent_id_col,
        lats_module=lats_module,
    )
    reference_macro_f1 = float(reference_parent["macro_f1"])

    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )

    selected_policies: dict[str, Any] = {}
    all_sweep_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_assignment_frame = metadata_df[[args.parent_id_col]].copy()

    for transition_name, transition in TRANSITIONS.items():
        source_exit = int(transition["source_exit"])
        source_idx = source_exit - 1
        source_probs = probabilities[source_idx]
        source_pred = predictions[source_idx]
        previous_probs = p1 if source_exit == 2 else None

        features, feature_names, diagnostics = build_parent_aware_features(
            current_probabilities=source_probs,
            previous_probabilities=previous_probs,
            parent_ids=parent_ids,
            current_thresholds=thresholds[source_idx],
            rules=rules,
        )
        unsafe_targets, baseline_parent_by_row, counterfactual_parent_by_row = (
            counterfactual_parent_unsafe_targets(
                y_true=y_true,
                source_probabilities=source_probs,
                deeper_probabilities=p3,
                parent_ids=parent_ids,
                rules=rules,
            )
        )
        del baseline_parent_by_row, counterfactual_parent_by_row
        oof_probs, fold_index, fold_records = grouped_oof_gate_probabilities(
            features=features,
            targets=unsafe_targets,
            groups=parent_ids,
            n_splits=args.cv_folds,
            seed=args.seed + source_exit * 1000,
        )
        fold_assignment_frame[f"{transition_name}_fold"] = fold_index
        final_models = fit_label_gate_models(
            features=features,
            targets=unsafe_targets,
            seed=args.seed + source_exit * 10000,
        )
        model_path = out_dir / f"{transition_name}_gate_v014.joblib"
        joblib.dump(
            {
                "models": final_models,
                "feature_names": feature_names,
                "labels": labels,
                "transition": transition,
                "target_definition": (
                    "unsafe per label when substituting the source exit for one "
                    "segment makes a correct all-Exit3 parent prediction wrong"
                ),
            },
            model_path,
        )

        oof_frame = metadata_df[[args.parent_id_col]].copy()
        for label_idx, label in enumerate(labels):
            oof_frame[f"unsafe_target_{label}"] = unsafe_targets[:, label_idx]
            oof_frame[f"unsafe_probability_{label}"] = oof_probs[:, label_idx]
        oof_frame["fold"] = fold_index
        oof_frame.to_csv(out_dir / f"{transition_name}_oof_predictions.csv", index=False)

        source_flops = float(flops[f"exit{source_exit}"])
        deeper_flops = float(flops["exit3"])
        strategy_rows: list[dict[str, Any]] = []

        for target_recall in parse_float_list(args.unsafe_recall_grid):
            base_thresholds, positive_counts, used_fallback = (
                derive_label_probability_thresholds(
                    unsafe_targets=unsafe_targets,
                    unsafe_probabilities=oof_probs,
                    target_recall=target_recall,
                    minimum_positive_examples=args.minimum_unsafe_examples,
                )
            )
            for threshold_scale in parse_float_list(args.threshold_scale_grid):
                label_thresholds = np.clip(
                    base_thresholds * float(threshold_scale), 0.01, 0.99
                ).astype(np.float32)
                for expected_harm in parse_float_list(args.expected_harm_grid):
                    harm_threshold = None if expected_harm > 1.0 else expected_harm
                    stop_mask, expected_harm_score, highest_risk_label = (
                        adaptive_label_stop_mask(
                            unsafe_probabilities=oof_probs,
                            label_thresholds=label_thresholds,
                            expected_harm_threshold=harm_threshold,
                            non_empty=diagnostics["non_empty"],
                            allow_empty_stop=False,
                        )
                    )
                    parameters = {
                        "unsafe_recall_target": target_recall,
                        "threshold_scale": threshold_scale,
                        "label_probability_thresholds": {
                            label: float(label_thresholds[idx])
                            for idx, label in enumerate(labels)
                        },
                        "unsafe_positive_counts": {
                            label: int(positive_counts[idx])
                            for idx, label in enumerate(labels)
                        },
                        "used_pooled_fallback": {
                            label: bool(used_fallback[idx])
                            for idx, label in enumerate(labels)
                        },
                        "expected_harm_threshold": harm_threshold,
                        "threshold_type": (
                            "per-label unsafe-probability thresholds plus "
                            "optional expected-parent-harm threshold"
                        ),
                    }
                    row = adaptive_candidate_result(
                        strategy=transition_name,
                        parameters=parameters,
                        stop_mask=stop_mask,
                        y_true=y_true,
                        source_probabilities=source_probs,
                        deeper_probabilities=p3,
                        source_predictions=source_pred,
                        deeper_predictions=predictions[2],
                        metadata_df=metadata_df,
                        labels=labels,
                        lats_config_json=lats_config_json,
                        parent_id_col=args.parent_id_col,
                        lats_module=lats_module,
                        reference_macro_f1=reference_macro_f1,
                        max_macro_f1_drop=args.max_macro_f1_drop,
                        min_source_fraction=args.min_source_fraction,
                        source_exit_no=source_exit,
                        source_flops=source_flops,
                        deeper_flops=deeper_flops,
                    )

                    fold_drops: list[float] = []
                    fold_micro: list[float] = []
                    for fold_no in np.unique(fold_index):
                        fold_mask = fold_index == fold_no
                        fold_df = metadata_df.loc[fold_mask].reset_index(drop=True)
                        fold_reference = parent_level_metrics(
                            metadata_df=fold_df,
                            labels=labels,
                            probabilities=p3[fold_mask],
                            lats_config_json=lats_config_json,
                            parent_id_col=args.parent_id_col,
                            lats_module=lats_module,
                        )
                        fold_selected_probs = np.where(
                            stop_mask[fold_mask].reshape(-1, 1),
                            source_probs[fold_mask],
                            p3[fold_mask],
                        )
                        fold_candidate = parent_level_metrics(
                            metadata_df=fold_df,
                            labels=labels,
                            probabilities=fold_selected_probs,
                            lats_config_json=lats_config_json,
                            parent_id_col=args.parent_id_col,
                            lats_module=lats_module,
                        )
                        fold_drops.append(
                            float(fold_reference["macro_f1"])
                            - float(fold_candidate["macro_f1"])
                        )
                        fold_micro.append(float(fold_candidate["micro_f1"]))
                    robust = robust_drop_statistics(
                        fold_drops, one_sided_z=args.one_sided_z
                    )
                    row.update(robust)
                    row["fold_parent_micro_f1_mean"] = float(np.mean(fold_micro))
                    row["robust_quality_constraint_met"] = bool(
                        row["base_quality_constraint_met"]
                        and robust["fold_macro_f1_drop_upper_confidence"]
                        <= float(args.max_macro_f1_drop) + 1e-12
                    )
                    row["oof_unsafe_target_fraction"] = float(unsafe_targets.mean())
                    row["oof_expected_harm_mean"] = float(expected_harm_score.mean())
                    row["oof_highest_risk_label_mode"] = int(
                        np.bincount(highest_risk_label).argmax()
                    )
                    strategy_rows.append(row)

        strategy_df = pd.DataFrame(strategy_rows)
        selected_row, selection_status = select_robust_candidate(strategy_df)
        selected_parameters = json.loads(str(selected_row["parameters_json"]))
        selected_policies[transition_name] = {
            "selection_status": selection_status,
            "source_exit": source_exit,
            "deeper_exit": 3,
            "gate_model_filename": model_path.name,
            "feature_names": feature_names,
            "parameters": selected_parameters,
            "selection_metrics": {
                key: jsonable(value)
                for key, value in selected_row.items()
                if key != "parameters_json"
            },
            "fold_records": fold_records,
            "unsafe_target_counts": {
                label: int(unsafe_targets[:, idx].sum())
                for idx, label in enumerate(labels)
            },
        }
        selected_rows.append(
            {
                **{
                    key: jsonable(value)
                    for key, value in selected_row.items()
                    if key != "parameters_json"
                },
                "selection_status": selection_status,
            }
        )
        all_sweep_rows.extend(strategy_rows)

        target_summary = pd.DataFrame(
            {
                "label": labels,
                "unsafe_parent_counterfactual_count": unsafe_targets.sum(axis=0),
                "unsafe_parent_counterfactual_rate": unsafe_targets.mean(axis=0),
            }
        )
        target_summary.to_csv(
            out_dir / f"{transition_name}_unsafe_target_summary.csv", index=False
        )

    fold_assignment_frame.to_csv(
        out_dir / "v014_validation_parent_folds.csv", index=False
    )
    sweep_df = pd.DataFrame(all_sweep_rows)
    sweep_path = out_dir / "v014_parent_aware_gate_validation_sweep.csv"
    sweep_df.sort_values(
        [
            "strategy",
            "robust_quality_constraint_met",
            "estimated_flops_saved_pct",
            "selection_macro_f1",
        ],
        ascending=[True, False, False, False],
    ).to_csv(sweep_path, index=False)
    selected_path = out_dir / "v014_selected_parent_aware_policies.csv"
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)

    frozen = {
        "schema_version": 1,
        "experiment": "v0.14_EE_parent_aware_cross_validated_adaptive_gate",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "validation_manifest": str(manifest),
        "validation_features_root": str(features_root),
        "labels_json": str(labels_json),
        "lats_config_json": str(lats_config_json),
        "labels": labels,
        "architecture": {
            "model": "ExitNet/TinyAudioCNN",
            "tap_blocks": list(tap_blocks),
            "num_exits": 3,
            "n_mels": n_mels,
            "frames_observed": frames,
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {
            f"exit{idx + 1}": threshold_mapping(labels, threshold)
            for idx, threshold in enumerate(thresholds)
        },
        "validation_protocol": {
            "method": "parent-grouped out-of-fold prediction",
            "cv_folds": int(args.cv_folds),
            "one_sided_z": float(args.one_sided_z),
            "robust_constraint": (
                "overall parent Macro-F1 drop and one-sided fold-drop upper "
                "confidence bound must both remain within the limit"
            ),
            "max_macro_f1_drop": float(args.max_macro_f1_drop),
            "minimum_source_exit_fraction": float(args.min_source_fraction),
            "parent_id_col": args.parent_id_col,
        },
        "gate_design": {
            "model": (
                "one class-balanced logistic regression per label; constant "
                "probability fallback when a label target has one class"
            ),
            "target": (
                "parent-level counterfactual harm relative to all-Exit3 frozen "
                "LATS-v2 prediction"
            ),
            "decision": (
                "label-specific unsafe-probability thresholds; continue when "
                "any label crosses its own threshold or expected harm exceeds "
                "the selected budget threshold"
            ),
        },
        "reference_always_exit3_validation": {
            "segment_metrics": reference_segment,
            "parent_metrics": reference_parent,
        },
        "selected_policies": selected_policies,
        "estimated_flops_by_exit": {
            key: float(value) for key, value in flops.items()
        },
        "important_note": (
            "The final gate models were fitted on validation only after OOF "
            "threshold selection. The corrected holdout must not be used to "
            "alter model parameters or label-specific thresholds."
        ),
    }
    frozen_path = out_dir / "frozen_parent_aware_gate_v014.json"
    save_json(frozen, frozen_path)

    print("\nV0.14 parent-aware gate tuning complete")
    print("-" * 118)
    print(f"Validation segments: {len(y_true)}")
    print(f"Validation parents:  {metadata_df[args.parent_id_col].nunique()}")
    print(f"Frozen policy:       {frozen_path}")
    print(f"Validation sweep:    {sweep_path}")
    display_columns = [
        "strategy",
        "selection_status",
        "source_exit_fraction",
        "estimated_flops_saved_pct",
        "selection_macro_f1",
        "parent_micro_f1",
        "parent_exact_match",
        "macro_f1_drop",
        "fold_macro_f1_drop_upper_confidence",
        "robust_quality_constraint_met",
    ]
    print("\nSelected adaptive policies:")
    print(pd.DataFrame(selected_rows)[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
