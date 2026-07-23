#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tune five Early-Exit strategies under one matched validation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v013 import (
    candidate_result,
    collect_outputs,
    jsonable,
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
    select_candidate,
    threshold_mapping,
)
from data.datasets_multilabel import make_multilabel_loaders
from policies.early_exit_strategy_comparison import (
    GlobalRuleConfig,
    LabelRiskRuleConfig,
    PerLabelMarginConfig,
    build_gate_features,
    compute_common_diagnostics,
    derive_per_label_margin_thresholds,
    gate_safe_targets,
    global_rule_stop_mask,
    label_predictions,
    label_risk_stop_mask,
    logistic_gate_stop_mask,
    per_label_margin_stop_mask,
    split_parent_ids,
)
from policies.label_aware_early_exit_policy import derive_label_risk_profile
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


STRATEGIES = (
    "global_conf_margin",
    "global_conf_margin_delta",
    "label_risk",
    "per_label_margin",
    "logistic_gate",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune matched global, label-aware, and learned gate policies."
        )
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--features_root", type=Path, default=None)
    parser.add_argument("--labels_json", type=Path, default=None)
    parser.add_argument("--lats_config_json", type=Path, default=None)
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument(
        "--threshold_mode",
        choices=[
            "tuned_per_exit",
            "final_exit_tuned",
            "fixed_0p5",
        ],
        default="fixed_0p5",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--derivation_fraction", type=float, default=0.70)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--confidence_grid",
        default="0.55,0.65,0.75,0.85,0.95",
    )
    parser.add_argument(
        "--margin_grid",
        default="0.00,0.02,0.05,0.08",
    )
    parser.add_argument(
        "--delta_grid",
        default="0.05,0.10,0.20,1.00",
    )
    parser.add_argument(
        "--risk_grid",
        default="0.10,0.25,0.50,0.75,1.00",
    )
    parser.add_argument(
        "--capture_grid",
        default="0.25,0.50,0.75,0.90",
    )
    parser.add_argument(
        "--gate_threshold_grid",
        default=(
            "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
            "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"
        ),
    )
    parser.add_argument(
        "--minimum_label_improvement",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--minimum_corrected_examples",
        type=int,
        default=3,
    )
    parser.add_argument("--risk_margin_scale", type=float, default=0.25)
    parser.add_argument("--risk_margin_weight", type=float, default=0.5)
    parser.add_argument("--risk_delta_weight", type=float, default=0.5)
    parser.add_argument("--max_macro_f1_drop", type=float, default=0.01)
    parser.add_argument("--min_exit2_fraction", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(run_dir)

    manifest = (
        args.manifest.resolve()
        if args.manifest
        else Path(cfg["manifest"]).resolve()
    )
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
        required.append(args.lats_config_json.resolve())
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    labels = load_labels(labels_json, cfg)
    tap_blocks = parse_tap_blocks(cfg.get("tap_blocks", "1,3"))
    n_mels = int(cfg.get("n_mels", 64))
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    loader_seed = int(cfg.get("seed", 42))

    train_loader, val_loader, test_loader, loaded_labels = (
        make_multilabel_loaders(
            manifest_csv=manifest,
            features_root=features_root,
            labels_json=labels_json,
            batch_size=batch_size,
            num_workers=int(args.num_workers),
            seed=loader_seed,
            label_balance_power=0.0,
            synthetic_balance_power=0.0,
        )
    )
    del train_loader, test_loader
    if list(loaded_labels) != labels:
        raise RuntimeError(
            "Label order mismatch between schema and loader."
        )

    metadata_df = val_loader.dataset.df.reset_index(drop=True)
    if args.parent_id_col not in metadata_df.columns:
        raise RuntimeError(
            f"Validation manifest lacks {args.parent_id_col!r}."
        )

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
        val_loader,
        args.device,
    )
    if len(probabilities) != 3:
        raise RuntimeError(
            f"Expected 3 exits, got {len(probabilities)}."
        )
    if len(metadata_df) != len(y_true):
        raise RuntimeError(
            "Validation metadata and prediction rows differ."
        )

    thresholds = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=3,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
    )
    p1, p2, p3 = probabilities
    pred2 = label_predictions(p2, thresholds[1])
    pred3 = label_predictions(p3, thresholds[2])

    derivation_mask, selection_mask = split_parent_ids(
        metadata_df[args.parent_id_col].astype(str).tolist(),
        derivation_fraction=args.derivation_fraction,
        seed=args.split_seed,
    )
    derivation_df = metadata_df.loc[
        derivation_mask
    ].reset_index(drop=True)
    selection_df = metadata_df.loc[
        selection_mask
    ].reset_index(drop=True)

    split_assignments = metadata_df[[args.parent_id_col]].copy()
    split_assignments["v013_role"] = np.where(
        derivation_mask,
        "derivation_train",
        "policy_selection",
    )
    split_assignments.to_csv(
        out_dir / "v013_validation_parent_split.csv",
        index=False,
    )

    risk_profile = derive_label_risk_profile(
        labels=labels,
        y_true=y_true[derivation_mask],
        exit2_probabilities=p2[derivation_mask],
        exit3_probabilities=p3[derivation_mask],
        exit2_thresholds=thresholds[1],
        exit3_thresholds=thresholds[2],
        minimum_improvement=args.minimum_label_improvement,
    )
    risk_weights = np.asarray(
        risk_profile.risk_weights,
        dtype=np.float32,
    )

    diagnostics = compute_common_diagnostics(
        exit1_probabilities=p1,
        exit2_probabilities=p2,
        exit1_thresholds=thresholds[0],
        exit2_thresholds=thresholds[1],
        risk_weights=risk_weights,
        risk_margin_scale=args.risk_margin_scale,
        risk_margin_weight=args.risk_margin_weight,
        risk_delta_weight=args.risk_delta_weight,
    )
    selection_diagnostics = {
        key: np.asarray(value)[selection_mask]
        for key, value in diagnostics.items()
    }

    gate_features, gate_feature_names = build_gate_features(
        exit1_probabilities=p1,
        exit2_probabilities=p2,
        exit1_thresholds=thresholds[0],
        exit2_thresholds=thresholds[1],
    )
    gate_targets, exit3_error_improvement = gate_safe_targets(
        y_true=y_true,
        exit2_predictions=pred2,
        exit3_predictions=pred3,
    )
    if len(np.unique(gate_targets[derivation_mask])) < 2:
        raise RuntimeError(
            "Gate derivation subset contains only one target class."
        )

    gate_model = Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=args.split_seed,
                ),
            ),
        ]
    )
    gate_model.fit(
        gate_features[derivation_mask],
        gate_targets[derivation_mask],
    )
    gate_model_path = out_dir / "logistic_gate_v013.joblib"
    joblib.dump(gate_model, gate_model_path)
    gate_safe_probability = gate_model.predict_proba(
        gate_features
    )[:, 1].astype(np.float32)

    lats_module = (
        load_lats_module()
        if args.lats_config_json
        else None
    )
    lats_config = (
        args.lats_config_json.resolve()
        if args.lats_config_json
        else None
    )
    reference_segment = multilabel_metrics(
        y_true[selection_mask],
        pred3[selection_mask],
    )
    reference_parent = None
    if lats_module is not None and lats_config is not None:
        reference_parent = parent_level_metrics(
            metadata_df=selection_df,
            labels=labels,
            probabilities=p3[selection_mask],
            lats_config_json=lats_config,
            parent_id_col=args.parent_id_col,
            lats_module=lats_module,
        )
    reference_macro_f1 = (
        float(reference_parent["macro_f1"])
        if reference_parent is not None
        else float(reference_segment["macro_f1"])
    )

    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )
    exit2_flops = float(flops["exit2"])
    exit3_flops = float(flops["exit3"])

    common_candidate_kwargs = dict(
        y_true=y_true[selection_mask],
        exit2_probabilities=p2[selection_mask],
        exit3_probabilities=p3[selection_mask],
        exit2_predictions=pred2[selection_mask],
        exit3_predictions=pred3[selection_mask],
        metadata_df=selection_df,
        labels=labels,
        lats_config_json=lats_config,
        parent_id_col=args.parent_id_col,
        lats_module=lats_module,
        reference_macro_f1=reference_macro_f1,
        max_macro_f1_drop=args.max_macro_f1_drop,
        min_exit2_fraction=args.min_exit2_fraction,
        exit2_flops=exit2_flops,
        exit3_flops=exit3_flops,
    )

    rows: list[dict[str, Any]] = []
    confidence_grid = parse_float_list(args.confidence_grid)
    margin_grid = parse_float_list(args.margin_grid)
    delta_grid = parse_float_list(args.delta_grid)
    risk_grid = parse_float_list(args.risk_grid)

    for confidence in confidence_grid:
        for margin in margin_grid:
            config = GlobalRuleConfig(
                confidence,
                margin,
                1.0,
                True,
                False,
            )
            rows.append(
                candidate_result(
                    strategy="global_conf_margin",
                    parameters=config.to_dict(),
                    stop_mask=global_rule_stop_mask(
                        selection_diagnostics,
                        config,
                    ),
                    **common_candidate_kwargs,
                )
            )

            for delta in delta_grid:
                config_delta = GlobalRuleConfig(
                    confidence,
                    margin,
                    delta,
                    True,
                    False,
                )
                rows.append(
                    candidate_result(
                        strategy="global_conf_margin_delta",
                        parameters=config_delta.to_dict(),
                        stop_mask=global_rule_stop_mask(
                            selection_diagnostics,
                            config_delta,
                        ),
                        **common_candidate_kwargs,
                    )
                )

                for risk_threshold in risk_grid:
                    risk_config = LabelRiskRuleConfig(
                        confidence,
                        margin,
                        delta,
                        True,
                        False,
                        risk_threshold,
                    )
                    rows.append(
                        candidate_result(
                            strategy="label_risk",
                            parameters=risk_config.to_dict(),
                            stop_mask=label_risk_stop_mask(
                                selection_diagnostics,
                                risk_config,
                            ),
                            **common_candidate_kwargs,
                        )
                    )

    margin_profiles: dict[str, dict[str, Any]] = {}
    for capture_fraction in parse_float_list(args.capture_grid):
        per_label_margins, corrected_counts = (
            derive_per_label_margin_thresholds(
                y_true=y_true[derivation_mask],
                exit2_probabilities=p2[derivation_mask],
                exit3_probabilities=p3[derivation_mask],
                exit2_thresholds=thresholds[1],
                exit3_thresholds=thresholds[2],
                capture_fraction=capture_fraction,
                minimum_corrected_examples=(
                    args.minimum_corrected_examples
                ),
            )
        )
        profile_key = f"capture_{capture_fraction:.4f}"
        margin_profiles[profile_key] = {
            "capture_fraction": capture_fraction,
            "per_label_margins": {
                label: float(per_label_margins[idx])
                for idx, label in enumerate(labels)
            },
            "corrected_example_counts": {
                label: int(corrected_counts[idx])
                for idx, label in enumerate(labels)
            },
        }
        for confidence in confidence_grid:
            config = PerLabelMarginConfig(
                confidence,
                tuple(
                    float(value)
                    for value in per_label_margins
                ),
                True,
                False,
            )
            parameters = {
                **config.to_dict(),
                "capture_fraction": capture_fraction,
                "corrected_example_counts": (
                    corrected_counts.tolist()
                ),
            }
            rows.append(
                candidate_result(
                    strategy="per_label_margin",
                    parameters=parameters,
                    stop_mask=per_label_margin_stop_mask(
                        selection_diagnostics,
                        config,
                    ),
                    **common_candidate_kwargs,
                )
            )

    for gate_threshold in parse_float_list(
        args.gate_threshold_grid
    ):
        parameters = {
            "gate_probability_threshold": gate_threshold,
            "allow_empty_stop": False,
            "target_definition": (
                "safe_when_exit3_does_not_reduce_binary_error_count"
            ),
        }
        rows.append(
            candidate_result(
                strategy="logistic_gate",
                parameters=parameters,
                stop_mask=logistic_gate_stop_mask(
                    safe_probabilities=(
                        gate_safe_probability[selection_mask]
                    ),
                    threshold=gate_threshold,
                    diagnostics=selection_diagnostics,
                    allow_empty_stop=False,
                ),
                **common_candidate_kwargs,
            )
        )

    sweep_df = pd.DataFrame(rows)
    selected_policies: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        strategy_df = sweep_df[
            sweep_df["strategy"] == strategy
        ].copy()
        selected_row, selection_status = select_candidate(
            strategy_df
        )
        parameters = json.loads(
            str(selected_row["parameters_json"])
        )
        selected_policies[strategy] = {
            "selection_status": selection_status,
            "parameters": parameters,
            "selection_metrics": {
                key: jsonable(value)
                for key, value in selected_row.items()
                if key != "parameters_json"
            },
        }
        selected_rows.append(
            {
                **{
                    key: jsonable(value)
                    for key, value in selected_row.items()
                },
                "selection_status": selection_status,
            }
        )

    sweep_path = (
        out_dir / "v013_matched_policy_validation_sweep.csv"
    )
    sweep_df.sort_values(
        [
            "strategy",
            "quality_constraint_met",
            "estimated_flops_saved_pct",
            "selection_macro_f1",
        ],
        ascending=[True, False, False, False],
    ).to_csv(sweep_path, index=False)
    selected_path = (
        out_dir / "v013_selected_policy_comparison.csv"
    )
    pd.DataFrame(selected_rows).to_csv(
        selected_path,
        index=False,
    )

    policy = {
        "schema_version": 1,
        "experiment": "v0.13_EE_matched_policy_comparison",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "validation_manifest": str(manifest),
        "validation_features_root": str(features_root),
        "labels_json": str(labels_json),
        "lats_config_json": (
            str(lats_config) if lats_config else None
        ),
        "labels": labels,
        "architecture": {
            "model": "ExitNet/TinyAudioCNN",
            "tap_blocks": list(tap_blocks),
            "num_exits": 3,
            "n_mels": n_mels,
            "frames_observed": frames,
            "eligible_early_exit": 2,
            "final_exit": 3,
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {
            f"exit{idx + 1}": threshold_mapping(
                labels,
                threshold,
            )
            for idx, threshold in enumerate(thresholds)
        },
        "validation_protocol": {
            "parent_id_col": args.parent_id_col,
            "derivation_fraction": args.derivation_fraction,
            "split_seed": args.split_seed,
            "derivation_segments": int(
                derivation_mask.sum()
            ),
            "selection_segments": int(selection_mask.sum()),
            "derivation_parents": int(
                derivation_df[
                    args.parent_id_col
                ].nunique()
            ),
            "selection_parents": int(
                selection_df[
                    args.parent_id_col
                ].nunique()
            ),
            "derivation_purpose": (
                "derive label risks, per-label margins, and train "
                "logistic gate"
            ),
            "selection_purpose": (
                "select every strategy under identical constraints"
            ),
        },
        "selection_constraints": {
            "max_absolute_macro_f1_drop": (
                args.max_macro_f1_drop
            ),
            "minimum_exit2_fraction": (
                args.min_exit2_fraction
            ),
            "reference_level": (
                "parent_frozen_lats_v2"
                if reference_parent
                else "segment"
            ),
        },
        "reference_always_exit3_selection_subset": {
            "segment_metrics": reference_segment,
            "parent_metrics": reference_parent,
        },
        "label_risk_profile": risk_profile.to_dict(),
        "risk_definition": {
            "margin_scale": args.risk_margin_scale,
            "margin_weight": args.risk_margin_weight,
            "delta_weight": args.risk_delta_weight,
            "minimum_label_improvement": (
                args.minimum_label_improvement
            ),
        },
        "per_label_margin_profiles": margin_profiles,
        "logistic_gate": {
            "model_filename": gate_model_path.name,
            "feature_names": gate_feature_names,
            "target_definition": (
                "safe=1 when Exit3 does not reduce sample binary-error "
                "count relative to Exit2"
            ),
            "derivation_safe_fraction": float(
                gate_targets[derivation_mask].mean()
            ),
            "selection_safe_fraction": float(
                gate_targets[selection_mask].mean()
            ),
            "derivation_exit3_improvement_mean": float(
                exit3_error_improvement[
                    derivation_mask
                ].mean()
            ),
            "model": (
                "StandardScaler + class-balanced "
                "LogisticRegression(C=1.0)"
            ),
        },
        "selected_policies": selected_policies,
        "estimated_flops_by_exit": {
            key: float(value)
            for key, value in flops.items()
        },
        "important_note": (
            "The corrected holdout must not be used to alter any "
            "selected strategy or learned gate parameter."
        ),
    }
    policy_path = (
        out_dir / "frozen_matched_policy_comparison_v013.json"
    )
    save_json(policy, policy_path)

    risk_df = pd.DataFrame(
        {
            "label": labels,
            "exit2_f1": risk_profile.exit2_f1,
            "exit3_f1": risk_profile.exit3_f1,
            "exit3_minus_exit2_f1": (
                risk_profile.improvement
            ),
            "risk_weight": risk_profile.risk_weights,
        }
    ).sort_values("risk_weight", ascending=False)
    risk_df.to_csv(
        out_dir / "v013_derivation_label_risk_profile.csv",
        index=False,
    )

    print("\nV0.13 matched policy tuning complete")
    print("-" * 110)
    print(f"Derivation segments: {int(derivation_mask.sum())}")
    print(f"Selection segments:  {int(selection_mask.sum())}")
    print(f"Frozen comparison:   {policy_path}")
    print(f"Logistic gate:       {gate_model_path}")
    print(f"Validation sweep:    {sweep_path}")
    display = pd.DataFrame(selected_rows)[
        [
            "strategy",
            "selection_status",
            "exit2_fraction",
            "selection_macro_f1",
            "macro_f1_drop",
            "estimated_flops_saved_pct",
            "quality_constraint_met",
        ]
    ]
    print("\nSelected matched policies:")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
