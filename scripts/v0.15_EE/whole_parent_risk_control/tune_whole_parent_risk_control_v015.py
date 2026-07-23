#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tune v0.15 whole-parent selective risk-controlled Early Exit."""

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

from common_v015 import (
    collect_outputs,
    fit_empirical_risk_calibrators,
    fit_shared_parent_gate,
    jsonable,
    load_checkpoint,
    load_json,
    load_labels,
    load_run_config,
    load_thresholds_by_exit,
    multilabel_metrics,
    parent_oof_probabilities,
    parse_float_list,
    parse_optional_float_list,
    parse_tap_blocks,
    resolve_model_cfg,
    robust_drop_statistics,
    save_json,
    select_risk_controlled_candidate,
    threshold_mapping,
    whole_parent_candidate_result,
)
from data.datasets_multilabel import make_multilabel_loaders
from policies.parent_aware_adaptive_gate import (
    label_predictions,
    parse_lats_rules,
)
from policies.whole_parent_selective_exit import (
    build_whole_parent_features,
    derive_label_unsafe_thresholds,
    whole_parent_stop_mask,
    whole_parent_unsafe_targets,
)
from utils.model_factory import build_audio_exit_net
from utils.profiling import estimate_flops_tiny_audiocnn


STRATEGIES = (
    "nonparametric_parent_risk",
    "shared_logistic_parent_gate",
)


def _fold_candidate_drops(
    *,
    fold_index: np.ndarray,
    stop_parent_mask: np.ndarray,
    target_bundle: dict[str, np.ndarray],
    row_to_parent: np.ndarray,
    y_true_segments: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    pred2: np.ndarray,
    pred3: np.ndarray,
    exit2_flops: float,
    exit3_flops: float,
    max_macro_f1_drop: float,
    max_micro_f1_drop: float,
    max_exact_match_drop: float,
    max_overall_harm_fraction: float,
) -> dict[str, Any]:
    macro_drops: list[float] = []
    micro_drops: list[float] = []
    exact_drops: list[float] = []
    fold_harm_fractions: list[float] = []
    fold_records: list[dict[str, Any]] = []

    for fold_no in sorted(np.unique(fold_index).tolist()):
        parent_mask = fold_index == fold_no
        parent_indices = np.flatnonzero(parent_mask)
        mapping = np.full(len(parent_mask), -1, dtype=np.int64)
        mapping[parent_indices] = np.arange(len(parent_indices), dtype=np.int64)
        row_mask = parent_mask[row_to_parent]
        local_row_to_parent = mapping[row_to_parent[row_mask]]

        deeper_reference = multilabel_metrics(
            target_bundle["parent_truth"][parent_mask],
            target_bundle["deeper_predictions"][parent_mask],
        )
        fold_row = whole_parent_candidate_result(
            strategy="fold_diagnostic",
            parameters={},
            stop_parent_mask=stop_parent_mask[parent_mask],
            parent_truth=target_bundle["parent_truth"][parent_mask],
            source_parent_predictions=target_bundle["source_predictions"][parent_mask],
            deeper_parent_predictions=target_bundle["deeper_predictions"][parent_mask],
            unsafe_targets=target_bundle["unsafe_targets"][parent_mask],
            row_to_parent=local_row_to_parent,
            y_true_segments=y_true_segments[row_mask],
            source_segment_probabilities=p2[row_mask],
            deeper_segment_probabilities=p3[row_mask],
            source_segment_predictions=pred2[row_mask],
            deeper_segment_predictions=pred3[row_mask],
            source_flops=exit2_flops,
            deeper_flops=exit3_flops,
            reference_parent_metrics=deeper_reference,
            max_macro_f1_drop=max_macro_f1_drop,
            max_micro_f1_drop=max_micro_f1_drop,
            max_exact_match_drop=max_exact_match_drop,
            max_overall_harm_fraction=max_overall_harm_fraction,
            min_parent_stop_fraction=0.0,
        )
        macro_drops.append(float(fold_row["macro_f1_drop"]))
        micro_drops.append(float(fold_row["micro_f1_drop"]))
        exact_drops.append(float(fold_row["exact_match_drop"]))
        fold_harm_fractions.append(float(fold_row["overall_harm_fraction"]))
        fold_records.append(
            {
                "fold": int(fold_no),
                "parents": int(parent_mask.sum()),
                "parent_stop_fraction": float(fold_row["parent_stop_fraction"]),
                "macro_f1_drop": float(fold_row["macro_f1_drop"]),
                "micro_f1_drop": float(fold_row["micro_f1_drop"]),
                "exact_match_drop": float(fold_row["exact_match_drop"]),
                "overall_harm_fraction": float(fold_row["overall_harm_fraction"]),
            }
        )

    macro_stats = robust_drop_statistics(macro_drops)
    micro_stats = robust_drop_statistics(micro_drops)
    exact_stats = robust_drop_statistics(exact_drops)
    return {
        **macro_stats,
        "fold_micro_f1_drop_mean": micro_stats["fold_macro_f1_drop_mean"],
        "fold_micro_f1_drop_upper_confidence": micro_stats[
            "fold_macro_f1_drop_upper_confidence"
        ],
        "fold_micro_f1_drop_max": micro_stats["fold_macro_f1_drop_max"],
        "fold_exact_match_drop_mean": exact_stats["fold_macro_f1_drop_mean"],
        "fold_exact_match_drop_upper_confidence": exact_stats[
            "fold_macro_f1_drop_upper_confidence"
        ],
        "fold_exact_match_drop_max": exact_stats["fold_macro_f1_drop_max"],
        "fold_overall_harm_fraction_mean": float(np.mean(fold_harm_fractions)),
        "fold_overall_harm_fraction_max": float(np.max(fold_harm_fractions)),
        "fold_records": fold_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune whole-parent Exit-2/Exit-3 selective risk control."
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
    parser.add_argument("--empirical_bins", type=int, default=5)
    parser.add_argument("--minimum_positive_examples", type=int, default=3)
    parser.add_argument(
        "--target_recall_grid",
        default="0.80,0.90,0.95,0.98,1.00",
    )
    parser.add_argument(
        "--expected_harm_grid",
        default="none,0.005,0.01,0.02,0.05,0.10",
    )
    parser.add_argument("--max_macro_f1_drop", type=float, default=0.005)
    parser.add_argument("--max_micro_f1_drop", type=float, default=0.005)
    parser.add_argument("--max_exact_match_drop", type=float, default=0.01)
    parser.add_argument("--max_overall_harm_fraction", type=float, default=0.01)
    parser.add_argument("--min_parent_stop_fraction", type=float, default=0.02)
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
    required = [
        manifest,
        features_root,
        labels_json,
        checkpoint,
        args.lats_config_json.resolve(),
    ]
    for path in required:
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
        raise RuntimeError(f"Expected three exits, got {len(probabilities)}.")
    p1, p2, p3 = probabilities

    thresholds = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=3,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
    )
    pred2 = label_predictions(p2, thresholds[1])
    pred3 = label_predictions(p3, thresholds[2])
    rules = parse_lats_rules(load_json(args.lats_config_json.resolve()), labels)

    parent_features, feature_names, diagnostics = build_whole_parent_features(
        current_probabilities=p2,
        previous_probabilities=p1,
        parent_ids=parent_ids,
        rules=rules,
    )
    target_bundle = whole_parent_unsafe_targets(
        y_true=y_true,
        source_probabilities=p2,
        deeper_probabilities=p3,
        parent_ids=parent_ids,
        rules=rules,
    )
    if not np.array_equal(diagnostics["parent_ids"], target_bundle["parent_ids"]):
        raise RuntimeError("Parent features and target ordering differ.")
    row_to_parent = target_bundle["row_to_parent"]

    oof_probabilities, fold_index, fold_records = parent_oof_probabilities(
        parent_features=parent_features,
        raw_nonparametric_risk=diagnostics["raw_nonparametric_risk"],
        unsafe_targets=target_bundle["unsafe_targets"],
        n_splits=args.cv_folds,
        seed=args.seed,
        empirical_bins=args.empirical_bins,
        minimum_positive_examples=args.minimum_positive_examples,
    )

    shared_model = fit_shared_parent_gate(
        parent_features=parent_features,
        unsafe_targets=target_bundle["unsafe_targets"],
        seed=args.seed,
    )
    shared_path = out_dir / "shared_logistic_parent_gate_v015.joblib"
    joblib.dump(
        {
            "type": "shared_logistic_parent_gate",
            "model": shared_model,
            "feature_names": feature_names,
            "labels": labels,
        },
        shared_path,
    )
    calibrators, empirical_counts, empirical_pooled = fit_empirical_risk_calibrators(
        raw_scores=diagnostics["raw_nonparametric_risk"],
        unsafe_targets=target_bundle["unsafe_targets"],
        num_bins=args.empirical_bins,
        minimum_positive_examples=args.minimum_positive_examples,
    )
    empirical_path = out_dir / "nonparametric_parent_risk_v015.joblib"
    joblib.dump(
        {
            "type": "nonparametric_parent_risk",
            "calibrators": calibrators,
            "labels": labels,
            "positive_counts": empirical_counts,
            "used_pooled": empirical_pooled,
        },
        empirical_path,
    )

    reference_parent_metrics = multilabel_metrics(
        target_bundle["parent_truth"],
        target_bundle["deeper_predictions"],
    )
    reference_segment_metrics = multilabel_metrics(y_true, pred3)
    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )
    exit2_flops = float(flops["exit2"])
    exit3_flops = float(flops["exit3"])

    rows: list[dict[str, Any]] = []
    target_recalls = parse_float_list(args.target_recall_grid)
    expected_harm_values = parse_optional_float_list(args.expected_harm_grid)
    for strategy in STRATEGIES:
        unsafe_probabilities = oof_probabilities[strategy]
        for target_recall in target_recalls:
            label_thresholds, unsafe_counts, used_fallback = (
                derive_label_unsafe_thresholds(
                    unsafe_targets=target_bundle["unsafe_targets"],
                    unsafe_probabilities=unsafe_probabilities,
                    target_recall=target_recall,
                    minimum_positive_examples=args.minimum_positive_examples,
                )
            )
            for expected_harm_threshold in expected_harm_values:
                stop_mask, expected_harm, highest_risk = whole_parent_stop_mask(
                    unsafe_probabilities=unsafe_probabilities,
                    label_thresholds=label_thresholds,
                    expected_harm_threshold=expected_harm_threshold,
                    non_empty=diagnostics["non_empty"],
                    allow_empty_stop=False,
                )
                parameters = {
                    "target_unsafe_recall": float(target_recall),
                    "label_unsafe_probability_thresholds": {
                        label: float(label_thresholds[idx])
                        for idx, label in enumerate(labels)
                    },
                    "unsafe_positive_counts": {
                        label: int(unsafe_counts[idx])
                        for idx, label in enumerate(labels)
                    },
                    "used_fallback_threshold": {
                        label: bool(used_fallback[idx])
                        for idx, label in enumerate(labels)
                    },
                    "expected_harm_threshold": expected_harm_threshold,
                    "allow_empty_stop": False,
                }
                row = whole_parent_candidate_result(
                    strategy=strategy,
                    parameters=parameters,
                    stop_parent_mask=stop_mask,
                    parent_truth=target_bundle["parent_truth"],
                    source_parent_predictions=target_bundle["source_predictions"],
                    deeper_parent_predictions=target_bundle["deeper_predictions"],
                    unsafe_targets=target_bundle["unsafe_targets"],
                    row_to_parent=row_to_parent,
                    y_true_segments=y_true,
                    source_segment_probabilities=p2,
                    deeper_segment_probabilities=p3,
                    source_segment_predictions=pred2,
                    deeper_segment_predictions=pred3,
                    source_flops=exit2_flops,
                    deeper_flops=exit3_flops,
                    reference_parent_metrics=reference_parent_metrics,
                    max_macro_f1_drop=args.max_macro_f1_drop,
                    max_micro_f1_drop=args.max_micro_f1_drop,
                    max_exact_match_drop=args.max_exact_match_drop,
                    max_overall_harm_fraction=args.max_overall_harm_fraction,
                    min_parent_stop_fraction=args.min_parent_stop_fraction,
                )
                fold_stats = _fold_candidate_drops(
                    fold_index=fold_index,
                    stop_parent_mask=stop_mask,
                    target_bundle=target_bundle,
                    row_to_parent=row_to_parent,
                    y_true_segments=y_true,
                    p2=p2,
                    p3=p3,
                    pred2=pred2,
                    pred3=pred3,
                    exit2_flops=exit2_flops,
                    exit3_flops=exit3_flops,
                    max_macro_f1_drop=args.max_macro_f1_drop,
                    max_micro_f1_drop=args.max_micro_f1_drop,
                    max_exact_match_drop=args.max_exact_match_drop,
                    max_overall_harm_fraction=args.max_overall_harm_fraction,
                )
                row.update(
                    {
                        key: value
                        for key, value in fold_stats.items()
                        if key != "fold_records"
                    }
                )
                row["fold_records_json"] = json.dumps(
                    jsonable(fold_stats["fold_records"]), sort_keys=True
                )
                row["mean_expected_harm"] = float(expected_harm.mean())
                row["highest_risk_label_mode"] = int(
                    np.bincount(highest_risk, minlength=len(labels)).argmax()
                )
                row["robust_risk_constraint_met"] = bool(
                    row["base_risk_constraint_met"]
                    and row["fold_macro_f1_drop_upper_confidence"]
                    <= args.max_macro_f1_drop + 1e-12
                    and row["fold_micro_f1_drop_upper_confidence"]
                    <= args.max_micro_f1_drop + 1e-12
                    and row["overall_harm_fraction_upper_confidence"]
                    <= args.max_overall_harm_fraction + 1e-12
                )
                rows.append(row)

    sweep_df = pd.DataFrame(rows)
    selected_policies: dict[str, dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    bundle_names = {
        "nonparametric_parent_risk": empirical_path.name,
        "shared_logistic_parent_gate": shared_path.name,
    }
    for strategy in STRATEGIES:
        selected_row, selection_status = select_risk_controlled_candidate(
            sweep_df[sweep_df["strategy"] == strategy].copy()
        )
        parameters = json.loads(str(selected_row["parameters_json"]))
        selected_policies[strategy] = {
            "selection_status": selection_status,
            "deployment_eligible": bool(
                selected_row["robust_risk_constraint_met"]
            ),
            "bundle_filename": bundle_names[strategy],
            "parameters": parameters,
            "selection_metrics": {
                key: jsonable(value)
                for key, value in selected_row.items()
                if key not in {"parameters_json", "fold_records_json"}
            },
        }
        selected_rows.append(
            {
                **{
                    key: jsonable(value)
                    for key, value in selected_row.items()
                    if key not in {"parameters_json", "fold_records_json"}
                },
                "selection_status": selection_status,
            }
        )

    sweep_path = out_dir / "v015_whole_parent_validation_sweep.csv"
    sweep_df.sort_values(
        [
            "strategy",
            "robust_risk_constraint_met",
            "estimated_flops_saved_pct",
            "parent_macro_f1",
        ],
        ascending=[True, False, False, False],
    ).to_csv(sweep_path, index=False)
    selected_path = out_dir / "v015_selected_parent_policies.csv"
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)

    parent_diagnostics = pd.DataFrame(
        {
            args.parent_id_col: target_bundle["parent_ids"],
            "fold": fold_index,
            "any_unsafe_exit2_parent": target_bundle["any_unsafe"],
            "exit2_parent_error_count": target_bundle["source_error_count"],
            "exit3_parent_error_count": target_bundle["deeper_error_count"],
            "net_error_increase_exit2": target_bundle["net_error_increase"],
        }
    )
    for label_idx, label in enumerate(labels):
        parent_diagnostics[f"unsafe_{label}"] = target_bundle[
            "unsafe_targets"
        ][:, label_idx]
        parent_diagnostics[f"oof_nonparametric_{label}"] = oof_probabilities[
            "nonparametric_parent_risk"
        ][:, label_idx]
        parent_diagnostics[f"oof_shared_logistic_{label}"] = oof_probabilities[
            "shared_logistic_parent_gate"
        ][:, label_idx]
    diagnostics_path = out_dir / "v015_parent_oof_diagnostics.csv"
    parent_diagnostics.to_csv(diagnostics_path, index=False)

    frozen = {
        "schema_version": 1,
        "experiment": "v0.15_EE_whole_parent_selective_risk_control",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "validation_manifest": str(manifest),
        "validation_features_root": str(features_root),
        "labels_json": str(labels_json),
        "lats_config_json": str(args.lats_config_json.resolve()),
        "labels": labels,
        "architecture": {
            "model": "ExitNet/TinyAudioCNN",
            "tap_blocks": list(tap_blocks),
            "num_exits": 3,
            "decision_unit": "complete_parent",
            "source_exit": 2,
            "deeper_exit": 3,
            "n_mels": n_mels,
            "frames_observed": frames,
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {
            f"exit{idx + 1}": threshold_mapping(labels, threshold)
            for idx, threshold in enumerate(thresholds)
        },
        "validation_protocol": {
            "parents": int(len(target_bundle["parent_ids"])),
            "segments": int(len(y_true)),
            "cv_folds": int(min(args.cv_folds, len(parent_features))),
            "seed": int(args.seed),
            "fold_records": fold_records,
            "oof_selection": True,
        },
        "selection_constraints": {
            "max_parent_macro_f1_drop": float(args.max_macro_f1_drop),
            "max_parent_micro_f1_drop": float(args.max_micro_f1_drop),
            "max_parent_exact_match_drop": float(args.max_exact_match_drop),
            "max_overall_harm_fraction": float(args.max_overall_harm_fraction),
            "minimum_parent_stop_fraction": float(args.min_parent_stop_fraction),
            "upper_confidence_required": True,
        },
        "gate_design": {
            "decision": (
                "All segments in a parent stop at Exit 2 or all continue to Exit 3."
            ),
            "target": (
                "Per-label harm when the complete all-Exit2 parent prediction is "
                "wrong and the all-Exit3 parent prediction is correct."
            ),
            "strategies": {
                "nonparametric_parent_risk": (
                    "Empirical monotone bin calibration of transparent parent risk."
                ),
                "shared_logistic_parent_gate": (
                    "One shared class-balanced logistic model across parent-label pairs."
                ),
            },
            "feature_names": feature_names,
        },
        "reference_always_exit3_validation": {
            "segment_metrics": reference_segment_metrics,
            "parent_metrics": reference_parent_metrics,
        },
        "unsafe_label_counts": {
            label: int(target_bundle["unsafe_targets"][:, idx].sum())
            for idx, label in enumerate(labels)
        },
        "selected_policies": selected_policies,
        "estimated_flops_by_exit": {
            key: float(value) for key, value in flops.items()
        },
        "important_note": (
            "Models and thresholds were selected using validation-only out-of-fold "
            "parent predictions. The corrected holdout must not alter them."
        ),
    }
    frozen_path = out_dir / "frozen_whole_parent_policy_v015.json"
    save_json(frozen, frozen_path)

    print("\nV0.15 whole-parent risk-control tuning complete")
    print("-" * 126)
    print(f"Validation parents: {len(target_bundle['parent_ids'])}")
    print(f"Validation segments: {len(y_true)}")
    print(f"Frozen policy:       {frozen_path}")
    print(f"Validation sweep:    {sweep_path}")
    display_columns = [
        "strategy",
        "selection_status",
        "robust_risk_constraint_met",
        "parent_stop_fraction",
        "segment_stop_fraction",
        "estimated_flops_saved_pct",
        "parent_macro_f1",
        "parent_micro_f1",
        "parent_exact_match",
        "macro_f1_drop",
        "micro_f1_drop",
        "overall_harm_fraction",
        "overall_harm_fraction_upper_confidence",
    ]
    print("\nSelected whole-parent policies:")
    print(pd.DataFrame(selected_rows)[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
