#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tune a Pareto-optimal lightweight per-label margin Early-Exit policy."""

from __future__ import annotations

import argparse
import json
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

from common_v016 import (  # noqa: E402
    ParentMetricContext,
    collect_outputs,
    evaluate_margin_candidate,
    jsonable,
    load_checkpoint,
    load_json,
    load_labels,
    load_run_config,
    load_thresholds_by_exit,
    objective_matrix,
    parse_tap_blocks,
    resolve_model_cfg,
    save_json,
    threshold_mapping,
)
from data.datasets_multilabel import make_multilabel_loaders  # noqa: E402
from policies.early_exit_strategy_comparison import (  # noqa: E402
    compute_common_diagnostics,
    derive_per_label_margin_thresholds,
    label_predictions,
)
from policies.multiobjective_per_label_margin import (  # noqa: E402
    decode_genes,
    environmental_select,
    make_bounds,
    make_offspring,
    multiobjective_margin_stop_mask,
    pareto_front_mask,
    random_population,
)
from utils.model_factory import build_audio_exit_net  # noqa: E402
from utils.profiling import estimate_flops_tiny_audiocnn  # noqa: E402


def parse_pair(text: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"{name} must contain two increasing values.")
    return values[0], values[1]


def objectives(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            row["objective_compute"],
            row["objective_macro"],
            row["objective_micro"],
            row["objective_exact"],
            row["objective_hamming"],
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--features_root", type=Path, default=None)
    parser.add_argument("--labels_json", required=True, type=Path)
    parser.add_argument("--lats_config_json", required=True, type=Path)
    parser.add_argument("--v013_policy_json", type=Path, default=None)
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument(
        "--threshold_mode",
        choices=["tuned_per_exit", "final_exit_tuned", "fixed_0p5"],
        default="fixed_0p5",
    )
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--population_size", type=int, default=80)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crossover_probability", type=float, default=0.90)
    parser.add_argument("--mutation_probability", type=float, default=0.20)
    parser.add_argument("--mutation_scale", type=float, default=0.08)
    parser.add_argument("--confidence_bounds", default="0.50,0.99")
    parser.add_argument("--delta_bounds", default="0.01,1.00")
    parser.add_argument("--margin_bounds", default="0.00,0.50")
    parser.add_argument("--max_macro_f1_drop", type=float, default=0.01)
    parser.add_argument("--max_micro_f1_drop", type=float, default=0.005)
    parser.add_argument("--max_exact_match_drop", type=float, default=0.01)
    parser.add_argument("--max_hamming_increase", type=float, default=0.002)
    parser.add_argument("--min_exit2_fraction", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    if args.population_size < 8 or args.generations < 1:
        raise ValueError("Use population_size >= 8 and generations >= 1.")

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
    checkpoint = args.checkpoint.resolve() if args.checkpoint else run_dir / "ckpt" / "best.pt"
    for path in (
        manifest,
        features_root,
        checkpoint,
        args.labels_json.resolve(),
        args.lats_config_json.resolve(),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    labels = load_labels(args.labels_json.resolve(), cfg)
    tap_blocks = parse_tap_blocks(cfg.get("tap_blocks", "1,3"))
    n_mels = int(cfg.get("n_mels", 64))
    batch_size = int(args.batch_size or cfg.get("batch_size", 64))
    train_loader, val_loader, test_loader, loaded_labels = make_multilabel_loaders(
        manifest_csv=manifest,
        features_root=features_root,
        labels_json=args.labels_json.resolve(),
        batch_size=batch_size,
        num_workers=int(args.num_workers),
        seed=int(cfg.get("seed", args.seed)),
        label_balance_power=0.0,
        synthetic_balance_power=0.0,
    )
    del train_loader, test_loader
    if list(loaded_labels) != labels:
        raise RuntimeError("Label order mismatch between schema and loader.")
    metadata_df = val_loader.dataset.df.reset_index(drop=True)
    if args.parent_id_col not in metadata_df.columns:
        raise RuntimeError(f"Validation manifest lacks {args.parent_id_col!r}.")

    model = build_audio_exit_net(
        num_classes=len(labels),
        n_mels=n_mels,
        tap_blocks=tap_blocks,
        model_cfg=resolve_model_cfg(cfg),
    ).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()
    y_true, exit_probabilities, frames = collect_outputs(model, val_loader, args.device)
    if len(exit_probabilities) != 3:
        raise RuntimeError(f"Expected three exits, got {len(exit_probabilities)}.")
    p1, p2, p3 = exit_probabilities
    thresholds = load_thresholds_by_exit(
        run_dir=run_dir,
        labels=labels,
        num_exits=3,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.fixed_threshold,
    )
    pred2 = label_predictions(p2, thresholds[1])
    pred3 = label_predictions(p3, thresholds[2])
    diagnostics = compute_common_diagnostics(
        exit1_probabilities=p1,
        exit2_probabilities=p2,
        exit1_thresholds=thresholds[0],
        exit2_thresholds=thresholds[1],
    )
    parent_context = ParentMetricContext.build(
        metadata_df=metadata_df,
        labels=labels,
        parent_id_col=args.parent_id_col,
        lats_config_json=args.lats_config_json.resolve(),
        reference_probabilities=p3,
        cv_folds=args.cv_folds,
    )
    flops = estimate_flops_tiny_audiocnn(
        n_mels=n_mels,
        frames=frames,
        num_classes=len(labels),
        tap_blocks=tap_blocks,
    )

    lower, upper = make_bounds(
        len(labels),
        confidence_bounds=parse_pair(args.confidence_bounds, "confidence_bounds"),
        delta_bounds=parse_pair(args.delta_bounds, "delta_bounds"),
        margin_bounds=parse_pair(args.margin_bounds, "margin_bounds"),
    )
    seeds: list[np.ndarray] = [
        np.asarray([0.50, 1.00, *([0.0] * len(labels))], dtype=np.float64)
    ]
    for capture in (0.25, 0.50, 0.75, 0.90):
        margins, _ = derive_per_label_margin_thresholds(
            y_true=y_true,
            exit2_probabilities=p2,
            exit3_probabilities=p3,
            exit2_thresholds=thresholds[1],
            exit3_thresholds=thresholds[2],
            capture_fraction=capture,
            minimum_corrected_examples=3,
        )
        for confidence in (0.55, 0.65, 0.75):
            seeds.append(np.asarray([confidence, 1.0, *margins], dtype=np.float64))

    v013_seed_used = False
    if args.v013_policy_json and args.v013_policy_json.resolve().exists():
        previous = load_json(args.v013_policy_json.resolve())
        item = previous.get("selected_policies", {}).get("per_label_margin")
        if item:
            parameters = item["parameters"]
            seeds.append(
                np.asarray(
                    [
                        float(parameters["mean_confidence_threshold"]),
                        1.0,
                        *[float(value) for value in parameters["per_label_margins"]],
                    ],
                    dtype=np.float64,
                )
            )
            v013_seed_used = True

    rng = np.random.default_rng(int(args.seed))
    population = random_population(
        size=int(args.population_size), lower=lower, upper=upper, rng=rng, seeds=seeds
    )
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    order: list[tuple[float, ...]] = []

    def evaluate(genes: np.ndarray) -> dict[str, Any]:
        key = tuple(float(value) for value in np.round(genes, 6))
        if key in cache:
            return cache[key]
        config = decode_genes(genes, num_labels=len(labels))
        row = evaluate_margin_candidate(
            strategy="multiobjective_per_label_margin",
            parameters=config.to_dict(),
            stop_mask=multiobjective_margin_stop_mask(diagnostics, config),
            y_true=y_true,
            exit2_probabilities=p2,
            exit3_probabilities=p3,
            exit2_predictions=pred2,
            exit3_predictions=pred3,
            parent_context=parent_context,
            exit2_flops=float(flops["exit2"]),
            exit3_flops=float(flops["exit3"]),
            max_macro_drop=args.max_macro_f1_drop,
            max_micro_drop=args.max_micro_f1_drop,
            max_exact_drop=args.max_exact_match_drop,
            max_hamming_increase=args.max_hamming_increase,
            min_exit2_fraction=args.min_exit2_fraction,
        )
        row["candidate_id"] = len(cache)
        row["genes_json"] = json.dumps([float(value) for value in genes])
        row["mean_confidence_threshold"] = config.mean_confidence_threshold
        row["max_probability_delta"] = config.max_probability_delta
        for idx, label in enumerate(labels):
            row[f"margin_{label}"] = config.per_label_margins[idx]
        cache[key] = row
        order.append(key)
        return row

    history: list[dict[str, Any]] = []
    for generation in range(int(args.generations)):
        rows = [evaluate(genes) for genes in population]
        matrix = np.vstack([objectives(row) for row in rows])
        violations = np.asarray([row["constraint_violation"] for row in rows])
        feasible = [row for row in rows if row["quality_constraints_met"]]
        history.append(
            {
                "generation": generation,
                "unique_candidates": len(cache),
                "feasible_population": len(feasible),
                "best_feasible_flops_saved_pct": max(
                    [row["estimated_flops_saved_pct"] for row in feasible], default=0.0
                ),
                "minimum_constraint_violation": float(np.min(violations)),
            }
        )
        offspring = make_offspring(
            population=population,
            objectives=matrix,
            violations=violations,
            lower=lower,
            upper=upper,
            rng=rng,
            crossover_probability=args.crossover_probability,
            mutation_probability=args.mutation_probability,
            mutation_scale=args.mutation_scale,
        )
        offspring_rows = [evaluate(genes) for genes in offspring]
        population, _, _ = environmental_select(
            population=np.vstack([population, offspring]),
            objectives=np.vstack(
                [matrix, np.vstack([objectives(row) for row in offspring_rows])]
            ),
            violations=np.concatenate(
                [
                    violations,
                    np.asarray([row["constraint_violation"] for row in offspring_rows]),
                ]
            ),
            size=int(args.population_size),
        )

    final_df = pd.DataFrame([evaluate(genes) for genes in population]).drop_duplicates(
        subset=["genes_json"]
    )
    pareto_mask = pareto_front_mask(
        objective_matrix(final_df),
        final_df["constraint_violation"].to_numpy(dtype=np.float64),
    )
    pareto_df = final_df.loc[pareto_mask].copy()
    all_df = pd.DataFrame([cache[key] for key in order]).drop_duplicates(
        subset=["genes_json"]
    )
    feasible_df = all_df[all_df["quality_constraints_met"] == True].copy()  # noqa: E712
    if feasible_df.empty:
        selected = all_df.sort_values(
            ["constraint_violation", "estimated_flops_saved_pct"],
            ascending=[True, False],
        ).iloc[0]
        status = "fallback_minimum_constraint_violation"
        eligible = False
    else:
        selected = feasible_df.sort_values(
            [
                "estimated_flops_saved_pct",
                "parent_macro_f1",
                "parent_micro_f1",
                "parent_exact_match",
                "parent_hamming_loss",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        status = "feasible_pareto_max_compute_saving"
        eligible = True

    all_df.sort_values(
        ["quality_constraints_met", "estimated_flops_saved_pct"],
        ascending=[False, False],
    ).to_csv(out_dir / "v016_all_evaluated_candidates.csv", index=False)
    pareto_df.sort_values("estimated_flops_saved_pct", ascending=False).to_csv(
        out_dir / "v016_pareto_front.csv", index=False
    )
    pd.DataFrame(history).to_csv(out_dir / "v016_optimization_history.csv", index=False)
    pd.DataFrame(
        [
            {
                **{key: jsonable(value) for key, value in selected.items()},
                "selection_status": status,
                "deployment_eligible": eligible,
            }
        ]
    ).to_csv(out_dir / "v016_selected_policy.csv", index=False)

    frozen = {
        "schema_version": 1,
        "experiment": "v0.16_EE_multiobjective_per_label_margin",
        "branch": "active_budget_anytime_exit_v0.4",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "validation_manifest": str(manifest),
        "validation_features_root": str(features_root),
        "labels_json": str(args.labels_json.resolve()),
        "lats_config_json": str(args.lats_config_json.resolve()),
        "labels": labels,
        "architecture": {
            "model": "ExitNet/TinyAudioCNN",
            "tap_blocks": list(tap_blocks),
            "num_exits": 3,
            "eligible_early_exit": 2,
            "final_exit": 3,
            "n_mels": n_mels,
            "frames_observed": frames,
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {
            f"exit{idx + 1}": threshold_mapping(labels, threshold)
            for idx, threshold in enumerate(thresholds)
        },
        "optimisation": {
            "algorithm": "constraint-aware NSGA-II-style evolutionary search",
            "population_size": int(args.population_size),
            "generations": int(args.generations),
            "seed": int(args.seed),
            "genes": [
                "mean_confidence_threshold",
                "max_probability_delta",
                *[f"margin_{label}" for label in labels],
            ],
            "objectives": [
                "maximise estimated FLOPs saved",
                "minimise robust Parent Macro-F1 drop",
                "minimise robust Parent Micro-F1 drop",
                "minimise robust Parent Exact-Match drop",
                "minimise robust Parent Hamming increase",
            ],
            "bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
            "v013_seed_used": v013_seed_used,
            "unique_candidates_evaluated": int(len(all_df)),
            "pareto_candidates": int(len(pareto_df)),
        },
        "validation_protocol": {
            "segments": int(len(y_true)),
            "parents": int(len(parent_context.parent_ids)),
            "cv_folds": int(len(np.unique(parent_context.fold_index))),
            "fold_confidence_bound": "one-sided normal approximation, z=1.645",
        },
        "selection_constraints": {
            "max_parent_macro_f1_drop": float(args.max_macro_f1_drop),
            "max_parent_micro_f1_drop": float(args.max_micro_f1_drop),
            "max_parent_exact_match_drop": float(args.max_exact_match_drop),
            "max_parent_hamming_increase": float(args.max_hamming_increase),
            "minimum_exit2_fraction": float(args.min_exit2_fraction),
            "upper_confidence_required": True,
        },
        "reference_always_exit3_validation": parent_context.reference_metrics,
        "selected_policy": {
            "selection_status": status,
            "deployment_eligible": eligible,
            "parameters": json.loads(str(selected["parameters_json"])),
            "validation_metrics": {
                key: jsonable(value)
                for key, value in selected.items()
                if key not in {"parameters_json", "genes_json"}
            },
        },
        "estimated_flops_by_exit": {key: float(value) for key, value in flops.items()},
        "important_note": (
            "The corrected holdout must not alter the selected thresholds or "
            "quality constraints. A fallback policy is diagnostic only."
        ),
    }
    frozen_path = out_dir / "frozen_multiobjective_margin_policy_v016.json"
    save_json(frozen, frozen_path)

    columns = [
        "estimated_flops_saved_pct",
        "exit2_fraction",
        "parent_macro_f1",
        "parent_micro_f1",
        "parent_exact_match",
        "parent_hamming_loss",
        "macro_f1_drop",
        "micro_f1_drop",
        "exact_match_drop",
        "hamming_loss_increase",
        "constraint_violation",
        "quality_constraints_met",
    ]
    print("\nV0.16 multi-objective per-label margin tuning complete")
    print("-" * 122)
    print(f"Validation segments:      {len(y_true)}")
    print(f"Validation parents:       {len(parent_context.parent_ids)}")
    print(f"Unique candidates tested: {len(all_df)}")
    print(f"Pareto candidates:        {len(pareto_df)}")
    print(f"Selection status:         {status}")
    print(f"Deployment eligible:      {eligible}")
    print(f"Frozen policy:            {frozen_path}")
    print("\nSelected policy metrics:")
    print(pd.DataFrame([selected])[columns].to_string(index=False))


if __name__ == "__main__":
    main()
