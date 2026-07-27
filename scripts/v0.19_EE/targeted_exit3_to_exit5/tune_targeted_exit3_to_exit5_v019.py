#!/usr/bin/env python
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
V017_DIR = PROJECT_ROOT / "scripts" / "v0.17_EE" / "sequential_anytime_exit"
for path in (PROJECT_ROOT, V017_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v017 import evaluate_sequential_candidate, jsonable, objective_matrix, save_json, threshold_mapping
from policies.multiobjective_per_label_margin import environmental_select, make_offspring, pareto_front_mask, random_population
from policies.targeted_exit3_to_exit5_v019 import decode_targeted_genes, derive_grouped_continuation_risk, make_targeted_bounds, targeted_select
from tune_v017_helpers import objective_vector, parse_pair
from tune_v017_problem import prepare_problem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune the final v0.19 Exit-3-to-Exit-5 policy.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--features_root", type=Path)
    parser.add_argument("--labels_json", required=True, type=Path)
    parser.add_argument("--lats_config_json", required=True, type=Path)
    parser.add_argument("--parent_id_col", default="parent_clip_id")
    parser.add_argument("--threshold_mode", choices=["tuned_per_exit", "final_exit_tuned", "fixed_0p5"], default="fixed_0p5")
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--population_size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crossover_probability", type=float, default=0.90)
    parser.add_argument("--mutation_probability", type=float, default=0.18)
    parser.add_argument("--mutation_scale", type=float, default=0.05)
    parser.add_argument("--confidence_bounds", default="0.65,0.995")
    parser.add_argument("--delta_bounds", default="0.00,0.50")
    parser.add_argument("--risk_bounds", default="0.00,1.00")
    parser.add_argument("--max_label_risk_bounds", default="0.00,0.50")
    parser.add_argument("--risk_score_bounds", default="0.25,0.95")
    parser.add_argument("--risk_multiplier_bounds", default="1.00,4.00")
    parser.add_argument("--risk_band_bounds", default="0.01,0.30")
    parser.add_argument("--margin_bounds", default="0.00,0.50")
    parser.add_argument("--internal_max_macro_drop", type=float, default=0.005)
    parser.add_argument("--internal_max_micro_drop", type=float, default=0.0025)
    parser.add_argument("--internal_max_exact_drop", type=float, default=0.005)
    parser.add_argument("--internal_max_hamming_increase", type=float, default=0.001)
    parser.add_argument("--deployment_max_macro_drop", type=float, default=0.01)
    parser.add_argument("--deployment_max_micro_drop", type=float, default=0.005)
    parser.add_argument("--deployment_max_exact_drop", type=float, default=0.01)
    parser.add_argument("--deployment_max_hamming_increase", type=float, default=0.002)
    parser.add_argument("--min_exit3_fraction", type=float, default=0.02)
    parser.add_argument("--target_flops_saved_pct", type=float, default=7.0)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", required=True, type=Path)
    return parser


def seed_vectors(num_labels: int) -> list[np.ndarray]:
    def vector(conf: float, delta: float, max_risk: float, risk_threshold: float, multiplier: float, band: float, margin: float) -> np.ndarray:
        return np.asarray([conf, delta, max_risk, risk_threshold, multiplier, band, *([margin] * num_labels)], dtype=np.float64)
    return [
        vector(0.985, 0.04, 0.05, 0.35, 3.50, 0.22, 0.40),
        vector(0.950, 0.10, 0.10, 0.45, 3.00, 0.16, 0.28),
        vector(0.900, 0.18, 0.18, 0.55, 2.50, 0.12, 0.20),
        vector(0.820, 0.30, 0.30, 0.70, 1.75, 0.08, 0.12),
        vector(0.650, 0.50, 0.50, 0.95, 1.00, 0.01, 0.00),
    ]


def quality_utilisation(row: pd.Series, args: argparse.Namespace) -> float:
    values = [
        float(row["robust_macro_drop"]) / max(args.internal_max_macro_drop, 1e-9),
        float(row["robust_micro_drop"]) / max(args.internal_max_micro_drop, 1e-9),
        float(row["robust_exact_drop"]) / max(args.internal_max_exact_drop, 1e-9),
        float(row["robust_hamming_increase"]) / max(args.internal_max_hamming_increase, 1e-9),
    ]
    return float(max(values))


def select_final_candidate(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, str, bool]:
    feasible = frame[frame["quality_constraints_met"] == True].copy()
    if feasible.empty:
        selected = frame.sort_values(["constraint_violation", "estimated_flops_saved_pct"], ascending=[True, False]).iloc[0]
        return selected, "fallback_minimum_constraint_violation", False
    feasible["quality_utilisation"] = feasible.apply(lambda row: quality_utilisation(row, args), axis=1)
    target = feasible[feasible["estimated_flops_saved_pct"] >= float(args.target_flops_saved_pct)].copy()
    if not target.empty:
        selected = target.sort_values(["quality_utilisation", "robust_macro_drop", "estimated_flops_saved_pct"], ascending=[True, True, False]).iloc[0]
        return selected, "minimum_robust_risk_at_target_compute", True
    selected = feasible.sort_values(["estimated_flops_saved_pct", "quality_utilisation", "robust_macro_drop"], ascending=[False, True, True]).iloc[0]
    return selected, "maximum_compute_under_internal_constraints_below_target", True


def main() -> None:
    args = build_parser().parse_args()
    problem = prepare_problem(args)
    if problem.num_exits != 5 or list(problem.taps) != [1, 2, 3, 4]:
        raise RuntimeError("v0.19 requires the fair five-exit checkpoint with tap blocks 1,2,3,4.")
    groups = problem.parent.parent_ids[problem.parent.row_to_parent]
    profile = derive_grouped_continuation_risk(
        y_true=problem.y,
        exit3_probabilities=problem.probs[2],
        exit5_probabilities=problem.probs[4],
        exit3_thresholds=problem.thresholds[2],
        exit5_thresholds=problem.thresholds[4],
        group_ids=groups,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
    lower, upper = make_targeted_bounds(
        num_labels=len(problem.labels),
        confidence_bounds=parse_pair(args.confidence_bounds, "confidence_bounds"),
        delta_bounds=parse_pair(args.delta_bounds, "delta_bounds"),
        max_label_risk_bounds=parse_pair(args.max_label_risk_bounds, "max_label_risk_bounds"),
        risk_score_bounds=parse_pair(args.risk_score_bounds, "risk_score_bounds"),
        risk_multiplier_bounds=parse_pair(args.risk_multiplier_bounds, "risk_multiplier_bounds"),
        risk_band_bounds=parse_pair(args.risk_band_bounds, "risk_band_bounds"),
        margin_bounds=parse_pair(args.margin_bounds, "margin_bounds"),
    )
    rng = np.random.default_rng(int(args.seed))
    population = random_population(size=int(args.population_size), lower=lower, upper=upper, rng=rng, seeds=seed_vectors(len(problem.labels)))
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    order: list[tuple[float, ...]] = []

    def evaluate(genes: np.ndarray) -> dict[str, Any]:
        key = tuple(float(value) for value in np.round(genes, 6))
        if key in cache:
            return cache[key]
        config = decode_targeted_genes(genes, num_labels=len(problem.labels))
        selected = targeted_select(
            exit2_probabilities=problem.probs[1],
            exit3_probabilities=problem.probs[2],
            exit5_probabilities=problem.probs[4],
            exit2_thresholds=problem.thresholds[1],
            exit3_thresholds=problem.thresholds[2],
            risk_scores=profile["risk_scores"],
            config=config,
        )
        row = evaluate_sequential_candidate(
            strategy="targeted_exit3_to_exit5_v019",
            parameters=config.to_dict(),
            selected_probabilities=selected["selected_probabilities"],
            selected_exit=selected["selected_exit"],
            y_true=problem.y,
            thresholds_by_exit=problem.thresholds,
            parent_context=problem.parent,
            flops_by_exit=problem.flops,
            max_macro_drop=args.internal_max_macro_drop,
            max_micro_drop=args.internal_max_micro_drop,
            max_exact_drop=args.internal_max_exact_drop,
            max_hamming_increase=args.internal_max_hamming_increase,
            min_total_early_fraction=args.min_exit3_fraction,
            min_exit1_fraction=0.0,
        )
        row["candidate_id"] = len(cache)
        row["genes_json"] = json.dumps([float(value) for value in genes])
        cache[key] = row
        order.append(key)
        return row

    history: list[dict[str, Any]] = []
    for generation in range(int(args.generations)):
        rows = [evaluate(genes) for genes in population]
        objectives = np.vstack([objective_vector(row) for row in rows])
        violations = np.asarray([row["constraint_violation"] for row in rows], dtype=float)
        feasible = [row for row in rows if row["quality_constraints_met"]]
        history.append({
            "generation": generation,
            "unique_candidates": len(cache),
            "feasible_population": len(feasible),
            "best_feasible_flops_saved_pct": max([row["estimated_flops_saved_pct"] for row in feasible], default=0.0),
            "minimum_constraint_violation": float(violations.min()),
        })
        offspring = make_offspring(
            population=population,
            objectives=objectives,
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
            objectives=np.vstack([objectives, np.vstack([objective_vector(row) for row in offspring_rows])]),
            violations=np.concatenate([violations, np.asarray([row["constraint_violation"] for row in offspring_rows], dtype=float)]),
            size=int(args.population_size),
        )

    all_df = pd.DataFrame([cache[key] for key in order]).drop_duplicates("genes_json")
    final_df = pd.DataFrame([evaluate(genes) for genes in population]).drop_duplicates("genes_json")
    pareto = final_df.loc[pareto_front_mask(objective_matrix(final_df), final_df["constraint_violation"].to_numpy(float))].copy()
    selected, status, eligible = select_final_candidate(all_df, args)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.sort_values(["quality_constraints_met", "estimated_flops_saved_pct"], ascending=[False, False]).to_csv(out_dir / "v019_all_candidates.csv", index=False)
    pareto.sort_values("estimated_flops_saved_pct", ascending=False).to_csv(out_dir / "v019_pareto_front.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "v019_optimization_history.csv", index=False)
    selected_record = {**{key: jsonable(value) for key, value in selected.items()}, "selection_status": status, "validation_eligible": eligible}
    pd.DataFrame([selected_record]).to_csv(out_dir / "v019_selected_policy.csv", index=False)
    parameters = json.loads(str(selected["parameters_json"]))
    frozen = {
        "schema_version": 1,
        "experiment": "v0.19_EE_final_targeted_exit3_to_exit5",
        "branch": "active_budget_anytime_exit_v0.4",
        "run_dir": str(problem.run_dir),
        "checkpoint": str(problem.checkpoint),
        "validation_manifest": str(problem.manifest),
        "validation_features_root": str(problem.features),
        "labels_json": str(args.labels_json.resolve()),
        "lats_config_json": str(args.lats_config_json.resolve()),
        "labels": problem.labels,
        "architecture": {
            "model": "ExitNet/TinyAudioCNN",
            "tap_blocks": list(problem.taps),
            "num_exits": 5,
            "decision_route": "Exit 3 -> Exit 5",
            "important_note": "Exits 1, 2 and 4 are computed as required by the backbone but are not stopping points.",
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {f"exit{index + 1}": threshold_mapping(problem.labels, values) for index, values in enumerate(problem.thresholds)},
        "grouped_risk_profile": {
            "definition": "Worst-fold validation risk from Exit-3 errors corrected by Exit 5, F1 gain, FN/FP repair and label rarity.",
            **{key: value.tolist() for key, value in profile.items()},
        },
        "optimisation": {
            "algorithm": "constraint-aware NSGA-II-style targeted search",
            "population_size": int(args.population_size),
            "generations": int(args.generations),
            "seed": int(args.seed),
            "target_flops_saved_pct": float(args.target_flops_saved_pct),
            "selection_rule": "minimum robust validation risk among candidates reaching target compute",
            "unique_candidates_evaluated": len(all_df),
            "pareto_candidates": len(pareto),
        },
        "internal_selection_constraints": {
            "max_parent_macro_f1_drop": args.internal_max_macro_drop,
            "max_parent_micro_f1_drop": args.internal_max_micro_drop,
            "max_parent_exact_match_drop": args.internal_max_exact_drop,
            "max_parent_hamming_increase": args.internal_max_hamming_increase,
            "minimum_exit3_fraction": args.min_exit3_fraction,
        },
        "deployment_constraints": {
            "max_parent_macro_f1_drop": args.deployment_max_macro_drop,
            "max_parent_micro_f1_drop": args.deployment_max_micro_drop,
            "max_parent_exact_match_drop": args.deployment_max_exact_drop,
            "max_parent_hamming_increase": args.deployment_max_hamming_increase,
        },
        "reference_final_exit_validation": problem.parent.reference_metrics,
        "selected_policy": {
            "selection_status": status,
            "validation_eligible": eligible,
            "parameters": parameters,
            "validation_metrics": {key: jsonable(value) for key, value in selected.items() if key not in {"parameters_json", "genes_json"}},
        },
        "estimated_flops_by_exit": {key: float(value) for key, value in problem.flops.items()},
        "final_experiment_rule": "If this frozen policy fails any corrected-holdout deployment constraint, stop EE development and retain the previously established safe baseline.",
    }
    policy_path = out_dir / "frozen_targeted_exit3_to_exit5_policy_v019.json"
    save_json(frozen, policy_path)
    print("\nV0.19 final targeted tuning complete")
    print("-" * 130)
    print(f"Unique candidates:      {len(all_df)}")
    print(f"Pareto candidates:      {len(pareto)}")
    print(f"Selection status:       {status}")
    print(f"Validation eligible:    {eligible}")
    print(f"Target FLOPs saved:     {args.target_flops_saved_pct:.2f}%")
    print(f"Selected FLOPs saved:   {float(selected['estimated_flops_saved_pct']):.4f}%")
    print(f"Selected Exit-3 rate:   {float(selected['exit3_fraction']):.4%}")
    print(f"Frozen policy:          {policy_path}")


if __name__ == "__main__":
    main()
