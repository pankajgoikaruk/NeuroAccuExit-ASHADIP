#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
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
from policies.sequential_anytime_exit import environmental_select, make_offspring, pareto_front_mask, random_population
from policies.strict_sequential_anytime_exit_v018 import (
    decode_strict_genes,
    derive_strict_continuation_profile,
    make_strict_bounds,
    strict_sequential_select,
)
from tune_v017_helpers import objective_vector, parse_pair, select_buffered_pareto
from tune_v017_problem import prepare_problem


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune v0.18 strict sequential 3/5-exit policy.")
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--features_root", type=Path)
    p.add_argument("--labels_json", required=True, type=Path)
    p.add_argument("--lats_config_json", required=True, type=Path)
    p.add_argument("--parent_id_col", default="parent_clip_id")
    p.add_argument("--threshold_mode", choices=["tuned_per_exit", "final_exit_tuned", "fixed_0p5"], default="fixed_0p5")
    p.add_argument("--fixed_threshold", type=float, default=0.5)
    p.add_argument("--population_size", type=int, default=112)
    p.add_argument("--generations", type=int, default=70)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crossover_probability", type=float, default=0.90)
    p.add_argument("--mutation_probability", type=float, default=0.16)
    p.add_argument("--mutation_scale", type=float, default=0.05)
    p.add_argument("--confidence_bounds", default="0.55,0.995")
    p.add_argument("--delta_bounds", default="0.00,1.00")
    p.add_argument("--risk_bounds", default="0.00,1.00")
    p.add_argument("--risk_score_bounds", default="0.35,0.95")
    p.add_argument("--risk_multiplier_bounds", default="1.00,3.00")
    p.add_argument("--risk_band_bounds", default="0.01,0.25")
    p.add_argument("--exit1_boost_bounds", default="0.00,0.20")
    p.add_argument("--margin_bounds", default="0.00,0.50")
    p.add_argument("--max_macro_f1_drop", type=float, default=0.01)
    p.add_argument("--max_micro_f1_drop", type=float, default=0.005)
    p.add_argument("--max_exact_match_drop", type=float, default=0.01)
    p.add_argument("--max_hamming_increase", type=float, default=0.002)
    p.add_argument("--min_total_early_fraction", type=float, default=0.02)
    p.add_argument("--min_exit1_fraction", type=float, default=0.0025)
    p.add_argument("--safety_fraction", type=float, default=0.50)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", required=True, type=Path)
    return p


def stage_seed(num_labels: int, num_exits: int, kind: str) -> np.ndarray:
    values: list[float] = []
    for index in range(num_exits - 1):
        if kind == "conservative":
            block = [0.99, 0.02, 0.05, 0.35, 3.0, 0.25, 0.20, *([0.45] * num_labels)]
        elif kind == "permissive":
            block = [0.55, 1.00, 1.00, 0.95, 1.0, 0.01, 0.00, *([0.00] * num_labels)]
        else:
            block = [
                max(0.65, 0.95 - 0.06 * index),
                1.00 if index == 0 else min(0.50, 0.12 + 0.08 * index),
                min(0.85, 0.20 + 0.12 * index),
                min(0.85, 0.50 + 0.08 * index),
                max(1.25, 2.50 - 0.25 * index),
                max(0.04, 0.16 - 0.02 * index),
                0.12 if index == 0 else 0.0,
                *([max(0.03, 0.18 - 0.025 * index)] * num_labels),
            ]
        values.extend(block)
    return np.asarray(values, dtype=np.float64)


def optimise(problem: SimpleNamespace) -> SimpleNamespace:
    args = problem.args
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    order: list[tuple[float, ...]] = []
    population = problem.population

    def evaluate(genes: np.ndarray) -> dict[str, Any]:
        key = tuple(float(value) for value in np.round(genes, 6))
        if key in cache:
            return cache[key]
        config = decode_strict_genes(genes, num_exits=problem.num_exits, num_labels=len(problem.labels))
        selection = strict_sequential_select(
            exit_probabilities=problem.probs,
            thresholds_by_exit=problem.thresholds,
            risk_scores_by_exit=problem.strict_profile["risk_scores"],
            config=config,
        )
        row = evaluate_sequential_candidate(
            strategy=f"strict_sequential_{problem.num_exits}exit_v018",
            parameters=config.to_dict(),
            selected_probabilities=selection["selected_probabilities"],
            selected_exit=selection["selected_exit"],
            y_true=problem.y,
            thresholds_by_exit=problem.thresholds,
            parent_context=problem.parent,
            flops_by_exit=problem.flops,
            max_macro_drop=args.max_macro_f1_drop,
            max_micro_drop=args.max_micro_f1_drop,
            max_exact_drop=args.max_exact_match_drop,
            max_hamming_increase=args.max_hamming_increase,
            min_total_early_fraction=args.min_total_early_fraction,
            min_exit1_fraction=args.min_exit1_fraction,
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
            lower=problem.lower,
            upper=problem.upper,
            rng=problem.rng,
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
    pareto = final_df.loc[
        pareto_front_mask(objective_matrix(final_df), final_df["constraint_violation"].to_numpy(float))
    ].copy()
    feasible = all_df[all_df["quality_constraints_met"] == True].copy()
    if feasible.empty:
        selected = all_df.sort_values(["constraint_violation", "estimated_flops_saved_pct"], ascending=[True, False]).iloc[0]
        status = "fallback_minimum_constraint_violation"
        eligible = False
    else:
        selected, status = select_buffered_pareto(
            feasible,
            max_macro=args.max_macro_f1_drop,
            max_micro=args.max_micro_f1_drop,
            max_exact=args.max_exact_match_drop,
            max_hamming=args.max_hamming_increase,
            safety_fraction=args.safety_fraction,
        )
        eligible = True
    return SimpleNamespace(all_df=all_df, pareto=pareto, history=history, selected=selected, status=status, eligible=eligible)


def save_tuning(problem: SimpleNamespace, result: SimpleNamespace) -> None:
    args = problem.args
    prefix = f"v018_{problem.num_exits}exit"
    result.all_df.sort_values(["quality_constraints_met", "estimated_flops_saved_pct"], ascending=[False, False]).to_csv(problem.out_dir / f"{prefix}_all_candidates.csv", index=False)
    result.pareto.sort_values("estimated_flops_saved_pct", ascending=False).to_csv(problem.out_dir / f"{prefix}_pareto_front.csv", index=False)
    pd.DataFrame(result.history).to_csv(problem.out_dir / f"{prefix}_optimization_history.csv", index=False)
    pd.DataFrame([{**{key: jsonable(value) for key, value in result.selected.items()}, "selection_status": result.status, "deployment_eligible": result.eligible}]).to_csv(problem.out_dir / f"{prefix}_selected_policy.csv", index=False)
    parameters = json.loads(str(result.selected["parameters_json"]))
    frozen = {
        "schema_version": 1,
        "experiment": "v0.18_EE_strict_fair_sequential_anytime_exit",
        "branch": "active_budget_anytime_exit_v0.4",
        "architecture_name": f"{problem.num_exits}-exit",
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
            "num_exits": problem.num_exits,
            "n_mels": problem.n_mels,
            "sequential_route": " -> ".join(f"Exit {index}" for index in range(1, problem.num_exits + 1)),
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {f"exit{index + 1}": threshold_mapping(problem.labels, thresholds) for index, thresholds in enumerate(problem.thresholds)},
        "strict_risk_design": {
            "definition": "Validation-only per-label continuation risk combining final-exit correction rate and positive F1 gain.",
            **{key: value.tolist() for key, value in problem.strict_profile.items()},
            "exit1_safeguard": "Exit 1 receives an optimised confidence boost and a high-risk uncertainty veto.",
        },
        "optimisation": {
            "algorithm": "constraint-aware NSGA-II-style strict sequential search",
            "population_size": int(args.population_size),
            "generations": int(args.generations),
            "seed": int(args.seed),
            "selection": "safety-buffered Pareto knee",
            "safety_fraction": float(args.safety_fraction),
            "unique_candidates_evaluated": len(result.all_df),
            "pareto_candidates": len(result.pareto),
        },
        "selection_constraints": {
            "max_parent_macro_f1_drop": args.max_macro_f1_drop,
            "max_parent_micro_f1_drop": args.max_micro_f1_drop,
            "max_parent_exact_match_drop": args.max_exact_match_drop,
            "max_parent_hamming_increase": args.max_hamming_increase,
            "minimum_total_early_fraction": args.min_total_early_fraction,
            "minimum_exit1_fraction": args.min_exit1_fraction,
            "fold_upper_confidence_required": True,
        },
        "reference_final_exit_validation": problem.parent.reference_metrics,
        "selected_policy": {
            "selection_status": result.status,
            "deployment_eligible": result.eligible,
            "parameters": parameters,
            "validation_metrics": {key: jsonable(value) for key, value in result.selected.items() if key not in {"parameters_json", "genes_json"}},
        },
        "estimated_flops_by_exit": {key: float(value) for key, value in problem.flops.items()},
        "important_note": "The holdout remains untouched during tuning; eligibility must be reported separately for validation and holdout.",
    }
    path = problem.out_dir / f"frozen_strict_sequential_policy_{problem.num_exits}exit_v018.json"
    save_json(frozen, path)
    print(f"\nV0.18 strict sequential {problem.num_exits}-exit tuning complete")
    print("-" * 130)
    print(f"Unique candidates: {len(result.all_df)}")
    print(f"Pareto candidates: {len(result.pareto)}")
    print(f"Selection status: {result.status}")
    print(f"Validation eligible: {result.eligible}")
    print(f"Frozen policy: {path}")


def main() -> None:
    args = parser().parse_args()
    problem = prepare_problem(args)
    problem.strict_profile = derive_strict_continuation_profile(
        y_true=problem.y,
        exit_probabilities=problem.probs,
        thresholds_by_exit=problem.thresholds,
    )
    problem.lower, problem.upper = make_strict_bounds(
        num_exits=problem.num_exits,
        num_labels=len(problem.labels),
        confidence_bounds=parse_pair(args.confidence_bounds, "confidence_bounds"),
        delta_bounds=parse_pair(args.delta_bounds, "delta_bounds"),
        risk_bounds=parse_pair(args.risk_bounds, "risk_bounds"),
        risk_score_bounds=parse_pair(args.risk_score_bounds, "risk_score_bounds"),
        risk_multiplier_bounds=parse_pair(args.risk_multiplier_bounds, "risk_multiplier_bounds"),
        risk_band_bounds=parse_pair(args.risk_band_bounds, "risk_band_bounds"),
        exit1_boost_bounds=parse_pair(args.exit1_boost_bounds, "exit1_boost_bounds"),
        margin_bounds=parse_pair(args.margin_bounds, "margin_bounds"),
    )
    seeds = [stage_seed(len(problem.labels), problem.num_exits, kind) for kind in ("conservative", "permissive", "graduated")]
    problem.rng = np.random.default_rng(int(args.seed))
    problem.population = random_population(
        size=int(args.population_size),
        lower=problem.lower,
        upper=problem.upper,
        rng=problem.rng,
        seeds=seeds,
    )
    save_tuning(problem, optimise(problem))


if __name__ == "__main__":
    main()
