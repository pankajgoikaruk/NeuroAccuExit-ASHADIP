#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V017_DIR = PROJECT_ROOT / "scripts" / "v0.17_EE" / "sequential_anytime_exit"
for path in (PROJECT_ROOT, V017_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_v017 import evaluate_sequential_candidate, jsonable, objective_matrix, save_json, threshold_mapping
from policies.sequential_anytime_exit import environmental_select, make_offspring, pareto_front_mask, random_population
from policies.strict_sequential_anytime_exit_v018 import derive_strict_continuation_profile
from policies.exit3_to_exit5_v019 import bounds, decode_genes, select_exit3_or_exit5, to_v018_compatible_policy
from tune_v017_helpers import objective_vector
from tune_v017_problem import prepare_problem


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final targeted Exit-3 to Exit-5 policy optimisation.")
    p.add_argument("--run_dir", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--features_root", type=Path)
    p.add_argument("--labels_json", required=True, type=Path)
    p.add_argument("--lats_config_json", required=True, type=Path)
    p.add_argument("--parent_id_col", default="parent_clip_id")
    p.add_argument("--threshold_mode", choices=["tuned_per_exit", "final_exit_tuned", "fixed_0p5"], default="fixed_0p5")
    p.add_argument("--fixed_threshold", type=float, default=0.5)

    # Compatibility arguments required by the shared v0.17 prepare_problem() helper.
    # v0.19 does not optimise these sequential bounds directly; its focused Exit-3
    # search constructs its own bounds below. They are kept here so the shared data,
    # checkpoint, threshold, parent-metric and FLOP preparation path can be reused.
    p.add_argument("--confidence_bounds", default="0.55,0.995")
    p.add_argument("--delta_bounds", default="0.00,1.00")
    p.add_argument("--risk_bounds", default="0.00,1.00")
    p.add_argument("--margin_bounds", default="0.00,0.50")

    p.add_argument("--population_size", type=int, default=160)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--crossover_probability", type=float, default=0.90)
    p.add_argument("--mutation_probability", type=float, default=0.18)
    p.add_argument("--mutation_scale", type=float, default=0.04)
    p.add_argument("--max_macro_f1_drop", type=float, default=0.01)
    p.add_argument("--max_micro_f1_drop", type=float, default=0.005)
    p.add_argument("--max_exact_match_drop", type=float, default=0.01)
    p.add_argument("--max_hamming_increase", type=float, default=0.002)
    p.add_argument("--min_total_early_fraction", type=float, default=0.05)
    p.add_argument("--safety_fraction", type=float, default=0.35)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", required=True, type=Path)
    return p


def seed_vectors(num_labels: int) -> list[np.ndarray]:
    return [
        np.asarray([0.97, 0.08, 0.15, 0.45, 3.0, 0.20, *([0.40] * num_labels)]),
        np.asarray([0.90, 0.15, 0.30, 0.60, 2.5, 0.14, *([0.28] * num_labels)]),
        np.asarray([0.82, 0.25, 0.50, 0.72, 2.0, 0.10, *([0.18] * num_labels)]),
        np.asarray([0.72, 0.40, 0.75, 0.85, 1.5, 0.05, *([0.08] * num_labels)]),
    ]


def within_safety(row: pd.Series, args: argparse.Namespace) -> bool:
    return bool(
        float(row["robust_macro_drop"]) <= args.max_macro_f1_drop * args.safety_fraction
        and float(row["robust_micro_drop"]) <= args.max_micro_f1_drop * args.safety_fraction
        and float(row["robust_exact_drop"]) <= args.max_exact_match_drop * args.safety_fraction
        and float(row["robust_hamming_increase"]) <= args.max_hamming_increase * args.safety_fraction
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def progress_bar(completed: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    completed = min(max(0, int(completed)), total)
    filled = int(round(width * completed / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def optimise(problem: SimpleNamespace) -> SimpleNamespace:
    args = problem.args
    lower, upper = bounds(len(problem.labels))
    rng = np.random.default_rng(args.seed)
    population = random_population(
        size=args.population_size,
        lower=lower,
        upper=upper,
        rng=rng,
        seeds=seed_vectors(len(problem.labels)),
    )
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    order: list[tuple[float, ...]] = []
    history: list[dict[str, Any]] = []
    search_started = time.perf_counter()
    progress_csv = problem.out_dir / "v019_optimization_progress.csv"

    print(
        f"\nV0.19 optimisation started: {args.generations} generations x "
        f"{args.population_size} population (plus offspring evaluations).",
        flush=True,
    )
    print("Progress will be reported after every completed generation.", flush=True)

    def evaluate(genes: np.ndarray) -> dict[str, Any]:
        key = tuple(float(v) for v in np.round(genes, 6))
        if key in cache:
            return cache[key]
        config = decode_genes(genes, len(problem.labels))
        selection = select_exit3_or_exit5(
            exit2_probabilities=problem.probs[1],
            exit3_probabilities=problem.probs[2],
            exit5_probabilities=problem.probs[4],
            thresholds_by_exit=problem.thresholds,
            exit3_risk_scores=problem.profile["risk_scores"][2],
            config=config,
        )
        row = evaluate_sequential_candidate(
            strategy="v019_exit3_to_exit5_final",
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
            min_exit1_fraction=0.0,
        )
        row["candidate_id"] = len(cache)
        row["genes_json"] = json.dumps([float(v) for v in genes])
        cache[key] = row
        order.append(key)
        return row

    for generation in range(args.generations):
        rows = [evaluate(g) for g in population]
        objectives = np.vstack([objective_vector(r) for r in rows])
        violations = np.asarray([r["constraint_violation"] for r in rows], dtype=float)
        feasible = [r for r in rows if r["quality_constraints_met"]]

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
        offspring_rows = [evaluate(g) for g in offspring]
        population, _, _ = environmental_select(
            population=np.vstack([population, offspring]),
            objectives=np.vstack([objectives, np.vstack([objective_vector(r) for r in offspring_rows])]),
            violations=np.concatenate([violations, np.asarray([r["constraint_violation"] for r in offspring_rows])]),
            size=args.population_size,
        )

        completed = generation + 1
        elapsed = time.perf_counter() - search_started
        average_generation_seconds = elapsed / completed
        eta_seconds = average_generation_seconds * (args.generations - completed)
        all_feasible = [r for r in cache.values() if r["quality_constraints_met"]]
        best_feasible_flops = max(
            [float(r["estimated_flops_saved_pct"]) for r in all_feasible],
            default=0.0,
        )
        minimum_violation = min(
            [float(r["constraint_violation"]) for r in cache.values()],
            default=float("inf"),
        )
        progress_row = {
            "generation": completed,
            "generations_total": int(args.generations),
            "progress_pct": 100.0 * completed / max(int(args.generations), 1),
            "unique_candidates": len(cache),
            "feasible_candidates_seen": len(all_feasible),
            "best_feasible_flops_saved_pct": best_feasible_flops,
            "minimum_constraint_violation_seen": minimum_violation,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "average_generation_seconds": average_generation_seconds,
        }
        history.append(progress_row)
        pd.DataFrame(history).to_csv(progress_csv, index=False)

        print(
            f"{progress_bar(completed, args.generations)} "
            f"{completed:3d}/{args.generations} "
            f"({progress_row['progress_pct']:5.1f}%) | "
            f"candidates={len(cache):5d} | "
            f"feasible={len(all_feasible):4d} | "
            f"best_FLOPs={best_feasible_flops:6.2f}% | "
            f"elapsed={format_duration(elapsed)} | "
            f"ETA={format_duration(eta_seconds)}",
            flush=True,
        )

    all_df = pd.DataFrame([cache[k] for k in order]).drop_duplicates("genes_json")
    final_df = pd.DataFrame([evaluate(g) for g in population]).drop_duplicates("genes_json")
    pareto = final_df.loc[
        pareto_front_mask(objective_matrix(final_df), final_df["constraint_violation"].to_numpy(float))
    ].copy()
    feasible = all_df[all_df["quality_constraints_met"] == True].copy()
    safe = feasible[feasible.apply(lambda row: within_safety(row, args), axis=1)].copy() if not feasible.empty else feasible

    if not safe.empty:
        selected = safe.sort_values(["estimated_flops_saved_pct", "total_early_fraction"], ascending=False).iloc[0]
        status = "safety_buffered_feasible_max_compute"
        eligible = True
    elif not feasible.empty:
        selected = feasible.sort_values(["estimated_flops_saved_pct", "constraint_violation"], ascending=[False, True]).iloc[0]
        status = "feasible_max_compute_without_safety_buffer"
        eligible = True
    else:
        selected = all_df.sort_values(["constraint_violation", "estimated_flops_saved_pct"], ascending=[True, False]).iloc[0]
        status = "fallback_minimum_constraint_violation"
        eligible = False

    return SimpleNamespace(
        all_df=all_df,
        pareto=pareto,
        history=history,
        selected=selected,
        status=status,
        eligible=eligible,
    )


def main() -> None:
    args = parser().parse_args()
    problem = prepare_problem(args)
    if problem.num_exits != 5:
        raise RuntimeError(f"v0.19 requires the fair five-exit checkpoint, found {problem.num_exits} exits.")

    problem.profile = derive_strict_continuation_profile(
        y_true=problem.y,
        exit_probabilities=problem.probs,
        thresholds_by_exit=problem.thresholds,
    )
    result = optimise(problem)

    args.out_dir.resolve().mkdir(parents=True, exist_ok=True)
    result.all_df.to_csv(args.out_dir / "v019_all_candidates.csv", index=False)
    result.pareto.to_csv(args.out_dir / "v019_pareto_front.csv", index=False)
    pd.DataFrame(result.history).to_csv(args.out_dir / "v019_optimization_history.csv", index=False)
    pd.DataFrame([
        {
            **{k: jsonable(v) for k, v in result.selected.items()},
            "selection_status": result.status,
            "deployment_eligible": result.eligible,
        }
    ]).to_csv(args.out_dir / "v019_selected_policy.csv", index=False)

    focused = decode_genes(json.loads(str(result.selected["genes_json"])), len(problem.labels))
    adapted = to_v018_compatible_policy(focused)
    frozen = {
        "schema_version": 1,
        "experiment": "v0.18_EE_strict_fair_sequential_anytime_exit",
        "source_experiment": "v0.19_EE_final_targeted_exit3_to_exit5",
        "branch": "active_budget_anytime_exit_v0.4",
        "architecture_name": "5-exit",
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
            "n_mels": problem.n_mels,
            "sequential_route": "Exit 3 -> Exit 5 only",
            "disabled_stopping_exits": [1, 2, 4],
        },
        "threshold_mode": args.threshold_mode,
        "thresholds_by_exit": {
            f"exit{i + 1}": threshold_mapping(problem.labels, threshold)
            for i, threshold in enumerate(problem.thresholds)
        },
        "strict_risk_design": {
            "definition": "Validation-only continuation risk; only the Exit-3 risk vector is used.",
            **{k: v.tolist() for k, v in problem.profile.items()},
        },
        "optimisation": {
            "algorithm": "constraint-aware NSGA-II-style focused Exit-3 gate search",
            "population_size": args.population_size,
            "generations": args.generations,
            "seed": args.seed,
            "selection": result.status,
            "safety_fraction": args.safety_fraction,
            "unique_candidates_evaluated": len(result.all_df),
            "pareto_candidates": len(result.pareto),
        },
        "selection_constraints": {
            "max_parent_macro_f1_drop": args.max_macro_f1_drop,
            "max_parent_micro_f1_drop": args.max_micro_f1_drop,
            "max_parent_exact_match_drop": args.max_exact_match_drop,
            "max_parent_hamming_increase": args.max_hamming_increase,
            "minimum_total_early_fraction": args.min_total_early_fraction,
            "minimum_exit1_fraction": 0.0,
            "fold_upper_confidence_required": True,
        },
        "reference_final_exit_validation": problem.parent.reference_metrics,
        "selected_policy": {
            "selection_status": result.status,
            "deployment_eligible": result.eligible,
            "focused_exit3_parameters": focused.to_dict(),
            "parameters": adapted.to_dict(),
            "validation_metrics": {
                k: jsonable(v)
                for k, v in result.selected.items()
                if k not in {"parameters_json", "genes_json"}
            },
        },
        "estimated_flops_by_exit": {k: float(v) for k, v in problem.flops.items()},
        "important_note": (
            "The corrected holdout remains untouched during tuning. "
            "Exits 1, 2 and 4 are disabled as stopping points."
        ),
    }

    policy_path = args.out_dir / "frozen_exit3_to_exit5_policy_v019.json"
    save_json(frozen, policy_path)
    print("\nV0.19 final targeted Exit-3 to Exit-5 tuning complete")
    print("-" * 120)
    print(f"Unique candidates: {len(result.all_df)}")
    print(f"Pareto candidates: {len(result.pareto)}")
    print(f"Selection status: {result.status}")
    print(f"Validation eligible: {result.eligible}")
    print(f"Frozen policy: {policy_path}")


if __name__ == "__main__":
    main()
