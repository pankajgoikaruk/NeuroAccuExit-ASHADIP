#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fair v0.18 3-exit and 5-exit strict policies.")
    parser.add_argument("--training_audit", required=True, type=Path)
    parser.add_argument("--policy3", required=True, type=Path)
    parser.add_argument("--comparison3", required=True, type=Path)
    parser.add_argument("--policy5", required=True, type=Path)
    parser.add_argument("--comparison5", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    training = load_json(args.training_audit.resolve())
    policy3 = load_json(args.policy3.resolve())
    policy5 = load_json(args.policy5.resolve())
    frame3 = pd.read_csv(args.comparison3.resolve())
    frame5 = pd.read_csv(args.comparison5.resolve())

    fairness_checks = {
        "fair_training_valid": bool(training.get("fair_training_valid", False)),
        "same_labels": policy3.get("labels") == policy5.get("labels"),
        "same_validation_manifest": Path(policy3["validation_manifest"]).resolve() == Path(policy5["validation_manifest"]).resolve(),
        "same_validation_features": Path(policy3["validation_features_root"]).resolve() == Path(policy5["validation_features_root"]).resolve(),
        "same_lats_config": Path(policy3["lats_config_json"]).name == Path(policy5["lats_config_json"]).name,
        "same_threshold_mode": policy3.get("threshold_mode") == policy5.get("threshold_mode"),
        "same_population": policy3["optimisation"]["population_size"] == policy5["optimisation"]["population_size"],
        "same_generations": policy3["optimisation"]["generations"] == policy5["optimisation"]["generations"],
        "same_constraints": policy3.get("selection_constraints") == policy5.get("selection_constraints"),
    }
    fair = bool(all(fairness_checks.values()))
    fairness = {
        "experiment": "v0.18_EE_fair_architecture_comparison",
        "fair_comparison_valid": fair,
        "checks": fairness_checks,
        "interpretation": (
            "Valid direct 3-exit versus 5-exit comparison."
            if fair else
            "Do not claim architectural superiority until every failed fairness check is resolved."
        ),
    }
    with (out / "v018_fairness_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(fairness, handle, indent=2)

    combined = pd.concat([frame3, frame5], ignore_index=True, sort=False)
    combined["fair_architecture_comparison"] = fair
    combined.to_csv(out / "v018_all_methods.csv", index=False)

    headline = combined[combined["method"].isin(["always_final", "full_strict"])].copy()
    headline.to_csv(out / "v018_fair_3exit_vs_5exit_headline.csv", index=False)

    ablation = combined[combined["method"] != "always_final"].copy()
    ablation.to_csv(out / "v018_combined_ablation_table.csv", index=False)

    structure_columns = [
        "architecture", "method", "validation_eligible", "holdout_constraints_met",
        "total_early_fraction", "average_exit_depth", "estimated_flops_saved_pct",
        "measured_speedup_vs_always_final", "parent_macro_f1", "parent_micro_f1",
        "parent_samples_f1", "parent_exact_match", "parent_hamming_loss",
        "parent_macro_f1_drop_vs_own_final", "parent_micro_f1_drop_vs_own_final",
        "parent_exact_drop_vs_own_final", "parent_hamming_increase_vs_own_final",
    ]
    for column in structure_columns:
        if column not in headline.columns:
            headline[column] = pd.NA
    headline[structure_columns].to_csv(out / "v018_policy_structure_comparison.csv", index=False)

    selected = headline[headline["method"] == "full_strict"].copy()
    selected.to_csv(out / "v018_selected_policy_summary.csv", index=False)

    print("\nV0.18 fair architecture comparison complete")
    print("-" * 160)
    print(f"Fair comparison valid: {fair}")
    print(
        headline[
            [
                "architecture", "method", "holdout_constraints_met",
                "estimated_flops_saved_pct", "measured_speedup_vs_always_final",
                "parent_macro_f1", "parent_micro_f1", "parent_exact_match",
                "parent_hamming_loss",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved tables: {out}")


if __name__ == "__main__":
    main()
