#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Combine fair 3-exit and 5-exit v0.17 results and generate paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalise_path(value: str) -> str:
    return str(Path(value)).replace("\\", "/").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_3", required=True, type=Path)
    parser.add_argument("--comparison_3", required=True, type=Path)
    parser.add_argument("--policy_5", required=True, type=Path)
    parser.add_argument("--comparison_5", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy3 = load_json(args.policy_3.resolve())
    policy5 = load_json(args.policy_5.resolve())
    frame3 = pd.read_csv(args.comparison_3.resolve())
    frame5 = pd.read_csv(args.comparison_5.resolve())

    checks = {
        "same_labels": policy3.get("labels") == policy5.get("labels"),
        "same_validation_manifest": normalise_path(policy3["validation_manifest"])
        == normalise_path(policy5["validation_manifest"]),
        "same_validation_features": normalise_path(policy3["validation_features_root"])
        == normalise_path(policy5["validation_features_root"]),
        "same_lats_config": Path(policy3["lats_config_json"]).name
        == Path(policy5["lats_config_json"]).name,
        "same_threshold_mode": policy3.get("threshold_mode")
        == policy5.get("threshold_mode"),
        "same_optimisation_budget": (
            policy3["optimisation"]["population_size"]
            == policy5["optimisation"]["population_size"]
            and policy3["optimisation"]["generations"]
            == policy5["optimisation"]["generations"]
        ),
        "same_constraints": policy3.get("selection_constraints")
        == policy5.get("selection_constraints"),
    }
    fair = bool(all(checks.values()))
    fairness = {
        "fair_comparison_valid": fair,
        "checks": checks,
        "interpretation": (
            "Valid: both architectures used the same data, labels, parent aggregation, "
            "threshold mode, optimiser budget and quality constraints."
            if fair
            else "Invalid for a direct architecture claim until every failed check is resolved."
        ),
    }
    with (out_dir / "v017_fairness_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(fairness, handle, indent=2)

    combined = pd.concat([frame3, frame5], ignore_index=True, sort=False)
    combined.to_csv(out_dir / "v017_all_architecture_methods.csv", index=False)

    headline = combined[combined["method"].isin(["always_final", "full_sequential"])].copy()
    baseline_by_architecture = {
        row["architecture"]: row
        for _, row in headline[headline["method"] == "always_final"].iterrows()
    }
    for index, row in headline.iterrows():
        baseline = baseline_by_architecture[row["architecture"]]
        headline.loc[index, "parent_macro_f1_drop_vs_own_final"] = float(
            baseline["parent_macro_f1"] - row["parent_macro_f1"]
        )
        headline.loc[index, "parent_micro_f1_drop_vs_own_final"] = float(
            baseline["parent_micro_f1"] - row["parent_micro_f1"]
        )
        headline.loc[index, "parent_exact_drop_vs_own_final"] = float(
            baseline["parent_exact_match"] - row["parent_exact_match"]
        )
        headline.loc[index, "parent_hamming_increase_vs_own_final"] = float(
            row["parent_hamming_loss"] - baseline["parent_hamming_loss"]
        )
    headline["fair_architecture_comparison"] = fair
    headline.to_csv(out_dir / "v017_3exit_vs_5exit_headline.csv", index=False)

    ablation = combined[combined["method"] != "always_final"].copy()
    desired = [
        "architecture",
        "method",
        "validation_eligible",
        "exit1_fraction",
        "exit2_fraction",
        "exit3_fraction",
        "exit4_fraction",
        "exit5_fraction",
        "total_early_fraction",
        "average_exit_depth",
        "estimated_flops_saved_pct",
        "latency_median_per_segment_ms",
        "measured_speedup_vs_always_final",
        "parent_macro_f1",
        "parent_micro_f1",
        "parent_samples_f1",
        "parent_exact_match",
        "parent_hamming_loss",
    ]
    for column in desired:
        if column not in ablation.columns:
            ablation[column] = pd.NA
    ablation[desired].to_csv(out_dir / "v017_combined_ablation_table.csv", index=False)

    exit_distribution_columns = [
        "architecture",
        "method",
        "exit1_fraction",
        "exit2_fraction",
        "exit3_fraction",
        "exit4_fraction",
        "exit5_fraction",
        "average_exit_depth",
        "estimated_flops_saved_pct",
    ]
    headline_exit = combined[combined["method"] == "full_sequential"].copy()
    for column in exit_distribution_columns:
        if column not in headline_exit.columns:
            headline_exit[column] = pd.NA
    headline_exit[exit_distribution_columns].to_csv(
        out_dir / "v017_exit_distribution_comparison.csv", index=False
    )

    print("\nV0.17 cross-architecture comparison complete")
    print("-" * 150)
    print(f"Fair comparison valid: {fair}")
    print(
        headline[
            [
                "architecture",
                "method",
                "estimated_flops_saved_pct",
                "measured_speedup_vs_always_final",
                "parent_macro_f1",
                "parent_micro_f1",
                "parent_exact_match",
                "parent_hamming_loss",
                "parent_macro_f1_drop_vs_own_final",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved tables: {out_dir}")


if __name__ == "__main__":
    main()
