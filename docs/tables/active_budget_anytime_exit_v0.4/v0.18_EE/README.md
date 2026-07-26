# v0.18_EE — Fair Strict Sequential Anytime Exit

This package documents the completed fair 3-exit versus 5-exit study.

## Headline

- Fair training audit: **PASS**.
- 3-exit `full_strict`: 3.82% FLOPs, 1.018× speedup, fails 3/4 quality limits.
- 5-exit `full_strict`: 12.70% FLOPs, 1.057× speedup, fails 4/4 limits.
- 5-exit `no_exit1`: 9.18% FLOPs, 1.037× speedup, passes 3/4 limits and misses Macro-F1 by only 0.000819.

## Human-readable records

| File | Purpose |
|---|---|
| `EXPERIMENT_SETUP.md` | Data, fair training, architectures, optimiser settings, constraints, and protocol |
| `THEORY_AND_METHOD.md` | Strict sequential stopping theory, continuation risk, and genuine staged inference |
| `RESULTS_AND_ANALYSIS.md` | Training, validation, holdout, timing, transfer, parent-change, and per-label analysis |
| `ABLATIONS_AND_FINDINGS.md` | Exit-1, risk, stability, label-margin, and confidence-only ablations |
| `PAPER_READY_SUMMARY.md` | Final verdict, unsuccessful finding, safe wording, and non-claims |
| `PS_COMMANDS.md` | Training, audit, tuning, evaluation, reporting, and frozen-policy reuse |
| `CROSS_VERSION_AND_HISTORICAL_TABLES.md` | Canonical 3-exit, historical v0.17 5-exit, and fair v0.18 comparisons |
| `LATEX_TABLES.md` | Paper-ready canonical, historical, fair-architecture, and policy-structure tables |
| `FIGURES.md` | Figure links, interpretation, and cautions |
| `REPRODUCE_V018_EE.ps1` | Reproduction entry point |

## Machine-readable records

- `experiment_manifest.json`
- `fair_training_audit.json`
- `fair_architecture_audit.json`
- `holdout_constraint_check.csv`
- `cross_version_3exit_table.csv`
- `fair_architecture_headline.csv`
- `v018_fair_architecture_table.csv`
- `v018_3exit_holdout_comparison.csv`
- `v018_5exit_holdout_comparison.csv`
- `v018_combined_ablation_table.csv`
- `v018_policy_structure_comparison.csv`
- `v018_selected_policy_summary.csv`
- `v018_3exit_selected_policy.csv`
- `v018_5exit_selected_policy.csv`
- `v018_3exit_optimization_history.csv`
- `v018_5exit_optimization_history.csv`
- `v018_fair5_test_metrics_by_exit.csv`
- `validation_to_holdout_transfer.csv`
- `parent_change_summary.csv`
- `per_label_holdout_delta_3exit.csv`
- `per_label_holdout_delta_5exit.csv`

## Figures

- `v018_quality_compute_summary.svg`
- `cross_version_3exit_flops_v018.svg`
- `cross_version_3exit_quality_v018.svg`
- `v018_fair_architecture_flops.svg`
- `v018_exit_distribution.svg`
- `v018_constraint_utilisation.svg`

`CROSS_VERSION_AND_HISTORICAL_TABLES.md` is the authoritative comparison guide. The canonical ranking remains restricted to the 3-exit family; the historical v0.17 and fair v0.18 five-exit results remain separate.

## Safe conclusion

> Under matched training, the five-exit model provides more computation-saving capacity, but neither selected full sequential policy preserves all corrected-holdout quality constraints. The five-exit No-Exit-1 route is the closest current candidate.
