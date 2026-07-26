# v0.18_EE — Fair Strict Sequential Anytime Exit

This package documents the completed fair 3-exit versus 5-exit study.

## Headline

- Fair training audit: **PASS**.
- 3-exit `full_strict`: 3.82% FLOPs, 1.018× speedup, fails 3/4 quality limits.
- 5-exit `full_strict`: 12.70% FLOPs, 1.057× speedup, fails 4/4 limits.
- 5-exit `no_exit1`: 9.18% FLOPs, 1.037× speedup, passes 3/4 limits and misses Macro-F1 by only 0.000819.

## Package contents

- `EXPERIMENT_SETUP.md`
- `THEORY_AND_METHOD.md`
- `RESULTS_AND_ANALYSIS.md`
- `ABLATIONS_AND_FINDINGS.md`
- `PAPER_READY_SUMMARY.md`
- `PS_COMMANDS.md`
- `cross_version_3exit_table.csv`
- `fair_architecture_headline.csv`
- `v018_ablation_summary.csv`
- `v018_quality_compute_summary.svg`

## Safe conclusion

> Under matched training, the five-exit model provides more computation-saving capacity, but neither selected full sequential policy preserves all corrected-holdout quality constraints. The five-exit No-Exit-1 route is the closest current candidate.
