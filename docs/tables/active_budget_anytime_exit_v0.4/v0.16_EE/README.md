# NeuroAccuExit v0.16_EE

## Status

**Complete implementation and full integration; validation-eligible but not holdout-quality-compliant.**

The package records:

- exact staged/full checkpoint equivalence;
- five-fold parent-grouped validation optimisation;
- 4,078 evaluated policies and 20 Pareto candidates;
- frozen selected thresholds;
- genuine staged corrected-holdout evaluation;
- 30-repeat controlled CPU timing;
- v0.13 and full-depth matched comparators;
- per-label, cumulative, and paper-ready analysis.

## Headline result

| Item | v0.16 |
|---|---:|
| Exit-2 fraction | 7.87% |
| Average exit depth | 2.921338 |
| Estimated FLOPs saved | 5.055% |
| Median CPU speedup | 1.015× |
| Parent Macro-F1 | 0.849203 |
| Parent Micro-F1 | 0.942474 |
| Parent Samples-F1 | 0.950266 |
| Parent Exact Match | 0.854671 |
| Parent Hamming Loss | 0.016840 |

## Decision

The experiment worked as a multi-objective search and genuine compute-saving implementation, and it produced a small repeatable speedup. It did **not** fulfil the predefined holdout quality limits. Therefore:

- Always Exit 3 remains the full-quality reference;
- v0.13 per-label margin remains the selected adaptive baseline;
- v0.16 is retained as a compute-forward Pareto ablation.

## Package contents

| File | Purpose |
|---|---|
| `EXPERIMENT_SETUP.md` | Model, data, policy, search space, objectives, and constraints |
| `RESULTS_AND_ANALYSIS.md` | Validation, holdout, per-label, timing, and ablation analysis |
| `PAPER_READY_SUMMARY.md` | Reusable method/result/limitation wording |
| `PS_COMMANDS.md` | Full, reuse, custom-constraint, and reporting commands |
| `REPRODUCE_V016_EE.ps1` | Documentation entry point |
| `experiment_manifest.json` | Frozen machine-readable record |
| `checkpoint_staged_equivalence.json` | Numerical-equivalence evidence |
| `selected_policy.csv` | Selected validation candidate and genes |
| `pareto_front.csv` | Validation Pareto frontier |
| `optimization_history.csv` | Search progress |
| `holdout_comparison.csv` | Same-protocol method comparison |
| `holdout_constraint_check.csv` | Explicit holdout limit audit |
| `per_label_holdout_comparison.csv` | Label-level changes |
| `cumulative_comparison.csv` | v0.11–v0.16 summary |
| `*.svg` | Optimisation and quality-compute figures |
