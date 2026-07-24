# Active Budget and Anytime Exit v0.4 — Documentation Update Summary

## Scope

This record summarises the documentation freeze for branch:

```text
active_budget_anytime_exit_v0.4
```

The documented implementation milestone is `v0.16_EE`, which applies constraint-aware, NSGA-II-style multi-objective optimisation to the lightweight per-label Exit-2 margin policy. The neural checkpoint, TinyAudioCNN backbone, and exit heads remain frozen.

## Sources checked

The documentation values were checked against the completed v0.16 runtime package and the frozen repository artifacts:

```text
human_talk_workspace/active_budget_anytime_exit_v0.4/v0.16_EE/multiobjective_per_label_margin/
├── validation_tuning/
└── corrected_holdout_evaluation/
```

The documentation audit covered:

- staged/full checkpoint-equivalence output;
- selected frozen policy;
- validation candidate and Pareto tables;
- optimisation history;
- corrected-holdout method comparison;
- per-label holdout metrics;
- 30-repeat CPU timing;
- cumulative v0.11–v0.16 ablations.

## Confirmed v0.16 result

| Item | Confirmed value |
|---|---:|
| Validation segments / parents | 1,883 / 304 |
| Holdout segments / parents | 4,335 / 867 |
| Population / generations | 80 / 50 |
| Unique candidates evaluated | 4,078 |
| Validation Pareto candidates | 20 |
| Selected validation Exit-2 fraction | 19.6495% |
| Selected validation FLOPs saved | 12.6272% |
| Holdout Exit-2 fraction | 7.8662% |
| Holdout estimated FLOPs saved | 5.0550% |
| 30-repeat CPU speedup | 1.0153× |
| Holdout Parent Macro-F1 | 0.849203 |
| Holdout Parent Micro-F1 | 0.942474 |
| Holdout Parent Samples-F1 | 0.950266 |
| Holdout Parent Exact Match | 0.854671 |
| Holdout Parent Hamming Loss | 0.016840 |

## Requirement outcome

The selected policy was feasible on validation but failed all predefined holdout quality limits:

| Constraint | Limit | Observed | Outcome |
|---|---:|---:|---|
| Parent Macro-F1 drop | ≤ 0.010 | 0.013178 | Failed |
| Parent Micro-F1 drop | ≤ 0.005 | 0.010657 | Failed |
| Parent Exact-Match drop | ≤ 0.010 | 0.021915 | Failed |
| Parent Hamming increase | ≤ 0.002 | 0.003114 | Failed |

Repository wording must therefore distinguish:

```text
validation_eligible = true
holdout_constraints_met = false
```

## Confirmed findings versus interpretation

### Confirmed

- Genuine staged inference remained numerically equivalent to full-forward inference at all exits.
- v0.16 increased holdout Exit-2 coverage and estimated saving relative to v0.13.
- v0.16 produced a small same-protocol measured CPU speedup.
- v0.16 did not satisfy the predefined corrected-holdout quality limits.
- `audience_reaction_present` and `other_speaker_present` were important sources of quality degradation.
- v0.13 per-label margin remains the recommended quality-constrained adaptive baseline.

### Interpretation

- Selecting the maximum-compute feasible validation point was too aggressive for holdout transfer.
- The weak maximum-delta constraint and near-zero margins for several labels contributed to permissive stopping.
- A safety-buffered Pareto-knee rule or a new calibration/evaluation split may improve robustness, but this is future work rather than a confirmed v0.16 result.

## Documentation files covered

| File or directory | Documentation role |
|---|---|
| `README.md` | Authoritative branch identity, theory, v0.16 settings, selected policy, headline results, commands, limitations, and current decision |
| `DOC_STRUCTURE.md` | Repository path index, experiment-to-code mapping, package inventory, and documentation rules |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Concise branch-level research overview |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | Version-by-version implementation, settings, research questions, results, and findings from v0.11 to v0.16 |
| `docs/tables/active_budget_anytime_exit_v0.4/README.md` | Compact result-package index |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/EXPERIMENT_SETUP.md` | Architecture, data, search space, objective functions, constraints, and protocol |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/RESULTS_AND_ANALYSIS.md` | Validation, holdout, per-label, timing, ablation, and research interpretation |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/PAPER_READY_SUMMARY.md` | Paper/thesis-safe wording and non-claims |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/PS_COMMANDS.md` | PowerShell commands for full execution, tuning, evaluation, frozen-policy reuse, and reporting |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/*.csv` | Machine-readable selected policy, Pareto, optimisation, holdout, per-label, constraint, and cumulative summaries |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/*.svg` | Optimisation-progress, validation-Pareto, and holdout quality–compute figures |

## Non-claims

- v0.16 is not the deployment winner.
- Validation eligibility is not holdout approval.
- The measured speedup is hardware-, CPU-, threading-, batch-, and implementation-specific.
- Estimated FLOPs are not interchangeable with latency.
- v0.16 is not a budget-conditioned anytime sweep.
- v0.16 is not label-wise asynchronous exit.
- The corrected holdout must not be retuned and then described as untouched evaluation.
- The corrected-holdout result is not an independent external-test result because the frozen LATS-v2 configuration has related calibration provenance.
