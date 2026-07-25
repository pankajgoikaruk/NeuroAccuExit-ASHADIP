# Documentation Update Summary — v0.17_EE

## Scope

This update extends `active_budget_anytime_exit_v0.4` documentation from v0.16 through the completed v0.17 sequential study. Existing v0.11–v0.16 records were preserved and expanded only where needed for traceability.

## Source-of-truth hierarchy

1. Frozen runtime outputs under `human_talk_workspace/active_budget_anytime_exit_v0.4/v0.17_EE/`.
2. Console logs and archived v0.17 result package.
3. Existing committed v0.11–v0.16 documentation.
4. Interpretation recorded separately from confirmed measurements.

## Confirmed v0.17 additions

- genuine staged equivalence passed at every exit for both checkpoints;
- validation-only tuning used 96 individuals, 60 generations, and a safety-buffered Pareto knee;
- three-exit and five-exit routes evaluated every non-final exit;
- six ablations were executed under frozen policies;
- 30-repeat CPU timing was completed;
- five-exit full sequential met all within-model holdout limits;
- three-exit full sequential did not meet holdout limits;
- fairness audit rejected direct architecture comparison because training manifests differ.

## Updated files

| File | Update |
|---|---|
| `README.md` | Added cumulative v0.11–v0.17 traceability, v0.17 theory, settings, results, commands, findings, limitations, and verdict. |
| `DOC_STRUCTURE.md` | Added v0.17 code paths, compact package tree, artifact descriptions, documentation rules, and status table. |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Added v0.17 progression, headline results, decisions, and commands. |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | Added complete v0.17 implementation/settings/RQ/result/ablation/fairness record while preserving v0.11–v0.16. |
| `docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md` | Added this audit and file-by-file summary. |
| `docs/tables/active_budget_anytime_exit_v0.4/README.md` | Added v0.17 package link and headline status. |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/` | Added setup, results, paper wording, commands, compact tables, manifest, and figures. |

## Confirmed versus interpretive wording

### Confirmed

- The five-exit sequential policy met all predefined holdout quality limits relative to its own Always Exit 5 baseline.
- The three-exit full sequential policy did not meet the limits.
- Label margins and stability were strongly supported by ablations.
- The current risk term was not materially active.
- The fairness audit did not validate a direct architecture comparison.

### Interpretation

- More exits appear to offer richer routing opportunities within the historical five-exit checkpoint.
- Exit 1 appears to be the highest-risk compute-saving stage.
- Intermediate exits can occasionally improve the full label set, as reflected in improved five-exit Exact Match.

These interpretations are not equivalent to a causal architectural comparison.

## What must not be overclaimed

- Do not claim that five exits are generally superior to three exits.
- Do not claim an independent external-test result.
- Do not claim the risk component is validated.
- Do not claim global Pareto optimality.
- Do not omit the unsuccessful three-exit result or negative ablations.
- Do not describe the method as label-wise asynchronous exit.

## Remaining documentation-linked work

- Train a canonical five-exit model on the same training manifest and protocol as the three-exit checkpoint.
- Repeat the v0.17 comparison with a passing fairness audit.
- Reserve an independent calibration/evaluation split for future policy selection.
- Consider a redesigned label-risk model and safer Exit-1 constraints.
