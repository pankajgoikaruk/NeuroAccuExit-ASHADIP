# Documentation Structure — Active Budget and Anytime Exit v0.4

This file indexes the complete computation-adaptive inference documentation inherited by `active_budget_anytime_exit_v0.4` and the new v0.16 multi-objective package.

---

## Active branch

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.4` |
| Source branch | `active_budget_anytime_exit_v0.3` |
| Completed milestone | `v0.16_EE` multi-objective per-label margin optimisation |
| Canonical comparator | v0.10 no-hint + frozen historical LATS-v2 + Always Exit 3 |
| Current adaptive baseline | v0.13 per-label margin |
| v0.16 status | Fully integrated; validation-eligible; holdout quality constraints not met |

Historical records remain authoritative for their own versions and must not be silently rewritten.

---

## Top-level documentation

| Path | Purpose |
|---|---|
| `README.md` | Authoritative v0.4 branch summary, theory, cumulative results, commands, conclusions, and limitations |
| `DOC_STRUCTURE.md` | This documentation and artifact index |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Detailed branch-level experiment overview |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | Version-to-implementation and research-question traceability from v0.11 to v0.16 |
| `docs/tables/active_budget_anytime_exit_v0.4/README.md` | v0.4 compact-results package index |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/` | Complete committed v0.16 experiment record |
| `docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/` | Frozen full-depth baseline reproducibility package |
| `docs/active_budget_anytime_exit_v0.2/` | Historical v0.11 branch documentation |
| `docs/tables/active_budget_anytime_exit_v0.2/` | Historical v0.11 compact records |
| `docs/v0.10/` and `docs/v0.10_1/` | Earlier model, hint-pass, calibration, and low-energy-recovery documentation |
| `docs/archive/` | Archived documentation states |

---

## Experiment-to-code traceability

| Version | Primary implementation paths |
|---|---|
| v0.11 | `models/anytime_exit_net.py`; `scripts/v0.11_EE/fixed_policy/`; `scripts/v0.11_EE/dynamic_policy/`; `tests/test_anytime_exit_net.py` |
| v0.12 | `policies/label_aware_early_exit_policy.py`; `scripts/v0.12_EE/label_aware_policy/`; `tests/test_label_aware_early_exit_policy.py` |
| v0.13 | `policies/early_exit_strategy_comparison.py`; `scripts/v0.13_EE/matched_policy_comparison/`; `tests/test_early_exit_strategy_comparison.py` |
| v0.14 | `policies/parent_aware_adaptive_gate.py`; `scripts/v0.14_EE/parent_aware_gate/`; `tests/test_parent_aware_adaptive_gate.py` |
| v0.15 | `policies/whole_parent_selective_exit.py`; `scripts/v0.15_EE/whole_parent_risk_control/`; `tests/test_whole_parent_selective_exit.py` |
| v0.16 | `policies/multiobjective_per_label_margin.py`; `scripts/v0.16_EE/multiobjective_per_label_margin/`; `tests/test_multiobjective_per_label_margin.py` |

---

## v0.16 committed package

```text
docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/
├── README.md
├── EXPERIMENT_SETUP.md
├── RESULTS_AND_ANALYSIS.md
├── PAPER_READY_SUMMARY.md
├── PS_COMMANDS.md
├── REPRODUCE_V016_EE.ps1
├── experiment_manifest.json
├── checkpoint_staged_equivalence.json
├── selected_policy.csv
├── pareto_front.csv
├── optimization_history.csv
├── holdout_comparison.csv
├── holdout_constraint_check.csv
├── per_label_holdout_comparison.csv
├── cumulative_comparison.csv
├── optimization_progress.svg
├── validation_pareto_quality_compute.svg
└── holdout_quality_compute.svg
```

### Human-readable records

| File | Purpose |
|---|---|
| `README.md` | Status, headline results, package index, and decision |
| `EXPERIMENT_SETUP.md` | Architecture, data, search space, objectives, constraints, and evaluation protocol |
| `RESULTS_AND_ANALYSIS.md` | Validation, holdout, per-label, timing, ablation, and interpretation |
| `PAPER_READY_SUMMARY.md` | Reusable paper/thesis wording, tables, captions, and limitations |
| `PS_COMMANDS.md` | Windows commands for full run, tuning, evaluation, reuse, and reporting |
| `REPRODUCE_V016_EE.ps1` | Documentation-level entry point to the branch runner |

### Machine-readable records

| File | Purpose |
|---|---|
| `experiment_manifest.json` | Frozen branch, method, settings, selected policy, results, and status metadata |
| `checkpoint_staged_equivalence.json` | Real-checkpoint staged/full equivalence report |
| `selected_policy.csv` | Selected validation Pareto point and all 12 parameters |
| `pareto_front.csv` | Twenty validation Pareto candidates |
| `optimization_history.csv` | Feasible-search progress over 50 generations |
| `holdout_comparison.csv` | Same-protocol full-depth, v0.13, and v0.16 comparison |
| `holdout_constraint_check.csv` | Explicit post-hoc audit of predefined holdout limits |
| `per_label_holdout_comparison.csv` | Per-label precision, recall, F1, and error counts |
| `cumulative_comparison.csv` | v0.11–v0.16 ablation history |

---

## Runtime outputs not committed wholesale

Large predictions remain under:

```text
human_talk_workspace/active_budget_anytime_exit_v0.4/v0.16_EE/multiobjective_per_label_margin/
```

This includes full segment predictions, parent probability tables, and complete evaluated-candidate tables. The committed package contains compact records sufficient for interpretation and traceability, while the workspace artifacts remain the source for exact regeneration.

---

## Documentation rules

1. Always identify the canonical comparator as v0.10 no-hint + frozen historical LATS-v2 + Always Exit 3.
2. Do not confuse average predicted labels with average exit depth.
3. Distinguish validation eligibility from holdout constraint satisfaction.
4. Do not retune a policy after inspecting corrected-holdout results and then claim an untouched test.
5. Separate architecture-estimated FLOP saving from measured latency.
6. Report Parent Macro-F1 and Parent Micro-F1 together.
7. Preserve Exact Match and Hamming Loss because aggregate F1 can hide multi-label inconsistency.
8. Preserve per-label results, especially `audience_reaction_present`, `other_speaker_present`, and `silence_present`.
9. State whether the CNN was retrained, frozen, or only wrapped by a controller.
10. Label learned gate/controller training separately from backbone training.
11. Do not call v0.16 a full anytime controller; it selects one frozen operating point.
12. Do not call v0.16 the deployment winner; v0.13 remains the quality-constrained adaptive baseline.

---

## Current documentation status

| Area | Status |
|---|---|
| Root branch README | Updated through v0.16 |
| Root documentation index | Updated through v0.16 |
| v0.4 branch overview | Complete |
| v0.11–v0.16 version traceability | Complete |
| v0.16 theory and setup | Complete |
| v0.16 validation/Pareto records | Complete |
| v0.16 holdout and timing records | Complete |
| v0.16 per-label analysis | Complete |
| v0.16 paper-ready wording | Complete |
| Budget-conditioned anytime curve | Not yet implemented |
| Label-wise asynchronous exit | Future work, not part of v0.16 |
