# Documentation Structure — Active Budget and Anytime Exit v0.4

This index covers the complete `active_budget_anytime_exit_v0.4` programme through `v0.18_EE`.

## Active branch

| Item | Value |
|---|---|
| Branch | `active_budget_anytime_exit_v0.4` |
| Current milestone | `v0.18_EE` fair strict sequential anytime exit |
| Canonical comparator | v0.10 no-hint + frozen historical LATS-v2 + Always Exit 3 |
| Fair five-exit comparator | v0.18 auxiliary-budget-matched 5-exit checkpoint + Always Exit 5 |
| Fair-training audit | Passed |
| v0.18 full-policy status | Both full policies fail corrected-holdout constraints |
| Closest candidate | 5-exit `no_exit1`, passing 3/4 limits |

Historical records remain authoritative for their own versions and are not silently rewritten.

## Top-level documentation

| Path | Purpose |
|---|---|
| `README.md` | Authoritative branch summary, v0.18 verdict, theory, settings, results, commands, comparisons, limitations, and current decision |
| `DOC_STRUCTURE.md` | Documentation and artifact index |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Branch-level research overview through v0.18 |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | v0.11–v0.18 version traceability |
| `docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md` | File-by-file documentation update record |
| `docs/tables/active_budget_anytime_exit_v0.4/README.md` | Compact result-package index through v0.18 |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/` | v0.16 package |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/` | Historical v0.17 package |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.18_EE/` | Fair training, strict sequential policy, results, ablations, commands, tables, audits, LaTeX, and figures |
| `docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/` | Frozen canonical baseline package |

## Experiment-to-code traceability

| Version | Primary implementation paths |
|---|---|
| v0.11 | `models/anytime_exit_net.py`; `scripts/v0.11_EE/fixed_policy/`; `scripts/v0.11_EE/dynamic_policy/`; `tests/test_anytime_exit_net.py` |
| v0.12 | `policies/label_aware_early_exit_policy.py`; `scripts/v0.12_EE/label_aware_policy/`; `tests/test_label_aware_early_exit_policy.py` |
| v0.13 | `policies/early_exit_strategy_comparison.py`; `scripts/v0.13_EE/matched_policy_comparison/`; `tests/test_early_exit_strategy_comparison.py` |
| v0.14 | `policies/parent_aware_adaptive_gate.py`; `scripts/v0.14_EE/parent_aware_gate/`; `tests/test_parent_aware_adaptive_gate.py` |
| v0.15 | `policies/whole_parent_selective_exit.py`; `scripts/v0.15_EE/whole_parent_risk_control/`; `tests/test_whole_parent_selective_exit.py` |
| v0.16 | `policies/multiobjective_per_label_margin.py`; `scripts/v0.16_EE/multiobjective_per_label_margin/`; `tests/test_multiobjective_per_label_margin.py` |
| v0.17 | `policies/sequential_anytime_exit.py`; `scripts/v0.17_EE/sequential_anytime_exit/`; `tests/test_sequential_anytime_exit.py` |
| v0.18 | `policies/strict_sequential_anytime_exit_v018.py`; `scripts/v0.18_EE/fair_sequential_anytime_exit/`; `tests/test_strict_sequential_anytime_exit_v018.py` |

## v0.18 documentation package

```text
docs/tables/active_budget_anytime_exit_v0.4/v0.18_EE/
├── README.md
├── EXPERIMENT_SETUP.md
├── THEORY_AND_METHOD.md
├── RESULTS_AND_ANALYSIS.md
├── ABLATIONS_AND_FINDINGS.md
├── PAPER_READY_SUMMARY.md
├── PS_COMMANDS.md
├── CROSS_VERSION_AND_HISTORICAL_TABLES.md
├── LATEX_TABLES.md
├── FIGURES.md
├── REPRODUCE_V018_EE.ps1
├── experiment_manifest.json
├── fair_training_audit.json
├── fair_architecture_audit.json
├── holdout_constraint_check.csv
├── cross_version_3exit_table.csv
├── fair_architecture_headline.csv
├── v018_fair_architecture_table.csv
├── v018_3exit_holdout_comparison.csv
├── v018_5exit_holdout_comparison.csv
├── v018_combined_ablation_table.csv
├── v018_policy_structure_comparison.csv
├── v018_selected_policy_summary.csv
├── v018_3exit_selected_policy.csv
├── v018_5exit_selected_policy.csv
├── v018_3exit_optimization_history.csv
├── v018_5exit_optimization_history.csv
├── v018_fair5_test_metrics_by_exit.csv
├── validation_to_holdout_transfer.csv
├── parent_change_summary.csv
├── per_label_holdout_delta_3exit.csv
├── per_label_holdout_delta_5exit.csv
├── v018_quality_compute_summary.svg
├── cross_version_3exit_flops_v018.svg
├── cross_version_3exit_quality_v018.svg
├── v018_fair_architecture_flops.svg
├── v018_exit_distribution.svg
└── v018_constraint_utilisation.svg
```

## Runtime outputs not committed wholesale

Large outputs remain under:

```text
human_talk_workspace/active_budget_anytime_exit_v0.4/
├── v0.16_EE/
├── v0.17_EE/
└── v0.18_EE/fair_sequential_anytime_exit/
    ├── fair_5exit_training/
    ├── 3exit/
    ├── 5exit/
    └── fair_architecture_comparison/
```

Full segment predictions, feature caches, checkpoints, parent matrices, and complete optimisation candidate tables are not duplicated into `docs/`.

## Confirmed-result convention

Every v0.18 record separates confirmed measurements, interpretation, future work, and non-claims. Interpretations are not presented as proven causal mechanisms.

## Documentation rules

1. Identify Always Exit 3 + frozen historical LATS-v2 as the canonical 3-exit quality reference.
2. Report v0.18 five-exit policies relative to the fair v0.18 Always Exit 5 baseline.
3. State that the v0.18 training audit passed and replaced v0.17's unfair architecture comparison.
4. Keep the canonical cross-version ranking restricted to the 3-exit family.
5. Maintain separate historical v0.17 and fair v0.18 five-exit tables.
6. Distinguish validation eligibility from holdout constraint satisfaction.
7. Report Parent Macro-F1 and Micro-F1 together with Samples-F1, Exact Match, and Hamming Loss.
8. Preserve per-label findings, especially transient and context labels.
9. Separate estimated FLOPs from measured latency.
10. Record hardware, threading, batch size, and timing repetitions.
11. Do not claim v0.18 is optimal or deployment-ready.
12. Do not retune on corrected holdout and call it untouched evaluation.
13. Do not describe risk scores as training-loss penalties.
14. Do not call v0.18 label-wise asynchronous inference.
15. Do not mix v0.17's non-fair five-exit success into the fair v0.18 conclusion.
16. Preserve unsuccessful findings and negative ablations.
17. State that Exit 2 and Exit 4 had zero holdout coverage in the selected v0.18 five-exit policy.
18. Describe plots as summaries of frozen operating points, not learning curves.

## Current documentation status

| Area | Status |
|---|---|
| Root README | Updated through v0.18 |
| Documentation index | Updated through v0.18 |
| Branch overview | Updated through v0.18 |
| v0.11–v0.18 traceability | Complete |
| v0.18 fair-training protocol | Complete |
| v0.18 theory and optimiser | Complete |
| v0.18 corrected-holdout results | Complete |
| v0.18 ablations and per-label records | Complete |
| v0.18 commands and reproduction entry point | Complete |
| Canonical 3-exit table | Updated through v0.18 |
| Historical v0.17 five-exit table | Preserved separately |
| Fair v0.18 architecture table | Complete |
| Machine-readable audits and result tables | Complete |
| Cross-version and architecture figures | Complete |
| Final deployable policy | Not established |
| Dedicated 5-exit No-Exit-1 optimisation | Future work |
