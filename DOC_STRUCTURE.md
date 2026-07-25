# Documentation Structure — Active Budget and Anytime Exit v0.4

This file indexes the complete computation-adaptive inference documentation inherited by `active_budget_anytime_exit_v0.4`, including the completed v0.16 and v0.17 result packages.

---

## Active branch

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.4` |
| Source branch | `active_budget_anytime_exit_v0.3` |
| Current milestone | `v0.17_EE` sequential active-budget anytime exit |
| Previous milestone | `v0.16_EE` multi-objective per-label margin optimisation |
| Canonical comparator | v0.10 no-hint + frozen historical LATS-v2 + Always Exit 3 |
| 5-exit comparator | Tested v0.6 expanded checkpoint + Always Exit 5 |
| v0.17 3-exit status | Fully integrated; real speedup; holdout quality limits failed |
| v0.17 5-exit status | Fully integrated; all within-checkpoint holdout limits met |
| Cross-architecture status | Not fair because validation/training manifests differ |

Historical records remain authoritative for their own versions and must not be silently rewritten.

---

## Top-level documentation

| Path | Purpose |
|---|---|
| `README.md` | Authoritative v0.4 summary, theory, cumulative results, cross-version tables/figures, commands, decisions, and non-claims |
| `DOC_STRUCTURE.md` | This documentation and artifact index |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Detailed branch-level v0.16/v0.17 overview |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | v0.11–v0.17 implementation, settings, research questions, results, and findings |
| `docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md` | Result provenance and file-by-file update record |
| `docs/tables/active_budget_anytime_exit_v0.4/README.md` | Compact package index |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/` | Complete v0.16 experiment package |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/` | Complete v0.17 sequential experiment package, cross-version tables, and figures |
| `docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/` | Frozen canonical full-depth baseline package |
| `docs/active_budget_anytime_exit_v0.2/` | Historical v0.11 branch documentation |
| `docs/tables/active_budget_anytime_exit_v0.2/` | Historical v0.11 compact records |
| `docs/v0.10/` and `docs/v0.10_1/` | Earlier model, hint-pass, calibration, and low-energy records |
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
| v0.17 | `policies/sequential_anytime_exit.py`; `scripts/v0.17_EE/sequential_anytime_exit/`; `tests/test_sequential_anytime_exit.py` |

---

## v0.17 package

```text
docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/
├── README.md
├── EXPERIMENT_SETUP.md
├── THEORY_AND_METHOD.md
├── RESULTS_AND_ANALYSIS.md
├── ABLATIONS_AND_FINDINGS.md
├── PAPER_READY_SUMMARY.md
├── PS_COMMANDS.md
├── REPRODUCE_V017_EE.ps1
├── FIGURES.md
├── experiment_manifest.json
├── holdout_constraint_check.csv
├── cross_architecture_headline.csv
├── cross_version_3exit_table.csv
├── v017_architecture_table.csv
├── ablation_3exit.csv
├── ablation_5exit.csv
├── per_label_holdout_delta_3exit.csv
├── per_label_holdout_delta_5exit.csv
├── cross_version_3exit_flops.svg
├── cross_version_3exit_quality.svg
├── cross_version_3exit_hamming.svg
├── v017_architecture_flops.svg
└── fairness_audit.json
```

The human-readable records separate method, setup, results, interpretation, commands, paper-safe wording, and non-claims. Compact machine-readable records preserve the headline comparison, fair 3-exit cross-version table, architecture-specific v0.17 comparison, quality audit, ablations, per-label changes, and fairness decision.

The cross-version line plots connect discrete frozen-policy operating points. They are descriptive summaries, not continuous training or optimiser trajectories.

---

## Runtime outputs not committed wholesale

Large v0.17 predictions remain under:

```text
human_talk_workspace/active_budget_anytime_exit_v0.4/v0.17_EE/sequential_anytime_exit/
├── 3exit/
├── 5exit/
└── architecture_comparison/
```

Full segment predictions, parent score matrices, complete candidate tables, checkpoints, and features are not duplicated into `docs/`.

---

## Confirmed-result and interpretation convention

Every v0.17 document separates:

- **Confirmed measurements:** frozen validation, corrected-holdout, ablation, and 30-repeat timing outputs.
- **Interpretation:** explanations for why the 5-exit policy transfers better, why Exit 1 is risky, and which labels remain difficult.
- **Future work:** fair 5-exit retraining, stronger Exit-1 calibration, risk redesign, explicit budget curves, and label-wise asynchronous exit.

Interpretations must not be presented as confirmed causal mechanisms.

---

## Documentation rules

1. Always identify the canonical 3-exit comparator as v0.10 no-hint + frozen historical LATS-v2 + Always Exit 3.
2. Report the tested 5-exit result relative to Always Exit 5.
3. Do not compare 3-exit and 5-exit absolute quality as a fair architecture experiment while manifests differ.
4. Keep the canonical cross-version table restricted to the comparable 3-exit family.
5. Report the tested 5-exit result in a separate within-checkpoint companion table.
6. Do not confuse average predicted labels with average exit depth.
7. Distinguish validation eligibility from holdout constraint satisfaction.
8. Do not retune after viewing corrected-holdout results and then claim untouched evaluation.
9. Separate architecture-estimated FLOP saving from measured latency.
10. Report Parent Macro-F1 and Parent Micro-F1 together.
11. Preserve Samples-F1, Exact Match, and Hamming Loss.
12. Preserve per-label results, especially `audience_reaction_present`, `Nick_Vujicic`, `Eric_Thomas`, `other_speaker_present`, and `silence_present`.
13. State that the backbone and exit heads are frozen.
14. Label controller optimisation separately from model training.
15. Record that v0.17 has no backbone-training command.
16. Do not claim that the current risk term improved performance; the ablation was non-binding.
17. Do not call v0.17 label-wise asynchronous exit.
18. Do not include evidence accumulation or knowledge distillation as completed v0.17 components.
19. Record CPU, thread, batch, and repetition settings with latency.
20. Keep confirmed results, interpretation, future work, and non-claims separate.
21. Describe cross-version line plots as connected discrete operating points, not learning curves.

---

## Current documentation status

| Area | Status |
|---|---|
| Root branch README | Updated through v0.17, including cross-version tables and figures |
| Root documentation index | Updated through v0.17 |
| v0.4 branch overview | Updated through v0.17 |
| v0.11–v0.17 traceability | Complete |
| Canonical 3-exit cross-version comparison | Complete through v0.17 |
| Separate v0.17 5-exit companion comparison | Complete |
| Cross-version SVG figures | Complete |
| v0.17 theory and setup | Complete |
| v0.17 validation/Pareto records | Complete |
| v0.17 holdout and timing records | Complete |
| v0.17 ablations | Complete |
| v0.17 per-label analysis | Complete |
| v0.17 fairness audit | Complete |
| v0.17 paper-ready wording | Complete |
| Fair same-manifest 5-exit checkpoint | Not yet trained |
| Runtime budget–quality curve | Not yet implemented |
| Label-wise asynchronous exit | Future work |
