# Documentation Structure — Active Budget and Anytime Exit v0.4

This file indexes the computation-adaptive inference documentation for `active_budget_anytime_exit_v0.4`, including the completed v0.16 multi-objective policy and v0.17 fully sequential three-exit/five-exit study.

## Active branch

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.4` |
| Source branch | `active_budget_anytime_exit_v0.3` |
| Completed milestone | `v0.17_EE` sequential active-budget anytime exit |
| Canonical three-exit comparator | v0.10 no-hint + frozen historical LATS-v2 + Always Exit 3 |
| Current fair three-exit adaptive baseline | v0.13 per-label margin |
| v0.16 status | Fully integrated; speedup achieved; holdout quality constraints failed |
| v0.17 three-exit status | Genuine speedup achieved; holdout quality constraints failed |
| v0.17 five-exit status | Within-model success; all holdout quality constraints met |
| Cross-architecture fairness | Not established because the three-exit and five-exit training manifests differ |

Historical records remain authoritative for their own versions and must not be silently rewritten.

## Top-level documentation

| Path | Purpose |
|---|---|
| `README.md` | Authoritative branch summary, theory, cumulative results, commands, findings, cautions, and scientific verdict |
| `DOC_STRUCTURE.md` | This documentation and artifact index |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Detailed v0.4 branch overview through v0.17 |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | Version-to-implementation, settings, RQs, results, and findings from v0.11 through v0.17 |
| `docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md` | Documentation audit, provenance rules, and file-by-file update summary |
| `docs/tables/active_budget_anytime_exit_v0.4/README.md` | Compact-results package index |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/` | Complete v0.16 multi-objective record |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/` | Complete v0.17 sequential record |
| `docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/` | Frozen full-depth baseline reproducibility package |
| `docs/active_budget_anytime_exit_v0.2/` | Historical v0.11 branch documentation |
| `docs/tables/active_budget_anytime_exit_v0.2/` | Historical v0.11 compact records |
| `docs/v0.10/` and `docs/v0.10_1/` | Earlier model, hint-pass, calibration, and low-energy-recovery documentation |
| `docs/archive/` | Archived documentation states |

## Experiment-to-code traceability

| Version | Primary implementation paths |
|---|---|
| v0.11 | `models/anytime_exit_net.py`; `scripts/v0.11_EE/fixed_policy/`; `scripts/v0.11_EE/dynamic_policy/`; `tests/test_anytime_exit_net.py` |
| v0.12 | `policies/label_aware_early_exit_policy.py`; `scripts/v0.12_EE/label_aware_policy/`; `tests/test_label_aware_early_exit_policy.py` |
| v0.13 | `policies/early_exit_strategy_comparison.py`; `scripts/v0.13_EE/matched_policy_comparison/`; `tests/test_early_exit_strategy_comparison.py` |
| v0.14 | `policies/parent_aware_adaptive_gate.py`; `scripts/v0.14_EE/parent_aware_gate/`; `tests/test_parent_aware_adaptive_gate.py` |
| v0.15 | `policies/whole_parent_selective_exit.py`; `scripts/v0.15_EE/whole_parent_risk_control/`; `tests/test_whole_parent_selective_exit.py` |
| v0.16 | `policies/multiobjective_per_label_margin.py`; `scripts/v0.16_EE/multiobjective_per_label_margin/`; `tests/test_multiobjective_per_label_margin.py` |
| v0.17 | `policies/sequential_anytime_exit.py`; `policies/sequential_anytime_exit_optim.py`; `scripts/v0.17_EE/sequential_anytime_exit/`; `tests/test_sequential_anytime_exit.py` |

## v0.17 committed package

```text
docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/
├── README.md
├── EXPERIMENT_SETUP.md
├── RESULTS_AND_ANALYSIS.md
├── PAPER_READY_SUMMARY.md
├── PS_COMMANDS.md
├── experiment_manifest.json
├── headline_results.csv
├── ablation_summary.csv
├── cumulative_version_comparison.csv
├── per_label_findings.csv
├── exit_distribution.svg
└── quality_compute_comparison.svg
```

### Human-readable records

| File | Purpose |
|---|---|
| `README.md` | Headline status, results, decisions, and package index |
| `EXPERIMENT_SETUP.md` | Architecture, sequential logic, optimiser, data, settings, fairness protocol, and RQs |
| `RESULTS_AND_ANALYSIS.md` | Validation, holdout, timing, ablations, per-label findings, interpretation, and limitations |
| `PAPER_READY_SUMMARY.md` | Reusable paper/thesis wording, tables, captions, successful and unsuccessful findings |
| `PS_COMMANDS.md` | PowerShell commands for three-exit, combined, timing, tuning, evaluation, and reporting |

### Machine-readable and visual records

| File | Purpose |
|---|---|
| `experiment_manifest.json` | Frozen branch, checkpoints, settings, constraints, results, and fairness status |
| `headline_results.csv` | Three-exit and five-exit baseline/full-sequential headline comparison |
| `ablation_summary.csv` | No Exit 1, no stability, no risk, no margins, and confidence-only ablations |
| `cumulative_version_comparison.csv` | v0.11–v0.17 quality–computation history |
| `per_label_findings.csv` | Confirmed direction and approximate magnitude of key per-label changes |
| `exit_distribution.svg` | Exit-routing distribution for full sequential policies |
| `quality_compute_comparison.svg` | FLOP saving against Parent Macro-F1 drop |

## Runtime outputs not committed wholesale

Large predictions remain under:

```text
human_talk_workspace/active_budget_anytime_exit_v0.4/v0.17_EE/sequential_anytime_exit/
```

This includes full segment predictions, parent probability tables, all evaluated candidates, frozen policies, checkpoint-equivalence reports, timing summaries, and complete ablation outputs. The committed package is a compact interpretation and traceability layer; workspace artifacts remain the source for exact regeneration.

## Confirmed-result and interpretation convention

Every v0.17 document separates:

- **Confirmed measurements:** values directly produced by frozen validation and corrected-holdout runs.
- **Interpretation:** explanations for route behaviour, validation-to-holdout transfer, per-label risks, or ablation effects.
- **Future work:** fair five-exit retraining, stronger Exit-1 safeguards, revised risk modelling, independent calibration/evaluation, and label-wise asynchronous exit.

Future-work proposals must not be described as completed v0.17 contributions.

## Documentation rules

1. Identify the three-exit canonical comparator as v0.10 no-hint + frozen historical LATS-v2 + Always Exit 3.
2. Compare the historical five-exit sequential policy only with its own Always Exit 5 baseline unless training fairness is established.
3. Do not claim five-exit architectural superiority while training manifests differ.
4. Distinguish validation eligibility from corrected-holdout quality-limit satisfaction.
5. Do not retune after inspecting the corrected holdout and then claim an untouched test.
6. Separate architecture-estimated FLOP saving from measured latency.
7. Report Parent Macro-F1 and Parent Micro-F1 together.
8. Preserve Exact Match and Hamming Loss because aggregate F1 can hide multi-label inconsistency.
9. Report per-label findings, especially `audience_reaction_present`, `other_speaker_present`, `Nick_Vujicic`, and `Eric_Thomas`.
10. State whether the CNN was retrained, frozen, or only wrapped by a controller.
11. Label gate/controller fitting separately from backbone training.
12. Do not omit Exit 1 from the primary sequential method; `No Exit 1` is an ablation.
13. Do not claim the current risk component is effective when `No Risk` is identical or nearly identical.
14. Do not describe confidence-only or no-margin policies as acceptable merely because they save more compute.
15. Do not describe v0.17 as label-wise asynchronous inference; one exit depth is selected per sample.
16. Keep successful, unsuccessful, and inconclusive findings explicit.

## Current documentation status

| Area | Status |
|---|---|
| Root branch README | Updated through v0.17 |
| Root documentation index | Updated through v0.17 |
| v0.4 branch overview | Updated through v0.17 |
| v0.11–v0.17 traceability | Complete |
| v0.17 theory and settings | Complete |
| v0.17 validation and holdout records | Complete |
| v0.17 30-repeat timing | Complete |
| v0.17 ablation analysis | Complete |
| v0.17 per-label interpretation | Complete |
| v0.17 paper-ready wording | Complete |
| Fair three-exit/five-exit architecture comparison | Pending canonical five-exit retraining |
| Budget-conditioned quality curve | Future work |
| Label-wise asynchronous exit | Future work |
