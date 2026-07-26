# Documentation Update Summary — v0.18_EE

This update completes the documentation for `active_budget_anytime_exit_v0.4` through `v0.18_EE`.

## Files updated

| File | Added or changed |
|---|---|
| `README.md` | Replaced v0.17-as-current status with v0.18; added fair-training protocol, strict policy theory, optimiser settings, corrected-holdout results, constraint audit, updated cross-version table, policy-structure table, ablations, per-label findings, commands, limitations, final verdict, and unsuccessful finding. |
| `DOC_STRUCTURE.md` | Added v0.18 code traceability, documentation package tree, runtime-output locations, documentation rules, and completion status. |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Added concise branch-level v0.18 overview, research questions, fairness settings, headline results, findings, and execution command. |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | Extended the complete experiment history from v0.11 through v0.18 with implementation, settings, research questions, results, and interpretation. |
| `docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md` | Added this provenance and file-by-file update record. |

## Files created under the v0.18 package

| File | Purpose |
|---|---|
| `README.md` | Package index and concise experiment verdict |
| `EXPERIMENT_SETUP.md` | Data, architecture, training, fairness, optimiser, holdout, and timing settings |
| `THEORY_AND_METHOD.md` | Sequential stopping equations, risk-veto design, objectives, and constraints |
| `RESULTS_AND_ANALYSIS.md` | Confirmed training, holdout, transfer, timing, architecture, and per-label analysis |
| `ABLATIONS_AND_FINDINGS.md` | Exit-1, risk, stability, margin, and confidence-only ablations |
| `PAPER_READY_SUMMARY.md` | Safe academic wording, final scientific verdict, unsuccessful finding, and non-claims |
| `PS_COMMANDS.md` | Training, fairness audit, tuning, evaluation, comparison, and full runner commands |
| `cross_version_3exit_table.csv` | Canonical three-exit cross-version comparison through v0.18 |
| `fair_architecture_headline.csv` | Fair v0.18 three-exit/five-exit headline comparison |
| `v018_ablation_summary.csv` | Compact v0.18 ablation summary |
| `v018_quality_compute_summary.svg` | Descriptive quality–compute summary of frozen v0.18 operating points |

## Result provenance

The v0.18 records are based on the completed fair five-exit training run, passed fair-training audit, passed staged/full equivalence tests, frozen validation-selected policies, corrected-holdout evaluation of 4,335 segments and 867 parents, 30-repeat CPU timing, and ablation outputs.

## Confirmed-versus-interpretation convention

- Numerical tables are confirmed measurements.
- Explanations of LATS calibration, distribution shift, and transient-label behaviour are interpretations.
- Dedicated No-Exit-1 optimisation and stronger transient-label risk modelling are future work.

## Important documentation decisions

1. The canonical cross-version table remains three-exit only.
2. Fair five-exit results are reported in a separate v0.18 architecture table.
3. The non-fair v0.17 five-exit result remains documented historically but is not used as the principal fair conclusion.
4. Both successful and unsuccessful v0.18 findings are retained.
5. Validation eligibility and holdout compliance are kept separate.
