# Active Budget and Anytime Exit v0.4 — Result Tables

This directory stores compact, version-controlled records for the v0.4 multi-objective and sequential Early-Exit branch.

## Packages

| Folder | Status | Purpose |
|---|---|---|
| `v0.16_EE/` | Complete | Multi-objective per-label Exit-2 margin search, Pareto records, genuine staged holdout evaluation, and cumulative ablations |
| `v0.17_EE/` | Complete | Fully sequential 3-exit/5-exit optimisation, frozen policies, holdout comparisons, six ablations, per-label analysis, timing, and fairness audit |

Historical v0.11 records remain under `docs/tables/active_budget_anytime_exit_v0.2/`. Runtime-heavy v0.12–v0.15 outputs remain under `human_talk_workspace`, while confirmed headline results are traced in the v0.4 version history and cumulative tables.

## v0.17 headline

| Architecture | Outcome |
|---|---|
| 3-exit | 8.64% estimated FLOPs saved and 1.037× speedup, but holdout quality limits failed |
| 5-exit | 30.71% estimated FLOPs saved and 1.114× speedup, with all within-checkpoint holdout limits met |
| Fair architecture comparison | Not valid because the training/validation manifests differ |

## Storage policy

Committed:

- compact validation and holdout summaries;
- selected policies and optimisation histories;
- Pareto frontiers;
- explicit constraint checks;
- per-label and parent-change comparisons;
- cross-architecture fairness audit;
- figures, commands, theory, and paper-ready wording.

Not committed wholesale:

- all 5,847/5,856 evaluated candidates;
- full segment probability/prediction matrices;
- full parent score and prediction matrices;
- checkpoints and feature caches.

These remain reproducible from workspace outputs and branch scripts.
