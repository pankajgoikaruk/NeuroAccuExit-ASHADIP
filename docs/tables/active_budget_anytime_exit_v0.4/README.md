# Active Budget and Anytime Exit v0.4 — Result Tables

This directory stores compact, version-controlled records for the v0.4 multi-objective and sequential Early-Exit branch.

## Packages

| Folder | Status | Purpose |
|---|---|---|
| `v0.16_EE/` | Complete | Multi-objective per-label margin search, Pareto records, genuine staged holdout evaluation, and cumulative ablations |
| `v0.17_EE/` | Complete | Fully sequential 3-exit/5-exit policies, timing, ablations, fairness audit summary, paper-ready wording, compact CSVs, and SVG figures |

Historical v0.11 records remain under `docs/tables/active_budget_anytime_exit_v0.2/`. Runtime-heavy v0.12–v0.15 outputs remain under `human_talk_workspace`, while confirmed headline results are traced in the v0.4 version history and cumulative tables.

## v0.17 headline

- three-exit full sequential: 8.64% estimated FLOPs saved and 1.037× speedup, but quality limits failed;
- five-exit full sequential: 30.71% estimated FLOPs saved and 1.114× speedup, with all within-model quality limits met;
- direct architecture comparison: not fair because training manifests differ.

## Storage policy

Committed:

- compact validation and holdout summaries;
- selected settings and experiment manifest;
- per-label, ablation, and cumulative comparisons;
- figures, commands, and paper-ready wording.

Not committed wholesale:

- all evaluated optimiser candidates;
- full segment probability/prediction matrices;
- parent score and prediction matrices;
- checkpoints and feature caches.

These remain reproducible from workspace outputs and branch scripts.
