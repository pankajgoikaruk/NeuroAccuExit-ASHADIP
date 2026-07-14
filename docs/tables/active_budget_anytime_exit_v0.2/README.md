# Active Budget and Anytime Exit v0.2 — Result Tables

This directory stores compact, version-controlled experiment records for the computation-adaptive NeuroAccuExit branch.

## Packages

| Folder | Status | Purpose |
|---|---|---|
| `v0.11_EE/` | Complete | Staged equivalence, fixed exits, validation-frozen standard Dynamic Early-Exit |
| `v0.12_budget_aware_EE/` | Planned | Explicit per-sample or dataset-level budget controller |
| `v0.13_anytime_inference/` | Planned | Quality-versus-cost evaluation across normalized budgets |

## Baseline dependency

The canonical full-depth package is intentionally not duplicated. It remains at:

```text
docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/
```

All v0.2 tables reference that frozen result.

## Storage policy

Committed here:

- compact summaries;
- per-label tables;
- policy settings;
- equivalence reports;
- paper-ready wording;
- reproducibility commands.

Not committed here:

- full segment probability matrices;
- per-segment dynamic predictions;
- large parent score/prediction tables;
- feature files;
- model checkpoints.

Those artifacts remain reproducible under `human_talk_workspace`.
