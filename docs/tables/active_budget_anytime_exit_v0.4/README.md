# Active Budget and Anytime Exit v0.4 — Result Tables

This directory stores compact, version-controlled records for the v0.4 multi-objective Early-Exit branch.

## Packages

| Folder | Status | Purpose |
|---|---|---|
| `v0.16_EE/` | Complete | Multi-objective per-label margin search, Pareto records, genuine staged holdout evaluation, and cumulative ablations |

Historical v0.11 records remain under `docs/tables/active_budget_anytime_exit_v0.2/`. Runtime-heavy v0.12–v0.15 outputs remain under `human_talk_workspace`, while their confirmed headline results are traced in the v0.4 version history and cumulative table.

## Storage policy

Committed:

- compact validation and holdout summaries;
- selected policy and optimisation history;
- Pareto frontier;
- per-label and cumulative comparisons;
- figures, commands, and paper-ready wording.

Not committed wholesale:

- all 4,078 evaluated candidates;
- full segment probability/prediction matrices;
- parent score and prediction matrices;
- checkpoints and features.

These remain reproducible from the workspace outputs and branch scripts.
