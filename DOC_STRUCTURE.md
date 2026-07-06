# Documentation Structure — Agentic Data Preprocessing v0.10 / v0.10_1

This index covers the v0.10 human-talk multi-label experiments and the v0.10_1 low-energy recovery ablation.

## Top-level files

| File | Purpose |
|---|---|
| `README.md` | Project-level summary, final decision, method comparison, research questions, and outcomes |
| `DOC_STRUCTURE.md` | Documentation index |

## v0.10 documentation files

| File | Purpose |
|---|---|
| `docs/v0.10/README.md` | v0.10 branch report and final decision |
| `docs/v0.10/EXPERIMENT_SETUP.md` | Dataset, paths, model, LATS, hint-pass, and pos_weight settings |
| `docs/v0.10/RESULTS_AND_ANALYSIS.md` | Full method comparison and interpretation |
| `docs/v0.10/SEED_STABILITY_ANALYSIS.md` | v0.10 no-hint seed stability analysis |
| `docs/v0.10/POS_WEIGHT_EXPERIMENT_ANALYSIS.md` | pos_weight cap5 diagnostic analysis |
| `docs/v0.10/RESEARCH_FINDINGS.md` | Research questions, findings, conclusions, and future work |
| `docs/v0.10/PS_COMMANDS.md` | PowerShell commands for v0.10 experiments |

## v0.10_1 documentation files

| File | Purpose |
|---|---|
| `docs/v0.10_1/LOW_ENERGY_RECOVERY_ABLATION_PLAN.md` | Non-destructive plan for the TATA-LAWYER low-energy recovery ablation |
| `docs/v0.10_1/LOW_ENERGY_RECOVERY_ABLATION_RESULTS.md` | Final results, comparison table, research finding, and decision for v0.10_1 |

## Main comparison snapshot

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Decision |
|---|---:|---:|---:|---:|---:|---|
| v0.9_4 frozen LATS-v2 | 0.8673 | 0.9458 | 0.9517 | 0.8604 | 0.0158 | Stable reference baseline |
| v0.10 no-hint LATS-v2 re-optimized | 0.8624 | 0.9531 | 0.9589 | 0.8766 | 0.0137 | Selected current final outcome |
| v0.10 no-hint + pos_weight cap5 | 0.8511 | 0.9413 | 0.9481 | 0.8478 | 0.0171 | Rejected |
| v0.10_1 low-energy recovery | 0.8581 | 0.9446 | 0.9519 | 0.8570 | 0.0160 | Rejected as final; diagnostic ablation |

## Final decision

| Area | Decision |
|---|---|
| Selected final result | `v0.10 no-hint + LATS-v2 coordinate re-optimized` |
| Stable reference baseline | `v0.9_4 / LATS-v2` |
| Hint-pass | Reject current standard hint-pass |
| pos_weight cap5 | Reject |
| v0.10_1 low-energy recovery | Reject as final, document as valid negative/diagnostic ablation |
| Strongest contribution | LATS-v2 metric-aware inference optimization |
