# Documentation Structure — Agentic Data Preprocessing v0.10

This structure records the updated documentation package after the v0.10 hint-pass, no-hint, LATS re-optimization, and seed-stability experiments.

---

## Top-level files

| File | Purpose |
|---|---|
| `README.md` | Current project-level v0.10 summary, final decision, results, and research interpretation |
| `DOC_STRUCTURE.md` | This documentation index |
| `docs/archive/README_v09_4_original.md` | Archived original v0.9_4 README provided before the v0.10 update |
| `docs/archive/DOC_STRUCTURE_v09_4_original.md` | Archived original v0.9_4 documentation structure |

---

## v0.10 documentation files

| File | Purpose |
|---|---|
| `docs/v0.10/README.md` | v0.10 branch report and final decision |
| `docs/v0.10/COMMANDS_V010_HINT_PASS_LATS.md` | Correct PowerShell commands for patching, training, evaluation, LATS re-optimization, and seed stability |
| `docs/v0.10/EXPERIMENT_SETUP.md` | Dataset, paths, model, hint-pass, and LATS experiment settings |
| `docs/v0.10/RESULTS_AND_ANALYSIS.md` | Main v0.9/v0.10 result comparison and interpretation |
| `docs/v0.10/SEED_STABILITY_ANALYSIS.md` | Seed 101/202/303 stability analysis |
| `docs/v0.10/RESEARCH_FINDINGS.md` | Research questions, scientific conclusions, and future work |
| `docs/v0.10/README_SAFE_HINT_ACTIVATION_PATCH.md` | Safe softmax/sigmoid hint activation patch notes |
| `docs/v0.10/APPLY_V010_DOC_UPDATES.md` | Suggested copy/apply instructions |

---

## v0.10 result tables

| File/folder | Purpose |
|---|---|
| `docs/tables/agentic_data_preprocessing_v0.10/v010_lats_reoptimized_comparison_summary.csv` | Main comparison table across v0.9 baseline, v0.10 no-hint, and v0.10 hint-pass |
| `docs/tables/agentic_data_preprocessing_v0.10/no_hint_lats_v2_coordinate_reoptimized_config.json` | Best single-run v0.10 no-hint LATS-v2 coordinate config |
| `docs/tables/agentic_data_preprocessing_v0.10/no_hint_lats_v2_coordinate_reoptimized_per_label.csv` | Per-label metrics for best single-run v0.10 no-hint LATS-v2 coordinate result |
| `docs/tables/agentic_data_preprocessing_v0.10/hint_pass_lats_v2_coordinate_reoptimized_config.json` | Best v0.10 hint-pass LATS-v2 coordinate config |
| `docs/tables/agentic_data_preprocessing_v0.10/hint_pass_lats_v2_coordinate_reoptimized_per_label.csv` | Per-label metrics for v0.10 hint-pass LATS-v2 coordinate result |
| `docs/tables/agentic_data_preprocessing_v0.10/v010_no_hint_seed_stability_summary.csv` | Compact seed 101/202/303 summary |
| `docs/tables/agentic_data_preprocessing_v0.10/v010_no_hint_seed_stability_stats.csv` | Cross-seed mean/std/min/max summary |
| `docs/tables/agentic_data_preprocessing_v0.10/seed_stability/` | Full per-seed LATS outputs and selected configurations |

---

## Final decision to report

| Area | Decision |
|---|---|
| Main stable result | Keep `v0.9_4 / LATS-v2` as the stable final baseline |
| v0.10 no-hint | Keep as diagnostic/stability ablation; promising but unstable |
| v0.10 hint-pass | Reject for current human-talk multi-label dataset |
| Strongest contribution | Metric-aware LATS-v2 inference policy, not hint-pass |
| Future work | Label-aware/gated hint passing rather than standard previous-exit probability hints |

---

## Main comparison snapshot

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Avg pred labels |
|---|---:|---:|---:|---:|---:|---:|
| v0.9_4 baseline — frozen LATS-v2 | 0.8673 | 0.9458 | 0.9517 | 0.8604 | 0.0158 | 1.4452 |
| v0.10 no-hint — frozen old LATS-v2 transfer | 0.8452 | 0.9247 | 0.9252 | 0.8062 | 0.0212 | 1.3495 |
| v0.10 no-hint — LATS-v1 re-optimized | 0.8658 | 0.9506 | 0.9562 | 0.8674 | 0.0145 | 1.4717 |
| v0.10 no-hint — LATS-v2 coordinate re-optimized | 0.8624 | 0.9531 | 0.9589 | 0.8766 | 0.0137 | 1.4591 |
| v0.10 hint-pass — frozen old LATS-v2 transfer | 0.8180 | 0.9155 | 0.9225 | 0.7878 | 0.0242 | 1.3956 |
| v0.10 hint-pass — LATS-v1 re-optimized | 0.8634 | 0.9447 | 0.9535 | 0.8639 | 0.0160 | 1.4291 |
| v0.10 hint-pass — LATS-v2 coordinate re-optimized | 0.8632 | 0.9440 | 0.9536 | 0.8570 | 0.0164 | 1.4556 |

---

## Seed-stability snapshot

| Seed | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Avg pred labels |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.8624 | 0.9353 | 0.9467 | 0.8374 | 0.0189 | 1.4556 |
| 202 | 0.8741 | 0.9471 | 0.9565 | 0.8674 | 0.0153 | 1.4314 |
| 303 | 0.8607 | 0.9492 | 0.9560 | 0.8731 | 0.0149 | 1.4614 |
| **Mean** | **0.8657** | **0.9439** | **0.9531** | **0.8593** | **0.0164** | **1.4494** |
| Std | 0.0073 | 0.0075 | 0.0055 | 0.0192 | 0.0022 | 0.0159 |
