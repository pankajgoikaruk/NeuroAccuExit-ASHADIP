# Documentation Structure — Agentic Data Preprocessing v0.9_4

This structure documents the current v0.9_4 documentation update for LATS-v2.

## Main updated files

| File | Purpose |
|---|---|
| `README.md` | Main project-level v0.9_4 summary and final decision |
| `DOC_STRUCTURE.md` | Index of updated documentation and result files |
| `docs/INDEX.md` | High-level documentation index |
| `docs/RESULTS.md` | Consolidated result pointer and v0.9_4 result summary |
| `docs/MULTILABEL_EXPERIMENT_LOG.md` | Chronological log including the LATS-v2 experiment |
| `docs/APPENDIX.md` | Appendix update with LATS-v2 notes |
| `docs/v0.9/v0.9_4/COMMANDS_V09.md` | Commands for branch creation, LATS-v2 execution, result inspection, and commit |
| `docs/v0.9/v0.9_4/EXPERIMENT_SETUP.md` | Full experiment setup for LATS-v2 |
| `docs/v0.9/v0.9_4/AGENTIC_V09_AGGREGATION_CALIBRATION_GUIDE.md` | Method and interpretation guide |
| `docs/v0.9/v0.9_4/MULTILABEL_EXPERIMENT_LOG.md` | v0.9_4-specific experiment log |
| `docs/v0.9/v0.9_4/APPENDIX.md` | v0.9_4 appendix tables and notes |
| `docs/v0.9/v0.9_4/APPLY_V09_DOC_UPDATES.md` | Copy/apply instructions |
| `docs/results/v0.9/V09_4_RESULTS_SUMMARY.md` | Final result summary |
| `docs/reports/v0.9/V09_4_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md` | HCB experiment report update |
| `docs/reports/v0.9/V09_4_LABELWISE_AGGREGATION_CALIBRATION_REPORT.md` | LATS-v2 method report |

## Main updated result tables

| File | Purpose |
|---|---|
| `docs/tables/agentic_data_preprocessing_v0.9/lats_v2_final_full_holdout_eval.csv` | Raw LATS-v2 final full-holdout metrics |
| `docs/tables/agentic_data_preprocessing_v0.9/lats_v2_final_full_holdout_per_label.csv` | Raw LATS-v2 per-label metrics |
| `docs/tables/agentic_data_preprocessing_v0.9/lats_v2_final_vs_init_comparison.csv` | LATS-v2 vs initial LATS-v1 config |
| `docs/tables/agentic_data_preprocessing_v0.9/lats_v2_repeated_eval_summary.csv` | Repeated split metrics |
| `docs/tables/agentic_data_preprocessing_v0.9/lats_v2_threshold_summary.csv` | Final LATS-v2 threshold/stability summary |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_lats_v2_global_result_comparison.md` | Markdown global comparison table |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_lats_v2_pair_changes.md` | LATS-v1 to LATS-v2 pair-change table |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_lats_v1_vs_v2_stability.md` | Combined stability table |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_lats_v2_research_findings_latex.tex` | LaTeX-ready research findings section |

## Final decision

LATS-v2 is the new recommended v0.9_4 result because it improves all major full-holdout global metrics compared with LATS-v1.

| Metric                |   LATS-v1 |   LATS-v2 |   Difference |
|:----------------------|----------:|----------:|-------------:|
| Macro-F1              |    0.8667 |    0.8673 |       0.0005 |
| Micro-F1              |    0.9436 |    0.9458 |       0.0022 |
| Samples-F1            |    0.9495 |    0.9517 |       0.0022 |
| Exact Match           |    0.8524 |    0.8604 |       0.0081 |
| Hamming Loss ↓        |    0.0165 |    0.0158 |      -0.0007 |
| Jaccard               |    0.9274 |    0.9309 |       0.0035 |
| Avg true labels       |    1.4694 |    1.4694 |       0      |
| Avg pred labels       |    1.4544 |    1.4452 |      -0.0092 |
| Label-count abs error |    0.015  |    0.0242 |       0.0092 |

---

## Previous structure notes

The previous v0.9_3 documentation is preserved under `docs/v0.9/v0.9_3/`. The new v0.9_4 documentation is added separately so the experiment history remains traceable.
