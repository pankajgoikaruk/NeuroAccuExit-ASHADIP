# Documentation Structure — agentic_data_preprocessing_v0.8

This branch package documents the v0.8-human-corrected-balanced ASHADIP experiment.

## Root files

| Path | Purpose |
|---|---|
| `README.md` | Main branch overview, headline results, and documentation map. |
| `DOC_STRUCTURE.md` | This file; explains the documentation layout. |

## Core docs

| Path | Purpose |
|---|---|
| `docs/APPENDIX.md` | Thesis-style appendix with methods, settings, results, and conclusion. |
| `docs/MULTILABEL_EXPERIMENT_LOG.md` | Chronological log of the v0.8 experiment. |
| `docs/COMMANDS_V08.md` | Full PowerShell command log with purpose and key output. |

## Reports and results

| Path | Purpose |
|---|---|
| `docs/reports/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md` | Detailed thesis-ready experiment report. |
| `docs/results/V08_RESULTS_SUMMARY.md` | Compact official result and comparison tables. |
| `docs/releases/AGENTIC_DATA_PREPROCESSING_V08_RELEASE_NOTES.md` | Release notes for the v0.8 documentation update. |

## Figures

| Figure | Purpose |
|---|---|
| `docs/figures/v08_training_validation_curve.png` | Validation Macro-F1, Micro-F1, and Exact Match over epochs. |
| `docs/figures/v08_training_loss_hamming_curve.png` | Training loss and validation Hamming Loss over epochs. |
| `docs/figures/v08_label_counts_before_after_balance.png` | Label counts before/after balancing. |
| `docs/figures/v08_internal_test_by_exit_lineplot.png` | Internal test metrics by exit. |
| `docs/figures/v08_corrected_holdout_fixed_by_exit_lineplot.png` | Corrected holdout metrics by exit. |
| `docs/figures/v08_vs_v06_corrected_holdout_bar.png` | Final fair comparison against v0.6. |
| `docs/figures/v08_vs_v06_hamming_loss_bar.png` | Hamming Loss comparison. |
| `docs/figures/v08_corrected_holdout_per_label_f1_bar.png` | Per-label final-exit F1 on corrected holdout. |
| `docs/figures/v08_avg_true_vs_pred_labels_bar.png` | Average true vs predicted labels per parent clip. |

## Tables

CSV files under `docs/tables/` are the source tables used in the Markdown docs. They can be imported into Excel, LaTeX, or thesis plotting scripts.

## File inventory

```text
DOC_STRUCTURE.md
README.md
docs/APPENDIX.md
docs/COMMANDS_V08.md
docs/MULTILABEL_EXPERIMENT_LOG.md
docs/figures/v08_avg_true_vs_pred_labels_bar.png
docs/figures/v08_corrected_holdout_fixed_by_exit_lineplot.png
docs/figures/v08_corrected_holdout_per_label_f1_bar.png
docs/figures/v08_internal_test_by_exit_lineplot.png
docs/figures/v08_label_counts_before_after_balance.png
docs/figures/v08_training_loss_hamming_curve.png
docs/figures/v08_training_validation_curve.png
docs/figures/v08_vs_v06_corrected_holdout_bar.png
docs/figures/v08_vs_v06_hamming_loss_bar.png
docs/releases/AGENTIC_DATA_PREPROCESSING_V08_RELEASE_NOTES.md
docs/reports/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md
docs/results/V08_RESULTS_SUMMARY.md
docs/tables/historical_v06_v07_v08_summary.csv
docs/tables/v08_corrected_holdout_parent_mean_fixed_by_exit.csv
docs/tables/v08_corrected_holdout_parent_mean_fixed_per_label_exit3.csv
docs/tables/v08_corrected_holdout_parent_mean_tuned_by_exit.csv
docs/tables/v08_experiment_commands_index.csv
docs/tables/v08_fair_comparison_corrected_holdout_parent_mean_fixed.csv
docs/tables/v08_final_exit_tuned_thresholds.csv
docs/tables/v08_internal_test_by_exit.csv
docs/tables/v08_internal_test_per_label_exit3.csv
docs/tables/v08_label_counts_before_after_balance.csv
docs/tables/v08_manifest_summary.csv
docs/tables/v08_threshold_tuning_internal_val_test.csv
docs/tables/v08_training_group_counts.csv
docs/tables/v08_training_validation_curve.csv
```
