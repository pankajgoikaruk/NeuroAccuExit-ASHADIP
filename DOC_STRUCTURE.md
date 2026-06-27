---

## v0.8 human-talk documentation layout update

The v0.8 documentation artifacts are organised under human-talk and version-specific folders so that v0.7 and v0.8 results do not overwrite each other.

### Reports

```text
docs/reports/human_talk/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md
```

Purpose: thesis-ready narrative report for the v0.8-human-corrected-balanced experiment, including methodology, training configuration, corrected-holdout results, fair v0.6 comparison, and label-aware aggregation analysis.

### Results summaries

```text
docs/results/human_talk/V08_RESULTS_SUMMARY.md
```

Purpose: compact results summary for GitHub readers, including headline metrics, corrected-holdout comparison, and final reporting decision.

### Tables

```text
docs/tables/agentic_data_preprocessing_v0.8/
```

Recommended v0.8 table files:

```text
v08_fair_comparison_corrected_holdout_parent_mean_fixed.csv
v08_corrected_holdout_parent_mean_fixed_by_exit.csv
v08_corrected_holdout_parent_mean_tuned_by_exit.csv
v08_corrected_holdout_parent_mean_fixed_per_label_exit3.csv
v08_internal_test_by_exit.csv
v08_internal_test_per_label_exit3.csv
v08_label_counts_before_after_balance.csv
v08_threshold_tuning_internal_val_test.csv
v08_final_exit_tuned_thresholds.csv
v08_experiment_commands_index.csv
v08_hcb_parent_aggregation_strategy_comparison.csv
v08_hcb_label_aware_fair_comparison_corrected_holdout.csv
v08_hcb_weak_label_f1_by_aggregation.csv
v08_hcb_per_label_mean_max_labelaware_exit3.csv
v08_hcb_label_aware_commands.csv
```

v0.7-related tables should stay separate:

```text
docs/tables/agentic_data_preprocessing_v0.7/
```

### Figures

All v0.8 human-talk figures should be under:

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.8/
```

Recommended v0.8 figure files:

```text
v08_training_validation_curve.png
v08_training_loss_hamming_curve.png
v08_label_counts_before_after_balance.png
v08_internal_test_by_exit_lineplot.png
v08_corrected_holdout_fixed_by_exit_lineplot.png
v08_vs_v06_corrected_holdout_bar.png
v08_vs_v06_hamming_loss_bar.png
v08_corrected_holdout_per_label_f1_bar.png
v08_avg_true_vs_pred_labels_bar.png
v08_hcb_aggregation_strategy_lineplot.png
v08_hcb_aggregation_hamming_loss_lineplot.png
v08_hcb_weak_label_f1_lineplot.png
v08_hcb_vs_v06_label_aware_lineplot.png
v08_hcb_macro_hamming_tradeoff_bar.png
v08_hcb_per_label_mean_vs_labelaware_bar.png
```

### Command and methodology docs

```text
docs/COMMANDS_V08.md
docs/APPENDIX.md
docs/MULTILABEL_EXPERIMENT_LOG.md
```

Purpose:

| File | Purpose |
|---|---|
| `docs/COMMANDS_V08.md` | Full reproducible PowerShell command history, including delta review, manifest build, training, corrected holdout evaluation, v0.6 re-evaluation, global max diagnostic, and label-aware aggregation. |
| `docs/APPENDIX.md` | Thesis-style methodology appendix. |
| `docs/MULTILABEL_EXPERIMENT_LOG.md` | Chronological experiment log and decisions. |

### Final reporting policy

| Result type | Method | Use |
|---|---|---|
| Main official v0.8-HCB result | Parent mean aggregation, fixed threshold 0.5, Exit 3 | Overall corrected-holdout headline result. |
| Label-aware research finding | Mean for 8 stable labels, max for `audience_reaction_present` and `silence_present` | Macro-F1 and transient-label analysis. |
| Global max diagnostic | Max for all labels | Diagnostic only; not final because it over-predicts labels. |

---

## v0.9 documentation layout

The v0.9 experiment should be stored separately from v0.8 so the two result sets remain traceable.

### Branch

```text
agentic_data_preprocessing_v0.9
```

### Scripts

```text
scripts/v0.9/run_labelwise_aggregation_threshold_calibration.py
scripts/v0.9/evaluate_frozen_labelwise_aggregation_v09.py
```

### Workspace outputs

```text
human_talk_workspace/tata_v0.9_labelwise_calibration/verification/v08_parent_mean_fixed/
human_talk_workspace/tata_v0.9_labelwise_calibration/verification/v08_label_aware_fixed/
human_talk_workspace/tata_v0.9_labelwise_calibration/repeated_v07_style_calibration/
human_talk_workspace/tata_v0.9_labelwise_calibration/final_frozen_v06_labelwise_fixed_0p5/
```

### Recommended tables

```text
docs/tables/agentic_data_preprocessing_v0.9/v09_final_full_holdout_comparison.csv
docs/tables/agentic_data_preprocessing_v0.9/v09_repeated_split_calibration_summary.csv
docs/tables/agentic_data_preprocessing_v0.9/v09_frozen_labelwise_method_map.csv
docs/tables/agentic_data_preprocessing_v0.9/v09_frozen_labelwise_per_label_metrics.csv
```

### Recommended figures

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.9/v09_final_comparison_bar.png
docs/figures/human_talk/agentic_data_preprocessing_v0.9/v09_repeated_split_strategy_comparison.png
docs/figures/human_talk/agentic_data_preprocessing_v0.9/v09_per_label_f1_frozen_labelwise.png
```

### v0.9 reporting policy

| Result type | Method | Use |
|---|---|---|
| v0.8 official baseline | Parent mean, fixed 0.5 | Baseline reference. |
| v0.8 simple label-aware | Mean for stable labels, max for audience/silence | Earlier post-hoc improvement. |
| v0.9 repeated split | 20 seeds, 50/50 calibration/evaluation | Stability test for strategy selection. |
| v0.9 final frozen result | Frozen labelwise map, fixed 0.5, full corrected holdout | New final best result. |

