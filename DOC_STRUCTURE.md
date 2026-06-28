# Documentation structure — Agentic Data Preprocessing v0.9

This file records the recommended documentation layout for `agentic_data_preprocessing_v0.9`. v0.9 is a post-hoc labelwise aggregation branch built on top of the v0.8-HCB model.

---

## Top-level files

| File | Purpose |
|---|---|
| `README.md` | Main branch summary and final v0.9 result |
| `DOC_STRUCTURE.md` | Version-specific documentation layout |
| `configs/v0.9/labelwise_aggregation_maps.json` | Candidate aggregation maps and optional thresholds |
| `scripts/v0.9/run_labelwise_aggregation_threshold_calibration.py` | Repeated split aggregation/threshold calibration |
| `scripts/v0.9/evaluate_frozen_labelwise_aggregation_v09.py` | Single frozen-map evaluator |
| `scripts/v0.9/evaluate_labelwise_mapping_bank_v09.py` | Mapping-bank evaluator |

---

## v0.9 documentation files

| Path | Purpose |
|---|---|
| `docs/v0.9/COMMANDS_V09.md` | Reproducible PowerShell commands |
| `docs/v0.9/AGENTIC_V09_AGGREGATION_CALIBRATION_GUIDE.md` | Method guide for aggregation and threshold calibration |
| `docs/v0.9/APPENDIX.md` | Appendix-ready technical details |
| `docs/v0.9/MULTILABEL_EXPERIMENT_LOG.md` | Chronological experiment log |
| `docs/v0.9/APPLY_V09_DOC_UPDATES.md` | Copy/apply instructions for this documentation package |

---

## Reports and result summaries

| Path | Purpose |
|---|---|
| `docs/reports/v0.9/V09_LABELWISE_AGGREGATION_CALIBRATION_REPORT.md` | Preferred full narrative report for the v0.9 aggregation branch |
| `docs/reports/v0.9/V09_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md` | Backward-compatible renamed report path, same content |
| `docs/results/v0.9/V09_RESULTS_SUMMARY.md` | Compact results summary for quick reference |

Although the report filename contains `HUMAN_CORRECTED_BALANCED`, the v0.9 experiment itself is not a new training run. It reuses the v0.8-HCB model and evaluates labelwise parent aggregation on the corrected holdout.

---

## Evidence tables

All v0.9 table evidence should be kept under:

```text
docs/tables/agentic_data_preprocessing_v0.9/
```

Recommended files:

```text
v09_repeated_eval_summary.csv
v09_v06_selection_frequency.csv
v09_v07_selection_frequency.csv
v09_v07_threshold_summary.csv
v09_mapping_bank_summary_fixed_0p5_plus_gary_mean.csv
v09_mapping_bank_per_label_fixed_0p5_plus_gary_mean.csv
v09_mapping_bank_summary_with_tata_lawyer_thresholds.csv
parent_holdout_eval_frozen_labelwise_v09_fixed_0p5.json
parent_holdout_per_label_frozen_labelwise_v09_fixed_0p5.csv
```

---

## Figures

Use this folder for any future v0.9 plots:

```text
docs/figures/human_talk/agentic_data_preprocessing_v0.9/
```

Recommended future figures:

```text
v09_mapping_bank_macro_f1_barplot.png
v09_mapping_bank_hamming_loss_barplot.png
v09_per_label_f1_final_map.png
v09_repeated_split_macro_f1_summary.png
```

No figure is required to reproduce the current result; all final metrics are available in CSV/JSON evidence files.

---

## Workspace evidence folders

The raw experiment outputs are expected under:

```text
human_talk_workspace/tata_v0.9_labelwise_calibration/
```

Important subfolders:

| Folder | Purpose |
|---|---|
| `verification/v08_parent_mean_fixed/` | Segment-probability source from the v0.8-HCB model |
| `verification/v08_label_aware_fixed/` | Simple mean/max diagnostic |
| `repeated_v07_style_calibration/` | Repeated split aggregation and threshold calibration |
| `final_frozen_v06_labelwise_fixed_0p5/` | Original frozen frequency-map full holdout |
| `mapping_bank_fixed_0p5_plus_gary_mean/` | Final selected mapping-bank result |
| `mapping_bank_with_tata_lawyer_thresholds/` | Diagnostic showing old threshold values do not transfer |

---

## Final reporting decision

Report the final method as:

```text
v09_frozen_frequency_plus_gary_mean
```

with:

```text
Macro-F1   = 0.8518
Micro-F1   = 0.9374
Samples-F1 = 0.9464
Exact      = 0.8431
Hamming    = 0.0183
```
