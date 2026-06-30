# Documentation structure — Agentic Data Preprocessing v0.9_3

This file records the frozen documentation layout for `agentic_data_preprocessing_v0.9_3`.

v0.9_3 freezes the LATS-v0.9 novelty work: a label-wise aggregation and threshold-search inference layer over the frozen v0.8-HCB model probabilities.

---

## Branch identity

| Item | Details |
|---|---|
| Branch | `agentic_data_preprocessing_v0.9` |
| Subbranch | `agentic_data_preprocessing_v0.9_3` |
| Main purpose | Freeze LATS-v0.9 as a label-wise aggregation and threshold-search novelty layer |
| Training status | No retraining in v0.9_3 |
| Base model reused | `main_v08_human_corrected_balanced_3exit_20260610_084027` |
| Evaluation level | Parent/clip level from segment-level probabilities |
| Holdout set | Corrected human-talk holdout |
| Parent clips | 867 |
| Segments | 4,335 |
| Labels | 10 labels: 6 target speakers + `other_speaker_present`, `music_present`, `audience_reaction_present`, `silence_present` |
| Previous best v0.9 method | `v09_frozen_frequency_plus_gary_mean` |
| Final frozen method | `lats_final_frozen_config_v09` |
| Final method name | LATS-v0.9: Label-wise Aggregation and Threshold Search |
| Final threshold policy | Label-specific thresholds selected from repeated calibration splits |
| Final aggregation methods | Label-specific `mean`, `max`, `top2mean`, and `top3mean` |


---

## Top-level files

| File | Purpose |
|---|---|
| `README.md` | Main branch/subbranch summary, research questions, final LATS result, and frozen novelty claim |
| `DOC_STRUCTURE.md` | Documentation layout for the frozen v0.9_3 package |
| `configs/v0.9/labelwise_aggregation_maps.json` | Candidate maps plus final `lats_final_frozen_config_v09` aggregation/threshold config |
| `scripts/v0.9/run_lats_labelwise_aggregation_threshold_search_v09.py` | Main LATS search script |
| `scripts/v0.9/run_labelwise_aggregation_threshold_calibration.py` | Earlier repeated split aggregation/threshold calibration |
| `scripts/v0.9/evaluate_labelwise_mapping_bank_v09.py` | Mapping-bank evaluator |
| `scripts/v0.9/evaluate_frozen_labelwise_aggregation_v09.py` | Frozen map evaluator |

---

## v0.9 documentation files

| Path | Purpose |
|---|---|
| `docs/v0.9/COMMANDS_V09.md` | Reproducible PowerShell commands, including LATS-v0.9 |
| `docs/v0.9/AGENTIC_V09_AGGREGATION_CALIBRATION_GUIDE.md` | Method guide for LATS and earlier aggregation experiments |
| `docs/v0.9/APPENDIX.md` | Appendix-ready formulation, final rules, and tables |
| `docs/v0.9/MULTILABEL_EXPERIMENT_LOG.md` | Chronological experiment log ending with LATS freeze |
| `docs/v0.9/APPLY_V09_DOC_UPDATES.md` | Instructions for applying the documentation update |

---

## Reports and result summaries

| Path | Purpose |
|---|---|
| `docs/reports/v0.9/V09_LABELWISE_AGGREGATION_CALIBRATION_REPORT.md` | Preferred full narrative report for LATS-v0.9 |
| `docs/reports/v0.9/V09_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md` | Backward-compatible report path containing the same frozen narrative |
| `docs/results/v0.9/V09_RESULTS_SUMMARY.md` | Compact result summary for quick reference |

---

## Evidence tables

All v0.9_3 table evidence should be kept under:

```text
docs/tables/agentic_data_preprocessing_v0.9/
```

Core LATS evidence:

```text
v09_lats_final_full_holdout_comparison.csv
v09_lats_final_full_holdout_comparison.md
v09_lats_repeated_split_std_summary.csv
v09_lats_repeated_split_std_summary.md
v09_lats_final_frozen_config.csv
v09_lats_final_frozen_config.md
v09_lats_final_per_label_metrics.csv
v09_lats_final_per_label_metrics.md
v09_lats_selection_stability.csv
v09_lats_selection_stability.md
v09_lats_repeated_seed_labelwise_stability.csv
v09_lats_repeated_seed_labelwise_stability.md
lats_final_full_holdout_eval.csv
lats_final_full_holdout_per_label.csv
lats_repeated_eval_summary.csv
lats_threshold_summary.csv
lats_label_selection_frequency.csv
```

Earlier v0.9 comparison evidence remains useful:

```text
v09_mapping_bank_summary_fixed_0p5_plus_gary_mean.csv
v09_mapping_bank_per_label_fixed_0p5_plus_gary_mean.csv
v09_mapping_bank_summary_with_tata_lawyer_thresholds.csv
v09_repeated_eval_summary.csv
v09_v06_selection_frequency.csv
v09_v07_selection_frequency.csv
v09_v07_threshold_summary.csv
```

---

## Raw workspace evidence folders

| Folder | Purpose |
|---|---|
| `human_talk_workspace/tata_v0.9_labelwise_calibration/lats_v09_search/` | Final LATS-v0.9 outputs |
| `human_talk_workspace/tata_v0.9_labelwise_calibration/mapping_bank_fixed_0p5_plus_gary_mean/` | Previous best mapping-bank result |
| `human_talk_workspace/tata_v0.9_labelwise_calibration/mapping_bank_with_tata_lawyer_thresholds/` | Old threshold-transfer diagnostic |
| `human_talk_workspace/tata_v0.9_labelwise_calibration/repeated_v07_style_calibration/` | Earlier repeated split calibration |
| `human_talk_workspace/tata_v0.9_labelwise_calibration/verification/v08_parent_mean_fixed/` | Segment-probability source CSV |

---

## Final reporting decision

Report the final frozen method as:

```text
Subbranch: agentic_data_preprocessing_v0.9_3
Method: LATS-v0.9 final frozen config
Internal config name: lats_final_frozen_config_v09
```

```text
Method     = lats_final_frozen_config_v09
Macro-F1   = 0.8667
Micro-F1   = 0.9436
Samples-F1 = 0.9495
Exact      = 0.8524
Hamming    = 0.0165
```
