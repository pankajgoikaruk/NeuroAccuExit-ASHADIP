# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing v0.8

This branch documents the active **`agentic_data_preprocessing_v0.8`** experiment for ASHADIP/NeuroAccuExit multi-label human-talk audio. The v0.8 branch extends the v0.6/v0.7 TATA-assisted preprocessing work with **LAWYER label-specific weak-label refinement**, **delta-only human correction**, **known non-target identity repair**, and a final **v0.8-human-corrected-balanced** training/evaluation pipeline.

```text
Branch: agentic_data_preprocessing_v0.8
Main experiment: v0.8-human-corrected-balanced
Final run: main_v08_human_corrected_balanced_3exit_20260610_084027
Task: 10-label multi-label human-talk audio classification
Evaluation focus: corrected holdout, parent/clip-level mean aggregation, fixed threshold 0.5
Final model: 3 exits, tap_blocks=1,3, final exit selected for official reporting
```

## Executive result

The best official v0.8 result is the **final exit** of the 3-exit model using **parent/clip-level mean aggregation** and **fixed threshold 0.5** on the corrected holdout:

| model           |   final_exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:----------------|-------------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| v0.6 3-exit     |            3 |     0.7537 |     0.8865 |       0.8992 |        0.7497 |         0.0315 |            1.4694 |            1.3045 |
| v0.6 5-exit     |            5 |     0.746  |     0.8771 |       0.8881 |        0.7232 |         0.0338 |            1.4694 |            1.2814 |
| v0.8-HCB 3-exit |            3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4694 |            1.4302 |

**Recommended headline result:**

```text
v0.8-human-corrected-balanced 3-exit
Corrected holdout: 867 parent clips / 4,335 segments
Parent-level aggregation: mean probability
Threshold: fixed 0.5
Exit: 3
Macro-F1     = 0.7801
Micro-F1     = 0.9332
Samples-F1   = 0.9406
Exact Match  = 0.8397
Hamming Loss = 0.0194
```

On the corrected parent-level holdout set containing 867 parent clips and 4,335 one-second segments, the v0.8-human-corrected-balanced 3-exit model achieved the strongest final-exit performance under mean probability aggregation and a fixed 0.5 threshold. Compared with the previous v0.6 3-exit model re-evaluated on the same corrected holdout, it improved Macro-F1 from 0.7537 to 0.7801, Micro-F1 from 0.8865 to 0.9332, Samples-F1 from 0.8992 to 0.9406, and Exact Match from 0.7497 to 0.8397, while reducing Hamming Loss from 0.0315 to 0.0194. The model also predicted a more realistic number of labels per clip, increasing average predicted labels from 1.3045 to 1.4302 against a corrected ground-truth average of 1.4694.

## Why v0.8 was needed

The earlier v0.6 broad experiment showed that TATA-assisted pseudo-manifest generation works, but later inspection revealed that some known non-target speakers had incomplete background/event labels and that the final holdout required a corrected label refresh. v0.7 removed five non-target source folders as a filtered ablation, but filtering alone did not solve weak background/event labels. v0.8 therefore moved to a safer strategy:

```text
v0.6 trusted base
+ corrected non-target context labels
+ corrected holdout labels
+ only reviewed LAWYER-new samples
+ known non-target identity repair
+ controlled balancing of background-heavy rows
= v0.8-human-corrected-balanced
```

## Label schema

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
audience_reaction_present
silence_present
```

Known non-target source folders are not target classes:

```text
Les_Brown
Mel_Robbins
Oprah_Winfrey
Rabin_Sharma
Simon_Sinek
```

For these folders, the identity rule is:

```text
all target speaker labels = 0
other_speaker_present = 1
music/audience/silence are context labels and are preserved from review
```

## v0.8 data construction summary

### Delta-only review design

The project deliberately avoided a full re-review of all data. Instead, v0.8 used **delta-only correction** from the trusted v0.6 base.

| Item                              |   Rows |
|:----------------------------------|-------:|
| v0.6 trusted base index           |   4465 |
| raw non-target context review     |   1860 |
| holdout non-target context review |    426 |
| LAWYER changed-label review queue |   2471 |
| LAWYER new-samples review queue   |    291 |
| training delta master review      |   3334 |

Only the reviewed safe subsets were included in this v0.8-HCB experiment. The large 2,471-row changed-label queue remains a future ablation queue and was not blindly trusted.

### Manifest and balance summary

| item                                        |   count |
|:--------------------------------------------|--------:|
| seed_segment_rows                           |   12469 |
| raw_expanded_segment_rows                   |   23780 |
| final_combined_segment_rows                 |   36249 |
| raw_parent_labels_used                      |    4756 |
| zero_active_corrected_needs_review_excluded |       0 |
| missing_parent_segment_groups               |       0 |

| label                     |   before_balance |   after_balance |
|:--------------------------|-----------------:|----------------:|
| Brene_Brown               |             2885 |            2885 |
| Eckhart_Tolle             |             3145 |            3145 |
| Eric_Thomas               |             2850 |            2850 |
| Gary_Vee                  |             3135 |            3135 |
| Jay_Shetty                |             4225 |            4225 |
| Nick_Vujicic              |             2425 |            2425 |
| other_speaker_present     |            15916 |            9030 |
| music_present             |            13045 |           11393 |
| audience_reaction_present |             5124 |            5124 |
| silence_present           |             1724 |            1724 |

Balancing reduced the heavy `other_speaker_present` dominance while preserving all target-speaker, audience, silence, and seed rows.

## Training settings

| Setting                             | Value                                                                                                                                                        |
|:------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| branch                              | agentic_data_preprocessing_v0.8                                                                                                                              |
| experiment                          | v0.8-human-corrected-balanced                                                                                                                                |
| run_name                            | main_v08_human_corrected_balanced_3exit                                                                                                                      |
| manifest                            | human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\final_expanded_training_dataset_balanced\metadata\multilabel_features_manifest_balanced.csv |
| features_root                       | .                                                                                                                                                            |
| tap_blocks                          | 1,3                                                                                                                                                          |
| epochs                              | 40                                                                                                                                                           |
| batch_size                          | 64                                                                                                                                                           |
| learning_rate                       | 0.001                                                                                                                                                        |
| threshold                           | 0.5                                                                                                                                                          |
| device                              | cpu                                                                                                                                                          |
| use_pos_weight                      | False                                                                                                                                                        |
| loss_weights                        | [0.3, 0.3, 1.0]                                                                                                                                              |
| best_epoch                          | 39                                                                                                                                                           |
| best_validation_final_exit_macro_f1 | 0.8105259060931592                                                                                                                                           |

## Internal test result

|   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
|      1 |     0.2185 |     0.358  |       0.2833 |        0.1535 |         0.1293 |            1.4493 |            0.565  |
|      2 |     0.6713 |     0.6837 |       0.6478 |        0.4472 |         0.0844 |            1.4493 |            1.2208 |
|      3 |     0.8305 |     0.8283 |       0.8285 |        0.6206 |         0.0502 |            1.4493 |            1.4737 |

## Corrected holdout result

### Fixed threshold 0.5, parent mean

| model           | threshold_mode   | aggregation   |   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   jaccard_score |   avg_true_labels |   avg_pred_labels |
|:----------------|:-----------------|:--------------|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|------------------:|------------------:|
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      1 |     0.113  |     0.3166 |       0.204  |        0.0288 |         0.1275 |          0.1596 |            1.4694 |            0.3956 |
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      2 |     0.6315 |     0.7739 |       0.7197 |        0.5467 |         0.0591 |          0.6752 |            1.4694 |            1.1419 |
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |          0.9174 |            1.4694 |            1.4302 |

### Tuned thresholds, parent mean

| model           | threshold_mode   | aggregation   |   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   jaccard_score |   avg_true_labels |   avg_pred_labels |
|:----------------|:-----------------|:--------------|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|------------------:|------------------:|
| v0.8-HCB 3-exit | tuned_per_exit   | mean          |      1 |     0.3756 |     0.5239 |       0.547  |        0.1546 |         0.2146 |          0.4356 |            1.4694 |            3.0392 |
| v0.8-HCB 3-exit | tuned_per_exit   | mean          |      2 |     0.7134 |     0.8107 |       0.8328 |        0.5409 |         0.0597 |          0.7671 |            1.4694 |            1.6863 |
| v0.8-HCB 3-exit | tuned_per_exit   | mean          |      3 |     0.7487 |     0.9139 |       0.921  |        0.8143 |         0.0243 |          0.8955 |            1.4694 |            1.3576 |

Fixed 0.5 is the final recommended setting. Threshold tuning improved internal validation/test Macro-F1 slightly, but it reduced corrected-holdout parent-level Macro-F1, Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

## Figures

- ![Training validation curve](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_training_validation_curve.png)
- ![v0.8 vs v0.6 corrected holdout bar](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_vs_v06_corrected_holdout_bar.png)
- ![Per-label corrected holdout F1](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_corrected_holdout_per_label_f1_bar.png)

- ![Training validation curve](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_avg_true_vs_pred_labels_bar.png)
- ![Training validation curve](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_corrected_holdout_fixed_by_exit_lineplot.png)
- ![Training validation curve](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_internal_test_by_exit_lineplot.png)
- ![Training validation curve](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_label_counts_before_after_balance.png)
- ![Training validation curve](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_training_loss_hamming_curve.png)
- ![Training validation curve](docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_vs_v06_hamming_loss_bar.png) 

## Documentation map

- `docs/reports/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md` — thesis-ready detailed report.
- `docs/results/V08_RESULTS_SUMMARY.md` — compact results and comparison tables.
- `docs/COMMANDS_V08.md` — full PowerShell command log and purpose.
- `docs/APPENDIX.md` — expanded methodology appendix.
- `docs/MULTILABEL_EXPERIMENT_LOG.md` — chronological experiment log.
- `docs/tables/` — CSV source tables used in the docs.
- `docs/figures/` — generated line/bar plots.

## Key conclusion

v0.8-HCB is the current strongest model and should replace the old v0.6 headline as the main ASHADIP/TATA-assisted preprocessing result. The fair corrected-holdout comparison shows that v0.8-HCB improves global reliability and produces a more realistic number of predicted labels per clip. Remaining work should focus on rare event labels, especially `audience_reaction_present` and `silence_present` on corrected holdout.
