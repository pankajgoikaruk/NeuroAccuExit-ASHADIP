# Documentation Structure — agentic_data_preprocessing_v0.6

This document defines the active documentation structure for **`agentic_data_preprocessing_v0.6`**.

```text
Branch: agentic_data_preprocessing_v0.6
Documentation focus: TATA-assisted human-in-the-loop raw-data preprocessing
Final recommended model: 3-exit fixed threshold 0.5 with parent-level mean aggregation
Final raw holdout result: Macro-F1 0.7598, Micro-F1 0.8976, Samples-F1 0.9048, Exact Match 0.8155, Hamming Loss 0.0271
```

## Current documentation scope

1. v0.6 motivation: move from TATA seed validation to full raw-data pseudo-manifest generation.
2. Label schema: 10 labels with merged `audience_reaction_present`.
3. TATA seed training and validation.
4. Raw dataset split into pseudo-pool and final holdout.
5. TATA hybrid routing on the pseudo-pool.
6. Human correction of `needs_review` rows.
7. Final expanded training manifest construction.
8. Main 3-exit and 5-exit model training.
9. Threshold tuning and dynamic policy comparison.
10. Final raw holdout evaluation at segment level and parent/clip level.
11. Aggregation comparison: segment-level vs parent max vs parent mean.
12. Research conclusions, limitations, and future work.

## Canonical paths

| Artifact | Path |
|---|---|
| Raw pipeline root | `human_talk_workspace/tata_v0.6_raw_pipeline/` |
| Final expanded training manifest | `human_talk_workspace/tata_v0.6_raw_pipeline/final_expanded_training_dataset/metadata/multilabel_features_manifest.csv` |
| Final holdout ground truth | `human_talk_workspace/tata_v0.6_raw_pipeline/manual_review_queue/01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv` |
| Main 3-exit run | `human_talk_workspace/tata_v0.6_raw_pipeline/main_models/runs/main_v06_expanded_3exit_20260603_194435` |
| Main 5-exit run | `human_talk_workspace/tata_v0.6_raw_pipeline/main_models/runs/main_v06_expanded_5exit_20260603_210324` |
| Final holdout evaluation root | `human_talk_workspace/tata_v0.6_raw_pipeline/final_holdout_evaluation/` |

## Canonical result tables

### TATA seed validation

| model                  |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |
|:-----------------------|-----------:|-----------:|-------------:|--------------:|---------------:|
| TATA v0.6 3-exit fixed |     0.8164 |     0.8272 |       0.8264 |        0.6594 |         0.0474 |
| TATA v0.6 3-exit tuned |     0.8291 |     0.8121 |       0.8075 |        0.5977 |         0.0543 |

### Raw routing counts

| mode      |   accepted |   accepted_with_warning |   needs_review |   rejected |   accepted_plus_warning |
|:----------|-----------:|------------------------:|---------------:|-----------:|------------------------:|
| fixed_0p5 |        537 |                    1189 |           2711 |        473 |                    1726 |
| hybrid    |        369 |                     925 |           3171 |        445 |                    1294 |

### Internal main-model comparison

| model            |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   compute_saved |
|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|
| 3-exit fixed 0.5 |     0.8225 |     0.8219 |       0.824  |        0.6221 |         0.0502 |            0    |
| 3-exit tuned     |     0.8217 |     0.8154 |       0.823  |        0.5875 |         0.0542 |            0    |
| 3-exit dynamic   |     0.8217 |     0.8154 |       0.823  |        0.5875 |         0.0542 |            0    |
| 5-exit fixed 0.5 |     0.812  |     0.7999 |       0.7892 |        0.5767 |         0.0562 |            0    |
| 5-exit tuned     |     0.8217 |     0.8079 |       0.8101 |        0.5752 |         0.0562 |            0    |
| 5-exit dynamic   |     0.7764 |     0.761  |       0.7624 |        0.4656 |         0.0727 |           19.75 |

### Final raw holdout: segment-level

| setting          |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   compute_saved |
|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|
| 3-exit fixed 0.5 |     0.729  |     0.8476 |       0.8507 |        0.7301 |         0.0409 |            0    |
| 3-exit tuned     |     0.726  |     0.8489 |       0.8639 |        0.7165 |         0.0417 |            0    |
| 5-exit tuned     |     0.725  |     0.8396 |       0.8518 |        0.6932 |         0.0441 |            0    |
| 5-exit dynamic   |     0.6766 |     0.7912 |       0.8092 |        0.6028 |         0.0597 |           24.01 |

### Final raw holdout: parent-level max

| setting          |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_pred_labels   |   compute_saved |
|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|----------------:|
| 3-exit fixed 0.5 |     0.7337 |     0.8073 | 0.8427       |        0.5767 |         0.0619 | 1.8720            |            0    |
| 3-exit tuned     |     0.7009 |     0.7858 | 0.8227       |        0.5063 |         0.0712 | 1.9804            |            0    |
| 5-exit tuned     |     0.7275 |     0.7844 | 0.8134       |        0.4787 |         0.0721 | 2.0012            |            0    |
| 5-exit dynamic   |     0.6892 |     0.7451 |              |        0.391  |         0.0893 |                   |           18.69 |

### Final raw holdout: parent-level mean

| setting          |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_pred_labels   |   compute_saved |
|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|----------------:|
| 3-exit fixed 0.5 |     0.7598 |     0.8976 | 0.9048       |        0.8155 |         0.0271 | 1.3045            |             0   |
| 3-exit tuned     |     0.7615 |     0.8937 | 0.9115       |        0.7785 |         0.0292 | 1.4014            |             0   |
| 5-exit tuned     |     0.77   |     0.8866 | 0.9032       |        0.7439 |         0.0311 | 1.4025            |             0   |
| 5-exit dynamic   |     0.7186 |     0.8283 |              |        0.6332 |         0.0498 |                   |            28.6 |

### Aggregation comparison for final recommended model

| evaluation    |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_pred_labels |
|:--------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|
| Segment-level |     0.729  |     0.8476 |       0.8507 |        0.7301 |         0.0409 |            1.3409 |
| Parent max    |     0.7337 |     0.8073 |       0.8427 |        0.5767 |         0.0619 |            1.872  |
| Parent mean   |     0.7598 |     0.8976 |       0.9048 |        0.8155 |         0.0271 |            1.3045 |

## Figure assets

| Figure | Path | Purpose |
|---|---|---|
| Final parent mean comparison | `figures/human_talk/agentic_data_preprocessing_v0.6/final_holdout_parent_mean_comparison.png` | Compare best final parent-level mean settings. |
| Aggregation comparison | `figures/human_talk/agentic_data_preprocessing_v0.6/aggregation_comparison_3exit_fixed.png` | Show segment vs parent max vs parent mean for 3-exit fixed. |
| Hamming loss comparison | `figures/human_talk/agentic_data_preprocessing_v0.6/final_holdout_hamming_loss_comparison.png` | Show reliability differences by evaluation setting. |
| Dynamic policy tradeoff | `figures/human_talk/agentic_data_preprocessing_v0.6/dynamic_policy_tradeoff_parent_level.png` | Compare accuracy vs compute saving. |
| Raw routing counts | `figures/human_talk/agentic_data_preprocessing_v0.6/raw_pseudo_routing_counts.png` | Show fixed vs hybrid routing volume. |
| Per-label F1 | `figures/human_talk/agentic_data_preprocessing_v0.6/per_label_f1_segment_3exit_fixed.png` | Show label-level strengths and weaknesses. |

## Documentation rules

- Use **parent-level mean aggregation** as the main final raw holdout result.
- Use **3-exit fixed threshold 0.5** as the best reliable final model.
- Do not claim that 5-exit is the best accuracy model. It is not.
- Describe 5-exit dynamic as an efficiency/accuracy trade-off ablation.
- Do not claim that TATA labels are perfect; describe them as useful pseudo labels protected by conservative routing and human correction.
- State clearly that final holdout labels are manually reviewed clip-level ground truth.
- State clearly that segment labels are weak inherited labels.
- Mention the known limitation: non-target speaker rows may require future background/event re-checking.
