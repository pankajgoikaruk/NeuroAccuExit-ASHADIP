# Multi-label Experiment Log — v0.8 Human-Corrected-Balanced

## Branch

```text
agentic_data_preprocessing_v0.8
```

## Chronological log

### 1. Problem identified

Earlier v0.6 and v0.7 results showed strong target-speaker recognition, but non-target/background context labels required correction. The main risk was that known non-target source folders were being used as broad `other_speaker_present` examples while their `music_present`, `audience_reaction_present`, and `silence_present` context labels were incomplete.

### 2. Strategy selected

Instead of deleting audio or fully re-reviewing everything, v0.8 used a delta-only strategy:

```text
v0.6 trusted base
+ reviewed v0.8 LAWYER-new samples
+ corrected non-target context labels
+ corrected holdout context labels
+ known non-target identity repair
+ balanced background-heavy rows
```

### 3. Manual review queues

| Queue                                     |   Rows |
|:------------------------------------------|-------:|
| 01_v06_trusted_base_index.csv             |   4465 |
| 03_raw_nontarget_context_REVIEW.csv       |   1860 |
| 04_holdout_nontarget_context_REVIEW.csv   |    426 |
| 05_lawyer_delta_changed_labels_REVIEW.csv |   2471 |
| 06_lawyer_new_samples_REVIEW.csv          |    291 |
| 07_training_delta_master_REVIEW.csv       |   3334 |

Completed manual review in this experiment:

```text
03_raw_nontarget_context_REVIEW.csv
04_holdout_nontarget_context_REVIEW.csv
06_lawyer_new_samples_REVIEW.csv
```

The large changed-label queue remains for future ablation.

### 4. Corrected outputs

| Output                                                |   Rows |
|:------------------------------------------------------|-------:|
| corrected raw hybrid needs-review                     |   3171 |
| corrected holdout ground truth                        |    867 |
| reviewed LAWYER-new samples                           |    291 |
| hybrid accepted-with-warning plus reviewed LAWYER-new |   1216 |

### 5. Manifest build and balance

| Item                                       |   Count |
|:-------------------------------------------|--------:|
| final combined segment rows before balance |   36249 |
| balanced segment rows                      |   29363 |
| dropped background-heavy rows              |    6886 |
| train rows after balance                   |   25519 |
| validation rows after balance              |    1883 |
| test rows after balance                    |    1961 |

### 6. Training

```text
Run: main_v08_human_corrected_balanced_3exit_20260610_084027
Best epoch: 39
Best validation final-exit Macro-F1: 0.8105
```

|   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
|      1 |     0.2185 |     0.358  |       0.2833 |        0.1535 |         0.1293 |            1.4493 |            0.565  |
|      2 |     0.6713 |     0.6837 |       0.6478 |        0.4472 |         0.0844 |            1.4493 |            1.2208 |
|      3 |     0.8305 |     0.8283 |       0.8285 |        0.6206 |         0.0502 |            1.4493 |            1.4737 |

### 7. Corrected holdout evaluation

| model           | threshold_mode   | aggregation   |   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   jaccard_score |   avg_true_labels |   avg_pred_labels |
|:----------------|:-----------------|:--------------|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|------------------:|------------------:|
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      1 |     0.113  |     0.3166 |       0.204  |        0.0288 |         0.1275 |          0.1596 |            1.4694 |            0.3956 |
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      2 |     0.6315 |     0.7739 |       0.7197 |        0.5467 |         0.0591 |          0.6752 |            1.4694 |            1.1419 |
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |          0.9174 |            1.4694 |            1.4302 |

### 8. Fair v0.6 comparison

| model           |   final_exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:----------------|-------------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| v0.6 3-exit     |            3 |     0.7537 |     0.8865 |       0.8992 |        0.7497 |         0.0315 |            1.4694 |            1.3045 |
| v0.6 5-exit     |            5 |     0.746  |     0.8771 |       0.8881 |        0.7232 |         0.0338 |            1.4694 |            1.2814 |
| v0.8-HCB 3-exit |            3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4694 |            1.4302 |

### 9. Decision

Official v0.8 result:

```text
v0.8-human-corrected-balanced 3-exit
parent-level mean aggregation
fixed threshold 0.5
Exit 3
Macro-F1=0.7801, Micro-F1=0.9332, Samples-F1=0.9406, Exact=0.8397, Hamming=0.0194
```

### 10. Remaining limitations

- `audience_reaction_present` has low corrected-holdout recall under parent mean.
- `silence_present` remains under-predicted on corrected holdout.
- Validation-tuned thresholds do not transfer as well to corrected holdout as fixed 0.5.
- A future event-aware pooling rule may be needed for bursty labels.
