# Documentation Structure — Human-Talk Incremental + Agentic Preprocessing Evaluation

This document defines the reporting structure for the staged human-talk evaluation branch.

```text
Branch: kexit_human_talk_incremental_eval
Study type: staged clean speaker-identification benchmark
Compared models: 3-exit no-hint vs 5-exit no-hint
Stages: clean2, clean3, clean4, clean5
Primary result type: segment-level multi-label early-exit metrics
Secondary result type: clip-level and segment-level confusion matrices
```

---

## 1. Recommended paper/report structure

### 1.1 Motivation

- NeuroAccuExit was previously evaluated on environmental/multi-label audio settings.
- The human-talk benchmark tests whether the same architecture can generalise to speaker-discriminative audio patterns.
- The staged design checks whether performance drops smoothly or sharply as more clean speakers are added.

### 1.2 Research questions

| ID | Question | Current answer |
|---|---|---|
| RQ1 | Does NeuroAccuExit generalise to clean human-talk speaker classification? | Yes; all clean stages achieve high Macro-F1. |
| RQ2 | Does performance collapse as speakers increase from 2 to 5? | No; degradation is smooth. |
| RQ3 | Is 5-exit more accurate than 3-exit? | No; 3-exit is the stronger accuracy baseline. |
| RQ4 | Does 5-exit provide an efficiency trade-off? | Yes; it saves about 16.9%–19.5% estimated depth compute. |
| RQ5 | Does clip aggregation help? | Yes; clip-level confusion accuracy is stronger than segment-level accuracy. |
| RQ6 | Which classes are most confused? | `Simon_Sinek` is the dominant weak/confused class in segment-level results. |

---

## 2. Dataset section

Report the staged design using this table:

| stage   |   n_labels | labels                                                           |   n_parent_clips |   n_segments |   train_segments |   val_segments |   test_segments |
|:--------|-----------:|:-----------------------------------------------------------------|-----------------:|-------------:|-----------------:|---------------:|----------------:|
| clean2  |          2 | Les_Brown, Simon_Sinek                                           |              944 |         8496 |             5940 |           1278 |            1278 |
| clean3  |          3 | Les_Brown, Simon_Sinek, Rabin_Sharma                             |             1416 |        12744 |             8910 |           1917 |            1917 |
| clean4  |          4 | Les_Brown, Simon_Sinek, Rabin_Sharma, Oprah_Winfrey              |             1888 |        16992 |            11880 |           2556 |            2556 |
| clean5  |          5 | Les_Brown, Mel_Robbins, Oprah_Winfrey, Rabin_Sharma, Simon_Sinek |             2205 |        19845 |            13905 |           2970 |            2970 |

Key points to write:

- Each parent clip is approximately 5 seconds.
- Each parent clip is segmented into 1-second windows with 0.5-second hop.
- Each parent clip produces 9 segment-level windows.
- Splits are file/parent based to avoid segment leakage.
- Stages are balanced by parent clips where possible.

---

## 3. Methodology section

### 3.1 Architecture

- TinyAudioCNN backbone.
- ExitNet wrapper.
- 3-exit model: tap blocks `1,3` plus final exit.
- 5-exit model: tap blocks `1,2,3,4` plus final exit.
- Sigmoid outputs are retained because the implementation is compatible with future true multi-label audio tagging.

### 3.2 Evaluation levels

| Level | Description | Main use |
|---|---|---|
| Segment-level | Each 1-second window is evaluated independently. | Early-exit quality and depth-compute saving. |
| Clip-level | Segment/window outputs are aggregated by parent clip. | Robustness and confusion analysis. |
| Confusion matrix | Single-label view of model output. | Speaker-identification interpretation. |

### 3.3 Metrics

Primary thresholded metrics:

- Macro-F1
- Micro-F1
- Samples-F1
- Exact Match
- Hamming Loss / Hamming Accuracy
- Jaccard Score
- Label Cardinality Error

Probability-based metrics:

- Macro-AUPRC / mAP
- Micro-AUPRC
- Per-label AUPRC

Efficiency/stability metrics:

- Average exit depth
- Exit distribution
- Estimated depth-compute saved
- Exit consistency
- Label-set flip-any rate
- Average label-set flip count
- Average label-bit flip count

Confusion metrics:

- Accuracy
- Error count
- Per-class precision, recall, F1
- TP / FP / TN / FN
- Worst-class F1

---

## 4. Results section

### 4.1 Segment-level selected-policy results

| stage   | model_type   |   macro_f1 |   exact_match |   hamming_loss |   hamming_accuracy |   jaccard_score |   macro_auprc |   avg_exit_depth |   depth_compute_saved_pct |   exit_consistency |   label_set_flip_any_rate |
|:--------|:-------------|-----------:|--------------:|---------------:|-------------------:|----------------:|--------------:|-----------------:|--------------------------:|-------------------:|--------------------------:|
| clean2  | 3-exit       |     0.9898 |        0.9898 |         0.0102 |             0.9898 |          0.9898 |        0.9996 |           3      |                    0      |             1      |                    0.1369 |
| clean2  | 5-exit       |     0.9898 |        0.9898 |         0.0102 |             0.9898 |          0.9898 |        0.9989 |           4.025  |                   19.4992 |             0.9953 |                    0.1385 |
| clean3  | 3-exit       |     0.9808 |        0.975  |         0.0127 |             0.9873 |          0.9763 |        0.9976 |           3      |                    0      |             1      |                    0.4956 |
| clean3  | 5-exit       |     0.9792 |        0.9729 |         0.0137 |             0.9863 |          0.9755 |        0.9955 |           4.121  |                   17.5796 |             0.987  |                    0.4971 |
| clean4  | 3-exit       |     0.9789 |        0.9667 |         0.0105 |             0.9895 |          0.9705 |        0.9976 |           3      |                    0      |             1      |                    0.7433 |
| clean4  | 5-exit       |     0.9696 |        0.9515 |         0.0154 |             0.9846 |          0.9634 |        0.995  |           4.1451 |                   17.097  |             0.9855 |                    0.8075 |
| clean5  | 3-exit       |     0.9758 |        0.9589 |         0.0096 |             0.9904 |          0.9655 |        0.9976 |           3      |                    0      |             1      |                    0.7428 |
| clean5  | 5-exit       |     0.9629 |        0.9414 |         0.0152 |             0.9848 |          0.954  |        0.994  |           4.1535 |                   16.9293 |             0.9778 |                    0.731  |

### 4.2 3-exit vs 5-exit selected-policy deltas

| stage   |   macro_f1_3exit |   macro_f1_5exit |   macro_f1_delta_5_minus_3 |   exact_match_3exit |   exact_match_5exit |   compute_saved_3exit_pct |   compute_saved_5exit_pct |   exit_consistency_5exit |   flip_rate_5exit |
|:--------|-----------------:|-----------------:|---------------------------:|--------------------:|--------------------:|--------------------------:|--------------------------:|-------------------------:|------------------:|
| clean2  |           0.9898 |           0.9898 |                     0      |              0.9898 |              0.9898 |                         0 |                   19.4992 |                   0.9953 |            0.1385 |
| clean3  |           0.9808 |           0.9792 |                    -0.0016 |              0.975  |              0.9729 |                         0 |                   17.5796 |                   0.987  |            0.4971 |
| clean4  |           0.9789 |           0.9696 |                    -0.0093 |              0.9667 |              0.9515 |                         0 |                   17.097  |                   0.9855 |            0.8075 |
| clean5  |           0.9758 |           0.9629 |                    -0.0129 |              0.9589 |              0.9414 |                         0 |                   16.9293 |                   0.9778 |            0.731  |

### 4.3 Confusion-matrix summary

| stage   | model_type   | level   | policy          |   n_samples |   accuracy |   errors | worst_class   |   worst_class_f1 |
|:--------|:-------------|:--------|:----------------|------------:|-----------:|---------:|:--------------|-----------------:|
| clean2  | 3-exit       | clip    | dynamic_policy  |         142 |     0.993  |        1 | Simon_Sinek   |           0.9929 |
| clean2  | 3-exit       | clip    | full_final      |         142 |     1      |        0 | Les_Brown     |           1      |
| clean2  | 5-exit       | clip    | dynamic_policy  |         142 |     1      |        0 | Les_Brown     |           1      |
| clean2  | 5-exit       | clip    | full_final      |         142 |     1      |        0 | Les_Brown     |           1      |
| clean2  | 3-exit       | segment | selected_policy |        1278 |     0.9898 |       13 | Simon_Sinek   |           0.9898 |
| clean2  | 5-exit       | segment | selected_policy |        1278 |     0.9898 |       13 | Les_Brown     |           0.9898 |
| clean3  | 3-exit       | clip    | dynamic_policy  |         213 |     0.9906 |        2 | Simon_Sinek   |           0.9859 |
| clean3  | 3-exit       | clip    | full_final      |         213 |     1      |        0 | Les_Brown     |           1      |
| clean3  | 5-exit       | clip    | dynamic_policy  |         213 |     0.9906 |        2 | Simon_Sinek   |           0.9859 |
| clean3  | 5-exit       | clip    | full_final      |         213 |     1      |        0 | Les_Brown     |           1      |
| clean3  | 3-exit       | segment | selected_policy |        1917 |     0.9812 |       36 | Simon_Sinek   |           0.9739 |
| clean3  | 5-exit       | segment | selected_policy |        1917 |     0.9802 |       38 | Simon_Sinek   |           0.9721 |
| clean4  | 3-exit       | clip    | dynamic_policy  |         284 |     0.993  |        2 | Simon_Sinek   |           0.9859 |
| clean4  | 3-exit       | clip    | full_final      |         284 |     0.9965 |        1 | Oprah_Winfrey |           0.9929 |
| clean4  | 5-exit       | clip    | dynamic_policy  |         284 |     1      |        0 | Les_Brown     |           1      |
| clean4  | 5-exit       | clip    | full_final      |         284 |     1      |        0 | Les_Brown     |           1      |
| clean4  | 3-exit       | segment | selected_policy |        2556 |     0.9812 |       48 | Simon_Sinek   |           0.9734 |
| clean4  | 5-exit       | segment | selected_policy |        2556 |     0.98   |       51 | Simon_Sinek   |           0.9719 |
| clean5  | 3-exit       | clip    | dynamic_policy  |         330 |     1      |        0 | Les_Brown     |           1      |
| clean5  | 3-exit       | clip    | full_final      |         330 |     1      |        0 | Les_Brown     |           1      |
| clean5  | 5-exit       | clip    | dynamic_policy  |         330 |     0.9879 |        4 | Oprah_Winfrey |           0.9688 |
| clean5  | 5-exit       | clip    | full_final      |         330 |     0.9939 |        2 | Oprah_Winfrey |           0.9846 |
| clean5  | 3-exit       | segment | selected_policy |        2970 |     0.9848 |       45 | Simon_Sinek   |           0.9702 |
| clean5  | 5-exit       | segment | selected_policy |        2970 |     0.9667 |       99 | Simon_Sinek   |           0.9319 |

### 4.4 Clean5 per-class confusion analysis

| model_type   | label         |   precision |   recall |     f1 |   support |   predicted |   tp |   fp |   fn |
|:-------------|:--------------|------------:|---------:|-------:|----------:|------------:|-----:|-----:|-----:|
| 3-exit       | Les_Brown     |      1      |   0.9949 | 0.9975 |       594 |         591 |  591 |    0 |    3 |
| 3-exit       | Mel_Robbins   |      0.9966 |   0.9916 | 0.9941 |       594 |         591 |  589 |    2 |    5 |
| 3-exit       | Oprah_Winfrey |      0.9832 |   0.9832 | 0.9832 |       594 |         594 |  584 |   10 |   10 |
| 3-exit       | Rabin_Sharma  |      0.9626 |   0.9966 | 0.9793 |       594 |         615 |  592 |   23 |    2 |
| 3-exit       | Simon_Sinek   |      0.9827 |   0.9579 | 0.9702 |       594 |         579 |  569 |   10 |   25 |
| 5-exit       | Les_Brown     |      0.9899 |   0.9882 | 0.989  |       594 |         593 |  587 |    6 |    7 |
| 5-exit       | Mel_Robbins   |      0.9949 |   0.9815 | 0.9881 |       594 |         586 |  583 |    3 |   11 |
| 5-exit       | Oprah_Winfrey |      0.9927 |   0.9209 | 0.9555 |       594 |         551 |  547 |    4 |   47 |
| 5-exit       | Rabin_Sharma  |      0.9895 |   0.9529 | 0.9708 |       594 |         572 |  566 |    6 |   28 |
| 5-exit       | Simon_Sinek   |      0.8802 |   0.9899 | 0.9319 |       594 |         668 |  588 |   80 |    6 |

---

## 5. Recommended figures

Use these figures in the main report:

| Figure | File |
|---|---|
| Selected Macro-F1 by stage | `docs/figures/human_talk/selected_macro_f1_by_stage.png` |
| Estimated compute saved | `docs/figures/human_talk/selected_compute_saved_by_stage.png` |
| Average exit depth | `docs/figures/human_talk/selected_avg_exit_depth_by_stage.png` |
| Label-set flip rate | `docs/figures/human_talk/selected_flip_rate_by_stage.png` |
| Segment selected-policy confusion accuracy | `docs/figures/human_talk/segment_selected_confusion_accuracy.png` |
| Clip dynamic-policy confusion accuracy | `docs/figures/human_talk/clip_dynamic_confusion_accuracy.png` |
| Clean5 confusion errors | `docs/figures/human_talk/clean5_confusion_errors.png` |
| Clean2 example confusion matrix | `docs/figures/human_talk/c2__3e__segment_confusion__segment_final_exit_confusion_matrix.png` |
| Clean5 3-exit selected confusion | `docs/figures/human_talk/c5__3e__segment_confusion__segment_selected_policy_confusion_matrix.png` |
| Clean5 5-exit selected confusion | `docs/figures/human_talk/c5__5e__segment_confusion__segment_selected_policy_confusion_matrix.png` |

---

## 6. Correct research narrative

### Strong claim supported by results

> NeuroAccuExit remains effective on clean human-talk speaker classification, and performance decreases smoothly rather than sharply as the number of clean speaker classes increases.

### Strong claim for 3-exit

> The 3-exit model is the strongest accuracy baseline in the current human-talk benchmark.

### Strong claim for 5-exit

> The 5-exit model provides the better dynamic early-exit efficiency trade-off, saving estimated depth compute while retaining high performance.

### Claim to avoid

> The 5-exit model is more accurate.

This is not supported by the staged results.

---

<!-- AGENTIC_RAW5_RESULTS_START -->

## 8. Agentic preprocessing extension — Raw5 cleaned stage

This documentation now includes an agentic preprocessing extension on the Raw5 speaker dataset.

### 8.1 Additional research questions

| ID | Question | Current answer |
|---|---|---|
| RQ7 | Can the agentic preprocessing workflow convert noisy raw speaker folders into a training-ready cleaned dataset? | Yes; 3,109 accepted files were built as 16 kHz mono cleaned copies, with one additional music-only file manually excluded after re-audit. |
| RQ8 | Does the cleaned Raw5 stage produce a strong first speaker-classification result? | Yes; the 3-exit greedy model achieved 96.83% segment accuracy, 99.57% full-clip accuracy, and 98.93% Depth×Time clip accuracy. |

### 8.2 Agentic preprocessing dataset summary

| Item | Count / value |
|---|---:|
| Raw5 files audited | 3,170 |
| Accepted by agent | 3,109 |
| Needs review | 27 |
| Rejected | 34 |
| Blocked | 0 |
| Cleaned files built | 3,109 |
| Manually excluded after cleaned re-audit | 1 |
| Final training-ready cleaned files | 3,108 |

| Class | Final cleaned files |
|---|---:|
| `Brene_Brown` | 595 |
| `Eckhart_Tolle` | 660 |
| `Eric_Thomas` | 593 |
| `Gary_Vee` | 642 |
| `Jay_Shetty` | 618 |
| **Total** | **3,108** |

### 8.3 Main Raw5 cleaned result

| Evaluation mode | Accuracy | Samples / windows | Avg exit depth | Avg windows used | Windows saved | Compute saved |
|---|---:|---:|---:|---:|---:|---:|
| Segment greedy policy | 96.83% | 4040 windows | 2.089 | — | — | 52.56% vs full-depth segment |
| Full-clip greedy aggregation | 99.57% | 467 clips / 4040 windows | 2.089 | 8.651 / 8.651 | 0.00% | 0.00% |
| Depth×Time clip greedy | 98.93% | 467 clips / 975 used windows | 2.092 | 2.088 / 8.651 | 75.87% | 75.82% |

### 8.4 Interpretation for reporting

The first Raw5 cleaned result supports the claim that the agentic preprocessing stage can produce a usable speaker dataset without modifying raw data. The strongest headline is the clip-level accuracy-efficiency trade-off: Depth×Time loses only **0.64 percentage points** against full-clip aggregation while saving **75.87%** windows and **75.82%** compute.

### 8.5 Remaining comparison needed

The next required baseline is `raw5_uncleaned_3exit_greedy` with the same architecture and evaluation settings. That comparison is needed before claiming that agentic preprocessing improves accuracy over the uncleaned raw dataset.

<!-- AGENTIC_RAW5_RESULTS_END -->

---

## 7. Remaining work

1. Add tuned threshold comparison against fixed threshold `0.5`.
2. Add measured FLOPs/device latency.
3. Run noisy/raw speaker stages.
4. Extend from speaker identity to true multi-label audio tagging.
5. Add event onset metadata before using detection latency for transient classes.
