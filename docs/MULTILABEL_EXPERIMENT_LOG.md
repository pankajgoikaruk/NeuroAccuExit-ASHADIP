# Human-Talk Multi-Label Early-Exit Experiment Log

This log records the current staged human-talk evaluation on the `kexit_human_talk_incremental_eval` branch.

```text
Project: NeuroAccuExit-ASHADIP
Branch: kexit_human_talk_incremental_eval
Current study: clean human-talk staged speaker classification + Raw5 agentic preprocessing extension
Compared models: 3-exit no-hint vs 5-exit no-hint
Evaluation levels: segment, clip, confusion matrix
```

---

## 1. Why this branch exists

The previous multi-label branches established a sigmoid/BCE early-exit pipeline. This branch tests whether the same K-exit architecture can generalise to a different audio domain: human-talk speaker classification.

The clean speaker stages are not intended to prove real-world noisy robustness yet. They are a controlled benchmark to test:

1. Whether the model learns speaker-discriminative log-mel features.
2. Whether performance degrades smoothly as classes increase.
3. Whether 5 exits give a useful accuracy/compute trade-off.
4. Whether clip aggregation improves robustness over 1-second segments.

---

## 2. Stage chronology

| Stage | Purpose | Status |
|---|---|---|
| `clean2_balanced` | Two-speaker sanity check | Completed |
| `clean3_balanced` | First scalability check | Completed |
| `clean4_balanced` | Intermediate scalability stage | Completed |
| `clean5_balanced` | Main clean scalability stage | Completed |
| Clip/confusion evaluation | Fairer segment + clip comparison | Completed |
| Noisy/raw speaker stage | Robustness stress test | Started via Raw5 agentic-cleaned experiment |
| `raw5_agentic_cleaned` | First agentic-cleaned Raw5 speaker experiment | Completed |
| Matched Raw5 uncleaned baseline | Required comparison against cleaned stage | Not yet run |

---

## 3. Dataset summary

| stage   |   n_labels | labels                                                           |   n_parent_clips |   n_segments |   train_segments |   val_segments |   test_segments |
|:--------|-----------:|:-----------------------------------------------------------------|-----------------:|-------------:|-----------------:|---------------:|----------------:|
| clean2  |          2 | Les_Brown, Simon_Sinek                                           |              944 |         8496 |             5940 |           1278 |            1278 |
| clean3  |          3 | Les_Brown, Simon_Sinek, Rabin_Sharma                             |             1416 |        12744 |             8910 |           1917 |            1917 |
| clean4  |          4 | Les_Brown, Simon_Sinek, Rabin_Sharma, Oprah_Winfrey              |             1888 |        16992 |            11880 |           2556 |            2556 |
| clean5  |          5 | Les_Brown, Mel_Robbins, Oprah_Winfrey, Rabin_Sharma, Simon_Sinek |             2205 |        19845 |            13905 |           2970 |            2970 |

---

## 4. Main segment-level results

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

---

## 5. Confusion results

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

---

## 6. Clean5 detailed confusion findings

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

Important clean5 notes:

- 3-exit selected-policy segment confusion accuracy: `98.48%`.
- 5-exit selected-policy segment confusion accuracy: `96.67%`.
- 3-exit clip dynamic-policy confusion accuracy: `100.00%`.
- 5-exit clip dynamic-policy confusion accuracy: `98.79%`.
- `Simon_Sinek` is the most recurring weak/confused class.
- 5-exit over-predicts `Simon_Sinek` in the clean5 selected policy.

---

## 7. Research conclusions

### 7.1 Performance scaling

The staged benchmark shows smooth degradation rather than sharp collapse:

```text
3-exit selected Macro-F1:
clean2 = 0.9898
clean3 = 0.9808
clean4 = 0.9789
clean5 = 0.9758

5-exit selected Macro-F1:
clean2 = 0.9898
clean3 = 0.9792
clean4 = 0.9696
clean5 = 0.9629
```

### 7.2 Accuracy baseline

The 3-exit model is the stronger accuracy baseline from `clean3` onward.

### 7.3 Efficiency model

The 5-exit model is the efficiency-oriented model:

```text
5-exit estimated depth-compute saving:
clean2 = 19.50%
clean3 = 17.58%
clean4 = 17.10%
clean5 = 16.93%
```

### 7.4 Clip-level robustness

Clip-level confusion accuracy is stronger than segment-level confusion accuracy, which suggests that aggregating windows from a parent clip reduces isolated 1-second mistakes.

### 7.5 Main weakness

The hardest recurring speaker is `Simon_Sinek`, especially at the segment level. This should be investigated using:

- confusion matrices,
- per-class precision/recall,
- spectrogram inspection,
- speaker similarity / background overlap checks.

---

## 8. Commands used

### Run a full stage

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
  -Stage clean5_balanced `
  -RawRoot human_talk_dataset `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -Clean `
  -ZipResults
```

### Refresh evaluation only

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
  -Stage clean5_balanced `
  -RawRoot human_talk_dataset `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -SkipPrepare `
  -SkipFeatures `
  -SkipTrain3 `
  -SkipTrain5 `
  -ZipResults
```

### Export confusion matrices

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_confusion_eval_all.ps1 `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -ZipResults
```

### Package existing confusion outputs

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_confusion_eval_all.ps1 `
  -WorkspaceRoot human_talk_workspace `
  -ZipOnly `
  -ZipResults
```

---

## 9. Interpretation rules

Use these rules when writing the paper/report:

1. Use Macro-F1 as the main result.
2. Use Hamming Loss / Jaccard / Exact Match as multi-label correctness checks.
3. Use AUPRC/mAP to discuss probability ranking.
4. Use confusion matrices as single-label speaker-identification interpretation.
5. Use 3-exit as the accuracy baseline.
6. Use 5-exit as the dynamic efficiency model.
7. Do not claim that 5-exit is more accurate.

---

## 10. Next experiments

1. Tuned-threshold evaluation.
2. Measured FLOPs and device latency.
3. Noisy/raw speaker classes.
4. True multi-label human-talk + environmental audio mixtures.
5. Event detection latency only after adding event-onset metadata.

---

<!-- AGENTIC_RAW5_RESULTS_START -->

## 11. Agentic Raw5 cleaned experiment

### 11.1 Run identity

```text
Run ID: raw5_agentic_cleaned_3exit_greedy_final_001
Timestamp UTC: 2026-05-22T11:32:45Z
Branch context: agentic_data_preprocessing
Dataset stage: raw5_agentic_cleaned
Final cleaned pool: 3,108 files
Classes: Brene_Brown, Eckhart_Tolle, Eric_Thomas, Gary_Vee, Jay_Shetty
Model: TinyAudioCNN + ExitNet
Exits: 3
Tap blocks: [1, 3]
Exit hint: false
Policy tau: 0.95
```

### 11.2 Command

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "human_talk_workspace\datasets\raw5_agentic_cleaned" `
  -Variant "raw5_agentic_cleaned_3exit_greedy_final" `
  -Policy greedy `
  -Device cpu `
  -InputMode segment `
  -Labels "Brene_Brown,Eckhart_Tolle,Eric_Thomas,Gary_Vee,Jay_Shetty" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -SampleRate 16000 `
  -Bandpass "50,7600" `
  -NMels 64 `
  -TapBlocks "1,3" `
  -SplitUnit file `
  -RunClipPolicy `
  -ForceRebuild
```

### 11.3 Dataset outcome

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

### 11.4 Main result table

| Evaluation mode | Accuracy | Samples / windows | Avg exit depth | Avg windows used | Windows saved | Compute saved |
|---|---:|---:|---:|---:|---:|---:|
| Segment greedy policy | 96.83% | 4040 windows | 2.089 | — | — | 52.56% vs full-depth segment |
| Full-clip greedy aggregation | 99.57% | 467 clips / 4040 windows | 2.089 | 8.651 / 8.651 | 0.00% | 0.00% |
| Depth×Time clip greedy | 98.93% | 467 clips / 975 used windows | 2.092 | 2.088 / 8.651 | 75.87% | 75.82% |

### 11.5 Per-exit test quality

| Exit | Accuracy | Macro-F1 | Weighted-F1 | Test windows |
|---|---:|---:|---:|---:|
| Exit 1 | 65.62% | 64.04% | 64.36% | 4040 |
| Exit 2 | 92.40% | 92.29% | 92.37% | 4040 |
| Exit 3 / Final | 97.60% | 97.56% | 97.59% | 4040 |

### 11.6 Exit behaviour

| Metric | Value |
|---|---:|
| Exit 1 usage | 18.71% |
| Exit 2 usage | 53.71% |
| Exit 3 usage | 27.57% |
| Average exit depth | 2.089 |
| Flip-any rate | 35.79% |
| Average flip count | 0.401 |
| Exit consistency | 99.13% |
| Policy threshold `tau` | 0.95 |
| Policy ECE | 0.0104 |

### 11.7 Clip-level mistakes

| Evaluation | Mistake summary |
|---|---|
| Full-clip greedy | 2 wrong clips: `Brene_Brown → Eric_Thomas` (1), `Gary_Vee → Jay_Shetty` (1) |
| Depth×Time greedy | 5 wrong clips: `Brene_Brown → Gary_Vee` (3), `Eric_Thomas → Jay_Shetty` (1), `Gary_Vee → Eckhart_Tolle` (1) |

### 11.8 Research interpretation

This is the first successful agentic-cleaned Raw5 result. It shows that the cleaned Raw5 speaker data is highly trainable and that clip aggregation improves reliability over individual 1-second windows. Depth×Time provides the strongest dynamic result: **98.93%** clip accuracy with **75.87%** windows saved and **75.82%** compute saved.

### 11.9 Next experiment

Run the matched uncleaned baseline:

```text
raw5_uncleaned_3exit_greedy
```

This will allow a direct table:

| Dataset stage | Segment Acc | Full Clip Acc | Depth×Time Acc | Windows Saved | Compute Saved |
|---|---:|---:|---:|---:|---:|
| Raw5 uncleaned | TBD | TBD | TBD | TBD | TBD |
| Raw5 agentic cleaned | 96.83% | 99.57% | 98.93% | 75.87% | 75.82% |

<!-- AGENTIC_RAW5_RESULTS_END -->
