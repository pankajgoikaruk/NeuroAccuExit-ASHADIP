# Human-Talk Multi-Label Early-Exit Experiment Log

This log records the current staged human-talk evaluation on the `kexit_human_talk_incremental_eval` branch.

```text
Project: NeuroAccuExit-ASHADIP
Branch: kexit_human_talk_incremental_eval
Current study: clean human-talk staged speaker classification
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
| Noisy/raw speaker stage | Robustness stress test | Not yet run |

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
