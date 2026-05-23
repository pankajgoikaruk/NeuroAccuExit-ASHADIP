# Appendix — Human-Talk Incremental Evaluation + Agentic Preprocessing

This appendix records the reproducibility protocol, commands, metrics, and result tables for the `kexit_human_talk_incremental_eval` branch.

---

## A1. Branch status

| Item | Value |
|---|---|
| Active branch | `kexit_human_talk_incremental_eval` |
| Task | Clean human-talk speaker classification |
| Compared models | 3-exit no-hint, 5-exit no-hint |
| Model | TinyAudioCNN + ExitNet |
| Feature type | 64-mel log-mel |
| Segment size | 1.0 second |
| Hop size | 0.5 second |
| Threshold mode | fixed sigmoid threshold `0.5` |
| Primary metric | Macro-F1 |
| Secondary metrics | Exact Match, Hamming Loss, Jaccard, AUPRC |
| Efficiency metric | Estimated depth-compute saving |
| Confusion outputs | Segment and clip confusion matrices |

---

## A2. Dataset stages

| stage   |   n_labels | labels                                                           |   n_parent_clips |   n_segments |   train_segments |   val_segments |   test_segments |
|:--------|-----------:|:-----------------------------------------------------------------|-----------------:|-------------:|-----------------:|---------------:|----------------:|
| clean2  |          2 | Les_Brown, Simon_Sinek                                           |              944 |         8496 |             5940 |           1278 |            1278 |
| clean3  |          3 | Les_Brown, Simon_Sinek, Rabin_Sharma                             |             1416 |        12744 |             8910 |           1917 |            1917 |
| clean4  |          4 | Les_Brown, Simon_Sinek, Rabin_Sharma, Oprah_Winfrey              |             1888 |        16992 |            11880 |           2556 |            2556 |
| clean5  |          5 | Les_Brown, Mel_Robbins, Oprah_Winfrey, Rabin_Sharma, Simon_Sinek |             2205 |        19845 |            13905 |           2970 |            2970 |

---

## A3. Full stage execution commands

### clean2

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
  -Stage clean2_balanced `
  -RawRoot human_talk_dataset `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -Clean `
  -ZipResults
```

### clean3

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
  -Stage clean3_balanced `
  -RawRoot human_talk_dataset `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -Clean `
  -ZipResults
```

### clean4

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
  -Stage clean4_balanced `
  -RawRoot human_talk_dataset `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -Clean `
  -ZipResults
```

### clean5

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
  -Stage clean5_balanced `
  -RawRoot human_talk_dataset `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -Clean `
  -ZipResults
```

---

## A4. Evaluation-only commands

### Segment/depth policy refresh

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

### Clip/window evaluation

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clip_eval_all.ps1 `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -ZipResults
```

### Confusion-matrix export

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_confusion_eval_all.ps1 `
  -WorkspaceRoot human_talk_workspace `
  -Device cpu `
  -ZipResults
```

### Confusion package-only rerun

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_confusion_eval_all.ps1 `
  -WorkspaceRoot human_talk_workspace `
  -ZipOnly `
  -ZipResults
```

---

## A5. Segment-level selected-policy metrics

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

## A6. Final-exit static metrics

| stage   | model_type   |   exit |   macro_f1 |   exact_match |   hamming_loss |   macro_auprc |
|:--------|:-------------|-------:|-----------:|--------------:|---------------:|--------------:|
| clean2  | 3-exit       |      3 |     0.9898 |        0.9898 |         0.0102 |        0.9996 |
| clean2  | 5-exit       |      5 |     0.993  |        0.993  |         0.007  |        0.9998 |
| clean3  | 3-exit       |      3 |     0.9808 |        0.975  |         0.0127 |        0.9976 |
| clean3  | 5-exit       |      5 |     0.9795 |        0.9697 |         0.0136 |        0.9972 |
| clean4  | 3-exit       |      3 |     0.9789 |        0.9667 |         0.0105 |        0.9976 |
| clean4  | 5-exit       |      5 |     0.9724 |        0.9566 |         0.0139 |        0.9967 |
| clean5  | 3-exit       |      3 |     0.9758 |        0.9589 |         0.0096 |        0.9976 |
| clean5  | 5-exit       |      5 |     0.9654 |        0.9451 |         0.0141 |        0.996  |

---

## A7. 3-exit vs 5-exit selected-policy deltas

| stage   |   macro_f1_3exit |   macro_f1_5exit |   macro_f1_delta_5_minus_3 |   exact_match_3exit |   exact_match_5exit |   compute_saved_3exit_pct |   compute_saved_5exit_pct |   exit_consistency_5exit |   flip_rate_5exit |
|:--------|-----------------:|-----------------:|---------------------------:|--------------------:|--------------------:|--------------------------:|--------------------------:|-------------------------:|------------------:|
| clean2  |           0.9898 |           0.9898 |                     0      |              0.9898 |              0.9898 |                         0 |                   19.4992 |                   0.9953 |            0.1385 |
| clean3  |           0.9808 |           0.9792 |                    -0.0016 |              0.975  |              0.9729 |                         0 |                   17.5796 |                   0.987  |            0.4971 |
| clean4  |           0.9789 |           0.9696 |                    -0.0093 |              0.9667 |              0.9515 |                         0 |                   17.097  |                   0.9855 |            0.8075 |
| clean5  |           0.9758 |           0.9629 |                    -0.0129 |              0.9589 |              0.9414 |                         0 |                   16.9293 |                   0.9778 |            0.731  |

---

## A8. Confusion-matrix summary

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

## A9. Clean5 per-class selected-policy confusion table

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

## A10. Confusion-matrix interpretation

The confusion matrices are single-label views of a model trained through a sigmoid multi-label pipeline. This is acceptable for the clean human-talk benchmark because each sample has exactly one true speaker label. However, if the model predicts no active label or multiple active labels, the confusion exporter converts the output to a single class using an argmax-style rule. Therefore:

- Macro-F1 / Hamming / Jaccard / AUPRC remain the main multi-label metrics.
- Confusion matrices should be used as speaker-identification interpretability evidence.
- Confusion accuracy may differ slightly from Exact Match.

---

## A11. Key observations

1. `clean2` is nearly solved by both models.
2. `clean3` and `clean4` remain strong and show no sharp collapse.
3. `clean5` is the most useful stress test.
4. The 3-exit selected policy gives better accuracy than the 5-exit selected policy in `clean5`.
5. The 5-exit selected policy gives non-zero depth-compute saving in every stage.
6. Clip-level full-final confusion is often perfect or near-perfect.
7. `Simon_Sinek` is the main recurring weak class in segment-level confusion.
8. For 5-exit clean5, `Simon_Sinek` is over-predicted, while `Oprah_Winfrey` and `Rabin_Sharma` lose recall.

---

## A12. Output artifacts

Expected result package locations:

```text
human_talk_workspace/packages/
├─ human_talk_clean2_balanced_results_to_share_*.zip
├─ human_talk_clean3_balanced_results_to_share_*.zip
├─ human_talk_clean4_balanced_results_to_share_*.zip
├─ human_talk_clean5_balanced_results_to_share_*.zip
├─ human_talk_clip_window_eval_results_*.zip
└─ human_talk_confusion_eval_results_*.zip
```

Expected important output folders inside runs:

```text
multilabel_greedy_policy/
├─ dynamic_early_exit_efficiency.csv
├─ static_per_exit_quality.csv
├─ full_policy_sweep.csv
├─ selected_policy_per_label.csv
└─ confusion/

multilabel_clip_window_policy/
└─ confusion/
```

---

## A13. Git commands

```powershell
git status
git add README.md DOC_STRUCTURE.md APPENDIX.md MULTILABEL_EXPERIMENT_LOG.md docs/figures/human_talk docs/results/human_talk
git commit -m "docs: update human-talk staged benchmark results"
git push origin kexit_human_talk_incremental_eval
```

---

<!-- AGENTIC_RAW5_RESULTS_START -->

## A14. Agentic Raw5 cleaned experiment

### A14.1 Run metadata

| Item | Value |
|---|---|
| Run ID | `raw5_agentic_cleaned_3exit_greedy_final_001` |
| Timestamp UTC | `2026-05-22T11:32:45Z` |
| Branch context | `agentic_data_preprocessing` |
| Dataset stage | `raw5_agentic_cleaned` |
| Classes | `Brene_Brown`, `Eckhart_Tolle`, `Eric_Thomas`, `Gary_Vee`, `Jay_Shetty` |
| Final cleaned files | 3,108 |
| Model | TinyAudioCNN + ExitNet |
| Exits | 3 |
| Tap blocks | `[1, 3]` |
| Feature type | 64-mel log-mel |
| Segment / hop | 1.0 s / 0.5 s |
| Greedy threshold `tau` | 0.95 |
| Temperature scaling | `0.7143, 0.9819, 1.3302` |
| Exit hint | `false` |

### A14.2 Preprocessing and dataset counts

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

### A14.3 Execution command

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

### A14.4 Segment and clip results

| Evaluation mode | Accuracy | Samples / windows | Avg exit depth | Avg windows used | Windows saved | Compute saved |
|---|---:|---:|---:|---:|---:|---:|
| Segment greedy policy | 96.83% | 4040 windows | 2.089 | — | — | 52.56% vs full-depth segment |
| Full-clip greedy aggregation | 99.57% | 467 clips / 4040 windows | 2.089 | 8.651 / 8.651 | 0.00% | 0.00% |
| Depth×Time clip greedy | 98.93% | 467 clips / 975 used windows | 2.092 | 2.088 / 8.651 | 75.87% | 75.82% |

### A14.5 Per-exit static report

| Exit | Accuracy | Macro-F1 | Weighted-F1 | Test windows |
|---|---:|---:|---:|---:|
| Exit 1 | 65.62% | 64.04% | 64.36% | 4040 |
| Exit 2 | 92.40% | 92.29% | 92.37% | 4040 |
| Exit 3 / Final | 97.60% | 97.56% | 97.59% | 4040 |

### A14.6 Exit behaviour diagnostics

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

### A14.7 Full-clip per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `Brene_Brown` | 100.00% | 98.88% | 99.44% | 89 |
| `Eckhart_Tolle` | 100.00% | 100.00% | 100.00% | 99 |
| `Eric_Thomas` | 98.89% | 100.00% | 99.44% | 89 |
| `Gary_Vee` | 100.00% | 98.97% | 99.48% | 97 |
| `Jay_Shetty` | 98.94% | 100.00% | 99.47% | 93 |

### A14.8 Depth×Time per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `Brene_Brown` | 100.00% | 96.63% | 98.29% | 89 |
| `Eckhart_Tolle` | 99.00% | 100.00% | 99.50% | 99 |
| `Eric_Thomas` | 100.00% | 98.88% | 99.44% | 89 |
| `Gary_Vee` | 96.97% | 98.97% | 97.96% | 97 |
| `Jay_Shetty` | 98.94% | 100.00% | 99.47% | 93 |

### A14.9 Clip confusion notes

| Evaluation | Mistake summary |
|---|---|
| Full-clip greedy | 2 wrong clips: `Brene_Brown → Eric_Thomas` (1), `Gary_Vee → Jay_Shetty` (1) |
| Depth×Time greedy | 5 wrong clips: `Brene_Brown → Gary_Vee` (3), `Eric_Thomas → Jay_Shetty` (1), `Gary_Vee → Eckhart_Tolle` (1) |

### A14.10 Reproducibility caveat

`config_used.yaml` records the base YAML bandpass `[100, 3000]`, while the cache path records the effective CLI override `bp50-7600`. The result should be documented as the `50–7600 Hz` run unless the pipeline is rerun after patching config override persistence.

<!-- AGENTIC_RAW5_RESULTS_END -->
