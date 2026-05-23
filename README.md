# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing Branch

This branch documents the **current agentic preprocessing workflow** for the Raw5 human-talk speaker dataset.
Older staged clean-speaker branch results are intentionally removed from this document so that the repository reflects the current branch only.

```text
Branch: agentic_data_preprocessing_v0.3
Current experiment: raw5_agentic_cleaned_3exit_greedy_final_001
Task: five-speaker human-talk classification
Dataset stage: raw5_agentic_cleaned
Model: TinyAudioCNN + ExitNet
Exits: 3
Tap blocks: [1, 3]
Policy: greedy segment policy + full-clip and Depth×Time clip evaluation
Input: 64-mel log-mel features
Windowing: 1.0 s windows, 0.5 s hop
Status: first cleaned Raw5 experiment completed
```

---

## 1. Branch purpose

The purpose of this branch is to test whether an **agentic AI preprocessing layer** can make a noisy/raw speaker dataset suitable for early-exit TinyML audio classification.

The branch focuses on:

1. Non-destructive dataset auditing.
2. Manifest-first accepted / needs-review / rejected routing.
3. Safe audio conversion into training-ready 16 kHz mono WAV files.
4. Manual traceability for excluded samples.
5. Early-exit evaluation after agentic cleaning.
6. Accuracy-efficiency analysis using segment, full-clip, and Depth×Time policies.

---

## 2. Current Raw5 speaker classes

| Class | Final training-ready files |
|---|---:|
| Brene_Brown | 595 |
| Eckhart_Tolle | 660 |
| Eric_Thomas | 593 |
| Gary_Vee | 642 |
| Jay_Shetty | 618 |
| **Total** | **3,108** |

One file was manually excluded after re-audit and listening:

```text
Eric_Thomas__0175.wav
Reason: pure music, no target speaker audio
Action: excluded from raw5_agentic_cleaned training root
```

---

## 3. Agentic preprocessing summary

### 3.1 Raw audit split

| Decision | Count |
|---|---:|
| Accepted | 3109 |
| Needs review | 27 |
| Rejected | 34 |
| Blocked | 0 |
| **Total** | **3170** |

### 3.2 Class-wise raw audit split

| Class | Accepted | Needs review | Rejected | Total |
|---|---:|---:|---:|---:|
| Brene_Brown | 595 | 0 | 6 | 601 |
| Eckhart_Tolle | 660 | 0 | 0 | 660 |
| Eric_Thomas | 594 | 26 | 10 | 630 |
| Gary_Vee | 642 | 1 | 7 | 650 |
| Jay_Shetty | 618 | 0 | 11 | 629 |

### 3.3 Build and final audit status

| Item | Result |
|---|---:|
| Built cleaned files | 3,109 |
| Build failures | 0 |
| Missing sources | 0 |
| Re-audited cleaned files before manual exclusion | 3,109 |
| Re-audit accepted | 3,108 |
| Re-audit rejected | 1 |
| Final files after manual exclusion | 3,108 |
| Sample-rate mismatch after build | 0 |
| Channel mismatch after build | 0 |

The cleaned dataset is therefore structurally valid for training after removing the single music-only file from the training root.

---

## 4. First current-branch result

```text
Run ID: raw5_agentic_cleaned_3exit_greedy_final_001
Timestamp UTC: 2026-05-22T11:32:45Z
Test segments: 4040
Test clips: 467
Threshold tau: 0.95
Temperatures: 0.7143, 0.9819, 1.3302
Exit hint: False
```

### 4.1 Main results

| Evaluation mode | Accuracy | Avg exit depth | Avg windows used | Windows saved | Compute saved |
|---|---:|---:|---:|---:|---:|
| Segment greedy policy | 96.83% | 2.089 | — | — | 52.56% |
| Full-clip greedy aggregation | 99.57% | 2.089 | 8.651 / 8.651 | 0.00% | 0.00% |
| Depth×Time clip greedy | 98.93% | 2.092 | 2.088 / 8.651 | 75.87% | 75.82% |

### 4.2 Per-exit test performance

| Exit | Accuracy | Macro F1 | Weighted F1 | Test segments |
|---|---:|---:|---:|---:|
| Exit 1 | 65.62% | 64.04% | 64.36% | 4040 |
| Exit 2 | 92.40% | 92.29% | 92.37% | 4040 |
| Exit 3 | 97.60% | 97.56% | 97.59% | 4040 |

### 4.3 Exit behaviour

| Metric | Value |
|---|---:|
| Exit 1 usage | 18.71% |
| Exit 2 usage | 53.71% |
| Exit 3 usage | 27.57% |
| Avg exit depth | 2.089 |
| Flip-any rate | 35.79% |
| Avg flip count | 0.401 |
| Exit consistency | 99.13% |

---

## 5. Interpretation

The first current-branch experiment is successful:

- Segment greedy accuracy is **96.83%**.
- Full-clip aggregation reaches **99.57%**.
- Depth×Time keeps **98.93%** while saving **75.87%** windows and **75.82%** compute.
- Most early exits happen at Exit 2, which suggests the model is using the dynamic architecture rather than always exiting at the final head.
- The single rejected cleaned file was correctly identified as invalid speaker training data after manual inspection.

---

## 6. Known reproducibility note

The cache path records:

```text
bp50-7600
```

but `config_used.yaml` still shows:

```yaml
audio:
  bandpass:
  - 100
  - 3000
```

This means the CLI/cache likely used the wider `50,7600` setting, but the saved YAML did not fully preserve the CLI override. This should be fixed before final paper reporting.

---

## 7. Next required experiment

The next required comparison is:

```text
raw5_uncleaned_3exit_greedy
```

The same model/settings should be run on the uncleaned Raw5 dataset. That will allow a direct table:

| Dataset stage | Segment Acc | Full Clip Acc | Depth×Time Acc | Windows Saved | Compute Saved |
|---|---:|---:|---:|---:|---:|
| Raw5 uncleaned | Not run yet | Not run yet | Not run yet | Not run yet | Not run yet |
| Raw5 agentic cleaned | 96.83% | 99.57% | 98.93% | 75.87% | 75.82% |

---

## 8. GitHub upload commands

```powershell
git add README.md DOC_STRUCTURE.md MULTILABEL_EXPERIMENT_LOG.md APPENDIX.md
git commit -m "docs: keep current agentic preprocessing results only"
git push origin agentic_data_preprocessing_v0.3
```
