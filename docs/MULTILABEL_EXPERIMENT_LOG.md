# Experiment Log — Agentic Data Preprocessing Branch

This log records only the current `agentic_data_preprocessing` branch.

```text
Project: NeuroAccuExit-ASHADIP
Branch: agentic_data_preprocessing
Current completed run: raw5_agentic_cleaned_3exit_greedy_final_001
Current dataset stage: raw5_agentic_cleaned
Task: five-speaker human-talk speaker classification
```

---

## 1. Branch objective

This branch tests whether agentic preprocessing can convert a noisy Raw5 speaker dataset into a safe and traceable training stage for the NeuroAccuExit early-exit audio pipeline.

The branch is not documenting previous clean-stage experiments. Those results belong to earlier branches and are intentionally excluded here.

---

## 2. Current stage chronology

| Step | Output | Status |
|---|---|---|
| Raw5 folder preparation | Brene_Brown, Eckhart_Tolle, Eric_Thomas, Gary_Vee, Jay_Shetty | Completed |
| Raw5 rename | Standardized class-based filenames | Completed |
| DatasetAuditorAgent | Raw5 audit report and recommended actions | Completed |
| ManifestBuilderAgent | Accepted / needs-review / rejected / blocked manifests | Completed |
| DatasetBuilderAgent | 16 kHz mono cleaned copies | Completed |
| Cleaned re-audit | One bad file detected | Completed |
| Manual exclusion | `Eric_Thomas__0175.wav` moved out of training root | Completed |
| First cleaned experiment | `raw5_agentic_cleaned_3exit_greedy_final_001` | Completed |
| Matched uncleaned baseline | `raw5_uncleaned_3exit_greedy` | Not yet run |

---

## 3. Raw5 audit result

| Decision | Count |
|---|---:|
| Accepted | 3,109 |
| Needs review | 27 |
| Rejected | 34 |
| Blocked | 0 |
| **Total** | **3,170** |

### Class-wise audit split

| Class | Accepted | Needs review | Rejected | Total |
|---|---:|---:|---:|---:|
| Brene_Brown | 595 | 0 | 6 | 601 |
| Eckhart_Tolle | 660 | 0 | 0 | 660 |
| Eric_Thomas | 594 | 26 | 10 | 630 |
| Gary_Vee | 642 | 1 | 7 | 650 |
| Jay_Shetty | 618 | 0 | 11 | 629 |

---

## 4. Cleaned dataset build result

| Item | Count |
|---|---:|
| Accepted manifest rows used for build | 3,109 |
| Files built | 3,109 |
| Build failures | 0 |
| Missing sources | 0 |
| Re-audit accepted | 3,108 |
| Re-audit rejected | 1 |
| Final training files after manual exclusion | 3,108 |

Final class counts:

| Class | Count |
|---|---:|
| Brene_Brown | 595 |
| Eckhart_Tolle | 660 |
| Eric_Thomas | 593 |
| Gary_Vee | 642 |
| Jay_Shetty | 618 |

---

## 5. Manual exclusion note

The cleaned re-audit rejected:

```text
Eric_Thomas__0175.wav
```

Manual listening confirmed that the file contains pure music and no target speaker. This file should not be trained as `Eric_Thomas`, because it would teach the model a wrong association:

```text
music = Eric_Thomas
```

The generated cleaned copy is therefore excluded from the training root, while traceability is preserved.

---

## 6. First cleaned Raw5 experiment

### 6.1 Run metadata

| Item | Value |
|---|---|
| Run ID | `raw5_agentic_cleaned_3exit_greedy_final_001` |
| Timestamp UTC | `2026-05-22T11:32:45Z` |
| Dataset | `raw5_agentic_cleaned` |
| Final training pool | 3,108 files |
| Test windows | 4040 |
| Test clips | 467 |
| Exits | 3 |
| Tap blocks | `[1, 3]` |
| Policy | greedy |
| Tau | 0.95 |
| Exit hint | disabled |
| Temperatures | 0.7143, 0.9819, 1.3302 |

### 6.2 Per-exit performance

| Exit | Accuracy | Macro F1 | Weighted F1 | Support |
|---|---:|---:|---:|---:|
| Exit 1 | 65.62% | 64.04% | 64.36% | 4040 |
| Exit 2 | 92.40% | 92.29% | 92.37% | 4040 |
| Exit 3 | 97.60% | 97.56% | 97.59% | 4040 |

### 6.3 Segment and clip results

| Evaluation mode | Accuracy | Avg exit depth | Avg windows used | Windows saved | Compute saved |
|---|---:|---:|---:|---:|---:|
| Segment greedy policy | 96.83% | 2.089 | — | — | 52.56% |
| Full-clip greedy aggregation | 99.57% | 2.089 | 8.651 / 8.651 | 0.00% | 0.00% |
| Depth×Time clip greedy | 98.93% | 2.092 | 2.088 / 8.651 | 75.87% | 75.82% |

### 6.4 Exit behaviour

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

## 7. Clip-level mistakes

### Full-clip greedy aggregation

| True class | Predicted class | Count |
|---|---|---:|
| Brene_Brown | Eric_Thomas | 1 |
| Gary_Vee | Jay_Shetty | 1 |

### Depth×Time clip greedy

| True class | Predicted class | Count |
|---|---|---:|
| Brene_Brown | Gary_Vee | 3 |
| Eric_Thomas | Jay_Shetty | 1 |
| Gary_Vee | Eckhart_Tolle | 1 |

---

## 8. Findings from current branch

1. The agentic preprocessing workflow successfully produced a usable Raw5 cleaned dataset.
2. The cleaned dataset trained a strong 3-exit speaker model.
3. Full-clip aggregation reached 99.57% accuracy.
4. Depth×Time retained 98.93% clip accuracy while saving 75.87% windows and 75.82% compute.
5. The re-audit and manual inspection caught a music-only file that would have polluted the `Eric_Thomas` class.
6. A matched uncleaned baseline is still required before claiming a preprocessing improvement.

---

## 9. Next experiment to run

```text
raw5_uncleaned_3exit_greedy
```

Use the same model settings and evaluation outputs so the comparison is fair.
