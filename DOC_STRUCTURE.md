# Documentation Structure — Agentic Data Preprocessing Branch

This document defines the documentation structure for the **current** `agentic_data_preprocessing_v0.3` branch only.

Previous clean-stage branch results and old execution tables are intentionally excluded.

```text
Branch: agentic_data_preprocessing_v0.3
Current dataset stage: raw5_agentic_cleaned
Current completed run: raw5_agentic_cleaned_3exit_greedy_final_001
Current task: five-speaker human-talk speaker classification
Primary contribution: agentic preprocessing + early-exit evaluation
```

---

## 1. Current report scope

The documentation should cover only:

1. Raw5 audit and routing.
2. Manifest-first preprocessing design.
3. DatasetBuilderAgent cleaned-data construction.
4. Manual exclusion of confirmed non-speaker data.
5. First 3-exit greedy early-exit result on `raw5_agentic_cleaned`.
6. Required next baseline: `raw5_uncleaned_3exit_greedy`.

The documentation should only include the active agentic preprocessing workflow, its current Raw5 outputs, and the current completed run.

---

## 2. Recommended paper/report structure for this branch

### 2.1 Motivation

Raw audio collected from online or uncontrolled sources often contains music, silence, clipping, interviewer speech, and other non-target content. Training directly on such folders can teach the model wrong class associations. This branch evaluates an agentic preprocessing layer that audits and routes data before training.

### 2.2 Research questions

| ID | Question | Current answer |
|---|---|---|
| RQ1 | Can the agentic preprocessing layer build a usable Raw5 speaker dataset? | Yes; 3,108 final training-ready files were produced. |
| RQ2 | Can the audit separate preprocessing requirements from quality rejection? | Yes; resampling/downmixing were handled without rejecting otherwise valid files. |
| RQ3 | Can the cleaned Raw5 dataset train an early-exit speaker model? | Yes; segment greedy accuracy reached 96.83%. |
| RQ4 | Does clip aggregation improve reliability? | Yes; full-clip accuracy reached 99.57%. |
| RQ5 | Can Depth×Time reduce compute while retaining accuracy? | Yes; 98.93% clip accuracy with 75.87% windows saved. |
| RQ6 | Is a matched uncleaned baseline still required? | Yes; `raw5_uncleaned_3exit_greedy` has not been run yet. |

---

## 3. Dataset section

### 3.1 Raw5 classes

| Class | Accepted after raw audit | Final training-ready files |
|---|---:|---:|
| Brene_Brown | 595 | 595 |
| Eckhart_Tolle | 660 | 660 |
| Eric_Thomas | 594 | 593 |
| Gary_Vee | 642 | 642 |
| Jay_Shetty | 618 | 618 |
| **Total** | **3,109** | **3,108** |

### 3.2 Audit-routing table

| Decision | Count | Meaning |
|---|---:|---|
| Accepted | 3,109 | Usable after standard preprocessing |
| Needs review | 27 | Borderline/manual inspection candidates |
| Rejected | 34 | Excluded from first cleaned stage |
| Blocked | 0 | Unreadable/missing/unsupported |
| **Total** | **3,170** | Raw5 audio files audited |

### 3.3 Manual exclusion

`Eric_Thomas__0175.wav` was rejected on cleaned re-audit and manually confirmed as music-only. It is excluded from the final training root and preserved in an excluded folder for traceability.

---

## 4. Methods section

### 4.1 Agentic preprocessing components

| Component | Role |
|---|---|
| `DatasetAuditorAgent` | Non-destructive audio scan and decision routing |
| `ManifestBuilderAgent` | Creates accepted/needs-review/rejected/blocked manifests |
| `DatasetBuilderAgent` | Builds cleaned 16 kHz mono WAV training copies |
| `TinyAudioTriageAgent` | Planned triage model for target speaker / other speaker / event detection |
| `ResultDiagnosisAgent` | Planned post-training diagnosis layer |

### 4.2 Current model setup

| Item | Value |
|---|---|
| Model | TinyAudioCNN + ExitNet |
| Exits | 3 |
| Tap blocks | `[1, 3]` |
| Exit hint | disabled |
| Classes | 5 |
| Feature type | log-mel |
| Mel bins | 64 |
| Segment length | 1.0 s |
| Hop | 0.5 s |
| Training epochs | 40 |
| Greedy threshold `tau` | 0.95 |
| Calibration | per-exit temperature scaling |

---

## 5. Results section structure

Use these tables in the report.

### 5.1 Main current result

| Evaluation mode | Accuracy | Avg exit depth | Avg windows used | Windows saved | Compute saved |
|---|---:|---:|---:|---:|---:|
| Segment greedy policy | 96.83% | 2.089 | — | — | 52.56% |
| Full-clip greedy aggregation | 99.57% | 2.089 | 8.651 / 8.651 | 0.00% | 0.00% |
| Depth×Time clip greedy | 98.93% | 2.092 | 2.088 / 8.651 | 75.87% | 75.82% |

### 5.2 Per-exit table

| Exit | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Exit 1 | 65.62% | 64.04% | 64.36% |
| Exit 2 | 92.40% | 92.29% | 92.37% |
| Exit 3 | 97.60% | 97.56% | 97.59% |

### 5.3 Clip-level interpretation

Full-clip aggregation makes only 2 clip mistakes. Depth×Time makes 5 clip mistakes while using only 24.13% of available windows.

---

## 6. Required next documentation update

After running the matched uncleaned baseline, update all documents with:

```text
raw5_uncleaned_3exit_greedy
raw5_agentic_cleaned_3exit_greedy_final
```

Do not mix in old clean-stage branch results.
