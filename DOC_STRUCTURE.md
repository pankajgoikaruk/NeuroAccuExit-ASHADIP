# ASHADIP / NeuroAccuExit – Documentation Structure for `kexit-greedy-hint-gunshot`

This document defines the recommended documentation structure for the **`kexit-greedy-hint-gunshot`** branch.

This branch is the gunshot-focused continuation of the generic K-exit / C-class pipeline. The main distinction is now:

- **historical `v0.1.6` tag** = frozen 3-exit greedy moth baseline
- **`kexit-greedy-hint` branch** = generic K-exit / hint branch validated on the moth task
- **`kexit-greedy-hint-gunshot` branch** = gunshot binary adaptation using the same core architecture, updated preprocessing, and new four-run greedy comparison on `gunshot` vs `non_gunshot`

This document should be used for the updated thesis mini-book / branch-level write-up of the gunshot study.

---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain why the gunshot branch was created and what changed relative to the earlier branch.

### Content to include

- **Domain**
  - acoustic gunshot detection
  - binary classification: `gunshot` vs `non_gunshot`
- **Motivation**
  - the generic K-exit branch already supported dynamic 3-exit and 5-exit configurations
  - however, the preprocessing path was still effectively moth-oriented
  - a fair cross-domain evaluation required reusing the same architecture on a new binary dataset without redesigning the model
- **High-level idea**
  - raw mixed-length audio → inventory scan → cleaning/resampling → file-level split → log-mel features → dynamic K-exit TinyAudioCNN + ExitNet → calibration → greedy thresholding → segment and clip policy evaluation
- **Main contribution of this branch**
  - successful adaptation of the generic K-exit pipeline to a new binary gunshot dataset
  - clear evidence that hint passing is **task-dependent** and not universally beneficial

---

## Chapter 2 – Dataset and Preprocessing

**Goal:** Describe the new data layout and preprocessing policy.

### 2.1 Raw data layout

```text
data2/
  gunshot/
  non_gunshot/
```

### 2.2 Data characteristics

Current dataset summary:

- total files: **1712**
- gunshot: **896**
- non_gunshot: **816**

Duration summary:

| Class | Files | Min sec | Median sec | Mean sec | Max sec |
|---|---:|---:|---:|---:|---:|
| gunshot | 896 | 0.4988 | 1.9461 | 1.5271 | 44.0000 |
| non_gunshot | 816 | 0.5151 | 4.9998 | 7.4386 | 994.3688 |

Shorter than 1 second:
- gunshot: **250**
- non_gunshot: **3**

### 2.3 Preprocessing policy

Important decisions to document:

- raw files are **not** globally forced to fixed 5 s / 10 s / 30 s clips
- audio is standardized to:
  - mono
  - 16 kHz
  - optional bandpass
- training windows are created directly from the cleaned files
- current main setting:
  - `segment_sec = 1.0`
  - `hop = 0.5`

### 2.4 Handling short and long files

Document the updated preprocessing behavior:

- short clips can be kept if `duration >= min_keep_sec`
- current setting:
  - `MinKeepSec = 0.25`
- very long non-gunshot recordings are capped to avoid dominating the segment count:
  - `MaxSegmentsPerFileGunshot = 0`
  - `MaxSegmentsPerFileNonGunshot = 5`

### 2.5 File-level split

Explain clearly:

- splitting is by **source file**, not by random segment
- this prevents leakage between train / val / test

Current segment split:

| Split | Gunshot | Non-gunshot | Total |
|---|---:|---:|---:|
| train | 1295 | 2323 | 3618 |
| val | 267 | 486 | 753 |
| test | 259 | 515 | 774 |

---

## Chapter 3 – System Overview

**Goal:** Give a top-down view of the current gunshot pipeline.

### End-to-end pipeline

1. raw audio in `data2/gunshot/` and `data2/non_gunshot/`
2. `scripts/prep_segments.py`
   - audio inventory summary
   - cleaning/resampling
   - file-level split
   - `segments.csv`
3. `scripts/extract_features.py`
   - log-mel features
   - `feat_relpath` update
   - short-feature handling via padded extraction
4. `data/datasets.py`
   - PyTorch dataset construction
5. `adapters/audio_adapter.py`
   - dynamic backbone taps
6. `models/exit_net.py`
   - dynamic exits + optional hint
7. `training/train.py`
8. `training/calibrate.py`
9. `training/thresholds_offline.py`
10. `scripts/policy_test.py`
11. `scripts/clip_policy_test.py`
12. `scripts/summarize_run.py`
13. `scripts/analyse_run.py`
14. `scripts/profile_latency.py`
15. `scripts/run_reports.ps1`

---

## Chapter 4 – Code Changes Specific to the Gunshot Branch

**Goal:** List the most important implementation changes and why they matter.

### Core updated files

#### `scripts/prep_segments.py`
Document that it now supports:

- auto-discovered labels
- mixed audio formats
- inventory CSV export
- `--min_keep_sec`
- `--max_segments_per_file_gunshot`
- `--max_segments_per_file_non_gunshot`
- shorter hashed cache filenames to avoid Windows path-length issues

#### `scripts/run_full.ps1`
Document that it now supports:

- `DataRoot = "data2"`
- gunshot-oriented cache naming
- `MinKeepSec`
- gun/non-gun per-file caps
- CLI hint override
- run metadata for the new preprocessing settings

#### Feature extraction path
Document that the pipeline uses padded short-feature extraction to avoid missing feature paths for short gunshot clips.

---

## Chapter 5 – Model Architecture

**Goal:** Explain the dynamic architecture, which remains the same conceptually as the earlier branch.

### 5.1 Dynamic backbone taps

Validated settings:

- `tap_blocks=(1,3)` → **3 exits**
- `tap_blocks=(1,2,3,4)` → **5 exits**

### 5.2 Generic ExitNet

- one head per tap
- one final head
- total exits = `len(tap_blocks) + 1`

### 5.3 Optional sequential hint passing

- later exits can consume a hint derived from the previous exit
- hint is toggled via:
  - `-ExitHint "true"`
  - `-ExitHint "false"`

### 5.4 Important interpretation for this branch

This branch does **not** show that hint passing always helps.  
Instead, it shows:

- hint passing remains a real architectural mechanism
- but its usefulness depends on the task and temporal regime

---

## Chapter 6 – Evaluation Protocol

**Goal:** Define the evaluation levels used in the current gunshot study.

### 6.1 Per-exit test evaluation

Report:

- exit1 accuracy
- exit2 accuracy
- exit3 accuracy
- exit4 accuracy
- exit5 accuracy

### 6.2 Segment-level greedy policy

Report:

- policy accuracy
- avg exit depth
- exit mix
- flip-any rate
- avg flip count
- exit consistency

### 6.3 Clip-level evaluation

Two modes:

- **full-clip baseline**
- **Depth×Time early stopping**

Report:

- clip accuracy
- processed/used-window segment accuracy
- avg windows used
- windows saved (%)
- avg compute units
- avg depth per used window
- compute saved (%)
- flip-rate
- exit consistency

---

## Chapter 7 – Validated Run Settings

**Goal:** Record the exact reusable commands for the four current runs.

### 7.1 3-exit no-hint (`gs3`)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs3" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy `
  -CacheId "s1_h05_t13_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0
```

### 7.2 3-exit hint (`gs3_hint`)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs3_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy `
  -CacheId "s1_h05_t13_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0 `
  -ExitHint "true"
```

### 7.3 5-exit no-hint (`gs5`)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs5" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy `
  -CacheId "s1_h05_t1234_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0
```

### 7.4 5-exit hint (`gs5_hint`)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs5_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy `
  -CacheId "s1_h05_t1234_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0 `
  -ExitHint "true"
```

---

## Chapter 8 – Main Results

**Goal:** Present the four-run gunshot comparison clearly.

### 8.1 Per-exit test accuracy

| Metric | `gs3` | `gs3_hint` | Δ (Hint−No-Hint) | `gs5` | `gs5_hint` | Δ (Hint−No-Hint) |
|---|---:|---:|---:|---:|---:|---:|
| Exit1 accuracy | 0.8269 | 0.8165 | -0.0104 | 0.8049 | 0.8140 | +0.0091 |
| Exit2 accuracy | 0.9380 | 0.9432 | +0.0052 | 0.9057 | 0.8915 | -0.0142 |
| Exit3 accuracy | 0.9587 | 0.9548 | -0.0039 | 0.9509 | 0.9496 | -0.0013 |
| Exit4 accuracy | — | — | — | 0.9651 | 0.9561 | -0.0090 |
| Exit5 accuracy | — | — | — | 0.9599 | 0.9625 | +0.0026 |

### 8.2 Segment-level greedy policy

| Metric | `gs3` | `gs3_hint` | Δ | `gs5` | `gs5_hint` | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Policy accuracy | 0.9548 | 0.9535 | -0.0013 | 0.9587 | 0.9561 | -0.0026 |
| Avg exit depth | 1.801 | 1.810 | +0.009 | 2.265 | 2.288 | +0.023 |
| Flip-any rate | 0.1628 | 0.1809 | +0.0181 | 0.2080 | 0.1744 | -0.0336 |
| Avg flip count | 0.1731 | 0.2003 | +0.0272 | 0.2532 | 0.2054 | -0.0478 |
| Exit consistency | 0.9935 | 0.9987 | +0.0052 | 0.9987 | 0.9910 | -0.0077 |

### 8.3 Full-clip baseline

| Setting | Clip acc | Processed-win acc | Avg windows used | Avg compute units | Avg depth per used window |
|---|---:|---:|---:|---:|---:|
| `gs3` | 0.9844 | 0.9548 | 3.012 / 3.012 | 5.424 | 1.801 |
| `gs3_hint` | 0.9650 | 0.9535 | 3.012 / 3.012 | 5.451 | 1.810 |
| `gs5` | 0.9767 | 0.9587 | 3.012 / 3.012 | 6.821 | 2.265 |
| `gs5_hint` | 0.9689 | 0.9561 | 3.012 / 3.012 | 6.891 | 2.288 |

### 8.4 Depth×Time

| Metric | `gs3` | `gs3_hint` | Δ | `gs5` | `gs5_hint` | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Clip accuracy | 0.9844 | 0.9650 | -0.0194 | 0.9728 | 0.9650 | -0.0078 |
| Used-window segment accuracy | 0.9558 | 0.9512 | -0.0046 | 0.9578 | 0.9551 | -0.0027 |
| Avg windows used | 1.759 / 3.012 | 1.755 / 3.012 | -0.004 / 3.012 | 1.751 / 3.012 | 1.732 / 3.012 | -0.019 / 3.012 |
| Windows saved | 41.60% | 41.73% | +0.13 pp | 41.86% | 42.51% | +0.65 pp |
| Avg compute units | 3.339 | 3.401 | +0.062 | 4.218 | 4.128 | -0.090 |
| Avg depth per used window | 1.898 | 1.938 | +0.040 | 2.409 | 2.384 | -0.025 |
| Compute saved | 38.45% | 37.62% | -0.83 pp | 38.16% | 40.09% | +1.93 pp |
| Flip-rate | 0.1770 | 0.1863 | +0.0093 | 0.2089 | 0.1798 | -0.0291 |
| Exit consistency | 0.9978 | 0.9978 | +0.0000 | 0.9978 | 0.9933 | -0.0045 |

---

## Chapter 9 – Interpretation

**Goal:** State clearly what the new gunshot comparison means.

### Main conclusions

1. The gunshot adaptation is successful.
   - the same generic K-exit pipeline now runs on a new binary dataset
   - preprocessing, feature extraction, calibration, thresholding, and reporting all work end-to-end

2. The current best practical model is **`gs3`**.
   - highest clip accuracy
   - strongest overall deployment-style result in the current branch

3. Hint passing is **not universally beneficial**.
   - in `gs3_hint`, exit2 improved slightly
   - but final decision quality and clip accuracy worsened

4. The deeper 5-exit system is still useful.
   - `gs5` remains the best deeper no-hint reference
   - `gs5_hint` is mixed, with small efficiency gains but weaker overall clip accuracy

5. The current novelty result is therefore **task-dependent**.
   - hint remains behaviorally meaningful
   - but this branch does not support a strong claim that hint always improves intermediate prediction

### Recommended reviewer-safe interpretation

> The gunshot binary study shows that sequential hint passing is task-dependent rather than universally beneficial. Although hint passing can change exit behavior and occasionally improve intermediate or final-exit metrics, it does not improve the overall greedy accuracy-efficiency trade-off on this short, binary gunshot dataset.

---

## Chapter 10 – Current Scope

**Goal:** State clearly what is included and excluded in this branch.

### Included

- gunshot binary classification
- greedy segment policy
- greedy clip-wise Depth×Time policy
- dynamic 3-exit and 5-exit validation
- optional sequential hint passing
- mixed-length audio preprocessing
- file-level split and segment capping
- dynamic profiling and reporting

### Not yet solved

- a hint configuration that clearly beats the no-hint gunshot baseline
- multiclass gun-type classification
- hierarchical gunshot → gun-type / non-gun → subtype pipeline
- controlled experiments with smaller hop sizes and more temporal evidence
- broader robustness testing across more negative classes

---

## Chapter 11 – Limitations and Outlook

**Goal:** State clearly what this branch establishes and what should come next.

### Why this branch matters

This branch establishes:

- a reusable gunshot-ready preprocessing path for the K-exit architecture
- a fair 3-exit vs 5-exit comparison on a new binary audio domain
- a clear no-hint vs hint comparison under the same preprocessing policy
- a realistic negative result showing the boundary condition of the current novelty idea

### Best current recommendation

- **best practical model:** `gs3`
- **best deeper no-hint reference:** `gs5`
- **mixed experimental hint result:** `gs5_hint`
- **clear negative hint result:** `gs3_hint`

### Best next experiment

The recommended next controlled experiment is:

- keep `segment_sec = 1.0`
- reduce `hop` from `0.5` to `0.25`
- compare:
  - `gs3_h025`
  - `gs3_hint_h025`

This directly tests whether hint becomes more useful when the model has richer temporal evidence per clip.

---

## Chapter 12 – How to Use This Document

Use this file as the master structure for the branch-level documentation / thesis mini-book for the gunshot study.

Keep the distinction clear:

- `v0.1.6` = historical frozen moth baseline
- `kexit-greedy-hint` = older generic K-exit moth branch
- `kexit-greedy-hint-gunshot` = current gunshot binary adaptation

When the implementation changes, update:

- the code
- `README.md`
- `DOC_STRUCTURE.md`

so the documentation stays aligned with the validated gunshot pipeline and findings.
