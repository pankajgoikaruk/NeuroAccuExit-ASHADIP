# ASHADIP / NeuroAccuExit – Documentation Structure for `kexit-greedy-gunshot-segment`

This document defines the recommended documentation structure for the **`kexit-greedy-gunshot-segment`** branch.

This branch is the generic audio-segmentation continuation of the earlier K-exit work. The main distinction is now:

- **historical `v0.1.6` tag** = frozen 3-exit greedy moth baseline
- **later K-exit / hint branches** = dynamic 3-exit and 5-exit validation on earlier tasks
- **`kexit-greedy-gunshot-segment`** = generic segmentation pipeline + current 10-class acoustic run on mixed-length audio

This document should be used for the updated thesis mini-book / branch-level write-up of the new segmentation and multiclass study.

---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain why the segmentation branch was created and what changed relative to the earlier branches.

### Content to include

- **Domain**
  - multiclass acoustic event/sound classification
  - current validated run uses 10 classes
- **Motivation**
  - the generic K-exit architecture already existed
  - however, preprocessing and dataset preparation were still too task-specific
  - a fair continuation required a reusable pipeline for mixed-length, mixed-format audio without redesigning the model
- **High-level idea**
  - raw mixed-length audio → inventory scan → cleaning/resampling → file-level split → log-mel features → dynamic K-exit TinyAudioCNN + ExitNet → calibration → greedy thresholding → segment-policy evaluation → clip-level greedy evaluation
- **Main contribution of this branch**
  - successful end-to-end generic segmentation pipeline
  - successful transfer of the K-exit model to a new 10-class dataset
  - validated full-clip sequential greedy and Depth×Time clip-greedy evaluation on the same branch
  - clear evidence that the current bottleneck is now data balance and segment selection, not pipeline failure

---

## Chapter 2 – Dataset and Preprocessing

**Goal:** Describe the new data layout and preprocessing policy.

### 2.1 Raw data layout

```text
data2/
  car_crash/
  conversation/
  engine_idling/
  fireworks/
  gun_shot/
  rain/
  road_traffic/
  scream/
  thunderstorm/
  wind/
```

### 2.2 Data characteristics

Current validated dataset summary:

- total files: **1011**
- total classes: **10**

Duration summary:

| Class | Files | Min sec | Median sec | Mean sec | Max sec |
|---|---:|---:|---:|---:|---:|
| car_crash | 92 | 1.0000 | 2.3064 | 2.9281 | 10.8639 |
| conversation | 81 | 1.4800 | 3.0000 | 14.6656 | 994.3688 |
| engine_idling | 65 | 1.0376 | 8.0000 | 11.3280 | 36.0000 |
| fireworks | 14 | 1.3095 | 21.0000 | 108.0361 | 770.2427 |
| gun_shot | 187 | 0.4988 | 1.5084 | 1.9196 | 44.0000 |
| rain | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |
| road_traffic | 121 | 3.9998 | 4.9999 | 5.4463 | 60.0000 |
| scream | 151 | 0.5151 | 1.5020 | 1.7642 | 6.5912 |
| thunderstorm | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |
| wind | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |

### 2.3 Preprocessing policy

Important decisions to document:

- raw files are not globally forced to one pre-published duration before the main pipeline
- audio is standardized to:
  - mono
  - 16 kHz
  - optional bandpass
- current validated setting:
  - `segment_sec = 1.0`
  - `hop = 0.5`
  - `input_mode = segment`
  - `split_unit = file`
- the branch can also support already-ready datasets and optional segment WAV export

### 2.4 Handling short and long files

Document the updated preprocessing behavior:

- short clips can be kept if `duration >= min_keep_sec`
- current validated setting:
  - `MinKeepSec = 0.25`
- very long files are currently limited through a global file-level cap:
  - `MaxSegmentsPerFileDefault = 5`

### 2.5 File-level split

Explain clearly:

- splitting is by **source file**, not by random segment
- this prevents leakage between train / val / test

Current validated file split:

| Split | Files |
|---|---:|
| train | 707 |
| val | 152 |
| test | 152 |

Current validated segment split:

| Split | Segments |
|---|---:|
| train | 2536 |
| val | 551 |
| test | 555 |

### 2.6 Rejected segments

Current validated rejection summary:

| Reason | Count |
|---|---:|
| `cap_dropped` | 6249 |
| `silent_window` | 1786 |

This section should explicitly discuss why `cap_dropped` is now one of the major research issues in this branch.

---

## Chapter 3 – System Overview

**Goal:** Give a top-down view of the current segmentation branch.

### End-to-end pipeline

1. raw audio in `data2/<class_name>/`
2. `scripts/prep_segments.py`
   - inventory summary
   - cleaning/resampling
   - file-level split
   - optional exported WAV segments
   - `segments.csv`
3. `scripts/extract_features.py`
   - log-mel feature extraction
   - short-feature handling
   - `feat_relpath` update
4. `data/datasets.py`
   - PyTorch dataset construction
5. `adapters/audio_adapter.py`
   - dynamic backbone taps
6. `models/exit_net.py`
   - generic exits
7. `training/train.py`
8. `training/calibrate.py`
9. `training/thresholds_offline.py`
10. `scripts.policy_test.py`
11. `scripts.clip_policy_test.py`
12. `scripts.summarize_run.py`
13. `scripts.analyse_run.py`
14. `scripts.profile_latency.py`
15. `scripts.run_reports.ps1`

---

## Chapter 4 – Code Changes Specific to the Segmentation Branch

**Goal:** List the most important implementation changes and why they matter.

### Core updated files

#### `scripts/prep_segments.py`
Document that it now supports:

- auto-discovered labels
- mixed audio formats
- inventory CSV export
- generic segmentation from raw mixed-length files
- `--input_mode segment|ready`
- `--min_keep_sec`
- optional exported segment WAVs
- file-level split assignment
- current cap-based control of long files

#### `scripts/run_full.ps1`
Document that it now supports:

- `DataRoot = "data2"`
- `InputMode`
- `SplitUnit`
- `GroupMode`
- `MinKeepSec`
- `MaxSegmentsPerFileDefault`
- `ExportSegmentWavs`
- `RunClipPolicy`
- `ForceRebuild`
- run metadata for new preprocessing settings

#### `scripts/extract_features.py`
Document that it now supports:

- stable `feat_relpath` generation
- robust handling of short clips
- shorter feature filenames to avoid overwrite/path issues

#### Dynamic class handling
Document that training/evaluation now:

- can detect dataset class count at runtime
- currently override outdated config class count automatically

---

## Chapter 5 – Model Architecture

**Goal:** Explain the dynamic architecture, which remains conceptually continuous with earlier branches.

### 5.1 Dynamic backbone taps

Validated current setting:

- `tap_blocks=(1,2,3,4)` → **5 exits**

### 5.2 Generic ExitNet

- one head per tap
- one final head
- total exits = `len(tap_blocks) + 1`

### 5.3 Hint passing status

State clearly:

- this validated multiclass run is **greedy no-hint**
- hint passing remains part of the broader project history
- but the current validated result section in this branch is about the generic segmentation pipeline and the multiclass greedy baseline

### 5.4 Important interpretation for this branch

This branch should currently be interpreted as:

- a **generic preprocessing and segmentation branch first**
- a **multiclass greedy baseline branch second**
- now including validated clip-policy comparison on the same run

---

## Chapter 6 – Evaluation Protocol

**Goal:** Define the evaluation levels used in the current validated run.

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

Report the two validated clip-level policies:

- full-clip sequential greedy
- Depth×Time clip greedy

For each, report:

- clip accuracy
- segment accuracy over processed/used windows
- avg windows used
- windows saved
- avg compute units
- compute saved
- avg depth per used window
- flip-rate
- exit-consistency

---

## Chapter 7 – Validated Run Setting

**Goal:** Record the exact reusable command for the current stable run.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "kexit_greedy_gunshot_segment" `
  -Policy "greedy" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -NMels 64 `
  -TapBlocks "1,2,3,4" `
  -InputMode "segment" `
  -SplitUnit "file" `
  -GroupMode "none" `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 5 `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3 `
  -TimeMargin 0.0 `
  -ForceRebuild `
  -RunClipPolicy
```

---

## Chapter 8 – Main Results

**Goal:** Present the current stable multiclass baseline clearly.

### 8.1 Per-exit test accuracy

| Metric | Value |
|---|---:|
| Exit1 accuracy | 0.2613 |
| Exit2 accuracy | 0.4162 |
| Exit3 accuracy | 0.5946 |
| Exit4 accuracy | 0.7027 |
| Exit5 accuracy | 0.6739 |

### 8.2 Segment-level greedy policy

| Metric | Value |
|---|---:|
| Policy accuracy | 0.6739 |
| Avg exit depth | 4.589 |
| Flip-any rate | 0.8198 |
| Avg flip count | 1.2793 |
| Exit consistency | 1.0000 |

### 8.3 Exit mix

| Exit | Usage |
|---|---:|
| e1 | 0.0000 |
| e2 | 0.0649 |
| e3 | 0.0721 |
| e4 | 0.0721 |
| e5 | 0.7910 |

### 8.4 Threshold selection

| Metric | Value |
|---|---:|
| Best tau | 0.92 |
| Validation macro-F1 | 0.6708 |
| Validation accuracy | 0.7169 |

### 8.5 Full-clip sequential greedy

| Metric | Value |
|---|---:|
| Clip accuracy | 0.7829 |
| Segment acc over processed windows | 0.6739 |
| Avg windows used | 3.651 / 3.651 |
| Windows saved | 0.00% |
| Avg compute units | 16.757 |
| Compute saved | 0.00% |
| Avg depth per used window | 4.589 |
| Flip-rate | 0.8198 |
| Exit-consistency | 1.0000 |

### 8.6 Depth×Time clip greedy

| Metric | Value |
|---|---:|
| Clip accuracy | 0.7829 |
| Segment acc over used windows | 0.6209 |
| Avg windows used | 2.638 / 3.651 |
| Windows saved | 27.75% |
| Avg compute units | 11.993 |
| Compute saved | 28.43% |
| Avg depth per used window | 4.546 |
| Flip-rate | 0.8254 |
| Exit-consistency | 1.0000 |

### 8.7 Key structural observations

- Exit4 is currently stronger than Exit5:
  - Exit4 = **0.7027**
  - Exit5 = **0.6739**
- Depth×Time preserves the same clip accuracy as the full-clip baseline while reducing windows and compute.
- Rare classes such as `fireworks` remain the main failure point.

---

## Chapter 9 – Interpretation

**Goal:** State clearly what the new multiclass comparison means.

### Main conclusions

1. The generic segmentation adaptation is successful.
   - preprocessing, feature extraction, training, calibration, clip evaluation, and report generation all work end to end

2. The branch is no longer blocked by engineering instability.
   - the main issues are now:
     - class imbalance
     - data loss from capping
     - multiclass model quality

3. The current 5-exit greedy model is deep but not yet optimal.
   - most decisions are taken at Exit5
   - but Exit4 currently has the best per-exit test accuracy

4. Dynamic class-count handling already works in practice.
   - the runtime detects the true number of classes
   - this reduces manual reconfiguration burden

5. Validated clip-policy results now exist for this branch.
   - full-clip sequential greedy and Depth×Time clip greedy are no longer pending
   - the current result shows moderate savings with no clip-accuracy loss under Depth×Time

### Recommended reviewer-safe interpretation

> The `kexit-greedy-gunshot-segment` branch establishes a stable generic mixed-length audio preprocessing and K-exit greedy baseline for a 10-class acoustic dataset. The main limitations now arise from class balance, aggressive segment capping, and multiclass decision quality rather than from preprocessing or training-pipeline instability. The branch already shows that Depth×Time can reduce windows and compute without reducing clip accuracy on this dataset.

---

## Chapter 10 – Current Scope

**Goal:** State clearly what is included and excluded in this branch.

### Included

- generic mixed-length segmentation
- multiclass label auto-discovery
- file-level split
- robust feature extraction
- dynamic 5-exit greedy training
- calibration, thresholding, analysis, and profiling
- validated full-clip sequential greedy results
- validated Depth×Time clip-greedy results

### Not yet finalized

- finalized per-class capping strategy
- soft training-time balancing instead of preprocessing-time destruction
- final cleaned config defaults for fully dynamic class count
- stronger rare-class performance
- broader multiclass comparison beyond the first stable greedy run

---

## Chapter 11 – Limitations and Outlook

**Goal:** State clearly what this branch establishes and what should come next.

### Why this branch matters

This branch establishes:

- a reusable generic preprocessing path for the K-exit architecture
- successful transfer of the existing architecture to mixed-length multiclass audio
- a stable multiclass greedy baseline on the new branch
- a validated clip-policy comparison on the same branch
- a clear experimental basis for improving balancing and clip-level policies next

### Best current recommendation

- keep the generic segmentation design
- keep file-level splitting
- keep clip-policy evaluation
- improve capping and balancing next

### Best next experiment

The recommended next controlled experiment is:

1. reduce dependence on hard cap-dropped removal
2. compare:
   - no cap / softer cap
   - class-aware caps
   - training-time balancing
3. explicitly improve rare-class handling for `fireworks`
4. keep the same clip-policy evaluation so the compute/accuracy trade-off remains directly comparable

---

## Chapter 12 – How to Use This Document

Use this file as the master structure for the branch-level documentation / thesis mini-book for the segmentation study.

Keep the distinction clear:

- `v0.1.6` = historical frozen moth baseline
- earlier K-exit / hint branches = prior dynamic architecture studies
- `kexit-greedy-gunshot-segment` = current generic segmentation + multiclass greedy baseline branch with validated segment- and clip-level results

When the implementation changes, update:

- the code
- `README.md`
- `DOC_STRUCTURE.md`
- `APPENDIX.md`

so the documentation stays aligned with the validated pipeline and the most recent result set.
