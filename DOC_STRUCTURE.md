# ASHADIP / NeuroAccuExit – Documentation Structure for `kexit-dev`

This document defines the recommended documentation structure for the **`kexit-dev` branch**, which refactors the historical `v0.1.6` code line into a **generic K-exit / C-class** audio early-exit pipeline.

The key distinction is:

- **`v0.1.6` tag** = frozen historical **3-exit greedy baseline**
- **`kexit-dev` branch** = reusable **K-exit / C-class** refactor validated on both **3 exits** and **5 exits**

This document should be used for the updated mini-book / Overleaf write-up of the refactored branch, not for the frozen historical baseline.

---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain why the `kexit-dev` refactor was created and what it changes.

### Content to include

- **Domain**
  - acoustic classification of moth wingbeat audio
  - binary classification: male vs female
- **Motivation**
  - the original code line was hardcoded for 3 exits
  - later research required comparing 3-exit and 5-exit settings fairly
  - a reusable K-exit implementation was needed to support future early-exit variants without rewriting the full stack
- **High-level idea**
  - raw audio → segmentation → log-mel features → dynamic K-exit TinyAudioCNN + ExitNet → calibration → greedy threshold selection → segment and clip policy evaluation
- **Main contribution of the branch**
  - successful refactor from a fixed 3-exit path to a generic **K-exit / C-class** path

---

## Chapter 2 – System Overview

**Goal:** Give a top-down view of the new dynamic pipeline.

### End-to-end pipeline

1. raw WAV data in `data/male/` and `data/female/`
2. `scripts/prep_segments.py`
   - create `segments.csv`
   - define train / val / test split
3. `scripts/extract_features.py`
   - compute log-mel `.npy` features
   - update `feat_relpath`
4. `data/datasets.py`
   - build PyTorch datasets and loaders
5. `adapters/audio_adapter.py`
   - define dynamic `TinyAudioCNN` with configurable `tap_blocks`
6. `models/exit_net.py`
   - define generic `ExitNet` with one head per tap plus one final head
7. `training/train.py`
   - train dynamic K-exit model and save `ckpt/best.pt`
8. `training/calibrate.py`
   - fit per-exit temperatures and save `temperature.json`
9. `training/thresholds_offline.py`
   - select greedy `tau`
   - save `thresholds.json`
10. `scripts/policy_test.py`
   - segment-level greedy policy evaluation
11. `scripts/clip_policy_test.py`
   - clip-wise full baseline and Depth×Time evaluation
12. `scripts/summarize_run.py`
   - consolidate artifacts into `summary.json`
13. `scripts/analyse_run.py`
   - confusion matrices, ROC, plots, analysis summary
14. `scripts/profile_latency.py`
   - latency profiling and on-device summary
15. `scripts/run_reports.ps1`
   - report regeneration and LaTeX outputs
16. `scripts/compare_variants.py`, `scripts/variants_to_latex.py`, `scripts/ondevice_to_latex.py`, `scripts/analysis_to_latex.py`
   - variant comparison and LaTeX table generation

### Configuration and run orchestration

- `configs/audio_moth.yaml`
- `scripts/run_full.ps1`
- `scripts/run_reports.ps1`

---

## Chapter 3 – Data and Preprocessing

**Goal:** Document raw data layout, preprocessing, segmentation, and split logic.

### Main points

- raw input folders:
  - `data/male/*.wav`
  - `data/female/*.wav`
- audio preprocessing:
  - resampling to 16 kHz
  - mono conversion
  - band-pass filtering
  - silence removal
- segmentation:
  - default window = 1.0 s
  - default hop = 0.5 s
- manifest fields in `segments.csv`:
  - `wav_relpath`
  - `label`
  - `start`
  - `duration`
  - `split`
  - `feat_relpath`

### Important implementation note

Feature caches should now remain separated by tap configuration when using the main runner, so 3-exit and 5-exit runs are not accidentally mixed.

---

## Chapter 4 – Features and Representation

**Goal:** Explain how audio becomes model-ready features.

### Main points

- log-mel spectrogram representation
- parameters:
  - `n_mels`
  - `n_fft`
  - `win_ms`
  - `hop_ms`
  - optional CMVN
- `scripts/extract_features.py`
  - creates unique per-segment feature files
- `data/datasets.py`
  - loads `(1, M, T)` tensors
  - builds train / val / test loaders

---

## Chapter 5 – Model Architecture

**Goal:** Describe the dynamic K-exit TinyAudioCNN + ExitNet used in `kexit-dev`.

### Backbone

- `TinyAudioCNN`
- 5 convolutional blocks in the refactored branch
- configurable early-exit tap points via `tap_blocks`
- final representation after the last block

### ExitNet

- builds one classifier per tap feature
- adds one final classifier at the deepest layer
- total exits = `len(tap_blocks) + 1`

### Validated configurations

#### 3-exit configuration

- `tap_blocks=(1,3)`
- exits:
  - `exit1` from block 1
  - `exit2` from block 3 tap
  - `exit3` final

#### 5-exit configuration

- `tap_blocks=(1,2,3,4)`
- exits:
  - `exit1`
  - `exit2`
  - `exit3`
  - `exit4`
  - `exit5` final

### Interpretation

- shallow exits are cheaper but weaker
- deeper exits are more accurate but more expensive
- 5-exit adds stronger late decision points, especially exit4 and exit5

---

## Chapter 6 – Training, Calibration, and Threshold Selection

**Goal:** Explain model fitting and greedy policy calibration under the dynamic K-exit path.

### 6.1 Training

- train multi-exit model with per-exit losses
- validation accuracy used for checkpoint selection
- save `metrics.json` and `ckpt/best.pt`
- dynamic loops used instead of hardcoded `range(3)`

### 6.2 Temperature calibration

- fit one scalar temperature per exit on validation data
- save `temperature.json`
- validated outputs:
  - 3 temperatures for the 3-exit run
  - 5 temperatures for the 5-exit run

### 6.3 Greedy threshold selection

- sweep candidate `tau` values on validation set
- choose threshold using macro-F1 then accuracy
- save `thresholds.json`

### Validated thresholds

- 3-exit run: `tau = 0.95`
- 5-exit run: `tau = 0.92`

---

## Chapter 7 – Evaluation, Metrics, and Reporting

**Goal:** Document the evaluation outputs used in tables and graphs.

### 7.1 Segment policy test (`scripts/policy_test.py`)

Outputs:

- `policy_results.json`
- printed metrics:
  - `Policy test accuracy`
  - `Avg exit depth`
  - `Exit mix`
  - `Flip-any rate`
  - `Avg flip count`
  - `Exit consistency`

### 7.2 Clip policy test (`scripts/clip_policy_test.py`)

This script performs clip-wise window ordering and optional window skipping.

#### Full-clip baseline

Reports:

- `clip_accuracy`
- `segment_accuracy_over_processed_windows`
- fixed-position diagnostics
- `avg_compute_units_sum_depth_over_used_windows`
- `avg_depth_per_used_window`
- `exit_mix_over_used_windows`

Saved file:

- `clip_policy_results_full.json`

#### Depth×Time

Reports:

- `clip_accuracy`
- `segment_accuracy_over_used_windows`
- `avg_windows_used`
- `windows_saved_pct`
- `avg_compute_units_sum_depth_over_used_windows`
- `avg_depth_per_used_window`
- `compute_saved_pct`
- `flip_rate_over_used_windows`
- `exit_consistency_taken_vs_final_over_used_windows`
- `exit_mix_over_used_windows`

Saved files:

- `clip_policy_results_time.json`
- `clip_policy_results.json`
- `windows_used_hist.json`

### 7.3 Summary, analysis, and profiling

- `scripts/summarize_run.py`
  - `summary.json`
- `scripts/analyse_run.py`
  - confusion matrices, ROC curves, `analysis_run.json`
- `scripts/profile_latency.py`
  - `profiling.json`
  - dynamic latency and FLOPs summaries

### 7.4 Dynamic reporting support

The refactor also updated:

- `compare_variants.py`
- `variants_to_latex.py`
- `ondevice_to_latex.py`
- `analysis_to_latex.py`

so reporting now supports both 3-exit and 5-exit rows.

---

## Chapter 8 – Validated Results and Comparative Findings

**Goal:** Present the actual findings from the 3-exit vs 5-exit comparison.

### 8.1 Segment-level greedy results

#### 3-exit (`tap_blocks=(1,3)`)

- policy accuracy: **0.9754**
- avg exit depth: **1.982**
- exit mix:
  - e1 = 0.3631
  - e2 = 0.2923
  - e3 = 0.3446
- flip-any rate: **0.1785**
- exit consistency: **1.0000**

#### 5-exit (`tap_blocks=(1,2,3,4)`)

- policy accuracy: **0.9846**
- avg exit depth: **2.449**
- exit mix:
  - e1 = 0.3815
  - e2 = 0.1323
  - e3 = 0.2308
  - e4 = 0.1662
  - e5 = 0.0892
- flip-any rate: **0.2215**
- exit consistency: **0.9969**

### Main takeaway

The 5-exit model improves segment-level policy accuracy, but becomes deeper and slightly less stable.

### 8.2 Per-exit test accuracy

#### 3-exit

- exit1 = **0.8338**
- exit2 = **0.9385**
- exit3 = **0.9754**

#### 5-exit

- exit1 = **0.8400**
- exit2 = **0.8400**
- exit3 = **0.9600**
- exit4 = **0.9846**
- exit5 = **0.9877**

### Main takeaway

The strongest advantage of 5-exit comes from the additional deeper exits, not from uniformly improving all intermediate exits.

### 8.3 Depth×Time comparison

#### 3-exit

- clip accuracy: **1.0000**
- used-window segment accuracy: **0.9778**
- avg windows used: **2.045 / 14.773**
- windows saved: **86.15%**
- avg compute units: **5.364**
- compute saved: **81.68%**

#### 5-exit

- clip accuracy: **1.0000**
- used-window segment accuracy: **0.9778**
- avg windows used: **2.045 / 14.773**
- windows saved: **86.15%**
- avg compute units: **7.045**
- compute saved: **80.53%**

### Main takeaway

Under the current greedy Depth×Time rule, the 5-exit model does **not** improve clip-level efficiency. It matches clip accuracy and windows used, but uses more compute per used window.

### 8.4 Retuning findings for the 5-exit time policy

Tested settings:

- `time_conf=0.97, time_stable_k=2, time_margin=0.00`
- `time_conf=0.95, time_stable_k=3, time_margin=0.00`
- `time_conf=0.97, time_stable_k=3, time_margin=0.00`
- `time_conf=0.97, time_stable_k=3, time_margin=0.05`

#### Observed outcome

- increasing `time_conf` from 0.95 to 0.97 had **no practical effect** in the tested runs
- adding `time_margin=0.05` also had **no practical effect**
- increasing `time_stable_k` from 2 to 3:
  - increased avg windows used from **2.045** to **3.000**
  - reduced windows saved from **86.15%** to **79.69%**
  - reduced compute saved from **80.53%** to **73.12%**
  - only slightly reduced flip-rate

#### Current best 5-exit Depth×Time setting

- `time_conf=0.95`
- `time_stable_k=2`
- `time_min_windows=2`
- `time_margin=0.00`

---

## Chapter 9 – Reproducibility and Run Management

**Goal:** Describe how to rerun and compare experiments safely.

### Main runner

- `scripts/run_full.ps1`

Important flags now include:

- `-Variant`
- `-Policy`
- `-SegmentSec`
- `-HopSec`
- `-NMels`
- `-TapBlocks`
- `-RunClipPolicy`
- `-TimeConf`
- `-TimeStableK`
- `-TimeMinWindows`
- `-EvalFixedKWindows`
- `-TimeMargin`

### Typical 3-exit command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_dev" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy
```

### Typical 5-exit command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_dev" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy
```

### Important run-management note

The refactor updated the run/report scripts so that:

- tap configuration is recorded in metadata
- reporting regenerates safely for both 3-exit and 5-exit runs
- schema conflicts in old CSV logs fall back to `_kexit.csv` outputs instead of corrupting older files

---

## Chapter 10 – Limitations and Outlook

**Goal:** State clearly what this branch does and does not yet solve.

### Current scope

- binary classification only
- greedy segment policy
- greedy clip-wise Depth×Time policy
- validated 3-exit and 5-exit settings
- generic K-exit / C-class implementation

### What is not yet solved

- a 5-exit time policy that clearly beats the 3-exit baseline in clip-level efficiency
- EA policy integration and retuning in this branch
- hint-passing results in this branch
- multiclass experiments

### Why this branch matters

This branch establishes:

- a reusable implementation foundation for future early-exit work
- a fair 3-exit vs 5-exit comparison framework
- a clean bridge from the historical `v0.1.6` baseline to later K-exit, EA, and hint-based variants

---

## How to Use This Document

Use this file as the master structure for the updated branch-level documentation / thesis mini-book.

Keep the distinction clear:

- `v0.1.6` = historical frozen 3-exit greedy baseline
- `kexit-dev` = generic K-exit refactor validated on both 3 and 5 exits

When the implementation changes, update both:

- the code
- this structure file

so the documentation remains aligned with the actual validated pipeline and findings.
