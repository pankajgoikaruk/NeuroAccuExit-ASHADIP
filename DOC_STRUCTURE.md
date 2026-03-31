# ASHADIP v0.1.6 – Documentation & Mini-Book Structure

This document defines the recommended documentation structure for the **v0.1.6 baseline** of the ASHADIP project.

The purpose of this version is to document a **stable, reproducible 3-exit greedy audio baseline** that now includes:

- segment-level greedy policy evaluation
- clip-wise Depth×Time evaluation
- standardized printed metrics
- standardized JSON artifacts for later comparison tables and plots

---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain why ASHADIP v0.1.6 exists and what problem it solves.

### Content to include

- **Domain**
  - acoustic classification of moth wingbeat audio
  - binary classification: male vs female
- **Motivation**
  - build an efficient and reproducible early-exit baseline
  - study trade-offs between accuracy and computation
  - prepare a clean baseline for later 3-exit and 5-exit variants
- **High-level idea**
  - raw audio -> segmentation -> log-mel features -> 3-exit CNN -> calibration -> greedy threshold selection -> segment and clip policy evaluation
- **Main contribution of v0.1.6**
  - stable greedy baseline with both:
    - segment-policy metrics
    - clip-wise Depth×Time metrics

---

## Chapter 2 – System Overview

**Goal:** Give a top-down view of the v0.1.6 pipeline.

### End-to-end pipeline

1. raw WAV data in `data/male/` and `data/female/`
2. `scripts/prep_segments.py`
   - create `segments.csv`
   - define train/val/test split
3. `scripts/extract_features.py`
   - compute log-mel `.npy` features
   - update `feat_relpath`
4. `data/datasets.py`
   - build PyTorch datasets and loaders
5. `adapters/audio_adapter.py` + `models/exit_net.py`
   - define 3-exit network
6. `training/train.py`
   - train model and save `ckpt/best.pt`
7. `training/calibrate.py`
   - save `temperature.json`
8. `training/thresholds_offline.py`
   - select greedy `tau`
   - save `thresholds.json`
9. `scripts/policy_test.py`
   - segment-level greedy policy evaluation
   - save `policy_results.json`
10. `scripts/clip_policy_test.py`
    - clip-wise evaluation
    - full-clip baseline and Depth×Time
    - save clip policy JSON artifacts
11. `scripts/summarize_run.py`
    - consolidate metrics into `summary.json`
12. `scripts/analyse_run.py`, `scripts/profile_latency.py`, `scripts/run_reports.ps1`
    - analysis plots, latency profiling, tables, reports

### Configuration and run orchestration

- `configs/audio_moth.yaml`
- `scripts/run_full.ps1`
- `scripts/run_reports.ps1`

---

## Chapter 3 – Data and Preprocessing

**Goal:** Document the raw data layout, preprocessing, segmentation, and split logic.

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
  - builds train/val/test loaders

---

## Chapter 5 – Model Architecture

**Goal:** Describe the 3-exit TinyAudioCNN + ExitNet used in v0.1.6.

### Backbone

- `TinyAudioCNN`
- 3 convolutional stages
- tap features after early/mid blocks
- final global pooled representation

### ExitNet

- `exit1`: shallow
- `exit2`: intermediate
- `exit3`: final

### Interpretation

- `exit1` is cheapest and least accurate
- `exit3` is deepest and most accurate
- greedy policy selects the earliest confident exit

---

## Chapter 6 – Training, Calibration, and Threshold Selection

**Goal:** Explain model fitting and greedy policy calibration.

### 6.1 Training

- train multi-exit model with per-exit losses
- validation accuracy used for checkpoint selection
- save `metrics.json` and `ckpt/best.pt`

### 6.2 Temperature calibration

- fit per-exit scalar temperatures on validation data
- save `temperature.json`

### 6.3 Greedy threshold selection

- sweep candidate `tau` values on validation set
- choose threshold using macro-F1 / accuracy trade-off
- save `thresholds.json`

### Scope note

- v0.1.6 is **greedy-only** at policy level
- EA, hint passing, and K-exit logic belong to later versions

---

## Chapter 7 – Evaluation, Metrics, and Reporting

**Goal:** Document the exact evaluation outputs used in tables and graphs.

### 7.1 Segment policy test (`scripts/policy_test.py`)

Outputs:

- `policy_results.json`
- printed metrics:
  - `Policy test accuracy: x.xxxx (n_segments=...)`
  - `Avg exit depth`
  - `Exit mix`
  - `Flip-any rate`
  - `Exit consistency`

Saved fields include:

- `accuracy`
- `avg_exit_depth`
- `n_samples` / `n_segments`
- `exit_mix`
- `flip_any_rate`
- `avg_flip_count`
- `exit_consistency`

### 7.2 Clip policy test (`scripts/clip_policy_test.py`)

This script performs **clip-wise window ordering and optional window skipping**.

#### Full-clip baseline

Processes all windows in a clip and reports:

- `clip_accuracy`
- `segment_accuracy_over_processed_windows`
- `n_segments_processed_windows`
- fixed-position diagnostics
- `avg_compute_units_sum_depth_over_used_windows`
- `avg_depth_per_used_window`
- `exit_mix_over_used_windows`

Saved file:

- `clip_policy_results_full.json`

#### Depth×Time

Processes windows sequentially and stops early when clip-level confidence/stability conditions are met.

Reports:

- `clip_accuracy`
- `segment_accuracy_over_used_windows`
- `n_segments_used_windows`
- `avg_windows_used`
- `avg_windows_total`
- `windows_saved_pct`
- `avg_compute_units_sum_depth_over_used_windows`
- `avg_depth_per_used_window`
- `compute_saved_pct`
- `flip_rate_over_used_windows`
- `exit_consistency_taken_vs_final_over_used_windows`
- `exit_mix_over_used_windows`

Saved files:

- `clip_policy_results_time.json`
- `clip_policy_results.json` (legacy alias)
- `windows_used_hist.json`

### 7.3 Summary and experiments log (`scripts/summarize_run.py`)

`summary.json` should carry:

- base training/validation/test artifacts
- `policy_summary`
- `policy_results`
- `clip_policy_results_full`
- `clip_policy_results_time`

This keeps all metrics in one place for:

- result tables
- version comparison sheets
- plots

---

## Chapter 8 – Reproducibility and Run Management

**Goal:** Describe how to rerun and compare experiments safely.

### Main runner

- `scripts/run_full.ps1`

Important flags in v0.1.6:

- `-Variant`
- `-Policy`
- `-SegmentSec`
- `-HopSec`
- `-NMels`
- `-RunClipPolicy`
- `-TimeConf`
- `-TimeStableK`
- `-TimeMinWindows`
- `-EvalFixedKWindows`
- `-TimeMargin`

### Typical command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.1" `
  -Policy "greedy" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -NMels 64 `
  -RunClipPolicy `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3
```

### Run artifacts

Each run directory should contain the policy/clip JSON files needed for direct comparison.

---

## Chapter 9 – Limitations and Outlook

**Goal:** State clearly what v0.1.6 does and does not do.

### Current scope

- binary classification only
- 3 exits only
- greedy segment policy
- greedy clip-wise Depth×Time

### Not included in v0.1.6

- EA policy
- hint passing
- 5-exit / K-exit logic
- multiclass evaluation

### Why v0.1.6 still matters

This version is the clean baseline for:

- historical greedy comparison
- 3-exit fair comparison
- later comparison against EA, hint, and K-exit variants

---

## How to Use This Document

Use this file as the master structure for the mini-book / Overleaf documentation of the v0.1.6 baseline.

When the code changes, update both:

- the implementation
- this structure file

so the documentation stays aligned with the real pipeline.
