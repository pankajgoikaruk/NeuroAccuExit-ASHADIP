# ASHADIP / NeuroAccuExit – Documentation Structure for `kexit-greedy`

This document defines the recommended documentation structure for the **`kexit-greedy` branch**, which refactors the historical `v0.1.6` code line into a **generic K-exit / C-class** audio early-exit pipeline and validates the **greedy no-hint baseline** on both **3-exit** and **5-exit** settings.

The key distinction is:

- **`v0.1.6` tag** = frozen historical **3-exit greedy baseline**
- **`kexit-greedy` branch** = reusable **K-exit / C-class** refactor validated under **greedy no-hint** comparison on both **3 exits** and **5 exits**

This document should be used for the updated mini-book / Overleaf write-up of the refactored greedy branch, not for the frozen historical baseline.

---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain why the `kexit-greedy` refactor was created and what it changes.

### Content to include

- **Domain**
  - acoustic classification of moth wingbeat audio
  - binary classification: male vs female
- **Motivation**
  - the original code line was hardcoded for 3 exits
  - later research required comparing 3-exit and 5-exit settings fairly
  - a reusable K-exit implementation was needed to support future early-exit variants without rewriting the full stack
  - a clean **greedy no-hint** baseline was needed before testing **sequential hint passing**
- **High-level idea**
  - raw audio → segmentation → log-mel features → dynamic K-exit TinyAudioCNN + ExitNet → calibration → greedy threshold selection → segment and clip policy evaluation
- **Main contribution of the branch**
  - successful refactor from a fixed 3-exit path to a generic **K-exit / C-class** path
  - fair validation of **3-exit vs 5-exit** under the same greedy no-hint pipeline

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

Feature caches should remain separated by tap configuration when using the main runner, so 3-exit and 5-exit runs are not accidentally mixed.

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

**Goal:** Describe the dynamic K-exit TinyAudioCNN + ExitNet used in `kexit-greedy`.

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
- however, without hint passing, stronger deep exits do not automatically translate into better clip-level efficiency

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

## Chapter 8 – Results, Interpretation, and Master Tables

**Goal:** Present the actual findings from the 3-exit vs 5-exit comparison and define the exact repo-note table structure to maintain.

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

### 8.4 Core interpretation

The cleanest interpretation is:

> **Without sequential hint passing, 5 exits improve segment-level decision quality, but do not improve clip-level efficiency under the current greedy Depth×Time stopping rule.**

Therefore:

- **3-exit greedy** is the stronger **efficiency baseline**
- **5-exit greedy** is the stronger **segment-quality baseline**

This branch is valuable because it proves that simply increasing the number of exits is not enough; a stronger mechanism is needed to turn those additional exits into genuinely useful early decisions.

### 8.5 Full detailed interpretation of the no-hint greedy baseline

The no-hint greedy comparison provides a clear and useful baseline story for the refactored branch. At the **segment-policy** level, the 5-exit model improves greedy policy accuracy from **0.9754** to **0.9846**, which is the strongest positive result of the K-exit refactor. This shows that the extra exits are not redundant: the deeper decision points improve local decision quality. However, the 5-exit model also exits later on average (**1.982 → 2.449**), has a higher flip-any rate (**0.1785 → 0.2215**), and has slightly worse exit consistency (**1.0000 → 0.9969**). Therefore, the extra exits improve quality, but they do not make the greedy policy more decisive or more stable.

At the **per-exit** level, the strongest evidence that the refactor is working comes from the deeper 5-exit heads: `exit4 = 0.9846` and `exit5 = 0.9877`. These results show that the added late exits are genuinely strong. The weaker value at `exit2 = 0.8400` also shows that the benefit is not uniform across all intermediate exits. The main gain comes from the later decision stages, not from every new exit equally. This distinction matters, because it explains why the architecture can improve segment quality without automatically improving early stopping efficiency.

At the **full-clip** level, both variants already saturate at **1.0000 clip accuracy** on the current test set, which contains only **22 test files**. Because clip accuracy is saturated, it is not the best metric for separating the two models. In this setting, the more informative comparison becomes the trade-off between segment-level quality and compute cost. On that dimension, the 5-exit model is more expensive, increasing average compute units from **29.273** to **36.182**. So the 5-exit configuration gives better segment predictions, but the improvement is bought with higher computational cost.

The **Depth×Time** results are the most revealing. Both 3-exit and 5-exit variants achieve the same **1.0000 clip accuracy**, use the same average number of windows (**2.045 / 14.773**), and save the same percentage of windows (**86.15%**). However, the 5-exit model still uses more compute per used window (**5.364 → 7.045**) and therefore saves less compute overall (**81.68% → 80.53%**). This means the extra exits are not yet being converted into better time-aware efficiency. Instead, they make each used window more expensive while leaving clip-level outcomes unchanged under the current greedy stopping rule.

The most honest scientific interpretation is therefore: **without sequential hint passing, increasing the architecture from 3 exits to 5 exits improves segment-level decision quality, but does not improve clip-level time-aware efficiency under greedy stopping**. This is not a failure. It is a useful baseline result. It shows that the K-exit refactor successfully creates stronger deeper exits, but also shows that a stronger mechanism is needed to make those extra exits useful earlier in the inference process. That is the direct motivation for sequential hint passing.

### 8.6 Paper/thesis-ready text

#### Short paper-style paragraph

In the no-hint greedy setting, increasing the architecture from **3 exits** to **5 exits** improved segment-level policy accuracy from **97.54%** to **98.46%**, confirming that the additional deeper exits enhance local decision quality. However, this gain did not translate into better clip-level efficiency under **Depth×Time** stopping: both variants achieved **100% clip accuracy**, used the same average number of windows (**2.045 / 14.773**), and saved the same proportion of windows (**86.15%**), while the 5-exit model incurred higher average compute cost (**5.364 → 7.045** average compute units per used window). Thus, without sequential hint passing, the additional exits improved representational quality but did not improve time-aware efficiency.

#### Expanded thesis-style paragraph

The comparison between the 3-exit and 5-exit greedy no-hint configurations reveals an important trade-off. The 5-exit model achieves better segment-level decision quality, as shown by the increase in greedy policy accuracy from **0.9754** to **0.9846**, and by the strong performance of the late exits (`exit4 = 0.9846`, `exit5 = 0.9877`). However, this improvement comes with a higher average exit depth and slightly weaker internal stability. More importantly, the stronger late exits do not improve clip-level time-aware efficiency under the present greedy stopping rule. Both variants achieve perfect clip accuracy on the current test set and use the same average number of windows in the Depth×Time setting, yet the 5-exit model consumes more compute. This indicates that the additional exits are currently beneficial mainly as stronger deep decision stages rather than as more efficient early decision stages. Consequently, the 3-exit model should be viewed as the stronger efficiency baseline, whereas the 5-exit model should be viewed as the stronger decision-quality baseline. This result directly motivates sequential hint passing, whose purpose is to make the additional exits improve earlier decisions instead of simply adding computational cost.

#### Reviewer-safe answer

If a reviewer asks, **"If greedy no-hint already works better for efficiency, why use sequential hint passing?"**, the answer is: the 5-exit no-hint baseline already shows that extra exits improve segment decision quality, but they do not yet improve clip-time efficiency by themselves. Sequential hint passing is therefore introduced to make those extra exits more useful, so that the added depth improves earlier decisions rather than only increasing compute.

### 8.7 Excel-aligned master table section

This section should be maintained in the repo notes and branch-level README using the **same structure as the reference spreadsheet**.

#### Required column layout

1. **Metric**
2. **Segment Random Greedy Policy Test**
3. **Full-Clip Sequential Greedy Policy Test**
4. **Depth × Time Clip Greedy Policy Test**
5. **Key Change**

#### Required row order

1. PowerShell Command  
2. Files  
3. Segments  
4. Policy  
5. Device  
6. SegmentSec  
7. HopSec  
8. NMels  
9. Exit accuracies (`Exit1`, `Exit2`, ... as applicable)  
10. Policy test accuracy  
11. Avg exit depth  
12. Exit mix  
13. Flip-rate (any flip)  
14. Avg flip-count  
15. Exit-consistency (taken==final)  
16. Windows Saved (%)  
17. Compute Saved (%)  
18. Window distribution mode  
19. Clip-metrics (segment-policy, for fair comparison)  
20. Clip accuracy  
21. Avg windows used  
22. Avg compute units (sum depth over used windows)  
23. Fixed-position diagnostic (independent of time-exit)  
24. 3-window diagnostic `Acc_firstK`  
25. 3-window diagnostic `Acc_midK`  
26. 3-window diagnostic `Acc_lastK`  
27. Stop-speed group diagnostic (Depth×Time only; first-K accuracy)  
28. stop-speed group rows (`stop_1`, `stop_2`, `stop_3` for 3-exit; `stop_2`, `stop_3_4`, `stop_5_plus` for 5-exit if populated)  
29. Confusion matrix (clip-level)  
30. Per-class: Female  
31. Per-class: Male  
32. Total Time  

> **Important note:** `3-window diagnostic (K=3)` means **three windows**, not **three exits**.

#### Required filled master tables

The branch documentation should keep the two filled master tables exactly as follows:
- one for **3exit_greedy**
- one for **5exit_greedy**

For the current validated run set, the values should match the README master tables exactly.

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

### Typical 3-exit command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "3exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy
```

### Typical 5-exit command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "5exit_greedy" `
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
- greedy no-hint baseline comparison

### What is not yet solved

- a 5-exit time policy that clearly beats the 3-exit baseline in clip-level efficiency
- EA policy integration and retuning in this branch
- sequential hint-passing results in this branch
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
- `kexit-greedy` = generic K-exit refactor validated on both 3 and 5 exits under greedy no-hint comparison

When the implementation changes, update both:

- the code
- this structure file
- the README master tables

so the documentation remains aligned with the actual validated pipeline and findings.
