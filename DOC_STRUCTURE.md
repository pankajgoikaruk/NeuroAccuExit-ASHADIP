# ASHADIP / NeuroAccuExit – Documentation Structure for `kexit-hint`

This document defines the recommended documentation structure for the **`kexit-hint` branch**, which generalizes the historical `v0.1.6` code line into a **generic K-exit / C-class** audio early-exit pipeline and includes an optional **5-exit sequential hint-passing** path.

The key distinction is:

- **`v0.1.6` tag** = frozen historical **3-exit greedy baseline**
- **`kexit-hint` branch** = reusable **K-exit / C-class** refactor validated on both **3 exits** and **5 exits**, with an additional **5-exit sequential hint-passing experiment**
- **`kexit_greedy_no_hint`** = current best greedy 5-exit reference on this branch
- **`kexit_greedy_hint`** = experimental 5-exit sequential hint-passing run on this branch

This document should be used for the updated mini-book / Overleaf write-up of the refactored branch, not for the frozen historical baseline.

---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain why the `kexit-hint` branch was created and what it changes.

### Content to include

- **Domain**
  - acoustic classification of moth wingbeat audio
  - binary classification: male vs female
- **Motivation**
  - the original code line was hardcoded for 3 exits
  - later research required comparing 3-exit and 5-exit settings fairly
  - a reusable K-exit implementation was needed to support future early-exit variants without rewriting the full stack
  - later work also required testing whether **sequential exit-to-exit hint passing** could improve the 5-exit path
- **High-level idea**
  - raw audio → segmentation → log-mel features → dynamic K-exit TinyAudioCNN + ExitNet → calibration → greedy threshold selection → segment and clip policy evaluation
- **Main contribution of the branch**
  - successful refactor from a fixed 3-exit path to a generic **K-exit / C-class** path
  - optional integration of a lightweight **sequential hint-passing** mechanism

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
   - optionally attach **sequential exit-to-exit hints**
7. `utils/model_factory.py`
   - build the correct non-hint or hint-enabled model from `config_used.yaml`
8. `training/train.py`
   - train dynamic K-exit model and save `ckpt/best.pt`
9. `training/calibrate.py`
   - fit per-exit temperatures and save `temperature.json`
10. `training/thresholds_offline.py`
   - select greedy `tau`
   - save `thresholds.json`
11. `scripts/policy_test.py`
   - segment-level greedy policy evaluation
12. `scripts/clip_policy_test.py`
   - clip-wise full baseline and Depth×Time evaluation
13. `scripts/summarize_run.py`
   - consolidate artifacts into `summary.json`
14. `scripts/analyse_run.py`
   - confusion matrices, ROC, plots, analysis summary
15. `scripts/profile_latency.py`
   - on-device latency and FLOPs profiling
16. `scripts/run_reports.ps1`
   - regenerate per-run and cross-run reports / LaTeX tables

---

## Chapter 3 – Model Architecture

**Goal:** Explain the generic K-exit structure and the optional hint mechanism.

### 3.1 Dynamic backbone taps

- `TinyAudioCNN` exposes configurable `tap_blocks`
- validated settings:
  - `tap_blocks=(1,3)` → 3 exits
  - `tap_blocks=(1,2,3,4)` → 5 exits

### 3.2 Generic ExitNet

- one classifier head per tap
- one final classifier head
- total exits = `len(tap_blocks) + 1`

### 3.3 Optional sequential hint passing

- each later exit can consume a small hint derived from the previous exit
- hint source can be probabilities or logits
- optional confidence / margin / entropy summary statistics can be included
- hint passing is controlled from `model.exit_hint` in `configs/audio_moth.yaml`

### 3.4 Important interpretation

- the branch supports hint passing as an **optional architectural mechanism**
- current results show it is **behaviorally meaningful**, but **not yet the best final greedy model**

---

## Chapter 4 – Training, Calibration, and Thresholding

**Goal:** Explain the training and greedy stopping workflow.

### Include

- weighted multi-exit training
- temperature scaling per exit
- greedy threshold selection (`tau`)
- dynamic reconstruction of hint and no-hint models from `config_used.yaml`
- consistent run metadata and reporting paths

### Important implementation note

The refactor updated the run/report scripts so that:

- tap configuration is recorded in metadata
- hint and no-hint models are reconstructed consistently from `config_used.yaml`
- reporting regenerates safely for both 3-exit and 5-exit runs
- schema conflicts in old CSV logs fall back to `_kexit.csv` outputs instead of corrupting older files

---

## Chapter 5 – Evaluation Protocol

**Goal:** Describe the three evaluation levels used in this branch.

### 5.1 Per-exit test evaluation

Report:

- exit1 accuracy
- exit2 accuracy
- exit3 accuracy
- exit4 accuracy
- exit5 accuracy

### 5.2 Segment-level greedy policy

Report:

- policy accuracy
- avg exit depth
- exit mix
- flip-any rate
- avg flip count
- exit consistency

### 5.3 Clip-level evaluation

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

## Chapter 6 – Validated Run Settings

**Goal:** Record the main validated experiments in a reusable way.

### 6.1 3-exit reference on this branch

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_3exit_ref" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy
```

### 6.2 5-exit greedy no-hint

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_greedy_no_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3 `
  -TimeMargin 0.0
```

### 6.3 5-exit greedy hint passing

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3 `
  -TimeMargin 0.0
```

---

## Chapter 7 – Main Results

**Goal:** Present the validated 5-exit no-hint vs 5-exit hint comparison clearly.

### 7.1 Per-exit test accuracy

| Metric | Greedy No Hint | Greedy Hint Passing | Better |
|---|---:|---:|---|
| Exit1 accuracy | 0.8400 | 0.8215 | No hint |
| Exit2 accuracy | 0.8400 | 0.8677 | Hint |
| Exit3 accuracy | 0.9600 | 0.9292 | No hint |
| Exit4 accuracy | 0.9846 | 0.9723 | No hint |
| Exit5 accuracy | 0.9877 | 0.9692 | No hint |

### 7.2 Segment-level greedy policy

| Metric | Greedy No Hint | Greedy Hint Passing | Better |
|---|---:|---:|---|
| Policy accuracy | 0.9846 | 0.9723 | No hint |
| Avg exit depth | 2.449 | 3.105 | No hint |
| Flip-any rate | 0.2215 | 0.2246 | No hint |
| Avg flip count | 0.2954 | 0.2954 | Tie |
| Exit consistency | 0.9969 | 0.9969 | Tie |

### 7.3 Full-clip baseline

| Setting | Clip acc | Processed-win acc | First-3 diag | Avg compute units | Avg depth per used window |
|---|---:|---:|---:|---:|---:|
| 5-exit no hint | 1.0000 | 0.9846 | 0.9692 | 36.182 | 2.449 |
| 5-exit hint | 1.0000 | 0.9723 | 0.9385 | 45.864 | 3.105 |

### 7.4 Depth×Time

| Metric | Greedy No Hint | Greedy Hint Passing | Better |
|---|---:|---:|---|
| Clip accuracy | 1.0000 | 1.0000 | Tie |
| Used-window segment accuracy | 0.9778 | 0.9375 | No hint |
| Avg windows used | 2.045 | 2.182 | No hint |
| Windows saved (%) | 86.15 | 85.23 | No hint |
| Avg compute units | 7.045 | 8.909 | No hint |
| Avg depth per used window | 3.444 | 4.083 | No hint |
| Compute saved (%) | 80.53 | 80.57 | Tie |
| Flip-rate | 0.4222 | 0.5208 | No hint |
| Exit consistency | 1.0000 | 1.0000 | Tie |

---

## Chapter 8 – Interpretation

**Goal:** State clearly what the comparison means.

### Main conclusions

1. The refactor is successful.
   - the same pipeline now runs both 3-exit and 5-exit settings
   - hint and no-hint reconstruction both work end-to-end

2. The current best 5-exit greedy system is the **no-hint** version.
   - better segment-policy accuracy
   - lower average exit depth
   - better Depth×Time used-window accuracy
   - fewer windows used
   - lower flip-rate

3. The hint mechanism is still behaviorally meaningful.
   - segment-policy `exit2` usage increased from **0.1323** to **0.3877**
   - per-exit `exit2` test accuracy improved from **0.8400** to **0.8677**

4. Hint passing is not yet the better final method.
   - it became deeper and more expensive
   - it did not beat the no-hint greedy baseline overall

5. The strongest current advantage of 5 exits still comes from deeper exits.
   - `exit4 = 0.9846`
   - `exit5 = 0.9877`

### Recommended reviewer-safe interpretation

> We do not claim that the current greedy hint version is the best final model. We keep it because it tests a real architectural idea: whether passing information from one exit to the next can improve intermediate decision quality and make earlier exits more useful. Our results show that hint passing clearly changes exit behavior and improves exit2, but it does not yet improve the overall greedy accuracy-efficiency trade-off.

---

## Chapter 9 – Current Scope

**Goal:** State clearly what is included and excluded.

### Included in this branch

- greedy segment policy
- greedy clip-wise Depth×Time policy
- dynamic K-exit architecture
- 3-exit and 5-exit validation
- optional sequential exit-to-exit hint passing
- dynamic profiling and reporting
- binary classification

### Not yet solved

- EA policy tuning for the K-exit branch
- a hint-passing configuration that clearly beats the no-hint greedy baseline
- multiclass experiments
- a 5-exit time policy that clearly beats the 3-exit baseline in clip-level efficiency

---

## Chapter 10 – Limitations and Outlook

**Goal:** State clearly what this branch does and does not yet solve.

### Why this branch matters

This branch establishes:

- a reusable implementation foundation for future early-exit work
- a fair 3-exit vs 5-exit comparison framework
- a concrete no-hint vs hint comparison for the greedy 5-exit setting
- a clean bridge from the historical `v0.1.6` baseline to later K-exit, EA, and hint-based variants

### Best current recommendation

- **historical baseline:** `v0.1.6`
- **best current greedy 5-exit reference:** `kexit_greedy_no_hint`
- **experimental architectural branch:** `kexit_greedy_hint`
- **main development/documentation branch:** `kexit-hint`

---

## How to Use This Document

Use this file as the master structure for the updated branch-level documentation / thesis mini-book.

Keep the distinction clear:

- `v0.1.6` = historical frozen 3-exit greedy baseline
- `kexit-hint` = current generic K-exit branch built on top of the earlier refactor work
- `kexit_greedy_no_hint` = current best greedy 5-exit reference
- `kexit_greedy_hint` = experimental 5-exit sequential hint-passing run

When the implementation changes, update both:

- the code
- this structure file

so the documentation remains aligned with the actual validated pipeline and findings.
