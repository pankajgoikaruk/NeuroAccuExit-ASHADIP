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


## Files Updated in This Refactor

Core code path updated for dynamic K-exit support and corrected hinted model reconstruction:

- `adapters/audio_adapter.py`
- `models/exit_net.py`
- `training/train.py`
- `training/eval.py`
- `training/calibrate.py`
- `training/thresholds_offline.py`
- `scripts/policy_test.py`
- `scripts/clip_policy_test.py`
- `scripts/summarize_run.py`
- `scripts/analyse_run.py`
- `scripts/profile_latency.py`
- `utils/model_factory.py`
- `utils/profiling.py`
- `scripts/run_full.ps1`
- `scripts/run_reports.ps1`
- `scripts/compare_variants.py`
- `scripts/variants_to_latex.py`
- `scripts/ondevice_to_latex.py`
- `scripts/analysis_to_latex.py`

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


---

## Chapter 11 – Additional Comparison Tables from Workbook Draft

This chapter converts the workbook draft tables into paper-ready and repo-ready forms with corrected names and calculated difference columns.

### 11.1 Recommended naming

Use the following final names in the thesis / repo notes:

1. **Segment-level greedy policy comparison**
2. **Per-exit test accuracy comparison**
3. **Exit mix comparison (segment policy and Depth×Time)**
4. **Depth×Time comparison: no-hint vs hint passing**
5. **Full-clip vs Depth×Time accuracy-efficiency tradeoff**

These names are more precise than the original labels and match the actual evaluation targets.

### 11.2 Segment-level greedy policy comparison

| Metric | 3exit No-Hint | 3exit Hint | Δ (Hint−No-Hint) | 5exit No-Hint | 5exit Hint | Δ (Hint−No-Hint) |
|---|---:|---:|---:|---:|---:|---:|
| Policy accuracy | 0.9754 | 0.9908 | +0.0154 | 0.9908 | 0.9723 | -0.0185 |
| Avg exit depth | 1.982 | 1.895 | -0.0870 | 2.637 | 2.465 | -0.1720 |
| Flip-any rate | 0.1785 | 0.1908 | +0.0123 | 0.1908 | 0.2123 | +0.0215 |
| Avg flip count | 0.2031 | 0.2154 | +0.0123 | 0.2338 | 0.2492 | +0.0154 |
| Exit consistency | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 0.9908 | -0.0092 |

### 11.3 Per-exit test accuracy comparison

| Metric | 3exit No-Hint | 3exit Hint | Δ (Hint−No-Hint) | 5exit No-Hint | 5exit Hint | Δ (Hint−No-Hint) |
|---|---:|---:|---:|---:|---:|---:|
| Exit1 accuracy | 0.8338 | 0.8369 | +0.0031 | 0.8369 | 0.8308 | -0.0061 |
| Exit2 accuracy | 0.9385 | 0.9662 | +0.0277 | 0.8892 | 0.8646 | -0.0246 |
| Exit3 accuracy | 0.9754 | 0.9908 | +0.0154 | 0.9723 | 0.9231 | -0.0492 |
| Exit4 accuracy | — | — | — | 0.9754 | 0.9538 | -0.0216 |
| Exit5 accuracy | — | — | — | 0.9908 | 0.9692 | -0.0216 |

### 11.4 Exit mix comparison (segment policy and Depth×Time)

#### 3-exit exit mix

| Exit | 3exit No-Hint Segment | 3exit Hint Segment | Δ | 3exit No-Hint Depth×Time | 3exit Hint Depth×Time | Δ |
|---|---:|---:|---:|---:|---:|---:|
| e1 | 0.3631 | 0.3846 | +0.0215 | 0.0889 | 0.0909 | +0.0020 |
| e2 | 0.2923 | 0.3354 | +0.0431 | 0.2000 | 0.3409 | +0.1409 |
| e3 | 0.3446 | 0.2800 | -0.0646 | 0.7111 | 0.5680 | -0.1431 |

#### 5-exit exit mix

| Exit | 5exit No-Hint Segment | 5exit Hint Segment | Δ | 5exit No-Hint Depth×Time | 5exit Hint Depth×Time | Δ |
|---|---:|---:|---:|---:|---:|---:|
| e1 | 0.3569 | 0.3723 | +0.0154 | 0.0889 | 0.0870 | -0.0019 |
| e2 | 0.0308 | 0.1354 | +0.1046 | 0.0000 | 0.0652 | +0.0652 |
| e3 | 0.3538 | 0.2677 | -0.0861 | 0.3778 | 0.3043 | -0.0735 |
| e4 | 0.1354 | 0.1046 | -0.0308 | 0.3111 | 0.2826 | -0.0285 |
| e5 | 0.1231 | 0.1200 | -0.0031 | 0.2222 | 0.2609 | +0.0387 |

### 11.5 Depth×Time comparison: no-hint vs hint passing

| Metric | 3exit No-Hint | 3exit Hint | Δ (Hint−No-Hint) | 5exit No-Hint | 5exit Hint | Δ (Hint−No-Hint) |
|---|---:|---:|---:|---:|---:|---:|
| Used-window segment accuracy | 0.9778 | 1.0000 | +0.0222 | 0.9778 | 0.9783 | +0.0005 |
| Avg depth per used window | 2.622 | 2.477 | -0.1450 | 3.578 | 3.565 | -0.0130 |
| Clip accuracy | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| Avg windows used | 2.045 / 14.773 | 2.000 / 14.773 | -0.045 / 14.773 | 2.045 / 14.773 | 2.091 / 14.773 | +0.046 / 14.773 |
| Windows saved | 86.15% | 86.46% | +0.31 pp | 86.15% | 85.85% | -0.30 pp |
| Compute saved | 81.68% | 82.31% | +0.63 pp | 80.53% | 79.53% | -1.00 pp |
| Avg compute units | 5.364 | 4.955 | -0.4090 | 7.318 | 7.455 | +0.1370 |
| Flip-rate | 0.4000 | 0.3864 | -0.0136 | 0.4222 | 0.4565 | +0.0343 |
| Exit consistency | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |

### 11.6 Full-clip vs Depth×Time accuracy-efficiency tradeoff

| Metric | 3exit NH Full | 3exit NH Depth×Time | Δ (DT−Full) | 5exit NH Full | 5exit NH Depth×Time | Δ (DT−Full) | 3exit Hint Full | 3exit Hint Depth×Time | Δ (DT−Full) | 5exit Hint Full | 5exit Hint Depth×Time | Δ (DT−Full) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Used-window segment accuracy | 0.9754 | 0.9778 | +0.0024 | 0.9908 | 0.9778 | -0.0130 | 0.9908 | 1.0000 | +0.0092 | 0.9723 | 0.9783 | +0.0060 |
| Clip accuracy | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| Avg depth per used window | 1.982 | 2.622 | +0.640 | 2.637 | 3.578 | +0.941 | 1.895 | 2.477 | +0.582 | 2.465 | 3.565 | +1.100 |
| Avg windows used | 14.773 / 14.773 | 2.045 / 14.773 | -12.728 / 14.773 | 14.773 / 14.773 | 2.045 / 14.773 | -12.728 / 14.773 | 14.773 / 14.773 | 2.000 / 14.773 | -12.773 / 14.773 | 14.773 / 14.773 | 2.091 / 14.773 | -12.682 / 14.773 |
| Windows saved | 0.00% | 86.15% | +86.15 pp | 0.00% | 86.15% | +86.15 pp | 0.00% | 86.46% | +86.46 pp | 0.00% | 85.85% | +85.85 pp |
| Compute saved | 0.00% | 81.68% | +81.68 pp | 0.00% | 80.53% | +80.53 pp | 0.00% | 82.31% | +82.31 pp | 0.00% | 79.53% | +79.53 pp |
| Avg compute units | 29.273 | 5.364 | -23.909 | 38.955 | 7.318 | -31.637 | 28.000 | 4.955 | -23.045 | 36.409 | 7.455 | -28.954 |
| Flip-rate | 0.1785 | 0.4000 | +0.2215 | 0.1908 | 0.4222 | +0.2314 | 0.1908 | 0.3864 | +0.1956 | 0.2123 | 0.4565 | +0.2442 |
| Exit consistency | 1.0000 | 1.0000 | +0.0000 | 0.9969 | 1.0000 | +0.0031 | 1.0000 | 1.0000 | +0.0000 | 0.9908 | 1.0000 | +0.0092 |

### 11.7 Coverage of evaluation

### Are all evaluation aspects covered?

You covered the **core evaluation space well**. The five proposed tables are useful and, after renaming and filling the difference columns, they capture the most important views of the results:

1. **Segment-level greedy policy comparison** — suitable name  
2. **Per-exit test accuracy comparison** — suitable name  
3. **Exit mix comparison (segment policy and Depth×Time)** — better than just “Exit Mix”  
4. **Depth×Time comparison: no-hint vs hint passing** — clearer than “Greedy-Depth×Time No-Hint or Hint Passing”  
5. **Full-clip vs Depth×Time accuracy-efficiency tradeoff** — better than “Full Clip vs Depth * Time-Based Accuracy vs Efficiency Tradeoff”

The only important correction is that your original draft of Table 5 did **not** include the **5exit_greedy_hint** block. That row group should be included, otherwise the tradeoff analysis is incomplete.

### Optional additions

The current five tables are enough for `README.md` and `DOC_STRUCTURE.md`. If you want one more table later, the most useful optional addition would be:

- **Calibration / threshold summary**  
  Columns could include: temperatures, selected `tau`, and maybe ECE.

That is useful for appendix/thesis detail, but it is **not required** for the main README.


### 11.8 Interpretation of the difference tables

The difference columns reinforce the main corrected story. In the compact setting, **3exit_greedy_hint** improves segment policy accuracy by **+0.0154**, improves exit2 and exit3 quality, lowers average exit depth, reduces Depth×Time compute by **-0.409**, and slightly improves both windows saved and compute saved. This is consistent with the conclusion that hint passing works well in the compact regime.

In the deeper setting, **5exit_greedy_hint** does not improve the current greedy design. Its segment policy accuracy decreases by **-0.0185**, later exits become weaker, and Depth×Time compute increases by **+0.137** while compute saved drops by **-1.00 pp**. The deeper hint setting therefore remains a limitation rather than a success case in the present branch.

The tradeoff table also shows that Depth×Time early stopping is valuable across all four runs, since it consistently reduces windows used and compute relative to the full-clip baseline. However, the best deployment-oriented configuration remains **3exit_greedy_hint**, while the best deep-capacity no-hint reference remains **5exit_greedy**.
