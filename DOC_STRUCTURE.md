# ASHADIP / NeuroAccuExit – Documentation Structure for `kexit-greedy-hint`

This document defines the recommended documentation structure for the **`kexit-greedy-hint`** branch, which generalizes the historical `v0.1.6` line into a reusable **generic K-exit / C-class** audio early-exit pipeline and supports both **hint** and **no-hint** greedy evaluation.

The key distinction is:

- **`v0.1.6` tag** = frozen historical 3-exit greedy baseline
- **`kexit-greedy-hint` branch** = corrected reusable branch for:
  - 3-exit greedy
  - 5-exit greedy
  - 3-exit greedy + hint
  - 5-exit greedy + hint

This file should be used as the long-form thesis / paper-note skeleton for the corrected branch, not for the frozen historical baseline.

---

## Chapter 1 – Introduction and Motivation

**Goal:** Explain why `kexit-greedy-hint` exists and why the branch had to be updated.

### Include
- binary moth wingbeat gender classification
- motivation for early exit under compute constraints
- need to compare **compact** and **deeper** exit hierarchies fairly
- need to compare **hint** and **no-hint** using the same codebase
- need to eliminate configuration mismatch so the results reflect real behavior rather than stale settings

### Main branch contribution
- generic **K-exit / C-class** refactor
- CLI-controlled **hint / no-hint**
- dynamic loss weights from actual exit count
- corrected effective-config saving
- validated corrected four-run comparison

---

## Chapter 2 – System Overview

**Goal:** Give a top-down view of the corrected dynamic pipeline.

### End-to-end pipeline
1. raw WAV data
2. segment manifest generation
3. log-mel feature extraction
4. dynamic `TinyAudioCNN` with configurable `tap_blocks`
5. dynamic `ExitNet`
6. optional sequential exit-to-exit hint passing
7. training with dynamic loss weights
8. per-exit temperature calibration
9. greedy threshold selection
10. segment policy evaluation
11. full-clip evaluation
12. Depth×Time evaluation
13. summary, analysis, and profiling
14. LaTeX/report generation

### Important implementation update
The branch now supports:
- `-TapBlocks "1,3"` or `-TapBlocks "1,2,3,4"`
- `-ExitHint "true"` or `-ExitHint "false"`

This removes the need to edit YAML for every experiment.

---

## Chapter 3 – Model Architecture

**Goal:** Describe the generic K-exit model and the optional hint path.

### 3.1 Dynamic backbone taps
- `tap_blocks=(1,3)` → 3 exits
- `tap_blocks=(1,2,3,4)` → 5 exits

### 3.2 Generic ExitNet
- one classifier head per tap
- one final classifier head
- total exits = `len(tap_blocks) + 1`

### 3.3 Optional sequential hint passing
- later exits may consume a small hint derived from the previous exit
- hint can use probabilities/logits and optional summary statistics
- hint enable/disable is controlled at runtime

### 3.4 Important corrected config behavior
The effective run config must now be derived **after model construction**, so that:
- `num_exits` is correct
- `loss_weights` match the actual exit count
- `config_used.yaml` matches the real run

---

## Chapter 4 – Training, Calibration, and Thresholding

**Goal:** Explain the corrected train-time logic.

### 4.1 Dynamic loss weights
- **3 exits** → `[0.3, 0.3, 1.0]`
- **5 exits** → `[0.3, 0.3, 0.6, 0.8, 1.0]`
- avoid stale 3-exit schedules when the built model is actually 5-exit

### 4.2 Hint override
- YAML keeps a safe default
- actual experiment mode is chosen from the command line

### 4.3 Calibration and thresholds
- per-exit temperature scaling
- greedy threshold selection
- segment policy and clip policy evaluated under the same effective configuration

---

## Chapter 5 – Evaluation Protocol

**Goal:** Describe the three levels of evaluation used in the corrected comparison.

### 5.1 Per-exit test evaluation
Report:
- per-exit accuracies
- later-exit strength vs early-exit strength

### 5.2 Segment-level greedy policy
Report:
- policy accuracy
- average exit depth
- exit mix
- flip-any rate
- average flip count
- exit consistency

### 5.3 Clip-level evaluation
Two modes:
- full-clip baseline
- Depth×Time early stopping

Report:
- clip accuracy
- used-window / processed-window segment accuracy
- average windows used
- windows saved
- average compute units
- compute saved
- flip-rate
- exit consistency

---

## Chapter 6 – Exact Validated Commands

### 6.1 3-exit no-hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "3exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -ExitHint "false" `
  -RunClipPolicy
```

### 6.2 3-exit hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "3exit_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -ExitHint "true" `
  -RunClipPolicy
```

### 6.3 5-exit no-hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "5exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -ExitHint "false" `
  -RunClipPolicy
```

### 6.4 5-exit hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "5exit_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -ExitHint "true" `
  -RunClipPolicy
```

---

## Chapter 7 – Main Results

### 7.1 Summary comparison

| Setting | Hint | Policy acc | Avg exit depth | Full-clip avg compute | Depth×Time used-win acc | Depth×Time avg windows | Depth×Time avg compute | Depth×Time compute saved | Best interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `3exit_greedy` | No | 0.9754 | 1.982 | 29.273 | 0.9778 | 2.045 / 14.773 | 5.364 | 81.68% | Strong simple no-hint efficiency baseline |
| `5exit_greedy` | No | 0.9908 | 2.637 | 38.955 | 0.9778 | 2.045 / 14.773 | 7.318 | 80.53% | Best deep-capacity no-hint baseline |
| `3exit_greedy_hint` | Yes | 0.9908 | 1.895 | 28.000 | 1.0000 | 2.000 / 14.773 | 4.955 | 82.31% | Best overall efficiency-quality tradeoff |
| `5exit_greedy_hint` | Yes | 0.9723 | 2.465 | 36.409 | 0.9783 | 2.091 / 14.773 | 7.455 | 79.53% | Hint not yet beneficial in current 5-exit setup |

### 7.2 Per-exit accuracy

| Setting | Exit1 | Exit2 | Exit3 | Exit4 | Exit5 |
|---|---:|---:|---:|---:|---:|
| `3exit_greedy` | 0.8338 | 0.9385 | 0.9754 | — | — |
| `5exit_greedy` | 0.8369 | 0.8892 | 0.9723 | 0.9754 | 0.9908 |
| `3exit_greedy_hint` | 0.8369 | 0.9662 | 0.9908 | — | — |
| `5exit_greedy_hint` | 0.8308 | 0.8646 | 0.9231 | 0.9538 | 0.9692 |

### 7.3 What the corrected results now show

- `3exit_greedy_hint` and `5exit_greedy` tie for best segment policy accuracy at **0.9908**
- `3exit_greedy_hint` is the best **efficiency-quality** result:
  - lowest Depth×Time compute among the top-accuracy models
  - perfect clip accuracy
  - strongest compute saving
- `5exit_greedy` is the best **deep-capacity no-hint** result
- `5exit_greedy_hint` is still weaker than `5exit_greedy`

### 7.4 Full detailed interpretation

The corrected four-run comparison gives a much stronger and more credible research story than the earlier provisional state. The compact **3-exit hint-enabled** model and the deeper **5-exit no-hint** model now tie for best segment-level greedy policy accuracy at **0.9908**, but they occupy different parts of the accuracy-efficiency space. The compact hinted model reaches that accuracy with a shallower average exit depth (**1.895**), lower full-clip compute (**28.000**), lower Depth×Time compute (**4.955**), and perfect clip accuracy. This makes **`3exit_greedy_hint`** the strongest overall result for efficiency-oriented deployment.

The corrected **5-exit no-hint** run is now clearly the strongest deep-capacity baseline. It achieves the same top segment policy accuracy (**0.9908**) and very strong later-exit accuracies, especially at the deepest exit. However, it remains more compute-expensive than the compact hinted model, both in full-clip and Depth×Time evaluation. Therefore, it should be interpreted as the strongest **high-capacity no-hint** result rather than the best deployment-efficient model.

The most important negative result is the current **5-exit hint-enabled** run. Even after the configuration mismatch was fixed, hint passing still does not improve the deeper greedy pipeline under the current design. Its segment policy accuracy drops to **0.9723**, its later exits are weaker than the corrected 5-exit no-hint baseline, and its Depth×Time compute (**7.455**) is not competitive with either `3exit_greedy_hint` or `5exit_greedy`. This suggests that the present hint mechanism is effective in the compact 3-exit regime but does not yet transfer successfully to the deeper 5-exit regime.

### 7.5 Paper-ready short paragraph

In the corrected four-run comparison, `3exit_greedy_hint` and `5exit_greedy` achieve the best segment-level greedy policy accuracy at 99.08%, but they do so with different trade-offs. `3exit_greedy_hint` provides the best efficiency-quality balance, with lower average exit depth, lower full-clip and Depth×Time compute, and perfect clip accuracy. In contrast, `5exit_greedy` is the strongest deep-capacity no-hint baseline. The current `5exit_greedy_hint` configuration does not improve the deeper greedy pipeline, indicating that sequential hint passing is highly effective in the compact 3-exit setting but not yet beneficial in the present 5-exit design.

### 7.6 Reviewer-safe answer

**Question:** If hint passing helps, why is the best 5-exit result still no-hint?

**Answer:** The corrected results show that hint passing is helpful in the compact 3-exit architecture, where it improves both decision quality and efficiency. However, the current 5-exit hint formulation does not yet improve the deeper greedy pipeline and remains weaker than the corrected 5-exit no-hint baseline. Therefore, the honest conclusion is that hint passing is promising, but its benefit is currently architecture-dependent and stronger in the compact setting than in the deeper setting.

---

## Chapter 8 – Excel-Aligned Master Tables

This section should preserve the exact workbook structure used for repo notes and release documentation.

### Required column layout
1. Metric
2. Segment Random Greedy Policy Test
3. Full-Clip Sequential Greedy Policy Test
4. Depth × Time Clip Greedy Policy Test
5. Key Change

### Required sheet set
- `3exit-greedy`
- `5exit-greedy`
- `3exit-greedy-hint`
- `5exit-greedy-hint`

### Key change text to preserve

- **3exit-greedy:** strong no-hint greedy efficiency baseline, but surpassed by `3exit_greedy_hint`
- **5exit-greedy:** best deep-capacity no-hint baseline after corrected dynamic configuration
- **3exit-greedy-hint:** best overall efficiency-quality result
- **5exit-greedy-hint:** still weaker than `5exit_greedy` under the current design

---

## Chapter 9 – Reproducibility and Run Management

### Expected config behavior after the fix
For each run, `config_used.yaml` should correctly show:
- actual `tap_blocks`
- actual `exits`
- actual `loss_weights`
- actual `exit_hint.enable`

### Example expected values

#### 3-exit hint
- `tap_blocks: [1, 3]`
- `exits: 3`
- `loss_weights: [0.3, 0.3, 1.0]`
- `exit_hint.enable: true`

#### 5-exit hint
- `tap_blocks: [1, 2, 3, 4]`
- `exits: 5`
- `loss_weights: [0.3, 0.3, 0.6, 0.8, 1.0]`
- `exit_hint.enable: true`

---

## Chapter 10 – Limitations and Outlook

### Current limitations
- the current 5-exit hint design is not yet superior to the 5-exit no-hint greedy baseline
- hint gains appear stronger in the compact regime than in the deeper regime
- the present conclusions are greedy-policy specific
- the branch still focuses on binary classification

### Forward-looking directions
- retune 5-exit hint architecture and loss balance
- investigate why 5-exit hint underuses its depth advantage
- compare hint passing with EA or hybrid stopping
- extend to multiclass settings
- study whether different hint dimensionality / detach / stats settings improve deeper hinted models

---

## How to Use This Document

Use this file as the master long-form documentation scaffold for the corrected `kexit-greedy-hint` branch. Keep the distinction clear between:
- historical `v0.1.6`
- corrected greedy no-hint baselines
- corrected greedy hint-enabled baselines

The README should stay branch/release-note oriented, while this file should carry the longer interpretation and paper/thesis-ready wording.
