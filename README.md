# ASHADIP / NeuroAccuExit — `kexit-hint` Generic K-Exit Audio Pipeline

This branch documents the **`kexit-hint`** implementation built on top of the historical **`v0.1.6`** line.

The original `v0.1.6` tag remains the **locked 3-exit greedy baseline**. In contrast, `kexit-hint` generalizes that code path into a **generic K-exit / C-class structure** so the same model and evaluation pipeline can run either:

- **3 exits** with `tap_blocks=(1,3)`
- **5 exits** with `tap_blocks=(1,2,3,4)`

The task remains:

- **binary audio classification**
- moth wingbeat gender classification (**male** vs **female**)
- **log-mel spectrogram** input
- **TinyAudioCNN + ExitNet** early-exit model family

This branch now supports:

- dynamic backbone taps
- dynamic `ExitNet` heads
- dynamic training / evaluation / calibration loops
- dynamic greedy threshold selection
- dynamic segment-policy evaluation
- dynamic clip-wise **Depth×Time** evaluation
- dynamic reporting, profiling, and LaTeX table generation
- optional **sequential exit-to-exit hint passing** for the 5-exit path

---

## Why this branch exists

The old `v0.1.6` code path was still hardcoded around a **3-exit** model. This branch refactors that line into a reusable structure that can support multiple exit counts without rewriting the architecture or downstream scripts each time.

The main objective of `kexit-hint` is:

1. preserve the historical 3-exit baseline behavior
2. add a reusable **K-exit / C-class** implementation
3. validate both **3-exit** and **5-exit** runs end-to-end
4. compare the resulting accuracy / efficiency trade-offs honestly
5. test whether **sequential exit-to-exit hint passing** improves the 5-exit greedy pipeline

---

## Main architectural change

### Before

The old line assumed:

- a fixed 3-exit backbone
- a fixed 3-head `ExitNet`
- training / eval loops written around `range(3)`
- calibration, thresholding, policy tests, profiling, and reports written around `exit1`, `exit2`, `exit3`

### Now

The branch supports a **generic K-exit structure**:

- `TinyAudioCNN` exposes configurable `tap_blocks`
- `ExitNet` builds one head per tap, plus one final head
- total exits = `len(tap_blocks) + 1`

Validated settings:

- `tap_blocks=(1,3)` → **3 exits**
- `tap_blocks=(1,2,3,4)` → **5 exits**

The branch also supports an optional **sequential exit-to-exit hint path**:

- each later exit can consume a small hint derived from the previous exit
- hint passing is controlled from `model.exit_hint` in `configs/audio_moth.yaml`
- the current comparison focuses on:
  - **5-exit greedy no-hint**
  - **5-exit greedy sequential hint passing**

---

## Current pipeline overview

1. prepare segmented audio manifest
2. extract log-mel features
3. train dynamic K-exit model
4. calibrate per-exit temperatures
5. select greedy threshold `tau`
6. evaluate segment policy
7. evaluate clip policy
   - full-clip baseline
   - Depth×Time early stopping
8. summarize, analyze, and profile the run
9. generate reports and LaTeX tables

---

## Files updated in this refactor

Core code path updated for dynamic K-exit support and hinted model reconstruction:

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

## Validated run settings

### Example 3-exit reference run on this branch

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_3exit_ref" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy
```

### 5-exit greedy no-hint run

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

### 5-exit greedy hint-passing run

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

## Main run artifacts

Each run directory such as `runs/kexit_greedy_no_hint/kexit_greedy_no_hint_001/` contains:

- `ckpt/best.pt`
- `metrics.json`
- `temperature.json`
- `thresholds.json`
- `policy_results.json`
- `clip_policy_results_full.json`
- `clip_policy_results_time.json`
- `clip_policy_results.json`
- `summary.json`
- `report.json`
- `analysis_run.json`
- `profiling.json`
- `confusion_matrices.json`
- `roc_curves.json`
- `windows_used_hist.json`
- plots under `plots/`

---

## Validated results and findings

### Per-exit test accuracy comparison

| Metric | Greedy No Hint | Greedy Hint Passing | Difference | Better |
|---|---:|---:|---:|---|
| Exit1 accuracy | 0.8400 | 0.8215 | -0.0185 | No hint |
| Exit2 accuracy | 0.8400 | 0.8677 | +0.0277 | Hint |
| Exit3 accuracy | 0.9600 | 0.9292 | -0.0308 | No hint |
| Exit4 accuracy | 0.9846 | 0.9723 | -0.0123 | No hint |
| Exit5 accuracy | 0.9877 | 0.9692 | -0.0185 | No hint |

### Segment-level greedy policy comparison

| Metric | Greedy No Hint | Greedy Hint Passing | Difference | Better |
|---|---:|---:|---:|---|
| Policy accuracy | 0.9846 | 0.9723 | -0.0123 | No hint |
| Avg exit depth | 2.449 | 3.105 | +0.656 | No hint |
| Flip-any rate | 0.2215 | 0.2246 | +0.0031 | No hint |
| Avg flip count | 0.2954 | 0.2954 | 0.0000 | Tie |
| Exit consistency | 0.9969 | 0.9969 | 0.0000 | Tie |

### Segment-policy exit mix

| Exit | Greedy No Hint | Greedy Hint Passing |
|---|---:|---:|
| e1 | 0.3815 | 0.0554 |
| e2 | 0.1323 | 0.3877 |
| e3 | 0.2308 | 0.1385 |
| e4 | 0.1662 | 0.2338 |
| e5 | 0.0892 | 0.1846 |

### Full-clip baseline comparison

| Setting | Clip acc | Processed-win acc | First-3 diag | Avg compute units | Avg depth per used window |
|---|---:|---:|---:|---:|---:|
| 5-exit no hint (`1,2,3,4`) | 1.0000 | 0.9846 | 0.9692 | 36.182 | 2.449 |
| 5-exit hint (`1,2,3,4`) | 1.0000 | 0.9723 | 0.9385 | 45.864 | 3.105 |

### Depth×Time comparison

| Metric | Greedy No Hint | Greedy Hint Passing | Difference | Better |
|---|---:|---:|---:|---|
| Clip accuracy | 1.0000 | 1.0000 | 0.0000 | Tie |
| Used-window segment accuracy | 0.9778 | 0.9375 | -0.0403 | No hint |
| Avg windows used | 2.045 | 2.182 | +0.137 | No hint |
| Windows saved (%) | 86.15 | 85.23 | -0.92 | No hint |
| Avg compute units | 7.045 | 8.909 | +1.864 | No hint |
| Avg depth per used window | 3.444 | 4.083 | +0.639 | No hint |
| Compute saved (%) | 80.53 | 80.57 | +0.04 | Tie |
| Flip-rate | 0.4222 | 0.5208 | +0.0986 | No hint |
| Exit consistency | 1.0000 | 1.0000 | 0.0000 | Tie |

### Depth×Time exit mix

| Exit | Greedy No Hint | Greedy Hint Passing |
|---|---:|---:|
| e1 | 0.0889 | 0.0208 |
| e2 | 0.0667 | 0.1250 |
| e3 | 0.3556 | 0.0625 |
| e4 | 0.2889 | 0.3333 |
| e5 | 0.2000 | 0.4583 |

---

## Main conclusions from the comparison

### 1. The refactor is successful

The same pipeline now runs both:

- **3-exit**
- **5-exit**

through one generic implementation path.

### 2. The current best 5-exit greedy system is the no-hint version

Compared with the hinted 5-exit run, the **5-exit greedy no-hint** run is better overall for the current greedy pipeline:

- segment-policy accuracy: **0.9846 vs 0.9723**
- avg exit depth: **2.449 vs 3.105**
- full-clip processed-window accuracy: **0.9846 vs 0.9723**
- Depth×Time used-window accuracy: **0.9778 vs 0.9375**
- avg windows used: **2.045 vs 2.182**
- flip-rate under Depth×Time: **0.4222 vs 0.5208**

So under the current greedy stopping rule, **no hint** gives the best overall accuracy-efficiency trade-off.

### 3. The hint mechanism is still behaviorally meaningful

Sequential hint passing clearly changed the exit behavior:

- segment-policy `exit2` usage increased from **0.1323 -> 0.3877**
- per-exit `exit2` test accuracy improved from **0.8400 -> 0.8677**

So the hint mechanism is not a null result. It makes **exit2** more active and somewhat stronger.

### 4. But hint passing is not yet the better final method

The current hinted model became deeper and more expensive:

- avg exit depth: **2.449 -> 3.105**
- full-clip avg compute: **36.182 -> 45.864**
- Depth×Time avg compute: **7.045 -> 8.909**
- Depth×Time avg depth per used window: **3.444 -> 4.083**

Thus, the current hint design is **architecturally meaningful**, but **not yet beneficial** for the final greedy accuracy-efficiency trade-off.

### 5. The strongest advantage of 5 exits still comes from deep exits

For the non-hint 5-exit system:

- `exit4 = 0.9846`
- `exit5 = 0.9877`

So the strongest benefit of the 5-exit refactor still comes from the additional deeper exits, not from uniformly improving every intermediate exit.

---

## Recommended interpretation of this branch

This branch should now be described as:

> a successful **generic K-exit / C-class refactor** of the historical `v0.1.6` line, extended with an optional sequential hint-passing mechanism and validated on both non-hint and hint-enabled 5-exit greedy runs.

The most honest scientific interpretation is:

- **3 exits** remains the best historical lightweight reference
- **5-exit greedy no-hint** is the best current greedy K-exit configuration
- **5-exit hint passing** is a meaningful architectural experiment because it activates exit2 and changes routing behavior
- **hint passing still needs better policy/training design** to beat the no-hint greedy baseline

---

## Current scope

### Included in this branch

- greedy segment policy
- greedy clip-wise Depth×Time policy
- dynamic K-exit architecture
- 3-exit and 5-exit validation
- optional sequential exit-to-exit hint passing
- dynamic profiling and reporting
- binary classification

### Not yet included as a solved result

- EA policy tuning for the K-exit branch
- a hint-passing configuration that clearly beats the no-hint greedy baseline
- multiclass experiments
- a 5-exit time policy that clearly beats the 3-exit baseline on clip-level efficiency

---

## Recommended use of this branch

Use `kexit-hint` for:

- validating the **generic K-exit / C-class code path**
- comparing **3-exit vs 5-exit** fairly under the same greedy pipeline
- comparing **5-exit greedy no-hint vs 5-exit greedy hint passing**
- documenting the transition from a fixed 3-exit baseline to a reusable early-exit framework
- preparing later work on:
  - EA for K-exit
  - improved hint passing
  - improved clip-level stopping rules

---

## Historical note

- **`v0.1.6` tag** should remain the frozen **historical 3-exit greedy baseline**
- **`kexit-hint`** is the current generic K-exit branch built on top of the earlier refactor work
- the current branch history includes a **5-exit greedy hint-passing experiment**, but that should be reported as an **experimental architectural branch**, not the current best model

This separation should be kept clear in all documentation and reporting.
