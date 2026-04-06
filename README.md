# ASHADIP / NeuroAccuExit — kexit-dev Generic K-Exit Audio Pipeline

This branch documents the **`kexit-dev` refactor** built from the historical **`v0.1.6`** line.

The original `v0.1.6` tag remains the **locked 3-exit greedy baseline**. In contrast, `kexit-dev` converts that code path into a **generic K-exit / C-class structure**, so the same model and evaluation pipeline can run either:

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

---

## Why this branch exists

The old `v0.1.6` code path was still hardcoded around a **3-exit** model. This branch refactors that line into a reusable structure that can support multiple exit counts without rewriting the architecture or downstream scripts each time.

The main objective of `kexit-dev` is:

1. preserve the historical 3-exit baseline behavior
2. add a reusable **K-exit / C-class** implementation
3. validate both **3-exit** and **5-exit** runs end-to-end
4. compare the resulting accuracy / efficiency trade-offs honestly

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

Core code path updated for dynamic K-exit support:

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
- `utils/profiling.py`
- `scripts/run_full.ps1`
- `scripts/run_reports.ps1`
- `scripts/compare_variants.py`
- `scripts/variants_to_latex.py`
- `scripts/ondevice_to_latex.py`
- `scripts/analysis_to_latex.py`

---

## Validated run settings

### 3-exit reference run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_dev" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy
```

### 5-exit run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "kexit_dev" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy
```

---

## Main run artifacts

Each run directory such as `runs/kexit_dev/kexit_dev_002/` contains:

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

### Segment-level greedy policy comparison

| Setting | Tau | Policy acc | Avg exit depth | Flip-any | Exit consistency |
|---|---:|---:|---:|---:|---:|
| 3-exit (`1,3`) | 0.95 | 0.9754 | 1.982 | 0.1785 | 1.0000 |
| 5-exit (`1,2,3,4`) | 0.92 | 0.9846 | 2.449 | 0.2215 | 0.9969 |

### Per-exit test accuracy

| Setting | Exit1 | Exit2 | Exit3 | Exit4 | Exit5 |
|---|---:|---:|---:|---:|---:|
| 3-exit (`1,3`) | 0.8338 | 0.9385 | 0.9754 | -- | -- |
| 5-exit (`1,2,3,4`) | 0.8400 | 0.8400 | 0.9600 | 0.9846 | 0.9877 |

### Depth×Time comparison

| Setting | Clip acc | Used-win acc | Avg windows used | Windows saved | Avg compute units | Compute saved |
|---|---:|---:|---:|---:|---:|---:|
| 3-exit (`1,3`) | 1.0000 | 0.9778 | 2.045 | 86.15% | 5.364 | 81.68% |
| 5-exit (`1,2,3,4`) | 1.0000 | 0.9778 | 2.045 | 86.15% | 7.045 | 80.53% |

---

## Main conclusions from the comparison

### 1. The refactor is successful

The same pipeline now runs both:

- **3-exit**
- **5-exit**

through one generic implementation path.

### 2. The 5-exit model improves segment-level decision quality

Compared with the 3-exit run, the 5-exit run improved greedy segment-policy accuracy from:

- **0.9754 → 0.9846**

The strongest gains came from the deeper exits:

- `exit4 = 0.9846`
- `exit5 = 0.9877`

### 3. The 5-exit model is not yet better for Depth×Time efficiency

Under the current greedy Depth×Time rule, the 5-exit model did **not** improve:

- clip accuracy
- windows used
- windows saved

and it increased compute per used window:

- **5.364 → 7.045** average compute units

So the extra exits currently provide **stronger deep decisions**, but not yet **better clip-level time-aware efficiency**.

### 4. Current best 5-exit Depth×Time setting

Among the tested 5-exit time-policy settings, the best remained the original:

- `time_conf=0.95`
- `time_stable_k=2`
- `time_min_windows=2`
- `time_margin=0.00`

Increasing `time_stable_k` from 2 to 3 reduced flip-rate slightly, but made the policy clearly less efficient:

- avg windows used: **2.045 → 3.000**
- compute saved: **80.53% → 73.12%**

Changing `time_conf` from `0.95` to `0.97` and adding `time_margin=0.05` did not provide additional benefit.

---

## Recommended interpretation of this branch

This branch should be described as:

> a successful **generic K-exit / C-class refactor** of the historical v0.1.6 line, validated on both 3-exit and 5-exit configurations.

The most honest scientific interpretation is:

- **3 exits** remains the better configuration for the current greedy **Depth×Time efficiency** setting
- **5 exits** improves **segment-level policy accuracy** and provides stronger deep exits
- **5 exits still needs policy retuning or a stronger stopping rule** to translate that extra depth into better clip-level efficiency

---

## Current scope

### Included in this branch

- greedy segment policy
- greedy clip-wise Depth×Time policy
- dynamic K-exit architecture
- 3-exit and 5-exit validation
- dynamic profiling and reporting
- binary classification

### Not yet included as a solved result

- EA policy tuning for the K-exit branch
- hint passing in this branch
- multiclass experiments
- a 5-exit time policy that clearly beats the 3-exit baseline on clip-level efficiency

---

## Recommended use of this branch

Use `kexit-dev` for:

- validating the **generic K-exit / C-class code path**
- comparing **3-exit vs 5-exit** fairly under the same greedy pipeline
- documenting the transition from a fixed 3-exit baseline to a reusable early-exit framework
- preparing later work on:
  - EA for K-exit
  - hint passing
  - improved clip-level stopping rules

---

## Historical note

- **`v0.1.6` tag** should remain the frozen **historical 3-exit greedy baseline**
- **`kexit-dev`** is the refactored branch that generalizes that line into a reusable K-exit implementation

This separation should be kept clear in all documentation and reporting.
