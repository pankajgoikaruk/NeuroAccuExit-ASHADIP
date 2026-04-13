# ASHADIP / NeuroAccuExit — `kexit-greedy-hint` Generic K-Exit Greedy + Hint Audio Pipeline

This branch documents the corrected **`kexit-greedy-hint`** workflow built on top of the historical **`v0.1.6`** line.

The original `v0.1.6` tag remains the frozen **3-exit greedy baseline**. In contrast, `kexit-greedy-hint` provides a reusable **generic K-exit / C-class** implementation that supports both:

- **3 exits** with `tap_blocks=(1,3)`
- **5 exits** with `tap_blocks=(1,2,3,4)`

and both:

- **greedy no-hint**
- **greedy hint-enabled**

The task remains:

- binary moth wingbeat gender classification (**male** vs **female**)
- log-mel spectrogram input
- **TinyAudioCNN + ExitNet** early-exit model family

This branch now supports:

- dynamic backbone taps
- dynamic `ExitNet` heads
- dynamic training / evaluation / calibration loops
- dynamic greedy threshold selection
- dynamic segment-policy evaluation
- dynamic clip-wise **Depth×Time** evaluation
- optional **sequential exit-to-exit hint passing**
- CLI control of hint / no-hint through `-ExitHint "true|false"`
- corrected saving of the effective run config into `config_used.yaml`

---

## Why this branch exists

The old `v0.1.6` line was still hardcoded around a **3-exit** model. This branch generalizes that path so the same codebase can compare:

- compact vs deeper exit hierarchies
- no-hint vs hint-enabled runs
- segment-level vs clip-level behavior
- accuracy vs compute trade-offs

The branch also fixes the earlier configuration mismatch that affected the provisional 5-exit hinted setup. The current results are based on the corrected dynamic configuration.

---

## Main architectural change

### Before
The historical line assumed:

- a fixed 3-exit backbone
- a fixed 3-head `ExitNet`
- training / eval loops written around `range(3)`
- fixed 3-exit configuration assumptions in downstream scripts

### Now
The branch supports a **generic K-exit structure**:

- `TinyAudioCNN` exposes configurable `tap_blocks`
- `ExitNet` builds one head per tap plus one final head
- total exits = `len(tap_blocks) + 1`

Validated settings:

- `tap_blocks=(1,3)` → **3 exits**
- `tap_blocks=(1,2,3,4)` → **5 exits**

The branch also supports optional **sequential exit-to-exit hint passing**:

- later exits may consume a small hint derived from the previous exit
- hint can be switched on/off from the CLI
- the same shared YAML can be used for all four validated runs

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

## Validated run settings

### 3-exit no-hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "3exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -ExitHint "false" `
  -RunClipPolicy
```

### 3-exit hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "3exit_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -ExitHint "true" `
  -RunClipPolicy
```

### 5-exit no-hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "5exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -ExitHint "false" `
  -RunClipPolicy
```

### 5-exit hint
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

## Main run artifacts

Each run directory contains the usual outputs:

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
- `windows_used_hist.json`
- `meta.json`
- plots under `plots/`

---

## Validated results and findings

### Per-exit test accuracy

| Setting | Exit1 | Exit2 | Exit3 | Exit4 | Exit5 |
|---|---:|---:|---:|---:|---:|
| `3exit_greedy` | 0.8338 | 0.9385 | 0.9754 | — | — |
| `5exit_greedy` | 0.8369 | 0.8892 | 0.9723 | 0.9754 | 0.9908 |
| `3exit_greedy_hint` | 0.8369 | 0.9662 | 0.9908 | — | — |
| `5exit_greedy_hint` | 0.8308 | 0.8646 | 0.9231 | 0.9538 | 0.9692 |

### Segment-level results table

| Setting | Hint | Policy acc | Avg exit depth | Flip-any rate | Avg flip-count | Exit consistency | Best interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| `3exit_greedy` | No | 0.9754 | 1.982 | 0.1785 | 0.2031 | 1.0000 | Strong simple no-hint efficiency baseline |
| `5exit_greedy` | No | 0.9908 | 2.637 | 0.1908 | 0.2338 | 1.0000 | Best deep-capacity no-hint baseline |
| `3exit_greedy_hint` | Yes | 0.9908 | 1.895 | 0.1908 | 0.2154 | 1.0000 | Best overall efficiency-quality tradeoff |
| `5exit_greedy_hint` | Yes | 0.9723 | 2.465 | 0.2123 | 0.2492 | 0.9908 | Hint not yet beneficial in current 5-exit setup |

### Clip-based results table

| Setting | Hint | Clip mode | Clip acc | Used-window acc | Avg depth per used window | Avg windows used | Flip rate | Windows saved | Avg compute units | Compute saved | Best interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3exit_greedy` | No | Full-clip | 1.0000 | 0.9754 | 1.982 | 14.773 / 14.773 | 0.1785 | 0.00% | 29.273 | 0.00% | Full-window baseline for 3-exit no-hint |
| `3exit_greedy` | No | Depth×Time | 1.0000 | 0.9778 | 2.622 | 2.045 / 14.773 | 0.4000 | 86.15% | 5.364 | 81.68% | Strong no-hint clip efficiency baseline |
| `5exit_greedy` | No | Full-clip | 1.0000 | 0.9908 | 2.637 | 14.773 / 14.773 | 0.1908 | 0.00% | 38.955 | 0.00% | Strong deep no-hint full-window baseline |
| `5exit_greedy` | No | Depth×Time | 1.0000 | 0.9778 | 3.578 | 2.045 / 14.773 | 0.4222 | 86.15% | 7.318 | 80.53% | Best deep-capacity no-hint clip baseline |
| `3exit_greedy_hint` | Yes | Full-clip | 1.0000 | 0.9908 | 1.895 | 14.773 / 14.773 | 0.1908 | 0.00% | 28.000 | 0.00% | Best compact hinted full-window baseline |
| `3exit_greedy_hint` | Yes | Depth×Time | 1.0000 | 1.0000 | 2.477 | 2.000 / 14.773 | 0.3864 | 86.46% | 4.955 | 82.31% | Best overall efficiency-quality clip result |
| `5exit_greedy_hint` | Yes | Full-clip | 1.0000 | 0.9723 | 2.465 | 14.773 / 14.773 | 0.2123 | 0.00% | 36.409 | 0.00% | Corrected 5-exit hinted full-window result |
| `5exit_greedy_hint` | Yes | Depth×Time | 1.0000 | 0.9783 | 3.565 | 2.091 / 14.773 | 0.4565 | 85.85% | 7.455 | 79.53% | Hint still not beneficial in current 5-exit setup |

---

## Short interpretation of the corrected four-run comparison

The corrected four-run comparison now gives a much cleaner and more credible research story. **`3exit_greedy_hint`** and **`5exit_greedy`** tie for the best segment-level greedy policy accuracy at **0.9908**, but they reach that result through different trade-offs. **`3exit_greedy_hint`** provides the best overall **efficiency-quality tradeoff**, with lower average exit depth, lower full-clip and Depth×Time compute, perfect clip accuracy, and the strongest compute saving. In contrast, **`5exit_greedy`** is now the strongest **deep-capacity no-hint baseline**, with very strong deeper exits and matched best segment accuracy, but it remains more compute-expensive than the compact hinted model.

The most important negative result is that **`5exit_greedy_hint`** still does not improve the deeper greedy pipeline under the current design. Its policy accuracy remains lower than `5exit_greedy`, its deeper exits are weaker, and its Depth×Time compute is not competitive with either `3exit_greedy_hint` or `5exit_greedy`. Therefore, the corrected evidence supports the claim that **sequential hint passing works very well in the compact 3-exit setting, but does not yet benefit the deeper 5-exit greedy setting**.

### Reviewer-safe conclusion

- **`3exit_greedy_hint`** = best **efficiency-quality** result
- **`5exit_greedy`** = best **deep-capacity no-hint** result
- **`5exit_greedy_hint`** = not yet beneficial under the current setup

This is a strong and balanced branch-level conclusion because it shows both a clear success case and a clear limitation.

---

## Current research takeaway

This branch should now be described as:

> a corrected and reusable **generic K-exit / C-class** greedy branch with **CLI-controlled hint vs no-hint**, validated across four runs and showing that hint passing is highly effective in the **3-exit** setting but not yet beneficial in the current **5-exit** greedy setting.

That is the cleanest documentation position for `kexit-greedy-hint` right now.


---

## Additional comparison tables from workbook draft

The workbook includes several useful comparison tables. They are worth keeping, but a few names are clearer after minor revision and all difference columns should be computed explicitly.

### Recommended table names

- **Segment-level greedy policy comparison**
- **Per-exit test accuracy comparison**
- **Exit mix comparison (segment policy and Depth×Time)**
- **Depth×Time comparison: no-hint vs hint passing**
- **Full-clip vs Depth×Time accuracy-efficiency tradeoff**

> In the tables below, **Δ** means:
> - **Hint−No-Hint** for Tables 1–4
> - **Depth×Time−Full** for Table 5

### 1) Segment-level greedy policy comparison

| Metric | 3exit No-Hint | 3exit Hint | Δ (Hint−No-Hint) | 5exit No-Hint | 5exit Hint | Δ (Hint−No-Hint) |
|---|---:|---:|---:|---:|---:|---:|
| Policy accuracy | 0.9754 | 0.9908 | +0.0154 | 0.9908 | 0.9723 | -0.0185 |
| Avg exit depth | 1.982 | 1.895 | -0.0870 | 2.637 | 2.465 | -0.1720 |
| Flip-any rate | 0.1785 | 0.1908 | +0.0123 | 0.1908 | 0.2123 | +0.0215 |
| Avg flip count | 0.2031 | 0.2154 | +0.0123 | 0.2338 | 0.2492 | +0.0154 |
| Exit consistency | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 0.9908 | -0.0092 |

### 2) Per-exit test accuracy comparison

| Metric | 3exit No-Hint | 3exit Hint | Δ (Hint−No-Hint) | 5exit No-Hint | 5exit Hint | Δ (Hint−No-Hint) |
|---|---:|---:|---:|---:|---:|---:|
| Exit1 accuracy | 0.8338 | 0.8369 | +0.0031 | 0.8369 | 0.8308 | -0.0061 |
| Exit2 accuracy | 0.9385 | 0.9662 | +0.0277 | 0.8892 | 0.8646 | -0.0246 |
| Exit3 accuracy | 0.9754 | 0.9908 | +0.0154 | 0.9723 | 0.9231 | -0.0492 |
| Exit4 accuracy | — | — | — | 0.9754 | 0.9538 | -0.0216 |
| Exit5 accuracy | — | — | — | 0.9908 | 0.9692 | -0.0216 |

### 3) Exit mix comparison (segment policy and Depth×Time)

#### 3-exit

| Exit | 3exit No-Hint Segment | 3exit Hint Segment | Δ | 3exit No-Hint Depth×Time | 3exit Hint Depth×Time | Δ |
|---|---:|---:|---:|---:|---:|---:|
| e1 | 0.3631 | 0.3846 | +0.0215 | 0.0889 | 0.0909 | +0.0020 |
| e2 | 0.2923 | 0.3354 | +0.0431 | 0.2000 | 0.3409 | +0.1409 |
| e3 | 0.3446 | 0.2800 | -0.0646 | 0.7111 | 0.5680 | -0.1431 |

#### 5-exit

| Exit | 5exit No-Hint Segment | 5exit Hint Segment | Δ | 5exit No-Hint Depth×Time | 5exit Hint Depth×Time | Δ |
|---|---:|---:|---:|---:|---:|---:|
| e1 | 0.3569 | 0.3723 | +0.0154 | 0.0889 | 0.0870 | -0.0019 |
| e2 | 0.0308 | 0.1354 | +0.1046 | 0.0000 | 0.0652 | +0.0652 |
| e3 | 0.3538 | 0.2677 | -0.0861 | 0.3778 | 0.3043 | -0.0735 |
| e4 | 0.1354 | 0.1046 | -0.0308 | 0.3111 | 0.2826 | -0.0285 |
| e5 | 0.1231 | 0.1200 | -0.0031 | 0.2222 | 0.2609 | +0.0387 |

### 4) Depth×Time comparison: no-hint vs hint passing

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

### 5) Full-clip vs Depth×Time accuracy-efficiency tradeoff

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

