# `kexit-greedy` — Generic K-Exit Greedy No-Hint Baseline

This release documents the **`kexit-greedy`** branch, which refactors the historical **`v0.1.6`** line into a **generic K-exit / C-class** early-exit audio pipeline and validates the **greedy no-hint baseline** on both **3-exit** and **5-exit** settings.

## What this release is

The original **`v0.1.6`** tag remains the frozen **3-exit greedy baseline**. In contrast, **`kexit-greedy`** converts that code path into a reusable architecture so the same training, calibration, evaluation, profiling, and reporting pipeline can run either:

- **3 exits** with `tap_blocks=(1,3)`
- **5 exits** with `tap_blocks=(1,2,3,4)`

This release is specifically a **greedy no-hint baseline release**. The results here were obtained **without sequential exit-to-exit hint passing**.

## What was implemented

- Refactored the old fixed 3-exit path into a **generic K-exit / C-class** implementation
- Added dynamic support for configurable `tap_blocks`
- Generalized `TinyAudioCNN` and `ExitNet` for both 3-exit and 5-exit settings
- Updated training, calibration, threshold selection, policy testing, clip-policy testing, summarization, analysis, profiling, and reporting scripts to work with dynamic exit counts
- Standardized reporting for:
  - segment-level greedy policy
  - full-clip baseline
  - clip-wise **Depth×Time** evaluation
- Preserved the historical role of `v0.1.6` while creating a reusable baseline for later work such as **sequential hint passing**

## Validated run commands

### 3-exit greedy no-hint

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "3exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy
```

### 5-exit greedy no-hint

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "5exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy
```

## Main validated results

### Segment-level greedy policy

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

## Interpretation of the no-hint greedy baseline

These runs provide a clean greedy no-hint comparison between the **3-exit** configuration and the **5-exit** configuration. The 5-exit model improves **segment-level policy accuracy** from **0.9754** to **0.9846**, which shows that the additional deeper exits improve local decision quality. However, that gain comes with a higher average exit depth (**1.982 → 2.449**) and slightly worse internal stability, reflected by a higher flip-any rate and slightly lower exit consistency.

At the **full-clip** level, both models achieve **1.0000 clip accuracy**, but the 5-exit model requires more compute (**29.273 → 36.182** average compute units). Under **Depth×Time**, both models again achieve the same clip accuracy (**1.0000**), use the same average number of windows (**2.045 / 14.773**), and save the same proportion of windows (**86.15%**). However, the 5-exit model still uses more compute per used window (**5.364 → 7.045**) and therefore provides lower compute savings (**81.68% → 80.53%**).

The honest interpretation is that, **without sequential hint passing**, moving from 3 exits to 5 exits improves **segment-level decision quality** but does **not** improve **clip-level time-aware efficiency** under the current greedy stopping rule. This makes the comparison scientifically useful:

- **3-exit greedy** is the stronger **efficiency baseline**
- **5-exit greedy** is the stronger **segment-quality / deeper-head baseline**

This is exactly why later **sequential hint passing** experiments are motivated: the 5-exit no-hint baseline already has stronger later exits, but those stronger exits are not yet being converted into better early decisions or better clip-time efficiency.

## Main conclusions

1. The refactor is successful: the same pipeline now supports both **3-exit** and **5-exit** greedy no-hint runs through one generic implementation path.
2. The **5-exit** model improves **segment-level policy accuracy** and produces very strong deeper exits.
3. The **3-exit** model remains the better **efficiency baseline** under the current greedy **Depth×Time** stopping rule.
4. The extra exits in the 5-exit model improve representational quality, but without sequential hint passing they do **not yet** improve clip-level time-aware efficiency.
5. This release provides the correct baseline for later work on **sequential hint passing**, **EA for K-exit**, and stronger clip-level stopping strategies.

## Scope of this release

### Included

- greedy segment policy
- greedy clip-wise Depth×Time policy
- dynamic K-exit architecture
- validated **3-exit** and **5-exit** greedy no-hint baselines
- dynamic profiling and reporting
- binary classification

### Not included as a solved result

- sequential hint-passing results in this release note
- EA retuning for the K-exit release note
- multiclass experiments
- a 5-exit time policy that clearly beats the 3-exit baseline on clip-level efficiency

## Recommended description of this release

> A successful **generic K-exit / C-class refactor** of the historical `v0.1.6` greedy baseline, validated under a clean **3-exit vs 5-exit greedy no-hint** comparison. The 5-exit setting improves segment-level decision quality, while the 3-exit setting remains the stronger clip-level efficiency baseline under the current greedy Depth×Time rule.
