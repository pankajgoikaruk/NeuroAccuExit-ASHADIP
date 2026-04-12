# ASHADIP / NeuroAccuExit — `kexit-greedy` Generic K-Exit Greedy No-Hint Audio Pipeline

This branch documents the **`kexit-greedy`** refactor and validation pass built from the historical **`v0.1.6`** line.

The original `v0.1.6` tag remains the **frozen 3-exit greedy baseline**. In contrast, `kexit-greedy` converts that code path into a **generic K-exit / C-class structure** so that the same architecture and evaluation pipeline can run either:

- **3 exits** with `tap_blocks=(1,3)`
- **5 exits** with `tap_blocks=(1,2,3,4)`

This README documents the **greedy no-hint baseline comparison only**. The results below were obtained **without sequential exit-to-exit hint passing**.

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

The main objective of `kexit-greedy` is to:

1. preserve the historical 3-exit greedy behavior
2. add a reusable **K-exit / C-class** implementation
3. validate both **3-exit** and **5-exit** greedy no-hint runs end-to-end
4. compare the resulting accuracy / efficiency trade-offs honestly
5. establish a clean baseline before testing **sequential hint passing**

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

### 3-exit greedy no-hint run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data\moth_sounds" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "3exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -NMels 64 `
  -TapBlocks "1,3" `
  -RunClipPolicy `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3 `
  -TimeMargin 0.0
```

### 5-exit greedy no-hint run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data\moth_sounds" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "5exit_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -NMels 64 `
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

Each run directory such as `runs/3exit_greedy/3exit_greedy_001/` or `runs/5exit_greedy/5exit_greedy_001/` contains:

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

## Interpretation of the no-hint greedy baseline

These runs provide a clean no-hint greedy comparison between the **3-exit** configuration (`tap_blocks=(1,3)`) and the **5-exit** configuration (`tap_blocks=(1,2,3,4)`). The 5-exit model improves **segment-level policy accuracy** from **0.9754** to **0.9846**, which shows that the additional later exits improve local decision quality. However, that gain comes with a higher average exit depth (**1.982 → 2.449**) and slightly worse internal stability, seen in the higher flip-any rate (**0.1785 → 0.2215**) and slightly lower exit consistency (**1.0000 → 0.9969**).

At the **full-clip** level, both models already achieve **1.0000 clip accuracy**, but the 5-exit model requires more compute (**29.273 → 36.182** average compute units). Under **Depth×Time** evaluation, both models again achieve the same clip accuracy (**1.0000**), use the same average number of windows (**2.045 / 14.773**), and save the same proportion of windows (**86.15%**). However, the 5-exit model still uses more compute per used window (**5.364 → 7.045**) and therefore provides lower compute savings (**81.68% → 80.53%**).

The honest interpretation is that, **without sequential hint passing**, moving from **3 exits** to **5 exits** improves **segment-level decision quality** but does **not** improve **clip-level time-aware efficiency** under the current greedy stopping rule. This makes the baseline scientifically useful: **3-exit greedy** is the stronger **efficiency baseline**, while **5-exit greedy** is the stronger **decision-quality baseline**. This is also the motivation for **sequential hint passing**, whose goal is to make the additional exits improve earlier decisions rather than simply increase compute.

---

## Excel-aligned master results tables

The following two tables mirror the spreadsheet structure used for this branch. They preserve the same result blocks:

- **Segment Random Greedy Policy Test**
- **Full-Clip Sequential Greedy Policy Test**
- **Depth × Time Clip Greedy Policy Test**
- **Key Change**

> **Note:** `3-window diagnostic (K=3)` refers to **three windows per clip**, not three exits.

### Master table — 3-exit greedy (`tap_blocks=(1,3)`)

<table>
  <tr>
    <th>Metric</th>
    <th>Segment Random Greedy Policy Test</th>
    <th>Full-Clip Sequential Greedy Policy Test</th>
    <th>Depth × Time Clip Greedy Policy Test</th>
    <th>Key Change</th>
  </tr>
  <tr>
    <td>PowerShell Command</td>
    <td><code>powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `<br>-Variant "3exit_greedy" `<br>-Policy "greedy" `<br>-Device "cpu" `<br>-TapBlocks "1,3" `<br>-RunClipPolicy</code></td>
    <td>—</td>
    <td>—</td>
    <td><strong>Strong no-hint greedy efficiency baseline.</strong> Achieves high segment accuracy (97.54%) and perfect clip accuracy with lower average compute than 5-exit. Best current baseline for efficiency-focused comparison.</td>
  </tr>
  <tr><td>Files</td><td>Train: 99, Test: 22, Val: 21</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Segments</td><td>Train: 1646, Test: 325, Val: 246</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy</td><td>Greedy</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Device</td><td>CPU</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>SegmentSec</td><td>1.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>HopSec</td><td>0.5</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>NMels</td><td>64</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit1 Accuracy</td><td>0.8338</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit2 Accuracy</td><td>0.9385</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit3 Accuracy</td><td>0.9754</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy test accuracy</td><td>0.9754 (n_segments=325)</td><td>0.9754 (n_segments=325)</td><td>0.9778 (n_segments=45)</td><td>—</td></tr>
  <tr><td>Avg exit depth</td><td>1.982</td><td>1.982</td><td>2.622</td><td>—</td></tr>
  <tr><td>Exit mix</td><td>e1=0.3631, e2=0.2923, e3=0.3446</td><td>e1=0.3631, e2=0.2923, e3=0.3446</td><td>e1=0.0889, e2=0.2000, e3=0.7111</td><td>—</td></tr>
  <tr><td>Flip-rate (any flip)</td><td>0.1785</td><td>0.1785 (used windows)</td><td>0.4000 (used windows)</td><td>—</td></tr>
  <tr><td>Avg flip-count</td><td>0.2031</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Exit-consistency (taken==final)</td><td>1.0000</td><td>1.0000 (used windows)</td><td>1.0000 (used windows)</td><td>—</td></tr>
  <tr><td>Windows Saved (%)</td><td>0.00%</td><td>0.00%</td><td>86.15%</td><td>—</td></tr>
  <tr><td>Compute Saved (%)</td><td>0.00%</td><td>0.00%</td><td>81.68%</td><td>—</td></tr>
  <tr><td>Window distribution mode</td><td>N/A</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td><strong>Clip-metrics (segment-policy, for fair comparison)</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Clip accuracy</td><td>N/A</td><td>1.0000 (n_clips=22)</td><td>1.0000 (n_clips=22)</td><td>—</td></tr>
  <tr><td>Avg windows used</td><td>N/A</td><td>14.773 / 14.773 (100.00%)</td><td>2.045 / 14.773</td><td>—</td></tr>
  <tr><td>Avg compute units (sum depth over used windows)</td><td>N/A</td><td>29.273</td><td>5.364</td><td>—</td></tr>
  <tr><td><strong>Fixed-position diagnostic (independent of time-exit)</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_firstK</td><td>N/A</td><td>0.9538 (n_segments=65)</td><td>0.9538 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_midK</td><td>N/A</td><td>0.9538 (n_segments=65)</td><td>0.9538 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_lastK</td><td>N/A</td><td>0.9692 (n_segments=65)</td><td>0.9692 (n_segments=65)</td><td>—</td></tr>
  <tr><td><strong>Stop-speed group diagnostic (Depth×Time only; first-K accuracy)</strong></td><td>—</td><td>—</td><td>Not populated in this run</td><td>—</td></tr>
  <tr><td>stop_1</td><td>N/A</td><td>N/A</td><td>—</td><td>—</td></tr>
  <tr><td>stop_2</td><td>N/A</td><td>N/A</td><td>—</td><td>—</td></tr>
  <tr><td>stop_3</td><td>N/A</td><td>N/A</td><td>—</td><td>—</td></tr>
  <tr><td><strong>Confusion matrix (clip-level)</strong></td><td>N/A</td><td>[[12, 0], [0, 10]]</td><td>[[12, 0], [0, 10]]</td><td>—</td></tr>
  <tr><td>Per-class: Female</td><td>N/A</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 12</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 12</td><td>—</td></tr>
  <tr><td>Per-class: Male</td><td>N/A</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 10</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 10</td><td>—</td></tr>
  <tr><td>Total Time</td><td>207.38 seconds (~3.46 minutes)</td><td>—</td><td>—</td><td>—</td></tr>
</table>

### Master table — 5-exit greedy (`tap_blocks=(1,2,3,4)`)

<table>
  <tr>
    <th>Metric</th>
    <th>Segment Random Greedy Policy Test</th>
    <th>Full-Clip Sequential Greedy Policy Test</th>
    <th>Depth × Time Clip Greedy Policy Test</th>
    <th>Key Change</th>
  </tr>
  <tr>
    <td>PowerShell Command</td>
    <td><code>powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `<br>-Variant "5exit_greedy" `<br>-Policy "greedy" `<br>-Device "cpu" `<br>-TapBlocks "1,2,3,4" `<br>-RunClipPolicy</code></td>
    <td>—</td>
    <td>—</td>
    <td><strong>Extends the model from 3 exits to 5 exits without sequential hint passing.</strong> Improves segment-level greedy accuracy (98.46%) and provides stronger deeper exits, but does not improve clip accuracy or window saving under Depth×Time, while increasing compute cost.</td>
  </tr>
  <tr><td>Files</td><td>Train: 99, Test: 22, Val: 21</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Segments</td><td>Train: 1646, Test: 325, Val: 246</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy</td><td>Greedy</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Device</td><td>CPU</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>SegmentSec</td><td>1.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>HopSec</td><td>0.5</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>NMels</td><td>64</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit1 Accuracy</td><td>0.8400</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit2 Accuracy</td><td>0.8400</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit3 Accuracy</td><td>0.9600</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit4 Accuracy</td><td>0.9846</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit5 Accuracy</td><td>0.9877</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy test accuracy</td><td>0.9846 (n_segments=325)</td><td>0.9846 (n_segments=325)</td><td>0.9778 (n_segments=45)</td><td>—</td></tr>
  <tr><td>Avg exit depth</td><td>2.449</td><td>2.449</td><td>3.444</td><td>—</td></tr>
  <tr><td>Exit mix</td><td>e1=0.3815, e2=0.1323, e3=0.2308, e4=0.1662, e5=0.0892</td><td>e1=0.3815, e2=0.1323, e3=0.2308, e4=0.1662, e5=0.0892</td><td>e1=0.0889, e2=0.0667, e3=0.3556, e4=0.2889, e5=0.2000</td><td>—</td></tr>
  <tr><td>Flip-rate (any flip)</td><td>0.2215</td><td>0.2215 (used windows)</td><td>0.4222 (used windows)</td><td>—</td></tr>
  <tr><td>Avg flip-count</td><td>0.2954</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Exit-consistency (taken==final)</td><td>0.9969</td><td>0.9969 (used windows)</td><td>1.0000 (used windows)</td><td>—</td></tr>
  <tr><td>Windows Saved (%)</td><td>0.00%</td><td>0.00%</td><td>86.15%</td><td>—</td></tr>
  <tr><td>Compute Saved (%)</td><td>0.00%</td><td>0.00%</td><td>80.53%</td><td>—</td></tr>
  <tr><td>Window distribution mode</td><td>N/A</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td><strong>Clip-metrics (segment-policy, for fair comparison)</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Clip accuracy</td><td>N/A</td><td>1.0000 (n_clips=22)</td><td>1.0000 (n_clips=22)</td><td>—</td></tr>
  <tr><td>Avg windows used</td><td>N/A</td><td>14.773 / 14.773 (100.00%)</td><td>2.045 / 14.773</td><td>—</td></tr>
  <tr><td>Avg compute units (sum depth over used windows)</td><td>N/A</td><td>36.182</td><td>7.045</td><td>—</td></tr>
  <tr><td><strong>Fixed-position diagnostic (independent of time-exit)</strong></td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_firstK</td><td>N/A</td><td>0.9692 (n_segments=65)</td><td>0.9692 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_midK</td><td>N/A</td><td>0.9846 (n_segments=65)</td><td>0.9846 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_lastK</td><td>N/A</td><td>0.9538 (n_segments=65)</td><td>0.9538 (n_segments=65)</td><td>—</td></tr>
  <tr><td><strong>Stop-speed group diagnostic (Depth×Time only; first-K accuracy)</strong></td><td>—</td><td>—</td><td>Not populated in this run</td><td>—</td></tr>
  <tr><td>stop_2</td><td>N/A</td><td>N/A</td><td>—</td><td>—</td></tr>
  <tr><td>stop_3_4</td><td>N/A</td><td>N/A</td><td>—</td><td>—</td></tr>
  <tr><td>stop_5_plus</td><td>N/A</td><td>N/A</td><td>—</td><td>—</td></tr>
  <tr><td><strong>Confusion matrix (clip-level)</strong></td><td>N/A</td><td>[[12, 0], [0, 10]]</td><td>[[12, 0], [0, 10]]</td><td>—</td></tr>
  <tr><td>Per-class: Female</td><td>N/A</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 12</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 12</td><td>—</td></tr>
  <tr><td>Per-class: Male</td><td>N/A</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 10</td><td>Precision 1.0, Recall 1.0, F1 1.0, Support 10</td><td>—</td></tr>
  <tr><td>Total Time</td><td>217.69 seconds (~3.63 minutes)</td><td>—</td><td>—</td><td>—</td></tr>
</table>

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

### 4. Core scientific interpretation

The most honest interpretation of these results is:

> **Without sequential hint passing, 5 exits improve segment-level decision quality, but do not improve clip-level efficiency under the current greedy Depth×Time stopping rule.**

Both variants already reach **1.0000 clip accuracy** on the current test set, so the more informative comparison is the trade-off between:

- segment-level policy quality
- average exit depth
- compute per used window
- compute saved under Depth×Time

In this comparison:

- **3-exit greedy** is the stronger **efficiency baseline**
- **5-exit greedy** is the stronger **segment-quality baseline**

This is exactly why the branch matters: it shows that simply adding more exits is not enough. A stronger mechanism is needed to turn those extra exits into genuinely useful earlier decisions.

### 5. Best reviewer-safe answer

If a reviewer asks:

> “If greedy no-hint already works better for efficiency, why use sequential hint passing?”

the clean answer is:

> **Because the 5-exit no-hint baseline already shows that extra exits improve segment decision quality, but they do not yet improve clip-time efficiency by themselves. Sequential hint passing is therefore introduced to make those extra exits more useful, so that the added depth can improve earlier decisions instead of only adding compute.**

---

## Current scope

### Included in this branch

- greedy segment policy
- greedy clip-wise Depth×Time policy
- dynamic K-exit architecture
- 3-exit and 5-exit greedy no-hint validation
- dynamic profiling and reporting
- binary classification

### Not yet included as a solved result

- EA policy tuning for the K-exit branch
- sequential hint passing results in this branch
- multiclass experiments
- a 5-exit time policy that clearly beats the 3-exit baseline on clip-level efficiency

---

## Recommended use of this branch

Use `kexit-greedy` for:

- validating the **generic K-exit / C-class code path**
- comparing **3-exit vs 5-exit** fairly under the same greedy no-hint pipeline
- documenting the transition from a fixed 3-exit baseline to a reusable early-exit framework
- preparing later work on:
  - sequential hint passing
  - EA for K-exit
  - improved clip-level stopping rules

---

## Historical note

- **`v0.1.6` tag** should remain the frozen historical **3-exit greedy baseline**
- **`kexit-greedy`** is the refactored greedy no-hint branch that generalizes that line into a reusable K-exit implementation

This separation should be kept clear in all documentation and reporting.
