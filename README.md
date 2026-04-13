# ASHADIP / NeuroAccuExit — `kexit-greedy-hint` Generic K-Exit Greedy + Hint Audio Pipeline

This branch documents the **`kexit-greedy-hint`** line, which now contains the corrected and unified workflow for:

- **3 exits** with `tap_blocks=(1,3)`
- **5 exits** with `tap_blocks=(1,2,3,4)`
- **greedy no-hint** runs
- **greedy hint-enabled** runs

The original `v0.1.6` tag remains the frozen historical **3-exit greedy baseline**. In contrast, `kexit-greedy-hint` is the current reusable branch for **generic K-exit / C-class**, **CLI-controlled hint vs no-hint**, and the corrected four-run comparison.

---

## What this branch is

This branch provides a single code path that can run all four validated variants:

- `3exit_greedy`
- `5exit_greedy`
- `3exit_greedy_hint`
- `5exit_greedy_hint`

The branch now supports:

- dynamic backbone taps via `tap_blocks`
- dynamic `ExitNet` heads
- dynamic reconstruction of **3-exit** and **5-exit** models
- dynamic reconstruction of **hint** and **no-hint** models
- dynamic loss weights based on the actual number of exits
- CLI control of `exit_hint.enable` through `-ExitHint "true|false"`
- corrected saving of effective run config into `config_used.yaml`
- segment policy, full-clip, and Depth×Time evaluation
- reporting, profiling, and LaTeX table generation

---

## What was implemented / corrected

### Generic K-exit support
- `TinyAudioCNN` exposes configurable `tap_blocks`
- total exits are derived from `len(tap_blocks) + 1`

### Hint / no-hint control from CLI
The workflow no longer requires manual YAML edits for every run.  
You now choose hinting from the command line:

- `-ExitHint "false"` → no-hint
- `-ExitHint "true"` → hint-enabled

### Correct effective config saving
The training code now:
1. builds the model first
2. derives the actual number of exits
3. resolves dynamic loss weights
4. saves the effective `config_used.yaml`

This matters because earlier 5-exit hinted experiments were affected by a config mismatch between the declared exit count and the actual built model. That issue has now been cleaned up.

---

## Exact validated run commands

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

## Main results table

| Setting | Hint | Policy acc | Avg exit depth | Full-clip avg compute | Depth×Time used-win acc | Depth×Time avg windows | Depth×Time avg compute | Depth×Time compute saved | Best interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `3exit_greedy` | No | 0.9754 | 1.982 | 29.273 | 0.9778 | 2.045 / 14.773 | 5.364 | 81.68% | Strong simple no-hint efficiency baseline |
| `5exit_greedy` | No | 0.9908 | 2.637 | 38.955 | 0.9778 | 2.045 / 14.773 | 7.318 | 80.53% | Best deep-capacity no-hint baseline |
| `3exit_greedy_hint` | Yes | 0.9908 | 1.895 | 28.000 | 1.0000 | 2.000 / 14.773 | 4.955 | 82.31% | Best overall efficiency-quality tradeoff |
| `5exit_greedy_hint` | Yes | 0.9723 | 2.465 | 36.409 | 0.9783 | 2.091 / 14.773 | 7.455 | 79.53% | Hint not yet beneficial in current 5-exit setup |

### Per-exit test accuracy

| Setting | Exit1 | Exit2 | Exit3 | Exit4 | Exit5 |
|---|---:|---:|---:|---:|---:|
| `3exit_greedy` | 0.8338 | 0.9385 | 0.9754 | — | — |
| `5exit_greedy` | 0.8369 | 0.8892 | 0.9723 | 0.9754 | 0.9908 |
| `3exit_greedy_hint` | 0.8369 | 0.9662 | 0.9908 | — | — |
| `5exit_greedy_hint` | 0.8308 | 0.8646 | 0.9231 | 0.9538 | 0.9692 |

---

## Short interpretation of the corrected four-run comparison

The corrected four-run comparison now tells a much cleaner story than the earlier provisional results. **`3exit_greedy_hint`** and **`5exit_greedy`** tie for the best segment-level greedy policy accuracy at **0.9908**, but they reach that result in different ways. **`3exit_greedy_hint`** delivers the best overall **efficiency-quality tradeoff**, with the lowest full-clip compute among the high-accuracy models, the lowest Depth×Time compute (**4.955**), perfect clip accuracy, and the highest compute saving (**82.31%**). In contrast, **`5exit_greedy`** is now the strongest **deep-capacity no-hint baseline** after the corrected dynamic configuration, with very strong later exits and matched best segment accuracy, but it remains more compute-expensive than the compact 3-exit hinted model.

The most important negative result is that **`5exit_greedy_hint`** still does not improve the deeper greedy pipeline under the current design. Its segment policy accuracy falls to **0.9723**, its later exits are weaker than `5exit_greedy`, and its Depth×Time compute (**7.455**) is worse than both `3exit_greedy_hint` and `5exit_greedy`. This means the current evidence supports the claim that **sequential hint passing works very well in the compact 3-exit setting, but does not yet benefit the deeper 5-exit greedy setting**.

### Reviewer-safe conclusion

- **`3exit_greedy_hint`** = best **efficiency-quality** result
- **`5exit_greedy`** = best **deep-capacity no-hint** result
- **`5exit_greedy_hint`** = not yet beneficial under the current setup

This is a strong and balanced research story because it shows both a clear success case and a clear limitation.

---

## Excel-aligned master table section

The following tables mirror the updated workbook structure and keep the same blocks:

- Segment Random Greedy Policy Test
- Full-Clip Sequential Greedy Policy Test
- Depth × Time Clip Greedy Policy Test
- Key Change

> **Note:** `3-window diagnostic (K=3)` means **3 windows per clip**, not 3 exits.

### Master table — `3exit-greedy`

<table>
  <tr><th>Metric</th><th>Segment Random Greedy Policy Test</th><th>Full-Clip Sequential Greedy Policy Test</th><th>Depth × Time Clip Greedy Policy Test</th><th>Key Change</th></tr>
  <tr><td>PowerShell Command</td><td><code>powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `</code><br><code>-Variant "3exit_greedy" `</code><br><code>-Policy "greedy" `</code><br><code>-Device "cpu" `</code><br><code>-TapBlocks "1,3" `</code><br><code>-ExitHint "false" `</code><br><code>-RunClipPolicy</code></td><td>—</td><td>—</td><td>Strong no-hint greedy efficiency baseline. Achieves 97.54% segment policy accuracy and perfect clip accuracy with relatively low compute. Useful simple baseline, but now surpassed by 3exit_greedy_hint in both accuracy and Depth×Time efficiency.</td></tr>
  <tr><td>Files</td><td>Train: 99, Test: 22, Val: 21</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Segment</td><td>Train: 1646, Test: 325, Val: 246</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy</td><td>Greedy</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Device</td><td>CPU</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>SegmentSec</td><td>1.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>HopSec</td><td>0.5</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>NMels</td><td>64.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeConf</td><td>0.95</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeStableK</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMinWindows</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>EvalFixedKWindows</td><td>3.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMargin</td><td>0.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit1 Accuracy</td><td>0.8338</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit2 Accuracy</td><td>0.9385</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit3 Accuracy</td><td>0.9754</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy test accuracy</td><td>0.9754 (n_segments=325)</td><td>0.9754 (n_segments=325)</td><td>0.9778 (n_segments=45)</td><td>—</td></tr>
  <tr><td>Avg exit depth</td><td>1.982</td><td>1.982</td><td>2.622</td><td>—</td></tr>
  <tr><td>Exit mix</td><td>e1=0.3631, e2=0.2923, e3=0.3446</td><td>e1=0.3631, e2=0.2923, e3=0.3446</td><td>e1=0.0889, e2=0.2000, e3=0.7111</td><td>—</td></tr>
  <tr><td>Flip-rate (any flip)</td><td>0.1785</td><td>0.1785 (used windows)</td><td>0.4000 (used windows)</td><td>—</td></tr>
  <tr><td>Avg flip-count</td><td>0.2031</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Exit-consistency (taken==final)</td><td>1.0</td><td>1 (used windows)</td><td>1 (used windows)</td><td>—</td></tr>
  <tr><td>Windows Saved (%)</td><td>0.0</td><td>0.0</td><td>0.8615</td><td>—</td></tr>
  <tr><td>Compute Saved (%)</td><td>0.0</td><td>0.0</td><td>0.8168</td><td>—</td></tr>
  <tr><td>Window distribution mode</td><td>N/A</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Clip-metrics (segment-policy, for fair comparison)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Clip accuracy</td><td>N/A</td><td>1.0</td><td>1.0</td><td>—</td></tr>
  <tr><td>Avg windows used</td><td>N/A</td><td>14.773 / 14.773 (100%)</td><td>2.045 / 14.773</td><td>—</td></tr>
  <tr><td>Avg compute units (sum depth over used windows)</td><td>N/A</td><td>29.273</td><td>5.364</td><td>—</td></tr>
  <tr><td>Fixed-position diagnostic (independent of time-exit)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_firstK</td><td>—</td><td>0.9538 (n_segments=65)</td><td>0.9538 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_midK</td><td>—</td><td>0.9538 (n_segments=65)</td><td>0.9538 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_lastK</td><td>—</td><td>0.9692 (n_segments=65)</td><td>0.9692 (n_segments=65)</td><td>—</td></tr>
  <tr><td>Stop-speed group diagnostic (Depth×Time only; first-K accuracy)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_1</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_2</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_3</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Confusion matrix (clip-level)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Per_Class: Female</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>—</td></tr>
  <tr><td>Per_Class: Male</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>—</td></tr>
  <tr><td>Total Time</td><td>207.38 seconds (~3.46 minutes)</td><td>—</td><td>—</td><td>—</td></tr>
</table>

### Master table — `5exit-greedy`

<table>
  <tr><th>Metric</th><th>Segment Random Greedy Policy Test</th><th>Full-Clip Sequential Greedy Policy Test</th><th>Depth × Time Clip Greedy Policy Test</th><th>Key Change</th></tr>
  <tr><td>PowerShell Command</td><td><code>powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `</code><br><code>-Variant "5exit_greedy" `</code><br><code>-Policy "greedy" `</code><br><code>-Device "cpu" `</code><br><code>-TapBlocks "1,2,3,4" `</code><br><code>-ExitHint "false" `</code><br><code>-RunClipPolicy</code></td><td>—</td><td>—</td><td>Best deep-capacity no-hint baseline after the corrected dynamic configuration. Ties for best segment policy accuracy at 99.08% and provides very strong late exits, but remains more compute-expensive than 3exit_greedy_hint.</td></tr>
  <tr><td>Files</td><td>Train: 99, Test: 22, Val: 21</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Segment</td><td>Train: 1646, Test: 325, Val: 246</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy</td><td>Greedy</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Device</td><td>CPU</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>SegmentSec</td><td>1.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>HopSec</td><td>0.5</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>NMels</td><td>64.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeConf</td><td>0.95</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeStableK</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMinWindows</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>EvalFixedKWindows</td><td>3.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMargin</td><td>0.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit1 Accuracy</td><td>0.8369</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit2 Accuracy</td><td>0.8892</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit3 Accuracy</td><td>0.9723</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit4 Accuracy</td><td>0.9754</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit5 Accuracy</td><td>0.9908</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy test accuracy</td><td>0.9908 (n_segments=325)</td><td>0.9908 (n_segments=325)</td><td>0.9778 (n_segments=45)</td><td>—</td></tr>
  <tr><td>Avg exit depth</td><td>2.637</td><td>2.637</td><td>3.578</td><td>—</td></tr>
  <tr><td>Exit mix</td><td>e1=0.3569, e2=0.0308, e3=0.3538, e4=0.1354, e5=0.1231</td><td>e1=0.3569, e2=0.0308, e3=0.3538, e4=0.1354, e5=0.1231</td><td>e1=0.0889, e2=0.0000, e3=0.3778, e4=0.3111, e5=0.2222</td><td>—</td></tr>
  <tr><td>Flip-rate (any flip)</td><td>0.1908</td><td>0.1908 (used windows)</td><td>0.4222 (used windows)</td><td>—</td></tr>
  <tr><td>Avg flip-count</td><td>0.2338</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Exit-consistency (taken==final)</td><td>1.0</td><td>0.9969 (used windows)</td><td>1 (used windows)</td><td>—</td></tr>
  <tr><td>Windows Saved (%)</td><td>0.0</td><td>0.0</td><td>0.86152</td><td>—</td></tr>
  <tr><td>Compute Saved (%)</td><td>0.0</td><td>0.0</td><td>0.8053</td><td>—</td></tr>
  <tr><td>Window distribution mode</td><td>N/A</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Clip-metrics (segment-policy, for fair comparison)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Clip accuracy</td><td>N/A</td><td>1.0000 (n_clips=22)</td><td>1.0000 (n_clips=22)</td><td>—</td></tr>
  <tr><td>Avg windows used</td><td>N/A</td><td>14.773 / 14.773 (100%)</td><td>2.045 / 14.773</td><td>—</td></tr>
  <tr><td>Avg compute units (sum depth over used windows)</td><td>N/A</td><td>38.955</td><td>7.318</td><td>—</td></tr>
  <tr><td>Fixed-position diagnostic (independent of time-exit)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_firstK</td><td>—</td><td>0.9846 (n_segments=65)</td><td>0.9846 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_midK</td><td>—</td><td>1.0000 (n_segments=65)</td><td>1.0000 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_lastK</td><td>—</td><td>0.9846 (n_segments=65)</td><td>0.9846 (n_segments=65)</td><td>—</td></tr>
  <tr><td>Stop-speed group diagnostic (Depth×Time only; first-K accuracy)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_2</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_3_4</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_5_plus</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Confusion matrix (clip-level)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Per_Class: Female</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>—</td></tr>
  <tr><td>Per_Class: Male</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>—</td></tr>
  <tr><td>Total Time</td><td>217.69 seconds (~3.63 minutes)</td><td>—</td><td>—</td><td>—</td></tr>
</table>

### Master table — `3exit-greedy-hint`

<table>
  <tr><th>Metric</th><th>Segment Random Greedy Policy Test</th><th>Full-Clip Sequential Greedy Policy Test</th><th>Depth × Time Clip Greedy Policy Test</th><th>Key Change</th></tr>
  <tr><td>PowerShell Command</td><td><code>powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `</code><br><code>-Variant "3exit_greedy_hint" `</code><br><code>-Policy "greedy" `</code><br><code>-Device "cpu" `</code><br><code>-TapBlocks "1,3" `</code><br><code>-ExitHint "true" `</code><br><code>-RunClipPolicy</code></td><td>—</td><td>—</td><td>Best overall efficiency-quality result. Sequential hint passing raises segment policy accuracy from 97.54% to 99.08%, keeps perfect clip accuracy, reduces average exit depth and full/depth-time compute, and gives the strongest deployment-ready tradeoff.</td></tr>
  <tr><td>Files</td><td>Train: 99, Test: 22, Val: 21</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Segment</td><td>Train: 1646, Test: 325, Val: 246</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy</td><td>Greedy</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Device</td><td>CPU</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>SegmentSec</td><td>1.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>HopSec</td><td>0.5</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>NMels</td><td>64.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeConf</td><td>0.95</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeStableK</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMinWindows</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>EvalFixedKWindows</td><td>3.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMargin</td><td>0.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit1 Accuracy</td><td>0.8369</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit2 Accuracy</td><td>0.9662</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit3 Accuracy</td><td>0.9908</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy test accuracy</td><td>0.9908 (n_segments=325)</td><td>0.9908 (n_segments=325)</td><td>1.0000 (n_segments=44)</td><td>—</td></tr>
  <tr><td>Avg exit depth</td><td>1.895</td><td>1.895</td><td>2.477</td><td>—</td></tr>
  <tr><td>Exit mix</td><td>e1=0.3846, e2=0.3354, e3=0.2800</td><td>e1=0.3846, e2=0.3354, e3=0.2800</td><td>e1=0.0909, e2=0.3409, e3=0.568</td><td>—</td></tr>
  <tr><td>Flip-rate (any flip)</td><td>0.1908</td><td>0.1908 (used windows)</td><td>0.3864 (used windows)</td><td>—</td></tr>
  <tr><td>Avg flip-count</td><td>0.2154</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Exit-consistency (taken==final)</td><td>1.0</td><td>1 (used windows)</td><td>1 (used windows)</td><td>—</td></tr>
  <tr><td>Windows Saved (%)</td><td>0.0</td><td>0.0</td><td>0.8646</td><td>—</td></tr>
  <tr><td>Compute Saved (%)</td><td>0.0</td><td>0.0</td><td>0.8231</td><td>—</td></tr>
  <tr><td>Window distribution mode</td><td>N/A</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Clip-metrics (segment-policy, for fair comparison)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Clip accuracy</td><td>N/A</td><td>1.0</td><td>1.0</td><td>—</td></tr>
  <tr><td>Avg windows used</td><td>N/A</td><td>14.773 / 14.773 (100%)</td><td>2.000 / 14.773</td><td>—</td></tr>
  <tr><td>Avg compute units (sum depth over used windows)</td><td>N/A</td><td>28.0</td><td>4.955</td><td>—</td></tr>
  <tr><td>Fixed-position diagnostic (independent of time-exit)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_firstK</td><td>—</td><td>1.0000 (n_segments=65)</td><td>1.0000 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_midK</td><td>—</td><td>1.0000 (n_segments=65)</td><td>1.0000 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_lastK</td><td>—</td><td>0.9692 (n_segments=65)</td><td>0.9692 (n_segments=65)</td><td>—</td></tr>
  <tr><td>Stop-speed group diagnostic (Depth×Time only; first-K accuracy)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_1</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_2</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_3</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Confusion matrix (clip-level)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Per_Class: Female</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>—</td></tr>
  <tr><td>Per_Class: Male</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>—</td></tr>
  <tr><td>Total Time</td><td>208.9 seconds (~3.48 minutes)</td><td>—</td><td>—</td><td>—</td></tr>
</table>

### Master table — `5exit-greedy-hint`

<table>
  <tr><th>Metric</th><th>Segment Random Greedy Policy Test</th><th>Full-Clip Sequential Greedy Policy Test</th><th>Depth × Time Clip Greedy Policy Test</th><th>Key Change</th></tr>
  <tr><td>PowerShell Command</td><td><code>powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `</code><br><code>-Variant "5exit_greedy_hint" `</code><br><code>-Policy "greedy" `</code><br><code>-Device "cpu" `</code><br><code>-TapBlocks "1,2,3,4" `</code><br><code>-ExitHint "true" `</code><br><code>-RunClipPolicy</code></td><td>—</td><td>—</td><td>Still weaker than 5exit_greedy under the corrected setup. Hint passing does not improve the deeper greedy pipeline yet: policy accuracy drops to 97.23%, and Depth×Time compute/efficiency remain worse than both 3exit_greedy_hint and 5exit_greedy.</td></tr>
  <tr><td>Files</td><td>Train: 99, Test: 22, Val: 21</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Segment</td><td>Train: 1646, Test: 325, Val: 246</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy</td><td>Greedy</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Device</td><td>CPU</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>SegmentSec</td><td>1.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>HopSec</td><td>0.5</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>NMels</td><td>64.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeConf</td><td>0.95</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeStableK</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMinWindows</td><td>2.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>EvalFixedKWindows</td><td>3.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>TimeMargin</td><td>0.0</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit1 Accuracy</td><td>0.8308</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit2 Accuracy</td><td>0.8646</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit3 Accuracy</td><td>0.9231</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit4 Accuracy</td><td>0.9538</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Exit5 Accuracy</td><td>0.9692</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Policy test accuracy</td><td>0.9723 (n_segments=325)</td><td>0.9723 (n_segments=325)</td><td>0.9783 (n_segments=48)</td><td>—</td></tr>
  <tr><td>Avg exit depth</td><td>2.465</td><td>2.465</td><td>3.565</td><td>—</td></tr>
  <tr><td>Exit mix</td><td>e1=0.3723, e2=0.1354, e3=0.2677, e4=0.1046, e5=0.1200</td><td>e1=0.0554, e2=0.3877, e3=0.1385, e4=0.2338, e5=0.1846</td><td>e1=0.0870, e2=0.0652, e3=0.3043, e4=0.2826, e5=0.2609</td><td>—</td></tr>
  <tr><td>Flip-rate (any flip)</td><td>0.2123</td><td>0.2123 (used windows)</td><td>0.4565 (used windows)</td><td>—</td></tr>
  <tr><td>Avg flip-count</td><td>0.2492</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Exit-consistency (taken==final)</td><td>0.9908</td><td>0.9908 (used windows)</td><td>1 (used windows)</td><td>—</td></tr>
  <tr><td>Windows Saved (%)</td><td>0.0</td><td>0.0</td><td>0.8585</td><td>—</td></tr>
  <tr><td>Compute Saved (%)</td><td>0.0</td><td>0.0</td><td>0.7953</td><td>—</td></tr>
  <tr><td>Window distribution mode</td><td>N/A</td><td>N/A</td><td>N/A</td><td>—</td></tr>
  <tr><td>Clip-metrics (segment-policy, for fair comparison)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Clip accuracy</td><td>N/A</td><td>1.0000 (n_clips=22)</td><td>1.0000 (n_clips=22)</td><td>—</td></tr>
  <tr><td>Avg windows used</td><td>N/A</td><td>14.773 / 14.773 (100%)</td><td>2.091 / 14.773</td><td>—</td></tr>
  <tr><td>Avg compute units (sum depth over used windows)</td><td>N/A</td><td>36.409</td><td>7.455</td><td>—</td></tr>
  <tr><td>Fixed-position diagnostic (independent of time-exit)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_firstK</td><td>—</td><td>0.9538 (n_segments=65)</td><td>0.9538 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_midK</td><td>—</td><td>0.9538 (n_segments=65)</td><td>0.9538 (n_segments=65)</td><td>—</td></tr>
  <tr><td>3-window diagnostic Acc_lastK</td><td>—</td><td>0.9385 (n_segments=65)</td><td>0.9385 (n_segments=65)</td><td>—</td></tr>
  <tr><td>Stop-speed group diagnostic (Depth×Time only; first-K accuracy)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_2</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_3_4</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>stop_5_plus</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Confusion matrix (clip-level)</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
  <tr><td>Per_Class: Female</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:12</td><td>—</td></tr>
  <tr><td>Per_Class: Male</td><td>—</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>Precision: 1.0, Recall:1.0, f1: 1.0, Support:10</td><td>—</td></tr>
  <tr><td>Total Time</td><td>217.69 seconds (~3.63 minutes)</td><td>—</td><td>—</td><td>—</td></tr>
</table>

---

## Main run artifacts

Each run directory contains the usual core outputs:

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

## Current research takeaway

This branch should now be described as:

> a corrected and reusable **generic K-exit / C-class** greedy branch with **CLI-controlled hint vs no-hint**, validated across four runs and showing that hint passing is highly effective in the **3-exit** setting but not yet beneficial in the current **5-exit** greedy setting.

That is the cleanest documentation position for `kexit-greedy-hint` right now.
