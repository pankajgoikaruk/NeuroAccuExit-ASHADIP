# NeuroAccuExit-ASHADIP (ASHADIP)

TinyML-friendly **multi-exit audio classifier** with **Depth×Time adaptive inference** for audio clips.

The repository now supports three connected ideas:

- **Depth early-exit (segment-level):** choose how deep to process each window using a policy such as greedy confidence or **Depth-EA** (evidence accumulation across exits).
- **Time early-exit (clip-level):** process windows sequentially and stop the remaining windows of the clip once the clip prediction is sufficiently **stable + confident**.
- **Hint passing between exits (v0.5.2):** later exits consume a compact learned hint derived from earlier exit predictions, improving early decision quality without changing the backbone into a heavy model.

This repo is designed for reproducible runs on cached log-mel features and produces paper-ready outputs such as accuracy, exit mix, stopping distributions, windows saved, compute saved, flip-rate, and clip-level diagnostics.

---

## What changed in v0.5.2

`v0.5.2` should be understood as:

**Hint Passing + Stabilized K=5 EA/Depth×Time**

It builds on the `v0.5` K-exit / C-class refactor and adds a clearer architectural and policy story.

### 1) K-exit + C-class generic pipeline
The model is generic for:

- **K exits** via `tap_blocks`
- **C classes** (binary or multi-class)

Examples:

- `tap_blocks = 1,3` → exits after block1 and block3 + final → **K=3**
- `tap_blocks = 1,2,3,4` → exits after blocks 1–4 + final → **K=5**

✅ **Rule:** a checkpoint trained with one `tap_blocks` must be calibrated and evaluated with the **same** `tap_blocks`.

### 2) Learned hint passing between exits
In `v0.5.2`, later exits are no longer forced to rely only on their local backbone features.
Instead, they also receive a **compact learned hint** derived from an earlier exit prediction.

Conceptually:

- exit1 predicts from early features
- exit2 uses **raw feature + hint1**
- exit3 uses **raw feature + hint2**
- later exits continue refining with progressively stronger evidence

This is a real **architectural change**, not just a threshold tweak.
It is intended to improve early-window decision quality while keeping the model TinyML-friendly.

### 3) Stabilized clip-level stopping
The clip-level stop posterior uses a **normalized accumulated posterior** rather than an unnormalized sum of confidence over time.
This avoids artificial confidence inflation when more windows are processed.

### 4) Safer intermediate-exit control
A stricter **Exit2 confirmation override** is used inside `Depth-EA` to let exit2 participate only in strong, agreement-based cases.
This gives a better balance between:

- reviving exit2 usage when appropriate
- avoiding weak intermediate exits
- preserving final clip accuracy

### 5) Stronger diagnostics
The evaluation pipeline now reports:

- segment policy accuracy
- clip accuracy
- windows saved (%)
- compute saved (%)
- average windows used
- average compute units
- average depth per used window
- exit mix
- flip-rate
- exit-consistency
- fixed-position diagnostics (`firstK`, `midK`, `lastK`)
- stop-speed group analysis

---

## Quickstart (recommended v0.5.2 pipeline)

### Environment
```powershell
conda activate ASHADIP_V0
```

### Full pipeline run (K=5, hinted model)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.5_hint" `
  -Policy "ea" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -AutoLambdaDepth `
  -EAMinExit 2 `
  -TapBlocks "1,2,3,4" `
  -NMels 64 `
  -RunClipPolicy `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3
```

> Tip: `Variant="v0.5_hint"` creates folders such as `runs/v0_5_hint/...` and `data_caches/v0_5_hint/...`.

---

## Recommended v0.5.2 interpretation

For `K=5`, the most reliable configuration is still:

- `ea_min_exit = 2` (force at least exit3 under normal EA gating)
- `lambda_depth = 0.02` (milder depth penalty)

On top of that, `v0.5.2` adds:

- hint passing between exits
- normalized time stopping
- stricter exit2 confirmation logic

This means:

- weak exit1-heavy policies are avoided
- later strong exits still dominate hard cases
- exit2 can contribute in safe cases
- Depth×Time can stop very early without losing clip accuracy

---

## Best confirmed v0.5.2 result (strict Exit2 confirmation)

A strong verified `v0.5.2` setting used the hinted K=5 model with:

- `ea_min_exit = 2`
- `lambda_depth = 0.02`
- strict exit2 confirmation override
  - `ea_exit2_conf_min = 0.99`
  - `ea_exit2_margin_min = 0.45`
  - `ea_exit2_conf_gain_min = 0.05`
  - `ea_exit2_margin_gain_min = 0.03`
  - `ea_exit2_require_agree = True`

### Segment policy (Depth-only, EA)
- **Policy accuracy:** `0.9662`
- **Clip accuracy:** `1.0000`
- **Avg exit depth:** `2.302`
- **Avg compute units:** `34.000`
- **Exit mix:** `e1=0.372, e2=0.114, e3=0.382, e4=0.105, e5=0.028`
- **Exit-consistency:** `0.9785`

### Clip policy (FULL baseline; no time stop)
- **Policy accuracy over processed windows:** `0.9662`
- **Clip accuracy:** `1.0000`
- **Avg windows used:** `14.773 / 14.773`
- **Avg compute units:** `34.000`
- **Avg depth per used window:** `2.302`

### Clip policy (Depth×Time)
- **Policy accuracy over processed windows:** `0.9149`
- **Clip accuracy:** `1.0000`
- **Avg windows used:** `2.136 / 14.773`
- **Windows saved:** `85.54%`
- **Avg compute units:** `6.591`
- **Compute saved:** `80.61%`
- **Avg depth per used window:** `3.085`
- **Exit mix:** `e1=0.106, e2=0.043, e3=0.574, e4=0.213, e5=0.064`
- **Exit-consistency:** `0.9362`

This is the most useful headline result for `v0.5.2`:

> **The hinted K=5 model preserves 100% clip accuracy while using only 2.136 windows on average, saving 85.54% of windows and 80.61% of compute.**

---

## Why segment accuracy over processed windows can drop under Depth×Time

Under Depth×Time, the policy intentionally evaluates mostly **early** windows.
Those windows are usually harder and noisier than later ones, so it is normal for segment accuracy over processed windows to be lower than the full-clip baseline.

Use the fixed-position diagnostic to interpret this correctly:

- `Acc_firstK`
- `Acc_midK`
- `Acc_lastK`

If `firstK` is lower than `midK` and `lastK`, the drop is mainly **early-window difficulty**, not a failure of the method.

---

## Meaning of the main PowerShell parameters (`scripts/run_full.ps1`)

### Paths and run organization
- **`-DataRoot`** *(default: `data\moth_sounds`)*
- **`-CacheRoot`** *(default: `data_caches`)*
- **`-CacheId`** *(default: auto)*
- **`-Config`** *(default: `configs\audio_moth.yaml`)*
- **`-RunsRoot`** *(default: `runs`)*
- **`-Variant`** *(default: `v0.5`)*

### Policy selection
- **`-Policy`** *(auto / greedy / ea; default: auto)*
- **`-LambdaDepth`** *(default: 0.08)*
- **`-AutoLambdaDepth`** *(switch)*
  - `K>=5 -> 0.02`
  - `K<5 -> 0.08`
- **`-EAMinExit`** *(default: -1 = auto)*
  - `-1` = auto (`K>=5 -> 2`, else `0`)
  - `0` = allow exit1
  - `1` = force at least exit2
  - `2` = force at least exit3

### Compute/device
- **`-Device`** *(default: cpu)*

### Segmentation
- **`-SegmentSec`** *(default: 1.0)*
- **`-HopSec`** *(default: 0.5)*

### K exits and features
- **`-TapBlocks`** *(default: `"1,3"`)*
- **`-NMels`** *(default: 64)*

### Clip / time-exit evaluation
- **`-RunClipPolicy`** *(switch)*
- **`-TimeMinWindows`** *(default: 2)*
- **`-TimeStableK`** *(default: 2)*
- **`-TimeConf`** *(default: 0.95)*
- **`-TimeMargin`** *(default: 0.0)*
- **`-EvalFixedKWindows`** *(default: 3)*
- **`-PrintClipWindows`** *(switch)*

---

## Key outputs created in `run_dir`

### Segment policy (`scripts/policy_test.py`)
- `policy_results.json`

### Clip policy (`scripts/clip_policy_test.py`)
- `clip_policy_results_full.json`
- `clip_policy_results_time.json`
- `clip_preds_full.csv`, `clip_preds_time.csv`
- `clip_length_hist.json`
- `windows_used_hist.json`

### Summaries and plots
- `summary.json`
- `analysis_run.json`
- `confusion_matrices.json`
- `roc_curves.json`
- `plots/`
- `profiling.json`
- `analysis/on_device_summary.csv`

---

## Notes / troubleshooting

- If `Compute Saved (%)` prints `N/A`, run the FULL clip baseline first so `clip_policy_results_full.json` exists.
- If you see `state_dict` mismatch errors, confirm that training, calibration, and evaluation all use the same `tap_blocks` and hinted/non-hinted model definition.
- If an EA sweep becomes exit1-heavy, re-run threshold selection with `ea_min_exit = 2` and `lambda_depth = 0.02`.
- If flip-rate is too high, try increasing `TimeStableK` or adding a positive `TimeMargin`.

---

## Suggested citation / interpretation line

If you describe this release in a thesis or paper, a concise summary is:

> `v0.5.2` adds learned hint passing between exits and stabilized K=5 Depth×Time inference, improving early-window decision quality while preserving 100% clip-level accuracy under aggressive temporal stopping.
