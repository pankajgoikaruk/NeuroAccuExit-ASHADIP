# Repository Structure (DOC_STRUCTURE)

This file documents the project layout and key scripts used for **Depth×Time early-exit** experiments.
Updated for **v0.5.2**, where the codebase supports:

- **K exits** (via `tap_blocks`)
- **C classes** (binary or multi-class)
- **hint passing between exits**
- **stabilized K=5 Depth-EA + Depth×Time evaluation**

---

## Top-level layout

```text
NeuroAccuExit-ASHADIP/
├─ adapters/                 # model backbones / adapters (TinyAudioCNN)
├─ configs/                  # YAML configs (dataset, features, training knobs)
├─ data/                     # raw audio (optional; depends on workflow)
├─ data_caches/              # cached segments + per-segment features (log-mel)
├─ models/                   # ExitNet and multi-exit heads
├─ policies/                 # early-exit policies (Depth-EA, greedy, etc.)
├─ scripts/                  # entry points for running and evaluating
├─ training/                 # training + calibration + threshold search
├─ utils/                    # helpers (profiling, logging, misc)
└─ runs/                     # experiment outputs (checkpoints, json, csv, plots)
```

---

## K-exit (`tap_blocks`) convention

`tap_blocks` is a comma-separated list of backbone blocks after which an exit head is attached.

Examples:

- `tap_blocks=1,3` → **K=3 exits**
- `tap_blocks=1,2,3,4` → **K=5 exits**

**Rule:** a checkpoint trained with a given `tap_blocks` must be evaluated and calibrated with the same `tap_blocks`.

---

## Core model files

### `adapters/audio_adapter.py` (TinyAudioCNN)
- 5-block CNN backbone
- configurable `tap_blocks`
- exposes `tap_dims` and `final_dim`
- acts as the shared feature extractor for all exits

### `models/exit_net.py` (ExitNet)
- builds `K = len(taps) + 1` classifier heads
- supports arbitrary `num_classes`
- in `v0.5.2`, later exits can consume a compact **hint** projected from earlier exit predictions
- this makes the hinted model an **architectural extension**, not only a policy change

### `policies/depth_ea.py` (Depth-EA)
- generic evidence accumulation across exits for arbitrary `K`
- returns `taken`, `pred_taken`, `pred_final`, `flip_count`, and `logp_taken`
- supports safer K=5 behavior via `ea_min_exit`
- in `v0.5.2`, includes stricter **Exit2 confirmation override** logic so exit2 can be used only in strong agreement-based cases

---

## Training and evaluation modules

### `training/train.py`
Trains the multi-exit network.

**Typical outputs**
- `ckpt/best.pt`
- `metrics.json`

For `v0.5.2`, this step may train either:
- the standard K-exit model, or
- the hinted K-exit model (depending on the active code/configuration)

### `training/eval.py`
Evaluates per-exit test metrics and writes:
- `report.json`

Useful for checking whether exits 1, 2, 3, 4, and 5 individually improved.

### `training/calibrate.py`
Temperature scaling per exit.

**Output**
- `temperature.json` (length = K)

### `training/thresholds_offline.py`
Greedy threshold search.

**Output**
- `thresholds.json`

### `training/ea_thresholds_offline.py`
EA threshold search.

**Outputs**
- `ea_thresholds.json`
- `ea_sweep_results.json`

> Note: `run_full.ps1` exposes `-EAMinExit` (auto by default). The Python module also accepts `--ea_min_exit` directly.
>
> For `K=5`, using `--ea_min_exit 2` is a robust way to avoid exit1-heavy policies.

---

## Policy evaluation scripts

### `scripts/policy_test.py`
Segment policy evaluation (greedy or EA).

**Reports**
- policy accuracy
- average exit depth
- exit mix (`e1..eK`)
- flip-rate
- exit-consistency
- clip-metrics without temporal stopping

**Output**
- `policy_results.json`

### `scripts/clip_policy_test.py`
Depth×Time clip policy evaluation.

This script:
- groups test windows by clip
- processes windows sequentially
- applies `Depth-EA` to choose the exit for each used window
- accumulates evidence over time
- stops early when the clip posterior becomes stable + confident

In `v0.5.2`, the time-stop posterior is normalized before confidence is checked, which avoids confidence blow-up over time.

**Outputs**
- `clip_policy_results_full.json` (full-clip baseline; no time stopping)
- `clip_policy_results_time.json` (Depth×Time)
- `clip_preds_full.csv`, `clip_preds_time.csv`
- `clip_length_hist.json`
- `windows_used_hist.json`

**Diagnostics reported**
- fixed-position first/mid/last-K window accuracy
- stop-speed group analysis
- confusion matrix
- windows saved / compute saved
- exit mix, flip-rate, exit-consistency

---

## Pipeline runner

### `scripts/run_full.ps1`
Runs the full pipeline:

1. prep segments  
2. extract features  
3. train  
4. calibrate  
5. threshold search  
6. `policy_test`  
7. `clip_policy_test` (optional)  
8. summarize  
9. analyse  
10. profile  

### Important runner knobs

#### Depth-EA controls
- `-LambdaDepth`  
  Depth penalty used during EA sweep. Higher = earlier exits, lower = deeper exits.
- `-AutoLambdaDepth`  
  Safe automatic default based on `K`.
- `-EAMinExit`  
  Minimum allowed exit index.
  For `K=5`, `2` means "force exit3+ under standard EA gating".

#### K-exit and feature controls
- `-TapBlocks`
- `-NMels`

#### Time-exit controls
- `-RunClipPolicy`
- `-TimeMinWindows`
- `-TimeStableK`
- `-TimeConf`
- `-TimeMargin`
- `-EvalFixedKWindows`
- `-PrintClipWindows`

**Logs**
- `analysis/pipeline_runtime.csv`

---

## Data cache expectations

`segments.csv` contains at least:
- `wav_relpath`
- `start`
- `duration`
- `feat_relpath`
- `label`
- `split`

Features referenced by `feat_relpath` are `.npy` arrays shaped `(n_mels, frames)`.

---

## Example commands (PowerShell)

### Full run for `v0.5.2` hinted K=5 setup
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

### Re-sweep EA thresholds for a trained hinted run
```powershell
$RUN="runs\v0_5_hint\v0_5_hint_002"
$CACHE="data_caches\v0_5_hint\seg1_hop0p5_bp100-3000_mels64"

python -m training.ea_thresholds_offline `
  --run_dir $RUN `
  --segments_csv "$CACHE\segments.csv" `
  --features_root "$CACHE\features" `
  --tap_blocks 1,2,3,4 `
  --n_mels 64 `
  --ea_min_exit 2 `
  --lambda_depth 0.02
```

### Re-run segment policy evaluation
```powershell
python -m scripts.policy_test `
  --policy ea `
  --run_dir $RUN `
  --segments_csv "$CACHE\segments.csv" `
  --features_root "$CACHE\features" `
  --tap_blocks 1,2,3,4 `
  --n_mels 64
```

### Re-run clip policy evaluation
```powershell
python -m scripts.clip_policy_test `
  --run_dir $RUN `
  --segments_csv "$CACHE\segments.csv" `
  --features_root "$CACHE\features" `
  --tap_blocks 1,2,3,4 `
  --n_mels 64 `
  --disable_time_exit `
  --eval_fixed_k_windows 3

python -m scripts.clip_policy_test `
  --run_dir $RUN `
  --segments_csv "$CACHE\segments.csv" `
  --features_root "$CACHE\features" `
  --tap_blocks 1,2,3,4 `
  --n_mels 64 `
  --time_conf 0.95 `
  --time_stable_k 2 `
  --time_min_windows 2 `
  --eval_fixed_k_windows 3 `
  --full_baseline_json "$RUN\clip_policy_results_full.json"
```

---

## How to read the main metrics

### `policy accuracy`
Segment-level accuracy over the windows that were actually processed.

### `clip accuracy`
Final clip decision accuracy.
This is the most important end-task metric for clip-level inference.

### `avg windows used`
How many windows were processed before stopping.

### `windows saved (%)`
Percentage of windows skipped relative to the full baseline.

### `avg compute units`
A simple proxy equal to the sum of exit depths used over the processed windows.

### `compute saved (%)`
Reduction in average compute units relative to the full-clip baseline.

### `avg depth per used window`
Average exit depth among processed windows only.

### `exit mix`
Fraction of processed windows that exited at each exit (`e1..eK`).

### `flip-rate`
How often the predicted class changed across exits/windows.
Higher values indicate lower internal stability.

### `exit-consistency`
How often the taken exit agreed with the final exit.
Higher is better.

---

## Practical interpretation of v0.5.2

At this stage, `v0.5.2` is best understood as:

- a **K=5 / C-class-capable multi-exit system**
- with an added **hint-passing architecture**
- plus a more stable **Depth×Time evaluation protocol**
- and safer **intermediate-exit control**

In other words, `v0.5.2` is not just "another threshold version"; it is the first version where architectural refinement and policy refinement are both clearly part of the method.
