# Repository Structure (DOC_STRUCTURE)

This file documents the project layout and key scripts used for **Depth×Time early-exit** experiments.
Updated for **v0.5+** where the codebase is generic for:
- **K exits** (via `tap_blocks`)
- **C classes** (binary/multi-class)

---

## Top-level layout

```
NeuroAccuExit-ASHADIP/
├─ adapters/                 # model backbones / adapters (TinyAudioCNN)
├─ configs/                  # YAML configs (dataset, features, training knobs)
├─ data/                     # raw audio (optional; depends on your workflow)
├─ data_caches/              # cached segments + per-segment features (log-mel)
├─ models/                   # ExitNet (multi-exit wrapper)
├─ policies/                 # early-exit policies (Depth-EA, greedy, etc.)
├─ scripts/                  # entry points for running and evaluating
├─ training/                 # training + calibration + threshold search
├─ utils/                    # helpers (profiling, logging, misc)
└─ runs/                     # experiment outputs (checkpoints, json, csv, plots)
```

---

## K-exit (`tap_blocks`) convention

`tap_blocks` is a comma list of backbone blocks after which an exit head is attached:

- `tap_blocks=1,3` → **K=3 exits**
- `tap_blocks=1,2,3,4` → **K=5 exits**

**Rule:** A checkpoint trained with a given `tap_blocks` must be evaluated/calibrated with the same `tap_blocks`.

---

## Core model files

### `adapters/audio_adapter.py` (TinyAudioCNN)
- 5-block CNN backbone
- configurable `tap_blocks`
- exposes `tap_dims` and `final_dim`

### `models/exit_net.py` (ExitNet)
- builds `K = len(taps) + 1` classifier heads
- supports arbitrary `num_classes`

### `policies/depth_ea.py` (Depth-EA)
- generic evidence accumulation across exits (K)
- returns `taken`, `pred_taken`, `pred_final`, `flip_count`, and (optionally) `logp_taken`

---

## Training and evaluation modules

### `training/train.py`
Trains the multi-exit network.

**Outputs (in run_dir)**
- `ckpt/best.pt`
- `metrics.json`

### `training/eval.py`
Evaluates per-exit test metrics → `report.json`.

### `training/calibrate.py`
Temperature scaling per exit → `temperature.json` (length K).

### `training/thresholds_offline.py`
Greedy threshold search → `thresholds.json`.

### `training/ea_thresholds_offline.py`
EA threshold search → `ea_thresholds.json` + `ea_sweep_results.json`.

> Note: `run_full.ps1` supports `-EAMinExit` (auto by default). The Python module still supports `--ea_min_exit` directly.  
> For K=5, using `--ea_min_exit 2` is a robust way to avoid exit1-heavy policies.

---

## Policy evaluation scripts

### `scripts/policy_test.py`
Segment policy evaluation (greedy / EA).

**Outputs**
- `policy_results.json` (accuracy, avg depth, exit mix e1..eK, flip metrics, clip metrics)

### `scripts/clip_policy_test.py`
Depth×Time clip policy evaluation.

**Outputs**
- `clip_policy_results_full.json` (no time-exit baseline)
- `clip_policy_results_time.json` (time-exit)
- `clip_preds_full.csv`, `clip_preds_time.csv`
- `clip_length_hist.json`, `windows_used_hist.json`

---

## Pipeline runner

### `scripts/run_full.ps1`
Runs the whole pipeline:
1) prep segments → 2) extract features → 3) train → 4) calibrate → 5) thresholds → 6) policy_test → 7) clip_policy_test (optional) → 8) summarize → 9) analyse → 10) profile

**Depth-EA knobs exposed by the runner**
- `-LambdaDepth`: depth penalty used during EA sweep (higher = earlier exits).
- `-EAMinExit`: minimum allowed exit index (0-indexed). `2` forces exit3+ for K=5 (recommended).

Logs:
- `analysis/pipeline_runtime.csv`

---

## Data cache expectations

`segments.csv` contains:
- `wav_relpath`, `start`, `duration`
- `feat_relpath`
- `label`, `split`

Features referenced by `feat_relpath` are `.npy` arrays shaped `(n_mels, frames)`.

---

## Example commands (PowerShell)

### Run everything (K=5)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.5" `
  -Policy "ea" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -AutoLambdaDepth `
  -TapBlocks "1,2,3,4" `
  -NMels 64 `
  -RunClipPolicy `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -TimeMargin 0.0 `
  -EvalFixedKWindows 3 `
  -PrintClipWindows

# Optional: set EAMinExit explicitly (default auto picks 2 for K>=5 when Policy=ea)
#   -EAMinExit 2
```

### (Optional) Force strong EA behavior for K=5
```powershell
python -m training.ea_thresholds_offline `
  --run_dir "runs\v0_5\v0_5_002" `
  --segments_csv "data_caches\v0_5\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_5\seg1_hop0p5_bp100-3000_mels64\features" `
  --tap_blocks 1,2,3,4 `
  --n_mels 64 `
  --ea_min_exit 2 `
  --lambda_depth 0.02
```
