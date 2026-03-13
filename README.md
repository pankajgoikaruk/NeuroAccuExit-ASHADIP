# NeuroAccuExit-ASHADIP (ASHADIP)

TinyML-friendly **multi-exit audio classifier** with **Depth×Time early-exit**:

- **Depth early-exit (segment-level)**: choose the exit depth per segment using *Depth-EA* (evidence accumulation across exits).
- **Time early-exit (clip-level)**: process segments sequentially and stop early when the clip prediction becomes **stable + confident** via **temporal evidence accumulation**.

This repo supports reproducible runs on cached log-mel features and produces paper-ready metrics (accuracy vs compute, stopping distributions, diagnostic baselines).

---

## What’s new in v0.5 (Step 0: K-exit + C-class)

### 1) K-exit via `tap_blocks`
The model is now generic for **K exits**.

`tap_blocks` controls where early-exit heads attach inside the backbone:

- `tap_blocks = 1,3` → exits after block1 and block3 + final → **K=3**
- `tap_blocks = 1,2,3,4` → exits after blocks 1–4 + final → **K=5**

✅ **Rule:** a checkpoint trained with one `tap_blocks` must be calibrated/evaluated with the **same** `tap_blocks`.

### 2) C-class generic
The pipeline now supports **binary or multi-class** classification (no hardcoded `num_classes=2`).

### 3) Updated end-to-end runner: `scripts/run_full.ps1`
The full pipeline can be run from scratch (cache → train → calibrate → thresholds → policy tests → analysis → profiling) using a single PowerShell command, and it exposes K-exit/time-exit knobs.

---

## Quickstart (full pipeline)

### Environment
```powershell
conda activate ASHADIP_V0
```

### Run the full pipeline (recommended baseline: K=5)
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

# Note: -EAMinExit is optional.
# By default (EAMinExit=-1) the runner auto-uses EAMinExit=2 for K>=5 when Policy=ea.
```

> Tip: `Variant="v0.5"` will create folders like `runs/v0_5/v0_5_001` and `data_caches/v0_5/...`.

---

## IMPORTANT: “exit1-heavy” EA thresholds (why accuracy can drop)
With **K=5**, EA threshold sweeping can accidentally choose a policy that exits too early (e.g., ~90% at exit1), which reduces accuracy.

A robust “known-good” EA setting for K=5 is:
- `ea_min_exit = 2` (force at least **exit3**)
- `lambda_depth = 0.02` (milder depth penalty)

`run_full.ps1` exposes `-EAMinExit` (auto by default) and `-AutoLambdaDepth` (auto lambda based on K).  
So, after the pipeline finishes training/calibration, run this once to lock the strong EA policy:

```powershell
# Replace with your run folder printed by run_full.ps1
$RUN="runs\v0_5\v0_5_002"
$CACHE="data_caches\v0_5\seg1_hop0p5_bp100-3000_mels64"

python -m training.ea_thresholds_offline `
  --run_dir $RUN `
  --segments_csv "$CACHE\segments.csv" `
  --features_root "$CACHE\features" `
  --tap_blocks 1,2,3,4 `
  --n_mels 64 `
  --ea_min_exit 2 `
  --lambda_depth 0.02

python -m scripts.policy_test `
  --policy ea `
  --run_dir $RUN `
  --segments_csv "$CACHE\segments.csv" `
  --features_root "$CACHE\features" `
  --tap_blocks 1,2,3,4 `
  --n_mels 64
```

This is the configuration that produced:
- segment EA accuracy ≈ **0.9569**
- clip accuracy ≈ **1.0000**
- Depth×Time clip accuracy **1.0000** with ~85% window savings and ~81% compute savings

---

## Meaning of the PowerShell parameters (run_full.ps1)

### Paths and run organization
- **`-DataRoot`** *(default: `data\moth_sounds`)*  
  Root folder for raw audio clips used by `scripts.prep_segments`.
- **`-CacheRoot`** *(default: `data_caches`)*  
  Parent folder where caches are created.
- **`-CacheId`** *(default: auto)*  
  Cache subfolder name. If empty, it’s auto-generated from `SegmentSec/HopSec/NMels`.
- **`-Config`** *(default: `configs\audio_moth.yaml`)*  
  YAML config used by preprocessing and training.
- **`-RunsRoot`** *(default: `runs`)*  
  Parent folder where run directories are created.
- **`-Variant`** *(default: `v0.5`)*  
  Groups runs/caches under `runs/<variant>/` and `data_caches/<variant>/`.

### Policy selection
- **`-Policy`** *(auto/greedy/ea; default: auto)*  
  Controls which segment policy is tested:
  - `ea`: Depth-EA (recommended)
  - `greedy`: greedy confidence threshold (tau)
  - `auto`: selects EA if Variant looks like EA/v0.2, otherwise greedy
- **`-LambdaDepth`** *(default: 0.08)*  
  Depth penalty used during EA threshold search.  
  Higher → earlier exits (more saving), lower → deeper exits (better accuracy).  
  (If you set `-AutoLambdaDepth` or pass `-LambdaDepth -1`, the runner will override this value.)
- **`-AutoLambdaDepth`** *(switch; default: off)*  
  If set, the runner chooses a safe default `lambda_depth` based on K:
  - **K>=5 → 0.02** (recommended for 5 exits)
  - K<5 → 0.08  
  You can also trigger the same behavior by passing `-LambdaDepth -1`.
- **`-EAMinExit`** *(default: -1 = auto)*  
  Minimum allowed exit index for Depth-EA (0-indexed).  
  - `-1` = auto (**K>=5 → 2**, else → 0)  
  - `0` = allow exit1  
  - `1` = force at least exit2  
  - `2` = force at least exit3 (**recommended for K=5** to avoid exit1-heavy policies)


### Compute/device
- **`-Device`** *(default: cpu)*  
  `cpu` or `cuda`.

### Segmentation (controls windows per clip)
- **`-SegmentSec`** *(default: 1.0)*  
  Segment/window length in seconds.
- **`-HopSec`** *(default: 0.5)*  
  Hop between windows in seconds (overlap if HopSec < SegmentSec).

### K exits and feature dimensions
- **`-TapBlocks`** *(default: "1,3")*  
  Comma list controlling exit heads; **K = len(tap_blocks) + 1**.
- **`-NMels`** *(default: 64)*  
  Number of mel bins used during feature extraction; must match the cache.

### Clip (time-exit) evaluation
- **`-RunClipPolicy`** *(switch)*  
  Also runs `scripts.clip_policy_test.py` (baseline + Depth×Time).
- **`-TimeMinWindows`** *(default: 2)*  
  Minimum windows before stopping.
- **`-TimeStableK`** *(default: 2)*  
  Require the same clip prediction for K consecutive steps.
- **`-TimeConf`** *(default: 0.95)*  
  Stop only if clip posterior max ≥ this value.
- **`-TimeMargin`** *(default: 0.0)*  
  Optional: require (top1 − top2) ≥ margin. 0 disables margin check.
- **`-EvalFixedKWindows`** *(default: 3)*  
  Diagnostics only: evaluate first/mid/last **K windows per clip**  
  (**K here is windows, not exits**).
- **`-PrintClipWindows`** *(switch)*  
  Print per-clip window counts (`windows_total`, and `windows_used` if time-exit).

### Flip-rate vs savings tuning (time policy)
If you want fewer flips with minimal loss of savings:
- increase `TimeStableK` (e.g., 3)
- and/or set `TimeMargin` to ~0.08

Example:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.5" -Policy "ea" -Device "cpu" `
  -SegmentSec 1.0 -HopSec 0.5 -LambdaDepth 0.02 -EAMinExit 2  # optional; default auto does this for K=5 `
  -TapBlocks "1,2,3,4" -NMels 64 `
  -RunClipPolicy -TimeConf 0.95 -TimeStableK 3 -TimeMinWindows 2 -TimeMargin 0.08 -EvalFixedKWindows 3
```

---

## Key outputs (created in `run_dir`)

### Segment policy (`scripts/policy_test.py`)
- `policy_results.json` (segment accuracy, exit mix e1..eK, avg depth, flip-rate, exit-consistency, plus clip-metrics without time stopping)

### Clip policy (`scripts/clip_policy_test.py`)
- `clip_policy_results_full.json` (full clip baseline; no time-exit)
- `clip_policy_results_time.json` (Depth×Time)
- `clip_preds_full.csv`, `clip_preds_time.csv`
- `clip_length_hist.json` (baseline distribution)
- `windows_used_hist.json` (time-exit stopping distribution)

### Summaries and plots
- `summary.json` (`scripts/summarize_run.py`)
- `analysis_run.json`, `confusion_matrices.json`, `roc_curves.json`, `plots/` (`scripts/analyse_run.py`)
- `profiling.json` + `analysis/on_device_summary.csv` (`scripts/profile_latency.py`)

---

## How to interpret the numbers (paper-ready)
It’s normal for **segment accuracy over processed windows** to drop under Depth×Time, because the policy evaluates mostly **early** windows.  
The fixed-position diagnostic provides an apples-to-apples proof:
- if `Acc_firstK < Acc_midK` and `Acc_firstK < Acc_lastK`, then the drop is **early-window difficulty**, not a bug.

---

## Notes / troubleshooting
- If `Compute Saved (%)` prints `N/A`, ensure you ran the **FULL clip baseline** first so `clip_policy_results_full.json` exists in `run_dir`.
- If you see state_dict mismatches (unexpected `exit_heads.*` keys), you likely changed `tap_blocks` between training and calibration/evaluation.
- W&B logging is optional: set `$env:ENABLE_WANDB="1"` and the runner will call `scripts.wandb_log_run`.

---

## Citation / attribution
If you build on this repo, please cite our paper (to be added once submitted).
