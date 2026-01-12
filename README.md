# NeuroAccuExit-ASHADIP

Early-exit (multi-exit) inference for **AudioMoth moth wingbeat classification**, with a reproducible pipeline for:
- window segmentation + feature caching (log-mel + CMVN),
- training a 3-exit ExitNet,
- calibration (temperature scaling per exit),
- **Greedy early-exit** thresholding (τ),
- **Depth Evidence Accumulation (Depth‑EA)** policy + offline EA sweep selection,
- policy testing + analysis plots + on-device profiling summaries.

This repo is organised around **versioned variants** (v0.1, v0.2, …) so we can track the trade-off between **accuracy** and **efficiency**.

---

## What’s new in v0.2 (Depth‑EA)

v0.2 introduces an alternative to greedy early exit:

- **Greedy (v0.1 style):** exit as soon as the current exit’s confidence ≥ τ  
- **Depth‑EA (v0.2):** accumulate evidence across exits (e1→e3) and exit when the evidence margin/log‑prob meets an EA threshold

In the pipeline, v0.2 adds:
- `training/ea_thresholds_offline.py` (offline sweep: EA threshold / stable_k / flip_penalty, plus a score that trades off F1/Acc vs depth)
- `scripts/policy_test.py --policy ea` (Depth‑EA policy test)
- a unified runner: `scripts/run_full.ps1` with `-Variant v0.2` and `-Policy ea|greedy|auto`

---

## Quickstart

### 1) Environment
Create/activate your environment and install dependencies.

```bash
pip install -r requirements.txt
```

### 2) Data layout
Point `-DataRoot` to a folder that contains `moth_sounds` (or use the defaults).

Example:
```
data/
  moth_sounds/
    <wav files...>
```

### 3) Run the full pipeline (recommended)

#### v0.2 Depth‑EA (default for v0.2)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.2" `
  -Policy "ea" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5
```

#### v0.2 Greedy (same model, greedy decision rule)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.2" `
  -Policy "greedy" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5
```

#### Notes on `-Policy`
- `auto`: follows the Variant rule (EA for `EA` / `v0.2`, otherwise greedy)
- `ea`: forces Depth‑EA threshold selection + EA policy test
- `greedy`: forces τ selection + greedy policy test

---

## Cache & run directory scheme

### Cache
Caches are stored as:
```
data_caches/<Variant>/<CacheId>/
  segments.csv
  features/
```

CacheId is auto-generated if not provided:
- `seg{SegmentSec}_hop{HopSec}_bp100-3000_mels64` (dots become `p`, e.g., `0.5` → `0p5`)

### Runs
Runs are stored as:
```
runs/<Variant>/<Variant>_###/
  meta.json
  ckpt.pt (or similar)
  temperature.json
  thresholds.json              # greedy τ selection (VAL)
  ea_thresholds.json           # Depth‑EA selection (VAL)
  policy_results.json          # policy test results (PT)
  summary.json
  analysis_run.json
  plots/
  profiling.json
```

---

## Controlling the EA “depth vs accuracy” trade-off (λ_depth)

Depth‑EA selection uses a score like:
> score = F1 + 0.10×Acc − λ_depth×AvgExitDepth  
(with optional constraints, depending on your selector settings)

A **higher λ_depth** pushes the selector to choose **lower depth** (often at the cost of accuracy).  
A **lower λ_depth** allows deeper exits more often (typically improving accuracy).

If your `run_full.ps1` exposes `-LambdaDepth`, run:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.2" `
  -Policy "ea" `
  -LambdaDepth 0.03 `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5
```

If you don’t have `-LambdaDepth` in your runner yet, you can still run the selector directly:
```powershell
python -m training.ea_thresholds_offline `
  --run_dir "runs\v0_2\v0_2_007" `
  --segments_csv "data_caches\v0_2\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_2\seg1_hop0p5_bp100-3000_mels64\features" `
  --lambda_depth 0.03
```

---

## Re-running policy_test without retraining
You **do not** need to retrain to compare policies on the **same trained model**.

Once a run exists (has a checkpoint + calibration):
- Greedy uses `thresholds.json` (τ)
- Depth‑EA uses `ea_thresholds.json` (EA threshold + stability params)

So you can re-run:
```powershell
python -m scripts.policy_test --policy greedy --run_dir "<RUN_DIR>" --segments_csv "<SEGMENTS>" --features_root "<FEATS>"
python -m scripts.policy_test --policy ea     --run_dir "<RUN_DIR>" --segments_csv "<SEGMENTS>" --features_root "<FEATS>"
```

If a run is missing `thresholds.json`, generate it:
```powershell
python -m training.thresholds_offline --run_dir "<RUN_DIR>" --segments_csv "<SEGMENTS>" --features_root "<FEATS>"
```

---

## Configuration
Main config: `configs/audio_moth.yaml`.

Common knobs we tune in v0.2.x:
- `train.loss_weights` (strength of each exit’s supervised loss)
- `train.kd` (distillation from exit3 → exit1/exit2)
- `train.specaug` (robustness; keep modest probability/strength to avoid collapsing accuracy)
- `ea.*` (mode, threshold grid, stability, flip penalty)

---

## Version history (high-level)

- **v0.1**: baseline greedy early exit (τ), temperature scaling, offline τ sweep, policy_test + plots
- **v0.2**: Depth‑EA (evidence accumulation across exits) + EA offline sweep selector
- **v0.2.x**: tuning KD + SpecAug + loss weights to improve accuracy without losing depth too much
- **v0.3**: (work-in-progress) major accuracy improvements / new research ideas (keep master table updated)

---

## Security note (PyTorch warning)
You may see a PyTorch warning about `torch.load(..., weights_only=False)`.  
This is expected; do not load untrusted checkpoints. If needed, migrate to `weights_only=True` once your code is ready.
