# NeuroAccuExit / ASHADIP — TinyML Audio Early-Exit (Moth Wingbeats)

This repository implements a lightweight **multi-exit** audio classifier for moth wingbeats, designed for **efficient (TinyML/edge)** inference.
The system trains an ExitNet (early exits + final head), calibrates per-exit temperatures, and selects an **evidence-accumulation (EA)** early-exit policy **offline on validation**.

> Data format: put raw WAVs under `data/moth_sounds/male/` and `data/moth_sounds/female/` (or the `DataRoot` you pass to the scripts).


## What this project does

### Pipeline (high-level)
1. **Segment audio** into fixed-length windows with overlap (streaming-friendly).
2. **Extract log-mel** features (optionally CMVN) and cache them deterministically.
3. **Train** a small CNN backbone + multiple exit heads (ExitNet).
4. **Calibrate** each exit head with temperature scaling (saved to `temperature.json`).
5. **Select** EA policy parameters (saved to `ea_thresholds.json`) via offline sweep on VAL.
6. **Evaluate** the chosen policy on TEST (saved to `policy_results.json`).


## Research contributions inside the repo (what’s “new” here)
Keep these as *project contributions*; reviewers like them when they are tied to measured behavior.

- **Stability-aware evidence accumulation across exits (Depth-EA):** decisions are made from *accumulated evidence* (log-prob/logits) across exits, with stability control (`ea_stable_k`) and optional flip penalty.
- **Exit1 high-confidence override (fixes a real failure mode):** when `ea_stable_k>1`, exit-1 can otherwise become impossible. We add a strict confidence/margin gate so easy samples can still exit early without collapsing accuracy.
- **Adaptive-but-offline policy selection:** EA knobs are selected on VAL with a compute–accuracy trade-off objective (via `lambda_depth`). Parameters are fixed at inference time (TinyML-friendly).
- **Observed “regime switch” behavior:** small changes in `lambda_depth` can trigger a discrete policy flip (e.g., shallow/low-acc vs deeper/high-acc), which is useful for a paper discussion on controllable trade-offs.


## Quickstart (recommended: one command)

### Windows PowerShell
```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.3" `
  -Policy "ea" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -LambdaDepth 0.08
```

This runs the whole pipeline end-to-end and writes a new run folder under `runs\v0_3\...`.

### Optional: Weights & Biases tracking (PowerShell)
```powershell
wandb login
$env:ENABLE_WANDB="1"
$env:WANDB_PROJECT="NeuroAccuExit-ASHADIP"
$env:WANDB_ENTITY="pankajgoikar-lancaster-university"
$env:WANDB_MODE="online"
$env:WANDB_TAGS="v0.3,posthoc"
$env:WANDB_DIR="$PWD\.wandb_runs"
New-Item -ItemType Directory -Force -Path "$env:WANDB_DIR" | Out-Null
```


## Manual run (if you want to run steps separately)

### 1) Build segments & manifest (cached)
```bash
python -m scripts.prep_segments --root data/moth_sounds --cache data_caches   --sr 16000 --segment_sec 1.0 --hop 0.5 --silence_dbfs -40 --bandpass 100 3000
```

### 2) Extract log-mel features (cached)
```bash
python -m scripts.extract_features --cache data_caches --n_mels 64   --n_fft 1024 --win_ms 25 --hop_ms 10 --cmvn
```

### 3) Train ExitNet
```bash
python -m training.train --config configs/audio_moth.yaml --run_dir runs/latest
```

### 4) Calibrate temperatures
```bash
python -m training.calibrate --run_dir runs/latest   --segments_csv data_caches/.../segments.csv --features_root data_caches/.../features
```

### 5) Select EA thresholds (offline on VAL)
```bash
python -m training.ea_thresholds_offline --run_dir runs/latest   --segments_csv data_caches/.../segments.csv --features_root data_caches/.../features   --lambda_depth 0.08
```

### 6) Policy test (TEST split)
```bash
python -m scripts.policy_test --policy ea --run_dir runs/latest   --segments_csv data_caches/.../segments.csv --features_root data_caches/.../features
```


## Outputs (per run folder)
A typical run directory contains:
- `ckpt/best.pt` — best checkpoint
- `temperature.json` — per-exit temperatures
- `ea_thresholds.json` — selected EA policy parameters
- `ea_sweep_results.json` — sweep table for analysis
- `policy_results.json` — final test metrics + exit mix


## EA knobs (the key policy parameters)
Saved in `ea_thresholds.json` and used by `scripts/policy_test.py`.

- `ea_threshold`: margin threshold on accumulated evidence
- `ea_stable_k`: require K consecutive consistent predictions before exiting
- `ea_flip_penalty`: penalize margin on prediction flips (optional)
- `ea_mode`: `logprob` or `logits` accumulation
- `ea_min_exit`: minimum exit index allowed

**Exit1 override knobs (to avoid “Exit1 deadlock” when `stable_k>1`):**
- `ea_exit1_conf_min`: minimum softmax confidence (e.g., 0.90 or 0.95)
- `ea_exit1_margin_mult`: require margin >= mult × `ea_threshold` for Exit1
- `ea_exit1_margin_min`: optional absolute margin floor for Exit1


## Reproducibility notes
- Feature caching and deterministic splits are used to reduce preprocessing confounders.
- On CPU, runs are often deterministic; if you want a robustness test, change training seed (if exposed) or run on GPU with/without deterministic settings.


## License / Citation
Add your preferred license and (when ready) a short citation block for the paper.
