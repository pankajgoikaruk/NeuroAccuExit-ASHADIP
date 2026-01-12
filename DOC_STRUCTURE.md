# DOC_STRUCTURE (Repo Map + Versioned Notes)

This file describes **where things live** in NeuroAccuExit‑ASHADIP and how the pipeline ties together.

---

## 1) Top-level layout

```
configs/
  audio_moth.yaml              # main experiment configuration

data/
  moth_sounds/                 # raw AudioMoth wav files (user-provided)

data_caches/
  <variant>/<cache_id>/
    segments.csv               # window metadata (wav_relpath, start, label, split, ...)
    features/                  # cached features for each window (logmel + CMVN)

runs/
  <variant>/<variant>_###/
    meta.json                  # run metadata (variant, cache, device, etc.)
    *.pt                       # model checkpoint(s)
    temperature.json           # per-exit temperature scaling params
    thresholds.json            # greedy τ selection (VAL)
    ea_thresholds.json         # Depth‑EA selection (VAL)
    policy_results.json        # policy_test outputs (PT)
    summary.json               # consolidated run summary
    analysis_run.json          # consolidated analysis JSON
    plots/                     # loss/acc curves, confusion matrices, ROC, etc.
    profiling.json             # latency profiling results

scripts/
  run_full.ps1                 # full pipeline runner (Windows PowerShell)
  prep_segments.py             # make segments.csv
  extract_features.py          # compute cached features
  policy_test.py               # policy evaluation (greedy or ea)
  analyse_run.py               # plots + consolidated analysis JSON
  summarize_run.py             # summary.json + experiments.csv logging
  profile_latency.py           # on-device timing/profiling + CSV summary

training/
  train.py                     # train ExitNet (multi-exit)
  calibrate.py                 # temperature scaling per exit
  thresholds_offline.py        # greedy τ sweep on VAL → thresholds.json
  ea_thresholds_offline.py     # Depth‑EA sweep on VAL → ea_thresholds.json (+ ea_sweep_results.json)

analysis/
  pipeline_runtime.csv         # wall-clock runtime log from run_full.ps1
  on_device_summary.csv        # profiling summary appended by profile_latency.py
```

---

## 2) Core concepts

### Windows vs WAV files
We operate at **window level**:
- each wav is segmented into overlapping windows (e.g., 1.0s with 0.5s hop),
- each window gets features and is classified,
- early exit is decided per window (v0.1 / v0.2).

*(Future versions may add file-level temporal accumulation.)*

### Multi-exit ExitNet
The model produces logits at exits:
- exit1 (early, cheap)
- exit2 (middle)
- exit3 (late, accurate)

---

## 3) Policies

### Greedy (v0.1)
- use `thresholds_offline.py` to select τ on VAL
- `policy_test --policy greedy` evaluates on PT
Outputs: `thresholds.json`, `policy_results.json`

### Depth‑EA (v0.2)
- use `ea_thresholds_offline.py` to select EA threshold / stability params on VAL
- `policy_test --policy ea` evaluates on PT
Outputs: `ea_thresholds.json`, `ea_sweep_results.json`, `policy_results.json`

Depth‑EA selection uses a score trading off F1/Acc vs depth:
- score = F1 + 0.10×Acc − λ_depth×AvgExitDepth  
Higher `λ_depth` → lower depth (often lower accuracy).

---

## 4) Versioned implementation notes

### v0.1 — baseline early exit
- Greedy τ selection + temperature scaling
- Policy test + analysis plots
- Focus: reproducible pipeline, leakage-safe splitting

### v0.2 — Depth Evidence Accumulation across exits
Adds:
- `training/ea_thresholds_offline.py`
- `--policy ea` in `scripts/policy_test.py`
- Runner support (`scripts/run_full.ps1`) for variant-based policy selection

### v0.2.x — tuning for better EA accuracy
Typical tuning axes:
- `train.loss_weights` (avoid e1 dominating too strongly)
- KD: exit3 teaches exit1/exit2 (`train.kd.*`)
- SpecAug strength/probability (robustness without accuracy collapse)
- EA selector λ_depth (accuracy vs depth)

---

## 5) “Master table” workflow (paper tracking)
We maintain a master table of runs (per variant / run_id) with:
- Greedy and EA policy_test (PT) results
- exit mix, avg exit depth
- comparison rows (ΔAcc vs Greedy, depth gain, ΔDepth%, Δ(Acc/Depth))

This is the main story device for the paper: **how each change shifts accuracy vs efficiency**.
