# Repository Structure (DOC_STRUCTURE)

This file documents the **current project layout** and the key scripts used for **Depth×Time early-exit** experiments.

---

## Top-level layout

```
NeuroAccuExit-ASHADIP/
├─ adapters/                 # model backbones / adapters (e.g., TinyAudioCNN)
├─ configs/                  # YAML configs (dataset, features, training knobs)
├─ data/                     # raw audio (optional; depends on your workflow)
├─ data_caches/              # cached segments + per-segment features (log-mel)
├─ models/                   # ExitNet (multi-exit) + components
├─ policies/                 # early-exit policies (Depth-EA, greedy, etc.)
├─ scripts/                  # entry points for running and evaluating
├─ training/                 # training utilities + EA threshold search/calibration
├─ utils/                    # helpers (logging, misc)
└─ runs/                     # experiment outputs (checkpoints, json, csv)
```

> Exact subfolders may vary by branch/tag; the entries below are the ones used in v0.3.x.

---

## Core files and what they do

### `scripts/policy_test.py` (segment policy evaluation)
Evaluates the **segment-by-segment** policy (Depth-EA / greedy) on the test split.

**Outputs (in `run_dir`)**
- `policy_results.json` containing:
  - segment accuracy
  - avg exit depth
  - exit mix (e1/e2/e3)
  - flip-rate and exit-consistency (taken vs final)
  - clip-metrics computed without time stopping (for fair comparison)

### `scripts/clip_policy_test.py` (Depth×Time evaluation)
Evaluates **clip-level** inference by grouping segments by `wav_relpath` and processing them in time order.

**Two modes**
1) **Full clip baseline** (`--disable_time_exit`)
   - processes all segments of each clip
   - prints **clip-length distribution** over `windows_total`
2) **Depth×Time** (default)
   - stops early when prediction is **stable + confident**
   - prints **stop-window distribution** over `windows_used`

**Key printed metrics**
- Segment accuracy over processed windows:
  - `Policy test accuracy (segments, processed windows): ... (n_segments=...)`
- Fixed-position diagnostic (reviewer-proof):
  - `Acc_firstK`, `Acc_midK`, `Acc_lastK` (computed for every clip regardless of stopping)
- Stop-speed groups (Depth×Time only):
  - stop_2 vs stop_3_4 vs stop_5_plus with early-window accuracy
- Clip accuracy, window savings, compute savings, exit mix, flip-rate, exit consistency

**Outputs (in `run_dir`)**
- JSON:
  - `clip_policy_results.json` (legacy)
  - `clip_policy_results_full.json` (full clip baseline)
  - `clip_policy_results_time.json` (Depth×Time)
- CSV:
  - `clip_preds.csv` (+ mode-specific variants)
- Distribution JSON:
  - `clip_length_hist.json` (+ `clip_length_hist_full.json`)
  - `windows_used_hist.json` (+ `windows_used_hist_time.json`)

---

## Data cache expectations

`segments.csv` must contain at least:
- `wav_relpath` : clip id (relative path for grouping)
- `start`       : start time for ordering segments within a clip
- `feat_relpath`: relative feature path under `features_root`
- `label`       : ground-truth clip label
- `split`       : train/val/test

Feature files referenced by `feat_relpath` are expected to be `.npy` arrays shaped like `(M, T)` (log-mel).

---

## Example commands (PowerShell)

### Segment policy
```powershell
python -m scripts.policy_test `
  --policy ea `
  --run_dir "runs\v0_3\v0_3_027" `
  --segments_csv "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\features"
```

### Clip baseline (full clip; no time-exit)
```powershell
python -m scripts.clip_policy_test `
  --run_dir "runs\v0_3\v0_3_027" `
  --segments_csv "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\features" `
  --disable_time_exit `
  --eval_fixed_k_windows 3 `
  --print_clip_windows
```

### Clip policy (Depth×Time)
```powershell
python -m scripts.clip_policy_test `
  --run_dir "runs\v0_3\v0_3_027" `
  --segments_csv "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\features" `
  --time_conf 0.95 `
  --time_stable_k 2 `
  --time_min_windows 2 `
  --eval_fixed_k_windows 3 `
  --print_clip_windows
```

---

## Where to record results for the paper

Recommended artifacts to keep per run:
- `policy_results.json`
- `clip_policy_results_full.json`
- `clip_policy_results_time.json`
- `windows_used_hist_time.json`
- `clip_preds_time.csv`

These provide:
- accuracy vs compute (full vs Depth×Time)
- stop-window distribution (median, histogram)
- fixed-position diagnostic (first/mid/last K)
- stop-speed group table (easy vs hard clips)
