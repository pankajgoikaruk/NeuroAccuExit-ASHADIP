# NeuroAccuExit-ASHADIP (ASHADIP)

TinyML-friendly **multi-exit audio classifier** with **Depth×Time early-exit**:
- **Depth early-exit (segment-level)**: choose the exit depth per segment using *Depth-EA*.
- **Time early-exit (clip-level)**: process segments sequentially and stop early when the clip prediction becomes **stable + confident** via **temporal evidence accumulation**.

This repo supports reproducible runs on cached log-mel features and produces paper-ready metrics (accuracy vs compute, stopping distributions, diagnostic baselines).

---

## What’s new (v0.3.6)

### 1) Clip-level (Depth×Time) policy evaluation
`scripts/clip_policy_test.py` now reports:
- **Segment accuracy over processed windows** (comparable to `policy_test.py`, but only on windows actually processed under time-exit)
- **Clip accuracy**
- **Windows Saved (%)** and **Compute Saved (%)** (requires a full-clip baseline JSON)
- **Stop-window distribution** (min/median/max/mean + histogram) when time-exit is enabled
- **Clip-length distribution** (min/median/max/mean + histogram) when `--disable_time_exit` is used

### 2) Reviewer-proof diagnostics (apples-to-apples)
To explain why segment accuracy can drop under aggressive time-exit (because early windows are harder), `clip_policy_test.py` adds:
- **Fixed-position diagnostic** on every clip (independent of time-exit stopping):
  - `Acc_firstK`, `Acc_midK`, `Acc_lastK` for **First-K / Middle-K / Last-K** windows
- **Stop-speed groups** under Depth×Time:
  - compare early-window accuracy for clips that stop at **2**, **3–4**, and **≥5** windows
  - demonstrates: *easy clips stop early; hard clips need more evidence* (while clip accuracy remains high)

### 3) Traceability: per-clip window counts
Optional printing of:
- `clip_i -> windows_total=... , windows_used=... | id=<wav_relpath>`
so you can trace outliers quickly.

---

## Quickstart

### Environment
Create/activate your environment (example):
```powershell
conda activate ASHADIP_V0
```

### Typical evaluation flow

#### (1) Segment policy (Depth-EA per segment)
```powershell
python -m scripts.policy_test `
  --policy ea `
  --run_dir "runs\v0_3\v0_3_027" `
  --segments_csv "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\features"
```

#### (2) Clip policy baseline (FULL clip; no time early-exit)
This produces the compute reference used by Depth×Time to compute Compute Saved (%).
```powershell
python -m scripts.clip_policy_test `
  --run_dir "runs\v0_3\v0_3_027" `
  --segments_csv "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\segments.csv" `
  --features_root "data_caches\v0_3\seg1_hop0p5_bp100-3000_mels64\features" `
  --disable_time_exit `
  --eval_fixed_k_windows 3 `
  --print_clip_windows
```

#### (3) Clip policy with TIME early-exit (Depth×Time)
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

## Key outputs (created in `run_dir`)

### Segment policy (`scripts/policy_test.py`)
- `policy_results.json` (segment accuracy, exit mix, avg depth, flip-rate, etc.)

### Clip policy (`scripts/clip_policy_test.py`)
- JSON:
  - `clip_policy_results.json` (legacy)
  - `clip_policy_results_full.json` (when `--disable_time_exit`)
  - `clip_policy_results_time.json` (when time-exit enabled)
- CSV:
  - `clip_preds.csv` (legacy)
  - `clip_preds_full.csv` / `clip_preds_time.csv`
- Distribution JSON:
  - baseline: `clip_length_hist.json` (+ mode-specific `clip_length_hist_full.json`)
  - time-exit: `windows_used_hist.json` (+ mode-specific `windows_used_hist_time.json`)

---

## How to interpret the numbers (paper-ready)
It’s normal for **segment accuracy over processed windows** to drop under Depth×Time, because the policy evaluates mostly **early** windows.  
The fixed-position diagnostic provides an apples-to-apples proof:
- if `Acc_firstK < Acc_midK` and `Acc_firstK < Acc_lastK`, then the drop is **early-window difficulty**, not a bug or “early-clip bias”.
The stop-speed grouping further shows:
- **easy clips stop at 2 windows** with higher early-window accuracy,
- **hard clips require ≥5 windows**.

---

## Notes / troubleshooting
- If `Compute Saved (%)` prints `N/A`, ensure you ran the **FULL clip baseline** first so `clip_policy_results_full.json` exists in `run_dir`.
- If you see missing feature files, confirm `--features_root` matches your cache directory and that `segments.csv` contains valid `feat_relpath` values.

---

## Citation / attribution
If you build on this repo, please cite our paper (to be added once submitted).
