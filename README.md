# ASHADIP v0.1.6 — Audio (Moth Wingbeats)

ASHADIP v0.1.6 is the stabilized **3-exit moth wingbeat baseline** for binary audio classification (**male** vs **female**) using:

- log-mel spectrogram features
- a lightweight **TinyAudioCNN + ExitNet** model
- per-exit temperature calibration
- offline greedy threshold selection (`tau`)
- standardized **segment-policy** and **clip-wise Depth×Time** evaluation

This version is designed to make **accuracy/efficiency trade-offs** easy to reproduce and compare across runs.

---

## What is new in v0.1.6

This update focuses on **evaluation consistency** and **fair-comparison reporting**.

### Added / standardized

- `scripts/policy_test.py` now:
  - accepts `--policy` and `--num_workers`
  - writes `policy_results.json`
  - prints policy metrics in a stable format
- `scripts/clip_policy_test.py` now evaluates **clip-wise greedy Depth×Time** by:
  - grouping windows clip-wise using `wav_relpath`
  - processing windows in time order
  - stopping early when the clip rule is satisfied
  - saving:
    - `clip_policy_results_full.json`
    - `clip_policy_results_time.json`
    - `clip_policy_results.json` (legacy alias)
    - `windows_used_hist.json`
- `scripts/summarize_run.py` now keeps the run summary aligned with the saved policy/clip JSON artifacts
- `scripts/run_full.ps1` now supports clip-policy evaluation flags for the v0.1 greedy workflow

### Standardized printed metrics

The pipeline now prints metrics in a style suitable for direct table entry, including:

- `Policy test accuracy: x.xxxx (n_segments=...)`
- `Segment acc over processed windows: x.xxxx (n_segments=...)`
- `Segment acc over used windows: x.xxxx (n_segments=...)`
- `Avg windows used`
- `Windows saved`
- `Avg compute units`
- `Compute saved`
- `Flip-rate`
- `Exit-consistency`
- `Exit mix`

---

## Pipeline overview

1. Prepare segmented audio manifest
2. Extract log-mel features
3. Train 3-exit ExitNet model
4. Calibrate per-exit temperatures
5. Select greedy threshold `tau`
6. Evaluate **segment policy**
7. Evaluate **clip policy**:
   - full-clip baseline
   - Depth×Time early stopping
8. Summarize, analyze, and profile the run
9. Generate reports and tables

---

## Main artifacts saved per run

A run directory such as `runs/v0_1/v0_1_007/` typically contains:

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
- plots under `plots/`

---

## Quickstart

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Put raw WAVs in

- `data/male/`
- `data/female/`

### 3. Run the full v0.1.6 greedy pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -Variant "v0.1" `
  -Policy "greedy" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -NMels 64 `
  -RunClipPolicy `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3
```

---

## Main metrics reported

### Segment policy

- Policy test accuracy
- Average exit depth
- Exit mix
- Flip-any rate
- Exit consistency

### Clip policy (full baseline)

- Clip accuracy
- Segment accuracy over processed windows
- Fixed-position diagnostics (first / mid / last K windows)
- Average compute units
- Average depth per used window

### Clip policy (Depth×Time)

- Clip accuracy
- Segment accuracy over used windows
- Average windows used
- Windows saved
- Average compute units
- Average depth per used window
- Compute saved
- Flip-rate over used windows
- Exit-consistency over used windows
- Exit mix over used windows

---

## Notes and current scope

- This v0.1.6 path is a **3-exit greedy baseline**.
- `scripts/clip_policy_test.py` in this line is intended for **greedy clip evaluation**, not EA.
- This release is mainly for:
  - stable baseline reproduction
  - fair comparison against later 3-exit and 5-exit variants
  - clean table/graph generation

---

## Recommended use of v0.1.6

Use this release as the reference point for:

- the **historical v0.1 greedy baseline**
- the **3-exit segment-vs-Depth×Time comparison**
- standardized metric printing and JSON output before moving to later EA / hint / K-exit variants
