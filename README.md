# ASHADIP / NeuroAccuExit — `kexit-greedy-hint-gunshot`

This branch documents the **gunshot adaptation** of the generic K-exit / hint pipeline.

It is built on top of the earlier K-exit / C-class refactor, but the dataset and reporting focus are now different:

- **old branch/docs**: moth wingbeat audio (`male` vs `female`)
- **this branch**: gunshot audio (`gunshot` vs `non_gunshot`)

The goal of this branch is **fair reuse of the existing architecture** on a new binary audio task, while preserving:

- dynamic **3-exit** and **5-exit** settings
- greedy no-hint and greedy hint-enabled runs
- segment-level and clip-level evaluation
- Depth×Time clip policy analysis

This branch is currently the correct place to document the gunshot binary study, not the older moth-only README.

---

## Current task

### Binary classification
- `gunshot`
- `non_gunshot`

### Non-gunshot includes
- fireworks
- engine / background-type clips
- other negative / confusing sounds placed under `data2/non_gunshot`

### Dataset root used in this branch
```text
data2/
  gunshot/
  non_gunshot/
```

---

## Why this branch exists

The earlier K-exit branch already generalized the model side well, but the data-preparation path was still moth-specific.

This branch adds the missing gunshot-specific practical pieces:

- generic label discovery from subfolders
- mixed-length audio support
- preprocessing for `.wav`, `.flac`, and other supported audio formats
- inventory summary before segmentation
- file-level train/val/test splitting
- support for short clips
- control over segment explosion from very long non-gunshot recordings
- short Windows-safe processed filenames
- run commands specialized for the gunshot binary dataset

---

## Main implementation changes in this branch

### 1. `scripts/prep_segments.py`
Updated so that it now:

- auto-discovers class folders under `--root`
- reads mixed audio formats
- writes `audio_inventory.csv`
- writes `audio_inventory_by_label.csv`
- supports `--min_keep_sec`
- supports `--max_segments_per_file_gunshot`
- supports `--max_segments_per_file_non_gunshot`
- performs file-level train/val/test splitting
- handles very short clips through padded 1-second segments when appropriate
- uses shorter hashed processed filenames to avoid Windows path-length failures

### 2. `scripts/run_full.ps1`
Updated so that it now supports gunshot data preparation directly through CLI:

- `-DataRoot "data2"`
- `-MinKeepSec`
- `-MaxSegmentsPerFileGunshot`
- `-MaxSegmentsPerFileNonGunshot`
- `-ExitHint "true|false"`
- gunshot-oriented cache naming
- run metadata logging for the new preprocessing controls

### 3. `scripts.extract_features.py` usage
The full pipeline now uses `--pad_short` during feature extraction so short gunshot clips do not produce missing `feat_relpath` entries.

---

## Current end-to-end pipeline

1. raw audio under `data2/gunshot` and `data2/non_gunshot`
2. `scripts/prep_segments.py`
   - scan inventory
   - create cleaned cache WAVs
   - create `segments.csv`
   - apply file-level split
3. `scripts.extract_features.py`
   - compute log-mel `.npy` features
   - update `feat_relpath`
4. `data/datasets.py`
   - build PyTorch datasets and loaders
5. `adapters/audio_adapter.py`
   - dynamic `TinyAudioCNN` with configurable tap blocks
6. `models/exit_net.py`
   - generic multi-exit classifier with optional sequential hint passing
7. `training/train.py`
   - train the K-exit model
8. `training/calibrate.py`
   - fit per-exit temperatures
9. `training/thresholds_offline.py`
   - greedy threshold selection
10. `scripts/policy_test.py`
   - segment-level greedy policy
11. `scripts/clip_policy_test.py`
   - full-clip baseline and Depth×Time evaluation
12. `scripts/summarize_run.py`
13. `scripts/analyse_run.py`
14. `scripts/profile_latency.py`
15. `scripts/run_reports.ps1`

---

## Dataset summary used for the current gunshot experiments

The inventory summary for the current binary dataset is:

- **1712 total files**
- **896 gunshot**
- **816 non_gunshot**

Duration statistics:

| Class | Files | Min sec | Median sec | Mean sec | Max sec |
|---|---:|---:|---:|---:|---:|
| gunshot | 896 | 0.4988 | 1.9461 | 1.5271 | 44.0000 |
| non_gunshot | 816 | 0.5151 | 4.9998 | 7.4386 | 994.3688 |

Shorter than 1 second:

- gunshot: **250**
- non_gunshot: **3**

Because very long non-gunshot files were flooding the segment count, the current branch uses:

- `MaxSegmentsPerFileGunshot = 0` → keep all
- `MaxSegmentsPerFileNonGunshot = 5` → cap non-gunshot windows per file

This produced the working split:

| Split | Gunshot | Non-gunshot | Total |
|---|---:|---:|---:|
| train | 1295 | 2323 | 3618 |
| val | 267 | 486 | 753 |
| test | 259 | 515 | 774 |

---

## Validated run settings on this branch

### 3-exit no-hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs3" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy `
  -CacheId "s1_h05_t13_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0
```

### 3-exit hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs3_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy `
  -CacheId "s1_h05_t13_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0 `
  -ExitHint "true"
```

### 5-exit no-hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs5" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy `
  -CacheId "s1_h05_t1234_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0
```

### 5-exit hint
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "gs5_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -TapBlocks "1,2,3,4" `
  -RunClipPolicy `
  -CacheId "s1_h05_t1234_ng5" `
  -MaxSegmentsPerFileNonGunshot 5 `
  -MaxSegmentsPerFileGunshot 0 `
  -ExitHint "true"
```

---

## Main artifacts per run

Each run still generates the standard output set:

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
- `meta.json`
- plots under `plots/`

---

## Main results on the gunshot dataset

## 1) Per-exit test accuracy

| Metric | `gs3` | `gs3_hint` | Δ (Hint−No-Hint) | `gs5` | `gs5_hint` | Δ (Hint−No-Hint) |
|---|---:|---:|---:|---:|---:|---:|
| Exit1 accuracy | 0.8269 | 0.8165 | -0.0104 | 0.8049 | 0.8140 | +0.0091 |
| Exit2 accuracy | 0.9380 | 0.9432 | +0.0052 | 0.9057 | 0.8915 | -0.0142 |
| Exit3 accuracy | 0.9587 | 0.9548 | -0.0039 | 0.9509 | 0.9496 | -0.0013 |
| Exit4 accuracy | — | — | — | 0.9651 | 0.9561 | -0.0090 |
| Exit5 accuracy | — | — | — | 0.9599 | 0.9625 | +0.0026 |

## 2) Segment-level greedy policy

| Metric | `gs3` | `gs3_hint` | Δ | `gs5` | `gs5_hint` | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Policy accuracy | 0.9548 | 0.9535 | -0.0013 | 0.9587 | 0.9561 | -0.0026 |
| Avg exit depth | 1.801 | 1.810 | +0.009 | 2.265 | 2.288 | +0.023 |
| Flip-any rate | 0.1628 | 0.1809 | +0.0181 | 0.2080 | 0.1744 | -0.0336 |
| Avg flip count | 0.1731 | 0.2003 | +0.0272 | 0.2532 | 0.2054 | -0.0478 |
| Exit consistency | 0.9935 | 0.9987 | +0.0052 | 0.9987 | 0.9910 | -0.0077 |

## 3) Full-clip baseline

| Setting | Clip acc | Processed-win acc | Avg windows used | Avg compute units | Avg depth per used window |
|---|---:|---:|---:|---:|---:|
| `gs3` | 0.9844 | 0.9548 | 3.012 / 3.012 | 5.424 | 1.801 |
| `gs3_hint` | 0.9650 | 0.9535 | 3.012 / 3.012 | 5.451 | 1.810 |
| `gs5` | 0.9767 | 0.9587 | 3.012 / 3.012 | 6.821 | 2.265 |
| `gs5_hint` | 0.9689 | 0.9561 | 3.012 / 3.012 | 6.891 | 2.288 |

## 4) Depth×Time

| Metric | `gs3` | `gs3_hint` | Δ | `gs5` | `gs5_hint` | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Clip accuracy | 0.9844 | 0.9650 | -0.0194 | 0.9728 | 0.9650 | -0.0078 |
| Used-window segment accuracy | 0.9558 | 0.9512 | -0.0046 | 0.9578 | 0.9551 | -0.0027 |
| Avg windows used | 1.759 / 3.012 | 1.755 / 3.012 | -0.004 / 3.012 | 1.751 / 3.012 | 1.732 / 3.012 | -0.019 / 3.012 |
| Windows saved | 41.60% | 41.73% | +0.13 pp | 41.86% | 42.51% | +0.65 pp |
| Avg compute units | 3.339 | 3.401 | +0.062 | 4.218 | 4.128 | -0.090 |
| Avg depth per used window | 1.898 | 1.938 | +0.040 | 2.409 | 2.384 | -0.025 |
| Compute saved | 38.45% | 37.62% | -0.83 pp | 38.16% | 40.09% | +1.93 pp |
| Flip-rate | 0.1770 | 0.1863 | +0.0093 | 0.2089 | 0.1798 | -0.0291 |
| Exit consistency | 0.9978 | 0.9978 | +0.0000 | 0.9978 | 0.9933 | -0.0045 |

---

## Short interpretation of the current four-run comparison

The current gunshot dataset gives a very different picture from the older moth branch.

### Main observations

1. **`gs3` (3-exit no-hint)** is the current **best overall practical model**.
   - highest full-clip accuracy: **0.9844**
   - highest Depth×Time clip accuracy: **0.9844**
   - lowest compute among the strong-performing settings

2. **`gs3_hint` is worse overall** on this dataset.
   - exit2 improves slightly
   - but final exit accuracy, segment-policy accuracy, and clip accuracy all decline

3. **`gs5` is the strongest deeper no-hint reference**.
   - good segment-level behavior
   - stronger deeper exits than the 3-exit model
   - but still lower clip accuracy than `gs3`

4. **`gs5_hint` is mixed**.
   - slight gain at the final exit
   - slightly better validation threshold accuracy
   - slightly better Depth×Time compute saving
   - but lower segment-policy and clip accuracy than `gs5`

---

## Reviewer-safe conclusion

The current gunshot binary study supports the following claim:

- **hint passing is not universally beneficial**
- it may still be behaviorally meaningful
- but on this short, binary gunshot dataset it does **not** improve the overall greedy accuracy-efficiency trade-off

The clean current recommendation is:

- **best overall practical model:** `gs3`
- **best deeper no-hint reference:** `gs5`
- **best hint result on this dataset:** `gs5_hint` only in a limited, mixed sense
- **clear negative result:** `gs3_hint`

---

## Why the result differs from the older branch

This dataset is different in ways that matter:

- binary gunshot vs non-gunshot is a simpler label space
- clips are much shorter on average
- the available windows per clip are only about **3.012**, not the much larger values seen in the older branch
- many gunshot files are shorter than 1 second and required careful handling

So the current evidence suggests that hint passing may need:

- richer temporal evidence
- more ambiguous intermediate decisions
- or a different task regime

before it becomes consistently beneficial.

---

## Current research takeaway

This branch should now be described as:

> a gunshot-adapted generic K-exit / hint pipeline that successfully transfers the existing early-exit architecture to a new binary audio domain, while showing that sequential hint passing is **task-dependent** rather than universally beneficial.

That is a strong and honest result.

---

## Next recommended experiment

The best next controlled follow-up is:

- keep `segment_sec = 1.0`
- reduce `hop` from `0.5` to `0.25`
- compare:
  - `gs3_h025`
  - `gs3_hint_h025`

That tests whether hint begins to help when the model receives more temporal evidence per clip, without changing the dataset or architecture fundamentally.
