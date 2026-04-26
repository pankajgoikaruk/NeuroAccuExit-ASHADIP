# ASHADIP / NeuroAccuExit — Generic K-Exit, C-Class Greedy + Hint Audio Pipeline

This branch documents the current **generic K-exit / C-class audio early-exit pipeline** for ASHADIP. It extends the earlier moth-only greedy/hint work into a reusable pipeline that can run:

- **2-class moth wingbeat classification**: `female`, `male`
- **10-class audio classification**: `car_crash`, `conversation`, `engine_idling`, `fireworks`, `gun_shot`, `rain`, `road_traffic`, `scream`, `thunderstorm`, `wind`
- **3-exit models** using `TapBlocks="1,3"`
- **5-exit models** using `TapBlocks="1,2,3,4"`
- **greedy no-hint** and **greedy sequential hint-passing** variants
- **segment-level greedy policy**, **full-clip policy**, and **Depth×Time clip policy**

The current results show that the **new generic preprocessing and physical segment-WAV export pipeline is working correctly**. The moth 2-class runs remain very strong, while the 10-class runs are substantially harder because of class overlap, shorter clips, lower support for `fireworks`, and acoustic similarity among environmental/background classes.

---

## Current reviewer-safe takeaway

The cleanest conclusion from the 8-run controlled study is:

- **Best 2-class moth segment-policy result:** `3exit_2class_greedy_hint`
- **Best 10-class C-class result:** `3exit_cclass_greedy`
- **Best current C-class clip accuracy:** `81.58%` using `3exit_cclass_greedy`
- **Hint passing helps the compact 3-exit binary moth setting**, but does **not** currently help the 10-class setting.
- **5 exits do not currently improve C-class performance**; for C-class, the 5-exit models mostly defer to the deepest exit, so the extra exits do not yet provide useful early decisions.
- **Depth×Time remains useful**, especially for reducing windows and compute while preserving clip-level accuracy.

---

## Key findings overview

| Key finding                                 | Variant                      | Value                         |
|:--------------------------------------------|:-----------------------------|:------------------------------|
| Best 2-class segment policy                 | 3exit_2class_greedy_hint     | 99.38%                        |
| Best 2-class clip accuracy                  | All four 2-class variants    | 100.00%                       |
| Best 10-class segment policy                | 3exit_cclass_greedy          | 69.90%                        |
| Best 10-class full/depth-time clip accuracy | 3exit_cclass_greedy          | 81.58%                        |
| Best 10-class compute saving                | 3exit_cclass_greedy          | 34.34%                        |
| Hint effect on C-class                      | Negative in all C-class runs | -8.22pp / -4.44pp segment acc |

---

## What changed in this branch

### 1. Generic class discovery and C-class support

The preprocessing no longer assumes only `male` and `female`. The pipeline can now receive class labels through the CLI and can support any number of class folders.

Example C-class labels:

```powershell
$CCLASS_LABELS="car_crash,conversation,engine_idling,fireworks,gun_shot,rain,road_traffic,scream,thunderstorm,wind"
```

### 2. Unified audio segmentation flow

The recommended research-ready flow is:

```text
raw parent audio
→ clean/resample/bandpass parent audio
→ split parent files into train/val/test using SplitUnit="file"
→ create 1-second segment rows from each split
→ export physical 1-second segment WAVs
→ extract log-mel features from those physical segment WAVs
→ train/evaluate dynamic K-exit model
```

This gives every feature a clean traceability chain:

```text
feature .npy → 1-second segment WAV → cleaned parent WAV → original parent file
```

The important metadata columns are expected in `segments.csv`, especially:

```text
orig_relpath
clean_relpath
segment_wav_relpath
feat_relpath
label
split
source_file_id
```

### 3. Physical 1-second WAV export is now part of the reproducible workflow

Physical segment WAVs are useful because they can be reused for:

- future experiments
- publication/inspection
- dataset auditing
- Edge Impulse or other TinyML workflows
- sanity-checking individual predictions

Large exported folders such as `segment_wavs/`, `features/`, and `clean/` should remain **ignored by Git**.

### 4. Generic class imbalance controls

For long and short mixed audio clips, per-file and per-class controls are needed. The C-class runs used:

```powershell
$CCLASS_CAPS="gun_shot=0,scream=0,car_crash=0,fireworks=8,rain=5,wind=5,road_traffic=5,engine_idling=5,conversation=5,thunderstorm=5"
```

Interpretation:

- `0` means no special per-class cap beyond other constraints.
- `5` or `8` limits per-file contribution for classes that can produce too many windows.
- This avoids long clips dominating training, validation, and test splits.

### 5. Dynamic K-exit and dynamic C-class model configuration

The code should derive:

```text
num_exits = len(tap_blocks) + 1
num_classes = number of classes found in the dataset / segments.csv
```

For clean reproducibility, avoid hardcoding this in `configs/audio_moth.yaml`:

```yaml
model:
  # num_classes: 2   # keep removed/commented for generic C-class work
```

If a warning appears like:

```text
[WARN] eval num_classes=2 but dataset has 10. Using dataset value.
```

the run can still complete, but the config should be cleaned so the warning disappears.

---

## Main experiment design

The 8-run controlled experiment compares:

| Axis | Values |
|---|---|
| Dataset | 2-class moth vs 10-class C-class audio |
| Exit depth | 3 exits vs 5 exits |
| Hint | No-hint vs sequential hint passing |
| Evaluation level | per-exit, segment policy, full clip, Depth×Time |
| Efficiency | exit depth, windows saved, compute saved, MFLOPs |

The 8 validated run variants are:

```text
3exit_2class_greedy
3exit_2class_greedy_hint
5exit_2class_greedy
5exit_2class_greedy_hint
3exit_cclass_greedy
3exit_cclass_greedy_hint
5exit_cclass_greedy
5exit_cclass_greedy_hint
```

---

## Common PowerShell variables

Paste these once before running the commands:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$MOTH_LABELS="female,male"

$CCLASS_LABELS="car_crash,conversation,engine_idling,fireworks,gun_shot,rain,road_traffic,scream,thunderstorm,wind"

$CCLASS_CAPS="gun_shot=0,scream=0,car_crash=0,fireworks=8,rain=5,wind=5,road_traffic=5,engine_idling=5,conversation=5,thunderstorm=5"
```

---

## Reproducible PowerShell commands

### 3exit_2class_greedy

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data\moth_sounds" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "3exit_2class_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $MOTH_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 0 `
  -SplitUnit "file" `
  -TapBlocks "1,3" `
  -ExitHint "false" `
  -ForceRebuild `
  -RunClipPolicy
```

### 3exit_2class_greedy_hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data\moth_sounds" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "3exit_2class_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $MOTH_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 0 `
  -SplitUnit "file" `
  -TapBlocks "1,3" `
  -ExitHint "true" `
  -ForceRebuild `
  -RunClipPolicy
```

### 5exit_2class_greedy

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data\moth_sounds" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "5exit_2class_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $MOTH_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 0 `
  -SplitUnit "file" `
  -TapBlocks "1,2,3,4" `
  -ExitHint "false" `
  -ForceRebuild `
  -RunClipPolicy
```

### 5exit_2class_greedy_hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data\moth_sounds" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "5exit_2class_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $MOTH_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 0 `
  -SplitUnit "file" `
  -TapBlocks "1,2,3,4" `
  -ExitHint "true" `
  -ForceRebuild `
  -RunClipPolicy
```

### 3exit_cclass_greedy

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "3exit_cclass_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 5 `
  -MaxSegmentsPerLabelJson $CCLASS_CAPS `
  -SplitUnit "file" `
  -TapBlocks "1,3" `
  -ExitHint "false" `
  -ForceRebuild `
  -RunClipPolicy
```

### 3exit_cclass_greedy_hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "3exit_cclass_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 5 `
  -MaxSegmentsPerLabelJson $CCLASS_CAPS `
  -SplitUnit "file" `
  -TapBlocks "1,3" `
  -ExitHint "true" `
  -ForceRebuild `
  -RunClipPolicy
```

### 5exit_cclass_greedy

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "5exit_cclass_greedy" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 5 `
  -MaxSegmentsPerLabelJson $CCLASS_CAPS `
  -SplitUnit "file" `
  -TapBlocks "1,2,3,4" `
  -ExitHint "false" `
  -ForceRebuild `
  -RunClipPolicy
```

### 5exit_cclass_greedy_hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "5exit_cclass_greedy_hint" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "segment" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 5 `
  -MaxSegmentsPerLabelJson $CCLASS_CAPS `
  -SplitUnit "file" `
  -TapBlocks "1,2,3,4" `
  -ExitHint "true" `
  -ForceRebuild `
  -RunClipPolicy
```


---

# Results

## Table 1 — Dataset and segmentation summary

| Dataset        | Variant                  |   Classes |   Train files |   Val files |   Test files |   Train segs |   Val segs |   Test segs |   Total segs |   Test avg windows/clip |   Test median windows/clip |
|:---------------|:-------------------------|----------:|--------------:|------------:|-------------:|-------------:|-----------:|------------:|-------------:|------------------------:|---------------------------:|
| 2-class moth   | 3exit_2class_greedy      |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 2-class moth   | 3exit_2class_greedy_hint |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 2-class moth   | 5exit_2class_greedy      |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 2-class moth   | 5exit_2class_greedy_hint |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 10-class audio | 3exit_cclass_greedy      |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |
| 10-class audio | 3exit_cclass_greedy_hint |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |
| 10-class audio | 5exit_cclass_greedy      |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |
| 10-class audio | 5exit_cclass_greedy_hint |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |

## Table 2 — Per-exit test accuracy and macro F1

| Dataset        | Variant                  |   Exits | Hint   | Exit 1 acc   | Exit 1 macro F1   | Exit 2 acc   | Exit 2 macro F1   | Exit 3 acc   | Exit 3 macro F1   |
|:---------------|:-------------------------|--------:|:-------|:-------------|:------------------|:-------------|:------------------|:-------------|:------------------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | 84.00%       | 79.44%            | 93.54%       | 91.18%            | 98.46%       | 97.90%            |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | 84.62%       | 80.38%            | 94.46%       | 92.72%            | 99.38%       | 99.14%            |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | 83.38%       | 78.81%            | 85.54%       | 82.36%            | 96.00%       | 94.68%            |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | 83.38%       | 78.81%            | 86.15%       | 82.77%            | 94.77%       | 93.15%            |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | 29.28%       | 21.29%            | 61.51%       | 56.79%            | 69.74%       | 64.74%            |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | 22.70%       | 16.52%            | 53.29%       | 48.89%            | 61.68%       | 57.44%            |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | 21.38%       | 14.90%            | 42.60%       | 35.56%            | 58.39%       | 51.89%            |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | 21.05%       | 15.04%            | 40.13%       | 34.20%            | 56.74%       | 51.71%            |

## Table 3 — Segment-level greedy policy comparison

| Dataset        | Variant                  |   Exits | Hint   | Policy acc   |   Avg exit depth | Flip-any rate   |   Avg flip count | Exit consistency   |   Tau |   N segments |
|:---------------|:-------------------------|--------:|:-------|:-------------|-----------------:|:----------------|-----------------:|:-------------------|------:|-------------:|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | 98.15%       |            1.862 | 17.23%          |            0.194 | 99.69%             |  0.9  |          325 |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | 99.38%       |            1.911 | 16.31%          |            0.172 | 100.00%            |  0.95 |          325 |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | 97.85%       |            2.462 | 20.31%          |            0.252 | 99.69%             |  0.95 |          325 |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | 96.92%       |            2.36  | 19.38%          |            0.243 | 99.69%             |  0.9  |          325 |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | 69.90%       |            2.873 | 71.38%          |            0.895 | 99.34%             |  0.92 |          608 |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | 61.68%       |            2.87  | 84.38%          |            1.086 | 100.00%            |  0.85 |          608 |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | 68.75%       |            4.612 | 81.91%          |            1.352 | 99.67%             |  0.9  |          608 |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | 64.31%       |            4.748 | 83.22%          |            1.339 | 100.00%            |  0.95 |          608 |

## Table 4 — Full-clip vs Depth×Time comparison

| Dataset        | Variant                  |   Exits | Hint   | Clip mode   | Clip acc   | Segment acc used   |   Avg windows used |   Avg windows total | Windows saved %   |   Avg compute units | Compute saved %   |   Avg depth/used window | Flip rate   | Exit consistency   |   N clips |
|:---------------|:-------------------------|--------:|:-------|:------------|:-----------|:-------------------|-------------------:|--------------------:|:------------------|--------------------:|:------------------|------------------------:|:------------|:-------------------|----------:|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Full clip   | 100.00%    | 98.15%             |             14.77  |               14.77 | 0.00%             |              27.5   | 0.00%             |                   1.862 | 17.23%      | 99.69%             |        22 |
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Depth×Time  | 100.00%    | 97.73%             |              2     |               14.77 | 86.46%            |               4.818 | 82.48%            |                   2.409 | 36.36%      | 100.00%            |        22 |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Full clip   | 100.00%    | 99.38%             |             14.77  |               14.77 | 0.00%             |              28.23  | 0.00%             |                   1.911 | 16.31%      | 100.00%            |        22 |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Depth×Time  | 100.00%    | 100.00%            |              2     |               14.77 | 86.46%            |               4.864 | 82.77%            |                   2.432 | 36.36%      | 100.00%            |        22 |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Full clip   | 100.00%    | 97.85%             |             14.77  |               14.77 | 0.00%             |              36.36  | 0.00%             |                   2.462 | 20.31%      | 99.69%             |        22 |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Depth×Time  | 100.00%    | 97.78%             |              2.045 |               14.77 | 86.15%            |               6.455 | 82.25%            |                   3.156 | 40.00%      | 97.78%             |        22 |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Full clip   | 100.00%    | 96.92%             |             14.77  |               14.77 | 0.00%             |              34.86  | 0.00%             |                   2.36  | 19.38%      | 99.69%             |        22 |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Depth×Time  | 100.00%    | 97.87%             |              2.136 |               14.77 | 85.54%            |               7.136 | 79.53%            |                   3.34  | 40.43%      | 100.00%            |        22 |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Full clip   | 81.58%     | 69.90%             |              4     |                4    | 0.00%             |              11.49  | 0.00%             |                   2.873 | 71.38%      | 99.34%             |       152 |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Depth×Time  | 81.58%     | 65.59%             |              2.638 |                4    | 34.05%            |               7.546 | 34.34%            |                   2.86  | 72.07%      | 99.50%             |       152 |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Full clip   | 78.29%     | 61.68%             |              4     |                4    | 0.00%             |              11.48  | 0.00%             |                   2.87  | 84.38%      | 100.00%            |       152 |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Depth×Time  | 78.29%     | 61.20%             |              2.967 |                4    | 25.82%            |               8.467 | 26.25%            |                   2.854 | 82.48%      | 100.00%            |       152 |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Full clip   | 78.95%     | 68.75%             |              4     |                4    | 0.00%             |              18.45  | 0.00%             |                   4.612 | 81.91%      | 99.67%             |       152 |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Depth×Time  | 78.95%     | 65.57%             |              2.809 |                4    | 29.77%            |              12.93  | 29.89%            |                   4.604 | 80.80%      | 99.53%             |       152 |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Full clip   | 73.68%     | 64.31%             |              4     |                4    | 0.00%             |              18.99  | 0.00%             |                   4.748 | 83.22%      | 100.00%            |       152 |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Depth×Time  | 73.03%     | 61.69%             |              2.73  |                4    | 31.74%            |              12.88  | 32.21%            |                   4.716 | 81.69%      | 100.00%            |       152 |

## Table 5 — Segment-policy exit mix

| Dataset        | Variant                  |   Exits | Hint   | e1     | e2     | e3     | e4     | e5     |
|:---------------|:-------------------------|--------:|:-------|:-------|:-------|:-------|:-------|:-------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | 39.38% | 35.08% | 25.54% | 0.00%  | 0.00%  |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | 36.31% | 36.31% | 27.38% | 0.00%  | 0.00%  |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | 36.31% | 6.46%  | 39.38% | 10.46% | 7.38%  |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | 38.46% | 13.85% | 29.23% | 10.15% | 8.31%  |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | 0.00%  | 12.66% | 87.34% | 0.00%  | 0.00%  |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | 0.00%  | 12.99% | 87.01% | 0.00%  | 0.00%  |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | 0.00%  | 2.96%  | 11.18% | 7.57%  | 78.29% |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | 0.00%  | 1.97%  | 8.55%  | 2.14%  | 87.34% |

## Table 6 — Full-clip exit mix

| Dataset        | Variant                  |   Exits | Hint   | Clip mode   | e1     | e2     | e3     | e4     | e5     |
|:---------------|:-------------------------|--------:|:-------|:------------|:-------|:-------|:-------|:-------|:-------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Full clip   | 39.38% | 35.08% | 25.54% | 0.00%  | 0.00%  |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Full clip   | 36.31% | 36.31% | 27.38% | 0.00%  | 0.00%  |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Full clip   | 36.31% | 6.46%  | 39.38% | 10.46% | 7.38%  |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Full clip   | 38.46% | 13.85% | 29.23% | 10.15% | 8.31%  |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Full clip   | 0.00%  | 12.66% | 87.34% | 0.00%  | 0.00%  |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Full clip   | 0.00%  | 12.99% | 87.01% | 0.00%  | 0.00%  |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Full clip   | 0.00%  | 2.96%  | 11.18% | 7.57%  | 78.29% |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Full clip   | 0.00%  | 1.97%  | 8.55%  | 2.14%  | 87.34% |

## Table 7 — Depth×Time exit mix

| Dataset        | Variant                  |   Exits | Hint   | Clip mode   | e1     | e2     | e3     | e4     | e5     |
|:---------------|:-------------------------|--------:|:-------|:------------|:-------|:-------|:-------|:-------|:-------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Depth×Time  | 11.36% | 36.36% | 52.27% | 0.00%  | 0.00%  |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Depth×Time  | 9.09%  | 38.64% | 52.27% | 0.00%  | 0.00%  |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Depth×Time  | 8.89%  | 6.67%  | 53.33% | 22.22% | 8.89%  |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Depth×Time  | 8.51%  | 10.64% | 36.17% | 27.66% | 17.02% |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Depth×Time  | 0.00%  | 13.97% | 86.03% | 0.00%  | 0.00%  |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Depth×Time  | 0.00%  | 14.63% | 85.37% | 0.00%  | 0.00%  |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Depth×Time  | 0.00%  | 4.22%  | 10.30% | 6.32%  | 79.16% |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Depth×Time  | 0.00%  | 2.89%  | 8.43%  | 2.89%  | 85.78% |

## Table 8 — Hint vs no-hint deltas

Positive delta means the hint-enabled run improved the metric. Negative delta means the hint-enabled run reduced it.

| Dataset        |   Exits | Metric                     |   No hint |   Hint |   Delta Hint-No |
|:---------------|--------:|:---------------------------|----------:|-------:|----------------:|
| 2-class moth   |       3 | Segment policy acc         |     0.982 |  0.994 |           0.012 |
| 2-class moth   |       3 | Full-clip acc              |     1     |  1     |           0     |
| 2-class moth   |       3 | Depth×Time clip acc        |     1     |  1     |           0     |
| 2-class moth   |       3 | Depth×Time compute saved % |    82.48  | 82.77  |           0.29  |
| 2-class moth   |       3 | Avg exit depth             |     1.862 |  1.911 |           0.049 |
| 2-class moth   |       5 | Segment policy acc         |     0.978 |  0.969 |          -0.009 |
| 2-class moth   |       5 | Full-clip acc              |     1     |  1     |           0     |
| 2-class moth   |       5 | Depth×Time clip acc        |     1     |  1     |           0     |
| 2-class moth   |       5 | Depth×Time compute saved % |    82.25  | 79.53  |          -2.719 |
| 2-class moth   |       5 | Avg exit depth             |     2.462 |  2.36  |          -0.102 |
| 10-class audio |       3 | Segment policy acc         |     0.699 |  0.617 |          -0.082 |
| 10-class audio |       3 | Full-clip acc              |     0.816 |  0.783 |          -0.033 |
| 10-class audio |       3 | Depth×Time clip acc        |     0.816 |  0.783 |          -0.033 |
| 10-class audio |       3 | Depth×Time compute saved % |    34.34  | 26.25  |          -8.098 |
| 10-class audio |       3 | Avg exit depth             |     2.873 |  2.87  |          -0.003 |
| 10-class audio |       5 | Segment policy acc         |     0.688 |  0.643 |          -0.044 |
| 10-class audio |       5 | Full-clip acc              |     0.789 |  0.737 |          -0.053 |
| 10-class audio |       5 | Depth×Time clip acc        |     0.789 |  0.73  |          -0.059 |
| 10-class audio |       5 | Depth×Time compute saved % |    29.89  | 32.21  |           2.327 |
| 10-class audio |       5 | Avg exit depth             |     4.612 |  4.748 |           0.137 |

## Table 9 — 3-exit vs 5-exit deltas

Positive delta means 5 exits improved the metric relative to 3 exits.

| Dataset        | Hint   | Metric                     |   3 exits |   5 exits |   Delta 5-3 |
|:---------------|:-------|:---------------------------|----------:|----------:|------------:|
| 2-class moth   | No     | Segment policy acc         |     0.982 |     0.978 |      -0.003 |
| 2-class moth   | No     | Full-clip acc              |     1     |     1     |       0     |
| 2-class moth   | No     | Depth×Time clip acc        |     1     |     1     |       0     |
| 2-class moth   | No     | Depth×Time compute saved % |    82.48  |    82.25  |      -0.229 |
| 2-class moth   | No     | Avg exit depth             |     1.862 |     2.462 |       0.6   |
| 2-class moth   | Yes    | Segment policy acc         |     0.994 |     0.969 |      -0.025 |
| 2-class moth   | Yes    | Full-clip acc              |     1     |     1     |       0     |
| 2-class moth   | Yes    | Depth×Time clip acc        |     1     |     1     |       0     |
| 2-class moth   | Yes    | Depth×Time compute saved % |    82.77  |    79.53  |      -3.239 |
| 2-class moth   | Yes    | Avg exit depth             |     1.911 |     2.36  |       0.449 |
| 10-class audio | No     | Segment policy acc         |     0.699 |     0.688 |      -0.012 |
| 10-class audio | No     | Full-clip acc              |     0.816 |     0.789 |      -0.026 |
| 10-class audio | No     | Depth×Time clip acc        |     0.816 |     0.789 |      -0.026 |
| 10-class audio | No     | Depth×Time compute saved % |    34.34  |    29.89  |      -4.459 |
| 10-class audio | No     | Avg exit depth             |     2.873 |     4.612 |       1.738 |
| 10-class audio | Yes    | Segment policy acc         |     0.617 |     0.643 |       0.026 |
| 10-class audio | Yes    | Full-clip acc              |     0.783 |     0.737 |      -0.046 |
| 10-class audio | Yes    | Depth×Time clip acc        |     0.783 |     0.73  |      -0.053 |
| 10-class audio | Yes    | Depth×Time compute saved % |    26.25  |    32.21  |       5.967 |
| 10-class audio | Yes    | Avg exit depth             |     2.87  |     4.748 |       1.878 |

## Table 10 — C-class full-clip per-class F1

| Class         | 3exit_cclass_greedy F1   |   3exit_cclass_greedy support | 3exit_cclass_greedy_hint F1   |   3exit_cclass_greedy_hint support | 5exit_cclass_greedy F1   |   5exit_cclass_greedy support | 5exit_cclass_greedy_hint F1   |   5exit_cclass_greedy_hint support |
|:--------------|:-------------------------|------------------------------:|:------------------------------|-----------------------------------:|:-------------------------|------------------------------:|:------------------------------|-----------------------------------:|
| car_crash     | 70.59%                   |                            14 | 73.33%                        |                                 14 | 69.23%                   |                            14 | 75.86%                        |                                 14 |
| conversation  | 95.65%                   |                            12 | 95.65%                        |                                 12 | 95.65%                   |                            12 | 95.65%                        |                                 12 |
| engine_idling | 82.35%                   |                            10 | 46.15%                        |                                 10 | 46.15%                   |                            10 | 33.33%                        |                                 10 |
| fireworks     | 0.00%                    |                             2 | 0.00%                         |                                  2 | 0.00%                    |                             2 | 0.00%                         |                                  2 |
| gun_shot      | 90.00%                   |                            28 | 91.53%                        |                                 28 | 94.74%                   |                            28 | 90.57%                        |                                 28 |
| rain          | 66.67%                   |                            15 | 58.33%                        |                                 15 | 57.14%                   |                            15 | 60.87%                        |                                 15 |
| road_traffic  | 90.00%                   |                            18 | 84.21%                        |                                 18 | 85.71%                   |                            18 | 66.67%                        |                                 18 |
| scream        | 87.80%                   |                            23 | 85.00%                        |                                 23 | 93.33%                   |                            23 | 93.33%                        |                                 23 |
| thunderstorm  | 40.00%                   |                            15 | 66.67%                        |                                 15 | 51.61%                   |                            15 | 40.00%                        |                                 15 |
| wind          | 100.00%                  |                            15 | 85.71%                        |                                 15 | 89.66%                   |                            15 | 73.17%                        |                                 15 |

## Table 11 — Runtime / profiling summary

| Dataset        | Variant                  |   Exits | Hint   |   Expected MFLOPs |   Full MFLOPs | Compute saving %   | Latency mean ms   | Latency p50 ms   |
|:---------------|:-------------------------|--------:|:-------|------------------:|--------------:|:-------------------|:------------------|:-----------------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     |             20.39 |         51.63 | 60.51%             | —                 | —                |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    |             21.51 |         51.63 | 58.33%             | —                 | —                |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     |             15.68 |         51.63 | 69.63%             | —                 | —                |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    |             15.18 |         51.63 | 70.59%             | —                 | —                |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     |             47.43 |         51.63 | 8.14%              | —                 | —                |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    |             47.32 |         51.63 | 8.35%              | —                 | —                |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     |             45.1  |         51.63 | 12.65%             | —                 | —                |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    |             47.55 |         51.63 | 7.89%              | —                 | —                |

## Table 12 — Threshold calibration summary

| Dataset        | Variant                  |   Exits | Hint   |   Tau | Val macro F1   | Val acc   |
|:---------------|:-------------------------|--------:|:-------|------:|:---------------|:----------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     |  0.9  | 97.51%         | 97.56%    |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    |  0.95 | 98.35%         | 98.37%    |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     |  0.95 | 98.34%         | 98.37%    |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    |  0.9  | 96.67%         | 96.75%    |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     |  0.92 | 67.65%         | 71.38%    |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    |  0.85 | 70.53%         | 71.73%    |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     |  0.9  | 67.59%         | 71.73%    |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    |  0.95 | 66.28%         | 69.96%    |

## Table 13 — Per-class split segment counts

| Dataset        | Variant                  | Class         |   Train |   Val |   Test |   Total |
|:---------------|:-------------------------|:--------------|--------:|------:|-------:|--------:|
| 2-class moth   | 3exit_2class_greedy      | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 3exit_2class_greedy      | male          |     502 |   141 |     76 |     719 |
| 2-class moth   | 3exit_2class_greedy_hint | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 3exit_2class_greedy_hint | male          |     502 |   141 |     76 |     719 |
| 2-class moth   | 5exit_2class_greedy      | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 5exit_2class_greedy      | male          |     502 |   141 |     76 |     719 |
| 2-class moth   | 5exit_2class_greedy_hint | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 5exit_2class_greedy_hint | male          |     502 |   141 |     76 |     719 |
| 10-class audio | 3exit_cclass_greedy      | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 3exit_cclass_greedy      | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 3exit_cclass_greedy      | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 3exit_cclass_greedy      | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 3exit_cclass_greedy      | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 3exit_cclass_greedy      | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 3exit_cclass_greedy      | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 3exit_cclass_greedy      | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 3exit_cclass_greedy      | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 3exit_cclass_greedy      | wind          |     350 |    75 |     75 |     500 |
| 10-class audio | 3exit_cclass_greedy_hint | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 3exit_cclass_greedy_hint | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 3exit_cclass_greedy_hint | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 3exit_cclass_greedy_hint | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 3exit_cclass_greedy_hint | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 3exit_cclass_greedy_hint | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 3exit_cclass_greedy_hint | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 3exit_cclass_greedy_hint | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 3exit_cclass_greedy_hint | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 3exit_cclass_greedy_hint | wind          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy      | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 5exit_cclass_greedy      | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 5exit_cclass_greedy      | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 5exit_cclass_greedy      | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 5exit_cclass_greedy      | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 5exit_cclass_greedy      | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy      | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 5exit_cclass_greedy      | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 5exit_cclass_greedy      | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 5exit_cclass_greedy      | wind          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy_hint | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 5exit_cclass_greedy_hint | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 5exit_cclass_greedy_hint | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 5exit_cclass_greedy_hint | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 5exit_cclass_greedy_hint | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 5exit_cclass_greedy_hint | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy_hint | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 5exit_cclass_greedy_hint | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 5exit_cclass_greedy_hint | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 5exit_cclass_greedy_hint | wind          |     350 |    75 |     75 |     500 |

---

# Interpretation

## 1. The generic pipeline is validated

The 2-class moth results remain excellent:

```text
Full-clip accuracy:      100%
Depth×Time clip accuracy: 100%
```

This confirms that the new generic segmentation, physical 1-second WAV export, and feature extraction flow did **not** damage the original moth pipeline.

## 2. C-class is genuinely harder than moth

The C-class task has:

- 10 classes instead of 2
- very different clip lengths
- similar acoustic classes, for example `rain`, `thunderstorm`, `wind`, `road_traffic`, and `engine_idling`
- weak support for `fireworks`
- short event sounds such as `gun_shot`, `scream`, and `car_crash`
- background/environment sounds that can dominate if not capped

This explains why C-class clip accuracy is much lower than moth clip accuracy.

## 3. Current best C-class model

The best current C-class setting is:

```text
3exit_cclass_greedy
```

It achieves:

```text
Segment policy accuracy: 69.90%
Full-clip accuracy:      81.58%
Depth×Time accuracy:    81.58%
Compute saved:          34.34%
```

This is the most reliable C-class baseline for the next research step.

## 4. Hint passing currently helps only in the compact binary case

For 2-class moth:

```text
3exit no-hint segment policy: 98.15%
3exit hint segment policy:    99.38%
```

But for C-class:

```text
3exit no-hint segment policy: 69.90%
3exit hint segment policy:    61.68%
```

This suggests that the current hint vector may propagate early-exit uncertainty/noise in the harder multiclass setting. The idea remains valuable, but the current implementation needs improvement for C-class audio.

## 5. 5 exits do not currently help the C-class task

For C-class no-hint:

```text
3exit segment policy: 69.90%
5exit segment policy: 68.75%
```

The 5-exit C-class model mostly exits at the final exit:

```text
5exit_cclass_greedy segment exit mix:
e1=0.00%, e2=2.96%, e3=11.18%, e4=7.57%, e5=78.29%
```

This means the extra exits do not yet provide strong confident early predictions on the 10-class dataset.

## 6. Depth×Time is still useful

Depth×Time reduces windows and compute while keeping clip-level predictions stable. For the best C-class model:

```text
3exit_cclass_greedy:
Full-clip accuracy:   81.58%
Depth×Time accuracy: 81.58%
Compute saved:       34.34%
```

This is important for TinyML-style deployment.

---

# Limitations

Current limitations:

- C-class `fireworks` has very small test support and F1 remains 0.00%.
- C-class hint passing is not yet beneficial.
- C-class 5-exit models mainly defer to the final exit.
- ROC/AUC reporting currently works only for binary tasks.
- Some scripts still show a config warning if `num_classes: 2` remains in the YAML.
- The current class balancing uses fixed per-file/per-label caps; it may need a more principled sampler or epoch-level class balancing.

---

# Future scope

Recommended next steps:

1. **Improve C-class data balance**
   - Add a `WeightedRandomSampler`.
   - Add source-file balancing so long clips cannot dominate.
   - Report macro-F1 as the main C-class metric, not only accuracy.

2. **Improve hint passing for multiclass**
   - Reduce hint strength for early uncertain exits.
   - Try detached logits vs detached probabilities.
   - Add entropy/margin gating before passing hints.
   - Try per-class adaptive hint temperature.

3. **Improve 5-exit C-class usefulness**
   - Train early exits with stronger auxiliary supervision.
   - Add exit-specific loss weighting for C-class.
   - Consider KD from final exit to early exits.
   - Tune greedy threshold separately for C-class.

4. **Improve C-class audio preprocessing**
   - Revisit bandpass `[100,3000]`, because some C-class sounds may have useful energy outside this range.
   - Compare 1s, 2s, and 3s windows.
   - Add augmentation for rare classes such as `fireworks`.

5. **Improve reporting**
   - Add multiclass ROC/AUC or one-vs-rest AUC.
   - Add macro-F1 and weighted-F1 to summary tables.
   - Add confusion-matrix heatmaps for C-class to the paper.

---

# Recommended branch-level conclusion

This branch should be described as:

> A reusable generic K-exit / C-class greedy audio early-exit branch with physical segment-WAV export, dynamic class handling, dynamic exit handling, and CLI-controlled sequential hint passing. The controlled 8-run comparison validates the pipeline on the original moth binary dataset and extends it to a harder 10-class audio dataset. The best current C-class baseline is `3exit_cclass_greedy`, while hint passing is currently beneficial only for the compact 3-exit moth setting and remains future work for the 10-class setting.
