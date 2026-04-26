# ASHADIP / NeuroAccuExit — Documentation Structure for Generic K-Exit / C-Class Greedy + Hint Pipeline

This document defines the recommended documentation / thesis mini-book structure for the current `kexit-greedy-hint` research branch.

The current branch has moved beyond the older moth-only setting. It now supports:

- dynamic **K-exit** inference
- dynamic **C-class** classification
- generic audio segmentation
- physical 1-second segment-WAV export
- greedy no-hint and greedy sequential hint-passing variants
- 2-class moth validation
- 10-class audio validation

---

## Recommended document map

| File | Purpose |
|---|---|
| `README.md` | Main repository-facing documentation, commands, main tables, and conclusions |
| `DOC_STRUCTURE.md` | Long-form thesis/report structure and writing plan |
| `APPENDIX.md` | Extended tables, per-class results, split counts, threshold details, and diagnostic notes |
| `ASHADIP_8_run_comparison_tables.xlsx` | Spreadsheet copy of all parsed run tables |

---

# Chapter 1 — Introduction and motivation

## Purpose

Explain why this branch exists.

## Key points to include

- The original pipeline was designed around a 3-exit moth binary classifier.
- The new research direction required a generic audio pipeline that can handle:
  - 2 classes or N classes
  - 3 exits or 5 exits
  - hint or no-hint inference
  - full-clip and Depth×Time policies
- The immediate research question became:

> Are the weak C-class results caused by a broken modified pipeline, or by the harder multiclass dataset?

The controlled 8-run experiment answers this clearly: the pipeline is not broken because moth performance remains strong. The C-class task is genuinely harder.

---

# Chapter 2 — Dataset design

## 2.1 Moth 2-class dataset

Classes:

```text
female
male
```

Purpose:

- pipeline sanity check
- direct comparison against previous moth results
- proof that generic segmentation/export did not damage performance

## 2.2 C-class 10-class dataset

Classes:

```text
car_crash
conversation
engine_idling
fireworks
gun_shot
rain
road_traffic
scream
thunderstorm
wind
```

Purpose:

- harder generic audio classification test
- multiclass validation of the K-exit/C-class refactor
- realistic test of hint passing under class ambiguity

## 2.3 Dataset summary table

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

---

# Chapter 3 — Generic preprocessing and segmentation

## 3.1 Recommended flow

Use the following flow for reproducible research:

```text
raw parent audio
→ clean/resample/bandpass parent audio
→ split by parent file or group
→ create logical 1-second segment rows
→ export physical 1-second segment WAVs
→ extract log-mel features from exported segment WAVs
```

## 3.2 Why this flow is preferred

This flow preserves traceability:

```text
feature .npy → segment WAV → cleaned parent WAV → original parent file
```

It also prevents segment-level train/test leakage because the split is assigned at the parent file or group level before model evaluation.

## 3.3 Modes to document

| Mode | Purpose |
|---|---|
| `segment` | Raw long/short audio → cleaned audio → segment rows → physical segment WAVs → log-mel features |
| `ready` | Already segmented clips → build manifest/splits → extract features |
| `export_wavs` | Save physical 1-second WAV segments for reuse, auditing, and publication |

## 3.4 Important CLI controls

| Argument | Purpose |
|---|---|
| `-InputMode "segment"` | Segment raw parent audio |
| `-SplitUnit "file"` | Split by parent file to reduce leakage |
| `-MinKeepSec 0.25` | Keep very short event clips if at least 0.25s |
| `-MaxSegmentsPerFileDefault` | Limit per-file segment contribution |
| `-MaxSegmentsPerLabelJson` | Per-class cap using key=value format |
| `-ForceRebuild` | Rebuild cache cleanly |
| `-RunClipPolicy` | Run full-clip and Depth×Time evaluation |

---

# Chapter 4 — Model architecture

## 4.1 Dynamic K-exit model

The model derives:

```text
num_exits = len(tap_blocks) + 1
```

Validated configurations:

| TapBlocks | Exits |
|---|---:|
| `1,3` | 3 |
| `1,2,3,4` | 5 |

## 4.2 Dynamic C-class output

The classifier should infer the number of classes from the dataset / `segments.csv`.

Avoid hardcoding:

```yaml
model:
  num_classes: 2
```

For generic C-class runs, this should be removed or commented out.

## 4.3 Sequential hint passing

Hint passing is controlled from the CLI:

```powershell
-ExitHint "true"
```

or:

```powershell
-ExitHint "false"
```

The current evidence shows:

- useful in compact 3-exit moth binary runs
- harmful or unstable in current C-class runs
- not yet beneficial for 5-exit C-class

---

# Chapter 5 — Training, calibration, and policy selection

## 5.1 Multi-exit training

Training uses dynamic loss weights:

| Exit setting | Dynamic loss weights |
|---|---|
| 3 exits | `[0.3, 0.3, 1.0]` |
| 5 exits | `[0.3, 0.3, 0.6, 0.8, 1.0]` |

## 5.2 Calibration

Each exit receives a fitted temperature in `temperature.json`.

## 5.3 Greedy threshold selection

The selected threshold `tau` is stored in `thresholds.json`.

Threshold summary:

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

---

# Chapter 6 — Evaluation protocol

Use three evaluation levels.

## 6.1 Per-exit test evaluation

Report:

- exit accuracy
- exit macro-F1
- final-exit quality

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

## 6.2 Segment-level greedy policy

Report:

- policy accuracy
- average exit depth
- exit mix
- flip-any rate
- average flip count
- exit consistency

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

## 6.3 Clip-level evaluation

Report both:

- **Full-clip baseline**
- **Depth×Time early stopping**

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

---

# Chapter 7 — Reproducibility commands

## Common variables

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$MOTH_LABELS="female,male"

$CCLASS_LABELS="car_crash,conversation,engine_idling,fireworks,gun_shot,rain,road_traffic,scream,thunderstorm,wind"

$CCLASS_CAPS="gun_shot=0,scream=0,car_crash=0,fireworks=8,rain=5,wind=5,road_traffic=5,engine_idling=5,conversation=5,thunderstorm=5"
```

## 8-run command set

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

# Chapter 8 — Main findings

## 8.1 Moth result

The moth binary task remains strong:

```text
All 2-class variants reached 100% full-clip accuracy.
All 2-class variants reached 100% Depth×Time clip accuracy.
```

This validates the new generic preprocessing/export pipeline.

## 8.2 C-class result

The best C-class setting is:

```text
3exit_cclass_greedy
```

with:

```text
Segment policy accuracy: 69.90%
Full-clip accuracy:      81.58%
Depth×Time accuracy:    81.58%
Compute saved:          34.34%
```

## 8.3 Hint effect

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

Interpretation:

- Hint helps `3exit_2class_greedy_hint`.
- Hint hurts both 3-exit and 5-exit C-class runs.
- The current C-class hint design needs further work.

## 8.4 3-exit vs 5-exit effect

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

Interpretation:

- 5 exits do not currently improve C-class.
- C-class 5-exit models mostly defer to later exits.
- The extra depth increases compute without a clear accuracy benefit.

---

# Chapter 9 — Exit-routing behaviour

## 9.1 Segment-policy exit mix

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

## 9.2 Full-clip exit mix

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

## 9.3 Depth×Time exit mix

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

Interpretation:

- Moth runs use early exits meaningfully.
- C-class 3-exit models mostly use exit 3.
- C-class 5-exit models mostly use exit 5.
- The early exits are not yet confident enough for difficult C-class prediction.

---

# Chapter 10 — C-class per-class behaviour

## 10.1 Full-clip per-class F1

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

## 10.2 Interpretation

Strong classes:

- `conversation`
- `gun_shot`
- `scream`
- `wind` in the 3-exit no-hint run

Weak classes:

- `fireworks`
- `engine_idling`
- `thunderstorm`
- `rain` in some variants

Reasonable explanation:

- `fireworks` has very low support.
- Rain/thunderstorm/wind share acoustic structure.
- Road traffic and engine idling can overlap.
- Event classes such as gunshot/scream are easier when clearly represented.

---

# Chapter 11 — Profiling and deployment view

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

Interpretation:

- Moth models save substantially more expected MFLOPs because early exits are more confident.
- C-class models save less expected MFLOPs because they defer to deeper exits.
- This reinforces that the C-class problem is harder and needs better early-exit training.

---

# Chapter 12 — Limitations

Document the following limitations clearly:

1. C-class is not yet fully optimized.
2. Current hint passing is not robust for 10-class audio.
3. 5-exit C-class mostly exits late.
4. `fireworks` has weak support and poor F1.
5. ROC/AUC is currently binary-only.
6. Fixed segment caps are useful but not enough; sampler-level balancing is still needed.
7. The bandpass setting `[100,3000]` may be moth-oriented and may not be ideal for every C-class sound.

---

# Chapter 13 — Future scope

Recommended future research directions:

## 13.1 Data balancing

- Add `WeightedRandomSampler`.
- Add source-file balancing.
- Report class-balanced macro-F1.
- Avoid letting long clips dominate training.

## 13.2 C-class preprocessing

- Try wider bandpass or no bandpass.
- Compare 1s vs 2s vs 3s windows.
- Add augmentation for weak classes.
- Revisit caps for `fireworks`.

## 13.3 Hint passing

- Add confidence-gated hints.
- Pass hints only when margin is high.
- Compare probability hints vs logit hints.
- Try entropy/margin summary statistics.
- Reduce hint dimension or regularize hint projection.

## 13.4 Early-exit training

- Tune loss weights for C-class.
- Try KD from final exit to early exits.
- Try class-balanced auxiliary exit losses.
- Add calibration-aware early-exit training.

## 13.5 Evaluation

- Add multiclass ROC/AUC.
- Add macro-F1 and weighted-F1 to all summary scripts.
- Add per-class confusion analysis.
- Add repeated-seed evaluation for robustness.

---

# Chapter 14 — Recommended final conclusion

Use the following wording in the thesis/report:

> The generic K-exit/C-class audio early-exit pipeline is validated because it preserves the strong moth binary results while extending the system to a harder 10-class audio dataset. The main C-class baseline is currently `3exit_cclass_greedy`, which achieves the best clip-level and segment-level trade-off among the tested C-class variants. Sequential hint passing remains promising, as it improves compact 3-exit moth performance, but the current hint formulation does not yet generalize to the harder 10-class dataset. Future work should focus on class-balanced training, stronger early-exit supervision, and confidence-gated hint passing for multiclass audio.
