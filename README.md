# ASHADIP / NeuroAccuExit — `kexit-greedy-gunshot-segment`

This branch documents the **generic audio-segmentation and multiclass adaptation** of the K-exit greedy pipeline.

It extends the earlier work in two important ways:

- **older branch/docs**: moth wingbeat (`male` vs `female`) and later gunshot binary adaptation
- **this branch**: **generic mixed-length audio preprocessing** plus a **10-class acoustic classification study**

The goal of this branch is to keep the **same K-exit architecture** while making the data pipeline reusable for raw audio collections that:

- contain **multiple classes**
- contain **mixed file lengths**
- contain **multiple input formats**
- may need **physical exported segment WAVs**
- may already be **ready-clipped** and therefore should skip re-segmentation

This branch is now the correct place to document the **generic segmentation pipeline** and the current **multiclass greedy baseline with segment- and clip-level evaluation**.

---

## Current task

### Multiclass acoustic classification

The current validated run uses these **10 classes**:

- `car_crash`
- `conversation`
- `engine_idling`
- `fireworks`
- `gun_shot`
- `rain`
- `road_traffic`
- `scream`
- `thunderstorm`
- `wind`

### Dataset root used in the current validated run

```text
data2/
  car_crash/
  conversation/
  engine_idling/
  fireworks/
  gun_shot/
  rain/
  road_traffic/
  scream/
  thunderstorm/
  wind/
```

---

## Why this branch exists

The earlier K-exit branch generalized the **model side** well, but the preprocessing path was still too tied to older task assumptions.

This branch adds the missing practical pieces required for a reusable acoustic pipeline:

- generic label discovery from subfolders
- mixed-length audio support
- preprocessing for `.wav`, `.flac`, and other supported formats
- inventory summary before segmentation
- file-level train/val/test splitting
- support for short clips
- optional export of physical segment WAV files
- support for already-ready datasets
- shorter processed filenames for Windows safety
- multiclass runs without manually redesigning the architecture

---

## Main implementation changes in this branch

### 1. `scripts/prep_segments.py`

Updated so that it now supports a more generic preprocessing flow:

- auto-discovers class folders under `--root`
- reads mixed audio formats
- writes `audio_inventory.csv`
- writes `audio_inventory_by_label.csv`
- supports `--input_mode segment|ready`
- supports `--min_keep_sec`
- supports generic file-level caps through `MaxSegmentsPerFileDefault`
- supports optional exported 1-second segment WAVs
- performs file-level train/val/test splitting
- handles short clips through padded 1-second segments when appropriate
- supports reusable `segments.csv` generation for later feature extraction and training

### 2. `scripts/run_full.ps1`

Updated so that it now supports generic data preparation directly through CLI:

- `-DataRoot "data2"`
- `-InputMode "segment"` or `"ready"`
- `-SplitUnit`
- `-GroupMode`
- `-MinKeepSec`
- `-MaxSegmentsPerFileDefault`
- `-ExportSegmentWavs`
- `-ForceRebuild`
- `-RunClipPolicy`
- run metadata logging for the new preprocessing controls

### 3. `scripts/extract_features.py`

Updated so that feature extraction is more robust for the new pipeline:

- uses padded short-feature extraction when needed
- writes unique feature paths back into `segments.csv`
- uses shorter per-segment feature names to avoid path and overwrite issues

### 4. Dynamic class handling in training/evaluation

The current branch now tolerates class-count mismatch between the old config and the current dataset:

- if `config num_classes` does not match the dataset,
- training and evaluation automatically use the dataset class count.

This allows the same architecture to be reused without manually changing every old config before testing a new dataset.

---

## Current end-to-end pipeline

1. raw audio under `data2/<class_name>/`
2. `scripts/prep_segments.py`
   - scan inventory
   - create cleaned cache WAVs
   - create `segments.csv`
   - apply file-level split
   - optionally export physical 1-second WAV segments
3. `scripts.extract_features.py`
   - compute log-mel `.npy` features
   - update `feat_relpath`
4. `data/datasets.py`
   - build PyTorch datasets and loaders
5. `adapters/audio_adapter.py`
   - dynamic `TinyAudioCNN` with configurable tap blocks
6. `models/exit_net.py`
   - generic multi-exit classifier
7. `training/train.py`
   - train the K-exit model
8. `training/calibrate.py`
   - fit per-exit temperatures
9. `training/thresholds_offline.py`
   - greedy threshold selection
10. `scripts/policy_test.py`
   - segment-level greedy policy
11. `scripts/clip_policy_test.py`
   - full-clip and Depth×Time clip evaluation
12. `scripts/summarize_run.py`
13. `scripts/analyse_run.py`
14. `scripts/profile_latency.py`
15. `scripts/run_reports.ps1`

---

## Current validated dataset summary

The current validated multiclass run used the following inventory:

- **1011 total files**
- **10 classes**

### Inventory summary

| Class | Files | Min sec | Median sec | Mean sec | Max sec |
|---|---:|---:|---:|---:|---:|
| car_crash | 92 | 1.0000 | 2.3064 | 2.9281 | 10.8639 |
| conversation | 81 | 1.4800 | 3.0000 | 14.6656 | 994.3688 |
| engine_idling | 65 | 1.0376 | 8.0000 | 11.3280 | 36.0000 |
| fireworks | 14 | 1.3095 | 21.0000 | 108.0361 | 770.2427 |
| gun_shot | 187 | 0.4988 | 1.5084 | 1.9196 | 44.0000 |
| rain | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |
| road_traffic | 121 | 3.9998 | 4.9999 | 5.4463 | 60.0000 |
| scream | 151 | 0.5151 | 1.5020 | 1.7642 | 6.5912 |
| thunderstorm | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |
| wind | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |

### Segmentation setting used in the validated run

- `segment_sec = 1.0`
- `hop = 0.5`
- `input_mode = segment`
- `split_unit = file`
- `min_keep_sec = 0.25`
- `max_segments_per_file_default = 5`
- `export_segment_wavs = False`
- `RunClipPolicy = True`

### File-level split

| Split | Files |
|---|---:|
| train | 707 |
| val | 152 |
| test | 152 |

### Segment-level split

| Split | Segments |
|---|---:|
| train | 2536 |
| val | 551 |
| test | 555 |

### Important preprocessing finding

The current run rejected:

- **6249** `cap_dropped`
- **1786** `silent_window`

This is a major result for this branch:

- the cap prevented segment explosion,
- but it also discarded a very large number of valid non-silent segments.

That means the branch is now **pipeline-stable**, but the **sampling/capping strategy still needs refinement**.

---

## Current validated run command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1 `
  -DataRoot "data2" `
  -CacheRoot "data_caches" `
  -Config "configs\audio_moth.yaml" `
  -RunsRoot "runs" `
  -Variant "kexit_greedy_gunshot_segment" `
  -Policy "greedy" `
  -Device "cpu" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -NMels 64 `
  -TapBlocks "1,2,3,4" `
  -InputMode "segment" `
  -SplitUnit "file" `
  -GroupMode "none" `
  -MinKeepSec 0.25 `
  -MaxSegmentsPerFileDefault 5 `
  -TimeConf 0.95 `
  -TimeStableK 2 `
  -TimeMinWindows 2 `
  -EvalFixedKWindows 3 `
  -TimeMargin 0.0 `
  -ForceRebuild `
  -RunClipPolicy
```

---

## Main artifacts per run

Each successful run still generates the standard artifact set:

- `ckpt/best.pt`
- `metrics.json`
- `temperature.json`
- `thresholds.json`
- `policy_results.json`
- `clip_policy_results.json`
- `clip_policy_results_full.json`
- `clip_policy_results_time.json`
- `summary.json`
- `report.json`
- `analysis_run.json`
- `profiling.json`
- `meta.json`
- plots under `plots/`

---

## Main results from the current validated multiclass run (`kexit_greedy_gunshot_segment_002`)

### 1) Per-exit test accuracy

| Metric | Value |
|---|---:|
| Exit1 accuracy | 0.2613 |
| Exit2 accuracy | 0.4162 |
| Exit3 accuracy | 0.5946 |
| Exit4 accuracy | 0.7027 |
| Exit5 accuracy | 0.6739 |

### 2) Segment-level greedy policy

| Metric | Value |
|---|---:|
| Policy accuracy | 0.6739 |
| Avg exit depth | 4.589 |
| Flip-any rate | 0.8198 |
| Avg flip count | 1.2793 |
| Exit consistency | 1.0000 |

### 3) Segment policy exit mix

- `e1 = 0.0000`
- `e2 = 0.0649`
- `e3 = 0.0721`
- `e4 = 0.0721`
- `e5 = 0.7910`

### 4) Greedy threshold selection

| Metric | Value |
|---|---:|
| Best tau | 0.92 |
| Validation macro-F1 | 0.6708 |
| Validation accuracy | 0.7169 |

### 5) Full-Clip Sequential Greedy

| Metric | Value |
|---|---:|
| Clip accuracy | 0.7829 |
| Segment acc over processed windows | 0.6739 |
| Avg windows used | 3.651 / 3.651 |
| Windows saved | 0.00% |
| Avg compute units | 16.757 |
| Compute saved | 0.00% |
| Avg depth per used window | 4.589 |
| Flip-rate (used windows) | 0.8198 |
| Exit-consistency (taken==final) | 1.0000 |

### 6) Depth×Time Clip Greedy

| Metric | Value |
|---|---:|
| Clip accuracy | 0.7829 |
| Segment acc over used windows | 0.6209 |
| Avg windows used | 2.638 / 3.651 |
| Windows saved | 27.75% |
| Avg compute units | 11.993 |
| Compute saved | 28.43% |
| Avg depth per used window | 4.546 |
| Flip-rate (used windows) | 0.8254 |
| Exit-consistency (taken==final) | 1.0000 |

### 7) Rare-class note

The clip-level per-class report shows that `fireworks` remains the weakest class in the current validated run:

- precision = **0.0**
- recall = **0.0**
- f1 = **0.0**
- support = **2**

This is consistent with the extremely small number of source files and the current blunt global cap.

---

## Short interpretation of the current validated run

The current validated run gives a clear picture.

### Main observations

1. **The engineering pipeline is now stable.**
   - preprocessing works
   - feature extraction works
   - training, calibration, policy test, clip-policy test, analysis, profiling, and report generation all complete end to end

2. **The current challenge is no longer pipeline failure.**
   The main challenge is now:
   - class imbalance
   - segment imbalance caused by very long source files
   - aggressive `cap_dropped` loss
   - weaker multiclass model quality than the earlier simpler tasks

3. **Exit4 is stronger than Exit5 in the current run.**
   - Exit4 test accuracy: **0.7027**
   - Exit5 test accuracy: **0.6739**

   This means the deepest exit is not currently the strongest one.

4. **Clip-policy results are now validated for this branch.**
   - Full-Clip Sequential Greedy clip accuracy: **0.7829**
   - Depth×Time Clip Greedy clip accuracy: **0.7829**
   - Depth×Time saves **27.75% windows** and **28.43% compute** relative to the full-clip baseline, with no drop in clip accuracy.

5. **Dynamic class handling already works in practice.**
   The run started from an older config with `num_classes=2`, but the code detected that the dataset has `10` classes and used the dataset value automatically.

### Main policy takeaway

For this current 10-class task, Depth×Time is already useful:

- it preserves the same clip accuracy as the full-clip baseline,
- while reducing both windows used and compute.

That is a meaningful positive result even though the overall multiclass accuracy is still moderate.

---

## Current research takeaway

This branch should now be described as:

> a generic mixed-length acoustic preprocessing and K-exit greedy pipeline that successfully transfers the existing early-exit architecture to a new 10-class audio setting, and already demonstrates useful clip-level compute savings through Depth×Time without reducing clip accuracy, while showing that the remaining bottlenecks are now class imbalance, capping strategy, and multiclass decision quality rather than engineering instability.

This is an important transition point:

- **before**: the branch was blocked by preprocessing and feature-extraction problems
- **now**: the branch is stable enough for controlled experimental improvement

---

## Current limitations

The current validated run also makes the next problems very clear:

1. **Class imbalance remains severe**
   - `fireworks` is especially underrepresented
   - the per-class clip-level report confirms it is currently the weakest class

2. **A global hard cap is too blunt**
   - the current `max_segments_per_file_default = 5` avoided explosion
   - but also discarded many potentially useful segments

3. **The config still contains older binary assumptions**
   - the runtime auto-corrects class count,
   - but the documentation and defaults should now catch up to the current generic multiclass branch

4. **Clip-policy savings are currently moderate rather than dramatic**
   - the 10-class problem is harder than the earlier simpler tasks
   - current routing remains heavily biased toward the deepest exit

---

## Best next steps

The strongest next follow-up experiments are:

### A. Improve class balancing
Move toward:

- keeping all valid non-silent segments in `segments.csv`
- class-aware sampling during training
- and/or per-class file-level caps instead of a single blunt global cap

### B. Reduce dependence on hard `cap_dropped` removal
The current branch should move toward:

- preserving more valid segments,
- then balancing or sub-sampling at training time rather than permanently discarding them during preprocessing

### C. Make class count fully automatic in the documented defaults
The runtime already corrects this dynamically.  
The documentation and default config logic should now reflect that design explicitly.

### D. Improve rare-class handling
The current branch should eventually support:

- per-class segment caps
- better balancing for rare classes like `fireworks`
- less destructive handling of long but informative source files

---

## Branch-level conclusion

The current recommendation for this branch is:

- keep the generic preprocessing and segmentation design
- keep file-level splitting
- keep clip-policy evaluation
- keep dynamic K-exit training
- improve class balancing and capping strategy next
- treat the current run as the first stable multiclass greedy baseline **with validated clip-policy results** for this branch
