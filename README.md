# ASHADIP / NeuroAccuExit — `kexit_cclass_greedy_multi-label_v0` Baseline

This documentation update records the first **multi-label audio classification baseline** created on the local branch:

```text
kexit_cclass_greedy_multi-label
```

The baseline should be saved as:

```text
kexit_cclass_greedy_multi-label_v0
```

This branch is separate from the earlier single-label C-class branch. The goal here is not to replace the multi-class experiments, but to create a controlled future-work extension where one audio segment can contain **more than one sound label**.

---

## What this branch adds

This branch implements the first controlled multi-label pipeline:

```text
clean single-label seed data
→ synthetic multi-label mixtures
→ multi-hot manifest
→ log-mel feature extraction
→ multi-label PyTorch dataset loader
→ 5-exit no-hint multi-label training baseline
```

The key conceptual change is:

| Setting | Single-label / multi-class | Multi-label |
|---|---|---|
| Target format | One integer class ID | Multi-hot vector |
| Example | `rain_thunderstorm` | `rain=1`, `thunderstorm=1` |
| Output activation | Softmax | Sigmoid |
| Loss | CrossEntropyLoss | BCEWithLogitsLoss |
| Prediction | One class only | Any number of labels |
| Main metrics | Accuracy, macro-F1 | Micro-F1, macro-F1, samples-F1, exact match, hamming loss, per-label F1 |

---

## Why this branch is needed

In real environmental audio, a one-second clip can contain more than one event:

```text
rain + thunderstorm
traffic + gunshot
fireworks + crowd/background
wind + rain
```

The previous C-class setup forced mixed audio into one class such as:

```text
rain_thunderstorm
```

The multi-label setup instead represents the same sound as:

```text
rain = 1
thunderstorm = 1
```

This is more realistic for overlapping acoustic events.

---

## Dataset design used in v0

The branch uses a controlled first experiment, not noisy full-dataset weak supervision.

### Source structure

The clean seed data is stored as class folders:

```text
multilabel_data/
└─ clean_seed/
   ├─ car_crash/
   ├─ conversation/
   ├─ engine_idling/
   ├─ fireworks/
   ├─ gun_shot/
   ├─ rain/
   ├─ road_traffic/
   ├─ scream/
   ├─ thunderstorm/
   └─ wind/
```

Each file was renamed in-place using class-prefixed names, for example:

```text
car_crash_0001.wav
conversation_0001.wav
fireworks_0001.flac
```

The `.m4a` files were excluded because the local environment could not decode them through `soundfile` or `librosa/audioread`.

---

## Table ML-v0.1 — Labels used

| ID | Label |
|---:|---|
| 0 | `car_crash` |
| 1 | `conversation` |
| 2 | `engine_idling` |
| 3 | `fireworks` |
| 4 | `gun_shot` |
| 5 | `rain` |
| 6 | `road_traffic` |
| 7 | `scream` |
| 8 | `thunderstorm` |
| 9 | `wind` |

---

## Table ML-v0.2 — Clean seed availability after excluding `.m4a`

| Split | Total | car_crash | conversation | engine_idling | fireworks | gun_shot | rain | road_traffic | scream | thunderstorm | wind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 724 | 64 | 57 | 32 | 57 | 155 | 66 | 80 | 106 | 38 | 69 |
| Val | 155 | 14 | 12 | 7 | 12 | 33 | 14 | 17 | 23 | 8 | 15 |
| Test | 156 | 14 | 12 | 7 | 13 | 33 | 15 | 17 | 22 | 8 | 15 |

---

## Synthetic mixture generation

Synthetic mixtures were created from the clean seed manifest using exactly two labels per synthetic sample:

```text
mix_size_min = 2
mix_size_max = 2
sample_rate  = 16000
clip_sec     = 1.0
seed         = 42
gain range   = -6 dB to 0 dB
```

The synthetic split was leakage-safe:

```text
synthetic train mixtures use only clean_seed train files
synthetic val mixtures use only clean_seed val files
synthetic test mixtures use only clean_seed test files
```

---

## Table ML-v0.3 — Synthetic mixture counts

| Split | Synthetic mixtures |
|---|---:|
| Train | 1000 |
| Val | 200 |
| Test | 200 |
| **Total** | **1400** |

---

## Table ML-v0.4 — Synthetic positive label counts

| Label | Positive count |
|---|---:|
| car_crash | 276 |
| conversation | 293 |
| engine_idling | 301 |
| fireworks | 256 |
| gun_shot | 248 |
| rain | 284 |
| road_traffic | 288 |
| scream | 289 |
| thunderstorm | 267 |
| wind | 298 |

The counts are reasonably balanced for a first baseline.

---

## Table ML-v0.5 — Combined clean + synthetic manifest

| Split | Rows |
|---|---:|
| Train | 1724 |
| Val | 355 |
| Test | 356 |
| **Total** | **2435** |

The combined manifest was saved as:

```text
multilabel_data/metadata/multilabel_train_manifest.csv
```

The extracted feature manifest was saved as:

```text
multilabel_cache/metadata/multilabel_features_manifest.csv
```

---

## Feature extraction settings

Log-mel feature extraction used:

| Setting | Value |
|---|---:|
| Sample rate | 16000 Hz |
| Clip length | 1.0 s |
| Mel bins | 64 |
| FFT size | 1024 |
| Window length | 25 ms |
| Hop length | 10 ms |
| CMVN | Enabled |
| Feature shape | `[1, 64, 101]` after dataset loading |

---

## Table ML-v0.6 — Final extracted feature dataset

| Item | Value |
|---|---:|
| Total rows | 2435 |
| Clean rows | 1035 |
| Synthetic rows | 1400 |
| Train rows | 1724 |
| Val rows | 355 |
| Test rows | 356 |
| Input tensor shape | `[batch, 1, 64, 101]` |
| Target tensor shape | `[batch, 10]` |

---

## Table ML-v0.7 — Final label-positive counts after feature extraction

| Label | Positive count |
|---|---:|
| car_crash | 368 |
| conversation | 374 |
| engine_idling | 347 |
| fireworks | 338 |
| gun_shot | 469 |
| rain | 379 |
| road_traffic | 402 |
| scream | 440 |
| thunderstorm | 321 |
| wind | 397 |

---

## Training baseline recorded in v0

The first completed baseline was:

```text
multilabel_5exit_nohint
```

with:

```text
tap_blocks = 1,2,3,4
num_exits  = 5
exit_hint  = disabled
epochs     = 40
batch_size = 64
lr         = 0.001
threshold  = 0.5
device     = cpu
loss       = BCEWithLogitsLoss
```

The best validation checkpoint was selected using final-exit macro-F1.

```text
Best epoch: 37
Best validation final-exit macro-F1: 0.5438
```

---

## Table ML-v0.8 — Test metrics by exit for `multilabel_5exit_nohint`

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0068 | 0.0072 | — | 0.0056 | 0.1556 | Too weak; earliest exit is not useful yet |
| 2 | 0.2411 | 0.3476 | — | 0.1601 | 0.1287 | Starts learning useful labels |
| 3 | 0.3708 | 0.4660 | — | 0.1994 | 0.1191 | Moderate intermediate representation |
| 4 | 0.5135 | 0.5686 | — | 0.2697 | 0.1104 | Strong early-exit candidate |
| 5 | **0.5302** | **0.5852** | — | **0.3062** | **0.1067** | Best final prediction |

`Samples-F1` was printed by the script, but the exact value was not available in the copied summary used for this documentation update. It remains available in `runs_multilabel/<run_id>/metrics.json`.

---

## Table ML-v0.9 — Final-exit per-label test metrics

| Label | Precision | Recall | F1 | Support | Predicted positives | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| car_crash | 0.6316 | 0.5106 | 0.5647 | 47 | — | Moderate |
| conversation | 0.9815 | 0.8833 | **0.9298** | 60 | 54 | Excellent |
| engine_idling | 0.7500 | 0.3673 | 0.4932 | — | — | Precise but misses many positives |
| fireworks | 0.8421 | 0.3077 | 0.4507 | — | — | High precision, low recall |
| gun_shot | 0.7458 | 0.5789 | 0.6519 | — | — | Good/moderate |
| rain | 0.8800 | 0.4231 | 0.5714 | — | — | Precise but low recall |
| road_traffic | 0.5000 | 0.1429 | 0.2222 | — | — | Weak |
| scream | 0.7971 | 0.9649 | **0.8730** | — | — | Very good |
| thunderstorm | 0.0000 | 0.0000 | 0.0000 | 49 | 6 | Failed in this baseline |
| wind | 0.5625 | 0.5294 | 0.5455 | — | — | Moderate |

`—` means the exact support or predicted-positive value was not available in the copied summary used for this documentation update. The full values should be read from `metrics.json` before final publication.

---

## Research findings from v0

### 1. Multi-label pipeline is functional

The v0 branch successfully builds a full multi-label path from clean seed audio to training-ready features and a working multi-label model.

### 2. Deeper exits behave logically

Performance improves consistently from exit 1 to exit 5. This confirms that the multi-exit architecture still behaves sensibly under a multi-label objective.

### 3. Exit 4 is a promising early-exit candidate

Exit 4 achieved macro-F1 `0.5135`, close to final-exit macro-F1 `0.5302`. This suggests that future multi-label early-exit policy work may be able to save computation with limited accuracy loss.

### 4. Label-wise behaviour is uneven

`conversation` and `scream` perform strongly, while `thunderstorm` and `road_traffic` are weak. This shows that one global threshold of `0.5` is not enough.

### 5. Thunderstorm needs special attention

The thunderstorm result is the clearest failure case:

```text
support = 49
predicted positives = 6
F1 = 0.0000
```

This may be due to weak/variable thunder cues, confusion with rain/fireworks/wind, or unsuitable thresholding.

---

## Current limitations

| Limitation | Meaning | Next fix |
|---|---|---|
| Synthetic-only mixed training | Real mixed audio transfer is not yet proven | Add small verified real mixed test set |
| Global threshold `0.5` | Some labels may need lower/higher thresholds | Tune per-label thresholds on validation set |
| No hint passing | Current v0 is no-hint only | Add sigmoid-aware hint passing later |
| No 3-exit multi-label comparison yet | Only 5-exit result is documented here | Run 3-exit no-hint baseline |
| `.m4a` excluded | Codec limitation in local environment | Convert `.m4a` to WAV/FLAC or install FFmpeg |
| Segment-level synthetic evaluation | Real weak-label clips not yet used | Add model-assisted dataset cleaning / real mixed test |

---

## Next updates after saving `kexit_cclass_greedy_multi-label_v0`

The next branch progress should focus on result improvement, not changing the whole pipeline.

Recommended order:

1. **Tune per-label thresholds** using validation probabilities.
2. **Re-evaluate test metrics** using label-specific thresholds instead of global `0.5`.
3. **Run 3-exit no-hint baseline** for fair 3-exit vs 5-exit comparison.
4. **Add threshold-calibrated early-exit policy** for multi-label inference.
5. **Add sigmoid-aware hint passing** after the no-hint baselines are stable.
6. **Create a small manually verified real mixed test set** to test transfer from synthetic to real overlapping audio.
7. **Document mAP / per-label AUC** if probability quality is important.

---

## Reproducibility commands used in v0

### Build clean seed manifest

```powershell
python scripts\build_multilabel_seed_manifest.py `
  --root "multilabel_data\clean_seed" `
  --out "multilabel_data\metadata\clean_seed_manifest.csv" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --audio_exts ".wav,.flac,.mp3,.ogg" `
  --seed 42
```

### Create synthetic mixtures

```powershell
python scripts\create_synthetic_multilabel_mixtures.py `
  --seed_manifest "multilabel_data\metadata\clean_seed_manifest.csv" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --out_audio_root "multilabel_data\synthetic_mixed\audio" `
  --out_manifest "multilabel_data\metadata\synthetic_mixed_manifest.csv" `
  --combined_out "multilabel_data\metadata\multilabel_train_manifest.csv" `
  --num_train 1000 `
  --num_val 200 `
  --num_test 200 `
  --mix_size_min 2 `
  --mix_size_max 2 `
  --sample_rate 16000 `
  --clip_sec 1.0 `
  --seed 42
```

### Extract multi-label features

```powershell
python scripts\extract_multilabel_features.py `
  --manifest "multilabel_data\metadata\multilabel_train_manifest.csv" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --out_cache "multilabel_cache" `
  --sample_rate 16000 `
  --clip_sec 1.0 `
  --n_mels 64 `
  --n_fft 1024 `
  --win_ms 25 `
  --hop_ms 10 `
  --cmvn
```

### Sanity-check dataset loader

```powershell
python -c "from data.datasets_multilabel import make_multilabel_loaders; dl_tr, dl_va, dl_te, labels = make_multilabel_loaders('multilabel_cache/metadata/multilabel_features_manifest.csv', 'multilabel_cache/features', 'multilabel_data/metadata/labels.json', batch_size=8, num_workers=0, seed=42); x,y = next(iter(dl_tr)); print('labels=', labels); print('x shape=', x.shape); print('y shape=', y.shape); print('y[0]=', y[0])"
```

### Train 5-exit multi-label no-hint baseline

```powershell
python -m training.train_multilabel `
  --manifest "multilabel_cache\metadata\multilabel_features_manifest.csv" `
  --features_root "multilabel_cache\features" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --runs_root "runs_multilabel" `
  --variant "multilabel_5exit_nohint" `
  --tap_blocks "1,2,3,4" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --threshold 0.5 `
  --device cpu
```

---

## Git saving plan for this baseline

After updating documentation, save this local branch state as:

```text
kexit_cclass_greedy_multi-label_v0
```

Recommended Git commands:

```powershell
git status

git add README.md DOC_STRUCTURE.md APPENDIX.md

git add scripts\rename_wavs_by_class.py `
        scripts\build_multilabel_seed_manifest.py `
        scripts\create_synthetic_multilabel_mixtures.py `
        scripts\extract_multilabel_features.py `
        data\datasets_multilabel.py `
        training\train_multilabel.py

git commit -m "Document and save multi-label baseline v0"

git branch kexit_cclass_greedy_multi-label_v0
```

When ready to upload:

```powershell
git push -u origin kexit_cclass_greedy_multi-label
git push -u origin kexit_cclass_greedy_multi-label_v0
```

Large generated folders should normally stay out of Git:

```text
multilabel_data/
multilabel_cache/
runs_multilabel/
```

Commit only lightweight result summaries if needed, such as `metrics.json` and `config_used.json`.

---

# Previous single-label C-class documentation

The content below is preserved from the earlier `kexit_cclass_greedy_v2` documentation.

# ASHADIP / NeuroAccuExit — `kexit_cclass_greedy_v2` Documentation Update

This update records the current **C-class prepared/grouped dataset experiments** and the latest refined-data audit for branch:

```text
kexit_cclass_greedy_v2
```

The branch now represents a more research-correct C-class evaluation path using:

- prepared `train/val/test/<class>` audio folders,
- `InputMode="ready"`,
- metadata-aware parent/source clip grouping,
- wider C-class audio bandpass `[50, 7600]`,
- 3-exit and 5-exit greedy policies,
- no-hint and sequential hint-passing comparisons,
- full-clip and Depth×Time clip-policy evaluation.

---

## Current reviewer-safe status

The most important point is that the **latest refined dataset was intended to contain 11 classes**, including:

```text
rain_thunderstorm
```

However, the available refined-run logs show a mismatch:

```text
CLI labels: car_crash,conversation,engine_idling,fireworks,gun_shot,rain,rain_thunderstorm,road_traffic,scream,thunderstorm,wind
Effective inventory labels: car_crash,conversation,engine_idling,fireworks,gun_shot,rain,road_traffic,scream,thunderstorm,wind
Effective num_classes: 10
```

Therefore, the current `*_refined11_grouped` logs should **not yet be reported as valid 11-class results**. They are useful diagnostic runs, but the final 11-class comparison must be rerun after confirming that `rain_thunderstorm` appears in:

```text
Labels: [..., rain_thunderstorm, ...]
num_classes: 11
segments: train=2453, val=462, test=418
```

---

## Main current conclusion

The lower C-class performance is **not simply because the code is broken**. The grouped ready-mode pipeline is now more scientifically honest because it reconstructs source/parent clip grouping instead of treating every 1-second WAV as an independent clip. This makes clip-policy evaluation meaningful again.

The strongest currently reportable prepared/grouped C-class result is:

```text
3exit_cclass_greedy_hint_prepared_grouped
```

It achieved:

```text
Segment policy accuracy: 68.16%
Full-clip accuracy:      84.56%
Depth×Time accuracy:    84.56%
Compute saved:          22.40%
```

This is an important change from the earlier raw 10-class C-class result, where no-hint was stronger. Under the stricter prepared/grouped dataset, **3-exit hint passing becomes useful**.

---

## Table V2.1 — Documentation state and branch status

| Item | Status | Research meaning |
|---|---|---|
| Branch | `kexit_cclass_greedy_v2` | Current branch for prepared/grouped C-class experiments |
| Dataset mode | `InputMode="ready"` | Uses already prepared `train/val/test/<class>` folders |
| Grouping | Metadata-aware source grouping | Clip policy groups windows by parent/source clip |
| Bandpass | `50,7600` | Better suited to non-moth general audio than `100,3000` |
| Valid prepared/grouped comparison | 10-class | Current fully comparable prepared/grouped result set |
| Intended refined comparison | 11-class | Requires rerun because effective logs still show 10 classes |
| Main valid winner so far | `3exit_cclass_greedy_hint_prepared_grouped` | Best prepared/grouped full-clip and Depth×Time result |

---

## Table V2.2 — Intended refined 11-class metadata counts

After adding/refining data, the intended prepared dataset should contain 11 balanced classes.

| Split | Classes | Clips per class | Expected total clips |
|---|---:|---:|---:|
| Train | 11 | 223 | 2453 |
| Val | 11 | 42 | 462 |
| Test | 11 | 38 | 418 |
| Total | 11 | — | 3333 |

Expected classes:

```text
car_crash
conversation
engine_idling
fireworks
gun_shot
rain
rain_thunderstorm
road_traffic
scream
thunderstorm
wind
```

---

## Table V2.3 — Refined-run label audit

| Check | Expected for true 11-class run | Observed in available refined logs | Status |
|---|---:|---:|---|
| CLI label list | 11 labels | 11 labels in command header | Partially correct |
| Inventory label list | 11 labels | 10 labels | Not correct |
| `rain_thunderstorm` in inventory | Yes | No | Missing |
| `num_classes` in JSON | 11 | 10 | Not correct |
| Train segments | 2453 | 2230 | Not correct |
| Val segments | 462 | 420 | Not correct |
| Test segments | 418 | 380 | Not correct |
| Scientific validity as 11-class result | Valid | Not valid yet | Rerun needed |

**Conclusion:** the new refined results should be treated as **effective 10-class diagnostic runs**, not final 11-class findings.

---

## Table V2.4 — Valid prepared/grouped 10-class comparison

These are the current valid prepared/grouped results using metadata-aware ready mode.

| Variant | Exits | Hint | Segment policy acc | Full-clip acc | Depth×Time acc | Avg windows used / total | Windows saved | Compute saved | Main interpretation |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `3exit_cclass_greedy_prepared_grouped` | 3 | No | 55.79% | 66.44% | 66.44% | 1.973 / 2.550 | 22.63% | 22.01% | Weak baseline; exit2 stronger than final exit |
| `3exit_cclass_greedy_hint_prepared_grouped` | 3 | Yes | 68.16% | **84.56%** | **84.56%** | 1.980 / 2.550 | 22.37% | **22.40%** | Best current prepared/grouped result |
| `5exit_cclass_greedy_prepared_grouped` | 5 | No | 61.58% | 75.17% | 75.17% | 1.987 / 2.550 | 22.11% | 21.99% | Improves over 3-exit no-hint but worse than 3-exit hint |
| `5exit_cclass_greedy_hint_prepared_grouped` | 5 | Yes | 68.42% | 79.87% | 79.19% | 1.960 / 2.550 | 23.16% | **23.53%** | Saves slightly more compute but lower clip accuracy than 3-exit hint |

---

## Table V2.5 — Prepared/grouped segment-policy details

| Variant | Policy acc | Avg exit depth | Exit mix | Flip-any rate | Avg flip count | Exit consistency | Tau | Val macro-F1 | Val acc |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `3exit_cclass_greedy_prepared_grouped` | 55.79% | 2.511 | e1=8.68%, e2=31.58%, e3=59.74% | 70.00% | — | 93.42% | 0.70 | 65.14% | 65.95% |
| `3exit_cclass_greedy_hint_prepared_grouped` | 68.16% | 2.808 | e1=0.53%, e2=18.16%, e3=81.32% | 67.63% | 0.884 | 99.47% | 0.90 | 71.56% | 72.14% |
| `5exit_cclass_greedy_prepared_grouped` | 61.58% | 4.547 | e1=0.00%, e2=7.63%, e3=10.26%, e4=1.84%, e5=80.26% | 69.74% | — | 99.74% | 0.95 | 66.95% | 67.14% |
| `5exit_cclass_greedy_hint_prepared_grouped` | 68.42% | 4.474 | e1=0.79%, e2=8.42%, e3=10.00%, e4=4.21%, e5=76.58% | 66.32% | — | 99.47% | 0.95 | 74.23% | 74.05% |

`—` means the exact value was not available from the accessible pasted logs.

---

## Table V2.6 — Full-clip vs Depth×Time details

| Variant | Clip mode | Clip acc | Segment acc used | Avg windows used | Avg windows total | Windows saved | Avg compute units | Compute saved | Avg depth/window | Flip rate | Exit consistency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3exit_cclass_greedy_prepared_grouped` | Full clip | 66.44% | 55.79% | 2.550 | 2.550 | 0.00% | 6.403 | 0.00% | 2.511 | 70.00% | 93.42% |
| `3exit_cclass_greedy_prepared_grouped` | Depth×Time | 66.44% | 57.14% | 1.973 | 2.550 | 22.63% | 4.993 | 22.01% | 2.531 | 71.77% | 93.54% |
| `3exit_cclass_greedy_hint_prepared_grouped` | Full clip | **84.56%** | 68.16% | 2.550 | 2.550 | 0.00% | 7.161 | 0.00% | 2.808 | 67.63% | 99.47% |
| `3exit_cclass_greedy_hint_prepared_grouped` | Depth×Time | **84.56%** | 71.86% | 1.980 | 2.550 | 22.37% | 5.557 | 22.40% | 2.807 | 70.51% | 99.66% |
| `5exit_cclass_greedy_prepared_grouped` | Full clip | 75.17% | 61.58% | 2.550 | 2.550 | 0.00% | 11.597 | 0.00% | 4.547 | 69.74% | 99.74% |
| `5exit_cclass_greedy_prepared_grouped` | Depth×Time | 75.17% | — | 1.987 | 2.550 | 22.11% | 9.047 | 21.99% | 4.554 | 71.62% | 99.66% |
| `5exit_cclass_greedy_hint_prepared_grouped` | Full clip | 79.87% | 68.42% | 2.550 | 2.550 | 0.00% | 11.409 | 0.00% | 4.474 | 66.32% | 99.47% |
| `5exit_cclass_greedy_hint_prepared_grouped` | Depth×Time | 79.19% | — | 1.960 | 2.550 | 23.16% | 8.725 | 23.53% | 4.452 | 67.81% | 99.66% |

---

## Table V2.7 — Refined diagnostic runs available so far

These runs are useful diagnostically, but should not be described as final 11-class results because effective processing still shows `num_classes=10`.

| Variant | Exits | Hint | Effective classes | Segment policy acc | Full-clip acc | Depth×Time acc | Main use |
|---|---:|---|---:|---:|---:|---:|---|
| `3exit_cclass_greedy_refined11_grouped` | 3 | No | 10 | 71.32% | 83.33% | 83.97% | Diagnostic only; missing `rain_thunderstorm` |
| `3exit_cclass_greedy_hint_refined11_grouped` | 3 | Yes | 10 | 73.68% | 80.77% | Not fully extracted | Diagnostic only; missing `rain_thunderstorm` |
| `5exit_cclass_greedy_refined11_grouped` | 5 | No | 10 | 71.05% | 79.49% | 79.49% | Diagnostic only; missing `rain_thunderstorm` |
| `5exit_cclass_greedy_hint_refined11_grouped` | 5 | Yes | 10 | 72.37% | 82.05% | Not fully extracted | Diagnostic only; missing `rain_thunderstorm` |

---

## Research findings from the current v2 work

### 1. Ready-mode grouped evaluation is now meaningful

The grouped ready-mode fix is essential because it prevents each 1-second prepared WAV from being treated as a separate clip. This restores meaningful full-clip and Depth×Time evaluation.

### 2. The stricter prepared dataset changes the hint-passing story

In the older raw 10-class run, hint passing reduced C-class performance. In the stricter prepared/grouped result, the best model is now `3exit_cclass_greedy_hint_prepared_grouped`.

This means the current conclusion should be updated:

> Hint passing is not universally helpful, but under the prepared/grouped C-class evaluation, the compact 3-exit hint model currently gives the best clip-level result.

### 3. 5 exits are still not clearly better

The 5-exit prepared/grouped models mostly use late exits, especially exit 5. They can improve over weak 3-exit no-hint baselines, but they do not beat 3-exit hint at clip level.

### 4. Depth×Time is still useful

Depth×Time saves about **22–24% compute** in the prepared/grouped C-class setup while preserving or nearly preserving clip accuracy.

### 5. The intended 11-class experiment must be rerun

Because `rain_thunderstorm` is missing from effective logs, the refined dataset cannot yet be reported as a valid 11-class experiment.

---

## Correct rerun commands for final 11-class comparison

Use dynamic label detection from the prepared training folders.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$DATA_ROOT="prepared_data2\final_dataset"
$CCLASS_LABELS=((Get-ChildItem -Directory "$DATA_ROOT\train" | Sort-Object Name).Name -join ",")

Write-Host "Detected labels:"
Write-Host $CCLASS_LABELS
```

Before training, verify this prints:

```text
car_crash,conversation,engine_idling,fireworks,gun_shot,rain,rain_thunderstorm,road_traffic,scream,thunderstorm,wind
```

### 3exit no-hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot $DATA_ROOT `
  -CacheRoot "data_caches" `
  -Config "configs\audio_cclass_ready.yaml" `
  -RunsRoot "runs" `
  -Variant "3exit_cclass_greedy_refined11_grouped" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "ready" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -SampleRate 16000 `
  -SilenceDbfs -50 `
  -Bandpass "50,7600" `
  -NMels 64 `
  -TapBlocks "1,3" `
  -ExitHint "false" `
  -ForceRebuild `
  -RunClipPolicy `
  -TimeMinWindows 1
```

### 3exit hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot $DATA_ROOT `
  -CacheRoot "data_caches" `
  -Config "configs\audio_cclass_ready.yaml" `
  -RunsRoot "runs" `
  -Variant "3exit_cclass_greedy_hint_refined11_grouped" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "ready" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -SampleRate 16000 `
  -SilenceDbfs -50 `
  -Bandpass "50,7600" `
  -NMels 64 `
  -TapBlocks "1,3" `
  -ExitHint "true" `
  -ForceRebuild `
  -RunClipPolicy `
  -TimeMinWindows 1
```

### 5exit no-hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot $DATA_ROOT `
  -CacheRoot "data_caches" `
  -Config "configs\audio_cclass_ready.yaml" `
  -RunsRoot "runs" `
  -Variant "5exit_cclass_greedy_refined11_grouped" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "ready" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -SampleRate 16000 `
  -SilenceDbfs -50 `
  -Bandpass "50,7600" `
  -NMels 64 `
  -TapBlocks "1,2,3,4" `
  -ExitHint "false" `
  -ForceRebuild `
  -RunClipPolicy `
  -TimeMinWindows 1
```

### 5exit hint

```powershell
.\scripts\run_full.ps1 `
  -DataRoot $DATA_ROOT `
  -CacheRoot "data_caches" `
  -Config "configs\audio_cclass_ready.yaml" `
  -RunsRoot "runs" `
  -Variant "5exit_cclass_greedy_hint_refined11_grouped" `
  -Policy "greedy" `
  -Device "cpu" `
  -InputMode "ready" `
  -Labels $CCLASS_LABELS `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -SampleRate 16000 `
  -SilenceDbfs -50 `
  -Bandpass "50,7600" `
  -NMels 64 `
  -TapBlocks "1,2,3,4" `
  -ExitHint "true" `
  -ForceRebuild `
  -RunClipPolicy `
  -TimeMinWindows 1
```

---

## Must-pass checklist before reporting final 11-class results

The next run is valid only if the terminal shows:

```text
Labels: ['car_crash', 'conversation', 'engine_idling', 'fireworks', 'gun_shot', 'rain', 'rain_thunderstorm', 'road_traffic', 'scream', 'thunderstorm', 'wind']
num_classes: 11
Segments: {'train': 2453, 'val': 462, 'test': 418}
```

If the terminal again shows only 10 labels or `num_classes: 10`, do not report it as an 11-class result.

---

## Recommended next research step

Do **not** change architecture again until the 11-class ingestion issue is fixed and the four final commands above are rerun.

After that, the next implementation work should be:

1. early stopping based on validation macro-F1,
2. `ReduceLROnPlateau`,
3. best-epoch checkpoint reporting,
4. class/source-aware sampling if needed,
5. confidence-gated hint passing,
6. stronger auxiliary loss or knowledge distillation for early exits.

---

## Git update commands for branch `kexit_cclass_greedy_v2`

After replacing the three documentation files:

```powershell
git checkout kexit_cclass_greedy_v2
git status

git add README.md DOC_STRUCTURE.md APPENDIX.md
git commit -m "docs: update kexit cclass greedy v2 findings"
git push origin kexit_cclass_greedy_v2
```

---



---

# Historical / Previous 8-Run Documentation

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
