# ASHADIP / NeuroAccuExit — `kexit_cclass_greedy_multi-label_pos-weight`

This document records the current **multi-label positive-label-weighting early-exit audio classification baseline** for the active branch:

```text
kexit_cclass_greedy_multi-label_pos-weight
```

This document is intentionally focused only on the active positive-weight multi-label branch.

---

## Executive summary

The current multi-label baseline is now thesis-ready as a controlled experimental record.

1. The pipeline works end-to-end: clean seed audio, synthetic mixtures, multi-hot labels, log-mel features, K-exit training, threshold tuning, and positive-label-weighting ablation.
2. Fixed global threshold `0.5` under-predicts labels. Per-label tuning improved no-weight final macro-F1 from `0.5319 → 0.6301` for 3-exit and `0.5302 → 0.6152` for 5-exit.
3. Positive label weighting improves label-balanced performance. The best final macro-F1 is now `0.6530` from `3exit_nohint_posweight` with tuned thresholds.
4. Positive weighting increases recall but risks false positives. This appears through higher average predicted label counts and worse hamming/exact-match behaviour in some settings.
5. For 5-exit models, Exit 4 is consistently the best tuned macro-F1 exit, not Exit 5. This is the strongest early-exit research signal.
6. Thunderstorm improves with positive weighting but remains the most difficult label.

---

## Multi-label task formulation

| Component | Multi-label setting |
|---|---|
| Target format | Multi-hot vector |
| Output activation | Sigmoid |
| Loss | BCEWithLogitsLoss |
| Prediction | Any number of labels |
| Thresholding | Fixed `0.5` baseline and tuned per-label thresholds |
| Main metrics | Macro-F1, micro-F1, samples-F1, exact match, hamming loss, per-label precision/recall/F1 |

---

## Labels

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

## Dataset construction

### Clean seed split counts after excluding `.m4a`

| Split | Total | car_crash | conversation | engine_idling | fireworks | gun_shot | rain | road_traffic | scream | thunderstorm | wind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 724 | 64 | 57 | 32 | 57 | 155 | 66 | 80 | 106 | 38 | 69 |
| Val | 155 | 14 | 12 | 7 | 12 | 33 | 14 | 17 | 23 | 8 | 15 |
| Test | 156 | 14 | 12 | 7 | 13 | 33 | 15 | 17 | 22 | 8 | 15 |


### Synthetic mixture design

| Setting | Value |
|---|---:|
| Synthetic train mixtures | 1000 |
| Synthetic validation mixtures | 200 |
| Synthetic test mixtures | 200 |
| Labels per synthetic clip | 2 |
| Sample rate | 16000 Hz |
| Clip duration | 1.0 s |
| Gain range | -6 dB to 0 dB |
| Split leakage control | Train mixtures use train seed files only; val mixtures use val seed files only; test mixtures use test seed files only |
| Seed | 42 |

### Synthetic positive label counts

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


### Combined manifest and feature settings

| Item | Value |
|---|---:|
| Train rows | 1724 |
| Validation rows | 355 |
| Test rows | 356 |
| Total rows | 2435 |
| Sample rate | 16000 Hz |
| Clip duration | 1.0 s |
| Mel bins | 64 |
| FFT size | 1024 |
| Window length | 25 ms |
| Hop length | 10 ms |
| CMVN | Enabled |
| Loaded feature tensor | `[batch, 1, 64, 101]` |
| Target tensor | `[batch, 10]` |

### Overall positive label counts after feature extraction

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


### Train positive label counts used by loader

| Label | Train positive count |
|---|---:|
| car_crash | 267 |
| conversation | 264 |
| engine_idling | 237 |
| fireworks | 226 |
| gun_shot | 329 |
| rain | 278 |
| road_traffic | 280 |
| scream | 323 |
| thunderstorm | 228 |
| wind | 292 |


---

## Model variants and training setup

| Variant | Tap blocks | Exits | Pos-weight | Loss weights | Best epoch | Best validation final-exit macro-F1 |
|---|---|---:|---|---|---:|---:|
| `multilabel_3exit_nohint` | `1,3` | 3 | No | `[0.3, 0.3, 1.0]` | 27 | 0.5469 |
| `multilabel_5exit_nohint` | `1,2,3,4` | 5 | No | `[0.3, 0.3, 0.6, 0.8, 1.0]` | 37 | 0.5438 |
| `multilabel_3exit_nohint_posweight` | `1,3` | 3 | Yes, cap 5.0 | `[0.3, 0.3, 1.0]` | 39 | 0.6251 |
| `multilabel_5exit_nohint_posweight` | `1,2,3,4` | 5 | Yes, cap 5.0 | `[0.3, 0.3, 0.6, 0.8, 1.0]` | 35 | 0.6199 |


### Positive label weights

The weighted-loss runs used:

```text
--use_pos_weight --pos_weight_max 5.0
```

| Label | Positive weight used |
|---|---:|
| car_crash | 5.0000 |
| conversation | 5.0000 |
| engine_idling | 5.0000 |
| fireworks | 5.0000 |
| gun_shot | 4.2401 |
| rain | 5.0000 |
| road_traffic | 5.0000 |
| scream | 4.3375 |
| thunderstorm | 5.0000 |
| wind | 4.9041 |


Positive weighting increases the penalty for missing active labels. It is expected to increase recall, but it can also increase false positives.

---

## Final-exit comparison across all four models

| Model | Exits | Exit | Pos-weight | Threshold | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg true labels | Avg predicted labels |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `3exit_nohint` | 3 | 3 | No | fixed 0.5 | 0.5319 | 0.5920 | 0.5598 | 0.3034 | **0.1065** | 1.5618 | 1.0478 |
| `3exit_nohint` | 3 | 3 | No | tuned | 0.6301 | 0.6361 | 0.6576 | 0.2725 | 0.1247 | 1.5618 | 1.8652 |
| `3exit_nohint_posweight` | 3 | 3 | Yes | fixed 0.5 | 0.6422 | 0.6379 | **0.6663** | 0.2500 | 0.1368 | 1.5618 | 2.2163 |
| `3exit_nohint_posweight` | 3 | 3 | Yes | tuned | **0.6530** | 0.6427 | 0.6580 | 0.2978 | 0.1256 | 1.5618 | 1.9522 |
| `5exit_nohint` | 5 | 5 | No | fixed 0.5 | 0.5302 | 0.5852 | 0.5545 | 0.3062 | 0.1067 | 1.5618 | 1.0112 |
| `5exit_nohint` | 5 | 5 | No | tuned | 0.6152 | **0.6454** | 0.6639 | **0.3343** | 0.1157 | 1.5618 | 1.7022 |
| `5exit_nohint_posweight` | 5 | 5 | Yes | fixed 0.5 | 0.6159 | 0.5931 | 0.6319 | 0.2556 | 0.1626 | 1.5618 | 2.4354 |
| `5exit_nohint_posweight` | 5 | 5 | Yes | tuned | 0.6232 | 0.6085 | 0.6341 | 0.2640 | 0.1424 | 1.5618 | 2.0758 |


### Main interpretation

The strongest final macro-F1 is:

```text
3exit_nohint_posweight + tuned thresholds = 0.6530
```

The strongest micro-F1 and exact-match model is:

```text
5exit_nohint + tuned thresholds
micro-F1    = 0.6454
exact match = 0.3343
```

So the best model depends on the research objective.

---

## Best tuned exit per model

| Model | Best tuned exit | Pos-weight | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg predicted labels |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `3exit_nohint_posweight` | 3 | Yes | **0.6530** | **0.6427** | 0.6580 | 0.2978 | 0.1256 | 1.9522 |
| `5exit_nohint_posweight` | 4 | Yes | 0.6433 | 0.6349 | **0.6645** | 0.2809 | 0.1292 | 1.9775 |
| `3exit_nohint` | 3 | No | 0.6301 | 0.6361 | 0.6576 | 0.2725 | **0.1247** | 1.8652 |
| `5exit_nohint` | 4 | No | 0.6281 | 0.6291 | 0.6475 | **0.3090** | 0.1258 | 1.8315 |


### Early-exit interpretation

The repeated finding is:

```text
5-exit models achieve their best macro-F1 at Exit 4, not Exit 5.
```

This supports compute-adaptive multi-label inference: an intermediate exit can be the best accuracy–depth trade-off point.

---

## Positive label weighting ablation

### 3-exit effect

```text
3exit_nohint tuned macro-F1            = 0.6301
3exit_nohint_posweight tuned macro-F1  = 0.6530
Improvement                            = +0.0229
```

This is the clearest gain from positive weighting.

### 5-exit effect

```text
5exit_nohint tuned macro-F1            = 0.6152
5exit_nohint_posweight tuned macro-F1  = 0.6232
Improvement                            = +0.0080
```

The 5-exit gain is smaller and comes with a worse exact match and hamming loss.

### Average predicted labels

True average labels per test sample:

```text
1.5618
```

Predicted labels after tuning:

| Model | Avg predicted labels |
|---|---:|
| `3exit_nohint` tuned | 1.8652 |
| `3exit_nohint_posweight` tuned | 1.9522 |
| `5exit_nohint` tuned | 1.7022 |
| `5exit_nohint_posweight` tuned | 2.0758 |

The positive-weighted models predict more labels, especially the 5-exit positive-weighted model.

---

## Positive-weight final-exit per-label analysis

| Model | Label | Threshold | Support | Fixed P | Fixed R | Fixed F1 | Tuned P | Tuned R | Tuned F1 | ΔF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3exit_posweight` | car_crash | 0.61 | 47 | 0.5556 | 0.7447 | 0.6364 | 0.6400 | 0.6809 | 0.6598 | +0.0234 |
| `3exit_posweight` | conversation | 0.47 | 60 | 0.9828 | 0.9500 | 0.9661 | 0.9667 | 0.9667 | 0.9667 | +0.0006 |
| `3exit_posweight` | engine_idling | 0.75 | 49 | 0.3889 | 0.8571 | 0.5350 | 0.5606 | 0.7551 | 0.6435 | +0.1085 |
| `3exit_posweight` | fireworks | 0.42 | 52 | 0.4375 | 0.6731 | 0.5303 | 0.4211 | 0.7692 | 0.5442 | +0.0139 |
| `3exit_posweight` | gun_shot | 0.77 | 76 | 0.6522 | 0.7895 | 0.7143 | 0.7812 | 0.6579 | 0.7143 | +0.0000 |
| `3exit_posweight` | rain | 0.88 | 52 | 0.5930 | 0.9808 | 0.7391 | 0.8269 | 0.8269 | 0.8269 | +0.0878 |
| `3exit_posweight` | road_traffic | 0.63 | 63 | 0.4937 | 0.6190 | 0.5493 | 0.4630 | 0.3968 | 0.4274 | -0.1219 |
| `3exit_posweight` | scream | 0.51 | 57 | 0.8413 | 0.9298 | 0.8833 | 0.8413 | 0.9298 | 0.8833 | +0.0000 |
| `3exit_posweight` | thunderstorm | 0.47 | 49 | 0.2800 | 0.4286 | 0.3387 | 0.2771 | 0.4694 | 0.3485 | +0.0098 |
| `3exit_posweight` | wind | 0.41 | 51 | 0.4235 | 0.7059 | 0.5294 | 0.3796 | 0.8039 | 0.5157 | -0.0137 |
| `5exit_posweight` | car_crash | 0.43 | 47 | 0.4444 | 0.7660 | 0.5625 | 0.4138 | 0.7660 | 0.5373 | -0.0252 |
| `5exit_posweight` | conversation | 0.45 | 60 | 0.9661 | 0.9500 | 0.9580 | 0.9661 | 0.9500 | 0.9580 | +0.0000 |
| `5exit_posweight` | engine_idling | 0.57 | 49 | 0.4483 | 0.7959 | 0.5735 | 0.5429 | 0.7755 | 0.6387 | +0.0652 |
| `5exit_posweight` | fireworks | 0.64 | 52 | 0.2714 | 0.7308 | 0.3958 | 0.4156 | 0.6154 | 0.4961 | +0.1003 |
| `5exit_posweight` | gun_shot | 0.74 | 76 | 0.6566 | 0.8553 | 0.7429 | 0.7606 | 0.7105 | 0.7347 | -0.0082 |
| `5exit_posweight` | rain | 0.46 | 52 | 0.8163 | 0.7692 | 0.7921 | 0.7843 | 0.7692 | 0.7767 | -0.0154 |
| `5exit_posweight` | road_traffic | 0.43 | 63 | 0.5088 | 0.4603 | 0.4833 | 0.4507 | 0.5079 | 0.4776 | -0.0057 |
| `5exit_posweight` | scream | 0.65 | 57 | 0.7794 | 0.9298 | 0.8480 | 0.8667 | 0.9123 | 0.8889 | +0.0409 |
| `5exit_posweight` | thunderstorm | 0.57 | 49 | 0.2149 | 0.5306 | 0.3059 | 0.1837 | 0.3673 | 0.2449 | -0.0610 |
| `5exit_posweight` | wind | 0.58 | 51 | 0.3679 | 0.7647 | 0.4968 | 0.3684 | 0.6863 | 0.4795 | -0.0173 |


### Label-level conclusions

- `conversation` is consistently the strongest and most stable label.
- `scream` is also strong, especially in the 5-exit positive-weighted tuned model.
- `rain` becomes very strong in the 3-exit positive-weighted tuned model, reaching F1 `0.8269`.
- `engine_idling` improves substantially with positive weighting.
- `fireworks` improves most in the 5-exit positive-weighted tuned model, but precision remains limited.
- `thunderstorm` reaches its best result so far in the 3-exit positive-weighted tuned model: F1 `0.3485`.
- `road_traffic` remains unstable and should be inspected separately.

---

## Positive-weight tuned thresholds

| Model | Final exit | car_crash | conversation | engine_idling | fireworks | gun_shot | rain | road_traffic | scream | thunderstorm | wind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3exit_posweight` | 3 | 0.61 | 0.47 | 0.75 | 0.42 | 0.77 | 0.88 | 0.63 | 0.51 | 0.47 | 0.41 |
| `5exit_posweight` | 5 | 0.43 | 0.45 | 0.57 | 0.64 | 0.74 | 0.46 | 0.43 | 0.65 | 0.57 | 0.58 |


The positive-weighted model thresholds are generally higher than the no-weight thresholds because positive weighting pushes the model toward predicting more positives. Threshold tuning then raises some label cutoffs to recover precision.

---

## Research conclusions

1. Multi-label adaptation is successful using the existing TinyAudioCNN + ExitNet family.
2. Threshold tuning is required before fair multi-label comparison.
3. Positive label weighting improves macro-F1, especially in the compact 3-exit model.
4. Positive weighting creates a recall–precision trade-off.
5. The best 5-exit macro-F1 occurs at Exit 4, making it the key early-exit candidate.
6. Thunderstorm improves but is still unresolved.

---

## Limitations and next work

| Limitation | Meaning | Next action |
|---|---|---|
| Synthetic mixtures only | Real overlapping audio transfer is not proven | Create a manually verified real mixed test set |
| Single seed | Robustness not proven | Repeat strongest settings with multiple seeds |
| Positive-weight cap fixed at 5.0 | May be too aggressive | Test `--pos_weight_max 3.0` |
| No dynamic multi-label early-exit policy | Current results are static per-exit evaluations | Add sigmoid confidence and label-set stability policy |
| No sigmoid-aware hint passing | Existing hint logic is not multi-label-safe | Add only after no-hint and pos-weight baselines are stable |
| Thunderstorm weak | Data quality or acoustic overlap may be limiting | Inspect and improve thunderstorm seed data |
| Calibration not fully studied | Thresholds vary strongly by model | Add mAP, AUC, and calibration diagnostics |

---

## Recommended next experiments

1. Rerun 3-exit and 5-exit with `--pos_weight_max 3.0`.
2. Implement multi-label early-exit policy, excluding Exit 1 initially.
3. Add mAP and per-label AUC reporting.
4. Build a small manually verified real mixed test set.
5. Add sigmoid-aware hint passing only after the no-hint baselines are stable.

## Reproducibility commands

Run commands from:

```powershell
cd C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP
```

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

### Extract features

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

### Train 3-exit no-hint

```powershell
python -m training.train_multilabel `
  --manifest "multilabel_cache\metadata\multilabel_features_manifest.csv" `
  --features_root "multilabel_cache\features" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --runs_root "runs_multilabel" `
  --variant "multilabel_3exit_nohint" `
  --tap_blocks "1,3" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --threshold 0.5 `
  --device cpu
```

### Train 5-exit no-hint

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

### Tune no-hint thresholds

```powershell
python scripts\tune_multilabel_thresholds.py `
  --run_dir "runs_multilabel\multilabel_3exit_nohint_20260509_002118" `
  --device cpu

python scripts\tune_multilabel_thresholds.py `
  --run_dir "runs_multilabel\multilabel_5exit_nohint_20260509_001254" `
  --device cpu
```

### Train 3-exit positive-weight model

```powershell
python -m training.train_multilabel `
  --manifest "multilabel_cache\metadata\multilabel_features_manifest.csv" `
  --features_root "multilabel_cache\features" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --runs_root "runs_multilabel" `
  --variant "multilabel_3exit_nohint_posweight" `
  --tap_blocks "1,3" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --threshold 0.5 `
  --use_pos_weight `
  --pos_weight_max 5.0 `
  --device cpu
```

### Train 5-exit positive-weight model

```powershell
python -m training.train_multilabel `
  --manifest "multilabel_cache\metadata\multilabel_features_manifest.csv" `
  --features_root "multilabel_cache\features" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --runs_root "runs_multilabel" `
  --variant "multilabel_5exit_nohint_posweight" `
  --tap_blocks "1,2,3,4" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --threshold 0.5 `
  --use_pos_weight `
  --pos_weight_max 5.0 `
  --device cpu
```

### Tune positive-weight thresholds

```powershell
python scripts\tune_multilabel_thresholds.py `
  --run_dir "runs_multilabel\multilabel_3exit_nohint_posweight_20260510_094046" `
  --device cpu

python scripts\tune_multilabel_thresholds.py `
  --run_dir "runs_multilabel\multilabel_5exit_nohint_posweight_20260510_094350" `
  --device cpu
```

### Generate final 4-model summary

```powershell
python scripts\summarize_multilabel_threshold_runs.py `
  --run_dirs `
    "runs_multilabel\multilabel_3exit_nohint_20260509_002118" `
    "runs_multilabel\multilabel_5exit_nohint_20260509_001254" `
    "runs_multilabel\multilabel_3exit_nohint_posweight_20260510_094046" `
    "runs_multilabel\multilabel_5exit_nohint_posweight_20260510_094350" `
  --names `
    "3exit_nohint" `
    "5exit_nohint" `
    "3exit_nohint_posweight" `
    "5exit_nohint_posweight" `
  --out_dir "runs_multilabel\summary_thresholds_posweight"
```

## Git saving plan

These commands save the documentation directly on the active branch:

```text
kexit_cclass_greedy_multi-label_pos-weight
```

Check that the current branch is correct:

```powershell
git branch --show-current
```

Expected output:

```text
kexit_cclass_greedy_multi-label_pos-weight
```

Then stage and commit the documentation and selected lightweight summary tables:

```powershell
git status

git add README.md DOC_STRUCTURE.md APPENDIX.md

git add runs_multilabel\summary_thresholds_posweight\all_exit_metrics.csv `
        runs_multilabel\summary_thresholds_posweight\all_exit_metrics.md `
        runs_multilabel\summary_thresholds_posweight\final_exit_comparison.csv `
        runs_multilabel\summary_thresholds_posweight\final_exit_comparison.md `
        runs_multilabel\summary_thresholds_posweight\best_exit_comparison.csv `
        runs_multilabel\summary_thresholds_posweight\best_exit_comparison.md `
        runs_multilabel\summary_thresholds_posweight\final_exit_per_label.csv `
        runs_multilabel\summary_thresholds_posweight\final_exit_per_label.md `
        runs_multilabel\summary_thresholds_posweight\final_exit_thresholds.csv `
        runs_multilabel\summary_thresholds_posweight\final_exit_thresholds.md `
        runs_multilabel\summary_thresholds_posweight\README_TABLES.md

git commit -m "docs: record pos-weight multi-label branch findings"

git push -u origin kexit_cclass_greedy_multi-label_pos-weight
```

Do not commit the generated heavy data/cache folders unless there is a specific reason:

```text
multilabel_data/
multilabel_cache/
runs_multilabel/
```

Only documentation, scripts, configs, and selected lightweight summary tables should normally be committed.
