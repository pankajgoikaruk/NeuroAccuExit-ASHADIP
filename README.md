
# ASHADIP / NeuroAccuExit — Multi-label Early-exit Audio Baseline v0.1

This document records the updated **multi-label early-exit audio classification baseline** for the local branch:

```text
kexit_cclass_greedy_multi-label
```

The recommended snapshot name for this state is:

```text
kexit_cclass_greedy_multi-label_v0.1_threshold_tuned
```

This update supersedes the earlier `v0` note that only documented the first 5-exit fixed-threshold run. The branch now contains a complete controlled baseline:

```text
clean seed audio
→ leakage-safe clean seed manifest
→ synthetic 2-label mixtures
→ multi-hot combined manifest
→ log-mel feature cache
→ multi-label PyTorch loader
→ 3-exit and 5-exit no-hint training
→ per-label threshold tuning
→ detailed result summary tables
```

---

## Executive summary

The multi-label branch is now functional and research-readable. The most important findings are:

1. **The full multi-label pipeline works end-to-end.** Audio files are converted into multi-hot targets and feature tensors of shape `[batch, 1, 64, 101]`.
2. **A fixed global sigmoid threshold of `0.5` is not suitable.** Per-label threshold tuning improved final-exit macro-F1 from `0.5319 → 0.6301` for the 3-exit model and from `0.5302 → 0.6152` for the 5-exit model.
3. **The 3-exit tuned model gives the best final-exit macro-F1.** Its final exit achieved `macro-F1 = 0.6301`.
4. **The 5-exit tuned model gives better micro-F1 and exact-match accuracy.** Its final exit achieved `micro-F1 = 0.6454` and `exact match = 0.3343`.
5. **The most important early-exit finding is 5-exit Exit 4.** After threshold tuning, Exit 4 achieved `macro-F1 = 0.6281`, almost matching the best 3-exit final exit at `0.6301`.
6. **Exit 1 is not deployable yet.** Threshold tuning can improve its macro-F1, but exact match remains very poor and hamming loss becomes high.
7. **Thunderstorm remains the weakest label.** Threshold tuning improves it, but it still requires dataset-level or training-level correction.

---

## Task formulation

| Item | Single-label C-class | Multi-label v0.1 |
|---|---|---|
| Target | One class ID | Multi-hot vector |
| Example mixed clip | `rain_thunderstorm` class | `rain=1`, `thunderstorm=1` |
| Output activation | Softmax | Sigmoid |
| Loss | CrossEntropyLoss | BCEWithLogitsLoss |
| Prediction type | One class only | Any number of labels |
| Primary metrics | Accuracy, macro-F1 | Macro-F1, micro-F1, samples-F1, exact match, hamming loss, per-label F1 |
| Hint passing | Existing softmax-aware logic | Disabled for v0.1; sigmoid-aware hint passing is future work |

---

## Label list

|   ID | Label         |
|-----:|:--------------|
|    0 | car_crash     |
|    1 | conversation  |
|    2 | engine_idling |
|    3 | fireworks     |
|    4 | gun_shot      |
|    5 | rain          |
|    6 | road_traffic  |
|    7 | scream        |
|    8 | thunderstorm  |
|    9 | wind          |

---

## Clean seed dataset after excluding `.m4a`

The `.m4a` files were excluded because the local environment could not decode them through `soundfile` or `librosa/audioread`. This avoided codec-related failures during mixture generation and feature extraction.

| split   |   total |   car_crash |   conversation |   engine_idling |   fireworks |   gun_shot |   rain |   road_traffic |   scream |   thunderstorm |   wind |
|:--------|--------:|------------:|---------------:|----------------:|------------:|-----------:|-------:|---------------:|---------:|---------------:|-------:|
| train   |     724 |          64 |             57 |              32 |          57 |        155 |     66 |             80 |      106 |             38 |     69 |
| val     |     155 |          14 |             12 |               7 |          12 |         33 |     14 |             17 |       23 |              8 |     15 |
| test    |     156 |          14 |             12 |               7 |          13 |         33 |     15 |             17 |       22 |              8 |     15 |

---

## Synthetic mixture design

Synthetic mixtures used exactly two labels per generated audio sample. This was chosen because it gives controlled multi-label supervision without blindly trusting weak labels from long real-world clips.

| Setting | Value |
|---|---:|
| Synthetic train mixtures | 1000 |
| Synthetic validation mixtures | 200 |
| Synthetic test mixtures | 200 |
| Labels per synthetic clip | 2 |
| Sample rate | 16000 Hz |
| Clip duration | 1.0 s |
| Gain range | -6 dB to 0 dB |
| Split leakage control | Train mixtures use train files only; val mixtures use val files only; test mixtures use test files only |
| Seed | 42 |

### Synthetic positive label counts

| label         |   positive_count |
|:--------------|-----------------:|
| car_crash     |              276 |
| conversation  |              293 |
| engine_idling |              301 |
| fireworks     |              256 |
| gun_shot      |              248 |
| rain          |              284 |
| road_traffic  |              288 |
| scream        |              289 |
| thunderstorm  |              267 |
| wind          |              298 |

---

## Combined manifest and feature dataset

| split   |   rows |
|:--------|-------:|
| train   |   1724 |
| val     |    355 |
| test    |    356 |
| total   |   2435 |

### Feature extraction settings

| Setting | Value |
|---|---:|
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

| label         |   positive_count |
|:--------------|-----------------:|
| car_crash     |              368 |
| conversation  |              374 |
| engine_idling |              347 |
| fireworks     |              338 |
| gun_shot      |              469 |
| rain          |              379 |
| road_traffic  |              402 |
| scream        |              440 |
| thunderstorm  |              321 |
| wind          |              397 |

### Train positive label counts used by the loader

| label         |   train_positive_count |
|:--------------|-----------------------:|
| car_crash     |                    267 |
| conversation  |                    264 |
| engine_idling |                    237 |
| fireworks     |                    226 |
| gun_shot      |                    329 |
| rain          |                    278 |
| road_traffic  |                    280 |
| scream        |                    323 |
| thunderstorm  |                    228 |
| wind          |                    292 |

---

## Model and training setup

| Setting | 3-exit no-hint | 5-exit no-hint |
|---|---:|---:|
| Variant | `multilabel_3exit_nohint` | `multilabel_5exit_nohint` |
| Tap blocks | `1,3` | `1,2,3,4` |
| Number of exits | 3 | 5 |
| Exit hint | Disabled | Disabled |
| Loss | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Activation for evaluation | Sigmoid | Sigmoid |
| Fixed threshold baseline | 0.5 | 0.5 |
| Epochs | 40 | 40 |
| Batch size | 64 | 64 |
| Learning rate | 0.001 | 0.001 |
| Device | CPU | CPU |
| Best epoch | 27 | 37 |
| Best validation final-exit macro-F1 | 0.5469 | 0.5438 |

---

## Fixed-threshold vs tuned-threshold final-exit comparison

| model        |   num_exits |   exit | threshold_mode   |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:-------------|------------:|-------:|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| 3exit_nohint |           3 |      3 | fixed_0p5        |     0.5319 |     0.592  |       0.5598 |        0.3034 |         0.1065 |            1.5618 |            1.0478 |
| 3exit_nohint |           3 |      3 | tuned            |     0.6301 |     0.6361 |       0.6576 |        0.2725 |         0.1247 |            1.5618 |            1.8652 |
| 5exit_nohint |           5 |      5 | fixed_0p5        |     0.5302 |     0.5852 |       0.5545 |        0.3062 |         0.1067 |            1.5618 |            1.0112 |
| 5exit_nohint |           5 |      5 | tuned            |     0.6152 |     0.6454 |       0.6639 |        0.3343 |         0.1157 |            1.5618 |            1.7022 |

### Interpretation

The tuned 3-exit model achieved the highest **macro-F1**, which means it gives the best equal-weight average across labels. The tuned 5-exit model achieved the highest **micro-F1**, **samples-F1**, and **exact match**, which means it is better when all label decisions or complete predicted label sets are considered together.

The average true labels per sample is `1.5618`. With fixed thresholding, both models under-predict labels (`1.0478` and `1.0112` predicted labels per sample). After tuning, both models predict more positives. The 3-exit tuned model is more aggressive (`1.8652` predicted labels per sample), which explains why macro-F1 improves but exact match decreases.

---

## Best tuned exit per model

| model        |   num_exits |   exit | threshold_mode   |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:-------------|------------:|-------:|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| 3exit_nohint |           3 |      3 | tuned            |     0.6301 |     0.6361 |       0.6576 |        0.2725 |         0.1247 |            1.5618 |            1.8652 |
| 5exit_nohint |           5 |      4 | tuned            |     0.6281 |     0.6291 |       0.6475 |        0.309  |         0.1258 |            1.5618 |            1.8315 |

### Early-exit interpretation

The strongest early-exit result is the **5-exit no-hint model at Exit 4**:

```text
5-exit Exit 4 tuned macro-F1 = 0.6281
5-exit Exit 5 tuned macro-F1 = 0.6152
3-exit Exit 3 tuned macro-F1 = 0.6301
```

This is a meaningful research finding. The 5-exit model does not simply need the deepest exit. Exit 4 is nearly equal to the 3-exit final exit and slightly better than the 5-exit final exit in macro-F1. This suggests that the extra intermediate exits can provide useful compute-adaptive decision points in a multi-label setting.

---

## All exit-level fixed/tuned test metrics

| model        |   num_exits |   exit | threshold_mode   |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_true_labels   | avg_pred_labels   |
|:-------------|------------:|-------:|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|:------------------|
| 3exit_nohint |           3 |      1 | fixed_0p5        |     0      |     0      | 0.0000       |        0      |         0.1562 |                   |                   |
| 3exit_nohint |           3 |      1 | tuned            |     0.3802 |     0.3561 |              |        0.0112 |         0.3809 |                   |                   |
| 3exit_nohint |           3 |      2 | fixed_0p5        |     0.3024 |     0.4091 | 0.3408       |        0.191  |         0.1242 |                   |                   |
| 3exit_nohint |           3 |      2 | tuned            |     0.557  |     0.5278 |              |        0.1601 |         0.198  |                   |                   |
| 3exit_nohint |           3 |      3 | fixed_0p5        |     0.5319 |     0.592  | 0.5598       |        0.3034 |         0.1065 | 1.5618            | 1.0478            |
| 3exit_nohint |           3 |      3 | tuned            |     0.6301 |     0.6361 | 0.6576       |        0.2725 |         0.1247 | 1.5618            | 1.8652            |
| 5exit_nohint |           5 |      1 | fixed_0p5        |     0.0068 |     0.0072 | 0.0056       |        0.0056 |         0.1556 |                   |                   |
| 5exit_nohint |           5 |      1 | tuned            |     0.3879 |     0.3619 |              |        0.0028 |         0.3635 |                   |                   |
| 5exit_nohint |           5 |      2 | fixed_0p5        |     0.2411 |     0.3476 | 0.2739       |        0.1601 |         0.1287 |                   |                   |
| 5exit_nohint |           5 |      2 | tuned            |     0.502  |     0.4722 |              |        0.1067 |         0.2242 |                   |                   |
| 5exit_nohint |           5 |      3 | fixed_0p5        |     0.3708 |     0.466  | 0.3928       |        0.1994 |         0.1191 |                   |                   |
| 5exit_nohint |           5 |      3 | tuned            |     0.5723 |     0.5563 |              |        0.191  |         0.1747 |                   |                   |
| 5exit_nohint |           5 |      4 | fixed_0p5        |     0.5135 |     0.5686 | 0.5325       |        0.2697 |         0.1104 |                   |                   |
| 5exit_nohint |           5 |      4 | tuned            |     0.6281 |     0.6291 | 0.6475       |        0.309  |         0.1258 | 1.5618            | 1.8315            |
| 5exit_nohint |           5 |      5 | fixed_0p5        |     0.5302 |     0.5852 | 0.5545       |        0.3062 |         0.1067 | 1.5618            | 1.0112            |
| 5exit_nohint |           5 |      5 | tuned            |     0.6152 |     0.6454 | 0.6639       |        0.3343 |         0.1157 | 1.5618            | 1.7022            |

### Interpretation of Exit 1

Exit 1 remains too shallow. With fixed thresholding, it predicts almost nothing useful. With tuned thresholds, macro-F1 increases, but exact match remains extremely low and hamming loss becomes high. This means threshold tuning makes Exit 1 predict many positives, but the full label sets are not reliable. Therefore, Exit 1 should not be used for deployment or early-exit stopping yet.

---

## Final-exit per-label fixed-vs-tuned results

| model        | label         |   threshold |   support |   fixed_precision |   fixed_recall |   fixed_f1 |   tuned_precision |   tuned_recall |   tuned_f1 |   delta_f1 |   delta_recall |   delta_precision |
|:-------------|:--------------|------------:|----------:|------------------:|---------------:|-----------:|------------------:|---------------:|-----------:|-----------:|---------------:|------------------:|
| 3exit_nohint | car_crash     |        0.43 |        47 |            0.6327 |         0.6596 |     0.6458 |            0.5818 |         0.6809 |     0.6275 |    -0.0183 |         0.0213 |           -0.0509 |
| 3exit_nohint | conversation  |        0.34 |        60 |            1      |         0.9    |     0.9474 |            0.9322 |         0.9167 |     0.9244 |    -0.023  |         0.0167 |           -0.0678 |
| 3exit_nohint | engine_idling |        0.21 |        49 |            0.7895 |         0.3061 |     0.4412 |            0.4844 |         0.6327 |     0.5487 |     0.1075 |         0.3266 |           -0.3051 |
| 3exit_nohint | fireworks     |        0.09 |        52 |            0.75   |         0.3462 |     0.4737 |            0.3761 |         0.7885 |     0.5093 |     0.0356 |         0.4423 |           -0.3739 |
| 3exit_nohint | gun_shot      |        0.23 |        76 |            0.84   |         0.5526 |     0.6667 |            0.7746 |         0.7237 |     0.7483 |     0.0816 |         0.1711 |           -0.0654 |
| 3exit_nohint | rain          |        0.05 |        52 |            1      |         0.0769 |     0.1429 |            0.6176 |         0.8077 |     0.7    |     0.5571 |         0.7308 |           -0.3824 |
| 3exit_nohint | road_traffic  |        0.36 |        63 |            0.5882 |         0.4762 |     0.5263 |            0.5513 |         0.6825 |     0.6099 |     0.0836 |         0.2063 |           -0.0369 |
| 3exit_nohint | scream        |        0.47 |        57 |            0.9259 |         0.8772 |     0.9009 |            0.8929 |         0.8772 |     0.885  |    -0.0159 |         0      |           -0.033  |
| 3exit_nohint | thunderstorm  |        0.24 |        49 |            1      |         0.0408 |     0.0784 |            0.2609 |         0.2449 |     0.2526 |     0.1742 |         0.2041 |           -0.7391 |
| 3exit_nohint | wind          |        0.54 |        51 |            0.4394 |         0.5686 |     0.4957 |            0.4655 |         0.5294 |     0.4954 |    -0.0003 |        -0.0392 |            0.0261 |
| 5exit_nohint | car_crash     |        0.23 |        47 |            0.6316 |         0.5106 |     0.5647 |            0.4932 |         0.766  |     0.6    |     0.0353 |         0.2554 |           -0.1384 |
| 5exit_nohint | conversation  |        0.3  |        60 |            0.9815 |         0.8833 |     0.9298 |            0.9655 |         0.9333 |     0.9492 |     0.0194 |         0.05   |           -0.016  |
| 5exit_nohint | engine_idling |        0.3  |        49 |            0.75   |         0.3673 |     0.4932 |            0.4872 |         0.3878 |     0.4318 |    -0.0614 |         0.0205 |           -0.2628 |
| 5exit_nohint | fireworks     |        0.19 |        52 |            0.8421 |         0.3077 |     0.4507 |            0.5167 |         0.5962 |     0.5536 |     0.1029 |         0.2885 |           -0.3254 |
| 5exit_nohint | gun_shot      |        0.21 |        76 |            0.7458 |         0.5789 |     0.6519 |            0.6786 |         0.75   |     0.7125 |     0.0606 |         0.1711 |           -0.0672 |
| 5exit_nohint | rain          |        0.25 |        52 |            0.88   |         0.4231 |     0.5714 |            0.8571 |         0.8077 |     0.8317 |     0.2603 |         0.3846 |           -0.0229 |
| 5exit_nohint | road_traffic  |        0.18 |        63 |            0.5    |         0.1429 |     0.2222 |            0.4725 |         0.6825 |     0.5584 |     0.3362 |         0.5396 |           -0.0275 |
| 5exit_nohint | scream        |        0.57 |        57 |            0.7971 |         0.9649 |     0.873  |            0.806  |         0.9474 |     0.871  |    -0.002  |        -0.0175 |            0.0089 |
| 5exit_nohint | thunderstorm  |        0.46 |        49 |            0      |         0      |     0      |            0.2727 |         0.0612 |     0.1    |     0.1    |         0.0612 |            0.2727 |
| 5exit_nohint | wind          |        0.33 |        51 |            0.5625 |         0.5294 |     0.5455 |            0.4595 |         0.6667 |     0.544  |    -0.0015 |         0.1373 |           -0.103  |

### Per-label research findings

- `conversation` and `scream` are consistently strong labels.
- `rain` improves dramatically after threshold tuning: in 3-exit it improves from `0.1429 → 0.7000`; in 5-exit it improves from `0.5714 → 0.8317`.
- `road_traffic` improves strongly in the 5-exit model after tuning: `0.2222 → 0.5584`.
- `gun_shot` improves in both models after tuning and remains one of the more stable event labels.
- `thunderstorm` remains weak even after tuning. It improves from `0.0784 → 0.2526` in 3-exit and from `0.0000 → 0.1000` in 5-exit, but this is still not satisfactory.
- Very low tuned thresholds, such as `rain=0.05` in the 3-exit model, suggest calibration issues. The model may rank examples correctly but assign low absolute sigmoid probabilities.

---

## Final-exit tuned thresholds

| model        |   num_exits |   final_exit | label         |   threshold |
|:-------------|------------:|-------------:|:--------------|------------:|
| 3exit_nohint |           3 |            3 | car_crash     |        0.43 |
| 3exit_nohint |           3 |            3 | conversation  |        0.34 |
| 3exit_nohint |           3 |            3 | engine_idling |        0.21 |
| 3exit_nohint |           3 |            3 | fireworks     |        0.09 |
| 3exit_nohint |           3 |            3 | gun_shot      |        0.23 |
| 3exit_nohint |           3 |            3 | rain          |        0.05 |
| 3exit_nohint |           3 |            3 | road_traffic  |        0.36 |
| 3exit_nohint |           3 |            3 | scream        |        0.47 |
| 3exit_nohint |           3 |            3 | thunderstorm  |        0.24 |
| 3exit_nohint |           3 |            3 | wind          |        0.54 |
| 5exit_nohint |           5 |            5 | car_crash     |        0.23 |
| 5exit_nohint |           5 |            5 | conversation  |        0.3  |
| 5exit_nohint |           5 |            5 | engine_idling |        0.3  |
| 5exit_nohint |           5 |            5 | fireworks     |        0.19 |
| 5exit_nohint |           5 |            5 | gun_shot      |        0.21 |
| 5exit_nohint |           5 |            5 | rain          |        0.25 |
| 5exit_nohint |           5 |            5 | road_traffic  |        0.18 |
| 5exit_nohint |           5 |            5 | scream        |        0.57 |
| 5exit_nohint |           5 |            5 | thunderstorm  |        0.46 |
| 5exit_nohint |           5 |            5 | wind          |        0.33 |

---

## Research conclusions

### Conclusion 1 — Multi-label adaptation is functional

The TinyAudioCNN + ExitNet architecture can be trained under a multi-label formulation by changing the target representation, loss, and evaluation logic. The core architecture does not need to be replaced.

### Conclusion 2 — Fixed global threshold is not research-safe

A global `0.5` threshold substantially under-detects several labels. Per-label threshold tuning is essential before comparing model variants or discussing label-level weaknesses.

### Conclusion 3 — 3-exit and 5-exit models behave differently

The 3-exit model gives the best final-exit macro-F1 after tuning. The 5-exit model gives better micro-F1 and exact match. This means neither model is universally superior; the choice depends on the evaluation goal.

### Conclusion 4 — Exit 4 in the 5-exit model is the key early-exit candidate

The 5-exit tuned model achieved its best macro-F1 at Exit 4. This supports the early-exit hypothesis: intermediate exits can offer useful compute-accuracy trade-offs.

### Conclusion 5 — Thunderstorm needs dataset or training attention

Threshold tuning alone does not solve thunderstorm. Possible causes include unclear thunder events, high overlap with rain/wind/fireworks, weak seed quality, or insufficient representation of clear thunder cues.

---

## Current limitations

| Limitation | Meaning | Correct next action |
|---|---|---|
| Synthetic mixtures only | Real overlapping audio transfer is not proven | Build a small manually verified real mixed test set |
| No sigmoid-aware hint passing | Existing hint module was designed around softmax probabilities | Add multi-label hint only after no-hint baselines are stable |
| No multi-label early-exit policy yet | Current results evaluate exits, not dynamic stopping | Add sigmoid confidence / label-set stability policy |
| Thunderstorm weak | Threshold tuning does not fully solve it | Inspect data, add clean examples, or use positive weighting |
| Some thresholds are very low | Probability calibration is weak for some labels | Add calibration, mAP/AUC, or threshold regularization |
| Single seed only | Robustness not proven | Repeat with multiple seeds later |

---

## Recommended next experiments

1. **Positive label weighting** using `--use_pos_weight` for both 3-exit and 5-exit no-hint.
2. **Multi-label policy evaluation**, excluding Exit 1 initially.
3. **Sigmoid-aware hint passing**, using sigmoid probabilities rather than softmax probabilities.
4. **mAP / per-label AUC reporting**, because thresholded F1 alone does not fully describe probability quality.
5. **Small real mixed test set**, manually verified, to test synthetic-to-real transfer.


## Reproducibility commands

### 1) Rename clean seed audio files

```powershell
python scripts\rename_wavs_by_class.py `
  --root "multilabel_data\clean_seed" `
  --digits 4 `
  --skip_already_named `
  --apply
```

### 2) Build clean seed manifest, excluding `.m4a`

```powershell
python scripts\build_multilabel_seed_manifest.py `
  --root "multilabel_data\clean_seed" `
  --out "multilabel_data\metadata\clean_seed_manifest.csv" `
  --labels_json "multilabel_data\metadata\labels.json" `
  --audio_exts ".wav,.flac,.mp3,.ogg" `
  --seed 42
```

### 3) Create synthetic multi-label mixtures

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

### 4) Extract log-mel features

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

### 5) Sanity-check the multi-label dataset loader

```powershell
python -c "from data.datasets_multilabel import make_multilabel_loaders; dl_tr, dl_va, dl_te, labels = make_multilabel_loaders('multilabel_cache/metadata/multilabel_features_manifest.csv', 'multilabel_cache/features', 'multilabel_data/metadata/labels.json', batch_size=8, num_workers=0, seed=42); x,y = next(iter(dl_tr)); print('labels=', labels); print('x shape=', x.shape); print('y shape=', y.shape); print('y[0]=', y[0])"
```

Expected shape:

```text
x shape = torch.Size([8, 1, 64, 101])
y shape = torch.Size([8, 10])
```

### 6) Train 3-exit no-hint multi-label baseline

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

### 7) Train 5-exit no-hint multi-label baseline

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

### 8) Tune per-label thresholds for both runs

```powershell
python scripts\tune_multilabel_thresholds.py `
  --run_dir "runs_multilabel\multilabel_3exit_nohint_20260509_002118" `
  --device cpu

python scripts\tune_multilabel_thresholds.py `
  --run_dir "runs_multilabel\multilabel_5exit_nohint_20260509_001254" `
  --device cpu
```

### 9) Summarise threshold-tuning results

```powershell
python scripts\summarize_multilabel_threshold_runs.py `
  --run_dirs `
    "runs_multilabel\multilabel_3exit_nohint_20260509_002118" `
    "runs_multilabel\multilabel_5exit_nohint_20260509_001254" `
  --names "3exit_nohint" "5exit_nohint" `
  --out_dir "runs_multilabel\summary_thresholds"
```


---

## Git saving plan

```powershell
git status

git add README.md DOC_STRUCTURE.md APPENDIX.md

git add scripts
ename_wavs_by_class.py `
        scriptsuild_multilabel_seed_manifest.py `
        scripts\create_synthetic_multilabel_mixtures.py `
        scripts\extract_multilabel_features.py `
        scripts	une_multilabel_thresholds.py `
        scripts\summarize_multilabel_threshold_runs.py `
        data\datasets_multilabel.py `
        training	rain_multilabel.py

git commit -m "docs: record multi-label threshold-tuned baseline v0.1"

git branch kexit_cclass_greedy_multi-label_v0.1_threshold_tuned
```

When ready to upload:

```powershell
git push -u origin kexit_cclass_greedy_multi-label
git push -u origin kexit_cclass_greedy_multi-label_v0.1_threshold_tuned
```

Generated large folders should usually stay out of Git:

```text
multilabel_data/
multilabel_cache/
runs_multilabel/
```

Commit only lightweight scripts, documentation, configs, and selected summary tables if needed.


---

# Preserved previous documentation

The content below is preserved from the uploaded README for historical continuity. The multi-label v0.1 threshold-tuned section above supersedes the earlier v0 fixed-threshold-only section where they overlap.

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

---

# Relationship to previous single-label C-class work

This branch is derived from the earlier generic K-exit / C-class codebase, but it is **not** a continuation of the single-label C-class experiment.

The earlier C-class branches used single-label multi-class classification:

```text
one audio segment → one class only
softmax activation
CrossEntropyLoss
greedy / hint / Depth×Time policies for single-label decisions
```

This branch changes the learning problem to multi-label audio tagging:

```text
one audio segment → zero, one, or multiple active labels
sigmoid activation
BCEWithLogitsLoss
per-label threshold tuning
future sigmoid-aware early-exit policy
```

For the full older C-class documentation and result tables, use the dedicated C-class branches, especially:

```text
kexit_cclass_greedy_v2
```

The current branch should report only the multi-label dataset construction, synthetic-mixture design, multi-label training, threshold tuning, and multi-label early-exit findings. The old C-class tables are intentionally **not preserved inside this branch documentation** to avoid mixing two different experimental tracks.

---

# Git saving plan for this cleaned multi-label documentation state

```powershell
git status

git add README.md DOC_STRUCTURE.md APPENDIX.md

git add runs_multilabel\summary_thresholds\all_exit_metrics.csv `
        runs_multilabel\summary_thresholds\all_exit_metrics.md `
        runs_multilabel\summary_thresholds\final_exit_comparison.csv `
        runs_multilabel\summary_thresholds\final_exit_comparison.md `
        runs_multilabel\summary_thresholds\best_exit_comparison.csv `
        runs_multilabel\summary_thresholds\best_exit_comparison.md `
        runs_multilabel\summary_thresholds\final_exit_per_label.csv `
        runs_multilabel\summary_thresholds\final_exit_per_label.md `
        runs_multilabel\summary_thresholds\final_exit_thresholds.csv `
        runs_multilabel\summary_thresholds\final_exit_thresholds.md `
        runs_multilabel\summary_thresholds\README_TABLES.md

git commit -m "docs: clean multi-label threshold-tuned baseline documentation"

git branch kexit_cclass_greedy_multi-label_v0.1_threshold_tuned
```

When ready to upload:

```powershell
git push -u origin kexit_cclass_greedy_multi-label
git push -u origin kexit_cclass_greedy_multi-label_v0.1_threshold_tuned
```
