
# Appendix — Multi-label Early-exit Audio Baseline v0.1

This appendix provides the detailed result record for:

```text
kexit_cclass_greedy_multi-label_v0.1_threshold_tuned
```

---

## ML-A1. Branch and artifact status

| Item | Value |
|---|---|
| Working branch | `kexit_cclass_greedy_multi-label` |
| Recommended snapshot | `kexit_cclass_greedy_multi-label_v0.1_threshold_tuned` |
| Task | Multi-label audio tagging |
| Labels | 10 |
| Input | 1-second log-mel audio windows |
| Model family | TinyAudioCNN + ExitNet |
| Compared variants | 3-exit no-hint, 5-exit no-hint |
| Thresholding | Fixed 0.5 and tuned per-label thresholds |
| Status | End-to-end baseline complete |

---

## ML-A2. Label list

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

## ML-A3. Clean seed split counts after excluding `.m4a`

| split   |   total |   car_crash |   conversation |   engine_idling |   fireworks |   gun_shot |   rain |   road_traffic |   scream |   thunderstorm |   wind |
|:--------|--------:|------------:|---------------:|----------------:|------------:|-----------:|-------:|---------------:|---------:|---------------:|-------:|
| train   |     724 |          64 |             57 |              32 |          57 |        155 |     66 |             80 |      106 |             38 |     69 |
| val     |     155 |          14 |             12 |               7 |          12 |         33 |     14 |             17 |       23 |              8 |     15 |
| test    |     156 |          14 |             12 |               7 |          13 |         33 |     15 |             17 |       22 |              8 |     15 |

---

## ML-A4. Synthetic mixture settings

| Setting | Value |
|---|---:|
| Number of labels per synthetic clip | 2 |
| Train mixtures | 1000 |
| Val mixtures | 200 |
| Test mixtures | 200 |
| Sample rate | 16000 Hz |
| Clip duration | 1.0 s |
| Gain range | -6 dB to 0 dB |
| Seed | 42 |
| Leakage rule | Each synthetic split uses only clean seed files from the same split |

---

## ML-A5. Synthetic positive label counts

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

## ML-A6. Combined manifest counts

| split   |   rows |
|:--------|-------:|
| train   |   1724 |
| val     |    355 |
| test    |    356 |
| total   |   2435 |

---

## ML-A7. Feature extraction summary

| Item | Value |
|---|---:|
| Total feature rows | 2435 |
| Clean rows | 1035 |
| Synthetic rows | 1400 |
| Train rows | 1724 |
| Validation rows | 355 |
| Test rows | 356 |
| Loaded x shape | `[batch, 1, 64, 101]` |
| Loaded y shape | `[batch, 10]` |

---

## ML-A8. Label-positive counts after feature extraction

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

---

## ML-A9. Training setup

| variant                 | tap_blocks   |   exits | loss_weights              |   best_epoch |   best_val_final_macro_f1 |
|:------------------------|:-------------|--------:|:--------------------------|-------------:|--------------------------:|
| multilabel_3exit_nohint | 1,3          |       3 | [0.3, 0.3, 1.0]           |           27 |                    0.5469 |
| multilabel_5exit_nohint | 1,2,3,4      |       5 | [0.3, 0.3, 0.6, 0.8, 1.0] |           37 |                    0.5438 |

---

## ML-A10. All exit-level metrics

| model        |   num_exits |   exit | split   | threshold_mode   |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_true_labels   | avg_pred_labels   |
|:-------------|------------:|-------:|:--------|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|:------------------|
| 3exit_nohint |           3 |      1 | val     | fixed_0p5        |     0      |     0      |              |        0      |         0.1563 |                   |                   |
| 3exit_nohint |           3 |      1 | val     | tuned            |     0.4038 |     0.3848 |              |        0.0197 |         0.3566 |                   |                   |
| 3exit_nohint |           3 |      1 | test    | fixed_0p5        |     0      |     0      | 0.0000       |        0      |         0.1562 |                   |                   |
| 3exit_nohint |           3 |      1 | test    | tuned            |     0.3802 |     0.3561 |              |        0.0112 |         0.3809 |                   |                   |
| 3exit_nohint |           3 |      2 | val     | fixed_0p5        |     0.2755 |     0.3719 |              |        0.1775 |         0.1285 |                   |                   |
| 3exit_nohint |           3 |      2 | val     | tuned            |     0.5448 |     0.5174 |              |        0.169  |         0.2076 |                   |                   |
| 3exit_nohint |           3 |      2 | test    | fixed_0p5        |     0.3024 |     0.4091 | 0.3408       |        0.191  |         0.1242 |                   |                   |
| 3exit_nohint |           3 |      2 | test    | tuned            |     0.557  |     0.5278 |              |        0.1601 |         0.198  |                   |                   |
| 3exit_nohint |           3 |      3 | val     | fixed_0p5        |     0.5469 |     0.5924 |              |        0.2986 |         0.1062 |                   |                   |
| 3exit_nohint |           3 |      3 | val     | tuned            |     0.6563 |     0.645  |              |        0.2817 |         0.1231 |                   |                   |
| 3exit_nohint |           3 |      3 | test    | fixed_0p5        |     0.5319 |     0.592  | 0.5598       |        0.3034 |         0.1065 | 1.5618            | 1.0478            |
| 3exit_nohint |           3 |      3 | test    | tuned            |     0.6301 |     0.6361 | 0.6576       |        0.2725 |         0.1247 | 1.5618            | 1.8652            |
| 5exit_nohint |           5 |      1 | val     | fixed_0p5        |     0.0152 |     0.0178 |              |        0.0141 |         0.1552 |                   |                   |
| 5exit_nohint |           5 |      1 | val     | tuned            |     0.4225 |     0.3877 |              |        0.0113 |         0.3524 |                   |                   |
| 5exit_nohint |           5 |      1 | test    | fixed_0p5        |     0.0068 |     0.0072 | 0.0056       |        0.0056 |         0.1556 |                   |                   |
| 5exit_nohint |           5 |      1 | test    | tuned            |     0.3879 |     0.3619 |              |        0.0028 |         0.3635 |                   |                   |
| 5exit_nohint |           5 |      2 | val     | fixed_0p5        |     0.2388 |     0.329  |              |        0.1746 |         0.1321 |                   |                   |
| 5exit_nohint |           5 |      2 | val     | tuned            |     0.5179 |     0.4896 |              |        0.1324 |         0.2273 |                   |                   |
| 5exit_nohint |           5 |      2 | test    | fixed_0p5        |     0.2411 |     0.3476 | 0.2739       |        0.1601 |         0.1287 |                   |                   |
| 5exit_nohint |           5 |      2 | test    | tuned            |     0.502  |     0.4722 |              |        0.1067 |         0.2242 |                   |                   |
| 5exit_nohint |           5 |      3 | val     | fixed_0p5        |     0.3593 |     0.4436 |              |        0.2254 |         0.1208 |                   |                   |
| 5exit_nohint |           5 |      3 | val     | tuned            |     0.5823 |     0.5608 |              |        0.1944 |         0.1792 |                   |                   |
| 5exit_nohint |           5 |      3 | test    | fixed_0p5        |     0.3708 |     0.466  | 0.3928       |        0.1994 |         0.1191 |                   |                   |
| 5exit_nohint |           5 |      3 | test    | tuned            |     0.5723 |     0.5563 |              |        0.191  |         0.1747 |                   |                   |
| 5exit_nohint |           5 |      4 | val     | fixed_0p5        |     0.5079 |     0.5535 |              |        0.2845 |         0.1104 |                   |                   |
| 5exit_nohint |           5 |      4 | val     | tuned            |     0.6342 |     0.6251 |              |        0.2732 |         0.1287 |                   |                   |
| 5exit_nohint |           5 |      4 | test    | fixed_0p5        |     0.5135 |     0.5686 | 0.5325       |        0.2697 |         0.1104 |                   |                   |
| 5exit_nohint |           5 |      4 | test    | tuned            |     0.6281 |     0.6291 | 0.6475       |        0.309  |         0.1258 | 1.5618            | 1.8315            |
| 5exit_nohint |           5 |      5 | val     | fixed_0p5        |     0.5438 |     0.5827 |              |        0.3042 |         0.1045 |                   |                   |
| 5exit_nohint |           5 |      5 | val     | tuned            |     0.6571 |     0.6599 |              |        0.3577 |         0.1132 |                   |                   |
| 5exit_nohint |           5 |      5 | test    | fixed_0p5        |     0.5302 |     0.5852 | 0.5545       |        0.3062 |         0.1067 | 1.5618            | 1.0112            |
| 5exit_nohint |           5 |      5 | test    | tuned            |     0.6152 |     0.6454 | 0.6639       |        0.3343 |         0.1157 | 1.5618            | 1.7022            |

---

## ML-A11. Final-exit comparison

| model        |   num_exits |   exit | threshold_mode   |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:-------------|------------:|-------:|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| 3exit_nohint |           3 |      3 | fixed_0p5        |     0.5319 |     0.592  |       0.5598 |        0.3034 |         0.1065 |            1.5618 |            1.0478 |
| 3exit_nohint |           3 |      3 | tuned            |     0.6301 |     0.6361 |       0.6576 |        0.2725 |         0.1247 |            1.5618 |            1.8652 |
| 5exit_nohint |           5 |      5 | fixed_0p5        |     0.5302 |     0.5852 |       0.5545 |        0.3062 |         0.1067 |            1.5618 |            1.0112 |
| 5exit_nohint |           5 |      5 | tuned            |     0.6152 |     0.6454 |       0.6639 |        0.3343 |         0.1157 |            1.5618 |            1.7022 |

---

## ML-A12. Best tuned exit per model

| model        |   num_exits |   exit | threshold_mode   |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:-------------|------------:|-------:|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| 3exit_nohint |           3 |      3 | tuned            |     0.6301 |     0.6361 |       0.6576 |        0.2725 |         0.1247 |            1.5618 |            1.8652 |
| 5exit_nohint |           5 |      4 | tuned            |     0.6281 |     0.6291 |       0.6475 |        0.309  |         0.1258 |            1.5618 |            1.8315 |

---

## ML-A13. Final-exit per-label fixed-vs-tuned metrics

| model        | label         |   threshold |   support |   fixed_precision |   fixed_recall |   fixed_f1 |   fixed_pred_pos |   tuned_precision |   tuned_recall |   tuned_f1 |   delta_f1 |   delta_recall |   delta_precision |
|:-------------|:--------------|------------:|----------:|------------------:|---------------:|-----------:|-----------------:|------------------:|---------------:|-----------:|-----------:|---------------:|------------------:|
| 3exit_nohint | car_crash     |        0.43 |        47 |            0.6327 |         0.6596 |     0.6458 |               49 |            0.5818 |         0.6809 |     0.6275 |    -0.0183 |         0.0213 |           -0.0509 |
| 3exit_nohint | conversation  |        0.34 |        60 |            1      |         0.9    |     0.9474 |               54 |            0.9322 |         0.9167 |     0.9244 |    -0.023  |         0.0167 |           -0.0678 |
| 3exit_nohint | engine_idling |        0.21 |        49 |            0.7895 |         0.3061 |     0.4412 |               19 |            0.4844 |         0.6327 |     0.5487 |     0.1075 |         0.3266 |           -0.3051 |
| 3exit_nohint | fireworks     |        0.09 |        52 |            0.75   |         0.3462 |     0.4737 |               24 |            0.3761 |         0.7885 |     0.5093 |     0.0356 |         0.4423 |           -0.3739 |
| 3exit_nohint | gun_shot      |        0.23 |        76 |            0.84   |         0.5526 |     0.6667 |               50 |            0.7746 |         0.7237 |     0.7483 |     0.0816 |         0.1711 |           -0.0654 |
| 3exit_nohint | rain          |        0.05 |        52 |            1      |         0.0769 |     0.1429 |                4 |            0.6176 |         0.8077 |     0.7    |     0.5571 |         0.7308 |           -0.3824 |
| 3exit_nohint | road_traffic  |        0.36 |        63 |            0.5882 |         0.4762 |     0.5263 |               51 |            0.5513 |         0.6825 |     0.6099 |     0.0836 |         0.2063 |           -0.0369 |
| 3exit_nohint | scream        |        0.47 |        57 |            0.9259 |         0.8772 |     0.9009 |               54 |            0.8929 |         0.8772 |     0.885  |    -0.0159 |         0      |           -0.033  |
| 3exit_nohint | thunderstorm  |        0.24 |        49 |            1      |         0.0408 |     0.0784 |                2 |            0.2609 |         0.2449 |     0.2526 |     0.1742 |         0.2041 |           -0.7391 |
| 3exit_nohint | wind          |        0.54 |        51 |            0.4394 |         0.5686 |     0.4957 |               66 |            0.4655 |         0.5294 |     0.4954 |    -0.0003 |        -0.0392 |            0.0261 |
| 5exit_nohint | car_crash     |        0.23 |        47 |            0.6316 |         0.5106 |     0.5647 |               38 |            0.4932 |         0.766  |     0.6    |     0.0353 |         0.2554 |           -0.1384 |
| 5exit_nohint | conversation  |        0.3  |        60 |            0.9815 |         0.8833 |     0.9298 |               54 |            0.9655 |         0.9333 |     0.9492 |     0.0194 |         0.05   |           -0.016  |
| 5exit_nohint | engine_idling |        0.3  |        49 |            0.75   |         0.3673 |     0.4932 |               24 |            0.4872 |         0.3878 |     0.4318 |    -0.0614 |         0.0205 |           -0.2628 |
| 5exit_nohint | fireworks     |        0.19 |        52 |            0.8421 |         0.3077 |     0.4507 |               19 |            0.5167 |         0.5962 |     0.5536 |     0.1029 |         0.2885 |           -0.3254 |
| 5exit_nohint | gun_shot      |        0.21 |        76 |            0.7458 |         0.5789 |     0.6519 |               59 |            0.6786 |         0.75   |     0.7125 |     0.0606 |         0.1711 |           -0.0672 |
| 5exit_nohint | rain          |        0.25 |        52 |            0.88   |         0.4231 |     0.5714 |               25 |            0.8571 |         0.8077 |     0.8317 |     0.2603 |         0.3846 |           -0.0229 |
| 5exit_nohint | road_traffic  |        0.18 |        63 |            0.5    |         0.1429 |     0.2222 |               18 |            0.4725 |         0.6825 |     0.5584 |     0.3362 |         0.5396 |           -0.0275 |
| 5exit_nohint | scream        |        0.57 |        57 |            0.7971 |         0.9649 |     0.873  |               69 |            0.806  |         0.9474 |     0.871  |    -0.002  |        -0.0175 |            0.0089 |
| 5exit_nohint | thunderstorm  |        0.46 |        49 |            0      |         0      |     0      |                6 |            0.2727 |         0.0612 |     0.1    |     0.1    |         0.0612 |            0.2727 |
| 5exit_nohint | wind          |        0.33 |        51 |            0.5625 |         0.5294 |     0.5455 |               48 |            0.4595 |         0.6667 |     0.544  |    -0.0015 |         0.1373 |           -0.103  |

---

## ML-A14. Tuned thresholds

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

## ML-A15. Research finding notes

### Fixed threshold under-predicts positives

The fixed `0.5` threshold produced average predicted label counts of `1.0478` for 3-exit and `1.0112` for 5-exit, while the true test average was `1.5618`. This means the models were conservative and missed labels.

### Tuning improves F1 by increasing recall

Threshold tuning increased average predicted labels to `1.8652` for the 3-exit model and `1.7022` for the 5-exit model. This explains the improvement in macro-F1 and micro-F1, but also explains the increase in hamming loss.

### 5-exit Exit 4 is important

Exit 4 of the 5-exit model reached tuned macro-F1 `0.6281`, compared with final Exit 5 tuned macro-F1 `0.6152`. This means the best macro-F1 exit in the 5-exit model is not the deepest exit.

### Thunderstorm remains unresolved

Thunderstorm improved after threshold tuning but remained weak:

```text
3-exit thunderstorm F1: 0.0784 → 0.2526
5-exit thunderstorm F1: 0.0000 → 0.1000
```

This should be treated as a dataset/training issue, not only a threshold issue.

---

## ML-A16. Commands


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

## ML-A17. Generated summary files

```text
runs_multilabel\summary_thresholdsll_exit_metrics.csv
runs_multilabel\summary_thresholdsll_exit_metrics.md
runs_multilabel\summary_thresholdsinal_exit_comparison.csv
runs_multilabel\summary_thresholdsinal_exit_comparison.md
runs_multilabel\summary_thresholdsest_exit_comparison.csv
runs_multilabel\summary_thresholdsest_exit_comparison.md
runs_multilabel\summary_thresholdsinal_exit_per_label.csv
runs_multilabel\summary_thresholdsinal_exit_per_label.md
runs_multilabel\summary_thresholdsinal_exit_thresholds.csv
runs_multilabel\summary_thresholdsinal_exit_thresholds.md
runs_multilabel\summary_thresholds\README_TABLES.md
```

---

# Preserved previous appendix

The content below is preserved from the uploaded `APPENDIX.md`. The v0.1 threshold-tuned appendix above supersedes earlier v0 fixed-threshold-only multi-label notes where they overlap.

# Appendix — `kexit_cclass_greedy_multi-label_v0`

This appendix records the first controlled multi-label baseline for NeuroAccuExit-ASHADIP.

The baseline branch name is:

```text
kexit_cclass_greedy_multi-label_v0
```

---

# ML-A0. Branch status

| Item | Value |
|---|---|
| Working local branch | `kexit_cclass_greedy_multi-label` |
| Baseline branch to save | `kexit_cclass_greedy_multi-label_v0` |
| Task type | Multi-label audio classification |
| Label count | 10 |
| Training source | Clean seed + synthetic mixed audio |
| First trained model | `multilabel_5exit_nohint` |
| Hint passing | Disabled |
| Status | Functional v0 baseline |

---

# ML-A1. Multi-label target representation

| Audio content | Multi-class representation | Multi-label representation |
|---|---|---|
| rain only | `rain` | `rain=1` |
| thunderstorm only | `thunderstorm` | `thunderstorm=1` |
| rain + thunderstorm | `rain_thunderstorm` | `rain=1`, `thunderstorm=1` |
| traffic + gunshot | requires artificial combined class | `road_traffic=1`, `gun_shot=1` |

---

# ML-A2. Label list

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

# ML-A3. Clean seed split counts

| Split | Total | car_crash | conversation | engine_idling | fireworks | gun_shot | rain | road_traffic | scream | thunderstorm | wind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 724 | 64 | 57 | 32 | 57 | 155 | 66 | 80 | 106 | 38 | 69 |
| Val | 155 | 14 | 12 | 7 | 12 | 33 | 14 | 17 | 23 | 8 | 15 |
| Test | 156 | 14 | 12 | 7 | 13 | 33 | 15 | 17 | 22 | 8 | 15 |

`.m4a` files were excluded from the manifest because the local environment could not decode them.

---

# ML-A4. Synthetic mixture generation settings

| Setting | Value |
|---|---:|
| Number of labels per synthetic clip | 2 |
| Train mixtures | 1000 |
| Val mixtures | 200 |
| Test mixtures | 200 |
| Sample rate | 16000 Hz |
| Clip duration | 1.0 s |
| Gain range | -6 dB to 0 dB |
| Seed | 42 |
| Output format | WAV |

---

# ML-A5. Synthetic mixture counts

| Split | Synthetic mixtures |
|---|---:|
| Train | 1000 |
| Val | 200 |
| Test | 200 |
| **Total** | **1400** |

---

# ML-A6. Synthetic positive label counts

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

---

# ML-A7. Combined manifest counts

| Split | Rows |
|---|---:|
| Train | 1724 |
| Val | 355 |
| Test | 356 |
| **Total** | **2435** |

---

# ML-A8. Feature extraction settings

| Setting | Value |
|---|---:|
| Sample rate | 16000 Hz |
| Clip duration | 1.0 s |
| Mel bins | 64 |
| FFT size | 1024 |
| Window length | 25 ms |
| Hop length | 10 ms |
| CMVN | Enabled |
| Loaded tensor shape | `[batch, 1, 64, 101]` |
| Target tensor shape | `[batch, 10]` |

---

# ML-A9. Feature dataset summary

| Item | Value |
|---|---:|
| Total rows | 2435 |
| Clean rows | 1035 |
| Synthetic rows | 1400 |
| Train rows | 1724 |
| Val rows | 355 |
| Test rows | 356 |

---

# ML-A10. Label-positive counts after feature extraction

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

# ML-A11. Training setup for `multilabel_5exit_nohint`

| Setting | Value |
|---|---|
| Model | TinyAudioCNN + ExitNet |
| Exits | 5 |
| Tap blocks | `1,2,3,4` |
| Labels | 10 |
| Loss | BCEWithLogitsLoss |
| Activation for prediction | Sigmoid |
| Threshold | 0.5 global |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Device | CPU |
| Hint passing | Disabled |
| Best epoch | 37 |
| Best validation final-exit macro-F1 | 0.5438 |

---

# ML-A12. Test metrics by exit

| Exit | Macro-F1 | Micro-F1 | Exact match | Hamming loss | Interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | 0.0068 | 0.0072 | 0.0056 | 0.1556 | Too shallow; almost no useful label prediction |
| 2 | 0.2411 | 0.3476 | 0.1601 | 0.1287 | Weak but learning |
| 3 | 0.3708 | 0.4660 | 0.1994 | 0.1191 | Moderate |
| 4 | 0.5135 | 0.5686 | 0.2697 | 0.1104 | Strong early-exit candidate |
| 5 | **0.5302** | **0.5852** | **0.3062** | **0.1067** | Best final prediction |

---

# ML-A13. Final-exit per-label test metrics

| Label | Precision | Recall | F1 | Support | Predicted positives | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| car_crash | 0.6316 | 0.5106 | 0.5647 | 47 | — | Moderate |
| conversation | 0.9815 | 0.8833 | **0.9298** | 60 | 54 | Excellent |
| engine_idling | 0.7500 | 0.3673 | 0.4932 | — | — | Precise but misses many |
| fireworks | 0.8421 | 0.3077 | 0.4507 | — | — | High precision, low recall |
| gun_shot | 0.7458 | 0.5789 | 0.6519 | — | — | Good/moderate |
| rain | 0.8800 | 0.4231 | 0.5714 | — | — | High precision, low recall |
| road_traffic | 0.5000 | 0.1429 | 0.2222 | — | — | Weak |
| scream | 0.7971 | 0.9649 | **0.8730** | — | — | Very good |
| thunderstorm | 0.0000 | 0.0000 | 0.0000 | 49 | 6 | Failed in v0 |
| wind | 0.5625 | 0.5294 | 0.5455 | — | — | Moderate |

Full support/predicted-positive values should be copied from `metrics.json` before final paper submission.

---

# ML-A14. Main findings

1. The multi-label data pipeline works end-to-end.
2. The model learns useful multi-label predictions from clean seed + synthetic mixture data.
3. Exit performance improves with depth.
4. Exit 4 is close to exit 5, making it promising for future early-exit inference.
5. Conversation and scream are strong labels.
6. Thunderstorm and road_traffic require threshold/data-quality improvement.
7. A single global threshold of 0.5 is not sufficient.
8. Hint passing should not be added until threshold tuning and no-hint baselines are stable.

---

# ML-A15. Next implementation checklist

| Priority | Task | Why |
|---:|---|---|
| 1 | Tune per-label thresholds | Fix weak labels like thunderstorm and road_traffic |
| 2 | Re-evaluate test with tuned thresholds | Establish stronger v0.1 result |
| 3 | Run 3-exit no-hint baseline | Fair comparison with 5-exit baseline |
| 4 | Add multi-label policy evaluation | Needed for early-exit compute/accuracy results |
| 5 | Add sigmoid-aware hint passing | Multi-label hint passing differs from softmax hint passing |
| 6 | Create small real mixed test set | Test synthetic-to-real generalisation |
| 7 | Add mAP / label-wise AUC | Better probability-quality reporting |

---

# ML-A16. Git commands for saving this baseline

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

Upload later with:

```powershell
git push -u origin kexit_cclass_greedy_multi-label
git push -u origin kexit_cclass_greedy_multi-label_v0
```

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
