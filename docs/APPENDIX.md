# Appendix — `kexit_cclass_greedy_multi-label_pos-weight`

This appendix records the detailed results for the active branch:

```text
kexit_cclass_greedy_multi-label_pos-weight
```

It includes no-weight threshold tuning and the positive-label-weighting ablation.

---

## ML-A1. Branch and artifact status

| Item | Value |
|---|---|
| Working branch | `kexit_cclass_greedy_multi-label_pos-weight` |
| Active branch | `kexit_cclass_greedy_multi-label_pos-weight` |
| Task | Multi-label audio tagging |
| Labels | 10 |
| Input | 1-second log-mel windows |
| Feature shape | `[batch, 1, 64, 101]` |
| Target shape | `[batch, 10]` |
| Compared variants | 3-exit no-hint, 5-exit no-hint, 3-exit pos-weight, 5-exit pos-weight |
| Thresholding | Fixed 0.5 and tuned per-label thresholds |
| Status | 4-model comparison complete |

---

## ML-A2. Label list

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

## ML-A3. Clean seed split counts after excluding `.m4a`

| Split | Total | car_crash | conversation | engine_idling | fireworks | gun_shot | rain | road_traffic | scream | thunderstorm | wind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 724 | 64 | 57 | 32 | 57 | 155 | 66 | 80 | 106 | 38 | 69 |
| Val | 155 | 14 | 12 | 7 | 12 | 33 | 14 | 17 | 23 | 8 | 15 |
| Test | 156 | 14 | 12 | 7 | 13 | 33 | 15 | 17 | 22 | 8 | 15 |


---

## ML-A4. Synthetic mixture settings and counts

| Setting | Value |
|---|---:|
| Labels per synthetic clip | 2 |
| Train mixtures | 1000 |
| Validation mixtures | 200 |
| Test mixtures | 200 |
| Sample rate | 16000 Hz |
| Clip duration | 1.0 s |
| Gain range | -6 dB to 0 dB |
| Seed | 42 |
| Leakage rule | Each synthetic split uses only source files from same split |

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

## ML-A5. Combined manifest and feature summary

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

## ML-A6. Positive weights

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


---

## ML-A7. Training setup

| Variant | Tap blocks | Exits | Pos-weight | Loss weights | Best epoch | Best validation final-exit macro-F1 |
|---|---|---:|---|---|---:|---:|
| `multilabel_3exit_nohint` | `1,3` | 3 | No | `[0.3, 0.3, 1.0]` | 27 | 0.5469 |
| `multilabel_5exit_nohint` | `1,2,3,4` | 5 | No | `[0.3, 0.3, 0.6, 0.8, 1.0]` | 37 | 0.5438 |
| `multilabel_3exit_nohint_posweight` | `1,3` | 3 | Yes, cap 5.0 | `[0.3, 0.3, 1.0]` | 39 | 0.6251 |
| `multilabel_5exit_nohint_posweight` | `1,2,3,4` | 5 | Yes, cap 5.0 | `[0.3, 0.3, 0.6, 0.8, 1.0]` | 35 | 0.6199 |


---

## ML-A8. Final-exit comparison

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


---

## ML-A9. Best tuned exit per model

| Model | Best tuned exit | Pos-weight | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg predicted labels |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `3exit_nohint_posweight` | 3 | Yes | **0.6530** | **0.6427** | 0.6580 | 0.2978 | 0.1256 | 1.9522 |
| `5exit_nohint_posweight` | 4 | Yes | 0.6433 | 0.6349 | **0.6645** | 0.2809 | 0.1292 | 1.9775 |
| `3exit_nohint` | 3 | No | 0.6301 | 0.6361 | 0.6576 | 0.2725 | **0.1247** | 1.8652 |
| `5exit_nohint` | 4 | No | 0.6281 | 0.6291 | 0.6475 | **0.3090** | 0.1258 | 1.8315 |


---

## ML-A10. All exit-level test metrics

| Model | Exit | Threshold | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg predicted labels |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `3exit_nohint` | 1 | fixed 0.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1562 | — |
| `3exit_nohint` | 1 | tuned | 0.3802 | 0.3561 | — | 0.0112 | 0.3809 | — |
| `3exit_nohint` | 2 | fixed 0.5 | 0.3024 | 0.4091 | 0.3408 | 0.1910 | 0.1242 | — |
| `3exit_nohint` | 2 | tuned | 0.5570 | 0.5278 | — | 0.1601 | 0.1980 | — |
| `3exit_nohint` | 3 | fixed 0.5 | 0.5319 | 0.5920 | 0.5598 | 0.3034 | 0.1065 | 1.0478 |
| `3exit_nohint` | 3 | tuned | 0.6301 | 0.6361 | 0.6576 | 0.2725 | 0.1247 | 1.8652 |
| `3exit_posweight` | 1 | fixed 0.5 | 0.3901 | 0.4076 | 0.3901 | 0.0197 | 0.2997 | — |
| `3exit_posweight` | 1 | tuned | 0.4271 | 0.4020 | — | 0.0028 | 0.3410 | — |
| `3exit_posweight` | 2 | fixed 0.5 | 0.5552 | 0.5467 | 0.5841 | 0.1657 | 0.2031 | — |
| `3exit_posweight` | 2 | tuned | 0.5715 | 0.5680 | — | 0.2079 | 0.1598 | — |
| `3exit_posweight` | 3 | fixed 0.5 | 0.6422 | 0.6379 | 0.6663 | 0.2500 | 0.1368 | 2.2163 |
| `3exit_posweight` | 3 | tuned | 0.6530 | 0.6427 | 0.6580 | 0.2978 | 0.1256 | 1.9522 |
| `5exit_nohint` | 1 | fixed 0.5 | 0.0068 | 0.0072 | 0.0056 | 0.0056 | 0.1556 | — |
| `5exit_nohint` | 1 | tuned | 0.3879 | 0.3619 | — | 0.0028 | 0.3635 | — |
| `5exit_nohint` | 2 | fixed 0.5 | 0.2411 | 0.3476 | 0.2739 | 0.1601 | 0.1287 | — |
| `5exit_nohint` | 2 | tuned | 0.5020 | 0.4722 | — | 0.1067 | 0.2242 | — |
| `5exit_nohint` | 3 | fixed 0.5 | 0.3708 | 0.4660 | 0.3928 | 0.1994 | 0.1191 | — |
| `5exit_nohint` | 3 | tuned | 0.5723 | 0.5563 | — | 0.1910 | 0.1747 | — |
| `5exit_nohint` | 4 | fixed 0.5 | 0.5135 | 0.5686 | 0.5325 | 0.2697 | 0.1104 | — |
| `5exit_nohint` | 4 | tuned | 0.6281 | 0.6291 | 0.6475 | 0.3090 | 0.1258 | 1.8315 |
| `5exit_nohint` | 5 | fixed 0.5 | 0.5302 | 0.5852 | 0.5545 | 0.3062 | 0.1067 | 1.0112 |
| `5exit_nohint` | 5 | tuned | 0.6152 | 0.6454 | 0.6639 | 0.3343 | 0.1157 | 1.7022 |
| `5exit_posweight` | 1 | fixed 0.5 | 0.3887 | 0.4031 | 0.3880 | 0.0309 | 0.3020 | — |
| `5exit_posweight` | 1 | tuned | 0.4162 | 0.3964 | — | 0.0084 | 0.3362 | — |
| `5exit_posweight` | 2 | fixed 0.5 | 0.4655 | 0.4669 | 0.4796 | 0.0646 | 0.2604 | — |
| `5exit_posweight` | 2 | tuned | 0.5042 | 0.4653 | — | 0.0758 | 0.2531 | — |
| `5exit_posweight` | 3 | fixed 0.5 | 0.5495 | 0.5416 | 0.5768 | 0.1489 | 0.2087 | — |
| `5exit_posweight` | 3 | tuned | 0.5751 | 0.5600 | — | 0.2022 | 0.1730 | — |
| `5exit_posweight` | 4 | fixed 0.5 | 0.6244 | 0.6121 | 0.6466 | 0.2219 | 0.1545 | — |
| `5exit_posweight` | 4 | tuned | 0.6433 | 0.6349 | 0.6645 | 0.2809 | 0.1292 | 1.9775 |
| `5exit_posweight` | 5 | fixed 0.5 | 0.6159 | 0.5931 | 0.6319 | 0.2556 | 0.1626 | 2.4354 |
| `5exit_posweight` | 5 | tuned | 0.6232 | 0.6085 | 0.6341 | 0.2640 | 0.1424 | 2.0758 |


---

## ML-A11. Positive-weight final-exit per-label fixed-vs-tuned metrics

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


---

## ML-A12. Positive-weight tuned thresholds

| Model | Final exit | car_crash | conversation | engine_idling | fireworks | gun_shot | rain | road_traffic | scream | thunderstorm | wind |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3exit_posweight` | 3 | 0.61 | 0.47 | 0.75 | 0.42 | 0.77 | 0.88 | 0.63 | 0.51 | 0.47 | 0.41 |
| `5exit_posweight` | 5 | 0.43 | 0.45 | 0.57 | 0.64 | 0.74 | 0.46 | 0.43 | 0.65 | 0.57 | 0.58 |


---

## ML-A13. Key deltas

| Comparison | Before | After | Delta |
|---|---:|---:|---:|
| 3-exit no-weight fixed → tuned macro-F1 | 0.5319 | 0.6301 | +0.0982 |
| 5-exit no-weight fixed → tuned macro-F1 | 0.5302 | 0.6152 | +0.0850 |
| 3-exit no-weight tuned → pos-weight tuned macro-F1 | 0.6301 | 0.6530 | +0.0229 |
| 5-exit no-weight tuned → pos-weight tuned macro-F1 | 0.6152 | 0.6232 | +0.0080 |
| 3-exit no-weight tuned → pos-weight tuned exact match | 0.2725 | 0.2978 | +0.0253 |
| 5-exit no-weight tuned → pos-weight tuned exact match | 0.3343 | 0.2640 | -0.0703 |
| 3-exit no-weight tuned → pos-weight tuned hamming loss | 0.1247 | 0.1256 | +0.0009 |
| 5-exit no-weight tuned → pos-weight tuned hamming loss | 0.1157 | 0.1424 | +0.0267 |

---

## ML-A14. Research notes

1. Fixed `0.5` threshold under-predicts positives.
2. Per-label threshold tuning is required for fair multi-label comparison.
3. Positive weighting improves the compact 3-exit macro-F1 most clearly.
4. Positive weighting is too aggressive for the 5-exit model at cap `5.0` because it predicts `2.0758` labels/sample.
5. Exit 4 is the best tuned macro-F1 exit for both 5-exit settings.
6. Thunderstorm is improved but unresolved.

---

## ML-A15. Generated result folders

No-weight summary:

```text
runs_multilabel\summary_thresholds```

4-model positive-weight summary:

```text
runs_multilabel\summary_thresholds_posweight```

Important files:

```text
runs_multilabel\summary_thresholds_posweightll_exit_metrics.csv
runs_multilabel\summary_thresholds_posweightll_exit_metrics.md
runs_multilabel\summary_thresholds_posweight
inal_exit_comparison.csv
runs_multilabel\summary_thresholds_posweight
inal_exit_comparison.md
runs_multilabel\summary_thresholds_posweightest_exit_comparison.csv
runs_multilabel\summary_thresholds_posweightest_exit_comparison.md
runs_multilabel\summary_thresholds_posweight
inal_exit_per_label.csv
runs_multilabel\summary_thresholds_posweight
inal_exit_per_label.md
runs_multilabel\summary_thresholds_posweight
inal_exit_thresholds.csv
runs_multilabel\summary_thresholds_posweight
inal_exit_thresholds.md
runs_multilabel\summary_thresholds_posweight\README_TABLES.md
```

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
