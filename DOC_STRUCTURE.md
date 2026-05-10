# Documentation Structure — `kexit_cclass_greedy_multi-label_pos-weight`

This document defines the thesis/report writing structure for the active branch:

```text
kexit_cclass_greedy_multi-label_pos-weight
```

This document is intentionally focused only on the active positive-weight multi-label branch.

---

## Current research story

The branch now contains a complete multi-label study:

```text
clean seed data
→ synthetic two-label mixtures
→ feature extraction
→ 3-exit and 5-exit no-hint training
→ threshold tuning
→ positive-label-weighting ablation
→ 4-model comparison
```

Research question:

> Can a K-exit audio network be adapted to multi-label audio tagging, and can threshold tuning or positive label weighting improve label-balanced performance while retaining useful early-exit behaviour?

---

## Chapter 1 — Motivation

Real environmental audio can contain overlapping events. Use examples such as:

```text
rain + thunderstorm
road_traffic + gun_shot
fireworks + conversation
wind + rain
```

Suggested wording:

> Environmental sound scenes often contain multiple simultaneous events. A one-second recording may include both rain and thunder, or traffic and a gunshot. To model this overlap, this branch reformulates the classification problem as audio tagging, where each label is predicted independently using sigmoid outputs.

---

## Chapter 2 — Dataset design

Sections to include:

1. Clean seed data.
2. Synthetic two-label mixtures.
3. Combined multi-hot manifest.
4. Feature cache.
5. Future real mixed test set.

Required tables:

| Table | Title | Purpose |
|---:|---|---|
| ML-1 | Label list | Stable label order |
| ML-2 | Clean seed split counts | Data availability after excluding `.m4a` |
| ML-3 | Synthetic mixture settings | Reproducibility |
| ML-4 | Synthetic positive label counts | Synthetic balance |
| ML-5 | Combined manifest counts | Final data size |
| ML-6 | Feature extraction settings | Input representation |
| ML-7 | Train positive counts | Explains pos-weight calculation |

---

## Chapter 3 — Preprocessing and feature pipeline

| Stage | Script | Output |
|---|---|---|
| Rename clean seed files | `scripts/rename_wavs_by_class.py` | class-prefixed filenames |
| Build seed manifest | `scripts/build_multilabel_seed_manifest.py` | `clean_seed_manifest.csv`, `labels.json` |
| Create synthetic mixtures | `scripts/create_synthetic_multilabel_mixtures.py` | synthetic WAVs and manifest |
| Extract log-mel features | `scripts/extract_multilabel_features.py` | `.npy` feature cache |
| Load dataset | `data/datasets_multilabel.py` | `[x, y_multi_hot]` tensors |

Methodological point:

> Synthetic mixtures are generated after train/validation/test splitting. A validation or test mixture never uses source files from the training split.

---

## Chapter 4 — Model formulation

| Component | Setting |
|---|---|
| Backbone | TinyAudioCNN |
| Wrapper | ExitNet |
| Output at each exit | `[batch, 10]` logits |
| Activation | Sigmoid |
| Target | Multi-hot vector |
| Loss | BCEWithLogitsLoss |
| Initial decision rule | Fixed threshold 0.5 |
| Improved decision rule | Tuned per-label thresholds |

K-exit settings:

| Configuration | Tap blocks | Exits |
|---|---|---:|
| 3-exit | `1,3` | 3 |
| 5-exit | `1,2,3,4` | 5 |

### Positive label weighting

Positive label weighting increases the loss penalty for missed active labels. It helps recall-sensitive labels, but may increase false positives.

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

## Chapter 5 — Results

Experiment sequence:

| Experiment | Purpose |
|---|---|
| 3-exit no-hint fixed 0.5 | compact baseline |
| 5-exit no-hint fixed 0.5 | deeper baseline |
| 3-exit no-hint tuned | threshold calibration |
| 5-exit no-hint tuned | threshold calibration |
| 3-exit pos-weight fixed 0.5 | weighted-loss ablation |
| 5-exit pos-weight fixed 0.5 | weighted-loss ablation |
| 3-exit pos-weight tuned | weighted loss + threshold tuning |
| 5-exit pos-weight tuned | weighted loss + threshold tuning |

Core final-exit table:

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


Best tuned exits:

| Model | Best tuned exit | Pos-weight | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg predicted labels |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `3exit_nohint_posweight` | 3 | Yes | **0.6530** | **0.6427** | 0.6580 | 0.2978 | 0.1256 | 1.9522 |
| `5exit_nohint_posweight` | 4 | Yes | 0.6433 | 0.6349 | **0.6645** | 0.2809 | 0.1292 | 1.9775 |
| `3exit_nohint` | 3 | No | 0.6301 | 0.6361 | 0.6576 | 0.2725 | **0.1247** | 1.8652 |
| `5exit_nohint` | 4 | No | 0.6281 | 0.6291 | 0.6475 | **0.3090** | 0.1258 | 1.8315 |


---

## Chapter 6 — Discussion

### Threshold tuning

Fixed thresholding under-predicts active labels:

```text
3exit fixed avg predicted labels = 1.0478
5exit fixed avg predicted labels = 1.0112
true avg labels                 = 1.5618
```

### Positive weighting

Strongest macro-F1:

```text
3exit_nohint_posweight tuned macro-F1 = 0.6530
```

Positive weighting increases predicted label count:

```text
3exit pos-weight tuned avg_pred_labels = 1.9522
5exit pos-weight tuned avg_pred_labels = 2.0758
true avg labels                        = 1.5618
```

### Early-exit result

```text
5exit_nohint Exit 4 tuned macro-F1      = 0.6281
5exit_posweight Exit 4 tuned macro-F1   = 0.6433
```

This should be written as one of the central findings.

### Thunderstorm

```text
3exit no-weight tuned thunderstorm F1   = 0.2526
3exit pos-weight tuned thunderstorm F1  = 0.3485
```

Positive weighting improves thunderstorm but does not fully solve it.

---

## Chapter 7 — Limitations and future work

| Limitation | Explanation | Planned fix |
|---|---|---|
| Synthetic mixtures only | Controlled mixtures may not match real overlap | Verified real mixed test set |
| Single seed | Robustness not proven | Multi-seed repeats |
| Pos-weight cap may be high | `5.0` can over-predict | Try `--pos_weight_max 3.0` |
| Static exit evaluation only | No dynamic stopping yet | Multi-label early-exit policy |
| No sigmoid-aware hint passing | Hint not adapted yet | Add later |
| Thunderstorm weak | Likely data/acoustic issue | Inspect and improve thunderstorm clips |
| Calibration not fully studied | Thresholds vary by model | Add mAP/AUC/calibration diagnostics |

Suggested thesis wording:

> The multi-label experiments show that the K-exit audio architecture can be adapted to audio tagging by replacing softmax classification with sigmoid outputs and BCEWithLogitsLoss. Fixed global thresholding under-detects active labels, while per-label threshold tuning substantially improves macro-F1. Positive label weighting further improves label-balanced performance, with the 3-exit positive-weighted model achieving the strongest final-exit macro-F1 of 0.6530. However, positive weighting also increases predicted label counts, demonstrating a recall–precision trade-off. In the 5-exit model, Exit 4 consistently achieves the best tuned macro-F1, suggesting that intermediate exits can provide useful compute-adaptive decision points for multi-label audio tagging.

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

