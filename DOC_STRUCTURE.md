
# Documentation Structure — Multi-label Early-exit Audio Baseline v0.1

This document updates the writing structure for the local branch:

```text
kexit_cclass_greedy_multi-label
```

Recommended snapshot name:

```text
kexit_cclass_greedy_multi-label_v0.1_threshold_tuned
```

The previous v0 documentation recorded only the first 5-exit fixed-threshold run. The current v0.1 documentation must now include:

1. clean seed preparation,
2. synthetic multi-label mixture generation,
3. feature extraction,
4. multi-label dataset loading,
5. 3-exit no-hint training,
6. 5-exit no-hint training,
7. per-label threshold tuning,
8. summary-table generation,
9. detailed interpretation and future work.

---

## Updated project story

| Track | Task type | Output | Current status |
|---|---|---|---|
| Moth binary | Single-label binary | one class | Historical validated baseline |
| C-class audio | Single-label multi-class | one class | Prepared/grouped experiments preserved |
| Multi-label audio | Multi-label audio tagging | one or more labels | v0.1 threshold-tuned baseline completed |

The multi-label track should be framed as a controlled extension. It does not replace the C-class track. It answers a different question: can the K-exit architecture handle overlapping sounds when one segment may contain multiple labels?

---

## Recommended chapter placement

### Chapter 1 — Motivation

Explain why overlapping environmental audio needs multi-label modelling. Examples:

```text
rain + thunderstorm
road_traffic + gun_shot
fireworks + conversation
wind + rain
```

The key motivation is that a single-label classifier must either ignore overlap or create artificial combined classes. A multi-label classifier can represent the overlap directly.

### Chapter 2 — Dataset design

Separate the datasets clearly:

1. Original moth binary dataset.
2. Single-label C-class environmental dataset.
3. Clean seed multi-label dataset.
4. Synthetic mixed multi-label dataset.
5. Future manually verified real mixed test set.

Important wording:

> The multi-label v0.1 dataset avoids blind weak supervision from noisy long clips. Instead, it uses clean single-label seed clips and controlled synthetic two-label mixtures, preserving leakage-safe train/validation/test separation.

### Chapter 3 — Preprocessing pipeline

Add a dedicated subsection called:

```text
Multi-label seed and synthetic mixture preparation
```

Pipeline table:

| Stage | Script | Output |
|---|---|---|
| Rename clean seed files | `scripts/rename_wavs_by_class.py` | class-prefixed audio files |
| Build clean seed manifest | `scripts/build_multilabel_seed_manifest.py` | `clean_seed_manifest.csv`, `labels.json` |
| Create synthetic mixtures | `scripts/create_synthetic_multilabel_mixtures.py` | synthetic mixed WAVs and manifest |
| Extract log-mel features | `scripts/extract_multilabel_features.py` | `.npy` feature cache |
| Load multi-label dataset | `data/datasets_multilabel.py` | `[x, y_multi_hot]` tensors |
| Train multi-label model | `training/train_multilabel.py` | no-hint K-exit baseline |
| Tune thresholds | `scripts/tune_multilabel_thresholds.py` | per-label thresholds and comparison JSON |
| Summarise results | `scripts/summarize_multilabel_threshold_runs.py` | CSV/Markdown result tables |

### Chapter 4 — Model formulation

Use this comparison:

| Component | Single-label C-class | Multi-label v0.1 |
|---|---|---|
| Logits | `[B, C]` | `[B, L]` |
| Target | integer class ID | multi-hot vector |
| Activation | softmax | sigmoid |
| Loss | cross entropy | binary cross entropy with logits |
| Prediction | one class | label set |
| Threshold | greedy confidence tau | per-label sigmoid thresholds |
| Hint passing | softmax-aware | disabled until sigmoid-aware hint is implemented |

### Chapter 5 — Results

Recommended result order:

1. Dataset and feature summary.
2. 3-exit and 5-exit fixed-threshold baselines.
3. Per-label threshold tuning.
4. Final-exit comparison.
5. Best-exit comparison.
6. Per-label analysis.
7. Research findings and limitations.

---

## Required tables for Chapter 5

| Table | Title | Include |
|---:|---|---|
| ML-1 | Label list | 10 labels |
| ML-2 | Clean seed split counts | Train/val/test counts after excluding `.m4a` |
| ML-3 | Synthetic mixture settings | mix size, sample rate, gain, seed |
| ML-4 | Synthetic positive counts | per-label counts |
| ML-5 | Combined manifest and features | rows, shapes, clean/synthetic split |
| ML-6 | Training setup | 3-exit vs 5-exit |
| ML-7 | Fixed vs tuned final-exit metrics | macro/micro/samples/exact/hamming |
| ML-8 | Best tuned exit per model | highlights 5-exit Exit 4 |
| ML-9 | All exit metrics | fixed/tuned by exit |
| ML-10 | Per-label fixed-vs-tuned metrics | precision/recall/F1/threshold |
| ML-11 | Tuned thresholds | per-model, per-label |
| ML-12 | Limitations and next work | threshold, hint, real mixed data |

---

## Core result tables to include

### Final-exit comparison

| model        |   num_exits |   exit | threshold_mode   |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:-------------|------------:|-------:|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| 3exit_nohint |           3 |      3 | fixed_0p5        |     0.5319 |     0.592  |       0.5598 |        0.3034 |         0.1065 |            1.5618 |            1.0478 |
| 3exit_nohint |           3 |      3 | tuned            |     0.6301 |     0.6361 |       0.6576 |        0.2725 |         0.1247 |            1.5618 |            1.8652 |
| 5exit_nohint |           5 |      5 | fixed_0p5        |     0.5302 |     0.5852 |       0.5545 |        0.3062 |         0.1067 |            1.5618 |            1.0112 |
| 5exit_nohint |           5 |      5 | tuned            |     0.6152 |     0.6454 |       0.6639 |        0.3343 |         0.1157 |            1.5618 |            1.7022 |

### Best tuned exit per model

| model        |   num_exits |   exit | threshold_mode   |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:-------------|------------:|-------:|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| 3exit_nohint |           3 |      3 | tuned            |     0.6301 |     0.6361 |       0.6576 |        0.2725 |         0.1247 |            1.5618 |            1.8652 |
| 5exit_nohint |           5 |      4 | tuned            |     0.6281 |     0.6291 |       0.6475 |        0.309  |         0.1258 |            1.5618 |            1.8315 |

---

## Suggested thesis/report wording

> The multi-label v0.1 experiments demonstrate that the K-exit audio architecture can be adapted from single-label classification to multi-label audio tagging by replacing softmax and cross entropy with sigmoid outputs and BCEWithLogitsLoss. A fixed global threshold of 0.5 was found to under-detect several labels, especially rain and thunderstorm. Per-label threshold tuning substantially improved macro-F1, increasing the 3-exit final-exit score from 0.5319 to 0.6301 and the 5-exit final-exit score from 0.5302 to 0.6152. The 5-exit model produced a particularly interesting early-exit result: its tuned Exit 4 achieved macro-F1 0.6281, nearly matching the best 3-exit final exit. This suggests that intermediate exits may provide useful compute-adaptive decision points in multi-label audio classification.

---

## Reviewer-safe limitations

| Limitation | Transparent explanation | Planned response |
|---|---|---|
| Synthetic mixtures | Controlled mixtures may not match real overlap | Add verified real mixed test set |
| Single seed | Robustness not yet proven | Repeat with multiple seeds |
| No sigmoid-aware hint | Existing hint is softmax-designed | Add sigmoid probability hints later |
| No dynamic multi-label policy yet | Current result evaluates exits statically | Add threshold/stability-based early-exit policy |
| Thunderstorm weak | Could reflect data quality or overlap ambiguity | Inspect data and run pos_weight experiments |
| Very low thresholds | Indicates calibration weakness | Add calibration/mAP/AUC analysis |

---

## Next implementation order

1. Positive label weighting experiments:
   - `3exit_nohint_posweight`
   - `5exit_nohint_posweight`
2. Multi-label early-exit policy using sigmoid confidence and label-set stability.
3. mAP / per-label AUC reporting.
4. Sigmoid-aware hint passing.
5. Small manually verified real mixed test set.
6. Multi-seed robustness.


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

# Preserved previous documentation structure

The content below is preserved from the uploaded `DOC_STRUCTURE.md`. The v0.1 section above supersedes earlier v0 fixed-threshold-only multi-label notes where they overlap.

# Documentation Structure — `kexit_cclass_greedy_multi-label_v0`

This update adds a new documentation layer for the first **multi-label early-exit audio classification baseline**.

The branch to preserve this baseline is:

```text
kexit_cclass_greedy_multi-label_v0
```

This branch should be treated as a controlled baseline, not the final multi-label result.

---

## Updated project story

The project now has two related but separate experimental tracks:

| Track | Task type | Output type | Status |
|---|---|---|---|
| C-class track | Single-label multi-class | One class per segment | Existing prepared/grouped experiments |
| Multi-label track | Multi-label audio tagging | Any number of labels per segment | New v0 baseline |

The multi-label track should be framed as a future-work extension of the K-exit/C-class system, designed for overlapping real-world audio events.

---

## Recommended report/chapter placement

### Chapter 1 — Motivation

Add the motivation that real environmental audio often contains overlapping events. A single-label classifier is useful but limited when sounds co-occur.

### Chapter 2 — Dataset design

Separate the dataset discussion into:

1. **Single-label C-class dataset**
2. **Clean seed multi-label dataset**
3. **Synthetic mixed multi-label dataset**
4. **Future real mixed/weak-label dataset**

Key point:

> Multi-label v0 avoids blindly using noisy original folders. It uses a small manually verified clean seed subset and automatically generated synthetic mixtures to create controlled multi-label supervision.

### Chapter 3 — Preprocessing pipeline

Add a subsection:

```text
Multi-label seed and synthetic mixture preparation
```

Include these pipeline stages:

| Stage | Script | Output |
|---|---|---|
| Rename raw seed files | `scripts/rename_wavs_by_class.py` | Class-prefixed audio filenames |
| Build clean seed manifest | `scripts/build_multilabel_seed_manifest.py` | `clean_seed_manifest.csv`, `labels.json` |
| Create synthetic mixtures | `scripts/create_synthetic_multilabel_mixtures.py` | Synthetic mixed WAVs + manifest |
| Extract features | `scripts/extract_multilabel_features.py` | Log-mel `.npy` features |
| Load dataset | `data/datasets_multilabel.py` | Multi-hot target tensors |
| Train model | `training/train_multilabel.py` | Multi-label K-exit baseline |

### Chapter 4 — Model formulation

Add a distinction between single-label and multi-label heads.

| Component | Single-label C-class | Multi-label v0 |
|---|---|---|
| Final logits | `[batch, num_classes]` | `[batch, num_labels]` |
| Target | class ID | multi-hot vector |
| Activation | softmax | sigmoid |
| Loss | cross entropy | binary cross entropy with logits |
| Early-exit output | one class | label set |

The architecture still uses TinyAudioCNN + ExitNet. The output interpretation changes.

### Chapter 5 — Results

Add a new subsection:

```text
5.x Multi-label baseline v0
```

Report:

1. dataset counts,
2. synthetic mixture setup,
3. feature extraction settings,
4. 5-exit no-hint baseline results,
5. per-label F1,
6. limitations and next steps.

---

## Multi-label v0 result summary for Chapter 5

| Item | Value |
|---|---|
| Branch baseline | `kexit_cclass_greedy_multi-label_v0` |
| Task | 10-label audio tagging |
| Data | clean seed + synthetic two-label mixtures |
| Synthetic mixtures | 1000 train, 200 val, 200 test |
| Combined rows | 2435 |
| Feature shape | `[1, 64, 101]` |
| Model | 5-exit TinyAudioCNN + ExitNet |
| Taps | `1,2,3,4` |
| Hint passing | disabled |
| Loss | BCEWithLogitsLoss |
| Threshold | global 0.5 |
| Best epoch | 37 |
| Best validation final-exit macro-F1 | 0.5438 |
| Test final-exit macro-F1 | 0.5302 |
| Test final-exit micro-F1 | 0.5852 |
| Test exact match | 0.3062 |
| Test hamming loss | 0.1067 |

---

## Interpretation for writing

Use this wording:

> The first multi-label baseline confirms that the K-exit audio architecture can be trained under a multi-hot target formulation using sigmoid outputs and BCEWithLogitsLoss. The 5-exit no-hint baseline shows a clear depth-performance pattern, with final-exit macro-F1 reaching 0.5302 and exit 4 approaching the final exit at 0.5135. This suggests that early-exit computation saving may be possible in the multi-label setting. However, label-wise performance is uneven: conversation and scream are strong, while thunderstorm and road_traffic remain weak. The next methodological requirement is per-label threshold calibration before adding hint passing or stronger early-exit policies.

---

## Required tables for the multi-label section

| Table | Content | Location |
|---:|---|---|
| ML-v0.1 | Label list | README + Appendix |
| ML-v0.2 | Clean seed split availability | README + Appendix |
| ML-v0.3 | Synthetic mixture counts | README + Appendix |
| ML-v0.4 | Synthetic positive label counts | README + Appendix |
| ML-v0.5 | Combined manifest counts | README + Appendix |
| ML-v0.6 | Feature extraction settings | README + Appendix |
| ML-v0.7 | Test metrics by exit | README + Appendix |
| ML-v0.8 | Per-label final-exit metrics | README + Appendix |
| ML-v0.9 | Limitations and next updates | README + DOC_STRUCTURE |

---

## Next update plan after v0

### Step 1 — Per-label threshold tuning

Current results use a global threshold:

```text
threshold = 0.5
```

This is probably not suitable for all labels. For example:

```text
conversation may work well at 0.5 or higher
thunderstorm may need 0.2–0.3
road_traffic may need a lower threshold
```

Implement:

```text
scripts/tune_multilabel_thresholds.py
```

Expected outputs:

```text
runs_multilabel/<run_id>/thresholds_multilabel.json
runs_multilabel/<run_id>/test_metrics_thresholded.json
```

### Step 2 — 3-exit no-hint baseline

Run:

```text
multilabel_3exit_nohint
```

This is needed for a fair 3-exit vs 5-exit comparison.

### Step 3 — Multi-label early-exit policy

A multi-label greedy policy should not use softmax max-confidence. It should use sigmoid-aware confidence, for example:

```text
mean distance from 0.5
stable predicted label set between exits
per-label threshold satisfaction
```

### Step 4 — Sigmoid-aware hint passing

Existing hint passing should not be reused blindly because it was designed around single-label softmax predictions. Multi-label hint passing should use sigmoid probabilities and uncertainty statistics.

### Step 5 — Real mixed test set

Create a small manually verified real mixed test set to test whether synthetic mixture training transfers to real mixed audio.

---

## Reviewer-safe limitations

| Limitation | Reviewer-safe response |
|---|---|
| Synthetic mixtures may not perfectly match real mixed audio | v0 is a controlled baseline; real mixed evaluation is future work |
| Global threshold may be unfair to rare/weak labels | Per-label threshold tuning is the next planned ablation |
| Thunderstorm failed in v0 | This is reported transparently and motivates threshold/data-quality analysis |
| No hint passing yet | The no-hint baseline is necessary before adding sigmoid-aware hint passing |
| No 3-exit result in this baseline document | v0 records the first 5-exit result; 3-exit baseline is next |

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
