# Documentation Structure — `kexit_multi-label_greedy_EE`

This document defines the thesis/report writing structure for the active branch:

```text
kexit_multi-label_greedy_EE
```

This branch extends the multi-label BCE/sigmoid baseline into a **dynamic neural network / early-exit study**. The core contribution is the evaluation of a sigmoid-aware greedy label-set stability policy that measures prediction quality, exit distribution, average exit depth, and estimated depth-compute saving.

---

## Current research story

```text
clean seed data
→ synthetic two-label mixtures
→ log-mel feature extraction
→ 3-exit and 5-exit multi-label training
→ per-exit/per-label threshold tuning
→ static exit-quality comparison
→ greedy label-set stability policy evaluation
→ accuracy-efficiency trade-off analysis
```

Research question:

> Can a K-exit audio network perform multi-label audio tagging while supporting useful dynamic early-exit behaviour, and how do policy settings such as `min_exit` and `stable_k` affect the trade-off between macro-F1 and estimated compute saving?

---

## Chapter 1 — Motivation

Environmental audio often contains overlapping sound events. A one-second clip may contain more than one meaningful label, such as:

```text
rain + thunderstorm
road_traffic + gun_shot
fireworks + conversation
wind + rain
```

Suggested wording:

> Environmental sound scenes often contain simultaneous events. A recording may include both rain and thunder, or road traffic and a gunshot. To model this overlap, the task is formulated as multi-label audio tagging, where each label is predicted independently using sigmoid outputs. The early-exit objective then becomes more complex than in single-label classification because the system must decide whether an entire predicted label set is stable enough to stop computation.

---

## Chapter 2 — Dataset design

Sections to include:

1. Clean seed data.
2. Synthetic two-label mixtures.
3. Combined multi-hot manifest.
4. Feature cache.
5. Train/validation/test split.
6. Future real mixed test set.

Required tables:

| Table | Title | Purpose |
|---:|---|---|
| ML-1 | Label list | Stable label order |
| ML-2 | Clean seed split counts | Data availability after excluding unsupported formats |
| ML-3 | Synthetic mixture settings | Reproducibility |
| ML-4 | Synthetic positive label counts | Synthetic balance |
| ML-5 | Combined manifest counts | Final data size |
| ML-6 | Feature extraction settings | Input representation |
| ML-7 | Train positive counts | Explains positive-label weighting |

Methodological point:

> Synthetic mixtures are generated after train/validation/test splitting. A validation or test mixture never uses source files from the training split.

---

## Chapter 3 — Preprocessing and feature pipeline

| Stage | Script | Output |
|---|---|---|
| Rename clean seed files | `scripts/rename_wavs_by_class.py` | class-prefixed filenames |
| Build seed manifest | `scripts/build_multilabel_seed_manifest.py` | `clean_seed_manifest.csv`, `labels.json` |
| Create synthetic mixtures | `scripts/create_synthetic_multilabel_mixtures.py` | synthetic WAVs and manifest |
| Extract log-mel features | `scripts/extract_multilabel_features.py` | `.npy` feature cache |
| Load dataset | `data/datasets_multilabel.py` | `[x, y_multi_hot]` tensors |

Feature settings to report:

| Item | Value |
|---|---:|
| Sample rate | 16000 Hz |
| Clip length | 1.0 s |
| Feature type | log-mel |
| Mel bands | 64 |
| FFT size | 1024 |
| Window | 25 ms |
| Hop | 10 ms |
| CMVN | enabled |
| Input tensor | `[batch, 1, 64, 101]` |
| Target tensor | `[batch, 10]` |

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
| Improved decision rule | Tuned per-exit/per-label thresholds |
| Dynamic policy | Greedy label-set stability |

K-exit settings:

| Configuration | Tap blocks | Exits | Purpose |
|---|---|---:|---|
| 3-exit | `1,3` | 3 | Compact Tiny baseline |
| 5-exit | `1,2,3,4` | 5 | More intermediate exit opportunities |

Positive label weighting:

| Setting | Value |
|---|---:|
| `--use_pos_weight` | enabled for pos-weight variants |
| `--pos_weight_max` | 20.0 |

Important note:

> The structured greedy-EE experiments use `--pos_weight_max 20.0`. These results should not be numerically mixed with earlier positive-weight runs that used a different cap.

---

## Chapter 5 — Greedy label-set stability policy

For exit `e`, the model produces sigmoid probabilities:

```text
p_e = sigmoid(logits_e)
```

A multi-label prediction is created using tuned per-exit/per-label thresholds:

```text
y_hat_e[label] = 1 if p_e[label] >= tau_e,label else 0
```

The policy scans exits from `min_exit` onward and stops when the predicted label set is stable for `stable_k` consecutive considered exits. If no stable point is found, it falls back to the final exit.

Policy variants:

| Policy | Setting | Purpose |
|---|---|---|
| Policy 001 | `min_exit=2, stable_k=2` | Conservative policy; ignores Exit 1 |
| Policy 002 | `min_exit=1, stable_k=2` | Fair early-exit test for 3-exit models |
| Policy 003 | `min_exit=1, stable_k=1` | Aggressive ablation |
| Policy 004 | `min_exit=1, stable_k=2, allow_empty_stop=True` | Empty-label stopping ablation |

There is no `min_exit=0` because exits are numbered from `1`.

---

## Chapter 6 — Experiment store and reproducibility layout

```text
runs_multilabel/
│
├─ training/
│  ├─ multilabel_nohint/
│  │  ├─ multilabel_nohint_001_3exit_YYYYMMDD_HHMMSS/
│  │  └─ multilabel_nohint_002_5exit_YYYYMMDD_HHMMSS/
│  │
│  └─ multilabel_posweight/
│     ├─ multilabel_posweight_001_3exit_YYYYMMDD_HHMMSS/
│     └─ multilabel_posweight_002_5exit_YYYYMMDD_HHMMSS/
│
├─ policy_eval/
│  └─ multilabel_greedy_policy/
│     ├─ multilabel_greedy_policy_001_minexit2_stable2/
│     ├─ multilabel_greedy_policy_002_minexit1_stable2/
│     ├─ multilabel_greedy_policy_003_minexit1_stable1/
│     └─ multilabel_greedy_policy_004_minexit1_stable2_allowempty/
│
└─ summary/
   └─ threshold_summary_001_static_tuned/
```

This structure separates training checkpoints from policy-evaluation experiments because multiple policies can be evaluated against the same trained model.

---

## Chapter 7 — Results structure

### 7.1 Static threshold-tuned results

Required tables:

| Table | Purpose |
|---|---|
| Static final-exit comparison | Compare fixed 0.5 vs tuned thresholds |
| Best static exit per model | Identify the strongest standalone exit |
| Static per-exit quality | Diagnose early-exit reliability |
| Per-label F1 | Show label-specific strengths and weaknesses |

Main static findings:

```text
5exit_posweight Exit 4 macro-F1 = 0.6538
3exit_posweight final macro-F1  = 0.6451
```

Interpretation:

> The strongest standalone 5-exit head is Exit 4, not the final exit. This supports compute-adaptive inference because an intermediate head can be shallower and still highly competitive.

### 7.2 Dynamic policy results

Required tables:

| Table | Purpose |
|---|---|
| Selected policy comparison | Compare Policies 001–004 |
| Policy 001 vs Policy 002 | Main fair comparison for 3-exit early-exit behaviour |
| Exit distribution | Show how many samples stop at each exit |
| Compute-depth unit table | Estimate dynamic computation saved |
| Full policy sweep | Identify best practical trade-offs |

Main dynamic findings:

```text
Policy 001: 5-exit models save about 14–16% depth compute.
Policy 002: 3-exit models become true early-exit models but save only about 2.5–3.8%.
Policy 003: immediate Exit-1 stopping saves 66–80% but collapses prediction quality.
Best practical trade-off: 5exit_posweight, min_exit=3, stable_k=2.
```

Headline dynamic result:

```text
Model: 5exit_posweight
Policy: min_exit=3, stable_k=2
Macro-F1: 0.6449
Samples-F1: 0.6690
Avg exit depth: 4.5337 / 5
Estimated depth-compute saved: 9.33%
```

---

## Chapter 8 — Discussion

### Threshold tuning

Fixed thresholding under-predicts active labels in no-hint models:

```text
3exit fixed avg predicted labels = 1.0478
5exit fixed avg predicted labels = 1.0112
true avg labels                 = 1.5618
```

Per-label threshold tuning is therefore required for fair multi-label evaluation.

### Positive weighting

Positive weighting improves label-balanced performance but can increase false positives. This is visible through higher predicted-label counts and, in some cases, higher hamming loss.

### Early-exit reliability

Exit 1 is currently weak and over-predicts labels:

```text
True avg labels per test sample = 1.5618
Exit 1 avg predicted labels     = 3.7 to 4.35
```

This explains why aggressive early stopping performs poorly.

### Policy 001 vs Policy 002

Policy 001 is conservative and safe, but it structurally prevents 3-exit models from saving compute. Policy 002 enables Exit 1 and allows 3-exit models to stop at Exit 2 when Exit 1 and Exit 2 agree.

Recommended wording:

> Allowing Exit 1 makes the compact 3-exit model behave as a true early-exit network, but savings remain modest because only a small subset of samples produces stable label sets before the final exit.

### Best practical trade-off

Recommended wording:

> The strongest practical trade-off is obtained by the 5-exit positive-weighted model with `min_exit=3, stable_k=2`, which preserves high macro-F1 while still saving estimated depth compute.

---

## Chapter 9 — Limitations and future work

| Limitation | Explanation | Planned fix |
|---|---|---|
| Synthetic mixtures only | Controlled mixtures may not match real overlap | Verified real mixed test set |
| Single seed | Robustness not proven | Multi-seed repeats |
| Depth units are approximate | Compute saving is not measured FLOPs/latency | Add FLOPs and hardware latency profiling |
| Exit 1 is weak | It over-predicts labels and hurts aggressive early stopping | Strengthen early-exit supervision |
| Positive weighting may over-predict | `pos_weight_max=20.0` can increase false positives | Try lower caps such as 3.0 or 5.0 under the same structure |
| No sigmoid-aware hint passing | Existing hint logic was designed for softmax-style outputs | Add sigmoid-aware hint passing later |
| Calibration not fully studied | Thresholds vary by exit and label | Add mAP, AUC, calibration/error diagnostics |

Suggested thesis wording:

> The multi-label greedy early-exit experiments show that the NeuroAccuExit architecture can support dynamic audio tagging, but the usefulness of early exits depends strongly on exit reliability. Conservative label-set stability preserves quality and enables compute savings in 5-exit models, while compact 3-exit models require Exit 1 to be enabled before any early stopping is possible. However, Exit 1 currently over-predicts labels, so immediate stopping leads to unacceptable degradation. The strongest practical result comes from later stable exits, particularly the 5-exit positive-weighted model with `min_exit=3, stable_k=2`.

---

## Chapter 10 — Next update plan

The next branch should focus on strengthening early exits before adding hint passing.

Recommended next branch:

```text
kexit_multi-label_EE_lossweight
```

Research question:

> Can stronger early-exit supervision improve Exit 1 and Exit 2 enough to increase compute saving without causing major macro-F1 degradation?

Candidate loss-weight settings:

```text
3-exit:
[0.6, 0.6, 1.0]
[0.8, 0.6, 1.0]

5-exit:
[0.6, 0.6, 0.7, 0.9, 1.0]
[0.8, 0.7, 0.7, 0.9, 1.0]
```

Primary comparison against this branch:

```text
Policy 001: min_exit=2, stable_k=2
Policy 002: min_exit=1, stable_k=2
Best 5-exit sweep: min_exit=3, stable_k=2
```
