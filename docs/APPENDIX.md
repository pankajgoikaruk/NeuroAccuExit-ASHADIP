# Appendix — `kexit_multi-label_greedy_EE`

This appendix records the detailed protocol and extended tables for the active branch:

```text
kexit_multi-label_greedy_EE
```

This branch evaluates **multi-label greedy early-exit (EE)** behaviour using a sigmoid-aware label-set stability policy.

---

## ML-GEE-A1. Branch and artifact status

| Item | Value |
|---|---|
| Active branch | `kexit_multi-label_greedy_EE` |
| Base branch | `kexit_cclass_greedy_multi-label_policy` |
| Task | Multi-label audio tagging |
| Labels | 10 |
| Input | 1-second log-mel windows |
| Feature shape | `[batch, 1, 64, 101]` |
| Target shape | `[batch, 10]` |
| Main script | `scripts/multilabel_greedy_policy.py` |
| Compared variants | `3exit_nohint`, `5exit_nohint`, `3exit_posweight`, `5exit_posweight` |
| Positive-weight cap in structured runs | `20.0` |
| Thresholding | Tuned per-exit/per-label thresholds |
| Main policy | Greedy label-set stability |
| Dynamic metrics | Exit distribution, avg exit depth, depth units, estimated compute saved |
| Status | Static and dynamic 4-model comparison complete |

Important note:

> These structured greedy-EE results use `--pos_weight_max 20.0`. They should not be numerically mixed with earlier positive-weight experiments that used a different cap.

---

## ML-GEE-A2. Label list

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

## ML-GEE-A3. Combined manifest and feature summary

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

## ML-GEE-A4. Experiment store

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

Important generated files per policy/model folder:

```text
dynamic_early_exit_efficiency.csv
dynamic_early_exit_efficiency.md
static_per_exit_quality.csv
static_per_exit_quality.md
exit_distribution.csv
exit_distribution.md
compute_depth_units.csv
compute_depth_units.md
full_policy_sweep.csv
full_policy_sweep.md
selected_policy_per_label.csv
selected_policy_per_label.md
policy_results.json
policy_summary.md
```

Important static summary files:

```text
runs_multilabel\summary\threshold_summary_001_static_tuned\all_exit_metrics.csv
runs_multilabel\summary\threshold_summary_001_static_tuned\all_exit_metrics.md
runs_multilabel\summary\threshold_summary_001_static_tuned\final_exit_comparison.csv
runs_multilabel\summary\threshold_summary_001_static_tuned\final_exit_comparison.md
runs_multilabel\summary\threshold_summary_001_static_tuned\best_exit_comparison.csv
runs_multilabel\summary\threshold_summary_001_static_tuned\best_exit_comparison.md
runs_multilabel\summary\threshold_summary_001_static_tuned\final_exit_per_label.csv
runs_multilabel\summary\threshold_summary_001_static_tuned\final_exit_per_label.md
runs_multilabel\summary\threshold_summary_001_static_tuned\final_exit_thresholds.csv
runs_multilabel\summary\threshold_summary_001_static_tuned\final_exit_thresholds.md
runs_multilabel\summary\threshold_summary_001_static_tuned\README_TABLES.md
```

---

## ML-GEE-A5. Model variants

| Model | Tap blocks | Exits | Pos-weight | `pos_weight_max` | Purpose |
|---|---|---:|---|---:|---|
| `3exit_nohint` | `1,3` | 3 | No | — | Compact Tiny baseline |
| `5exit_nohint` | `1,2,3,4` | 5 | No | — | More dynamic exit opportunities |
| `3exit_posweight` | `1,3` | 3 | Yes | 20.0 | Compact recall-sensitive model |
| `5exit_posweight` | `1,2,3,4` | 5 | Yes | 20.0 | Dynamic recall-sensitive model |

---

## ML-GEE-A6. Greedy policy definitions

For each exit `e`, the model produces sigmoid probabilities:

```text
p_e = sigmoid(logits_e)
```

The predicted label set is obtained using tuned per-exit/per-label thresholds:

```text
y_hat_e[label] = 1 if p_e[label] >= tau_e,label else 0
```

The policy stops when the predicted label set is stable for `stable_k` consecutive considered exits.

| Policy | Setting | Interpretation |
|---|---|---|
| Policy 001 | `min_exit=2, stable_k=2, allow_empty_stop=False` | Conservative; ignores Exit 1 |
| Policy 002 | `min_exit=1, stable_k=2, allow_empty_stop=False` | Allows Exit 1; fair test for 3-exit early stopping |
| Policy 003 | `min_exit=1, stable_k=1, allow_empty_stop=False` | Aggressive; mostly stops at Exit 1 |
| Policy 004 | `min_exit=1, stable_k=2, allow_empty_stop=True` | Empty-label stopping ablation |

There is no `min_exit=0` because exits are numbered from `1`.

---

## ML-GEE-A7. Static final-exit comparison

| Model | Final Exit | Threshold | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `3exit_nohint` | 3 | fixed 0.5 | 0.5319 | 0.5920 | 0.5598 | 0.3034 | **0.1065** | 1.0478 |
| `3exit_nohint` | 3 | tuned | 0.6301 | 0.6361 | 0.6576 | 0.2725 | 0.1247 | 1.8652 |
| `3exit_posweight` | 3 | fixed 0.5 | 0.6356 | 0.6173 | 0.6499 | 0.2191 | 0.1525 | 2.4242 |
| `3exit_posweight` | 3 | tuned | **0.6451** | 0.6338 | 0.6600 | 0.2753 | 0.1272 | 1.9129 |
| `5exit_nohint` | 5 | fixed 0.5 | 0.5302 | 0.5852 | 0.5545 | 0.3062 | 0.1067 | 1.0112 |
| `5exit_nohint` | 5 | tuned | 0.6152 | **0.6454** | **0.6639** | **0.3343** | 0.1157 | 1.7022 |
| `5exit_posweight` | 5 | fixed 0.5 | 0.6058 | 0.5764 | 0.6224 | 0.2247 | 0.1775 | 2.6292 |
| `5exit_posweight` | 5 | tuned | 0.6384 | 0.6223 | 0.6489 | 0.2612 | 0.1419 | 2.1938 |

---

## ML-GEE-A8. Best static exit per model

| Model | Best Static Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5exit_posweight` | **4** | **0.6538** | **0.6536** | **0.6825** | **0.3315** | **0.1191** | 1.8764 |
| `3exit_posweight` | 3 | 0.6451 | 0.6338 | 0.6600 | 0.2753 | 0.1272 | 1.9129 |
| `3exit_nohint` | 3 | 0.6301 | 0.6361 | 0.6576 | 0.2725 | 0.1247 | 1.8652 |
| `5exit_nohint` | **4** | 0.6281 | 0.6291 | 0.6475 | 0.3090 | 0.1258 | 1.8315 |

Static conclusion:

```text
5exit_posweight Exit 4 macro-F1 = 0.6538
```

The best 5-exit static head is not the final exit. This supports the idea that intermediate exits can be useful compute-adaptive decision points.

---

## ML-GEE-A9. Static per-exit quality

| Model | Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `3exit_nohint` | 1 | 0.3802 | 0.3561 | 0.3560 | 0.0112 | 0.3809 | 4.3539 |
| `3exit_nohint` | 2 | 0.5570 | 0.5278 | 0.5586 | 0.1601 | 0.1980 | 2.6320 |
| `3exit_nohint` | 3 | **0.6301** | **0.6361** | **0.6576** | **0.2725** | **0.1247** | 1.8652 |
| `3exit_posweight` | 1 | 0.4268 | 0.4038 | 0.4047 | 0.0000 | 0.3508 | 4.3230 |
| `3exit_posweight` | 2 | 0.5727 | 0.5649 | 0.5841 | 0.1770 | 0.1705 | 2.3567 |
| `3exit_posweight` | 3 | **0.6451** | **0.6338** | **0.6600** | **0.2753** | **0.1272** | 1.9129 |
| `5exit_nohint` | 1 | 0.3879 | 0.3619 | 0.3565 | 0.0028 | 0.3635 | 4.1348 |
| `5exit_nohint` | 2 | 0.5020 | 0.4722 | 0.4805 | 0.1067 | 0.2242 | 2.6854 |
| `5exit_nohint` | 3 | 0.5723 | 0.5563 | 0.5893 | 0.1910 | 0.1747 | 2.3764 |
| `5exit_nohint` | 4 | **0.6281** | 0.6291 | 0.6475 | 0.3090 | 0.1258 | 1.8315 |
| `5exit_nohint` | 5 | 0.6152 | **0.6454** | **0.6639** | **0.3343** | **0.1157** | 1.7022 |
| `5exit_posweight` | 1 | 0.4116 | 0.3957 | 0.3913 | 0.0253 | 0.3191 | 3.7191 |
| `5exit_posweight` | 2 | 0.4777 | 0.4412 | 0.4392 | 0.0815 | 0.2455 | 2.8315 |
| `5exit_posweight` | 3 | 0.5932 | 0.5881 | 0.6202 | 0.2528 | 0.1542 | 2.1826 |
| `5exit_posweight` | 4 | **0.6538** | **0.6536** | **0.6825** | **0.3315** | **0.1191** | 1.8764 |
| `5exit_posweight` | 5 | 0.6384 | 0.6223 | 0.6489 | 0.2612 | 0.1419 | 2.1938 |

Early-exit diagnostic:

```text
True avg labels per test sample = 1.5618

Exit 1 avg predicted labels:
3exit_nohint      = 4.3539
3exit_posweight   = 4.3230
5exit_nohint      = 4.1348
5exit_posweight   = 3.7191
```

Exit 1 is currently weak and over-predicts labels. This explains why immediate Exit-1 stopping performs poorly.

---

## ML-GEE-A10. Selected policy experiments

| Experiment | Model | min_exit | stable_k | empty stop | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Exit Depth | Compute Saved |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Policy 001 | `3exit_nohint` | 2 | 2 | False | 0.6301 | 0.6361 | 0.6576 | 0.2725 | 0.1247 | 3.0000 / 3 | 0.00% |
| Policy 001 | `3exit_posweight` | 2 | 2 | False | **0.6451** | 0.6338 | 0.6600 | 0.2753 | 0.1272 | 3.0000 / 3 | 0.00% |
| Policy 001 | `5exit_nohint` | 2 | 2 | False | 0.6246 | 0.6340 | 0.6601 | 0.2978 | 0.1301 | 4.1994 / 5 | 16.01% |
| Policy 001 | `5exit_posweight` | 2 | 2 | False | 0.6321 | 0.6207 | **0.6636** | 0.2949 | 0.1421 | 4.2893 / 5 | 14.21% |
| Policy 002 | `3exit_nohint` | 1 | 2 | False | 0.6179 | 0.6160 | 0.6495 | 0.2612 | 0.1419 | 2.8848 / 3 | 3.84% |
| Policy 002 | `3exit_posweight` | 1 | 2 | False | 0.6293 | 0.6162 | 0.6530 | 0.2725 | 0.1396 | 2.9242 / 3 | 2.53% |
| Policy 002 | `5exit_nohint` | 1 | 2 | False | 0.6097 | 0.6112 | 0.6451 | 0.2837 | 0.1458 | 3.9551 / 5 | 20.90% |
| Policy 002 | `5exit_posweight` | 1 | 2 | False | 0.6260 | 0.6124 | 0.6580 | 0.2865 | 0.1497 | 4.0534 / 5 | 18.93% |
| Policy 003 | `3exit_nohint` | 1 | 1 | False | 0.3806 | 0.3569 | 0.3588 | 0.0140 | 0.3806 | 1.0028 / 3 | 66.57% |
| Policy 003 | `3exit_posweight` | 1 | 1 | False | 0.4268 | 0.4038 | 0.4047 | 0.0000 | 0.3508 | 1.0000 / 3 | 66.67% |
| Policy 003 | `5exit_nohint` | 1 | 1 | False | 0.3890 | 0.3627 | 0.3594 | 0.0056 | 0.3632 | 1.0028 / 5 | 79.94% |
| Policy 003 | `5exit_posweight` | 1 | 1 | False | 0.4116 | 0.3957 | 0.3913 | 0.0253 | 0.3191 | 1.0000 / 5 | 80.00% |
| Policy 004 | `3exit_nohint` | 1 | 2 | True | 0.6179 | 0.6160 | 0.6495 | 0.2612 | 0.1419 | 2.8848 / 3 | 3.84% |
| Policy 004 | `3exit_posweight` | 1 | 2 | True | 0.6293 | 0.6162 | 0.6530 | 0.2725 | 0.1396 | 2.9242 / 3 | 2.53% |
| Policy 004 | `5exit_nohint` | 1 | 2 | True | 0.6094 | 0.6107 | 0.6423 | 0.2809 | 0.1458 | 3.9410 / 5 | 21.18% |
| Policy 004 | `5exit_posweight` | 1 | 2 | True | 0.6244 | 0.6101 | 0.6477 | 0.2781 | 0.1497 | 4.0112 / 5 | 19.78% |

---

## ML-GEE-A11. Policy 001 vs Policy 002

This is the main fair comparison because it tests whether allowing Exit 1 makes the 3-exit Tiny model behave like a real early-exit model.

| Model | Policy 001 Macro-F1 | Policy 002 Macro-F1 | F1 Change | Policy 001 Saved | Policy 002 Saved | Saving Gain |
|---|---:|---:|---:|---:|---:|---:|
| `3exit_nohint` | 0.6301 | 0.6179 | -0.0122 | 0.00% | 3.84% | +3.84% |
| `3exit_posweight` | 0.6451 | 0.6293 | -0.0158 | 0.00% | 2.53% | +2.53% |
| `5exit_nohint` | 0.6246 | 0.6097 | -0.0149 | 16.01% | 20.90% | +4.89% |
| `5exit_posweight` | 0.6321 | 0.6260 | -0.0061 | 14.21% | 18.93% | +4.72% |

Interpretation:

```text
3exit_nohint:
  Exit 2 samples = 41 / 356
  compute saved  = 3.84%
  macro-F1 drop  = -0.0122

3exit_posweight:
  Exit 2 samples = 27 / 356
  compute saved  = 2.53%
  macro-F1 drop  = -0.0158
```

Conclusion:

> Allowing Exit 1 makes the compact 3-exit model behave as a true early-exit model, but savings are still modest because early label-set stability occurs for only a small proportion of samples.

---

## ML-GEE-A12. Exit distribution

| Experiment | Model | Exit 1 | Exit 2 | Exit 3 | Exit 4 | Exit 5 |
|---|---|---:|---:|---:|---:|---:|
| Policy 001 | `3exit_nohint` | 0 | 0 | 356 | — | — |
| Policy 001 | `3exit_posweight` | 0 | 0 | 356 | — | — |
| Policy 001 | `5exit_nohint` | 0 | 0 | 101 | 83 | 172 |
| Policy 001 | `5exit_posweight` | 0 | 0 | 59 | 135 | 162 |
| Policy 002 | `3exit_nohint` | 0 | 41 | 315 | — | — |
| Policy 002 | `3exit_posweight` | 0 | 27 | 329 | — | — |
| Policy 002 | `5exit_nohint` | 0 | 41 | 85 | 79 | 151 |
| Policy 002 | `5exit_posweight` | 0 | 47 | 33 | 130 | 146 |
| Policy 003 | `3exit_nohint` | 355 | 1 | 0 | — | — |
| Policy 003 | `3exit_posweight` | 356 | 0 | 0 | — | — |
| Policy 003 | `5exit_nohint` | 355 | 1 | 0 | 0 | 0 |
| Policy 003 | `5exit_posweight` | 356 | 0 | 0 | 0 | 0 |
| Policy 004 | `3exit_nohint` | 0 | 41 | 315 | — | — |
| Policy 004 | `3exit_posweight` | 0 | 27 | 329 | — | — |
| Policy 004 | `5exit_nohint` | 0 | 41 | 87 | 80 | 148 |
| Policy 004 | `5exit_posweight` | 0 | 47 | 40 | 131 | 138 |

---

## ML-GEE-A13. Best practical dynamic trade-offs from sweep

| Model | Best Practical Policy | Macro-F1 | Micro-F1 | Samples-F1 | Avg Depth | Compute Saved |
|---|---|---:|---:|---:|---:|---:|
| `3exit_nohint` | `min_exit=1, stable_k=2` | 0.6179 | 0.6160 | 0.6495 | 2.8848 / 3 | 3.84% |
| `3exit_posweight` | `min_exit=1, stable_k=2` | 0.6293 | 0.6162 | 0.6530 | 2.9242 / 3 | 2.53% |
| `5exit_nohint` | `min_exit=3, stable_k=2` | 0.6267 | **0.6486** | 0.6684 | 4.6433 / 5 | 7.13% |
| `5exit_posweight` | `min_exit=3, stable_k=2` | **0.6449** | 0.6313 | **0.6690** | 4.5337 / 5 | **9.33%** |

Recommended headline result:

```text
Model: 5exit_posweight
Policy: min_exit=3, stable_k=2
Macro-F1: 0.6449
Samples-F1: 0.6690
Avg exit depth: 4.5337 / 5
Estimated depth-compute saved: 9.33%
```

---

## ML-GEE-A14. Key research findings

1. Multi-label early-exit evaluation is now dynamic, not only static.
2. Per-label threshold tuning is required for fair multi-label comparison.
3. Positive weighting improves label-balanced behaviour but may increase false positives.
4. Exit 1 is not yet reliable for immediate stopping.
5. The 3-exit Tiny model can save compute only when Exit 1 is enabled, but current savings are small.
6. The 5-exit model provides better dynamic flexibility because it has more intermediate decision points.
7. Exit 4 is the strongest static early-exit candidate.
8. A later stability policy, especially `min_exit=3, stable_k=2`, gives the best current accuracy-efficiency trade-off.
9. Future work should strengthen earlier exits through loss design, calibration, and sigmoid-aware hint passing.

---

## ML-GEE-A15. Recommended thesis/README wording

> The multi-label greedy early-exit study shows that dynamic inference is possible in the NeuroAccuExit architecture, but the usefulness of early exits depends strongly on exit reliability. Conservative label-set stability preserves prediction quality and provides clear compute savings in 5-exit models, while compact 3-exit models require Exit 1 to be enabled before any early stopping is possible. However, because Exit 1 currently over-predicts labels, aggressive stopping produces unacceptable F1 degradation. The strongest practical trade-off is obtained by using later stable exits, particularly the 5-exit positive-weighted model with `min_exit=3, stable_k=2`, which maintains high macro-F1 while saving estimated depth compute.

---

## ML-GEE-A16. Reproducibility commands

Run from:

```powershell
cd C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP
git checkout kexit_multi-label_greedy_EE
```

Set common variables:

```powershell
$DEVICE = "cpu"

$MANIFEST = "multilabel_cache\metadata\multilabel_features_manifest.csv"
$FEATURES_ROOT = "multilabel_cache\features"
$LABELS_JSON = "multilabel_data\metadata\labels.json"

$ROOT = "runs_multilabel"

$TRAIN_NOHINT = "$ROOT\training\multilabel_nohint"
$TRAIN_POSWEIGHT = "$ROOT\training\multilabel_posweight"

$POLICY_ROOT = "$ROOT\policy_eval\multilabel_greedy_policy"
$SUMMARY_ROOT = "$ROOT\summary"

New-Item -ItemType Directory -Force -Path $TRAIN_NOHINT | Out-Null
New-Item -ItemType Directory -Force -Path $TRAIN_POSWEIGHT | Out-Null
New-Item -ItemType Directory -Force -Path $POLICY_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path $SUMMARY_ROOT | Out-Null
```

### Train four models

3-exit no-hint:

```powershell
python -m training.train_multilabel `
  --manifest $MANIFEST `
  --features_root $FEATURES_ROOT `
  --labels_json $LABELS_JSON `
  --runs_root $TRAIN_NOHINT `
  --variant "multilabel_nohint_001_3exit" `
  --tap_blocks "1,3" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --device $DEVICE
```

5-exit no-hint:

```powershell
python -m training.train_multilabel `
  --manifest $MANIFEST `
  --features_root $FEATURES_ROOT `
  --labels_json $LABELS_JSON `
  --runs_root $TRAIN_NOHINT `
  --variant "multilabel_nohint_002_5exit" `
  --tap_blocks "1,2,3,4" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --device $DEVICE
```

3-exit pos-weight:

```powershell
python -m training.train_multilabel `
  --manifest $MANIFEST `
  --features_root $FEATURES_ROOT `
  --labels_json $LABELS_JSON `
  --runs_root $TRAIN_POSWEIGHT `
  --variant "multilabel_posweight_001_3exit" `
  --tap_blocks "1,3" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --device $DEVICE `
  --use_pos_weight `
  --pos_weight_max 20.0
```

5-exit pos-weight:

```powershell
python -m training.train_multilabel `
  --manifest $MANIFEST `
  --features_root $FEATURES_ROOT `
  --labels_json $LABELS_JSON `
  --runs_root $TRAIN_POSWEIGHT `
  --variant "multilabel_posweight_002_5exit" `
  --tap_blocks "1,2,3,4" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --device $DEVICE `
  --use_pos_weight `
  --pos_weight_max 20.0
```

### Re-detect all four trained run directories

```powershell
$RUN_3_NOHINT = (Get-ChildItem $TRAIN_NOHINT -Directory |
  Where-Object { $_.Name -like "multilabel_nohint_001_3exit_*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).FullName

$RUN_5_NOHINT = (Get-ChildItem $TRAIN_NOHINT -Directory |
  Where-Object { $_.Name -like "multilabel_nohint_002_5exit_*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).FullName

$RUN_3_POSWEIGHT = (Get-ChildItem $TRAIN_POSWEIGHT -Directory |
  Where-Object { $_.Name -like "multilabel_posweight_001_3exit_*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).FullName

$RUN_5_POSWEIGHT = (Get-ChildItem $TRAIN_POSWEIGHT -Directory |
  Where-Object { $_.Name -like "multilabel_posweight_002_5exit_*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).FullName
```

### Tune thresholds

```powershell
python scripts\tune_multilabel_thresholds.py --run_dir $RUN_3_NOHINT --device $DEVICE
python scripts\tune_multilabel_thresholds.py --run_dir $RUN_5_NOHINT --device $DEVICE
python scripts\tune_multilabel_thresholds.py --run_dir $RUN_3_POSWEIGHT --device $DEVICE
python scripts\tune_multilabel_thresholds.py --run_dir $RUN_5_POSWEIGHT --device $DEVICE
```

### Static threshold summary

```powershell
$RUN_DIRS = @(
  $RUN_3_NOHINT,
  $RUN_5_NOHINT,
  $RUN_3_POSWEIGHT,
  $RUN_5_POSWEIGHT
)

$NAMES = @(
  "3exit_nohint",
  "5exit_nohint",
  "3exit_posweight",
  "5exit_posweight"
)

python scripts\summarize_multilabel_threshold_runs.py `
  --run_dirs $RUN_DIRS `
  --names $NAMES `
  --out_dir "$SUMMARY_ROOT\threshold_summary_001_static_tuned"
```

### Helper function for policy experiments

```powershell
function Run-MultilabelGreedyPolicy {
  param(
    [string]$RunDir,
    [string]$Name,
    [int]$MinExit,
    [int]$StableK,
    [string]$SweepMinExits,
    [string]$SweepStableK,
    [string]$OutDir,
    [switch]$AllowEmptyStop
  )

  $args = @(
    "scripts\multilabel_greedy_policy.py",
    "--run_dir", $RunDir,
    "--name", $Name,
    "--device", $DEVICE,
    "--threshold_mode", "tuned_per_exit",
    "--min_exit", $MinExit,
    "--stable_k", $StableK,
    "--sweep_min_exits", $SweepMinExits,
    "--sweep_stable_k", $SweepStableK,
    "--out_dir", $OutDir
  )

  if ($AllowEmptyStop) {
    $args += "--allow_empty_stop"
  }

  python @args
}
```

### Policy Experiment 001

```powershell
$EXP001 = "$POLICY_ROOT\multilabel_greedy_policy_001_minexit2_stable2"

Run-MultilabelGreedyPolicy -RunDir $RUN_3_NOHINT -Name "3exit_nohint" -MinExit 2 -StableK 2 -SweepMinExits "2" -SweepStableK "1,2,3" -OutDir "$EXP001\3exit_nohint"
Run-MultilabelGreedyPolicy -RunDir $RUN_5_NOHINT -Name "5exit_nohint" -MinExit 2 -StableK 2 -SweepMinExits "2,3" -SweepStableK "1,2,3" -OutDir "$EXP001\5exit_nohint"
Run-MultilabelGreedyPolicy -RunDir $RUN_3_POSWEIGHT -Name "3exit_posweight" -MinExit 2 -StableK 2 -SweepMinExits "2" -SweepStableK "1,2,3" -OutDir "$EXP001\3exit_posweight"
Run-MultilabelGreedyPolicy -RunDir $RUN_5_POSWEIGHT -Name "5exit_posweight" -MinExit 2 -StableK 2 -SweepMinExits "2,3" -SweepStableK "1,2,3" -OutDir "$EXP001\5exit_posweight"
```

### Policy Experiment 002

```powershell
$EXP002 = "$POLICY_ROOT\multilabel_greedy_policy_002_minexit1_stable2"

Run-MultilabelGreedyPolicy -RunDir $RUN_3_NOHINT -Name "3exit_nohint" -MinExit 1 -StableK 2 -SweepMinExits "1,2" -SweepStableK "1,2,3" -OutDir "$EXP002\3exit_nohint"
Run-MultilabelGreedyPolicy -RunDir $RUN_5_NOHINT -Name "5exit_nohint" -MinExit 1 -StableK 2 -SweepMinExits "1,2,3" -SweepStableK "1,2,3" -OutDir "$EXP002\5exit_nohint"
Run-MultilabelGreedyPolicy -RunDir $RUN_3_POSWEIGHT -Name "3exit_posweight" -MinExit 1 -StableK 2 -SweepMinExits "1,2" -SweepStableK "1,2,3" -OutDir "$EXP002\3exit_posweight"
Run-MultilabelGreedyPolicy -RunDir $RUN_5_POSWEIGHT -Name "5exit_posweight" -MinExit 1 -StableK 2 -SweepMinExits "1,2,3" -SweepStableK "1,2,3" -OutDir "$EXP002\5exit_posweight"
```

### Policy Experiment 003

```powershell
$EXP003 = "$POLICY_ROOT\multilabel_greedy_policy_003_minexit1_stable1"

Run-MultilabelGreedyPolicy -RunDir $RUN_3_NOHINT -Name "3exit_nohint" -MinExit 1 -StableK 1 -SweepMinExits "1,2" -SweepStableK "1,2,3" -OutDir "$EXP003\3exit_nohint"
Run-MultilabelGreedyPolicy -RunDir $RUN_5_NOHINT -Name "5exit_nohint" -MinExit 1 -StableK 1 -SweepMinExits "1,2,3" -SweepStableK "1,2,3" -OutDir "$EXP003\5exit_nohint"
Run-MultilabelGreedyPolicy -RunDir $RUN_3_POSWEIGHT -Name "3exit_posweight" -MinExit 1 -StableK 1 -SweepMinExits "1,2" -SweepStableK "1,2,3" -OutDir "$EXP003\3exit_posweight"
Run-MultilabelGreedyPolicy -RunDir $RUN_5_POSWEIGHT -Name "5exit_posweight" -MinExit 1 -StableK 1 -SweepMinExits "1,2,3" -SweepStableK "1,2,3" -OutDir "$EXP003\5exit_posweight"
```

### Policy Experiment 004

```powershell
$EXP004 = "$POLICY_ROOT\multilabel_greedy_policy_004_minexit1_stable2_allowempty"

Run-MultilabelGreedyPolicy -RunDir $RUN_3_NOHINT -Name "3exit_nohint" -MinExit 1 -StableK 2 -SweepMinExits "1,2" -SweepStableK "1,2,3" -OutDir "$EXP004\3exit_nohint" -AllowEmptyStop
Run-MultilabelGreedyPolicy -RunDir $RUN_5_NOHINT -Name "5exit_nohint" -MinExit 1 -StableK 2 -SweepMinExits "1,2,3" -SweepStableK "1,2,3" -OutDir "$EXP004\5exit_nohint" -AllowEmptyStop
Run-MultilabelGreedyPolicy -RunDir $RUN_3_POSWEIGHT -Name "3exit_posweight" -MinExit 1 -StableK 2 -SweepMinExits "1,2" -SweepStableK "1,2,3" -OutDir "$EXP004\3exit_posweight" -AllowEmptyStop
Run-MultilabelGreedyPolicy -RunDir $RUN_5_POSWEIGHT -Name "5exit_posweight" -MinExit 1 -StableK 2 -SweepMinExits "1,2,3" -SweepStableK "1,2,3" -OutDir "$EXP004\5exit_posweight" -AllowEmptyStop
```

---

## ML-GEE-A17. Recommended interpretation order

1. Static threshold-tuned results.
2. Policy 001: `min_exit=2, stable_k=2`.
3. Policy 002: `min_exit=1, stable_k=2`.
4. Policy 003: `min_exit=1, stable_k=1`.
5. Policy 004: `allow_empty_stop=True`.

The most important comparison is:

```text
Policy 001 vs Policy 002
```

because it directly tests whether allowing Exit 1 makes the 3-exit Tiny model behave like a real early-exit model.

---

## ML-GEE-A18. Next update plan

Recommended next branch:

```text
kexit_multi-label_EE_lossweight
```

Main research question:

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
