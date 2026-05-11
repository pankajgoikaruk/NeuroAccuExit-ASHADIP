# NeuroAccuExit Multi-Label Greedy Early Exit — `kexit_multi-label_greedy_EE`

This branch records the **multi-label greedy early-exit (EE) policy study** for NeuroAccuExit-ASHADIP.

```text
Branch: kexit_multi-label_greedy_EE
Base branch: kexit_cclass_greedy_multi-label_policy
Core script: scripts/multilabel_greedy_policy.py
Main policy: sigmoid-aware label-set stability
```

The previous branch established the static multi-label baseline with BCE/sigmoid outputs, tuned per-label thresholds, and positive-label weighting. This branch adds the missing dynamic-neural-network evidence: **when does the model exit, how much depth compute is saved, and what accuracy/F1 trade-off is produced?**

---

## Executive summary

1. The experiment store has been reorganised into a thesis-friendly structure under `runs_multilabel/`, separating training, policy evaluation, and summaries.
2. Four models were trained again under the structured experiment layout:
   - `3exit_nohint`
   - `5exit_nohint`
   - `3exit_posweight`
   - `5exit_posweight`
3. Threshold tuning remains essential for multi-label inference. Fixed `0.5` under-predicts labels in no-hint models and can over/under-shift positive-weighted models.
4. Static per-exit analysis shows that **5exit pos-weight Exit 4** is the strongest standalone exit with macro-F1 `0.6538`.
5. The conservative policy `min_exit=2, stable_k=2` proves useful dynamic behaviour for 5-exit models, saving `14–16%` depth compute, but it cannot save compute for 3-exit models because the earliest stable decision is the final exit.
6. The fair early-exit policy `min_exit=1, stable_k=2` allows the 3-exit Tiny model to early-exit, but savings are small: `3.84%` for no-hint and `2.53%` for pos-weight.
7. The aggressive policy `min_exit=1, stable_k=1` saves `66–80%` compute but collapses prediction quality, proving that Exit 1 is not currently reliable enough for immediate stopping.
8. The best practical dynamic trade-off is currently **5exit pos-weight with `min_exit=3, stable_k=2`**, achieving macro-F1 `0.6449` with `9.33%` estimated depth-compute saving.
9. The next research step is to strengthen early exits using better loss weighting, calibration, and later sigmoid-aware hint passing.

---

## Multi-label task formulation

| Component | Setting |
|---|---|
| Target format | Multi-hot vector |
| Output activation | Sigmoid |
| Loss | BCEWithLogitsLoss |
| Prediction | Any number of labels |
| Thresholding | Fixed `0.5`, tuned per-label thresholds, and tuned per-exit/per-label thresholds |
| Policy type | Greedy label-set stability |
| Main metrics | Macro-F1, micro-F1, samples-F1, exact match, hamming loss, per-label F1 |
| Dynamic metrics | Exit distribution, average exit depth, depth compute units, estimated compute saved |

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

## Experiment store

The branch uses this result organisation:

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

This structure is preferred because policy experiments can be repeated many times against the same trained checkpoint. Training runs and dynamic-policy outputs should therefore not be mixed.

---

## Model variants

| Model | Tap blocks | Exits | Pos-weight | `pos_weight_max` | Purpose |
|---|---|---:|---|---:|---|
| `3exit_nohint` | `1,3` | 3 | No | — | Compact Tiny baseline |
| `5exit_nohint` | `1,2,3,4` | 5 | No | — | More dynamic exit opportunities |
| `3exit_posweight` | `1,3` | 3 | Yes | 20.0 | Compact model with recall-sensitive loss |
| `5exit_posweight` | `1,2,3,4` | 5 | Yes | 20.0 | Dynamic model with recall-sensitive loss |

> Important: these structured runs used `--pos_weight_max 20.0`. Do not mix these numerical results with earlier positive-weight runs that used a different cap such as `5.0`.

---

## Greedy label-set stability policy

For each exit `e`, the model produces sigmoid probabilities:

```text
p_e = sigmoid(logits_e)
```

The prediction at that exit is converted into a multi-label set using tuned per-exit/per-label thresholds:

```text
y_hat_e[label] = 1 if p_e[label] >= tau_e,label else 0
```

The policy scans exits from `min_exit` onward and stops when the predicted label set is stable for `stable_k` consecutive considered exits. If no stable decision is found, the policy falls back to the final exit.

```text
min_exit=2, stable_k=2:
  Ignore Exit 1.
  Stop only after two consecutive matching label sets from Exit 2 onward.

min_exit=1, stable_k=2:
  Allow Exit 1.
  For a 3-exit model, this allows stopping at Exit 2 if Exit 1 and Exit 2 agree.

min_exit=1, stable_k=1:
  Aggressive ablation.
  Stop as soon as the first considered exit is reached.
```

There is no `min_exit=0` in the implementation because exits are numbered from `1`.

---

## Static final-exit comparison

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

## Best static exit per model

| Model | Best Static Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5exit_posweight` | **4** | **0.6538** | **0.6536** | **0.6825** | **0.3315** | **0.1191** | 1.8764 |
| `3exit_posweight` | 3 | 0.6451 | 0.6338 | 0.6600 | 0.2753 | 0.1272 | 1.9129 |
| `3exit_nohint` | 3 | 0.6301 | 0.6361 | 0.6576 | 0.2725 | 0.1247 | 1.8652 |
| `5exit_nohint` | **4** | 0.6281 | 0.6291 | 0.6475 | 0.3090 | 0.1258 | 1.8315 |

### Static interpretation

The strongest static finding is:

```text
5exit_posweight Exit 4 macro-F1 = 0.6538
```

This means the best macro-F1 in the 5-exit system is **not** always at the final exit. This is important because it supports compute-adaptive inference: an intermediate head can be both shallower and more accurate for macro-F1.

---

## Static per-exit quality

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

### Early-exit diagnostic

Exit 1 is still weak. It predicts too many labels compared with the true average label count:

```text
True avg labels per test sample = 1.5618

Exit 1 avg predicted labels:
3exit_nohint      = 4.3539
3exit_posweight   = 4.3230
5exit_nohint      = 4.1348
5exit_posweight   = 3.7191
```

This explains why aggressive immediate stopping performs poorly. The current early exits are not yet reliable enough to be used alone.

---

## Selected policy experiments

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

## Policy 001 vs Policy 002: main fair comparison

| Model | Policy 001 Macro-F1 | Policy 002 Macro-F1 | F1 Change | Policy 001 Saved | Policy 002 Saved | Saving Gain |
|---|---:|---:|---:|---:|---:|---:|
| `3exit_nohint` | 0.6301 | 0.6179 | -0.0122 | 0.00% | 3.84% | +3.84% |
| `3exit_posweight` | 0.6451 | 0.6293 | -0.0158 | 0.00% | 2.53% | +2.53% |
| `5exit_nohint` | 0.6246 | 0.6097 | -0.0149 | 16.01% | 20.90% | +4.89% |
| `5exit_posweight` | 0.6321 | 0.6260 | -0.0061 | 14.21% | 18.93% | +4.72% |

### Interpretation

Policy 001 is conservative and safe. However, for a 3-exit model, it cannot save compute because `min_exit=2` and `stable_k=2` make the earliest stable decision the final Exit 3.

Policy 002 enables Exit 1 and finally allows the 3-exit Tiny model to behave like a real early-exit model:

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

The result is valid but modest. It shows that the compact 3-exit model can early-exit, but only for a small proportion of samples under a safe stability rule.

---

## Exit distribution

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

## Best practical dynamic trade-offs from the sweep

| Model | Best Practical Policy | Macro-F1 | Micro-F1 | Samples-F1 | Avg Depth | Compute Saved |
|---|---|---:|---:|---:|---:|---:|
| `3exit_nohint` | `min_exit=1, stable_k=2` | 0.6179 | 0.6160 | 0.6495 | 2.8848 / 3 | 3.84% |
| `3exit_posweight` | `min_exit=1, stable_k=2` | 0.6293 | 0.6162 | 0.6530 | 2.9242 / 3 | 2.53% |
| `5exit_nohint` | `min_exit=3, stable_k=2` | 0.6267 | **0.6486** | 0.6684 | 4.6433 / 5 | 7.13% |
| `5exit_posweight` | `min_exit=3, stable_k=2` | **0.6449** | 0.6313 | **0.6690** | 4.5337 / 5 | **9.33%** |

### Recommended headline dynamic result

The strongest practical dynamic result is:

```text
Model: 5exit_posweight
Policy: min_exit=3, stable_k=2
Macro-F1: 0.6449
Samples-F1: 0.6690
Avg exit depth: 4.5337 / 5
Estimated depth-compute saved: 9.33%
```

This policy avoids the unreliable first two exits and focuses on the more reliable middle-to-late exits. It aligns with the static result where Exit 4 is the best macro-F1 exit.

---

## Research conclusions

1. Multi-label early-exit evaluation is now dynamic, not only static.
2. Threshold tuning is required for fair multi-label comparison.
3. Positive weighting improves macro-F1 but increases the risk of false positives.
4. Exit 1 is not yet reliable for immediate stopping.
5. The 3-exit Tiny model can save compute only when Exit 1 is enabled, but the saving is currently small.
6. The 5-exit model provides better dynamic flexibility because it has more intermediate decision points.
7. Exit 4 is the strongest static early-exit candidate.
8. A later stability policy, especially `min_exit=3, stable_k=2`, gives the best current accuracy-efficiency trade-off.
9. Future work should strengthen earlier exits through loss design, calibration, and sigmoid-aware hint passing.

---

## Recommended wording for thesis/README

> The multi-label greedy early-exit study shows that dynamic inference is possible in the NeuroAccuExit architecture, but the usefulness of early exits depends strongly on exit reliability. Conservative label-set stability preserves prediction quality and provides clear compute savings in 5-exit models, while compact 3-exit models require Exit 1 to be enabled before any early stopping is possible. However, because Exit 1 currently over-predicts labels, aggressive stopping produces unacceptable F1 degradation. The strongest practical trade-off is obtained by using later stable exits, particularly the 5-exit positive-weighted model with `min_exit=3, stable_k=2`, which maintains high macro-F1 while saving estimated depth compute.

---

## Documentation map

| File | Purpose |
|---|---|
| `README.md` | Branch-level summary, key tables, and main research findings |
| `DOC_STRUCTURE.md` | Thesis/report structure and detailed writing plan |
| `APPENDIX.md` | Full PowerShell commands, experiment protocol, and extended tables |
