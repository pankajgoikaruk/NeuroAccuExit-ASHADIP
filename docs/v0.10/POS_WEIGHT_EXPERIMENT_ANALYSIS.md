# v0.10 Pos-Weight Experiment Analysis

## Research question

Can label-imbalance-aware BCE improve weak/rare-label Macro-F1 more reliably than direct hint-pass?

## Method

The run used:

```text
v0.10 no-hint + BCEWithLogitsLoss(pos_weight) + PosWeightMax=5.0
```

Weights were computed from training-label imbalance:

```text
pos_weight(label) = negative_count / positive_count
```

then capped at:

```text
PosWeightMax = 5.0
```

---

## Important validity note

This output is a **single diagnostic run**, not true 3-seed stability.

The run name is:

```text
main_v010_no_hint_posweight_cap5_seed_101202303
```

Therefore, it should be reported as:

```text
single diagnostic run with combined run label seed_101202303
```

not as:

```text
three-seed stability result
```

---

## Raw fixed-threshold parent-level result

| exit | macro_f1 | micro_f1 | samples_f1 | exact_match | hamming_loss | avg_true_labels | avg_pred_labels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0000 | 0.3400 | 0.5642 | 0.5478 | 0.2341 | 0.1459 | 1.4694 | 1.8789 |
| 2.0000 | 0.6511 | 0.7518 | 0.7667 | 0.4706 | 0.0810 | 1.4694 | 1.7924 |
| 3.0000 | 0.8009 | 0.8939 | 0.9088 | 0.7232 | 0.0330 | 1.4694 | 1.6401 |

The raw Exit 3 result predicted too many labels:

```text
avg_true_labels = 1.4694
avg_pred_labels = 1.6401
```

This suggests `pos_weight cap5` pushed the model toward more positive predictions before LATS recalibration.

---

## LATS-v2 macro-priority result

```text
Macro-F1   = 0.8511
Micro-F1   = 0.9413
Samples-F1 = 0.9481
Exact      = 0.8478
Hamming    = 0.0171
Avg pred labels = 1.4371
```

---

## Per-label result after LATS-v2

| label | precision | recall | f1 | support | predicted_positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Brene_Brown | 0.9863 | 0.9863 | 0.9863 | 73 | 73 |
| Eckhart_Tolle | 1.0000 | 1.0000 | 1.0000 | 84 | 84 |
| Eric_Thomas | 0.9552 | 0.9412 | 0.9481 | 68 | 67 |
| Gary_Vee | 0.9855 | 1.0000 | 0.9927 | 68 | 69 |
| Jay_Shetty | 0.9278 | 1.0000 | 0.9626 | 90 | 97 |
| Nick_Vujicic | 1.0000 | 1.0000 | 1.0000 | 49 | 49 |
| other_speaker_present | 0.9363 | 0.9261 | 0.9311 | 460 | 455 |
| music_present | 0.9727 | 0.9413 | 0.9568 | 341 | 330 |
| audience_reaction_present | 0.5625 | 0.3103 | 0.4000 | 29 | 16 |
| silence_present | 0.5000 | 0.2500 | 0.3333 | 12 | 6 |

Rare labels remain weak after LATS:

```text
audience_reaction_present F1 = 0.4000
silence_present F1 = 0.3333
```

This means `pos_weight cap5` did not solve the rare-label problem.

---

## Selected LATS-v2 rules

| label | aggregation | threshold |
| --- | --- | ---: |
| Brene_Brown | mean | 0.3600 |
| Eckhart_Tolle | mean | 0.2000 |
| Eric_Thomas | p75 | 0.9300 |
| Gary_Vee | mean | 0.5400 |
| Jay_Shetty | top3mean | 0.9400 |
| Nick_Vujicic | mean | 0.3000 |
| other_speaker_present | mean | 0.6700 |
| music_present | mean | 0.4800 |
| audience_reaction_present | top2mean | 0.8800 |
| silence_present | max | 0.8500 |

---

## Comparison with baselines

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|
| v0.9_4 LATS-v2 baseline | 0.8673 | 0.9458 | 0.9517 | 0.8604 | 0.0158 |
| v0.10 no-hint seed mean | 0.8657 | 0.9439 | 0.9531 | 0.8593 | 0.0164 |
| pos_weight cap5 + LATS-v2 | 0.8511 | 0.9413 | 0.9481 | 0.8478 | 0.0171 |

---

## Conclusion

```text
pos_weight cap5 did not help.
```

It is worse than:

```text
1. v0.9_4 stable baseline
2. v0.10 no-hint seed mean
3. best single v0.10 no-hint LATS-v2 run
```

Recommended final status:

```text
Reject pos_weight cap5 as final method.
Do not continue pos_weight unless a much lower cap, such as 2.0 or 3.0, is explicitly needed as a small ablation.
```
