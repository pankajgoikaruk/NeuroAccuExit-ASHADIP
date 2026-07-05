# v0.10 Results and Analysis

## Complete method comparison

| method | macro_f1 | micro_f1 | samples_f1 | exact_match | hamming_loss | avg_pred_labels | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v0.9_4 baseline — frozen LATS-v2 | 0.8673 | 0.9458 | 0.9517 | 0.8604 | 0.0158 | 1.4452 | Stable final baseline |
| v0.10 no-hint — frozen old LATS-v2 transfer | 0.8452 | 0.9247 | 0.9252 | 0.8062 | 0.0212 | 1.3495 | Frozen policy transfer failed |
| v0.10 no-hint — LATS-v1 re-optimized | 0.8658 | 0.9506 | 0.9562 | 0.8674 | 0.0145 | 1.4717 | Strong single-run recovery |
| v0.10 no-hint — LATS-v2 coordinate re-optimized | 0.8624 | 0.9531 | 0.9589 | 0.8766 | 0.0137 | 1.4591 | Best single v0.10 global-consistency run |
| v0.10 hint-pass — frozen old LATS-v2 transfer | 0.8180 | 0.9155 | 0.9225 | 0.7878 | 0.0242 | 1.3956 | Hint-pass transfer weak |
| v0.10 hint-pass — LATS-v1 re-optimized | 0.8634 | 0.9447 | 0.9535 | 0.8639 | 0.0160 | 1.4291 | Recovered but not best |
| v0.10 hint-pass — LATS-v2 coordinate re-optimized | 0.8632 | 0.9440 | 0.9536 | 0.8570 | 0.0164 | 1.4556 | Rejected vs no-hint |
| v0.10 no-hint + pos_weight cap5 — fixed 0.5 mean | 0.8009 | 0.8939 | 0.9088 | 0.7232 | 0.0330 | 1.6401 | Over-predicted positives before LATS |
| v0.10 no-hint + pos_weight cap5 — LATS-v2 macro-priority | 0.8511 | 0.9413 | 0.9481 | 0.8478 | 0.0171 | 1.4371 | Rejected; worse than baseline and no-hint |

---

## What improved?

The only consistent positive result is that **model-specific LATS re-optimization** improves parent-level performance compared with frozen transfer.

Example:

```text
v0.10 no-hint frozen old LATS-v2 Macro-F1 = 0.8452
v0.10 no-hint re-optimized LATS-v2 Macro-F1 = 0.8624
```

---

## What did not improve?

### Hint-pass

Hint-pass after re-optimization:

```text
Macro-F1   = 0.8632
Micro-F1   = 0.9440
Samples-F1 = 0.9536
Exact      = 0.8570
Hamming    = 0.0164
```

This is weaker than no-hint LATS-v2 on Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

### Pos-weight cap5

After LATS-v2 macro-priority re-optimization:

```text
Macro-F1   = 0.8511
Micro-F1   = 0.9413
Samples-F1 = 0.9481
Exact      = 0.8478
Hamming    = 0.0171
```

This is worse than v0.9_4 and worse than previous v0.10 no-hint results.

---

## Final outcome

```text
v0.9_4 LATS-v2 remains final.
v0.10 no-hint is diagnostic only.
Current hint-pass is rejected.
pos_weight cap5 is rejected.
```
