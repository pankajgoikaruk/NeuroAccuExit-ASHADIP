# v0.10 No-Hint Seed Stability Analysis

## Purpose

The v0.10 no-hint model produced a strong single-run result after LATS-v2 re-optimization. To test whether this improvement was reliable, three no-hint seeds were trained and re-optimized using LATS.

Seeds:

```text
101, 202, 303
```

---

## Seed results

| Seed | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Avg pred labels |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.8624 | 0.9353 | 0.9467 | 0.8374 | 0.0189 | 1.4556 |
| 202 | 0.8741 | 0.9471 | 0.9565 | 0.8674 | 0.0153 | 1.4314 |
| 303 | 0.8607 | 0.9492 | 0.9560 | 0.8731 | 0.0149 | 1.4614 |
| **Mean** | **0.8657** | **0.9439** | **0.9531** | **0.8593** | **0.0164** | **1.4494** |
| Std | 0.0073 | 0.0075 | 0.0055 | 0.0192 | 0.0022 | 0.0159 |

---

## Cross-seed statistics

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| macro_f1 | 0.8657 | 0.0073 | 0.8607 | 0.8741 |
| micro_f1 | 0.9439 | 0.0075 | 0.9353 | 0.9492 |
| samples_f1 | 0.9531 | 0.0055 | 0.9467 | 0.9565 |
| exact_match | 0.8593 | 0.0192 | 0.8374 | 0.8731 |
| hamming_loss | 0.0164 | 0.0022 | 0.0149 | 0.0189 |
| avg_pred_labels | 1.4494 | 0.0159 | 1.4314 | 1.4614 |

---

## Baseline comparison

v0.9_4 baseline:

```text
Macro-F1   = 0.8673
Micro-F1   = 0.9458
Samples-F1 = 0.9517
Exact      = 0.8604
Hamming    = 0.0158
```

The v0.10 no-hint seed mean is:

```text
Macro-F1   = 0.8657
Micro-F1   = 0.9439
Samples-F1 = 0.9531
Exact      = 0.8593
Hamming    = 0.0164
```

---

## Stability conclusion

The result is not stable enough to replace v0.9_4:

- Seed 202 and Seed 303 are promising.
- Seed 101 is clearly weaker.
- The average result is below v0.9_4 on Macro-F1 and Micro-F1.
- Hamming Loss is slightly worse on average.

Final conclusion:

```text
v0.10 no-hint is a useful diagnostic ablation, not the new stable final method.
```
