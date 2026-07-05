# v0.10 No-Hint Seed Stability Analysis

## Purpose

The best single v0.10 no-hint run improved some global metrics. A seed-stability check was used to test whether this was reliable.

Seeds:

```text
101, 202, 303
```

---

## Seed results

| seed | macro_f1 | micro_f1 | samples_f1 | exact_match | hamming_loss | avg_pred_labels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.8624 | 0.9353 | 0.9467 | 0.8374 | 0.0189 | 1.4556 |
| 202 | 0.8741 | 0.9471 | 0.9565 | 0.8674 | 0.0153 | 1.4314 |
| 303 | 0.8607 | 0.9492 | 0.9560 | 0.8731 | 0.0149 | 1.4614 |

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

## Interpretation

Compared with v0.9_4:

```text
v0.9_4 Macro-F1   = 0.8673
v0.9_4 Micro-F1   = 0.9458
v0.9_4 Exact      = 0.8604
v0.9_4 Hamming    = 0.0158
```

The v0.10 no-hint seed mean is slightly weaker in Macro-F1, Micro-F1, Exact Match, and Hamming Loss. Therefore, v0.10 no-hint is not stable enough to replace v0.9_4.
