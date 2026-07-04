# Agentic Data Preprocessing v0.10 — Hint-Pass + LATS Re-optimization

## Purpose

v0.10 tested whether the old exit-to-exit hint-pass idea improves the current human-talk multi-label speaker/context pipeline. The experiment compared:

1. v0.9_4 frozen LATS-v2 baseline.
2. v0.10 no-hint retrained model.
3. v0.10 hint-pass retrained model.
4. Frozen old LATS-v2 transfer.
5. v0.10-specific LATS-v1 and LATS-v2 re-optimization.
6. 3-seed no-hint stability check.

---

## Final result

The current v0.10 hint-pass model should **not** replace the v0.9_4 baseline.

The best single v0.10 run is:

```text
v0.10 no-hint + LATS-v2 metric-aware coordinate re-optimization
```

but the seed-stability experiment shows this improvement is not stable enough across seeds to replace the v0.9_4 final result.

---

## Main results

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Avg pred labels |
|---|---:|---:|---:|---:|---:|---:|
| v0.9_4 baseline — frozen LATS-v2 | 0.8673 | 0.9458 | 0.9517 | 0.8604 | 0.0158 | 1.4452 |
| v0.10 no-hint — frozen old LATS-v2 transfer | 0.8452 | 0.9247 | 0.9252 | 0.8062 | 0.0212 | 1.3495 |
| v0.10 no-hint — LATS-v1 re-optimized | 0.8658 | 0.9506 | 0.9562 | 0.8674 | 0.0145 | 1.4717 |
| v0.10 no-hint — LATS-v2 coordinate re-optimized | 0.8624 | 0.9531 | 0.9589 | 0.8766 | 0.0137 | 1.4591 |
| v0.10 hint-pass — frozen old LATS-v2 transfer | 0.8180 | 0.9155 | 0.9225 | 0.7878 | 0.0242 | 1.3956 |
| v0.10 hint-pass — LATS-v1 re-optimized | 0.8634 | 0.9447 | 0.9535 | 0.8639 | 0.0160 | 1.4291 |
| v0.10 hint-pass — LATS-v2 coordinate re-optimized | 0.8632 | 0.9440 | 0.9536 | 0.8570 | 0.0164 | 1.4556 |

---

## Key interpretation

- Frozen old LATS-v2 did not transfer well to v0.10 probabilities.
- v0.10-specific LATS re-optimization recovered performance.
- Hint-pass still did not beat the no-hint control.
- v0.10 no-hint can improve global consistency metrics in a single run, but not stably across seeds.
- Therefore, the stable contribution remains LATS-v2 metric-aware inference optimization.

---

## Final recommendation

Use this wording in the report:

> The v0.10 hint-pass branch investigated whether exit-to-exit hint propagation could improve the human-talk multi-label audio pipeline. Although re-optimized LATS substantially improved both no-hint and hint-pass v0.10 outputs compared with frozen-policy transfer, hint-pass did not outperform the no-hint control. A three-seed stability check further showed that the promising v0.10 no-hint gains were not consistent enough to replace the v0.9_4 LATS-v2 baseline. Therefore, v0.10 is retained as a diagnostic ablation and negative hint-pass result, while v0.9_4 LATS-v2 remains the stable final baseline.
