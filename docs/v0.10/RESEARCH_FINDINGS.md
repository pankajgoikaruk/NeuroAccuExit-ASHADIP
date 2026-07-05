# v0.10 Research Questions and Findings

## Research questions

| ID | Research question | Answer |
|---|---|---|
| RQ1 | Can frozen v0.9_4 LATS-v2 transfer directly to v0.10 probabilities? | No. v0.10 probabilities require re-optimization. |
| RQ2 | Can v0.10-specific LATS recover performance? | Yes. LATS re-optimization is effective. |
| RQ3 | Does standard hint-pass improve the human-talk multi-label task? | No. It does not beat no-hint. |
| RQ4 | Is v0.10 no-hint a stable replacement for v0.9_4? | No. It is promising but seed-sensitive. |
| RQ5 | Does `pos_weight cap5` improve rare-label performance enough to help? | No. It reduces overall final performance. |
| RQ6 | What should be reported as the main contribution? | LATS-v2 metric-aware parent-level inference optimization. |

---

## Main findings

### Finding 1: LATS-v2 is the key improvement

LATS-v2 improves global parent-level performance without retraining the base model.

### Finding 2: Hint-pass does not transfer well

The older hint-pass idea may fit simpler single-label/binary audio tasks, but the current multi-label speaker/context setting is more complex. Early-exit probabilities can pass incomplete or biased beliefs forward.

### Finding 3: Pos-weight cap5 is too aggressive

The raw fixed 0.5 result predicted too many labels:

```text
avg_true_labels = 1.4694
avg_pred_labels = 1.6401
```

After LATS, the result was still weaker than baseline.

### Finding 4: v0.10 no-hint remains useful as an ablation

Some v0.10 no-hint seeds improve global consistency metrics, but the seed mean is not strong enough to replace v0.9_4.

---

## Final report wording

> The v0.10 experiments show that neither standard hint-pass nor capped `pos_weight` training reliably improves the human-talk multi-label pipeline. Although v0.10-specific LATS re-optimization recovers strong performance, seed-stability analysis indicates that v0.10 no-hint is not stable enough to replace the v0.9_4 LATS-v2 baseline. The final contribution is therefore the LATS-v2 metric-aware inference policy, while hint-pass and `pos_weight cap5` are retained as controlled negative ablations.

---

## Future work

Only consider more tuning if there is a clear new hypothesis:

1. Lower `pos_weight` cap, such as 2.0 or 3.0.
2. Label-aware/gated hint passing.
3. Speaker-only hint vectors.
4. Exit2-to-Exit3-only hints.
5. Consistency loss instead of feeding previous-exit probabilities.

Recommended current action:

```text
Stop tuning and write the report.
```
