# v0.10 Results and Analysis

## Main comparison

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

## Result interpretation

### 1. Frozen old LATS-v2 transfer failed

The old v0.9_4 LATS-v2 policy was applied directly to v0.10 probabilities. This produced weak results for both no-hint and hint-pass:

```text
v0.10 no-hint frozen transfer Macro-F1  = 0.8452
v0.10 hint-pass frozen transfer Macro-F1 = 0.8180
```

This shows that v0.10 probabilities were calibrated differently from v0.9_4 probabilities.

---

### 2. Re-optimized LATS recovered performance

After v0.10-specific LATS optimization:

```text
v0.10 no-hint LATS-v2 Macro-F1   = 0.8624
v0.10 no-hint LATS-v2 Micro-F1   = 0.9531
v0.10 no-hint LATS-v2 Exact      = 0.8766
v0.10 no-hint LATS-v2 Hamming    = 0.0137
```

This confirms that LATS is sensitive to each model's probability distribution and must be re-optimized after retraining.

---

### 3. Hint-pass did not beat no-hint

Best hint-pass after LATS-v2 coordinate re-optimization:

```text
Macro-F1   = 0.8632
Micro-F1   = 0.9440
Samples-F1 = 0.9536
Exact      = 0.8570
Hamming    = 0.0164
```

This is weaker than v0.10 no-hint LATS-v2 on Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

---

## Best single-run v0.10 no-hint LATS-v2 config

```json
{
  "Brene_Brown": {
    "aggregation": "p75",
    "threshold": 0.54
  },
  "Eckhart_Tolle": {
    "aggregation": "top3mean",
    "threshold": 0.5
  },
  "Eric_Thomas": {
    "aggregation": "top4mean",
    "threshold": 0.62
  },
  "Gary_Vee": {
    "aggregation": "mean",
    "threshold": 0.5
  },
  "Jay_Shetty": {
    "aggregation": "p75",
    "threshold": 0.91
  },
  "Nick_Vujicic": {
    "aggregation": "p75",
    "threshold": 0.34
  },
  "other_speaker_present": {
    "aggregation": "noisy_or",
    "threshold": 0.94
  },
  "music_present": {
    "aggregation": "mean",
    "threshold": 0.37
  },
  "audience_reaction_present": {
    "aggregation": "top3mean",
    "threshold": 0.23
  },
  "silence_present": {
    "aggregation": "p75",
    "threshold": 0.42
  }
}
```

---

## Best v0.10 hint-pass LATS-v2 config

```json
{
  "Brene_Brown": {
    "aggregation": "top4mean",
    "threshold": 0.46
  },
  "Eckhart_Tolle": {
    "aggregation": "p90",
    "threshold": 0.5
  },
  "Eric_Thomas": {
    "aggregation": "top4mean",
    "threshold": 0.32
  },
  "Gary_Vee": {
    "aggregation": "mean",
    "threshold": 0.5
  },
  "Jay_Shetty": {
    "aggregation": "p75",
    "threshold": 0.92
  },
  "Nick_Vujicic": {
    "aggregation": "mean",
    "threshold": 0.3
  },
  "other_speaker_present": {
    "aggregation": "top4mean",
    "threshold": 0.56
  },
  "music_present": {
    "aggregation": "noisy_or",
    "threshold": 0.93
  },
  "audience_reaction_present": {
    "aggregation": "p90",
    "threshold": 0.63
  },
  "silence_present": {
    "aggregation": "p75",
    "threshold": 0.52
  }
}
```

---

## Final analysis decision

The best single v0.10 no-hint result improves global consistency metrics compared with v0.9_4, but it does not improve Macro-F1. The later 3-seed stability check shows that this result is not stable enough to replace v0.9_4.

Therefore:

```text
Accept v0.9_4 as stable final baseline.
Record v0.10 no-hint as diagnostic/stability ablation.
Reject current hint-pass as final method.
```
