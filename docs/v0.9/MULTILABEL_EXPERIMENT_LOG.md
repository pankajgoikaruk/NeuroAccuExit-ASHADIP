---

### 11. Global max aggregation diagnostic

After the official parent-level mean corrected-holdout evaluation, global max aggregation was tested as a diagnostic.

Command section:

```text
13_eval_v08_global_max_parent_fixed
```

Result:

| Aggregation | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Parent mean | **0.7801** | **0.9332** | **0.9406** | **0.8397** | **0.0194** | 1.4302 |
| Global max | 0.7251 | 0.8203 | 0.8423 | 0.5121 | 0.0630 | 2.0346 |

Decision:

```text
Do not use global max as the final aggregation strategy.
```

Reason: global max over-predicts parent labels, increasing false positives and worsening Exact Match and Hamming Loss.

### 12. Weak-label improvement under max aggregation

The global max diagnostic showed that the two weak transient labels improved:

| Label | Parent mean F1 | Global max F1 |
|---|---:|---:|
| `audience_reaction_present` | 0.1250 | **0.4706** |
| `silence_present` | 0.0000 | **0.1739** |

Interpretation:

```text
The model has some segment-level evidence for weak transient labels,
but parent-level mean aggregation dilutes that evidence.
```

This motivated a label-aware parent aggregation rule.

### 13. Label-aware aggregation experiment

Command section:

```text
15_compute_v08_label_aware_parent_aggregation
```

Rule:

```text
mean aggregation for 8 stable labels:
  Brene_Brown
  Eckhart_Tolle
  Eric_Thomas
  Gary_Vee
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

max aggregation for 2 transient labels:
  audience_reaction_present
  silence_present
```

Result:

| Strategy | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Parent mean official | 0.7801 | **0.9332** | **0.9406** | **0.8397** | **0.0194** | 1.4302 |
| Label-aware mean/max | **0.8320** | 0.9285 | 0.9375 | 0.8235 | 0.0211 | 1.4844 |

Finding:

```text
Label-aware aggregation improved Macro-F1 from 0.7801 to 0.8320
without retraining the model.
```

### 14. Updated final decision

The final reporting strategy is:

| Reporting role | Method | Reason |
|---|---|---|
| Main official overall result | Parent mean, fixed threshold 0.5, Exit 3 | Best Micro-F1, Samples-F1, Exact Match, and Hamming Loss. |
| Research/ablation contribution | Label-aware mean/max, fixed threshold 0.5, Exit 3 | Best Macro-F1 and better handling of weak transient labels. |
| Diagnostic only | Global max, fixed threshold 0.5, Exit 3 | Helps transient labels but damages overall multi-label prediction. |

Official result remains:

```text
v0.8-HCB
parent-level mean
fixed threshold 0.5
Exit 3
Macro-F1=0.7801
Micro-F1=0.9332
Samples-F1=0.9406
Exact=0.8397
Hamming=0.0194
```

Additional label-aware research finding:

```text
v0.8-HCB
label-aware parent aggregation
mean for stable labels
max for transient labels
Exit 3
Macro-F1=0.8320
Micro-F1=0.9285
Samples-F1=0.9375
Exact=0.8235
Hamming=0.0211
```

### 15. Updated limitation and future work

The v0.8-HCB experiment shows that weak labels should not be treated only as a training-data problem. Parent-level aggregation also matters.

Remaining limitations:

- `audience_reaction_present` and `silence_present` remain difficult under parent mean.
- Global max is too aggressive for stable labels.
- Label-aware aggregation was post-hoc; future work should integrate this rule into a formal evaluation script or learn label-specific pooling automatically.
- Future early-exit policy should consider label type: stable labels may exit based on accumulated mean confidence, while transient labels may require max/event-detection evidence.

Updated future direction:

```text
Develop label-type-aware early-exit inference:
  stable labels -> evidence accumulation / mean confidence
  transient labels -> event-triggered max confidence
```

---

### 16. v0.9 branch created

A new branch was created:

```text
agentic_data_preprocessing_v0.9
```

The purpose was to test labelwise aggregation and threshold calibration on the already strong v0.8-HCB model output, before making additional data changes.

### 17. v0.8 results reproduced in v0.9

The v0.8 official parent-mean result was reproduced inside v0.9:

```text
Macro-F1=0.780096
Micro-F1=0.933174
Samples-F1=0.940600
Exact Match=0.839677
Hamming Loss=0.019377
```

The v0.8 simple label-aware result was also reproduced:

```text
Macro-F1=0.832047
Micro-F1=0.928544
Samples-F1=0.937513
Exact Match=0.823529
Hamming Loss=0.021107
```

This confirmed that the v0.9 branch correctly reproduced the v0.8 baseline before testing new strategies.

### 18. Repeated labelwise aggregation and threshold calibration

A repeated 20-seed split experiment was run using:

```text
50% calibration / 50% evaluation
Aggregation candidates: mean, max, top2mean
Threshold grid: 0.10 to 0.95
Fixed threshold baseline: 0.5
```

| Method | Macro-F1 mean | Micro-F1 mean | Samples-F1 mean | Exact mean | Hamming Loss ↓ mean | Role |
|---|---:|---:|---:|---:|---:|---|
| max fixed thresholds | 0.7187 | 0.8200 | 0.8419 | 0.5098 | 0.0632 | Rejected: over-predicts labels. |
| mean fixed thresholds | 0.7802 | 0.9315 | 0.9392 | 0.8371 | 0.0199 | Strong baseline. |
| top2mean fixed thresholds | 0.8023 | 0.8884 | 0.9060 | 0.6927 | 0.0358 | Helps Macro-F1 but hurts reliability. |
| **v06 selected aggregation fixed thresholds** | **0.8310** | **0.9345** | **0.9449** | **0.8368** | **0.0193** | Best balanced repeated-split strategy. |
| v07 aggregation + threshold calibrated | 0.8319 | 0.9288 | 0.9363 | 0.8185 | 0.0208 | Macro good, but weaker overall. |

Decision:

```text
Use v06 selected aggregation with fixed 0.5.
Reject v07 threshold calibration as final because the Macro-F1 gain is tiny and overall reliability is weaker.
```

### 19. Frozen v0.9 full-holdout evaluation

The final frozen v0.9 script was created:

```text
scripts/v0.9/evaluate_frozen_labelwise_aggregation_v09.py
```

It applies:

| Label | Frozen v0.9 aggregation | Threshold |
|---|---|---:|
| `Brene_Brown` | mean | 0.5 |
| `Eckhart_Tolle` | top2mean | 0.5 |
| `Eric_Thomas` | mean | 0.5 |
| `Gary_Vee` | top2mean | 0.5 |
| `Jay_Shetty` | mean | 0.5 |
| `Nick_Vujicic` | mean | 0.5 |
| `other_speaker_present` | mean | 0.5 |
| `music_present` | mean | 0.5 |
| `audience_reaction_present` | top2mean | 0.5 |
| `silence_present` | top2mean | 0.5 |

Full corrected-holdout result:

```text
Macro-F1=0.851166
Micro-F1=0.937206
Samples-F1=0.948163
Exact Match=0.841984
Hamming Loss=0.018454
Avg true labels=1.469435
Avg pred labels=1.469435
```

### 20. Updated final decision

The best current method is:

```text
v0.9 frozen labelwise aggregation
fixed threshold 0.5
full corrected holdout
no retraining
```

This replaces the earlier v0.8 label-aware mean/max analysis as the strongest post-training aggregation result.

### 21. Updated research finding

The v0.9 result shows that parent-level aggregation should be label-specific:

```text
Stable labels benefit from mean aggregation.
Some labels with strong evidence in a subset of segments benefit from top2mean.
Global max is too aggressive.
Threshold calibration is less reliable than fixed 0.5 for this holdout.
```

