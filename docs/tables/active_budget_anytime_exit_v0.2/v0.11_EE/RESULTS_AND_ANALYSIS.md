# v0.11_EE Results and Analysis

## 1. Staged equivalence

All four unit tests passed. The real canonical checkpoint also passed staged/full comparison at every exit.

| Exit | Max absolute logit difference | Mean absolute logit difference | Max probability difference |
|---:|---:|---:|---:|
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 |
| 3 | 0.0 | 0.0 | 0.0 |

Therefore any quality difference in later experiments comes from choosing a shallower exit, not from an altered computation at the same exit.

---

## 2. Fixed-exit segment results

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Avg predicted labels |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.120753 | 0.294082 | 0.191534 | 0.032295 | 0.133449 | 0.420992 |
| 2 | 0.597819 | 0.703483 | 0.632762 | 0.442215 | 0.078155 | 1.166321 |
| 3 | 0.737706 | 0.879322 | 0.871340 | 0.738408 | 0.034141 | 1.359631 |

Exit 1 is under-developed for the full ten-label task. Exit 2 recovers substantial information but is not a universal substitute for Exit 3.

---

## 3. Fixed-exit parent results

Frozen historical LATS-v2 rules were transferred to all exits.

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Avg predicted labels |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.162576 | 0.387702 | 0.290158 | 0.057670 | 0.135525 | 0.743945 |
| 2 | 0.692271 | 0.776025 | 0.765195 | 0.515571 | 0.065513 | 1.455594 |
| 3 | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 | 1.459054 |

Exit 2 retains approximately:

```text
80.3% of Macro-F1
81.4% of Micro-F1
79.8% of Samples-F1
58.8% of Exact Match
```

The near-identical average predicted-label count at Exit 2 does not imply equivalent correctness. Exit 2 often predicts a plausible number of labels but the wrong label set.

---

## 4. Per-label depth behaviour

### Labels relatively strong at Exit 2

| Label | Exit-2 F1 | Exit-3 F1 |
|---|---:|---:|
| Eckhart Tolle | 0.9941 | 1.0000 |
| Jay Shetty | 0.9048 | 0.9783 |
| Music present | 0.8917 | 0.9573 |
| Brene Brown | 0.8133 | 0.9861 |

### Labels requiring deeper evidence

| Label | Exit-2 F1 | Exit-3 F1 |
|---|---:|---:|
| Eric Thomas | 0.6299 | 0.9420 |
| Nick Vujicic | 0.7049 | 0.9800 |
| Audience reaction | 0.2677 | 0.5357 |
| Gary Vee | 0.7407 | 1.0000 |
| Other speaker | 0.7255 | 0.9587 |
| Silence | 0.2500 | 0.2857 |

Exit 1 produced zero F1 for Brene Brown, Eric Thomas, Gary Vee, Jay Shetty, Nick Vujicic, and silence under the transferred parent policy. Music was the strongest Exit-1 label with F1 0.7312.

This label heterogeneity motivates later label-aware policies, but v0.11 remains sample-wise.

---

## 5. Validation policy selection

Selected candidate:

| Setting/result | Value |
|---|---:|
| Confidence threshold | 0.55 |
| Margin threshold | 0.00 |
| Exit-1/Exit-2 agreement | required |
| Empty prediction stopping | disabled |
| Validation Exit-2 fraction | 0.233139 |
| Validation parent Macro-F1 | 0.892317 |
| Absolute Macro-F1 drop | 0.0103 |
| Estimated FLOPs saved | 14.982049% |
| Constraint status | met |

The selected margin threshold is zero, so the first policy is primarily an agreement-plus-confidence policy.

---

## 6. Genuine dynamic holdout result

| Measure | Value |
|---|---:|
| Segments | 4,335 |
| Exit-2 samples | 508 |
| Exit-3 samples | 3,827 |
| Exit-2 fraction | 11.72% |
| Average exit depth | 2.8828 |
| Estimated FLOPs saved | 7.53% |
| Model latency | 0.8552 ms/segment |
| Segment Macro-F1 | 0.721297 |
| Parent Macro-F1 | 0.842248 |
| Parent Micro-F1 | 0.935484 |
| Parent Samples-F1 | 0.943577 |
| Parent Exact Match | 0.838524 |
| Parent Hamming Loss | 0.018916 |
| Parent average predicted labels | 1.462514 |

### Quality change from full depth

| Metric | Dynamic | Full depth | Absolute change | Retention |
|---|---:|---:|---:|---:|
| Macro-F1 | 0.842248 | 0.862382 | −0.020134 | 97.67% |
| Micro-F1 | 0.935484 | 0.953131 | −0.017647 | 98.15% |
| Samples-F1 | 0.943577 | 0.958889 | −0.015312 | 98.40% |
| Exact Match | 0.838524 | 0.876586 | −0.038062 | 95.66% |
| Hamming Loss ↓ | 0.018916 | 0.013725 | +0.005191 | — |

---

## 7. Validation-to-holdout shift

| Split | Exit-2 fraction | Estimated FLOPs saved |
|---|---:|---:|
| Validation | 23.31% | 14.98% |
| Corrected holdout | 11.72% | 7.53% |

The frozen rule was less frequently satisfied on the holdout. This suggests a shift in confidence/agreement characteristics between the model-validation split and the human-corrected holdout.

Importantly, the policy was not relaxed after observing this result.

---

## 8. Compute interpretation

Architecture estimates:

| Exit | Cumulative cost | Potential saving vs Exit 3 |
|---:|---:|---:|
| 1 | 3.6% | 96.4% |
| 2 | 35.7% | 64.3% |
| 3 | 100% | 0% |

The dynamic policy saves 7.53% because only 11.72% of samples stop at Exit 2. The final-stage saving for each of those samples is large, but the policy is deliberately selective.

---

## 9. Main conclusions

1. The runtime implementation is correct and exactly checkpoint-equivalent.
2. Exit 1 should remain highly conservative or disabled.
3. Exit 2 is sufficiently mature for selective early stopping.
4. The first dynamic policy proves genuine computation avoidance.
5. The first policy sacrifices about 2.01 absolute parent Macro-F1 points for 7.53% estimated compute saving.
6. Exact Match degrades more sharply than global F1 metrics, showing that full label-set consistency is sensitive to early stopping.
7. A stricter validation quality constraint may produce a safer operating point, though it will likely reduce Exit-2 coverage.
8. Exit-specific calibration is a justified future ablation.
9. Budget-aware inference should build on this staged controller rather than on post-hoc exit selection.

---

## 10. Limitations

- Exit-1 and Exit-2 parent rules are not calibrated specifically for those exits.
- The canonical run lacks per-exit tuned threshold artifacts.
- The dynamic latency is not yet compared with Always Exit 3 using identical batching, synchronization, and timing scope.
- FLOP savings are estimated from architecture operations.
- CPU latency was measured for one run; repeated distributional timing is needed.
- v0.11 has no explicit budget forcing.
- v0.11 assigns one exit to the entire sample rather than different exits to individual labels.
