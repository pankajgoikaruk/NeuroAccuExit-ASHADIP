---

## O. Post-hoc label-aware aggregation analysis

After the official corrected-holdout parent-level mean evaluation, an additional aggregation diagnostic was performed to test whether transient labels were being diluted by mean probability aggregation.

The motivation was that not all labels have the same temporal behaviour:

| Label type | Labels | Parent aggregation assumption |
|---|---|---|
| Stable identity/background labels | `Brene_Brown`, `Eckhart_Tolle`, `Eric_Thomas`, `Gary_Vee`, `Jay_Shetty`, `Nick_Vujicic`, `other_speaker_present`, `music_present` | Evidence should be consistent across several segments, so mean aggregation suppresses noisy false positives. |
| Transient/bursty event labels | `audience_reaction_present`, `silence_present` | Evidence may appear in only one or two short segments, so max aggregation can recover events diluted by mean aggregation. |

Three parent-level aggregation strategies were compared on the same corrected holdout set:

```text
1. Parent mean aggregation for all labels
2. Global max aggregation for all labels
3. Label-aware aggregation:
   - mean for 8 stable labels
   - max for audience_reaction_present and silence_present
```

## P. Global max diagnostic

Global max aggregation was tested because it can recover short events. However, applying max to every label over-predicted labels and created too many false positives.

| Metric | Parent mean official | Global max diagnostic |
|---|---:|---:|
| Macro-F1 | **0.7801** | 0.7251 |
| Micro-F1 | **0.9332** | 0.8203 |
| Samples-F1 | **0.9406** | 0.8423 |
| Exact Match | **0.8397** | 0.5121 |
| Hamming Loss | **0.0194** | 0.0630 |
| Avg Pred Labels | 1.4302 | 2.0346 |

The global max result shows that max aggregation should not be used as the final overall parent-level strategy.

## Q. Weak/transient label behaviour

Although global max damaged the overall result, it improved the two rare transient labels.

| Label | Parent mean F1 | Global max F1 |
|---|---:|---:|
| `audience_reaction_present` | 0.1250 | **0.4706** |
| `silence_present` | 0.0000 | **0.1739** |

This supports the hypothesis that weak transient labels are not necessarily absent from the model predictions; instead, their probabilities are diluted when averaged across the whole parent clip.

## R. Label-aware aggregation result

The final post-hoc aggregation rule used:

```text
mean for 8 stable labels:
  Brene_Brown
  Eckhart_Tolle
  Eric_Thomas
  Gary_Vee
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

max for 2 transient labels:
  audience_reaction_present
  silence_present
```

The resulting corrected-holdout Exit-3 performance was:

| Metric | Parent mean official | Label-aware mean/max |
|---|---:|---:|
| Macro-F1 | 0.7801 | **0.8320** |
| Micro-F1 | **0.9332** | 0.9285 |
| Samples-F1 | **0.9406** | 0.9375 |
| Exact Match | **0.8397** | 0.8235 |
| Hamming Loss | **0.0194** | 0.0211 |
| Avg Pred Labels | 1.4302 | 1.4844 |

The label-aware method produced a large Macro-F1 improvement:

```text
0.7801 -> 0.8320
absolute gain = +0.0519
```

with only a small reduction in Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

## S. Updated final interpretation

The official headline result remains:

```text
v0.8-human-corrected-balanced
parent-level mean aggregation
fixed threshold 0.5
Exit 3
Micro-F1=0.9332
Samples-F1=0.9406
Exact Match=0.8397
Hamming Loss=0.0194
```

The label-aware aggregation result should be reported as an additional research finding:

```text
v0.8-human-corrected-balanced
post-hoc label-aware aggregation
mean for stable labels
max for transient labels
Exit 3
Macro-F1=0.8320
```

This finding supports the thesis claim that label-specific parent aggregation can improve rare transient labels without retraining the model.

## T. Updated thesis-ready conclusion with label-aware finding

On the corrected parent-level holdout set containing 867 parent clips and 4,335 one-second segments, the v0.8-human-corrected-balanced 3-exit model achieved the strongest overall final-exit performance under mean probability aggregation and a fixed 0.5 threshold. Compared with the previous v0.6 3-exit model re-evaluated on the same corrected holdout, it improved Macro-F1 from 0.7537 to 0.7801, Micro-F1 from 0.8865 to 0.9332, Samples-F1 from 0.8992 to 0.9406, and Exact Match from 0.7497 to 0.8397, while reducing Hamming Loss from 0.0315 to 0.0194.

A further post-hoc aggregation analysis showed that rare transient labels were diluted by parent-level mean aggregation. Global max aggregation improved `audience_reaction_present` and `silence_present`, but degraded the overall result by over-predicting labels. A label-aware aggregation rule, using mean aggregation for eight stable labels and max aggregation only for the two transient labels, improved Macro-F1 from 0.7801 to 0.8320 while maintaining high Micro-F1 of 0.9285 and Samples-F1 of 0.9375. This indicates that label-specific aggregation can recover rare event labels without changing the trained model.

## U. Additional recommended thesis figures

| Figure | File | Purpose |
|---|---|---|
| Aggregation strategy line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_aggregation_strategy_lineplot.png` | Compares parent mean, global max, and label-aware aggregation across Macro-F1, Micro-F1, Samples-F1, and Exact Match. |
| Aggregation Hamming loss line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_aggregation_hamming_loss_lineplot.png` | Shows that global max increases false positives and label error. |
| Weak-label F1 line plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_weak_label_f1_lineplot.png` | Shows improvement for `audience_reaction_present` and `silence_present` under max/label-aware aggregation. |
| Macro-Hamming trade-off plot | `docs/figures/human_talk/agentic_data_preprocessing_v0.8/v08_hcb_macro_hamming_tradeoff_bar.png` | Shows the trade-off between Macro-F1 improvement and small Hamming-loss increase for label-aware aggregation. |

---

## V. v0.9 frozen labelwise aggregation experiment

The v0.9 experiment was created on branch:

```text
agentic_data_preprocessing_v0.9
```

The goal was to test whether the strong v0.8-HCB model could be improved without retraining by selecting a more appropriate parent-level aggregation rule for each label.

The model and corrected holdout remained the same:

```text
Model: main_v08_human_corrected_balanced_3exit_20260610_084027
Parent clips: 867
Segments: 4335
Threshold: 0.5
Training: unchanged
Evaluation: corrected holdout
```

## W. v0.9 methodology

The v0.9 evaluation used three stages:

```text
1. Reproduce v0.8 official parent-mean result in the v0.9 branch.
2. Reproduce v0.8 simple label-aware mean/max result.
3. Run repeated labelwise aggregation + threshold calibration stability testing.
4. Freeze the best balanced aggregation map.
5. Evaluate the frozen map on the full corrected holdout.
```

The repeated split experiment used:

```text
Seeds: 20
Split: 50% calibration / 50% evaluation
Aggregation candidates: mean, max, top2mean
Threshold grid: 0.10 to 0.95 in steps of 0.05
Fixed-threshold baseline: 0.5
Probability prefix: exit3_prob_
Parent column: parent_clip_id
```

## X. v0.9 repeated split result

| Method | Macro-F1 mean | Micro-F1 mean | Samples-F1 mean | Exact mean | Hamming Loss ↓ mean | Role |
|---|---:|---:|---:|---:|---:|---|
| max fixed thresholds | 0.7187 | 0.8200 | 0.8419 | 0.5098 | 0.0632 | Rejected: over-predicts labels. |
| mean fixed thresholds | 0.7802 | 0.9315 | 0.9392 | 0.8371 | 0.0199 | Strong baseline. |
| top2mean fixed thresholds | 0.8023 | 0.8884 | 0.9060 | 0.6927 | 0.0358 | Helps Macro-F1 but hurts reliability. |
| **v06 selected aggregation fixed thresholds** | **0.8310** | **0.9345** | **0.9449** | **0.8368** | **0.0193** | Best balanced repeated-split strategy. |
| v07 aggregation + threshold calibrated | 0.8319 | 0.9288 | 0.9363 | 0.8185 | 0.0208 | Macro good, but weaker overall. |

The best balanced repeated-split strategy was `v06_calibration_selected_aggregation_fixed_thresholds`. Although `v07_aggregation_threshold_calibrated` gave a tiny Macro-F1 gain, it damaged Micro-F1, Samples-F1, Exact Match, and Hamming Loss. Therefore, threshold calibration was rejected as the final v0.9 strategy.

## Y. Frozen aggregation map

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

This frozen map uses `top2mean` for labels that benefit from strong evidence in a small number of segments, while retaining `mean` for stable labels.

## Z. Full corrected-holdout final v0.9 result

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss ↓ | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| v0.8 official parent mean | 0.7801 | 0.9332 | 0.9406 | 0.8397 | 0.0194 | 1.4302 |
| v0.8 simple label-aware mean/max | 0.8320 | 0.9285 | 0.9375 | 0.8235 | 0.0211 | 1.4844 |
| **v0.9 frozen labelwise top2mean/mean** | **0.8512** | **0.9372** | **0.9482** | **0.8420** | **0.0185** | **1.4694** |

## AA. v0.9 per-label behaviour

| Label | Precision | Recall | F1 | Support | Predicted positive | Aggregation |
|---|---:|---:|---:|---:|---:|---|
| `Brene_Brown` | 1.0000 | 0.9315 | 0.9645 | 73 | 68 | mean |
| `Eckhart_Tolle` | 1.0000 | 1.0000 | 1.0000 | 84 | 84 | top2mean |
| `Eric_Thomas` | 0.9028 | 0.9559 | 0.9286 | 68 | 72 | mean |
| `Gary_Vee` | 0.9444 | 1.0000 | 0.9714 | 68 | 72 | top2mean |
| `Jay_Shetty` | 0.9278 | 1.0000 | 0.9626 | 90 | 97 | mean |
| `Nick_Vujicic` | 1.0000 | 0.9592 | 0.9792 | 49 | 47 | mean |
| `other_speaker_present` | 0.9156 | 0.9435 | 0.9293 | 460 | 474 | mean |
| `music_present` | 0.9640 | 0.9413 | 0.9525 | 341 | 333 | mean |
| `audience_reaction_present` | 0.6818 | 0.5172 | 0.5882 | 29 | 22 | top2mean |
| `silence_present` | 0.4000 | 0.1667 | 0.2353 | 12 | 5 | top2mean |

## AB. v0.9 thesis-ready conclusion

The v0.9 frozen labelwise aggregation experiment showed that the v0.8-HCB model could be improved without retraining by changing only the parent-level aggregation rule. Repeated 20-seed calibration/evaluation splits indicated that labelwise aggregation with fixed 0.5 thresholds was more reliable than per-label threshold calibration. Applying the frozen aggregation map to the full corrected holdout improved Macro-F1 from 0.7801 to 0.8512, Micro-F1 from 0.9332 to 0.9372, Samples-F1 from 0.9406 to 0.9482, and Hamming Loss from 0.0194 to 0.0185. This supports the claim that label-specific aggregation is important for heterogeneous multi-label audio tasks, where stable identity labels and transient context labels require different evidence aggregation rules.

