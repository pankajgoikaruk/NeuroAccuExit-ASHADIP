# Experiment Log — agentic_data_preprocessing_v0.4

This log records the current **`agentic_data_preprocessing_v0.4`** branch outcomes only. Older results are intentionally removed.

```text
Branch: agentic_data_preprocessing_v0.4
Agenda: Agentic AI-based data preprocessing plus softmax-vs-sigmoid ablation on cleaned Raw5 human-talk data
Dataset stage: raw5_agentic_cleaned
Task: five-speaker human-talk speaker classification
Models: TinyAudioCNN + ExitNet, 3-exit and 5-exit
Classes: Brene_Brown, Eckhart_Tolle, Eric_Thomas, Gary_Vee, Jay_Shetty
Final cleaned files: 3,108
```

## Branch objective

This branch evaluates whether agentic preprocessing can convert noisy Raw5 speaker folders into a safe cleaned dataset and whether the NeuroAccuExit early-exit architecture behaves better under softmax speaker classification or sigmoid/BCE one-hot speaker supervision.

## Chronology

| Step | Output | Status |
|---|---|---|
| Raw5 folder preparation | Brene_Brown, Eckhart_Tolle, Eric_Thomas, Gary_Vee, Jay_Shetty | Completed |
| DatasetAuditorAgent | Raw5 audit and routing decisions | Completed |
| ManifestBuilderAgent | accepted / needs_review / rejected / blocked manifests | Completed |
| DatasetBuilderAgent | cleaned 16 kHz mono dataset | Completed |
| Cleaned re-audit | one bad file found | Completed |
| Manual exclusion | `Eric_Thomas__0175.wav` removed from training root | Completed |
| Softmax baseline | 3-exit and 5-exit on `raw5_agentic_cleaned` | Completed |
| Sigmoid ablation | 3-exit and 5-exit BCE/sigmoid one-hot runs | Completed |
| Sigmoid threshold tuning | fixed vs tuned comparison | Completed |
| Sigmoid label-set policy | dynamic early-exit policy sweep | Completed |


## Agentic preprocessing outcome

The agentic preprocessing layer was kept non-destructive. It audited raw files, separated technical preprocessing requirements from true quality concerns, generated accepted / needs-review / rejected / blocked manifests, built cleaned 16 kHz mono copies, and preserved traceability.

| Item | Result |
|---|---:|
| Raw files audited | 3,170 |
| Accepted by raw audit | 3,109 |
| Needs review | 27 |
| Rejected | 34 |
| Blocked | 0 |
| Cleaned files built | 3,109 |
| Build failures | 0 |
| Re-audit accepted | 3,108 |
| Manually excluded | 1 |
| Final training-ready files | 3,108 |

Final class counts:

| Class | Final files |
|---|---:|
| Brene_Brown | 595 |
| Eckhart_Tolle | 660 |
| Eric_Thomas | 593 |
| Gary_Vee | 642 |
| Jay_Shetty | 618 |
| **Total** | **3,108** |

Manual exclusion: `Eric_Thomas__0175.wav` was confirmed as pure music with no target speaker and was moved outside the training root while preserving traceability.


## Softmax and sigmoid comparison

| Setting | Model | Activation / loss | Final metric | Final Macro-F1 | Accuracy / Exact Match | Hamming loss | Policy metric | Avg exit depth | Compute saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Softmax | 3-exit | Softmax + CE | single-label accuracy | 0.9756 | 0.9760 | N/A | 0.9683 | 2.0886 | 52.56% |
| Softmax | 5-exit | Softmax + CE | single-label accuracy | 0.9610 | 0.9616 | N/A | 0.9520 | 2.7144 | 62.03% |
| Sigmoid fixed | 3-exit | Sigmoid + BCE | one-hot exact match | 0.9692 | 0.9535 | 0.0121 | N/A | N/A | N/A |
| Sigmoid fixed | 5-exit | Sigmoid + BCE | one-hot exact match | 0.9627 | 0.9426 | 0.0148 | N/A | N/A | N/A |
| Sigmoid tuned | 3-exit | Sigmoid + BCE | one-hot exact match | 0.9670 | 0.9505 | 0.0131 | 0.9670 | 3.0000 | 0.00% |
| Sigmoid tuned | 5-exit | Sigmoid + BCE | one-hot exact match | 0.9647 | 0.9465 | 0.0139 | 0.9561 | 3.4537 | 30.93% |

## Per-exit test quality

| Exit | Softmax 3 Macro-F1 | Softmax 5 Macro-F1 | Sigmoid 3 fixed Macro-F1 | Sigmoid 3 tuned Macro-F1 | Sigmoid 5 fixed Macro-F1 | Sigmoid 5 tuned Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.6404 | 0.6334 | 0.2943 | 0.5675 | 0.3085 | 0.5718 |
| 2 | 0.9229 | 0.8612 | 0.8854 | 0.9014 | 0.7395 | 0.7810 |
| 3 | 0.9756 | 0.9189 | 0.9692 | 0.9670 | 0.8872 | 0.9130 |
| 4 | N/A | 0.9537 | N/A | N/A | 0.9405 | 0.9511 |
| 5 | N/A | 0.9610 | N/A | N/A | 0.9627 | 0.9647 |

## Dynamic policy results

| Setting | Model | Policy | Metric | Avg exit depth | Compute saved | Exit consistency | Main observation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Softmax | 3-exit | Greedy confidence | accuracy 0.9683 | 2.0886 | 52.56% | 0.9913 | Best segment-level policy balance |
| Softmax | 5-exit | Greedy confidence | accuracy 0.9520 | 2.7144 | 62.03% | 0.9849 | Highest segment-depth compute saving |
| Sigmoid tuned | 3-exit | Label-set stability k=2 | macro-F1 0.9670 | 3.0000 | 0.00% | 1.0000 | No early exit; all samples reach final exit |
| Sigmoid tuned | 5-exit | Label-set stability k=2 | macro-F1 0.9561 | 3.4537 | 30.93% | 0.9673 | Works but less efficient than softmax |

## Sigmoid label-set-stability policy sweep

| Model | Stable K | Macro-F1 | Exact match | Avg exit depth | Compute saved | Exit consistency |
| --- | --- | --- | --- | --- | --- | --- |
| Sigmoid 3-exit | 1 | 0.9420 | 0.9183 | 2.0931 | 30.23% | 0.9384 |
| Sigmoid 3-exit | 2 | 0.9670 | 0.9505 | 3.0000 | 0.00% | 1.0000 |
| Sigmoid 3-exit | 3 | 0.9670 | 0.9505 | 3.0000 | 0.00% | 1.0000 |
| Sigmoid 5-exit | 1 | 0.8571 | 0.7832 | 2.2062 | 55.88% | 0.7891 |
| Sigmoid 5-exit | 2 | 0.9561 | 0.9376 | 3.4537 | 30.93% | 0.9673 |
| Sigmoid 5-exit | 3 | 0.9649 | 0.9488 | 4.3649 | 12.70% | 0.9923 |

## Clip-level softmax result

| Model | Clip policy | Clip accuracy | Avg windows used | Avg total windows | Windows saved | Compute saved |
| --- | --- | --- | --- | --- | --- | --- |
| Softmax 3-exit | Full-window aggregation | 0.9957 | 8.6510 | 8.6510 | 0.00% | 0.00% |
| Softmax 3-exit | Depth×Time | 0.9893 | 2.0878 | 8.6510 | 75.87% | 75.82% |
| Softmax 5-exit | Full-window aggregation | 0.9850 | 8.6510 | 8.6510 | 0.00% | 0.00% |
| Softmax 5-exit | Depth×Time | 0.9764 | 2.1649 | 8.6510 | 74.98% | 74.64% |

## Per-speaker final-exit F1

| Speaker | Softmax 3 | Softmax 5 | Sigmoid 3 fixed | Sigmoid 3 tuned | Sigmoid 5 fixed | Sigmoid 5 tuned |
| --- | --- | --- | --- | --- | --- | --- |
| Brene_Brown | 0.9637 | 0.9427 | 0.9547 | 0.9554 | 0.9266 | 0.9396 |
| Eckhart_Tolle | 0.9924 | 0.9918 | 0.9930 | 0.9913 | 0.9918 | 0.9918 |
| Eric_Thomas | 0.9671 | 0.9520 | 0.9510 | 0.9466 | 0.9546 | 0.9513 |
| Gary_Vee | 0.9708 | 0.9513 | 0.9611 | 0.9594 | 0.9553 | 0.9518 |
| Jay_Shetty | 0.9842 | 0.9671 | 0.9860 | 0.9823 | 0.9851 | 0.9891 |

## Threshold tuning observations

### Sigmoid 3-exit

| Metric | Fixed 0.5 | Tuned | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.9692 | 0.9670 | -0.0022 |
| Micro-F1 | 0.9696 | 0.9674 | -0.0022 |
| Exact match | 0.9535 | 0.9505 | -0.0030 |
| Hamming loss | 0.0121 | 0.0131 | worse |

For 3-exit sigmoid, fixed 0.5 performed better than tuned thresholds.

### Sigmoid 5-exit

| Label | Tuned threshold | Fixed F1 | Tuned F1 | ΔF1 | Fixed P | Tuned P | Fixed R | Tuned R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brene_Brown | 0.8700 | 0.9266 | 0.9396 | 0.0131 | 0.8924 | 0.9540 | 0.9635 | 0.9257 |
| Eckhart_Tolle | 0.3000 | 0.9918 | 0.9918 | 0.0000 | 0.9895 | 0.9872 | 0.9941 | 0.9965 |
| Eric_Thomas | 0.5500 | 0.9546 | 0.9513 | -0.0033 | 0.9516 | 0.9524 | 0.9576 | 0.9501 |
| Gary_Vee | 0.1900 | 0.9553 | 0.9518 | -0.0035 | 0.9701 | 0.9421 | 0.9409 | 0.9618 |
| Jay_Shetty | 0.3100 | 0.9851 | 0.9891 | 0.0040 | 0.9974 | 0.9948 | 0.9731 | 0.9834 |

For 5-exit sigmoid, threshold tuning improved the final Macro-F1 from 0.9627 to 0.9647 and improved exact match from 0.9426 to 0.9465.


### Figures

Generated comparison figures are saved under `figures/human_talk/agentic_data_preprocessing_v0.4/`:

![Final Macro-F1 comparison](figures/human_talk/agentic_data_preprocessing_v0.4/final_macro_f1_comparison.png)

![Dynamic policy quality and compute saving](figures/human_talk/agentic_data_preprocessing_v0.4/dynamic_policy_quality_compute.png)

![Softmax clip policy comparison](figures/human_talk/agentic_data_preprocessing_v0.4/softmax_clip_policy_comparison.png)

![Per-exit Macro-F1 comparison](figures/human_talk/agentic_data_preprocessing_v0.4/per_exit_macro_f1_comparison.png)

Original run plots also retained for reference: `softmax_3exit_val_acc_exits.png`, `softmax_5exit_val_acc_exits.png`, `softmax_3exit_policy_reliability.png`, and `softmax_5exit_policy_reliability.png`.



## Paper-safe conclusion

For the cleaned human-talk speaker dataset, the softmax formulation remains the most appropriate setting because each segment has one mutually exclusive speaker label. The 3-exit softmax model achieved the strongest final-exit performance and the best dynamic early-exit balance. The sigmoid/BCE ablation confirmed that the NeuroAccuExit architecture can also learn one-vs-rest speaker targets, but sigmoid is more threshold-sensitive and weaker for early-exit efficiency. Therefore, sigmoid should not replace softmax for the main speaker classifier; instead, sigmoid/BCE should be reserved for the future TinyAudioTriageAgent, where multiple audio-content tags such as target speaker, other speaker, music, silence, applause, and laughter can co-exist.

