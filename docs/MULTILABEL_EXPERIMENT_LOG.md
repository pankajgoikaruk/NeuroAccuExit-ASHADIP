# Experiment Log — agentic_data_preprocessing_v0.5_tata_2

This log records the active **`agentic_data_preprocessing_v0.5_tata_2`** branch after completion of both 3-exit and 5-exit weakclip TATA experiments.

```text
Branch: agentic_data_preprocessing_v0.5_tata_2
Agenda: TinyAudioTriageAgent weak clip-level multi-label preprocessing for human-talk audio
Dataset stage: reviewed 5-sec clip manifest -> weak inherited 1-sec segment manifest -> log-mel features
Task: multi-label detection of target speaker identity, non-target speech, and event/background audio
Models completed: TATA 3-exit and TATA 5-exit weakclip variants
Labels: 12 = 6 target speakers + other_speaker_present + 5 event/background labels
Current status: 3-exit fixed/tuned/policy and 5-exit fixed/tuned/policy completed
Best current quality result: 3-exit tuned threshold, Macro-F1 = 0.7916
```

## Completed chronology

| Step | Output | Status |
|---|---|---|
| Create clean branch from v0.4 | `agentic_data_preprocessing_v0.5_tata_2` | Completed |
| TATA label schema | 12 labels | Completed |
| Audio filename standardisation | Renamed before final manifest editing | Completed |
| 5-sec clip-level manifest | Manual multi-hot labels with auto-notes | Completed |
| Training-ready manifest | 2,074 parent clips after excluding 11 rows | Completed |
| Weak 1-sec segment manifest | 12,469 segments | Completed |
| Segment leakage check | 0 parent clips split across train/val/test | Passed |
| Feature extraction | Log-mel `.npy` feature cache | Completed |
| TATA 3-exit fixed threshold | `tata_2_3exit_weakclip_20260530_121030` | Completed |
| TATA 3-exit threshold tuning | Per-label thresholds | Completed |
| TATA 3-exit dynamic policy | Label-set stability policy | Completed |
| TATA 5-exit fixed threshold | `tata_2_5exit_weakclip_20260531_002128` | Completed |
| TATA 5-exit threshold tuning | Per-label thresholds | Completed |
| TATA 5-exit dynamic policy | Limited compute saving with quality loss | Completed |

## Main result comparison

| Result | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg pred labels | Avg exit depth | Compute saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-exit fixed threshold | 0.7774 | 0.7656 | 0.7503 | 0.4895 | 0.0616 | 1.5650 | 3.0000 | 0.00% |
| 3-exit tuned threshold | 0.7916 | 0.7796 | 0.7619 | 0.4732 | 0.0607 | 1.7175 | 3.0000 | 0.00% |
| 3-exit dynamic policy | 0.7916 | 0.7796 | 0.7619 | 0.4732 | 0.0607 | 1.7175 | 3.0000 | 0.00% |
| 5-exit fixed threshold | 0.7529 | 0.7665 | 0.7242 | 0.5013 | 0.0574 | 1.3641 | 5.0000 | 0.00% |
| 5-exit tuned threshold | 0.7578 | 0.7503 | 0.7287 | 0.4314 | 0.0693 | 1.7420 | 5.0000 | 0.00% |
| 5-exit dynamic policy | 0.7166 | 0.7081 | 0.6848 | 0.3376 | 0.0847 | 1.8970 | 4.1198 | 17.60% |

## Threshold tuning findings

### 3-exit

Threshold tuning improved the 3-exit final model:

| Metric | Fixed | Tuned | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.7774 | 0.7916 | 0.0142 |
| Micro-F1 | 0.7656 | 0.7796 | 0.0141 |
| Samples-F1 | 0.7503 | 0.7619 | 0.0116 |
| Exact match | 0.4895 | 0.4732 | -0.0163 |
| Hamming loss | 0.0616 | 0.0607 | -0.0009 |

### 5-exit

Threshold tuning slightly improved 5-exit Macro-F1 but worsened exact match and hamming loss:

| Metric | Fixed | Tuned | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.7529 | 0.7578 | 0.0049 |
| Micro-F1 | 0.7665 | 0.7503 | -0.0162 |
| Exact match | 0.5013 | 0.4314 | -0.0699 |
| Hamming loss | 0.0574 | 0.0693 | 0.0119 |

## Per-exit quality

| Exit | 3-exit fixed Macro-F1 | 3-exit tuned Macro-F1 | 5-exit fixed Macro-F1 | 5-exit tuned Macro-F1 |
| --- | --- | --- | --- | --- |
| 1 | 0.1730 | 0.4120 | 0.1621 | 0.4014 |
| 2 | 0.5468 | 0.6524 | 0.2565 | 0.5020 |
| 3 | 0.7774 | 0.7916 | 0.4807 | 0.6412 |
| 4 | N/A | N/A | 0.6886 | 0.7340 |
| 5 | N/A | N/A | 0.7529 | 0.7578 |

## Dynamic policy sweep

| Model | Stable K | Macro-F1 | Micro-F1 | Exact match | Hamming loss | Avg exit depth | Compute saved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3-exit | 1 | 0.6622 | 0.6574 | 0.2677 | 0.0995 | 2.0367 | 32.11% |
| 3-exit | 2 | 0.7916 | 0.7796 | 0.4732 | 0.0607 | 3.0000 | 0.00% |
| 3-exit | 3 | 0.7916 | 0.7796 | 0.4732 | 0.0607 | 3.0000 | 0.00% |
| 5-exit | 1 | 0.5034 | 0.5286 | 0.0811 | 0.1638 | 2.0076 | 59.85% |
| 5-exit | 2 | 0.7166 | 0.7081 | 0.3376 | 0.0847 | 4.1198 | 17.60% |
| 5-exit | 3 | 0.7569 | 0.7476 | 0.4227 | 0.0703 | 4.8899 | 2.20% |

## Per-label comparison

| Label | 3 fixed F1 | 3 tuned F1 | 5 tuned F1 | Δ 3 tuned-fixed | Δ 5 tuned-fixed | 3 tuned thr | 5 tuned thr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brene_Brown | 0.8213 | 0.8351 | 0.8043 | 0.0138 | -0.0027 | 0.8500 | 0.5800 |
| Eckhart_Tolle | 0.9219 | 0.9283 | 0.8930 | 0.0064 | -0.0031 | 0.6400 | 0.7100 |
| Eric_Thomas | 0.7359 | 0.7798 | 0.7385 | 0.0439 | 0.0041 | 0.2000 | 0.4500 |
| Gary_Vee | 0.8746 | 0.8958 | 0.8835 | 0.0211 | 0.0172 | 0.4100 | 0.2900 |
| Jay_Shetty | 0.8521 | 0.8394 | 0.8250 | -0.0128 | -0.0064 | 0.2700 | 0.3200 |
| Nick_Vujicic | 0.8099 | 0.8235 | 0.8188 | 0.0137 | 0.0045 | 0.4400 | 0.3300 |
| other_speaker_present | 0.6195 | 0.6056 | 0.5701 | -0.0139 | -0.0093 | 0.4000 | 0.2500 |
| music_present | 0.8383 | 0.8880 | 0.8957 | 0.0497 | -0.0028 | 0.1500 | 0.4900 |
| applause_present | 0.8587 | 0.8587 | 0.8690 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| laughter_present | 0.6306 | 0.6606 | 0.5135 | 0.0300 | 0.0597 | 0.7200 | 0.2900 |
| crowd_cheer_present | 0.6872 | 0.7028 | 0.6517 | 0.0156 | -0.0023 | 0.3500 | 0.3500 |
| silence_present | 0.6783 | 0.6812 | 0.6306 | 0.0029 | 0.0000 | 0.3000 | 0.4700 |


## Figures

Generated comparison figures are saved under `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/`.

![TATA weakclip Macro-F1 comparison](docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_weakclip_macro_f1_comparison.png)

![TATA dynamic policy tradeoff](docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_dynamic_policy_tradeoff.png)

![TATA per-exit tuned Macro-F1](docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_per_exit_tuned_macro_f1.png)

![TATA per-label F1 comparison](docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_per_label_f1_comparison_3_vs_5.png)



## Current research conclusion

The 3-exit tuned-threshold TATA model is the best current configuration for overall multi-label triage quality. It achieves the strongest Macro-F1, Micro-F1, and Samples-F1 among the evaluated TATA variants. The 5-exit model provides more intermediate decision points and achieves limited compute saving under dynamic policy, but it does not improve final detection quality and its dynamic policy loses too much F1. Therefore, the current branch supports TATA primarily as a **final-exit audio triage detector**, not yet as a reliable early-exit detector.

Paper-safe statement:

```text
The v0.5_tata_2 experiments show that a weakly supervised TinyAudioTriageAgent can learn a 12-label human-talk triage task from reviewed clip-level annotations. Per-label threshold tuning improves the 3-exit model from 0.7774 to 0.7916 Macro-F1. Adding more exits does not automatically improve final quality: the 5-exit model enables limited compute saving, but at a substantial quality cost. The best current configuration is therefore the 3-exit tuned-threshold TATA model, while future work should improve segment-level supervision through synthetic mixtures, label balancing, and calibration before relying on early exits.
```

