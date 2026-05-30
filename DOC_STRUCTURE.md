# Documentation Structure — agentic_data_preprocessing_v0.5_tata_2

This document defines the active documentation structure for the **`agentic_data_preprocessing_v0.5_tata_2`** branch.

```text
Branch: agentic_data_preprocessing_v0.5_tata_2
Agenda: TinyAudioTriageAgent weak clip-level multi-label preprocessing for human-talk audio
Dataset stage: TATA reviewed 5-sec clip manifest -> weak 1-sec segment manifest
Task: multi-label detection of target speaker identity, non-target speech, and event/background audio
Model: TinyAudioCNN + ExitNet, 3-exit baseline
Labels: 12 labels = 6 target speakers + other_speaker_present + 5 event/background labels
Current status: first fixed-threshold TATA 3-exit baseline completed; threshold tuning not yet applied
```

## Current documentation scope

This branch documentation should cover:

1. Why the project moved from v0.4 softmax/sigmoid ablation to true TATA multi-label triage.
2. The 12-label TATA schema and why `other_speaker_present` is collapsed into one non-target speech label.
3. The reviewed 5-sec clip-level manifest and the decision not to manually annotate event timing.
4. Weak 1-sec segment generation with parent-clip split protection.
5. Log-mel feature extraction and first TATA 3-exit BCE/sigmoid training.
6. Fixed-threshold test results and per-label findings.
7. Next steps: threshold tuning, multi-label greedy policy, 5-exit comparison, positive weighting/sampling, synthetic mixed data, and raw-data TATA pseudo-label inference.

## Recommended paper/report sections

| Section | Content to include |
|---|---|
| Motivation | Raw human-talk data may include target speakers, non-target speech, music, applause, laughter, crowd cheer, and silence. |
| TATA label design | 6 target speaker labels, `other_speaker_present`, and 5 event/background labels. |
| Manual manifest design | 5-sec clip-level multi-hot labels; no manual time-span annotation. |
| Weak segment supervision | 1-sec segments inherit parent clip labels; this is weak supervision. |
| Leakage control | Parent clip ID preserved; segments from a parent clip remain in one split. |
| First TATA experiment | 3-exit BCE/sigmoid baseline with fixed 0.5 threshold. |
| Findings | Final exit works; early exits not ready; threshold tuning required. |
| Future work | Threshold tuning, policy, 5-exit, positive weighting, synthetic mixing, raw pseudo-labels. |

## Research questions

| ID | Research question | Current answer from v0.5_tata_2 |
| --- | --- | --- |
| RQ1 | Can we build a reviewed multi-label TATA manifest from manually corrected 5-sec clips? | Yes: 2,074 training-ready parent clips were used after excluding 11 unusable/holdout rows. |
| RQ2 | Can clip-level labels be converted into weak 1-sec training segments while avoiding parent leakage? | Yes: 12,469 segments were created with 0 segment-build errors and 0 parent clips split across train/val/test. |
| RQ3 | Can a 3-exit TinyAudioCNN + ExitNet learn the 12-label TATA task with BCE/sigmoid? | Yes: final-exit test Macro-F1 reached 0.7774 at fixed threshold 0.5. |
| RQ4 | Is the model early-exit ready? | Not yet: Exit 1 and Exit 2 are weaker; final exit is currently the reliable output. |
| RQ5 | Which labels are strong and weak? | Strong labels include Eckhart_Tolle, Gary_Vee, Jay_Shetty, Nick_Vujicic, music, applause, and Brene_Brown. Weaker labels include other_speaker_present, laughter, crowd_cheer, silence, and Eric_Thomas recall. |
| RQ6 | What should happen before changing architecture or data? | Run per-label threshold tuning and a multi-label greedy policy evaluation. |

## Canonical result tables

### Dataset/segment status

| Item | Value |
| --- | --- |
| Reviewed clip-level training-ready rows | 2074 |
| Weak 1-sec segments created | 12469 |
| Segment build errors | 0 |
| Parent clips represented | 2074 |
| Parents split across train/val/test | 0 |
| Mean segments per parent clip | 6.01 |
| Min / max segments per parent clip | 2 / 109 |
| Mean active labels per segment | 1.6327 |
| Max active labels in a segment | 5 |

### Experiment settings

| Setting | Value |
| --- | --- |
| Branch | `agentic_data_preprocessing_v0.5_tata_2` |
| Run variant | `tata_2_3exit_weakclip` |
| Run directory | `human_talk_workspace\tata_2\runs\tata_2_3exit_weakclip_20260530_121030` |
| Task | `multi_label_audio` / TinyAudioTriageAgent |
| Model | TinyAudioCNN + ExitNet |
| Exits | 3 |
| Tap blocks | `1,3` |
| Labels | 12 |
| Loss / activation | BCEWithLogitsLoss + sigmoid |
| Threshold | 0.5 |
| Loss weights | `0.3, 0.3, 1.0` |
| Exit hint | `disabled` |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Device | `cpu` |
| Seed | 42 |
| Use positive class weighting | False |
| Runtime | 811.02 sec (~13.52 min) |

### Fixed-threshold exit metrics

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg predicted labels | Avg true labels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1730 | 0.2890 | 0.2036 | 0.0673 | 0.1219 | 0.4707 | 1.5869 |
| 2 | 0.5468 | 0.6192 | 0.5304 | 0.2968 | 0.0838 | 1.0551 | 1.5869 |
| 3 | 0.7774 | 0.7656 | 0.7503 | 0.4895 | 0.0616 | 1.5650 | 1.5869 |

### Per-label final-exit metrics

| Label | Precision | Recall | F1 | Support | Predicted positive |
| --- | --- | --- | --- | --- | --- |
| `Brene_Brown` | 0.7751 | 0.8733 | 0.8213 | 150 | 169 |
| `Eckhart_Tolle` | 0.9254 | 0.9185 | 0.9219 | 135 | 134 |
| `Eric_Thomas` | 0.8854 | 0.6296 | 0.7359 | 135 | 96 |
| `Gary_Vee` | 0.9804 | 0.7895 | 0.8746 | 190 | 153 |
| `Jay_Shetty` | 0.8622 | 0.8423 | 0.8521 | 260 | 254 |
| `Nick_Vujicic` | 0.8582 | 0.7667 | 0.8099 | 150 | 134 |
| `other_speaker_present` | 0.5654 | 0.6849 | 0.6195 | 511 | 619 |
| `music_present` | 0.9768 | 0.7342 | 0.8383 | 632 | 475 |
| `applause_present` | 0.9072 | 0.8151 | 0.8587 | 384 | 345 |
| `laughter_present` | 0.5609 | 0.7202 | 0.6306 | 243 | 312 |
| `crowd_cheer_present` | 0.6036 | 0.7976 | 0.6872 | 252 | 333 |
| `silence_present` | 0.8667 | 0.5571 | 0.6783 | 70 | 45 |

### Next strategy

| Step | Purpose | Status |
| --- | --- | --- |
| Threshold tuning | Tune per-label sigmoid thresholds and compare against fixed 0.5 | Next |
| Multi-label greedy policy | Check whether a 3-exit TATA model can exit early safely | After threshold tuning |
| Package tuned outputs | Share metrics/config/policy/log files in one ZIP | After policy |
| 5-exit TATA weakclip | Compare depth/compute tradeoff against 3-exit | Later |
| Positive class weighting / sampling | Improve weak labels such as other_speaker, laughter, crowd cheer, silence | Later ablation |
| Synthetic mixed data | Create controlled target+event/target+other-speaker mixtures | Later improvement |
| TATA inference on raw dataset | Generate pseudo-labels and routing manifests for main speaker model | After TATA is reliable |

## Figure assets

Use the following generated figures in reports and thesis notes:

| Figure | Path | Purpose |
|---|---|---|
| Validation progression | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_3exit_validation_progression.png` | Shows validation Macro-F1, Micro-F1, and exact match across epochs. |
| Training loss | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_3exit_training_loss.png` | Shows training-loss progression. |
| Test metrics by exit | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_test_metrics_by_exit.png` | Shows Exit 1 -> Exit 3 improvement. |
| Per-label F1 | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_per_label_f1.png` | Shows which labels are strong/weak. |
| Segment label distribution | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_segment_label_distribution.png` | Shows label imbalance in weak segments. |
| Split distribution | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_split_distribution.png` | Shows train/val/test segment counts. |

## Documentation rule

Do not claim TATA is finished. The correct statement is: **the first 3-exit weak-clip TATA baseline completed successfully, fixed-threshold final-exit performance is promising, and threshold tuning is the next required step**.
