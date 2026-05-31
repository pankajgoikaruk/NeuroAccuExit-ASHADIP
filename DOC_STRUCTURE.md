# Documentation Structure — agentic_data_preprocessing_v0.5_tata_2

This document defines the active documentation structure for **`agentic_data_preprocessing_v0.5_tata_2`** after both 3-exit and 5-exit TATA weakclip experiments.

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

## Current documentation scope

1. v0.4 motivation: why softmax is correct for final mutually exclusive speaker classification, but sigmoid/BCE is required for audio triage.
2. v0.5_tata_2 label schema: 12 labels with `other_speaker_present` collapsed into one non-target speech label.
3. Reviewed 5-sec clip-level manifest: primary label plus multi-hot labels; no manual event timing.
4. Weak 1-sec segment generation: parent clip hierarchy and leakage protection.
5. TATA 3-exit and 5-exit training with BCE/sigmoid.
6. Fixed threshold, tuned threshold, and dynamic policy comparisons.
7. Research conclusion: 3-exit tuned is best quality; 5-exit dynamic saves compute but loses too much F1.
8. Next work: synthetic mixed examples, calibration, balancing, and raw-data pseudo-label routing.

## Canonical result tables

### Main comparison

| Result | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg pred labels | Avg exit depth | Compute saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3-exit fixed threshold | 0.7774 | 0.7656 | 0.7503 | 0.4895 | 0.0616 | 1.5650 | 3.0000 | 0.00% |
| 3-exit tuned threshold | 0.7916 | 0.7796 | 0.7619 | 0.4732 | 0.0607 | 1.7175 | 3.0000 | 0.00% |
| 3-exit dynamic policy | 0.7916 | 0.7796 | 0.7619 | 0.4732 | 0.0607 | 1.7175 | 3.0000 | 0.00% |
| 5-exit fixed threshold | 0.7529 | 0.7665 | 0.7242 | 0.5013 | 0.0574 | 1.3641 | 5.0000 | 0.00% |
| 5-exit tuned threshold | 0.7578 | 0.7503 | 0.7287 | 0.4314 | 0.0693 | 1.7420 | 5.0000 | 0.00% |
| 5-exit dynamic policy | 0.7166 | 0.7081 | 0.6848 | 0.3376 | 0.0847 | 1.8970 | 4.1198 | 17.60% |

### Dynamic policy sweep

| Model | Stable K | Macro-F1 | Micro-F1 | Exact match | Hamming loss | Avg exit depth | Compute saved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3-exit | 1 | 0.6622 | 0.6574 | 0.2677 | 0.0995 | 2.0367 | 32.11% |
| 3-exit | 2 | 0.7916 | 0.7796 | 0.4732 | 0.0607 | 3.0000 | 0.00% |
| 3-exit | 3 | 0.7916 | 0.7796 | 0.4732 | 0.0607 | 3.0000 | 0.00% |
| 5-exit | 1 | 0.5034 | 0.5286 | 0.0811 | 0.1638 | 2.0076 | 59.85% |
| 5-exit | 2 | 0.7166 | 0.7081 | 0.3376 | 0.0847 | 4.1198 | 17.60% |
| 5-exit | 3 | 0.7569 | 0.7476 | 0.4227 | 0.0703 | 4.8899 | 2.20% |

### Best model by goal

| Goal | Best current result | Evidence |
| --- | --- | --- |
| Best overall Macro-F1 | 3-exit tuned threshold | Macro-F1 0.7916; Micro-F1 0.7796 |
| Best exact match | 5-exit fixed threshold | Exact match 0.5013 |
| Best hamming loss | 5-exit fixed threshold | Hamming loss 0.0574 |
| Best dynamic compute saving with usable quality | 5-exit stable_k=2 policy | 17.60% saved but Macro-F1 only 0.7166 |
| Most reliable current TATA configuration | 3-exit tuned threshold | Best F1 quality; should be used as final-exit triage detector, not early-exit policy yet |

## Figure assets

| Figure | Path | Purpose |
|---|---|---|
| Main Macro-F1 comparison | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_weakclip_macro_f1_comparison.png` | Compare 3-exit and 5-exit fixed/tuned/policy outcomes. |
| Dynamic policy tradeoff | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_dynamic_policy_tradeoff.png` | Show quality vs compute saving. |
| Per-exit tuned Macro-F1 | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_per_exit_tuned_macro_f1.png` | Show exit-depth behaviour. |
| Per-label F1 comparison | `docs/figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_per_label_f1_comparison_3_vs_5.png` | Compare label-level strengths/weaknesses. |

## Documentation rules

- Do not claim that 5-exit is better overall. It is not: 3-exit tuned has the best F1.
- Do not claim TATA is early-exit ready. The 5-exit dynamic policy saves compute but quality is too low.
- Do not describe weak 1-sec labels as perfect segment-level labels. They inherit parent clip labels and are weak supervision.
- Keep `3-exit tuned threshold` as the current best TATA baseline.
- Keep future work focused on synthetic mixtures, balancing, calibration, and raw pseudo-label routing.
