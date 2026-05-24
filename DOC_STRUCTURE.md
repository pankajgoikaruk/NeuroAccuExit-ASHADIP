# Documentation Structure — agentic_data_preprocessing_v0.4

This document defines the active documentation structure for the **`agentic_data_preprocessing_v0.4`** branch. Old branch results should not be mixed into this documentation set.

```text
Branch: agentic_data_preprocessing_v0.4
Agenda: Agentic AI-based data preprocessing plus softmax-vs-sigmoid ablation on cleaned Raw5 human-talk data
Dataset stage: raw5_agentic_cleaned
Task: five-speaker human-talk speaker classification
Models: TinyAudioCNN + ExitNet, 3-exit and 5-exit
Classes: Brene_Brown, Eckhart_Tolle, Eric_Thomas, Gary_Vee, Jay_Shetty
Final cleaned files: 3,108
```

## Current documentation scope

The branch documentation should cover:

1. Agentic AI preprocessing motivation and workflow.
2. Raw5 audit, accepted/review/rejected/blocked routing, and final cleaned dataset status.
3. Softmax 3-exit and 5-exit baselines on `raw5_agentic_cleaned`.
4. Sigmoid/BCE one-hot ablation for 3-exit and 5-exit.
5. Threshold tuning and label-set-stability policy results for sigmoid.
6. Final comparison and paper-safe interpretation.
7. Future TinyAudioTriageAgent direction for true multi-label audio-content triage.

## Recommended paper/report sections

| Section | Content to include |
|---|---|
| Motivation | Raw speaker data can contain silence, music, interviewer speech, applause, and other non-target audio. |
| Agentic preprocessing | DatasetAuditorAgent, ManifestBuilderAgent, DatasetBuilderAgent, final re-audit, manual exclusion. |
| Dataset | Raw5 classes, final training counts, traceability and exclusion policy. |
| Softmax baseline | 3-exit and 5-exit cross-entropy results; segment policy and clip policy. |
| Sigmoid ablation | BCE/sigmoid one-hot setup; fixed vs tuned threshold outcomes. |
| Discussion | Why softmax remains best for mutually exclusive speaker classification; why sigmoid belongs to TATA. |
| Future work | TinyAudioTriageAgent with true multi-label target speaker / other speaker / music / silence / applause / laughter tags. |

## Research questions

| ID | Research question | Answer from v0.4 |
| --- | --- | --- |
| RQ1 | Can agentic preprocessing create a safe, traceable Raw5 speaker dataset? | Yes: 3,108 final training-ready files were produced after non-destructive audit, build, re-audit, and one manual exclusion. |
| RQ2 | Does softmax remain suitable for cleaned human-talk speaker classification? | Yes: softmax 3-exit achieved the strongest final result and best dynamic policy balance. |
| RQ3 | Can the same architecture operate with sigmoid/BCE one-hot supervision? | Yes: sigmoid ablation worked, with best fixed 3-exit Macro-F1 = 0.9692. |
| RQ4 | Should sigmoid replace softmax for the main speaker model? | No: sigmoid is threshold-sensitive and less efficient for early exits; it is better reserved for true multi-label triage. |
| RQ5 | Which setting gives the best clip-level efficiency? | Softmax 3-exit Depth×Time: 0.9893 clip accuracy with 75.82% compute saved. |
| RQ6 | What is the next research step? | TinyAudioTriageAgent: true sigmoid/BCE multi-label content triage for target speaker, other speaker, music, silence, applause, and laughter. |

## Canonical result tables to keep

### Main final result table

| Setting | Model | Activation / loss | Final metric | Final Macro-F1 | Accuracy / Exact Match | Hamming loss | Policy metric | Avg exit depth | Compute saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Softmax | 3-exit | Softmax + CE | single-label accuracy | 0.9756 | 0.9760 | N/A | 0.9683 | 2.0886 | 52.56% |
| Softmax | 5-exit | Softmax + CE | single-label accuracy | 0.9610 | 0.9616 | N/A | 0.9520 | 2.7144 | 62.03% |
| Sigmoid fixed | 3-exit | Sigmoid + BCE | one-hot exact match | 0.9692 | 0.9535 | 0.0121 | N/A | N/A | N/A |
| Sigmoid fixed | 5-exit | Sigmoid + BCE | one-hot exact match | 0.9627 | 0.9426 | 0.0148 | N/A | N/A | N/A |
| Sigmoid tuned | 3-exit | Sigmoid + BCE | one-hot exact match | 0.9670 | 0.9505 | 0.0131 | 0.9670 | 3.0000 | 0.00% |
| Sigmoid tuned | 5-exit | Sigmoid + BCE | one-hot exact match | 0.9647 | 0.9465 | 0.0139 | 0.9561 | 3.4537 | 30.93% |

### Dynamic policy comparison

| Setting | Model | Policy | Metric | Avg exit depth | Compute saved | Exit consistency | Main observation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Softmax | 3-exit | Greedy confidence | accuracy 0.9683 | 2.0886 | 52.56% | 0.9913 | Best segment-level policy balance |
| Softmax | 5-exit | Greedy confidence | accuracy 0.9520 | 2.7144 | 62.03% | 0.9849 | Highest segment-depth compute saving |
| Sigmoid tuned | 3-exit | Label-set stability k=2 | macro-F1 0.9670 | 3.0000 | 0.00% | 1.0000 | No early exit; all samples reach final exit |
| Sigmoid tuned | 5-exit | Label-set stability k=2 | macro-F1 0.9561 | 3.4537 | 30.93% | 0.9673 | Works but less efficient than softmax |

### Clip-level softmax table

| Model | Clip policy | Clip accuracy | Avg windows used | Avg total windows | Windows saved | Compute saved |
| --- | --- | --- | --- | --- | --- | --- |
| Softmax 3-exit | Full-window aggregation | 0.9957 | 8.6510 | 8.6510 | 0.00% | 0.00% |
| Softmax 3-exit | Depth×Time | 0.9893 | 2.0878 | 8.6510 | 75.87% | 75.82% |
| Softmax 5-exit | Full-window aggregation | 0.9850 | 8.6510 | 8.6510 | 0.00% | 0.00% |
| Softmax 5-exit | Depth×Time | 0.9764 | 2.1649 | 8.6510 | 74.98% | 74.64% |

### Best model by goal

| Goal | Best current choice | Reason |
| --- | --- | --- |
| Best final segment classification | Softmax 3-exit | Final accuracy 0.9760, Macro-F1 0.9756 |
| Best segment-level dynamic early exit | Softmax 3-exit | Accuracy 0.9683 with 52.56% compute saving |
| Highest segment-depth compute saving | Softmax 5-exit | 62.03% compute saving |
| Best clip-level accuracy | Softmax 3-exit full aggregation | Clip accuracy 0.9957 |
| Best clip-level efficiency | Softmax 3-exit Depth×Time | Clip accuracy 0.9893 with 75.82% compute saving |
| Best sigmoid final result | Sigmoid 3-exit fixed | Macro-F1 0.9692 |
| Best sigmoid tuned result | Sigmoid 5-exit tuned | Macro-F1 0.9647 |
| Future true multi-label preprocessing | TinyAudioTriageAgent | Use sigmoid/BCE for co-existing tags: target speaker, other speaker, music, silence, applause, laughter |

## Figure assets

Use the following generated figures in README, appendix, and reports:

| Figure | Path | Purpose |
|---|---|---|
| Final Macro-F1 comparison | `docs/figures/human_talk/agentic_data_preprocessing_v0.4/final_macro_f1_comparison.png` | Compare softmax/sigmoid final exits. |
| Dynamic policy quality vs compute | `docs/figures/human_talk/agentic_data_preprocessing_v0.4/dynamic_policy_quality_compute.png` | Show early-exit quality and compute-saving tradeoff. |
| Softmax clip policy comparison | `docs/figures/human_talk/agentic_data_preprocessing_v0.4/softmax_clip_policy_comparison.png` | Show full-window vs Depth×Time clip results. |
| Per-exit Macro-F1 comparison | `docs/figures/human_talk/agentic_data_preprocessing_v0.4/per_exit_macro_f1_comparison.png` | Show how exit quality changes with depth. |

## Documentation rule

Do not claim that 5-exit is more accurate than 3-exit. In this branch, the strongest accuracy result is softmax 3-exit, while 5-exit is mainly useful for dynamic exit behaviour and compute-saving analysis. Do not describe the sigmoid one-hot ablation as true multi-label learning; true multi-label learning is reserved for the future TinyAudioTriageAgent.
