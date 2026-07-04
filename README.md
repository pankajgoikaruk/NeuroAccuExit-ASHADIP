# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing v0.10

This repository documentation now records the v0.10 human-talk speaker/context experiment. The aim of v0.10 was to test whether the old **exit-to-exit hint-pass** idea improves the current multi-label ASHADIP/LABLEX pipeline when combined with parent-level LATS inference optimisation.

The outcome is an important controlled finding: **standard hint-pass should not be accepted as the final method for the current human-talk multi-label dataset.** The strongest stable result remains **v0.9_4 LATS-v2**, while v0.10 no-hint is useful as a diagnostic/stability ablation.

---

## Current final decision

| Decision item | Outcome |
|---|---|
| Stable final baseline | `v0.9_4 / LATS-v2 metric-aware coordinate search` |
| Best single v0.10 run | `v0.10 no-hint + LATS-v2 coordinate re-optimized` |
| Hint-pass status | Rejected for current dataset; useful negative result |
| Reason | Hint-pass did not outperform no-hint after recalibration and was weaker in global consistency |
| Next research action | Document v0.10 as diagnostic; do not continue more tuning unless testing label-aware/gated hinting |

---

## Dataset and evaluation setting

| Item | Value |
|---|---|
| Task | Human-talk multi-label speaker/context detection |
| Parent clips | 867 |
| Segments | 4,335 |
| Train rows | 25,519 |
| Validation rows | 1,883 |
| Test rows | 1,961 |
| Labels | 10 |
| Label schema | `configs/human_talk_10label_schema.json` |
| Training manifest | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/final_expanded_training_dataset_balanced/metadata/multilabel_features_manifest_balanced.csv` |
| Corrected holdout manifest | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv` |
| Parent ID column | `parent_clip_id` |
| Probability prefix | `exit3_prob_` |

Labels:

```text
Brene_Brown, Eckhart_Tolle, Eric_Thomas, Gary_Vee, Jay_Shetty, Nick_Vujicic,
other_speaker_present, music_present, audience_reaction_present, silence_present
```

---

## Main result comparison

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

## Seed-stability check for v0.10 no-hint

| Seed | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Avg pred labels |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.8624 | 0.9353 | 0.9467 | 0.8374 | 0.0189 | 1.4556 |
| 202 | 0.8741 | 0.9471 | 0.9565 | 0.8674 | 0.0153 | 1.4314 |
| 303 | 0.8607 | 0.9492 | 0.9560 | 0.8731 | 0.0149 | 1.4614 |
| **Mean** | **0.8657** | **0.9439** | **0.9531** | **0.8593** | **0.0164** | **1.4494** |
| Std | 0.0073 | 0.0075 | 0.0055 | 0.0192 | 0.0022 | 0.0159 |

### Stability conclusion

The v0.10 no-hint model can improve global metrics in some seeds, but the result is not stable enough to replace the v0.9_4 final baseline.

Compared with v0.9_4:

- Seed 202 and Seed 303 are promising.
- Seed 101 is clearly weaker.
- The cross-seed mean is below v0.9_4 on Macro-F1 and Micro-F1 and slightly worse on Hamming Loss.

Therefore, v0.10 no-hint should be recorded as a **stability/diagnostic ablation**, not the new final method.

---

## Research questions and findings

| ID | Research question | Finding |
|---|---|---|
| RQ1 | Does old frozen v0.9_4 LATS-v2 transfer directly to v0.10 probabilities? | No. Frozen transfer is weak for both no-hint and hint-pass, showing that v0.10 probability distributions changed. |
| RQ2 | Does v0.10-specific label-wise aggregation and threshold search recover performance? | Yes. Re-optimized LATS greatly improves both v0.10 no-hint and hint-pass results. |
| RQ3 | Does hint-pass beat the no-hint control? | No. Hint-pass remains weaker than no-hint after both LATS-v1 and LATS-v2 re-optimization. |
| RQ4 | Is v0.10 no-hint better than v0.9_4? | Only as a single-run trade-off: it improves Micro-F1, Samples-F1, Exact Match, and Hamming Loss, but loses Macro-F1. Across seeds, the improvement is not stable. |
| RQ5 | What is the strongest scientific contribution? | LATS-v2 metric-aware inference optimisation remains the strongest stable contribution. Hint-pass is a useful negative result for multi-label speaker/context audio. |

---

## Why no-hint v0.10 sometimes improved without hint-pass

v0.10 no-hint is not identical to v0.9_4. It is a newly trained model checkpoint, so the segment-level probability distribution changes due to training variation, checkpoint selection, and run-specific calibration. The improvement in the best v0.10 no-hint run comes mainly from **v0.10-specific LATS re-optimization**, not from architecture changes.

In short:

```text
v0.10 no-hint improvement = retrained probability distribution + new LATS policy search
not hint-pass
```

---

## Why hint-pass did not transfer well

Hint-pass worked better in the older moth setting because that task was closer to single-label or binary classification. In human-talk multi-label detection, early exits may see only incomplete evidence. Passing early-exit probabilities forward can propagate partial or biased label beliefs, especially for rare/bursty labels such as `audience_reaction_present` and `silence_present`.

The current finding is:

```text
Standard exit-to-exit hint-pass does not reliably improve multi-label speaker/context detection.
Future versions need label-aware or gated hinting, not plain previous-exit probability hints.
```

---

## Recommended next step

Stop additional v0.10 tuning for now and write the report around these decisions:

1. Freeze **v0.9_4 LATS-v2** as the stable main result.
2. Include **v0.10 no-hint seed stability** as an ablation.
3. Include **v0.10 hint-pass** as a negative/diagnostic result.
4. Propose future work: label-aware hinting, speaker-only hints, Exit2-to-Exit3 hints only, or gated hint fusion.

---

## Updated documentation package

The v0.10 documentation is under:

```text
docs/v0.10/
```

Key result tables are under:

```text
docs/tables/agentic_data_preprocessing_v0.10/
```
