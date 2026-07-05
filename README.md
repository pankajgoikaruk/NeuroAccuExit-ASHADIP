# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing v0.10

This documentation records the latest human-talk speaker/context multi-label experiments for v0.9_4 and v0.10, including **hint-pass**, **LATS re-optimization**, **seed stability**, and the latest **no-hint + `pos_weight` cap5 diagnostic run**.

The final outcome is unchanged: **v0.9_4 LATS-v2 remains the stable final result**. v0.10 no-hint remains a useful diagnostic/stability ablation. Current standard hint-pass and `pos_weight cap5` should not be adopted as final methods.

---

## Current final decision

| Decision item | Outcome |
|---|---|
| Stable final baseline | `v0.9_4 / LATS-v2 metric-aware coordinate search` |
| Best single v0.10 run | `v0.10 no-hint + LATS-v2 coordinate re-optimized` |
| Stable v0.10 replacement? | No; seed stability is not strong enough |
| Hint-pass status | Rejected for current dataset; useful negative result |
| `pos_weight cap5` status | Rejected; worse than v0.9_4 and worse than previous v0.10 no-hint results |
| Strongest contribution | LATS-v2 metric-aware parent-level inference optimization |
| Next action | Report/document results; avoid more tuning unless testing a new hypothesis |

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

## Complete method comparison

| method | macro_f1 | micro_f1 | samples_f1 | exact_match | hamming_loss | avg_pred_labels | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v0.9_4 baseline — frozen LATS-v2 | 0.8673 | 0.9458 | 0.9517 | 0.8604 | 0.0158 | 1.4452 | Stable final baseline |
| v0.10 no-hint — frozen old LATS-v2 transfer | 0.8452 | 0.9247 | 0.9252 | 0.8062 | 0.0212 | 1.3495 | Frozen policy transfer failed |
| v0.10 no-hint — LATS-v1 re-optimized | 0.8658 | 0.9506 | 0.9562 | 0.8674 | 0.0145 | 1.4717 | Strong single-run recovery |
| v0.10 no-hint — LATS-v2 coordinate re-optimized | 0.8624 | 0.9531 | 0.9589 | 0.8766 | 0.0137 | 1.4591 | Best single v0.10 global-consistency run |
| v0.10 hint-pass — frozen old LATS-v2 transfer | 0.8180 | 0.9155 | 0.9225 | 0.7878 | 0.0242 | 1.3956 | Hint-pass transfer weak |
| v0.10 hint-pass — LATS-v1 re-optimized | 0.8634 | 0.9447 | 0.9535 | 0.8639 | 0.0160 | 1.4291 | Recovered but not best |
| v0.10 hint-pass — LATS-v2 coordinate re-optimized | 0.8632 | 0.9440 | 0.9536 | 0.8570 | 0.0164 | 1.4556 | Rejected vs no-hint |
| v0.10 no-hint + pos_weight cap5 — fixed 0.5 mean | 0.8009 | 0.8939 | 0.9088 | 0.7232 | 0.0330 | 1.6401 | Over-predicted positives before LATS |
| v0.10 no-hint + pos_weight cap5 — LATS-v2 macro-priority | 0.8511 | 0.9413 | 0.9481 | 0.8478 | 0.0171 | 1.4371 | Rejected; worse than baseline and no-hint |

---

## Main conclusion

```text
Keep v0.9_4 LATS-v2 as the stable final baseline.
Reject current standard hint-pass.
Reject pos_weight cap5.
Use v0.10 no-hint only as diagnostic/stability evidence.
```

---

## v0.10 no-hint seed-stability check

| seed | macro_f1 | micro_f1 | samples_f1 | exact_match | hamming_loss | avg_pred_labels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.8624 | 0.9353 | 0.9467 | 0.8374 | 0.0189 | 1.4556 |
| 202 | 0.8741 | 0.9471 | 0.9565 | 0.8674 | 0.0153 | 1.4314 |
| 303 | 0.8607 | 0.9492 | 0.9560 | 0.8731 | 0.0149 | 1.4614 |

Mean result:

```text
Macro-F1   = 0.8657
Micro-F1   = 0.9439
Samples-F1 = 0.9531
Exact      = 0.8593
Hamming    = 0.0164
```

Compared with v0.9_4, the seed mean is slightly weaker in Macro-F1, Micro-F1, Exact Match, and Hamming Loss. Therefore v0.10 no-hint is promising but not stable enough to replace v0.9_4.

---

## Latest `pos_weight cap5` finding

The `pos_weight cap5` experiment was intended to test whether label-imbalance-aware BCE can improve rare labels more reliably than hint-pass. It did **not** help.

Important note:

```text
The produced run is seed_101202303, so it is a single diagnostic run, not a true 3-seed stability run.
```

Best `pos_weight cap5` after LATS-v2 macro-priority re-optimization:

```text
Macro-F1   = 0.8511
Micro-F1   = 0.9413
Samples-F1 = 0.9481
Exact      = 0.8478
Hamming    = 0.0171
```

This is worse than v0.9_4 and worse than the previous no-hint seed mean.

---

## Research questions and findings

| ID | Research question | Finding |
|---|---|---|
| RQ1 | Does frozen v0.9_4 LATS-v2 transfer directly to v0.10 probabilities? | No. Frozen transfer is weak for both no-hint and hint-pass. |
| RQ2 | Does v0.10-specific LATS re-optimization recover performance? | Yes. Re-optimized LATS improves both no-hint and hint-pass outputs. |
| RQ3 | Does hint-pass beat no-hint? | No. Current standard hint-pass is weaker than no-hint after recalibration. |
| RQ4 | Is v0.10 no-hint better than v0.9_4? | Only in some single-run/global metrics, not stably across seeds. |
| RQ5 | Does `pos_weight cap5` improve rare-label Macro-F1 enough to help? | No. It lowers overall performance and should be rejected. |
| RQ6 | What is the strongest final contribution? | LATS-v2 metric-aware inference-policy optimization. |

---

## Updated documentation package

Detailed files are under:

```text
docs/v0.10/
docs/tables/agentic_data_preprocessing_v0.10/
```
