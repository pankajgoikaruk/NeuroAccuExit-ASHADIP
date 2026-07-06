# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing v0.10 / v0.10_1

This documentation records the latest human-talk speaker/context multi-label experiments, including **LATS re-optimization**, **hint-pass**, **seed stability**, **pos_weight cap5**, and the **v0.10_1 low-energy recovery ablation**.

The current selected outcome is **v0.10 no-hint + LATS-v2 coordinate re-optimized**. Although seed stability is not perfect, it gives the strongest single-run global multi-label consistency. v0.10_1 low-energy recovery is retained as a valid negative/diagnostic ablation because it did not beat v0.10.

---

## Current final decision

| Decision item | Outcome |
|---|---|
| Selected final result | `v0.10 no-hint + LATS-v2 coordinate re-optimized` |
| Stable reference baseline | `v0.9_4 / frozen LATS-v2` |
| v0.10 seed-stability caveat | Slightly unstable, but selected as best current outcome |
| Hint-pass status | Rejected for current dataset; useful negative result |
| `pos_weight cap5` status | Rejected; worse than v0.10 no-hint and v0.9_4 reference |
| v0.10_1 low-energy recovery status | Rejected as final model; useful negative/diagnostic ablation |
| Strongest contribution | LATS-v2 metric-aware parent-level inference optimization |
| Next action | Report/document results; avoid more tuning unless testing a new hypothesis |

---

## Dataset and evaluation setting

| Item | Value |
|---|---|
| Task | Human-talk multi-label speaker/context detection |
| Parent clips | 867 |
| Segments | 4,335 |
| Original train rows | 25,519 |
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
| v0.9_4 baseline — frozen LATS-v2 | 0.8673 | 0.9458 | 0.9517 | 0.8604 | 0.0158 | 1.4452 | Stable reference baseline |
| v0.10 no-hint — frozen old LATS-v2 transfer | 0.8452 | 0.9247 | 0.9252 | 0.8062 | 0.0212 | 1.3495 | Frozen policy transfer failed |
| v0.10 no-hint — LATS-v1 re-optimized | 0.8658 | 0.9506 | 0.9562 | 0.8674 | 0.0145 | 1.4717 | Strong single-run recovery |
| **v0.10 no-hint — LATS-v2 coordinate re-optimized** | **0.8624** | **0.9531** | **0.9589** | **0.8766** | **0.0137** | **1.4591** | **Selected current final outcome** |
| v0.10 hint-pass — frozen old LATS-v2 transfer | 0.8180 | 0.9155 | 0.9225 | 0.7878 | 0.0242 | 1.3956 | Hint-pass transfer weak |
| v0.10 hint-pass — LATS-v1 re-optimized | 0.8634 | 0.9447 | 0.9535 | 0.8639 | 0.0160 | 1.4291 | Recovered but not best |
| v0.10 hint-pass — LATS-v2 coordinate re-optimized | 0.8632 | 0.9440 | 0.9536 | 0.8570 | 0.0164 | 1.4556 | Rejected vs no-hint |
| v0.10 no-hint + pos_weight cap5 — fixed 0.5 mean | 0.8009 | 0.8939 | 0.9088 | 0.7232 | 0.0330 | 1.6401 | Over-predicted positives before LATS |
| v0.10 no-hint + pos_weight cap5 — LATS-v2 macro-priority | 0.8511 | 0.9413 | 0.9481 | 0.8478 | 0.0171 | 1.4371 | Rejected; worse than v0.10 no-hint |
| v0.10_1 low-energy recovery — LATS-v2 coordinate re-optimized | 0.8581 | 0.9446 | 0.9519 | 0.8570 | 0.0160 | 1.4268 | Rejected as final; useful diagnostic ablation |

---

## Main conclusion

```text
Select v0.10 no-hint + LATS-v2 coordinate re-optimized as the current final outcome.
Keep v0.9_4 LATS-v2 as the stable reference baseline.
Reject current standard hint-pass.
Reject pos_weight cap5.
Reject v0.10_1 low-energy recovery as a final model.
Document v0.10_1 as a valid negative/diagnostic ablation.
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

The selected v0.10 no-hint result is the strongest single-run global-consistency result, but the seed mean is slightly weaker than the best single run. This caveat should be reported honestly.

---

## v0.10_1 low-energy recovery finding

The v0.10_1 ablation tested whether manually reviewed TATA-LAWYER low-energy one-second samples improve difficult labels such as `silence_present` and `audience_reaction_present`.

Build summary:

```text
base_rows = 29363
selected_low_energy_rows = 667
final_rows = 30030
holdout_parent_overlap = 0
feature_resolution_mode = feat_relpath
```

Final LATS-v2 result:

```text
Macro-F1   = 0.8581
Micro-F1   = 0.9446
Samples-F1 = 0.9519
Exact      = 0.8570
Hamming    = 0.0160
```

The experiment was valid but did not beat the selected v0.10 no-hint result.

---

## Research questions and findings

| ID | Research question | Finding |
|---|---|---|
| RQ1 | Does frozen v0.9_4 LATS-v2 transfer directly to v0.10 probabilities? | No. Frozen transfer is weak for both no-hint and hint-pass. |
| RQ2 | Does v0.10-specific LATS re-optimization recover performance? | Yes. Re-optimized LATS improves both no-hint and hint-pass outputs. |
| RQ3 | Does hint-pass beat no-hint? | No. Current standard hint-pass is weaker than no-hint after recalibration. |
| RQ4 | Is v0.10 no-hint better than v0.9_4? | It is selected as the best single-run global-consistency result, but seed stability should be reported as a caveat. |
| RQ5 | Does `pos_weight cap5` improve rare-label Macro-F1 enough to help? | No. It lowers overall performance and should be rejected. |
| RQ6 | Does low-energy recovery improve the final corrected-holdout parent-level result? | No. v0.10_1 is useful as a negative/diagnostic ablation only. |
| RQ7 | What is the strongest final contribution? | LATS-v2 metric-aware inference-policy optimization. |

---

## Updated documentation package

Detailed files are under:

```text
docs/v0.10/
docs/v0.10_1/
docs/tables/agentic_data_preprocessing_v0.10/
```
