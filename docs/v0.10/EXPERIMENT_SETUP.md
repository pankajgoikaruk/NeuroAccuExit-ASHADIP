# v0.10 Experimental Settings

## Dataset

| Item | Value |
|---|---|
| Task | Human-talk multi-label speaker/context detection |
| Labels | 10 |
| Train rows | 25,519 |
| Validation rows | 1,883 |
| Test rows | 1,961 |
| Corrected holdout parent clips | 867 |
| Corrected holdout segments | 4,335 |
| Labels JSON | `configs/human_talk_10label_schema.json` |
| Training manifest | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/final_expanded_training_dataset_balanced/metadata/multilabel_features_manifest_balanced.csv` |
| Corrected holdout manifest | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv` |

---

## Base model settings

| Setting | Value |
|---|---|
| Training module | `python -m training.train_multilabel` |
| Number of exits | 3 |
| Tap blocks | `1,3` |
| Loss weights | `[0.3, 0.3, 1.0]` |
| Epochs | 40 |
| Batch size | 64 |
| LR | 0.001 |
| Device | CPU in reported runs |
| Raw threshold | 0.5 |
| Parent aggregation before LATS | mean |

---

## Hint-pass settings tested

| Setting | Value |
|---|---|
| `exit_hint.enable` | True for hint-pass; False for no-hint |
| `hint_dim` | 8 |
| `hint_source` | `probs` |
| `hint_activation` | `sigmoid` for multi-label |
| `hint_detach` | True |
| `hint_use_stats` | True |

Result: current standard hint-pass did not beat no-hint.

---

## Pos-weight settings tested

| Setting | Value |
|---|---|
| Model | v0.10 no-hint |
| Loss change | `BCEWithLogitsLoss(pos_weight=...)` |
| Formula | `negative_count / positive_count` |
| Cap | `PosWeightMax = 5.0` |
| Run label | `main_v010_no_hint_posweight_cap5_seed_101202303` |
| True stability? | No; this is a single diagnostic run, not valid 3-seed stability |

Positive weights used:

| Label | pos_weight |
|---|---:|
| Brene_Brown | 5.0000 |
| Eckhart_Tolle | 5.0000 |
| Eric_Thomas | 5.0000 |
| Gary_Vee | 5.0000 |
| Jay_Shetty | 5.0000 |
| Nick_Vujicic | 5.0000 |
| other_speaker_present | 2.2188 |
| music_present | 1.5105 |
| audience_reaction_present | 5.0000 |
| silence_present | 5.0000 |

---

## LATS settings

| Setting | Value |
|---|---|
| Parent ID column | `parent_clip_id` |
| Probability prefix | `exit3_prob_` |
| Threshold range | 0.10 to 0.95 |
| Threshold step | 0.01 |
| Aggregations | `mean,max,top2mean,top3mean,p75,p90` |
| LATS-v1 | Independent label-wise optimization |
| LATS-v2 | Metric-aware coordinate optimization |
| Pos-weight LATS objective | `macro_priority` |
