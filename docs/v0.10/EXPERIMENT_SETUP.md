# v0.10 Experiment Setup

## Branch and goal

| Item | Value |
|---|---|
| Branch | `agentic_data_preprocessing_v0.10` |
| Goal | Test hint-pass and v0.10-specific LATS re-optimization for human-talk multi-label audio |
| Main comparison | v0.9_4 LATS-v2 vs v0.10 no-hint vs v0.10 hint-pass |
| Final decision | Keep v0.9_4 as stable final result; document v0.10 as ablation |

---

## Data

| Item | Value |
|---|---|
| Train manifest | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/final_expanded_training_dataset_balanced/metadata/multilabel_features_manifest_balanced.csv` |
| Corrected holdout | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv` |
| Labels JSON | `configs/human_talk_10label_schema.json` |
| Train rows | 25,519 |
| Validation rows | 1,883 |
| Test rows | 1,961 |
| Corrected parent clips | 867 |
| Corrected segments | 4,335 |

---

## Model settings

| Setting | Value |
|---|---|
| Training module | `python -m training.train_multilabel` |
| Exits | 3 |
| Tap blocks | `1,3` |
| Loss weights | `[0.3, 0.3, 1.0]` |
| Epochs | 40 |
| Batch size | 64 |
| LR | 0.001 |
| Threshold during raw evaluation | 0.5 |
| Device | CPU in reported runs |

---

## Hint-pass settings

| Setting | No-hint | Hint-pass |
|---|---|---|
| `exit_hint.enable` | false | true |
| `hint_dim` | 8 configured but unused | 8 |
| `hint_source` | probs configured but unused | probs |
| `hint_activation` | sigmoid configured but unused | sigmoid |
| `hint_detach` | true configured but unused | true |
| `hint_use_stats` | true configured but unused | true |

Important compatibility decision:

```text
Shared ExitNet default remains softmax for older single-label/moth experiments.
Human-talk v0.10 explicitly uses sigmoid for multi-label hint probabilities.
```

---

## LATS settings

| Setting | Value |
|---|---|
| Parent ID column | `parent_clip_id` |
| Probability prefix | `exit3_prob_` |
| Calibration splits | 20 |
| Calibration fraction | 0.5 |
| Threshold range | 0.10 to 0.95 |
| Threshold step | 0.01 |
| Aggregation methods used in seed run | `mean,max,top2mean,top3mean,p75,p90` |
| Additional methods in fast comparison | includes `top4mean`, `top5mean`, `median`, `noisy_or` where available |
| LATS-v1 | independent label-wise aggregation/threshold optimization |
| LATS-v2 | metric-aware coordinate search over full multi-label objective |

---

## Acceptance rule used

A v0.10 method should only replace v0.9_4 if it is stable across seeds and matches or improves:

```text
Macro-F1   >= 0.8673
Micro-F1   >= 0.9458
Samples-F1 >= 0.9517
Exact      >= 0.8604
Hamming    <= 0.0158
```

v0.10 no-hint passed parts of this in individual seeds, but not consistently across all seeds.
