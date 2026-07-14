# v0.11_EE Experiment Setup

## Objective

Determine whether the trained three-exit human-talk model can make reliable predictions before full depth while genuinely skipping unnecessary CNN blocks.

## Canonical model

```text
main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845
```

| Setting | Value |
|---|---|
| Backbone | TinyAudioCNN |
| Backbone blocks | 5 |
| Tap blocks | `(1, 3)` |
| Exits | 3 |
| Hint passing | disabled |
| Labels | 10 |
| Input feature | log-mel |
| Observed input shape | `[B, 1, 64, 101]` |
| Model modification | none |
| Retraining | none |

## Exit mapping

| Exit | Cumulative blocks | Approx. cumulative compute |
|---|---|---:|
| 1 | Block 1 | 3.6% |
| 2 | Blocks 1–3 | 35.7% |
| 3 | Blocks 1–5 | 100% |

Compute percentages are architecture-derived estimates, not measured latency ratios.

## Data

| Split/use | Rows |
|---|---:|
| Training | 25,519 |
| Validation policy tuning | 1,883 |
| Original model test split | 1,961 |
| Corrected holdout segments | 4,335 |
| Corrected holdout parents | 867 |

Key paths:

```text
Training manifest:
human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/
final_expanded_training_dataset_balanced/metadata/
multilabel_features_manifest_balanced.csv

Corrected holdout:
human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/
corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv

Corrected holdout features:
human_talk_workspace/tata_v0.6_raw_pipeline/
final_holdout_feature_cache/features

Label schema:
configs/human_talk_10label_schema.json
```

## Canonical parent policy

Historical frozen LATS-v2:

| Label | Aggregation | Threshold |
|---|---|---:|
| Brene Brown | `p75` | 0.54 |
| Eckhart Tolle | `top3mean` | 0.50 |
| Eric Thomas | `top4mean` | 0.62 |
| Gary Vee | `mean` | 0.50 |
| Jay Shetty | `p75` | 0.91 |
| Nick Vujicic | `p75` | 0.34 |
| Other speaker present | `noisy_or` | 0.94 |
| Music present | `mean` | 0.37 |
| Audience reaction present | `top3mean` | 0.23 |
| Silence present | `p75` | 0.42 |

For fixed Exit 1 and Exit 2, these rules are transferred unchanged as a diagnostic.

## Equivalence protocol

1. Build the original `ExitNet`.
2. Load the canonical checkpoint into the original model.
3. Wrap it with `AnytimeExitNet`.
4. Evaluate the unchanged full forward and staged forward on the same real features.
5. Compare every exit using strict tolerances.

Result: zero observed difference for logits and probabilities at all exits.

## Fixed-exit protocol

The complete corrected-holdout sample set is evaluated three times:

```text
Scenario A: every sample uses Exit 1
Scenario B: every sample uses Exit 2
Scenario C: every sample uses Exit 3
```

Segment decisions use threshold 0.5. Parent results use frozen LATS-v2 transfer.

## Dynamic policy search

The first policy searches only Exit-2 stopping. Exit 1 always continues.

Candidate grids:

```text
confidence = 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
margin     = 0.00, 0.02, 0.05, 0.08, 0.10, 0.15
```

Constraints:

```text
maximum absolute validation parent Macro-F1 drop = 0.02
minimum validation Exit-2 fraction               = 0.05
```

Selected rule:

```text
Exit1 label set == Exit2 label set
Exit2 label set is non-empty
mean binary confidence at Exit2 >= 0.55
minimum decision margin at Exit2 >= 0.00
```

Threshold mode:

```text
fixed_0p5
```

Reason: the canonical run did not contain a per-exit threshold-comparison JSON.

## Genuine staged evaluation

Each batch reaches Exit 2. The staged state is sliced to retain only samples that fail the stopping rule. Only that active subset continues through Blocks 4–5.

Stop reasons:

```text
reliable_early_exit
final_exit
```

No holdout label is used to modify the frozen policy.

## Metrics

Quality:

- Macro-F1;
- Micro-F1;
- Samples-F1;
- Exact Match;
- Hamming Loss;
- per-label precision, recall, and F1.

Efficiency:

- exit distribution;
- average exit depth;
- architecture-estimated FLOPs;
- model-only latency.

## Reproducibility boundary

The scripts reproduce the experiment when the canonical checkpoint, manifests, features, and historical LATS configuration are available locally. Large runtime predictions are not committed.
