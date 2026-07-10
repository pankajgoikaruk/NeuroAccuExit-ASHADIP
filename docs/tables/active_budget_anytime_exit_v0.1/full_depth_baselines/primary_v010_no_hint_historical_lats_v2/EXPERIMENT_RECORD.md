# Primary Full-Depth Baseline Experiment Record

## Identity

- Method: **v0.10 no-hint + frozen historical LATS-v2**
- Role: canonical primary full-depth baseline
- Task: human-talk multi-label speaker and acoustic-context classification
- Segments: 4,335
- Parent clips: 867
- Labels: 10
- Full-depth output: Exit 3
- Hint passing: disabled

This is the only baseline used for standard Early-Exit, budget-aware Early-Exit,
anytime-inference, quality-versus-cost, latency, compute-saving, and exit-depth comparisons.

## Model and training settings

Source run: `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845`

| Setting | Value |
|---|---|
| Architecture | Three-exit ExitNet / TinyAudioCNN |
| Tap blocks | 1 and 3 |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Training threshold | 0.5 |
| Device | CPU |
| Output used | Exit 3 |

## Holdout evaluation settings

| Setting | Value |
|---|---|
| Evaluation batch size | 128 |
| Device | CPU |
| Initial threshold mode | `fixed_0p5` |
| Preliminary aggregation | `mean` |
| Probability columns | `exit3_prob_<label>` |
| Parent ID column | `parent_clip_id` |

The initial evaluation generated final-exit probabilities for 4,335 segments.
The frozen LATS-v2 rules aggregated these into 867 parent-level predictions.

## Frozen LATS-v2 rules

| Label | Aggregation | Threshold |
|---|---|---:|
| Brene Brown | p75 | 0.54 |
| Eckhart Tolle | top3mean | 0.50 |
| Eric Thomas | top4mean | 0.62 |
| Gary Vee | mean | 0.50 |
| Jay Shetty | p75 | 0.91 |
| Nick Vujicic | p75 | 0.34 |
| Other speaker present | noisy_or | 0.94 |
| Music present | mean | 0.37 |
| Audience reaction present | top3mean | 0.23 |
| Silence present | p75 | 0.42 |

Frozen replay performs no retraining, no threshold search, and no aggregation search.

## Canonical result

| Metric | Exact value | Paper value |
|---|---:|---:|
| Macro-F1 | 0.8623815322 | 0.8624 |
| Micro-F1 | 0.9531311540 | 0.9531 |
| Samples-F1 | 0.9588894381 | 0.9589 |
| Exact Match | 0.8765859285 | 0.8766 |
| Hamming Loss | 0.0137254902 | 0.0137 |
| Average predicted labels | 1.4590542099 | 1.4591 |
| Parent clips | 867 | 867 |

`1.4591` is average predicted labels, not average exit depth.

## Reporting limitation

Report this as a frozen corrected-holdout result, not as performance on an
independent external unseen-test set.
