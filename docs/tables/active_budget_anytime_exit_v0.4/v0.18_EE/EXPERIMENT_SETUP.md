# v0.18 Experiment Setup

## Research questions

1. Can five exits be compared fairly with the canonical three-exit model?
2. Does v0.17's strong five-exit result reproduce under matched training?
3. Can stricter Exit-1 and risk-label safeguards improve transfer?
4. Which safeguards are required for multi-label early exit?
5. Can either architecture satisfy the full quality-constraint set?

## Matched training

| Setting | 3-exit | 5-exit |
|---|---:|---:|
| Train / validation / test | 25,519 / 1,883 / 1,961 | 25,519 / 1,883 / 1,961 |
| Labels | 10 | 10 |
| Tap blocks | `1,3` | `1,2,3,4` |
| Epochs | 40 | 40 |
| Batch size | 64 | 64 |
| Learning rate | 0.001 | 0.001 |
| Seed | 42 | 42 |
| Threshold | 0.5 | 0.5 |
| Hint passing | Off | Off |
| Final loss weight | 1.0 | 1.0 |
| Auxiliary weights | `0.3,0.3` | `0.15,0.15,0.15,0.15` |
| Auxiliary budget | 0.60 | 0.60 |

The audit also verifies identical manifest, feature root, label schema, input dimensions, worker count, weight decay, and balance settings.

## Policy optimisation

| Setting | Value |
|---|---:|
| Algorithm | Constraint-aware NSGA-II-style |
| Population | 112 |
| Generations | 70 |
| Seed | 42 |
| Grouped folds | 5 |
| Safety fraction | 0.50 |
| Minimum early fraction | 0.020 |
| Minimum Exit-1 fraction | 0.0025 |
| Segment thresholds | Fixed 0.5 |

## Evaluation

| Item | Value |
|---|---:|
| Holdout segments | 4,335 |
| Parent clips | 867 |
| Parent aggregation | Frozen historical LATS-v2 |
| Batch size | 128 |
| Device | CPU |
| Threads | 1 |
| Timing repetitions | 30 |

## Constraints

| Metric | Maximum degradation |
|---|---:|
| Macro-F1 | 0.010 |
| Micro-F1 | 0.005 |
| Exact Match | 0.010 |
| Hamming Loss | +0.002 |
