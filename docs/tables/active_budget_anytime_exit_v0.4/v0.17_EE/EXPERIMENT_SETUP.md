# v0.17_EE Experiment Setup

## Research questions

| ID | Research question |
|---|---|
| RQ1 | Can a fully sequential controller use Exit 1, Exit 2, and the final exit rather than solving only an Exit-2/Exit-3 decision? |
| RQ2 | Can the same sequential policy formulation scale from three exits to five exits? |
| RQ3 | Does a safety-buffered Pareto-knee rule improve robustness over v0.16's maximum-compute Pareto selection? |
| RQ4 | Which decision signals are necessary for safe multi-label early exit: confidence, label margins, stability, probability change, and label risk? |
| RQ5 | Does adding more exit opportunities improve the quality–computation frontier? |
| RQ6 | Is Exit 1 practically useful, or does it introduce disproportionate quality risk? |

## Checkpoints

| Architecture | Run | Tap blocks | Route | Training rows |
|---|---|---|---|---:|
| 3-exit | `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845` | `(1,3)` | `Exit 1 → Exit 2 → Exit 3` | 25,519 |
| 5-exit | `main_v06_expanded_5exit_20260603_210324` | `(1,2,3,4)` | `Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5` | 30,950 |

The checkpoints are frozen. v0.17 does not retrain the TinyAudioCNN backbone or exit heads.

## Data and evaluation

| Setting | Value |
|---|---:|
| Labels | 10 |
| Validation segments / parents | 1,883 / 304 |
| Corrected-holdout segments / parents | 4,335 / 867 |
| Parent identifier | `parent_clip_id` |
| Feature shape used for equivalence | `[8,1,64,101]` |
| Segment decision threshold mode | `fixed_0p5` |
| Parent evaluator | Frozen historical LATS-v2 |
| Batch size | 128 |
| Device | CPU |
| Torch/BLAS threads | 1 |
| Publication timing repetitions | 30 |

## Validation-manifest difference

| Architecture | Validation/training manifest |
|---|---|
| 3-exit | `tata_v0.8_human_corrected_balanced_pipeline/.../multilabel_features_manifest_balanced.csv` |
| 5-exit | `tata_v0.6_raw_pipeline/.../multilabel_features_manifest.csv` |

The common corrected holdout, feature cache, ten-label schema, LATS-v2 configuration, threshold mode, optimiser budget, constraints, and timing protocol were shared. The different training/validation manifests invalidate a direct architecture-superiority claim.

## Architecture costs

### 3-exit cumulative FLOPs

| Exit | Cumulative FLOPs | Normalised to final |
|---|---:|---:|
| Exit 1 | 1,861,952 | 0.0361 |
| Exit 2 | 18,451,072 | 0.3574 |
| Exit 3 | 51,629,312 | 1.0000 |

### 5-exit cumulative FLOPs

| Exit | Cumulative FLOPs | Normalised to final |
|---|---:|---:|
| Exit 1 | 1,861,952 | 0.0361 |
| Exit 2 | 12,921,312 | 0.2503 |
| Exit 3 | 18,451,072 | 0.3574 |
| Exit 4 | 29,510,592 | 0.5716 |
| Exit 5 | 51,629,312 | 1.0000 |

## Optimiser

| Setting | Value |
|---|---:|
| Algorithm | Constraint-aware NSGA-II-style search |
| Population size | 96 |
| Generations | 60 |
| Random seed | 42 |
| Parent-grouped folds | 5 |
| Safety fraction | 0.75 |
| Minimum total early-exit fraction | 0.020 |
| Minimum Exit-1 fraction | 0.005 |
| 3-exit genes | 26 |
| 5-exit genes | 52 |
| 3-exit unique candidates / Pareto points | 5,847 / 19 |
| 5-exit unique candidates / Pareto points | 5,856 / 86 |

Each non-final exit owns one 13-value block:

```text
[mean confidence,
 maximum inter-exit probability delta,
 maximum label-risk score,
 ten label-specific decision margins]
```

## Quality constraints

| Metric | Maximum permitted degradation |
|---|---:|
| Parent Macro-F1 drop | 0.010 |
| Parent Micro-F1 drop | 0.005 |
| Parent Exact-Match drop | 0.010 |
| Parent Hamming increase | 0.002 |

Fold-level one-sided upper confidence checks are required. The selected candidate is a safety-buffered Pareto knee, not the maximum-compute boundary.

## Staged-equivalence result

Both checkpoints passed with maximum absolute logit and probability differences equal to `0.0`. This establishes that staged execution reproduces the original exit outputs before any stopping policy is applied.

## Evaluation protocol

1. Generate every exit's validation probabilities from the frozen checkpoint.
2. Derive validation-only label-risk weights.
3. Search stage-specific parameters under parent-level quality constraints.
4. Freeze the selected policy.
5. Execute genuine staged inference on the corrected holdout.
6. Remove stopped samples from the active batch before deeper blocks execute.
7. Evaluate parent predictions with the same frozen LATS-v2 configuration.
8. Run six ablations without holdout retuning.
9. Measure latency using 30 controlled repetitions.
10. Run the fairness audit before making any 3-exit versus 5-exit claim.
