# v0.16_EE Experiment Setup

## Research question

Can constraint-aware multi-objective optimisation find a **lightweight and interpretable per-label margin policy** that saves more genuine computation than v0.13 while preserving parent-level Macro-F1, Micro-F1, Exact Match, and Hamming Loss?

## Frozen model

| Item | Value |
|---|---|
| Model | ExitNet / five-block TinyAudioCNN |
| Taps | Blocks 1 and 3 |
| Exits | 3 |
| Eligible early stop | Exit 2 |
| Final exit | Exit 3 |
| Input | 1 × 64 × 101 log-mel tensor |
| Checkpoint | `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845/ckpt/best.pt` |
| Backbone/exit-head training in v0.16 | None; frozen |
| Hint passing | Disabled |

## Data

| Split | Segments | Parents | Purpose |
|---|---:|---:|---|
| Validation | 1,883 | 304 | Evolutionary search and grouped quality bounds |
| Corrected holdout | 4,335 | 867 | Frozen policy evaluation only |

Labels:

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
audience_reaction_present
silence_present
```

## Thresholds

The canonical run did not contain a per-exit threshold-comparison artifact, so segment-level Exit-1/2/3 predictions use fixed 0.5 thresholds. Parent metrics use the frozen historical LATS-v2 aggregation and thresholds.

## Policy

Let `p1,l` and `p2,l` be Exit-1 and Exit-2 probabilities for label `l`.

```text
confidence_l = max(p2,l, 1 - p2,l)
margin_l     = |p2,l - 0.5|
delta_l      = |p2,l - p1,l|
```

Stop at Exit 2 iff:

```text
thresholded label set at Exit 1 == thresholded label set at Exit 2
AND Exit-2 set is non-empty
AND mean(confidence_l) >= tau_conf
AND max(delta_l) <= tau_delta
AND margin_l >= m_l for every label l
```

The chromosome is:

```text
[tau_conf, tau_delta, m_1, ..., m_10]
```

## Evolutionary optimisation

| Setting | Value |
|---|---:|
| Algorithm | Constraint-aware NSGA-II-style evolutionary search |
| Population | 80 |
| Generations | 50 |
| Seed | 42 |
| Genes | 12 |
| Unique candidates | 4,078 |
| Pareto candidates | 20 |
| Crossover | blend crossover |
| Mutation | bounded Gaussian mutation |
| Constraint handling | Deb-style feasibility dominance |

Objectives:

1. maximise estimated FLOPs saved;
2. minimise upper-bound Parent Macro-F1 drop;
3. minimise upper-bound Parent Micro-F1 drop;
4. minimise upper-bound Parent Exact-Match drop;
5. minimise upper-bound Parent Hamming increase.

## Validation constraints

| Constraint | Limit |
|---|---:|
| Macro-F1 drop | 0.010 |
| Micro-F1 drop | 0.005 |
| Exact-Match drop | 0.010 |
| Hamming increase | 0.002 |
| Minimum Exit-2 fraction | 0.020 |
| Grouped folds | 5 |
| One-sided confidence `z` | 1.645 |

## Cost model

| Exit | Cumulative FLOPs |
|---|---:|
| Exit 1 | 1,861,952 |
| Exit 2 | 18,451,072 |
| Exit 3 | 51,629,312 |

Exit-2 stopping avoids 33,178,240 FLOPs per accepted segment, or 64.2624% of full-depth cumulative compute.

## Selected validation point

| Parameter | Value |
|---|---:|
| Mean confidence threshold | 0.678602930 |
| Maximum probability delta | 0.937177682 |
| Margin `Brene_Brown` | 0.004844179 |
| Margin `Eckhart_Tolle` | 0.003097406 |
| Margin `Eric_Thomas` | 0.024152159 |
| Margin `Gary_Vee` | 0.240166657 |
| Margin `Jay_Shetty` | 0.025998410 |
| Margin `Nick_Vujicic` | 0.181231066 |
| Margin `other_speaker_present` | 0.115068083 |
| Margin `music_present` | 0.005947636 |
| Margin `audience_reaction_present` | 0.122539506 |
| Margin `silence_present` | 0.009552024 |

## Evaluation protocol

1. Run unit tests and staged-checkpoint equivalence.
2. Compute all validation exits and evaluate candidates at parent level.
3. Select the maximum-saving feasible Pareto point.
4. Freeze the policy JSON.
5. Run genuine staged inference on the corrected holdout without retuning.
6. Compare Always Exit 3, frozen v0.13 per-label margin, and v0.16.
7. Repeat controlled CPU timing 30 times with one Torch/BLAS thread.
