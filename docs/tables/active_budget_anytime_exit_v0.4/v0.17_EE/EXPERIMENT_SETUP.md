# v0.17_EE Experiment Setup

## Objective

Replace the previous two-way `Exit 2 → Exit 3` controller with a fully sequential anytime policy that evaluates every non-final exit.

## Architectures

### Three exits

```text
Exit 1 → Exit 2 → Exit 3
```

Checkpoint: `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845`.

Tap blocks: `(1,3)`.

### Five exits

```text
Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5
```

Checkpoint: `main_v06_expanded_5exit_20260603_210324`.

Tap blocks: `(1,2,3,4)`.

## Genuine staged execution

The staged wrapper executes one incremental backbone section, produces the current exit probabilities, applies the frozen policy, removes stopped samples, and sends only unresolved samples deeper. This is genuine compute skipping.

Equivalence tests passed at every exit with maximum absolute logit and probability differences of `0.0`.

## Multi-label stopping evidence

For each exit and label:

```text
binary confidence = max(p, 1 − p)
margin            = |p − threshold|
delta             = |p_current − p_previous|
```

The primary policy stops only when:

1. mean binary confidence passes;
2. all ten label margins pass;
3. inter-exit probability change is below its limit;
4. label sets are stable from Exit 2 onward;
5. maximum validation-derived label risk is below its limit;
6. at least one label is predicted.

## Optimisation variables

Each non-final exit has:

- one confidence threshold;
- one maximum probability-delta threshold;
- one maximum label-risk budget;
- ten label-specific margin thresholds.

Therefore, three-exit and five-exit policies have different chromosome lengths but use the same parameter semantics.

## Optimisation objectives

The constraint-aware NSGA-II-style optimiser:

- maximises estimated FLOPs saved;
- minimises robust Parent Macro-F1 drop;
- minimises robust Parent Micro-F1 drop;
- minimises robust Exact-Match drop;
- minimises robust Hamming increase.

The final policy is selected as a safety-buffered Pareto knee using ratio `0.75` rather than maximum validation compute saving.

## Experimental settings

| Setting | Value |
|---|---:|
| Population | 96 |
| Generations | 60 |
| Seed | 42 |
| Threshold mode | fixed 0.5 |
| Validation rows / parents | 1,883 / 304 |
| Holdout rows / parents | 4,335 / 867 |
| Minimum total early fraction | 0.02 |
| Minimum Exit-1 fraction | 0.005 |
| Macro-F1 drop limit | 0.010 |
| Micro-F1 drop limit | 0.005 |
| Exact-Match drop limit | 0.010 |
| Hamming increase limit | 0.002 |
| Batch size | 128 |
| Device | CPU |
| Threads | 1 |
| Timing repetitions | 30 |

## Research questions

| ID | Research question |
|---|---|
| RQ1 | Can every available exit participate in genuine staged inference? |
| RQ2 | Can easy, moderate, and difficult samples be routed to different depths? |
| RQ3 | Does five-exit sequential routing provide a stronger within-model trade-off? |
| RQ4 | How much compute and latency benefit comes from Exit 1? |
| RQ5 | Which policy components protect multi-label quality? |
| RQ6 | Which labels remain vulnerable to early termination? |
| RQ7 | Is the 3-exit/5-exit architecture comparison fair? |

## Ablations

| Method | Modification |
|---|---|
| Always final | No Early Exit |
| Full sequential | All components active |
| No Exit 1 | First possible stop is Exit 2 |
| No stability | Previous-exit label-set agreement removed |
| No risk | Risk budget removed |
| No label margins | Label-specific margin protection removed |
| Confidence only | Only stage confidence thresholds retained |

## Fair-comparison protocol

A direct architecture comparison requires identical labels, validation/holdout manifests, feature cache, preprocessing, LATS-v2 rules, optimiser budget, constraints, seed, and timing procedure. The current fairness audit fails because the training manifests and training-row counts differ.
