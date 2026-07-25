# v0.17 — Sequential Active-Budget Anytime Exit

This experiment replaces the two-exit `Exit 2 -> Exit 3` controller with genuine sequential inference.

## Supported routes

- **3 exits:** `Exit 1 -> Exit 2 -> Exit 3`
- **5 exits:** `Exit 1 -> Exit 2 -> Exit 3 -> Exit 4 -> Exit 5`

Every non-final exit evaluates a stop/continue gate. Exit 1 is never omitted from the primary policy.

## Decision evidence at exit e

- mean multi-label binary confidence;
- per-label distance from the exit-specific classification threshold;
- maximum probability movement from exit e-1 to exit e;
- multi-label set stability with the previous exit (from Exit 2 onward);
- validation-derived risk weights for labels frequently corrected by the final exit.

A sample stops at the first exit satisfying all active conditions. Otherwise it continues to the next exit. Samples that stop do not execute later backbone blocks.

## Optimisation

A constraint-aware NSGA-II-style search learns a separate parameter block for every non-final exit:

1. confidence threshold;
2. maximum inter-exit probability delta;
3. maximum label-risk budget;
4. one decision-margin threshold per label.

Objectives maximise estimated FLOPs saved while minimising robust degradation in Parent Macro-F1, Parent Micro-F1, Exact Match and Hamming Loss. Selection uses a **safety-buffered Pareto knee**, not the maximum-compute boundary used in v0.16.

## Fair 3-exit versus 5-exit comparison

A direct architecture claim is valid only when both checkpoints use the identical:

- validation and holdout manifests;
- feature cache and label order;
- LATS-v2 configuration;
- threshold mode;
- optimiser population, generations and seed;
- quality constraints and timing protocol.

The comparison script writes `v017_fairness_audit.json` and marks the result invalid when any condition differs.

## Reported metrics

Predictive performance:

- Parent Macro-F1, Micro-F1 and Samples-F1;
- Parent Exact Match and Hamming Loss;
- segment-level multi-label metrics;
- per-label F1 through the frozen LATS-v2 evaluator.

Computational efficiency:

- fraction and count at every exit;
- total early-exit coverage;
- average exit depth;
- estimated FLOPs saved relative to each architecture's own final exit;
- median latency, latency IQR and measured speedup.

## Generated ablations

| Ablation | Purpose |
|---|---|
| Full sequential | Confidence + margins + stability + delta + risk at all exits |
| No Exit 1 | Quantifies the value of the first exit |
| No stability | Removes previous-exit label-set agreement |
| No risk | Removes label-risk weighting |
| No label margins | Removes label-specific safety distances |
| Confidence only | Retains only stage confidence thresholds |

Primary output tables:

- `v017_3exit_holdout_comparison.csv`
- `v017_5exit_holdout_comparison.csv`
- `v017_combined_ablation_table.csv`
- `v017_3exit_vs_5exit_headline.csv`
- `v017_exit_distribution_comparison.csv`
- `v017_fairness_audit.json`

## Older-experiment context

The older five-exit label-stability experiment stopped no samples at Exit 1 or Exit 2, saved about 17.6% depth units, and suffered substantial quality loss. v0.17 directly corrects that limitation by optimising all non-final exits and preserving explicit parent-level quality constraints. Evidence accumulation and distilled-knowledge ideas remain future policy/training ablations; they are not silently mixed into the primary v0.17 controller.
