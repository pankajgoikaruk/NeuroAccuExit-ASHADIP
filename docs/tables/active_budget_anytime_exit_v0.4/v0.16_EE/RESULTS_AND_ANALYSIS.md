# v0.16_EE Results and Analysis

## Execution validity

- Ten staged and optimisation tests passed.
- Real-checkpoint staged/full equivalence passed at all three exits.
- Maximum absolute logit and probability differences were `0.0`.
- The corrected holdout used the frozen validation policy; no holdout retuning occurred.
- Publication timing used 30 repetitions and one CPU thread for Torch/BLAS.

## Validation search

| Item | Value |
|---|---:|
| Unique policies evaluated | 4,078 |
| Pareto candidates | 20 |
| Selected status | `feasible_pareto_max_compute_saving` |
| Exit-2 fraction | 19.6495% |
| Average exit depth | 2.803505 |
| Estimated FLOPs saved | 12.6272% |
| Macro-F1 drop | 0.000389 |
| Micro-F1 drop | 0.000940 |
| Exact-Match drop | 0.000000 |
| Hamming increase | 0.000329 |

The search behaved as intended: feasible best compute saving improved from 8.09% at generation 0 to 12.63% by the final search.

## Corrected-holdout comparison

| Method | Exit-2 rate | Avg depth | FLOPs saved | Latency ms | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | 0.00% | 3.000000 | 0.00% | 1.522200 | 1.000× | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| v0.13 per-label margin | 2.24% | 2.977624 | 1.44% | 1.526888 | 0.997× | 0.858748 | 0.951556 | 0.957198 | 0.874279 | 0.014187 |
| v0.16 multi-objective | 7.87% | 2.921338 | 5.06% | 1.499292 | 1.015× | 0.849203 | 0.942474 | 0.950266 | 0.854671 | 0.016840 |

## Requirement audit

| Metric | Required maximum degradation | Observed degradation | Result |
|---|---:|---:|---|
| Parent Macro-F1 | 0.010 | 0.013178 | Fail |
| Parent Micro-F1 | 0.005 | 0.010657 | Fail |
| Parent Exact Match | 0.010 | 0.021915 | Fail |
| Parent Hamming Loss | +0.002 | +0.003114 | Fail |

The policy met compute objectives but not the predefined holdout quality requirements.

## Per-label holdout F1

| Label | Full depth | v0.13 | v0.16 | v0.16 change |
|---|---:|---:|---:|---:|
| `Brene_Brown` | 0.986111 | 0.986111 | 0.986111 | +0.000000 |
| `Eckhart_Tolle` | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| `Eric_Thomas` | 0.942029 | 0.948905 | 0.933333 | -0.008696 |
| `Gary_Vee` | 1.000000 | 0.992593 | 0.992593 | -0.007407 |
| `Jay_Shetty` | 0.978261 | 0.978261 | 0.972678 | -0.005583 |
| `Nick_Vujicic` | 0.980000 | 0.980000 | 0.989899 | +0.009899 |
| `other_speaker_present` | 0.958696 | 0.958606 | 0.947253 | -0.011443 |
| `music_present` | 0.957290 | 0.957290 | 0.955882 | -0.001408 |
| `audience_reaction_present` | 0.535714 | 0.500000 | 0.428571 | -0.107143 |
| `silence_present` | 0.285714 | 0.285714 | 0.285714 | +0.000000 |

Largest v0.16 effects:

- `audience_reaction_present`: 0.535714 → 0.428571;
- `other_speaker_present`: 0.958696 → 0.947253;
- `Eric_Thomas`: 0.942029 → 0.933333;
- `Nick_Vujicic`: 0.980000 → 0.989899, an improvement.

The overall loss is not uniformly distributed; transient/context and open-set labels remain particularly sensitive.

## Interpretation

### What worked

1. The optimiser correctly generated a validation Pareto frontier.
2. Genuine Exit-2 stopping increased from 2.24% in v0.13 to 7.87% in v0.16.
3. Estimated saving increased from 1.44% to 5.06%.
4. v0.16 achieved a controlled 1.015× CPU speedup, while the same-protocol v0.13 baseline was 0.997×.

### What did not work

1. Validation quality bounds did not transfer to the corrected holdout.
2. Selecting the maximum-saving feasible validation point was too aggressive.
3. The maximum probability-delta threshold (0.937) was effectively weak, while several label margins were near zero.
4. Aggregate validation feasibility did not protect difficult holdout labels sufficiently.

## Ablation conclusion

- v0.11/v0.12 demonstrated meaningful skipping but large quality degradation.
- v0.13 per-label margin established the strongest quality-constrained rule baseline.
- v0.14/v0.15 showed that learned parent-aware control can become either unsafe or too conservative.
- v0.16 showed that multi-objective optimisation can expand compute savings and produce real speedup, but maximum-compute validation selection is not robust enough.

## Research decision

Do not retune v0.16 on the corrected holdout. Preserve it as a compute-forward Pareto ablation. Continue to report v0.13 per-label margin as the adaptive baseline and Always Exit 3 as the full-quality reference.
