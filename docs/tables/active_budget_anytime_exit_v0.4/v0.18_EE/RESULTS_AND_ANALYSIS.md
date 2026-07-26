# v0.18 Results and Analysis

## Fairness and implementation checks

- Fair-training audit: **PASS**.
- All staged/full equivalence checks: **PASS** with zero logit and probability differences.
- Full training, tuning, holdout evaluation, ablations, and 30-repeat timing: complete.

## Five-exit training progression

| Exit | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|
| 1 | 0.2106 | 0.3472 | 0.1469 | 0.1296 |
| 2 | 0.3795 | 0.5075 | 0.2902 | 0.1109 |
| 3 | 0.6112 | 0.6542 | 0.4182 | 0.0898 |
| 4 | 0.7419 | 0.7356 | 0.4946 | 0.0727 |
| 5 | 0.8320 | 0.8216 | 0.6206 | 0.0502 |

Quality improves with depth; early heads are not interchangeable with the final exit.

## Corrected-holdout results

| Architecture | Policy | Early fraction | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3-exit | Always final | 0% | 0% | 1.000× | 0.862382 | 0.953131 | 0.876586 | 0.013725 |
| 3-exit | `full_strict` | 5.70% | 3.82% | 1.018× | 0.85285 | 0.94530 | 0.86159 | 0.01603 |
| 5-exit | Always final | 0% | 0% | 1.000× | 0.82097 | 0.90734 | 0.77970 | 0.02780 |
| 5-exit | `full_strict` | 17.42% | 12.70% | 1.057× | 0.79813 | 0.89832 | 0.75779 | 0.03010 |
| 5-exit | `no_exit1` | 14.28% | 9.18% | 1.037× | 0.81015 | 0.90375 | 0.77163 | 0.02872 |

## Exit distributions

### Three exits

```text
Exit 1: 0.48%
Exit 2: 5.21%
Exit 3: 94.30%
```

### Five exits

```text
Exit 1: 4.71%
Exit 2: 0.00%
Exit 3: 12.71%
Exit 4: 0.00%
Exit 5: 82.58%
```

The selected five-exit policy effectively behaves as `Exit 1 → Exit 3 → Exit 5`; Exits 2 and 4 receive no selected holdout samples.

## Constraint compliance

| Policy | Macro | Micro | Exact | Hamming | Overall |
|---|---|---|---|---|---|
| 3-exit `full_strict` | Pass | Fail | Fail | Fail | Fail |
| 5-exit `full_strict` | Fail | Fail | Fail | Fail | Fail |
| 5-exit `no_exit1` | Fail by 0.000819 | Pass | Pass | Pass | Nearly feasible |

## Validation-to-holdout transfer

Both selected policies appeared safe on validation but lost more quality and stopped fewer samples on holdout. This transfer gap, rather than insufficient optimiser capacity, is the principal remaining problem.

## Architecture interpretation

The five-exit model has more intermediate stopping opportunities and produces larger computation savings. This is a fair result for these checkpoints and data. It does not prove universal superiority of five-exit architectures.

## Per-label interpretation

The largest five-exit losses affect `audience_reaction_present`, `Jay_Shetty`, and `Eric_Thomas`. The current validation risk profile underestimates some transient-label failure modes.
