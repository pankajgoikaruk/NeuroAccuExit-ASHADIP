# v0.17_EE Results and Analysis

## Integration status

- ten staged/sequential tests passed;
- three-exit and five-exit checkpoint equivalence passed;
- validation tuning completed for both architectures;
- frozen policies evaluated on the corrected holdout;
- six ablations completed per architecture;
- 30-repeat CPU timing completed;
- fairness audit completed.

## Validation selection

| Architecture | Candidates | Pareto points | Selected validation FLOPs saved | Validation early fraction | Exit-1 fraction | Status |
|---|---:|---:|---:|---:|---:|---|
| Three exits | 5,847 | 19 | 13.98% | 18.69% | 6.11% | Validation eligible |
| Five exits | 5,856 | 86 | 29.39% | 46.10% | 10.52% | Validation eligible |

Validation eligibility is not treated as holdout approval.

## Three-exit holdout

| Method | Exit 1 | Exit 2 | Exit 3 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always final | 0.00% | 0.00% | 100.00% | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| Full sequential | 6.07% | 4.34% | 89.60% | 8.64% | 1.037× | 0.840128 | 0.937549 | 0.945653 | 0.840830 | 0.018224 |
| No Exit 1 | 0.00% | 5.28% | 94.72% | 3.39% | 1.022× | 0.854086 | 0.946871 | 0.954150 | 0.866205 | 0.015571 |

The full policy failed all four holdout limits: Macro-F1 drop 0.022254, Micro-F1 drop 0.015582, Exact-Match drop 0.035755, and Hamming increase 0.004498.

## Five-exit holdout

| Method | Exit 1 | Exit 2 | Exit 3 | Exit 4 | Exit 5 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always final | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 1.000× | 0.810761 | 0.869498 | 0.887906 | 0.673587 | 0.038985 |
| Full sequential | 6.83% | 1.22% | 18.59% | 26.30% | 47.06% | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.886945 | 0.688581 | 0.039100 |
| No Exit 1 | 0.00% | — | — | — | — | 26.80% | 1.096× | 0.809541 | 0.870906 | — | 0.687428 | approximately preserved |

The full five-exit policy met every limit: Macro-F1 drop 0.009406, Micro-F1 drop 0.000639, Exact Match improved by 0.014994, and Hamming increased by only 0.000115.

## Exit 1 analysis

Exit 1 is useful but risky. In three exits it raises FLOP saving from 3.39% to 8.64% but causes most quality loss. In five exits it raises saving from 26.80% to 30.71%; quality remains within limits, although No Exit 1 preserves Macro-F1 and Micro-F1 better.

## Ablation interpretation

- **Label margins:** removing them causes severe quality collapse, confirming that multi-label Early Exit requires label-specific safety distances.
- **Stability:** removing label-set stability gives modest extra saving but worse quality.
- **Risk:** `No Risk` is identical in three exits and nearly identical in five exits; the current risk term is effectively non-binding.
- **Confidence only:** saves the most compute but collapses Macro-F1, Exact Match, and Hamming.

## Per-label behaviour

Five-exit improvements include `silence_present` (~+0.072 F1), `music_present` (~+0.016), `Eckhart_Tolle` (~+0.006), and `Jay_Shetty` (~+0.005). Main degradations are `Nick_Vujicic` (~−0.095), `audience_reaction_present` (~−0.085), and `Eric_Thomas` (~−0.011). The three-exit model also exposes `audience_reaction_present`, `Eric_Thomas`, and `other_speaker_present` as recurring risks.

## Exact-Match improvement

Five-exit Exact Match improved from 0.673587 to 0.688581. Intermediate exits corrected the complete label set for some parents even when Macro-F1 declined slightly. This is a confirmed metric effect, not proof that intermediate exits are generally more accurate.

## Fairness audit

The architecture audit is invalid because training manifests and training rows differ: 25,519 rows for the three-exit model and 30,950 for the five-exit model. The valid claim is a within-model comparison against each architecture's own final exit. The invalid claim is that five-exit architectures are generally superior.

## Final verdict

### Successful

The five-exit full sequential policy is a major within-model success.

### Unsuccessful

The three-exit full sequential policy is not quality-safe, and the current risk term adds no measurable protection.

### Inconclusive

Architectural superiority remains unresolved until a canonical five-exit model is trained under the same data and protocol as the three-exit model.
