# v0.14_EE — Parent-Aware Adaptive Gate

## Status

Complete negative/diagnostic experiment. No candidate met the robust validation constraint.

## Research questions

- Can parent-aware harm targets align gate training with final parent evaluation?
- Can label-specific unsafe-probability thresholds control difficult labels?
- Is Exit 1 useful enough for a later hierarchical controller?

## Implementation

```text
policies/parent_aware_adaptive_gate.py
tests/test_parent_aware_adaptive_gate.py
scripts/v0.14_EE/parent_aware_gate/
```

## Method

For each segment, replace one all-Exit3 segment probability with its shallower probability, retain Exit 3 for the remaining parent segments, and mark a label unsafe if a correct parent prediction becomes wrong.

One logistic model is trained per label using five parent-grouped OOF folds. Separate unsafe-probability thresholds are derived for each label.

## Validation settings

| Setting | Value |
|---|---:|
| OOF folds | 5 |
| Maximum Macro-F1 drop | 0.01 |
| One-sided confidence check | Required |
| Segment thresholds | fixed 0.5 |
| Timing threads | 1 |
| Final timing repeats | 30 |

## Validation result

| Strategy | Source rate | FLOPs saved | Macro drop | Drop UCB | Robust pass |
|---|---:|---:|---:|---:|---|
| Exit 2→3 | 37.92% | 24.37% | 0.011629 | 0.012846 | No |
| Exit 1→3 | 11.58% | 11.16% | 0.018751 | 0.038275 | No |

Both policies were saved as `fallback_best_robust_quality`, not valid deployment policies.

## Holdout result

| Strategy | Source rate | FLOPs saved | Macro-F1 | Micro-F1 | Exact | Hamming ↓ | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exit 2→3 | 20.30% | 13.05% | 0.840798 | 0.933966 | 0.835063 | 0.019262 | 0.9858× |
| Exit 1→3 | 0.72% | 0.69% | 0.861442 | 0.952756 | 0.876586 | 0.013841 | 0.9582× |

## Interpretation

The Exit-2 gate was too aggressive and damaged all main metrics. Exit 1 was safe for very few samples and produced no latency advantage. Sparse unsafe examples and simultaneous stopping of multiple parent segments limited the counterfactual target.

## Command

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.14_EE\parent_aware_gate\run_parent_aware_gate_v014_EE.ps1" `
  -TimingRepeats 30
```

## Non-claim

Do not describe v0.14 as a successful parent-aware controller. It is a completed negative result showing that individual-segment parent counterfactuals did not transfer safely.
