# v0.13_EE — Matched Early-Exit Strategy Comparison

## Status

Complete. Five policies were selected under identical validation constraints and evaluated using the same checkpoint and corrected holdout.

## Research questions

- Do label-aware rules improve over global confidence/margin rules?
- Does a learned logistic gate find a stronger quality–compute point?
- Which policy should become the current adaptive baseline?

## Implementation

```text
policies/early_exit_strategy_comparison.py
tests/test_early_exit_strategy_comparison.py
scripts/v0.13_EE/matched_policy_comparison/
```

## Validation protocol

| Setting | Value |
|---|---:|
| Derivation / selection | 70% / 30% of validation parents |
| Maximum Macro-F1 drop | 0.01 |
| Minimum Exit-2 fraction | 0.02 |
| Segment thresholds | fixed 0.5 |
| Parent metric | frozen LATS-v2 Macro-F1 |

The derivation subset trains or derives the gate, label risks and per-label margins. The selection subset chooses every method under the same constraint.

## Compared policies and selected settings

| Policy | Selected configuration |
|---|---|
| Global confidence + margin | confidence 0.95; margin 0.00; agreement required |
| Global + delta | confidence 0.55; margin 0.00; max delta 0.20 |
| Label risk | global+delta settings; risk threshold 0.75 |
| Per-label margin | confidence 0.55; 75th-percentile corrected-example margins |
| Logistic gate | safe-probability threshold 0.75 |

## Holdout results

| Policy | Exit-2 rate | FLOPs saved | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Global confidence + margin | 1.18% | 0.76% | 0.861433 | 0.952719 | 0.875433 | 0.013841 |
| Global + delta | 2.42% | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| Label risk | 2.42% | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| **Per-label margin** | **2.24%** | **1.44%** | **0.858748** | **0.951556** | **0.874279** | **0.014187** |
| Logistic gate | 17.58% | 11.30% | 0.833034 | 0.943529 | 0.855825 | 0.016609 |

## Main findings

- Per-label margins provided the best reliable balance and remain the current adaptive recommendation.
- Label risk was non-binding and duplicated the global-delta decisions.
- The learned gate found many more early exits, but its validation-selected threshold transferred poorly.
- A learned gate should not be assumed superior unless it improves the quality–compute Pareto frontier.

## Command

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.13_EE\matched_policy_comparison\run_matched_policy_comparison_v013_EE.ps1"
```

## Timing caution

The v0.13 speedup values are preliminary. Later repeated timing showed that low stopping rates may not offset controller overhead.
