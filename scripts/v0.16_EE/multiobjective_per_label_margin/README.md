# v0.16 Multi-Objective Per-Label Margin Optimisation

This experiment is implemented on branch `active_budget_anytime_exit_v0.4`.
It keeps the Early-Exit controller lightweight and optimises the interpretable
v0.13 per-label margin policy rather than adding another learned runtime gate.

## Search space

The evolutionary chromosome contains:

- one global Exit-2 mean binary-confidence threshold;
- one global Exit-1-to-Exit-2 maximum probability-delta threshold;
- one decision-margin threshold for each of the ten labels.

Exit 2 is allowed only when Exit-1/Exit-2 label sets agree, the Exit-2
prediction is non-empty, all label margins pass, confidence passes and the
inter-exit probability change is sufficiently small.

## Objectives

The constraint-aware NSGA-II-style search simultaneously:

1. maximises estimated FLOPs saved;
2. minimises robust Parent Macro-F1 degradation;
3. minimises robust Parent Micro-F1 degradation;
4. minimises robust Parent Exact-Match degradation;
5. minimises robust Parent Hamming-Loss increase.

Quality constraints use five parent-grouped validation folds and one-sided
upper confidence bounds. The corrected holdout is never used for optimisation.

## Outputs

Validation output includes all evaluated candidates, the Pareto front,
optimisation history, the selected policy and a frozen JSON policy. Holdout
output compares Always Exit 3, the frozen v0.13 per-label margin baseline when
available, and the selected v0.16 Pareto policy using genuine staged execution.

A fallback point is marked `deployment_eligible=false` and remains diagnostic.

## Run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1"
```

For the final timing report, add `-TimingRepeats 30`.
