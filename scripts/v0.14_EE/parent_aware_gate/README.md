# v0.14 Parent-Aware Adaptive Gate

This experiment replaces the v0.13 scalar sample-level safe-probability gate with a parent-aware, label-sensitive utility controller.

## Main changes

- Reports Parent Macro-F1 **and Parent Micro-F1** in validation and holdout tables.
- Builds one unsafe-probability model per label.
- Uses the frozen LATS-v2 aggregation rule and threshold for each label.
- Constructs a counterfactual target by replacing one all-Exit3 segment with its shallower-exit probability and checking whether a correct parent label becomes wrong.
- Generates parent-grouped out-of-fold probabilities across the full validation set.
- Derives a separate unsafe-probability threshold for every label.
- Requires both overall quality and a one-sided cross-fold Macro-F1-drop upper confidence bound to satisfy the quality constraint.
- Evaluates Exit 2 -> Exit 3 as the primary method.
- Evaluates direct Exit 1 -> Exit 3 as a separate ablation.
- Uses complete-parent batches for parent-aware runtime decisions.
- Benchmarks methods repeatedly in randomized order and reports median and IQR latency.

## Why Exit 1 is an ablation

The first v0.14 experiment does not combine Exit-1 and Exit-2 gates hierarchically. A hierarchical controller changes the Exit-2 parent context because some segments may already have stopped at Exit 1. It therefore requires a separate sequential cross-fitting protocol. The direct Exit-1-or-Exit-3 ablation establishes whether Exit 1 contains enough reliable parent-level evidence before adding that complexity.

## Run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.14_EE\parent_aware_gate\run_parent_aware_gate_v014_EE.ps1"
```

For final runtime reporting:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.14_EE\parent_aware_gate\run_parent_aware_gate_v014_EE.ps1" `
  -TimingRepeats 30
```
