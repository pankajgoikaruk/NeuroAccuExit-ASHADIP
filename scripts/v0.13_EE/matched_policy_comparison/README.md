# v0.13 Matched Early-Exit Strategy Comparison

This experiment compares five Exit-2/Exit-3 policies under the same checkpoint,
probability thresholds, validation split, parent-level LATS-v2 quality
constraint, and corrected-holdout evaluation protocol.

## Strategies

1. `global_conf_margin`
2. `global_conf_margin_delta`
3. `label_risk`
4. `per_label_margin`
5. `logistic_gate`

`always_exit3` is evaluated with the same staged wrapper and timing scope as a
matched runtime baseline.

## Leakage control

Validation parents are divided into two disjoint groups:

- **derivation/train subset**: derives label-risk weights, derives direct
  per-label margin profiles, and trains the logistic-regression gate;
- **policy-selection subset**: tunes and selects every strategy under the same
  Macro-F1-drop and minimum-Exit-2 constraints.

The corrected holdout is used only after every policy has been frozen.

## Gate target

The logistic gate predicts whether Exit 3 is likely to reduce the sample's
binary-label error count relative to Exit 2. It consumes Exit-1 and Exit-2
probabilities, inter-exit changes, decision margins, confidence, entropy,
agreement, and predicted-label count. It never observes Exit-3 outputs at
runtime.

## Diagnostics

Each strategy exports, per segment:

- highest-risk label;
- one risk column per label;
- condition(s) that forced continuation;
- gate safe probability where applicable;
- Exit-3 binary-error improvement for continued samples;
- labels corrected and regressed by Exit 3.

## Run

```powershell
conda activate ASHADIP_V0
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.13_EE\matched_policy_comparison\run_matched_policy_comparison_v013_EE.ps1"
```
