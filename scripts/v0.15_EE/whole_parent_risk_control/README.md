# v0.15 Whole-Parent Selective Risk Control

This experiment replaces the v0.14 segment-wise counterfactual gate with a
single decision for the complete parent clip.

## Runtime decision

1. Every segment in the parent executes through Exit 2.
2. Exit-1 and Exit-2 probabilities are aggregated with the frozen LATS-v2 rules.
3. The controller estimates per-label risk that the all-Exit-2 parent decision
   will be worse than the all-Exit-3 parent decision.
4. The complete parent either stops at Exit 2 or all of its segments continue
   through Blocks 4-5 to Exit 3.

This removes the v0.14 mismatch where several individually safe segment
substitutions could jointly alter the parent prediction.

## Compared controllers

- `nonparametric_parent_risk`: transparent empirical risk calibration using
  parent LATS margin, Exit-1/Exit-2 stability and segment dispersion.
- `shared_logistic_parent_gate`: one shared class-balanced logistic model across
  all parent-label pairs, with label identity supplied as a feature.

Both controllers use label-specific unsafe-probability thresholds and an
optional total expected-harm budget.

## Validation protocol

- Five-fold shuffled parent-level out-of-fold prediction.
- No validation parent is scored by a model fitted on that same parent.
- Candidate policies must satisfy Parent Macro-F1, Parent Micro-F1, Exact Match,
  overall harmful-stop, and minimum-coverage constraints.
- One-sided confidence bounds are applied to fold quality drops and harmful-stop
  risk.
- Final controller models are refitted using validation only after policy
  selection.
- The corrected holdout is evaluation-only.

## Run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.15_EE\whole_parent_risk_control\run_whole_parent_risk_control_v015_EE.ps1"
```

For publication timing:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.15_EE\whole_parent_risk_control\run_whole_parent_risk_control_v015_EE.ps1" `
  -TimingRepeats 30
```
