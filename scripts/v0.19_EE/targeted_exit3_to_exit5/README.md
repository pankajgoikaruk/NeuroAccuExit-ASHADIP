# v0.19 — Final Targeted Exit 3 → Exit 5 Experiment

This is the final planned Early-Exit experiment on branch
`active_budget_anytime_exit_v0.4`.

## Motivation

The fair v0.18 five-exit `No Exit 1` ablation was the closest quality-safe
operating point. It saved about 9.18% of FLOPs and passed the Micro-F1,
Exact-Match and Hamming limits, but exceeded the Macro-F1 drop limit by only
0.000819. That ablation reused thresholds tuned for another route.

v0.19 therefore optimises the route directly:

```text
Exit 1: compute only
Exit 2: stability evidence only
Exit 3: stop or continue
Exit 4: compute only for continuing samples
Exit 5: final decision
```

The deployable decision route is `Exit 3 → Exit 5`; no sample may stop at
Exit 1, Exit 2 or Exit 4.

## Final policy design

The Exit-3 decision uses:

- mean multi-label binary confidence;
- Exit-2-to-Exit-3 label-set stability;
- maximum probability movement from Exit 2 to Exit 3;
- ten label-specific decision margins;
- grouped worst-fold continuation-risk scores;
- stricter margins for positive high-risk labels.

The grouped risk score combines:

- Exit-3 errors corrected by Exit 5;
- positive per-label F1 gain at Exit 5;
- false-negative repair;
- false-positive repair;
- label rarity.

## Conservative selection rule

The validation optimiser uses internal limits that are half the final deployment
limits by default. It targets 7% FLOP savings and freezes the candidate with the
lowest robust validation-risk utilisation among candidates reaching that target.
The corrected holdout is never used for tuning.

## Final decision rule

The experiment writes `v019_final_decision.json`.

- If the frozen policy passes all four corrected-holdout limits with non-zero
  savings, finalise v0.19.
- Otherwise stop EE development and retain the previously established safe
  adaptive baseline.

No further broad EE policy search is recommended after v0.19.

## Run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.19_EE\targeted_exit3_to_exit5\run_v019_EE.ps1" `
  -RunDir5 "N:\v018_fair5_20260725_231353" `
  -TimingRepeats 30
```

The runner can also resolve the standard v0.18 fair-five-exit run automatically
when it is present in the normal workspace location.

## Main outputs

```text
validation_tuning/
  frozen_targeted_exit3_to_exit5_policy_v019.json
  v019_all_candidates.csv
  v019_pareto_front.csv
  v019_selected_policy.csv

corrected_holdout_evaluation/
  v019_targeted_holdout_comparison.csv
  v019_targeted_holdout_comparison.json
  always_final/
  full_targeted/
  no_risk/
  no_stability/
  no_label_margins/
  confidence_only/

final_comparison/
  v019_final_decision.json
  v019_final_targeted_table.csv
  v019_final_targeted_table.tex
  v018_v019_targeted_comparison.csv
```
