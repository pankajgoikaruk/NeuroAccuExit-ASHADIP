# v0.15_EE — Whole-Parent Selective Risk Control

## Status

Complete methodological ablation. The whole-parent formulation worked correctly but neither controller became deployment-eligible.

## Research questions

- Can one decision for the complete parent remove v0.14 joint-substitution errors?
- Can quality and harmful-stop risk be controlled simultaneously?
- Can a nonparametric or shared logistic controller produce useful coverage?

## Implementation

```text
policies/whole_parent_selective_exit.py
tests/test_whole_parent_selective_exit.py
scripts/v0.15_EE/whole_parent_risk_control/
```

## Method

Every segment reaches Exit 2. Exit-1 and Exit-2 probabilities are aggregated using frozen LATS-v2. The complete parent stops at Exit 2 or all parent segments continue through Blocks 4–5.

Controllers:

- empirical nonparametric risk calibration;
- shared class-balanced logistic model across all parent-label pairs.

## Validation constraints

| Constraint | Limit |
|---|---:|
| Parent Macro-F1 drop | 0.005 |
| Parent Micro-F1 drop | 0.005 |
| Parent Exact Match drop | 0.01 |
| Overall harmful-stop rate | 0.01 |
| Minimum parent stop rate | 0.02 |
| OOF folds | 5 |

## Validation result

| Controller | Parent stop | FLOPs saved | Macro drop | Micro drop | Harm | Harm UCB | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| Nonparametric | 1.97% | 0.99% | 0 | 0 | 0 | 0.008823 | No |
| Shared logistic | 0.00% | 0.00% | 0 | 0 | 0 | 0.008823 | No |

The nonparametric controller missed the 2% coverage requirement by one parent. The threshold was not relaxed post hoc.

## Holdout result

| Controller | Parent stop | FLOPs saved | Macro-F1 | Micro-F1 | Exact | Hamming ↓ | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nonparametric | 0.69% | 0.44% | 0.863129 | 0.952681 | 0.875433 | 0.013841 | 0.9338× |
| Shared logistic | 0.00% | 0.00% | 0.862382 | 0.953131 | 0.876586 | 0.013725 | 0.9235× |

## Interpretation

Whole-parent decisions solved the conceptual mismatch in v0.14, but reducing the validation unit to 304 parents created a low-data risk-control problem. The safe controllers became too conservative, and policy overhead made them slower than full depth.

The nonparametric Macro-F1 increase was caused by a rare-label correction offset by errors in `other_speaker_present`; it should not be presented as overall superiority.

## Command

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.15_EE\whole_parent_risk_control\run_whole_parent_risk_control_v015_EE.ps1" `
  -TimingRepeats 30
```

## Final status

- Full integration: complete.
- Deployment eligibility: failed for both controllers.
- Practical compute saving: insufficient.
- Measured acceleration: not achieved.
