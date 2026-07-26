# Active Budget and Anytime Exit v0.4

## Scope

Branch: `active_budget_anytime_exit_v0.4`

This branch contains the completed computation-adaptive inference sequence from `v0.11_EE` through `v0.18_EE`.

```text
v0.11 staged global rule
→ v0.12 validation-derived label risk
→ v0.13 matched rules and logistic gate
→ v0.14 parent-aware counterfactual gate
→ v0.15 whole-parent risk control
→ v0.16 multi-objective Exit-2 optimisation
→ v0.17 full sequential 3-/5-exit anytime inference
→ v0.18 fair 3-/5-exit training and strict continuation-risk control
```

## v0.18 research questions

1. Can a five-exit model be trained fairly against the canonical three-exit model?
2. Does the strong v0.17 five-exit result reproduce under matched data and auxiliary-loss budget?
3. Can stricter Exit-1 protection reduce early-stage harm?
4. Can a redesigned label-risk veto become measurably active?
5. Which policy terms are necessary for multi-label early exit?
6. Can either architecture satisfy all four corrected-holdout quality constraints while saving computation?

## Fairness protocol

The audit passed the same manifest, feature root, label schema, labels, input size, epochs, batch size, optimiser settings, seed, threshold, class-balance settings, no-hint status, final-exit weight, and total auxiliary-loss budget.

| Architecture | Tap blocks | Loss weights | Auxiliary budget |
|---|---|---|---:|
| 3-exit | `1,3` | `0.3,0.3,1.0` | 0.60 |
| 5-exit | `1,2,3,4` | `0.15,0.15,0.15,0.15,1.0` | 0.60 |

## v0.18 headline

| Policy | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ | Constraint status |
|---|---:|---:|---:|---:|---:|---:|---|
| 3-exit Always Final | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.876586 | 0.013725 | Reference |
| 3-exit `full_strict` | 3.82% | 1.018× | 0.85285 | 0.94530 | 0.86159 | 0.01603 | Failed 3/4 |
| 5-exit Always Final | 0.00% | 1.000× | 0.82097 | 0.90734 | 0.77970 | 0.02780 | Reference |
| 5-exit `full_strict` | 12.70% | 1.057× | 0.79813 | 0.89832 | 0.75779 | 0.03010 | Failed 4/4 |
| 5-exit `no_exit1` | **9.18%** | **1.037×** | **0.81015** | **0.90375** | **0.77163** | **0.02872** | Passed 3/4 |

## Confirmed findings

- Fair 3-exit/5-exit training is now established.
- Five exits provide more computation-saving capacity under matched training.
- Neither full strict policy is holdout-safe.
- Exit 1 remains the riskiest stopping stage.
- The v0.18 risk veto is active and quality-protective.
- Label-specific margins and stability are essential.
- Confidence-only stopping is unsafe.
- Validation-to-holdout transfer is the principal unresolved problem.
- The five-exit No-Exit-1 route is the closest current candidate.

## Current decision

| Role | Method |
|---|---|
| Canonical full-quality reference | Always Exit 3 + frozen LATS-v2 |
| Quality-constrained adaptive baseline | v0.13 per-label margin |
| Fair architecture study | v0.18 |
| Closest-to-feasible candidate | v0.18 five-exit `no_exit1` |
| Unsuccessful result | Both selected `full_strict` policies |
| Next step | Dedicated validation-only optimisation of `Exit 3 → Exit 5` |

## Documentation entry points

```text
docs/active_budget_anytime_exit_v0.4/README.md
docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md
docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md

docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/
docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/
docs/tables/active_budget_anytime_exit_v0.4/v0.18_EE/
```

## Main command

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1" `
  -TimingRepeats 30
```
