# Active Budget and Anytime Exit v0.4

## Scope

Branch:

```text
active_budget_anytime_exit_v0.4
```

This branch extends the genuine staged-inference programme from v0.11 through v0.17. It contains two completed v0.4 milestones:

```text
v0.16_EE — multi-objective optimisation of an Exit-2/Exit-3 per-label margin rule
v0.17_EE — fully sequential active-budget anytime exit across 3 and 5 exits
```

The CNN backbone and all exit heads remain frozen. Versions tune rules, fit small controllers, optimise stopping parameters, or evaluate sequential staged execution.

## Research progression

```text
v0.11 staged global rule
→ v0.12 validation-derived label risk
→ v0.13 matched rule and gate comparison
→ v0.14 parent-aware segment gate
→ v0.15 whole-parent risk control
→ v0.16 multi-objective Exit-2 margin optimisation
→ v0.17 fully sequential 3-exit and 5-exit anytime policy
```

## v0.17 research questions

1. Can the controller make a decision at every non-final exit?
2. Can very easy samples exit at Exit 1 while harder samples continue?
3. Can the same optimisation formulation support both 3-exit and 5-exit models?
4. Which multi-label safeguards are necessary?
5. Does a safety-buffered Pareto knee transfer better than v0.16's maximum-compute selection?
6. Does a deeper multi-exit checkpoint provide a better within-model quality–compute trade-off?

## v0.17 headline

| Item | 3-exit full sequential | 5-exit full sequential |
|---|---:|---:|
| Total early-exit fraction | 10.40% | **52.94%** |
| Estimated FLOPs saved | 8.64% | **30.71%** |
| 30-repeat CPU speedup | 1.037× | **1.114×** |
| Parent Macro-F1 | 0.840128 | 0.801356 |
| Parent Micro-F1 | 0.937549 | 0.868859 |
| Parent Exact Match | 0.840830 | 0.688581 |
| Parent Hamming Loss | 0.018224 | 0.039100 |
| Own-baseline quality limits | **Failed** | **Passed** |

## Confirmed findings

- Both 3-exit and 5-exit checkpoints passed exact staged/full equivalence.
- Every non-final exit participates in the primary v0.17 policy.
- The 3-exit route produced real compute saving and speedup but failed all holdout-quality limits.
- The tested 5-exit route saved 30.71% estimated FLOPs, achieved 1.114× speedup, and met all within-checkpoint quality limits.
- Exit 1 contributed additional saving but was the riskiest stage.
- Label-specific margins and previous-exit stability were strongly supported by ablations.
- Confidence-only early exit was unsafe.
- The current risk component was non-binding.
- Difficult labels remain `audience_reaction_present`, `Nick_Vujicic`, `Eric_Thomas`, and `other_speaker_present`.

## Fairness limitation

The architecture-comparison audit reports:

```text
fair_comparison_valid = false
same_validation_manifest = false
```

The 3-exit model uses the human-corrected balanced v0.8/v0.10 manifest, while the 5-exit model uses the earlier v0.6 expanded manifest.

Safe wording:

> The tested five-exit checkpoint demonstrates a strong within-model sequential quality–efficiency trade-off. The present experiment does not establish that a five-exit architecture is superior to the canonical three-exit architecture.

## Current decision

| Role | Selected method |
|---|---|
| Canonical full-quality reference | Always Exit 3 + frozen LATS-v2 |
| 3-exit quality-constrained adaptive baseline | v0.13 per-label margin |
| v0.16 role | Compute-forward multi-objective ablation |
| v0.17 successful result | Tested 5-exit full sequential policy |
| v0.17 unsuccessful result | 3-exit full sequential holdout transfer |
| Required next confirmation | Fair 5-exit retraining on the canonical manifest |

## Documentation entry points

```text
docs/active_budget_anytime_exit_v0.4/README.md
docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md
docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md

docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/
docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/
```

## Main command

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -TimingRepeats 30
```

No new model training was performed in v0.17. The package `PS_COMMANDS.md` records execution, tuning, evaluation, frozen-policy reuse, and reporting commands.
