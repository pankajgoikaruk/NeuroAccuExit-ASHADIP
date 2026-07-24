# Active Budget and Anytime Exit v0.4

## Scope

Branch:

```text
active_budget_anytime_exit_v0.4
```

This branch extends the genuine staged-inference programme from v0.11 through v0.16. Its current contribution is a lightweight multi-objective optimisation of the v0.13 per-label margin policy.

The CNN backbone and all exit heads remain frozen. Different versions either tune rules, train small controllers, or optimise stopping thresholds.

## Research progression

```text
v0.11 staged global rule
→ v0.12 validation-derived label risk
→ v0.13 matched rule and gate comparison
→ v0.14 parent-aware segment gate
→ v0.15 whole-parent risk control
→ v0.16 multi-objective per-label margin optimisation
```

See `VERSION_HISTORY.md` for detailed traceability and `DOCUMENTATION_UPDATE_SUMMARY.md` for the documentation audit, result provenance, confirmed/interpretive separation, and file-by-file update record.

## v0.16 headline

Validation-only optimisation evaluated 4,078 unique policies and retained 20 Pareto candidates. The selected validation point predicted 12.63% FLOP saving with very small validation quality changes.

On the corrected holdout:

| Item | v0.16 |
|---|---:|
| Exit-2 fraction | 7.87% |
| Average exit depth | 2.921338 |
| Estimated FLOPs saved | 5.055% |
| 30-repeat CPU speedup | 1.015× |
| Parent Macro-F1 | 0.849203 |
| Parent Micro-F1 | 0.942474 |
| Parent Exact Match | 0.854671 |
| Parent Hamming Loss | 0.016840 |

The result demonstrates meaningful genuine skipping and a small measured speedup. It does not satisfy the predefined holdout quality limits.

## Confirmed result versus interpretation

### Confirmed

- staged/full equivalence passed at all exits;
- the policy was selected using validation only and frozen before holdout evaluation;
- v0.16 increased compute saving and measured speed relative to v0.13;
- every predefined holdout quality constraint failed;
- v0.13 remains the quality-constrained adaptive baseline.

### Interpretation and future work

The maximum-saving feasible validation point appears too aggressive for holdout transfer. Safety-buffered Pareto-knee selection and a newly reserved calibration/evaluation protocol are possible next steps, but they are not completed v0.16 results.

## Current decision

| Role | Selected method |
|---|---|
| Full-quality reference | Always Exit 3 + frozen LATS-v2 |
| Adaptive quality-constrained baseline | v0.13 per-label margin |
| Compute-forward optimisation ablation | v0.16 multi-objective margin |

## Documentation

The complete v0.16 package is stored at:

```text
docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/
```

Documentation entry points:

```text
docs/active_budget_anytime_exit_v0.4/README.md
docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md
docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md
```

## Main command

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -TimingRepeats 30
```

No model-training command is required for v0.16 because the canonical checkpoint is frozen. The package `PS_COMMANDS.md` separately records the full runner, direct tuning/evaluation entry points, frozen-policy reuse, and reporting commands.
