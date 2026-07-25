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

See `VERSION_HISTORY.md` for detailed traceability.

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

## Main command

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -TimingRepeats 30
```

No model-training command is required for v0.16 because the canonical checkpoint is frozen.
