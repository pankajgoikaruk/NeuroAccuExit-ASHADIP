# Active Budget and Anytime Exit v0.4

## Scope

Branch:

```text
active_budget_anytime_exit_v0.4
```

This branch extends genuine staged inference from v0.11 through v0.17. v0.16 optimised a lightweight Exit-2/Exit-3 policy. v0.17 moves to a fully sequential controller that evaluates every available exit in three-exit and five-exit checkpoints.

The CNN checkpoints are frozen during policy tuning and evaluation. Different versions tune rules, train small controllers, optimise stopping thresholds, or compare staged routes.

## Research progression

```text
v0.11 staged global rule
→ v0.12 validation-derived label risk
→ v0.13 matched rules and learned gate
→ v0.14 parent-aware segment gate
→ v0.15 whole-parent risk control
→ v0.16 multi-objective Exit-2/Exit-3 margins
→ v0.17 fully sequential 3-exit and 5-exit anytime inference
```

See `VERSION_HISTORY.md` for version-by-version traceability and `DOCUMENTATION_UPDATE_SUMMARY.md` for the documentation audit.

## v0.17 research questions

1. Can very easy samples exit at Exit 1 while harder samples progress sequentially?
2. Does full sequential routing improve the quality–computation trade-off over an Exit-2-only policy?
3. Does a five-exit route provide more useful operating points than a three-exit route?
4. Which safety components are essential: label margins, stability, risk, or confidence?
5. Which labels remain unsafe under early termination?
6. Is the available three-exit/five-exit comparison experimentally fair?

## v0.17 headline

### Three exits

The full sequential three-exit policy used Exit 1 for 6.07% of segments and Exit 2 for 4.34%, saving 8.64% estimated FLOPs and achieving 1.037× measured CPU speedup. However, all four predefined holdout quality limits failed.

### Five exits

The full sequential five-exit policy routed 6.83% / 1.22% / 18.59% / 26.30% / 47.06% of segments to Exits 1–5. It saved 30.71% estimated FLOPs, achieved 1.114× CPU speedup, and met every predefined within-model holdout quality limit.

| Metric | Always Exit 5 | Full sequential | Change |
|---|---:|---:|---:|
| Parent Macro-F1 | 0.810761 | 0.801356 | −0.009406 |
| Parent Micro-F1 | 0.869498 | 0.868859 | −0.000639 |
| Parent Exact Match | 0.673587 | 0.688581 | **+0.014994** |
| Parent Hamming Loss | 0.038985 | 0.039100 | +0.000115 |

## Confirmed findings

- The five-exit policy is the major success of v0.17.
- The three-exit policy is computationally successful but not quality-safe.
- Exit 1 adds meaningful compute saving but is currently the riskiest stage.
- Label-specific margins and inter-exit stability are essential.
- The current risk term is weakly active and needs redesign.
- Per-label failures concentrate on `Nick_Vujicic`, `audience_reaction_present`, `Eric_Thomas`, and `other_speaker_present`.
- The architecture comparison is not yet fair because training manifests differ.

## Current decision

| Role | Selected method |
|---|---|
| Three-exit full-quality reference | Always Exit 3 + frozen LATS-v2 |
| Fair three-exit adaptive baseline | v0.13 per-label margin |
| Successful within-model anytime result | v0.17 five-exit full sequential |
| Compute-successful but quality-unsafe result | v0.17 three-exit full sequential |
| Required next experiment | Train a canonical five-exit model using the same data/protocol as the three-exit model |

## Documentation

```text
docs/active_budget_anytime_exit_v0.4/
docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/
docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/
```

## Main commands

Three-exit only:

```powershell
conda activate ASHADIP_V0
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -Run3Only
```

Combined 3-/5-exit publication timing:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -TimingRepeats 30
```

No v0.17 backbone-training command exists for the completed run because both checkpoints were reused. A fair architecture claim requires a newly trained canonical five-exit checkpoint.
