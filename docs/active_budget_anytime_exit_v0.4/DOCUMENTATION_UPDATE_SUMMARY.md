# Active Budget and Anytime Exit v0.4 — Documentation Update Summary

## Scope

This record summarises the v0.17 documentation freeze for:

```text
active_budget_anytime_exit_v0.4
```

The previous v0.16 package remains unchanged and authoritative. The new documentation adds full traceability for `v0.17_EE`, which implements fully sequential active-budget inference for both 3-exit and 5-exit checkpoints.

## Sources checked

The documentation was audited against:

```text
human_talk_workspace/active_budget_anytime_exit_v0.4/v0.17_EE/
sequential_anytime_exit/
├── 3exit/
│   ├── validation_tuning/
│   └── corrected_holdout_evaluation/
├── 5exit/
│   ├── validation_tuning/
│   └── corrected_holdout_evaluation/
└── architecture_comparison/
```

The uploaded v0.17 archive contained:

- both staged-equivalence reports;
- both frozen sequential policies;
- 5,847 and 5,856 evaluated candidate records;
- 19 and 86 Pareto candidates;
- optimisation histories;
- 3-exit and 5-exit holdout comparisons;
- six ablations per architecture;
- per-method segment and parent predictions;
- per-label LATS-v2 reports;
- 30-repeat runtime summaries;
- exit-distribution tables;
- the cross-architecture fairness audit.

## Confirmed v0.17 results

### Three exits

| Item | Confirmed value |
|---|---:|
| Exit-1 / Exit-2 / Exit-3 fractions | 6.07% / 4.34% / 89.60% |
| Estimated FLOPs saved | 8.6350% |
| 30-repeat speedup | 1.0373× |
| Parent Macro-F1 | 0.840128 |
| Parent Micro-F1 | 0.937549 |
| Parent Samples-F1 | 0.945653 |
| Parent Exact Match | 0.840830 |
| Parent Hamming Loss | 0.018224 |
| Holdout constraints | Failed |

### Five exits

| Item | Confirmed value |
|---|---:|
| Exit fractions | 6.83% / 1.22% / 18.59% / 26.30% / 47.06% |
| Estimated FLOPs saved | 30.7130% |
| 30-repeat speedup | 1.1138× |
| Parent Macro-F1 | 0.801356 |
| Parent Micro-F1 | 0.868859 |
| Parent Samples-F1 | 0.886945 |
| Parent Exact Match | 0.688581 |
| Parent Hamming Loss | 0.039100 |
| Holdout constraints | Passed |

## Confirmed findings versus interpretation

### Confirmed

- Both checkpoints passed staged/full equivalence with zero probability difference.
- Every non-final exit participates in the primary policy.
- The three-exit route creates real speedup but fails all holdout-quality thresholds.
- The tested five-exit route meets all within-checkpoint holdout thresholds.
- Exit 1 increases saving but causes the strongest quality pressure.
- Label margins and stability materially protect quality.
- Confidence-only stopping is unsafe.
- The selected risk thresholds are non-binding.
- The fairness audit fails `same_validation_manifest`.

### Interpretation

- More sequential exit opportunities may allow finer compute allocation.
- The five-exit policy's improved Exact Match suggests some intermediate exits correct complete label sets that the final exit misses.
- The three-exit validation-to-holdout failure indicates that safety-buffered selection alone does not eliminate distribution shift.
- Difficult labels require stage-specific safeguards.

These interpretations are hypotheses supported by observed patterns; they are not controlled causal conclusions.

## Documentation files updated

| File or directory | What was added |
|---|---|
| `README.md` | v0.17 branch identity, theory, settings, 3-/5-exit results, ablations, fairness, commands, limitations, and current decision |
| `DOC_STRUCTURE.md` | v0.17 code traceability, package inventory, storage policy, documentation rules, and status |
| `docs/active_budget_anytime_exit_v0.4/README.md` | Branch-level v0.17 overview and result decision |
| `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md` | New v0.17 implementation, research questions, settings, results, and findings |
| `docs/active_budget_anytime_exit_v0.4/DOCUMENTATION_UPDATE_SUMMARY.md` | This provenance and file-by-file audit |
| `docs/tables/active_budget_anytime_exit_v0.4/README.md` | Added the v0.17 compact package |
| `docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/` | New complete human- and machine-readable package |

## Non-claims

- The five-exit result is not a fair architecture winner over the three-exit model.
- The 5-exit success is relative to its own Always Exit 5 baseline.
- The 3-exit validation eligibility does not imply holdout safety.
- Estimated FLOPs are not measured latency.
- CPU timing is hardware- and protocol-specific.
- The corrected holdout is not an independent external test.
- The risk mechanism should not be claimed as effective in v0.17.
- v0.17 is sample-wise, not label-wise asynchronous.
- Evidence accumulation and distilled knowledge are not part of the primary v0.17 method.
- No v0.17 backbone training was performed.
