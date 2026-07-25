# v0.17_EE — Sequential Active-Budget Anytime Exit

## Status

v0.17 is fully integrated and evaluated for:

```text
3 exits: Exit 1 → Exit 2 → Exit 3
5 exits: Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5
```

The policies were tuned on validation data, frozen, and evaluated through genuine staged inference on the corrected holdout. Thirty-repeat CPU timing and six ablations were completed.

## Headline decision

| Architecture | Outcome |
|---|---|
| Three exits | Real speedup and FLOP saving, but holdout quality constraints failed. |
| Five exits | 30.71% estimated FLOPs saved, 1.114× measured speedup, and all within-model holdout quality constraints met. |
| Direct 3-vs-5 claim | Not valid because the checkpoints used different training manifests. |

## Package contents

| File | Purpose |
|---|---|
| `EXPERIMENT_SETUP.md` | Architecture, theory, optimiser, settings, RQs, and fairness protocol |
| `RESULTS_AND_ANALYSIS.md` | Validation, holdout, timing, ablations, per-label findings, and verdict |
| `PAPER_READY_SUMMARY.md` | Reusable academic wording, tables, captions, limitations, and non-overclaim guidance |
| `PS_COMMANDS.md` | PowerShell commands for tuning, evaluation, timing, and reporting |
| `experiment_manifest.json` | Machine-readable experiment metadata |
| `headline_results.csv` | Main three-exit/five-exit results |
| `ablation_summary.csv` | Full and reduced-policy comparison |
| `cumulative_version_comparison.csv` | v0.11–v0.17 progression |
| `per_label_findings.csv` | Main per-label effects |
| `exit_distribution.svg` | Full sequential routing distribution |
| `quality_compute_comparison.svg` | Compute saving versus Macro-F1 change |

## Confirmed five-exit result

| Metric | Always Exit 5 | Full sequential | Change |
|---|---:|---:|---:|
| Parent Macro-F1 | 0.810761 | 0.801356 | −0.009406 |
| Parent Micro-F1 | 0.869498 | 0.868859 | −0.000639 |
| Parent Exact Match | 0.673587 | 0.688581 | +0.014994 |
| Parent Hamming Loss | 0.038985 | 0.039100 | +0.000115 |
| Estimated FLOPs saved | 0.00% | 30.71% | +30.71 pp |
| Measured CPU speedup | 1.000× | 1.114× | +11.4% |

## Scientific conclusion

The five-exit checkpoint demonstrates that fully sequential routing can create a strong within-model quality–computation trade-off. The result does not prove that a five-exit architecture is inherently better than the canonical three-exit architecture. A fair claim requires retraining the five-exit model on the same training manifest and protocol.
