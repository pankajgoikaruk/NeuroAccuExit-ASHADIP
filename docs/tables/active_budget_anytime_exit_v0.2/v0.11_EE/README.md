# NeuroAccuExit v0.11_EE

## Status

**Complete: first standard genuine Dynamic Early-Exit milestone**

The milestone includes:

- staged inference without model retraining;
- strict numerical equivalence;
- Always Exit 1/2/3 quality audit;
- validation-only policy selection;
- genuine Exit-2/Exit-3 runtime stopping;
- compact paper-ready and machine-readable records.

## Headline result

A validation-frozen policy stopped 508 of 4,335 corrected-holdout segments at Exit 2. These samples skipped Blocks 4–5.

| Item | Value |
|---|---:|
| Exit-2 fraction | 11.72% |
| Average exit depth | 2.8828 |
| Estimated FLOPs saved | 7.53% |
| Parent Macro-F1 | 0.842248 |
| Parent Micro-F1 | 0.935484 |
| Parent Samples-F1 | 0.943577 |
| Parent Exact Match | 0.838524 |
| Parent Hamming Loss | 0.018916 |

Canonical full depth:

```text
Macro-F1 = 0.862382
Micro-F1 = 0.953131
Average depth = 3.0
```

## Package contents

| File | Purpose |
|---|---|
| `EXPERIMENT_SETUP.md` | Exact model, dataset, policy, threshold, and evaluation configuration |
| `RESULTS_AND_ANALYSIS.md` | Fixed-exit, per-label, and dynamic interpretation |
| `PAPER_READY_SUMMARY.md` | Reusable paper text and table/caption wording |
| `PS_COMMANDS.md` | Reproduction and ablation commands |
| `REPRODUCE_V011_EE.ps1` | Combined runner |
| `experiment_manifest.json` | Machine-readable record |
| `checkpoint_staged_equivalence.json` | Real-checkpoint equivalence evidence |
| `fixed_exit_segment_summary.csv` | Segment-level fixed exits |
| `fixed_exit_parent_summary.csv` | Parent-level fixed exits |
| `fixed_exit_parent_per_label.csv` | Per-label parent metrics at each exit |
| `validation_policy_selection.csv` | Frozen selected validation policy |
| `dynamic_exit_summary.csv` | Dynamic holdout result |
| `cumulative_comparison.csv` | Full-depth, fixed, and dynamic comparison |

## Key conclusion

Exit 1 is too weak for general stopping. Exit 2 is useful for a conservative subset of samples. The staged implementation proves that those samples can avoid the final backbone stage without changing the trained checkpoint.

## Research status

This package establishes the **standard sample-wise Early-Exit baseline**. It does not yet represent budget-aware or label-wise asynchronous inference.
