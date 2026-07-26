# Cross-Version and Historical Comparison Tables

This document consolidates the canonical three-exit comparison, the historical v0.17 five-exit result, and the fair v0.18 architecture study.

## Reporting rule

- The **canonical cross-version ranking** contains only methods evaluated with the canonical three-exit checkpoint and frozen historical LATS-v2 protocol.
- The v0.17 five-exit result is retained as an important **within-checkpoint historical result**, but it is not evidence of fair five-exit superiority because its training manifest differed.
- The v0.18 five-exit model is the first **training-fair** comparator. Its policies are evaluated against the fair v0.18 Always-Exit-5 reference.
- Validation eligibility and corrected-holdout compliance are reported separately.

## Canonical three-exit corrected-holdout comparison through v0.18

| Method | Stop unit | Decision route | FLOPs saved | Parent Macro-F1 | Parent Micro-F1 | Samples-F1 | Exact Match | Hamming ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | None | Exit 3 only | 0.00% | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| v0.13 per-label margin | Segment | Exit 2 → Exit 3 | 1.44% | 0.858748 | 0.951556 | 0.957198 | 0.874279 | 0.014187 |
| v0.14 Exit-1 ablation | Segment | Exit 1 → Exit 3 | 0.69% | 0.861442 | 0.952756 | 0.958697 | 0.876586 | 0.013841 |
| v0.15 nonparametric parent risk | Parent | Exit 2 → Exit 3 | 0.44% | 0.863129 | 0.952681 | 0.958505 | 0.875433 | 0.013841 |
| v0.15 shared logistic parent gate | Parent | Exit 2 → Exit 3 | 0.00% | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| v0.16 multi-objective margin | Segment | Exit 2 → Exit 3 | 5.06% | 0.849203 | 0.942474 | 0.950266 | 0.854671 | 0.016840 |
| v0.17 sequential anytime | Segment, sequential | Exit 1 → Exit 2 → Exit 3 | 8.64% | 0.840128 | 0.937549 | 0.945653 | 0.840830 | 0.018224 |
| **v0.18 strict sequential** | Segment, sequential | Exit 1 → Exit 2 → Exit 3 | **3.82%** | **0.852849** | **0.945297** | **0.952277** | **0.861592** | **0.016032** |
| v0.13 logistic gate | Segment | Exit 2 → Exit 3 | 11.30% | 0.833034 | 0.943529 | 0.949750 | 0.855825 | 0.016609 |
| v0.14 Exit-2 parent-aware | Segment | Exit 2 → Exit 3 | 13.05% | 0.840798 | 0.933966 | 0.942473 | 0.835063 | 0.019262 |

The exact v0.18 Samples-F1 values are taken from the archived holdout CSV/JSON outputs.

## Historical v0.17 five-exit within-checkpoint result

| Method | Stop unit | Decision route | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 5 | None | Exit 5 only | 0.00% | 1.000× | 0.810761 | 0.869498 | 0.887906 | 0.673587 | 0.038985 |
| v0.17 five-exit sequential | Segment, sequential | Exit 1 → 2 → 3 → 4 → 5 | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.886945 | 0.688581 | 0.039100 |

The policy stopped 52.94% before Exit 5 and passed all four limits relative to its own baseline, but the architecture comparison was not fair because the checkpoints used different training manifests.

## Fair v0.18 architecture headline

| Architecture | Policy | Decision route | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | Holdout status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3-exit | Always final | Exit 3 only | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 | Reference |
| 3-exit | `full_strict` | Exit 1 → Exit 2 → Exit 3 | 3.82% | 1.018× | 0.852849 | 0.945297 | 0.952277 | 0.861592 | 0.016032 | Failed 3/4 limits |
| 5-exit | Always final | Exit 5 only | 0.00% | 1.000× | 0.820972 | 0.907343 | 0.923623 | 0.779700 | 0.027797 | Reference |
| 5-exit | `full_strict` | Exit 1 → Exit 3 → Exit 5 in practice | 12.70% | 1.057× | 0.798133 | 0.898325 | 0.913665 | 0.757785 | 0.030104 | Failed 4/4 limits |
| 5-exit | `no_exit1` | Exit 3 → Exit 5 | **9.18%** | **1.037×** | **0.810153** | **0.903750** | **0.918894** | **0.771626** | **0.028720** | Passed 3/4; Macro exceeded by 0.000819 |

## Main research interpretation

1. v0.17 demonstrated a strong five-exit operating point, but its cross-architecture comparison was not fair.
2. v0.18 solved the training-fairness problem by matching data, optimisation settings, no-hint status, final-exit weight, and total auxiliary-loss budget.
3. The strong v0.17 quality-preserving five-exit result did not reproduce under fair retraining.
4. Five exits still provided greater computation-saving capacity than three exits.
5. Exit 1 remained the highest-risk decision stage.
6. The redesigned v0.18 risk veto was active and quality-protective.
7. The five-exit `Exit 3 → Exit 5` route is the closest current candidate, but it is not yet fully compliant.
8. Validation-to-holdout transfer remains the principal unresolved challenge.

## Safe academic conclusion

> Under matched training and evaluation, the five-exit architecture provided greater opportunities for computation reduction than the three-exit architecture. However, the selected full sequential policies did not satisfy all corrected-holdout quality constraints. The most promising operating point excluded Exit 1 and routed selected samples from Exit 3 to the final Exit 5, achieving 9.18% estimated FLOP reduction and a 1.037× measured speedup while satisfying three of four quality constraints. These findings support deeper sequential routing as a promising direction, but do not establish an optimal or deployment-ready policy.

## Paper-ready LaTeX

Complete canonical, historical v0.17, fair v0.18, and policy-structure LaTeX tables are stored in `LATEX_TABLES.md`.

## Non-claims

Do not claim that v0.18 is optimal or deployment-ready; that five exits are universally superior; that v0.17 is a fair architecture comparison; that validation eligibility guarantees holdout compliance; that FLOP estimates equal measured speedup; that v0.18 is label-wise asynchronous; or that continuation risk is a training-loss penalty.
