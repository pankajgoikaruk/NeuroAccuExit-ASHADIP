# Cross-Version and Historical Comparison Tables

This document consolidates the canonical three-exit comparison, the historical v0.17 five-exit result, and the fair v0.18 architecture study.

## Reporting rule

- The **canonical cross-version ranking** contains only methods evaluated with the canonical three-exit checkpoint and frozen historical LATS-v2 protocol.
- The v0.17 five-exit result is retained as an important **within-checkpoint historical result**, but it must not be used as evidence of fair five-exit superiority because its training manifest differed.
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
| **v0.18 strict sequential** | Segment, sequential | Exit 1 → Exit 2 → Exit 3 | **3.82%** | **0.85285** | **0.94530** | See authoritative output | **0.86159** | **0.01603** |
| v0.13 logistic gate | Segment | Exit 2 → Exit 3 | 11.30% | 0.833034 | 0.943529 | 0.949750 | 0.855825 | 0.016609 |
| v0.14 Exit-2 parent-aware | Segment | Exit 2 → Exit 3 | 13.05% | 0.840798 | 0.933966 | 0.942473 | 0.835063 | 0.019262 |

The v0.18 compact execution record did not print a rounded parent Samples-F1 value for `full_strict`; the generated CSV/JSON output is authoritative and should be used when preparing the final paper table.

## Paper-ready LaTeX: canonical three-exit table

```latex
\begin{table}[H]
\centering
\caption{Cross-version corrected-holdout ablation for the canonical three-exit architecture.}
\label{tab:cross_version}
\resizebox{\textwidth}{!}{
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Method} & \textbf{FLOPs saved} & \textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{Samples-F1} & \textbf{Exact} & \textbf{Hamming $\downarrow$} \\
\midrule
Always Exit 3 & 0.00\% & 0.862382 & 0.953131 & 0.958889 & 0.876586 & 0.013725 \\
v0.13 per-label margin & 1.44\% & 0.858748 & 0.951556 & 0.957198 & 0.874279 & 0.014187 \\
v0.13 logistic gate & 11.30\% & 0.833034 & 0.943529 & 0.949750 & 0.855825 & 0.016609 \\
v0.14 Exit 1--3 ablation & 0.69\% & 0.861442 & 0.952756 & 0.958697 & 0.876586 & 0.013841 \\
v0.14 Exit 2--3 gate & 13.05\% & 0.840798 & 0.933966 & 0.942473 & 0.835063 & 0.019262 \\
v0.15 nonparametric parent risk & 0.44\% & 0.863129 & 0.952681 & 0.958505 & 0.875433 & 0.013841 \\
v0.15 shared logistic parent gate & 0.00\% & 0.862382 & 0.953131 & 0.958889 & 0.876586 & 0.013725 \\
v0.16 multi-objective margin & 5.06\% & 0.849203 & 0.942474 & 0.950266 & 0.854671 & 0.016840 \\
v0.17 sequential anytime exit & 8.64\% & 0.840128 & 0.937549 & 0.945653 & 0.840830 & 0.018224 \\
v0.18 strict sequential anytime exit & 3.82\% & 0.85285 & 0.94530 & -- & 0.86159 & 0.01603 \\
\bottomrule
\end{tabular}}
\end{table}
```

Use the authoritative v0.18 result CSV to replace `--` before paper submission.

## Historical v0.17 five-exit within-checkpoint result

| Method | Stop unit | Decision route | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 5 | None | Exit 5 only | 0.00% | 1.000× | 0.810761 | 0.869498 | 0.887906 | 0.673587 | 0.038985 |
| v0.17 five-exit sequential | Segment, sequential | Exit 1 → 2 → 3 → 4 → 5 | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.886945 | 0.688581 | 0.039100 |

### Historical interpretation

The v0.17 policy stopped 52.94% of segments before Exit 5 and passed all four quality-loss limits relative to its own full-depth baseline. This remains an important result showing that sequential multi-exit routing can work. It is not a fair architecture comparison because the three-exit and five-exit checkpoints were trained with different manifests.

## Paper-ready LaTeX: historical v0.17 five-exit result

```latex
\begin{table}[H]
\centering
\caption{Historical corrected-holdout result for the v0.17 five-exit sequential anytime architecture.}
\label{tab:v017_five_exit}
\resizebox{\textwidth}{!}{
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Method} & \textbf{FLOPs saved} & \textbf{Speedup} & \textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{Samples-F1} & \textbf{Exact} & \textbf{Hamming $\downarrow$} \\
\midrule
Always Exit 5 & 0.00\% & 1.000$\times$ & 0.810761 & 0.869498 & 0.887906 & 0.673587 & 0.038985 \\
v0.17 five-exit sequential anytime policy & 30.71\% & 1.114$\times$ & 0.801356 & 0.868859 & 0.886945 & \textbf{0.688581} & 0.039100 \\
\bottomrule
\end{tabular}}
\end{table}
```

## Fair v0.18 architecture headline

| Architecture | Policy | Decision route | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ | Holdout status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 3-exit | Always final | Exit 3 only | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.876586 | 0.013725 | Reference |
| 3-exit | `full_strict` | Exit 1 → Exit 2 → Exit 3 | 3.82% | 1.018× | 0.85285 | 0.94530 | 0.86159 | 0.01603 | Failed 3/4 limits |
| 5-exit | Always final | Exit 5 only | 0.00% | 1.000× | 0.82097 | 0.90734 | 0.77970 | 0.02780 | Reference |
| 5-exit | `full_strict` | Exit 1 → Exit 3 → Exit 5 in practice | 12.70% | 1.057× | 0.79813 | 0.89832 | 0.75779 | 0.03010 | Failed 4/4 limits |
| 5-exit | `no_exit1` | Exit 3 → Exit 5 | **9.18%** | **1.037×** | **0.81015** | **0.90375** | **0.77163** | **0.02872** | Passed 3/4; Macro drop exceeded by 0.000819 |

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

## Non-claims

Do not claim that:

- v0.18 is optimal or deployment-ready;
- five exits are universally superior;
- the v0.17 five-exit result is a fair architecture comparison;
- validation eligibility guarantees holdout compliance;
- estimated FLOPs and measured speedup are interchangeable;
- v0.18 performs label-wise asynchronous inference;
- the continuation-risk score is a training-loss penalty.
