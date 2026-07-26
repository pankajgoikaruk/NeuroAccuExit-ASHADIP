# v0.18_EE LaTeX Tables

## Canonical three-exit cross-version corrected-holdout table

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
v0.18 strict sequential anytime exit & 3.82\% & 0.852849 & 0.945297 & 0.952277 & 0.861592 & 0.016032 \\
\bottomrule
\end{tabular}}
\end{table}
```

## Historical v0.17 five-exit result

```latex
\begin{table}[H]
\centering
\caption{Historical corrected-holdout results for the v0.17 five-exit checkpoint.}
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

This table must retain the non-fair-training caution.

## Fair v0.18 architecture comparison

```latex
\begin{table}[H]
\centering
\caption{Fair corrected-holdout comparison of the v0.18 three-exit and five-exit architectures.}
\label{tab:v018_fair_architecture}
\resizebox{\textwidth}{!}{
\begin{tabular}{llrrrrrrr}
\toprule
\textbf{Architecture} & \textbf{Policy} & \textbf{FLOPs saved} & \textbf{Speedup} & \textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{Samples-F1} & \textbf{Exact} & \textbf{Hamming $\downarrow$} \\
\midrule
3-exit & Always final & 0.00\% & 1.000$\times$ & 0.862382 & 0.953131 & 0.958889 & 0.876586 & 0.013725 \\
3-exit & Full strict & 3.82\% & 1.018$\times$ & 0.852849 & 0.945297 & 0.952277 & 0.861592 & 0.016032 \\
5-exit & Always final & 0.00\% & 1.000$\times$ & 0.820972 & 0.907343 & 0.923623 & 0.779700 & 0.027797 \\
5-exit & Full strict & 12.70\% & 1.057$\times$ & 0.798133 & 0.898325 & 0.913665 & 0.757785 & 0.030104 \\
5-exit & No Exit 1 & 9.18\% & 1.037$\times$ & 0.810153 & 0.903750 & 0.918894 & 0.771626 & 0.028720 \\
\bottomrule
\end{tabular}}
\end{table}
```

## Policy-structure comparison through v0.18

```latex
\begin{table}[H]
\centering
\caption{Policy-structure comparison for corrected-holdout early-exit experiments.}
\label{tab:policy_structure}
\resizebox{\textwidth}{!}{
\begin{tabular}{lllrrrrr}
\toprule
\textbf{Method} & \textbf{Stop unit} & \textbf{Decision route} & \textbf{FLOPs saved} & \textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{Exact} & \textbf{Hamming $\downarrow$} \\
\midrule
Always Exit 3 & None & Exit 3 only & 0.00\% & 0.86238 & 0.95313 & 0.87659 & 0.01373 \\
v0.13 per-label margin & Segment & Exit 2 $\rightarrow$ Exit 3 & 1.44\% & 0.85875 & 0.95156 & 0.87428 & 0.01419 \\
v0.14 Exit-1 ablation & Segment & Exit 1 $\rightarrow$ Exit 3 & 0.69\% & 0.86144 & 0.95276 & 0.87659 & 0.01384 \\
v0.15 nonparametric parent & Parent & Exit 2 $\rightarrow$ Exit 3 & 0.44\% & 0.86313 & 0.95268 & 0.87543 & 0.01384 \\
v0.15 shared logistic & Parent & Exit 2 $\rightarrow$ Exit 3 & 0.00\% & 0.86238 & 0.95313 & 0.87659 & 0.01373 \\
v0.16 multi-objective margin & Segment & Exit 2 $\rightarrow$ Exit 3 & 5.06\% & 0.84920 & 0.94247 & 0.85467 & 0.01684 \\
v0.17 sequential anytime & Segment, sequential & Exit 1 $\rightarrow$ Exit 2 $\rightarrow$ Exit 3 & 8.64\% & 0.84013 & 0.93755 & 0.84083 & 0.01822 \\
v0.18 strict sequential & Segment, sequential & Exit 1 $\rightarrow$ Exit 2 $\rightarrow$ Exit 3 & 3.82\% & 0.85285 & 0.94530 & 0.86159 & 0.01603 \\
v0.18 fair 5-exit No Exit 1 & Segment, sequential & Exit 3 $\rightarrow$ Exit 5 & 9.18\% & 0.81015 & 0.90375 & 0.77163 & 0.02872 \\
\bottomrule
\end{tabular}}
\end{table}
```

### Caption cautions

- The canonical cross-version table is restricted to the comparable 3-exit family.
- The v0.17 five-exit table is historical and non-fair.
- The v0.18 architecture table is training-fair, but each policy is compared with its own final-exit reference.
- Validation eligibility and holdout compliance must be reported separately.
