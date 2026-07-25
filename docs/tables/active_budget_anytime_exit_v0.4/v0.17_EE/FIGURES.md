# v0.17_EE Figures

These repository-rendered figures summarise the confirmed v0.17 results. Exact values remain in the CSV and JSON records.

## Sequential routes

```mermaid
flowchart LR
    A[Input sample] --> E1[Exit 1]
    E1 -->|easy: stop| S1[Prediction]
    E1 -->|continue| E2[Exit 2]
    E2 -->|moderate: stop| S2[Prediction]
    E2 -->|continue| E3[Exit 3]
    E3 -->|3-exit final| F3[Final prediction]
    E3 -->|5-exit continue| E4[Exit 4]
    E4 -->|hard: stop| S4[Prediction]
    E4 -->|continue| E5[Exit 5 final]
```

## Exit distribution

```mermaid
xychart-beta
    title "v0.17 holdout exit distribution (%)"
    x-axis [E1, E2, E3, E4, E5]
    y-axis "Percent" 0 --> 100
    bar [6.07, 4.34, 89.60, 0, 0]
    bar [6.83, 1.22, 18.59, 26.30, 47.06]
```

First bar series: 3-exit policy. Second bar series: 5-exit policy.

## Estimated saving and measured speedup

```mermaid
xychart-beta
    title "Quality-constrained sequential efficiency"
    x-axis [Three-exit, Five-exit]
    y-axis "Estimated FLOPs saved (%)" 0 --> 35
    bar [8.64, 30.71]
```

| Architecture | Estimated FLOPs saved | Measured speedup | Holdout limits |
|---|---:|---:|---|
| 3-exit | 8.64% | 1.037× | Failed |
| 5-exit | 30.71% | 1.114× | Passed |

## Cross-version corrected-holdout plots

The following figures summarise the discrete corrected-holdout operating points in the canonical 3-exit experiment family:

![Cross-version corrected-holdout compute saving](cross_version_3exit_flops.svg)

![Cross-version corrected-holdout quality metrics](cross_version_3exit_quality.svg)

![Cross-version corrected-holdout Hamming loss](cross_version_3exit_hamming.svg)

The architecture-extension figure reports the two v0.17 policies against their own full-depth references:

![v0.17 3-exit and 5-exit architecture extension](v017_architecture_flops.svg)

Supporting tables:

```text
cross_version_3exit_table.csv
v017_architecture_table.csv
```

These are line plots over discrete policy versions, not continuous training or optimisation curves. The 5-exit point must remain separate from the canonical 3-exit ranking because the fairness audit failed `same_validation_manifest`.

## Ablation interpretation map

```mermaid
flowchart TD
    Full[Full sequential policy] --> Exit1[Remove Exit 1]
    Full --> Stable[Remove stability]
    Full --> Risk[Remove risk]
    Full --> Margin[Remove label margins]
    Full --> Conf[Confidence only]

    Exit1 -->|less saving, safer quality| A1[Exit 1 is useful but risky]
    Stable -->|more saving, worse quality| A2[Stability protects]
    Risk -->|same parent metrics| A3[Current risk is non-binding]
    Margin -->|severe quality collapse| A4[Label margins are essential]
    Conf -->|largest saving, unsafe| A5[Global confidence is insufficient]
```

## Fairness warning

```mermaid
flowchart LR
    C3[Canonical 3-exit checkpoint<br/>25,519 training rows] --> H[Shared holdout and evaluator]
    C5[Tested 5-exit checkpoint<br/>30,950 training rows] --> H
    H --> W[Within-checkpoint results valid]
    H --> X[Direct architecture superiority invalid]
```
