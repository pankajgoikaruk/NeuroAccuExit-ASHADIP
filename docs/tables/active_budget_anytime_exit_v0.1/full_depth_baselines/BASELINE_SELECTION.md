# Frozen Full-Depth Baselines

## Primary canonical baseline

This is the only full-depth baseline to be used for all subsequent standard
Early-Exit, budget-aware Early-Exit, and anytime-inference experiments.

- Method: v0.10 no-hint + frozen historical LATS-v2
- Evaluation level: parent/clip level
- Parent clips: 867
- Macro-F1: 0.8623815322333925
- Micro-F1: 0.9531311539976368
- Samples-F1: 0.9588894381281925
- Exact Match: 0.8765859284890427
- Hamming Loss: 0.013725490196078431
- Average predicted labels: 1.4590542099192618

Rounded reporting values:

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8624 |
| Micro-F1 | 0.9531 |
| Samples-F1 | 0.9589 |
| Exact Match | 0.8766 |
| Hamming Loss | 0.0137 |

Important: 1.4591 is the average number of predicted labels per parent clip.
It is not average exit depth.

## Secondary frozen result

The direct coordinate re-optimisation result is retained only as a secondary
post-hoc inference-policy result. It must not replace the primary baseline in
future Early-Exit comparisons.

Approximate secondary result:

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8599 |
| Micro-F1 | 0.9547 |
| Samples-F1 | 0.9620 |
| Exact Match | 0.8800 |
| Hamming Loss | 0.0131 |
| Average predicted labels | 1.4348 |

## Baseline rule

All future quality-retention, accuracy-loss, compute-saving, latency, exit-depth,
budget, and anytime comparisons must use the primary canonical baseline.
