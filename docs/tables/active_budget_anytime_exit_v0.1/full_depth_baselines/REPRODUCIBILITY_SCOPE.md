# Reproducibility Scope

## Exactly reproducible result

This package deterministically reproduces the canonical parent-level
v0.10 no-hint + historical frozen LATS-v2 metrics from the frozen
Exit 3 segment-probability CSV and frozen LATS-v2 configuration.

The deterministic replay performs:

- no neural-network retraining;
- no threshold optimisation;
- no aggregation-method search;
- no modification of the frozen probabilities.

## Canonical result

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8623815322333925 |
| Micro-F1 | 0.9531311539976368 |
| Samples-F1 | 0.9588894381281925 |
| Exact Match | 0.8765859284890427 |
| Hamming Loss | 0.013725490196078431 |
| Average predicted labels | 1.4590542099192618 |
| Parent clips | 867 |

## End-to-end training scope

The frozen package guarantees exact reproduction from the stored
segment probabilities onward. Exact neural-network retraining from
the original audio data is not guaranteed unless the original
checkpoint, random seeds, training configuration, data ordering,
and complete software environment are also preserved.

For all Early-Exit experiments, this frozen parent-level result is
the canonical full-depth quality baseline.
