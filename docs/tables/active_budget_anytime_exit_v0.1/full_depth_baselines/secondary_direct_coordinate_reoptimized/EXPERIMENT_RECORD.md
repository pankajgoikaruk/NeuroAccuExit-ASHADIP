# Secondary Direct Coordinate Re-optimisation Record

This is a secondary post-hoc inference-policy result and is not the Early-Exit baseline.

| Setting | Value |
|---|---|
| Input | Same frozen Exit 3 probabilities as the primary result |
| Threshold range | 0.10 to 0.95 |
| Threshold step | 0.01 |
| Maximum iterations | 20 |
| Objective | `global_consistency` |
| Aggregations | mean, max, top2mean, top3mean, p75, p90 |

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8599 |
| Micro-F1 | 0.9547 |
| Samples-F1 | 0.9620 |
| Exact Match | 0.8800 |
| Hamming Loss | 0.0131 |
| Average predicted labels | 1.4348 |
