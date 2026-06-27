| Method | Macro-F1 mean | Micro-F1 mean | Samples-F1 mean | Exact mean | Hamming Loss ↓ mean | Role |
|---|---:|---:|---:|---:|---:|---|
| max fixed thresholds | 0.7187 | 0.8200 | 0.8419 | 0.5098 | 0.0632 | Rejected: over-predicts labels. |
| mean fixed thresholds | 0.7802 | 0.9315 | 0.9392 | 0.8371 | 0.0199 | Strong baseline. |
| top2mean fixed thresholds | 0.8023 | 0.8884 | 0.9060 | 0.6927 | 0.0358 | Helps Macro-F1 but hurts reliability. |
| **v06 selected aggregation fixed thresholds** | **0.8310** | **0.9345** | **0.9449** | **0.8368** | **0.0193** | Best balanced repeated-split strategy. |
| v07 aggregation + threshold calibrated | 0.8319 | 0.9288 | 0.9363 | 0.8185 | 0.0208 | Macro good, but weaker overall. |
