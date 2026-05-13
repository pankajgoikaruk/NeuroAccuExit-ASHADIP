# Appendix — `kexit_multi-label_EE_lossweight`

This appendix records the reproducibility protocol and extended result summary for the active loss-weight branch.

```text
Branch: kexit_multi-label_EE_lossweight
Base branch: kexit_multi-label_greedy_EE
Task: multi-label audio tagging
Main idea: strengthen intermediate exits using larger early-exit loss weights
```

## A1. Branch status

| Item | Value |
|---|---|
| Active branch | `kexit_multi-label_EE_lossweight` |
| Base branch | `kexit_multi-label_greedy_EE` |
| Compared models | `3exit_lw060_posweight`, `3exit_lw080_posweight`, `5exit_lw060_posweight`, `5exit_lw080_posweight` |
| Positive weighting | enabled |
| `pos_weight_max` | `20.0` |
| Thresholding | tuned per-exit/per-label thresholds |
| Policy | greedy label-set stability |
| Main metric | macro-F1 with hamming loss and estimated depth-compute saving |

## A2. Loss-weight variants

| Model | Tap blocks | Exits | Loss weights |
|---|---|---:|---|
| `3exit_lw060_posweight` | `1,3` | 3 | `[0.6, 0.6, 1.0]` |
| `3exit_lw080_posweight` | `1,3` | 3 | `[0.8, 0.6, 1.0]` |
| `5exit_lw060_posweight` | `1,2,3,4` | 5 | `[0.6, 0.6, 0.7, 0.9, 1.0]` |
| `5exit_lw080_posweight` | `1,2,3,4` | 5 | `[0.8, 0.7, 0.7, 0.9, 1.0]` |

## A3. Experiment output layout

```text
runs_multilabel/
└─ lossweight/
   ├─ training/
   │  └─ multilabel_posweight_lossweight/
   ├─ summary/
   │  └─ threshold_summary_001_lossweight_static_tuned/
   └─ policy_eval/
      └─ multilabel_greedy_policy/
         ├─ lossweight_policy_001_minexit2_stable2/
         ├─ lossweight_policy_002_minexit1_stable2/
         └─ lossweight_policy_best_5exit_minexit3_stable2/
```

## A4. Static result summary

| Model | Best Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| `3exit_lw060_posweight` | 3 | **0.6671** | 0.6382 | 0.6628 | 0.2416 | 0.1382 |
| `3exit_lw080_posweight` | 3 | 0.6496 | 0.6344 | 0.6563 | 0.2809 | 0.1295 |
| `5exit_lw060_posweight` | 4 | 0.6508 | **0.6485** | **0.6732** | **0.3174** | **0.1197** |
| `5exit_lw080_posweight` | 5 | 0.6519 | 0.6443 | 0.6680 | 0.3034 | 0.1278 |

## A5. Early-exit diagnostic

| Model | Exit 2 Macro-F1 | Change vs baseline | Hamming Loss | Change vs baseline |
|---|---:|---:|---:|---:|
| `3exit_lw060_posweight` | 0.5884 | +0.0157 | 0.1756 | +0.0051 |
| `3exit_lw080_posweight` | **0.5909** | **+0.0182** | **0.1621** | **-0.0084** |
| `5exit_lw060_posweight` | 0.5030 | +0.0253 | 0.2312 | -0.0143 |
| `5exit_lw080_posweight` | **0.5067** | **+0.0290** | **0.2146** | **-0.0309** |

Exit 1 remains weak. Its macro-F1 remains around `0.41–0.42`, so it should not be used as an independent stopping point.

## A6. Dynamic policy summary

### Policy 001: `min_exit=2, stable_k=2`

| Model | Macro-F1 | Samples-F1 | Hamming Loss | Avg Exit Depth | Compute Saved |
|---|---:|---:|---:|---:|---:|
| `3exit_lw060_posweight` | **0.6671** | 0.6628 | 0.1382 | 3.0000 / 3 | 0.00% |
| `3exit_lw080_posweight` | 0.6496 | 0.6563 | 0.1295 | 3.0000 / 3 | 0.00% |
| `5exit_lw060_posweight` | 0.6251 | 0.6479 | 0.1396 | 4.2219 / 5 | 15.56% |
| `5exit_lw080_posweight` | **0.6418** | **0.6631** | **0.1315** | 4.2275 / 5 | 15.45% |

### Policy 002: `min_exit=1, stable_k=2`

| Model | Macro-F1 | Samples-F1 | Hamming Loss | Avg Exit Depth | Compute Saved |
|---|---:|---:|---:|---:|---:|
| `3exit_lw060_posweight` | **0.6579** | 0.6616 | 0.1503 | 2.8933 / 3 | 3.56% |
| `3exit_lw080_posweight` | 0.6475 | 0.6579 | **0.1362** | 2.8961 / 3 | 3.46% |
| `5exit_lw060_posweight` | 0.6225 | 0.6478 | 0.1480 | 3.9522 / 5 | **20.96%** |
| `5exit_lw080_posweight` | **0.6305** | **0.6579** | **0.1427** | 3.9551 / 5 | 20.90% |

### Best 5-exit target: `min_exit=3, stable_k=2`

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Exit Depth | Compute Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5exit_lw060_posweight` | 0.6345 | 0.6277 | 0.6543 | 0.2949 | 0.1323 | 4.5815 / 5 | 8.37% |
| `5exit_lw080_posweight` | **0.6504** | **0.6443** | **0.6693** | **0.3174** | **0.1253** | 4.5983 / 5 | 8.03% |

## A7. Main appendix conclusion

The loss-weight branch is a successful controlled ablation. It improves Exit 2 and strengthens the quality-focused early-exit trade-off, but it does not fully solve Exit 1 reliability. The recommended headline result is `5exit_lw080_posweight` with `min_exit=3, stable_k=2`, macro-F1 `0.6504`, and `8.03%` estimated depth-compute saving.
