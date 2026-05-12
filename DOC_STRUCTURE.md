# Documentation Structure — `kexit_multi-label_EE_lossweight`

This document records the report/thesis structure for the loss-weight branch.

```text
Branch: kexit_multi-label_EE_lossweight
Base branch: kexit_multi-label_greedy_EE
Main idea: strengthen intermediate exits using larger early-exit loss weights
```

## Main research question

> Can stronger early-exit supervision improve Exit 1 and Exit 2 enough to increase compute savings without major macro-F1 degradation?

## Main answer

Stronger loss weighting improves intermediate exits, especially Exit 2, and improves several dynamic policy trade-offs. However, Exit 1 remains weak, so loss weighting alone is insufficient to make the earliest exit reliable.

## Models

| Model | Tap blocks | Exits | Loss weights |
|---|---|---:|---|
| `3exit_lw060_posweight` | `1,3` | 3 | `[0.6, 0.6, 1.0]` |
| `3exit_lw080_posweight` | `1,3` | 3 | `[0.8, 0.6, 1.0]` |
| `5exit_lw060_posweight` | `1,2,3,4` | 5 | `[0.6, 0.6, 0.7, 0.9, 1.0]` |
| `5exit_lw080_posweight` | `1,2,3,4` | 5 | `[0.8, 0.7, 0.7, 0.9, 1.0]` |

## Key results

| Result | Value |
|---|---:|
| Best static result | `3exit_lw060_posweight`, Exit 3, macro-F1 `0.6671` |
| Best 3-exit dynamic result | `3exit_lw060_posweight`, Policy 002, macro-F1 `0.6579`, saved `3.56%` |
| Best 5-exit practical result | `5exit_lw080_posweight`, `min_exit=3, stable_k=2`, macro-F1 `0.6504`, saved `8.03%` |
| Exit 2 improvement, 3-exit | `0.5727 → 0.5909` |
| Exit 2 improvement, 5-exit | `0.4777 → 0.5067` |

## Interpretation

The loss-weight experiment is a successful controlled ablation because it improves Exit 2 and improves the quality-focused dynamic trade-off. It does not fully solve the earliest-exit problem because Exit 1 remains weak.

## Recommended thesis wording

> The early-exit loss-weighting experiment shows that stronger supervision of intermediate exits improves the multi-label early-exit trade-off. Exit 2 improves consistently across both 3-exit and 5-exit models. For the compact 3-exit model, enabling Exit 1 under a label-set stability policy now achieves higher macro-F1 and slightly greater estimated compute saving than the previous greedy-EE baseline. However, Exit 1 remains weak and over-predictive, indicating that loss weighting alone is insufficient to make the earliest head reliable. The strongest practical result is obtained by the 5-exit loss-weighted model with `min_exit=3, stable_k=2`, which achieves macro-F1 `0.6504` with `8.03%` estimated depth-compute saving.
