# Documentation Structure — `kexit_human_talk_incremental_eval`

This document records the report/thesis structure for the human-talk incremental evaluation branch.

```text
Branch: kexit_human_talk_incremental_eval
Base branch: kexit_multi-label_EE_lossweight
Main idea: test whether the K-exit audio model generalises to clean human-talk speaker classification and whether 5 exits improve the early-exit trade-off
```

## Main research question

> Can the K-exit NeuroAccuExit audio model generalise from the previous multi-label environmental setup to a clean human-talk speaker benchmark, and does a 5-exit design create a better early-exit accuracy/compute trade-off than the 3-exit baseline?

## Main answer

> Yes for Stage 1. On the clean two-speaker benchmark, both 3-exit and 5-exit models reached near-perfect final-exit performance. The 5-exit model produced a smoother intermediate-exit progression and a practical dynamic policy: macro-F1 `0.9883` with `19.56%` estimated depth-compute saving. However, this is a Stage 1 sanity-check result, not yet evidence of robust multi-speaker or noisy real-world generalisation.

## Stage design

| Stage | Classes | Data mode | Purpose | Status |
|---|---:|---|---|---|
| Stage 1 | 2 clean speakers | balanced | Check whether the model can separate two speakers clearly | Completed |
| Stage 2 | 3 clean speakers | balanced | Test class scalability | Next |
| Stage 3 | 4 clean speakers | balanced | Test added speaker confusion | Planned |
| Stage 4 | 5 clean speakers | balanced | Full clean-speaker benchmark | Planned |
| Stage 5 | 9 speakers | full / noisy | Realistic multi-speaker stress test | Later |

## Stage 1 models

| Model | Tap blocks | Exits | Loss weights | Main role |
|---|---|---:|---|---|
| `human_talk_clean2_3exit_nohint` | `1,3` | 3 | `[0.3, 0.3, 1.0]` | Compact baseline |
| `human_talk_clean2_5exit_nohint` | `1,2,3,4` | 5 | `[0.3, 0.3, 0.6, 0.8, 1.0]` | More early-exit opportunities |

## Key Stage 1 results

| Result | Value |
|---|---:|
| Total selected parent clips | `944` |
| Total one-second child segments | `8,496` |
| 3-exit final macro-F1 | `0.9926` |
| 5-exit final macro-F1 | `0.9930` |
| 5-exit selected policy | `min_exit=3, stable_k=2` |
| 5-exit selected-policy macro-F1 | `0.9883` |
| 5-exit selected-policy estimated saving | `19.56%` |
| Best accuracy/saving 5-exit policy | `min_exit=2, stable_k=3`, macro-F1 `0.9898`, saved `18.39%` |

## Interpretation

The human-talk Stage 1 experiment is a successful sanity-check. It shows that the same K-exit model can learn clean speaker-discriminative representations, and that adding more exits gives a smoother accuracy-depth curve. The 5-exit model is more useful for dynamic early exiting than the 3-exit model on this task because it can stop near Exit 4 with only a small macro-F1 reduction.

## Recommended thesis wording

> The human-talk Stage 1 experiment validates the generality of the NeuroAccuExit multi-exit audio pipeline on a clean two-speaker benchmark. Both 3-exit and 5-exit models reach near-perfect final-exit performance, while the 5-exit model provides a practical early-exit trade-off, achieving macro-F1 `0.9883` with `19.56%` estimated depth-compute saving under a label-set stability policy. This supports the use of additional exits as a mechanism for improving dynamic inference flexibility, although the result should be treated as a Stage 1 sanity check rather than final evidence of robust multi-speaker generalisation.

## Known issue to document

| Issue | Impact | Action |
|---|---|---|
| `renamed_format_ok=0` for already renamed files | Does not affect labels or accuracy, but makes parent IDs less clean | Fix parser before Stage 2 |
| Parent IDs duplicate class prefix | Traceability is less readable | Expected format should be `Les_Brown__0450` |

---

# Previous reference structure retained below

The previous loss-weight documentation structure is retained below so the original result-table style is preserved.

---

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

