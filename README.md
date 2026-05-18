# NeuroAccuExit Human-Talk Incremental Evaluation — `kexit_human_talk_incremental_eval`

This branch extends the multi-label early-exit pipeline to a **human-talk / speaker-classification sanity-check benchmark**. The goal is to test whether the same TinyAudioCNN + ExitNet multi-exit design can learn clean speaker-discriminative audio representations and whether additional exits improve the dynamic accuracy/compute trade-off.

```text
Branch: kexit_human_talk_incremental_eval
Base branch: kexit_multi-label_EE_lossweight
Task: clean human-talk speaker classification
Stage: clean2_balanced
Classes: Les_Brown vs Simon_Sinek
Model: TinyAudioCNN + ExitNet
Outputs: sigmoid/BCE label heads
Policy: greedy label-set stability
Workspace: human_talk_workspace/stages/clean2_balanced
```

---

## Executive summary

1. Stage 1 uses two manually cleaned speaker classes: `Les_Brown` and `Simon_Sinek`.
2. The dataset was balanced to `472` parent clips per class.
3. Each 5-second parent clip was segmented into `9` one-second child windows using `segment_sec=1.0` and `hop_sec=0.5`.
4. The final balanced Stage 1 dataset contains `944` parent clips and `8,496` one-second segments.
5. The 3-exit baseline reached final-exit macro-F1 `0.9926` on the test set.
6. The 5-exit baseline reached final-exit macro-F1 `0.9930` on the test set.
7. The selected 5-exit dynamic policy `min_exit=3, stable_k=2` reached macro-F1 `0.9883` with `19.56%` estimated depth-compute saving.
8. The 5-exit model shows a smoother exit-quality progression than the 3-exit model.
9. Exit 1 remains too weak for aggressive early stopping, consistent with the previous multi-label branches.
10. A metadata parser issue was detected: renamed files are classified correctly, but `renamed_format_ok` is currently `0` and parent IDs duplicate the class prefix. This should be fixed before Stage 2.

---

## Research question

> Can the K-exit NeuroAccuExit audio model generalise from the previous multi-label environmental setup to a clean human-talk speaker benchmark, and does a 5-exit design create a better early-exit accuracy/compute trade-off than the 3-exit baseline?

Answer:

> Yes for Stage 1. On the clean two-speaker benchmark, both 3-exit and 5-exit models reached near-perfect final-exit performance. The 5-exit model produced a smoother intermediate-exit progression and a practical dynamic policy: macro-F1 `0.9883` with `19.56%` estimated depth-compute saving. However, this is a Stage 1 sanity-check result, not yet evidence of robust multi-speaker or noisy real-world generalisation.

---

## Research finding from Stage 1

> The clean two-speaker benchmark confirms that NeuroAccuExit can learn strong speaker-discriminative audio representations and that a 5-exit model can provide a meaningful early-exit trade-off. Compared with the 3-exit model, the 5-exit model has a clearer monotonic improvement from shallow to deep exits and achieves nearly final-exit accuracy while saving about one-fifth of estimated depth-compute units.

---

## Stage 1 dataset summary

| Item | Value |
|---|---:|
| Stage | `clean2_balanced` |
| Raw dataset root | `human_talk_dataset` |
| Workspace | `human_talk_workspace/stages/clean2_balanced` |
| Classes | `Les_Brown`, `Simon_Sinek` |
| Original `Les_Brown` clips | 472 |
| Original `Simon_Sinek` clips | 608 |
| Selected clips per class | 472 |
| Total selected parent clips | 944 |
| Child segments per parent clip | 9 |
| Total child segments | 8496 |
| Sample rate after preparation | 16 kHz |
| Segment length | 1.0 sec |
| Hop length | 0.5 sec |

### Parent-clip split

| Split | Les_Brown | Simon_Sinek | Total |
|---|---:|---:|---:|
| Train | 330 | 330 | 660 |
| Val | 71 | 71 | 142 |
| Test | 71 | 71 | 142 |

### Segment split

| Split | Les_Brown | Simon_Sinek | Total |
|---|---:|---:|---:|
| Train | 2970 | 2970 | 5940 |
| Val | 639 | 639 | 1278 |
| Test | 639 | 639 | 1278 |

---

## Stage 1 models

| Model | Tap blocks | Exits | Loss weights | Pos-weight | Threshold mode | Purpose |
|---|---|---:|---|---|---|---|
| `human_talk_clean2_3exit_nohint` | `1,3` | 3 | `[0.3, 0.3, 1.0]` | No | fixed `0.5` | Compact baseline |
| `human_talk_clean2_5exit_nohint` | `1,2,3,4` | 5 | `[0.3, 0.3, 0.6, 0.8, 1.0]` | No | fixed `0.5` | More exit opportunities |

---

## Static per-exit test results

### 3-exit model

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.8765 | 0.8765 | 0.8732 | 0.8701 | 0.1232 | Too shallow |
| 2 | 0.9765 | 0.9765 | 0.9765 | 0.9765 | 0.0235 | Strong intermediate exit |
| 3 | 0.9926 | 0.9926 | 0.9922 | 0.9922 | 0.0074 | Excellent final exit |

### 5-exit model

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.8799 | 0.8799 | 0.8785 | 0.8748 | 0.1201 | Too shallow |
| 2 | 0.9259 | 0.9260 | 0.9244 | 0.9233 | 0.0739 | Improving but not enough |
| 3 | 0.9703 | 0.9703 | 0.9700 | 0.9695 | 0.0297 | Good intermediate exit |
| 4 | 0.9883 | 0.9883 | 0.9883 | 0.9883 | 0.0117 | Very strong practical exit |
| 5 | 0.9930 | 0.9930 | 0.9930 | 0.9930 | 0.0070 | Best final exit |

---

## Dynamic early-exit policy comparison

| Model | Selected policy | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Exit Depth | Compute Saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3exit_nohint` | `min_exit=2, stable_k=2` | 0.9926 | 0.9922 | 0.9922 | 0.0074 | 3.0000 / 3 | 0.00% |
| `5exit_nohint` | `min_exit=3, stable_k=2` | 0.9883 | 0.9883 | 0.9883 | 0.0117 | 4.0219 / 5 | 19.56% |

### 5-exit selected-policy exit distribution

| Exit | Samples | Fraction |
| ---: | ---: | ---: |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |
| 4 | 1250 | 97.81% |
| 5 | 28 | 2.19% |

---

## Policy sweep highlights

| Model | Policy | Macro-F1 | Samples-F1 | Hamming Loss | Avg Exit Depth | Compute Saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3exit_nohint` | `min_exit=1, stable_k=2` | 0.9836 | 0.9836 | 0.0164 | 2.1252 / 3 | 29.16% |
| `3exit_nohint` | `min_exit=2, stable_k=1` | 0.9765 | 0.9765 | 0.0235 | 2.0000 / 3 | 33.33% |
| `3exit_nohint` | `min_exit=2, stable_k=2` | 0.9926 | 0.9922 | 0.0074 | 3.0000 / 3 | 0.00% |
| `5exit_nohint` | `min_exit=1, stable_k=3` | 0.9765 | 0.9765 | 0.0235 | 3.1854 / 5 | 36.29% |
| `5exit_nohint` | `min_exit=2, stable_k=3` | 0.9898 | 0.9898 | 0.0102 | 4.0806 / 5 | 18.39% |
| `5exit_nohint` | `min_exit=3, stable_k=2` | 0.9883 | 0.9883 | 0.0117 | 4.0219 / 5 | 19.56% |
| `5exit_nohint` | `min_exit=3, stable_k=3` | 0.9930 | 0.9930 | 0.0070 | 5.0000 / 5 | 0.00% |

Recommended reporting options:

| Reporting goal | Recommended policy | Reason |
|---|---|---|
| Highest 5-exit dynamic accuracy with saving | `min_exit=2, stable_k=3` | Macro-F1 `0.9898`, saved `18.39%` |
| Slightly higher compute saving | `min_exit=3, stable_k=2` | Macro-F1 `0.9883`, saved `19.56%` |
| Accuracy-only reference | `min_exit=3, stable_k=3` | Macro-F1 `0.9930`, but no early-exit saving |

---

## Main Stage 1 findings

1. The preprocessing pipeline works for clean human-talk data.
2. The parent-child segmentation structure is valid for this benchmark.
3. Both 3-exit and 5-exit models achieve near-perfect final-exit performance.
4. The 5-exit model produces a clearer monotonic improvement across exits.
5. The selected 5-exit dynamic policy exits `97.81%` of samples at Exit 4 and only `2.19%` at Exit 5.
6. The 5-exit dynamic result gives a practical trade-off: macro-F1 `0.9883` with `19.56%` estimated depth-compute saving.
7. The 3-exit selected policy `min_exit=2, stable_k=2` is too conservative and gives no saving.
8. Aggressive Exit 1 stopping remains unreliable, which is consistent with the loss-weight branch findings.

---

## Known metadata caveat before Stage 2

The current Stage 1 output shows a filename parser issue:

```text
Expected: parent_clip_id = Les_Brown__0450
Observed: parent_clip_id = Les_Brown__Les_Brown_0450
```

The model results remain valid because labels, splits, and segments are correct. However, this should be fixed before Stage 2 so that parent-child traceability is clean and reviewer-safe.

---

## Recommended next step

1. Fix `audit_human_talk_dataset.py` and `prepare_human_talk_segments.py` so the renamed format `Class_Name__0001.wav` is recognised correctly.
2. Rerun Stage 1 preparation and feature extraction only if clean metadata is needed.
3. Move to `clean3_balanced` after the metadata fix.
4. Compare whether the 5-exit advantage remains as the number of speakers increases.

---

# Previous reference branch retained below

The previous loss-weight branch documentation is retained below so the old table format and findings are not lost.

---

# NeuroAccuExit Multi-Label Early Exit Loss-Weight Study — `kexit_multi-label_EE_lossweight`

This branch records the controlled **early-exit loss-weighting ablation** for the multi-label NeuroAccuExit-ASHADIP pipeline.

```text
Branch: kexit_multi-label_EE_lossweight
Base branch: kexit_multi-label_greedy_EE
Task: multi-label audio tagging
Model: TinyAudioCNN + ExitNet
Outputs: sigmoid/BCE multi-label exits
Policy: greedy label-set stability
```

The previous `kexit_multi-label_greedy_EE` branch showed that dynamic multi-label early-exit inference works, but also showed a weakness: **Exit 1 was too weak, Exit 2 was only partly useful, and Exit 4 was the strongest practical early-exit point**. This branch tests whether stronger early-exit loss weighting can improve the intermediate heads without changing the architecture.

---

## Executive summary

1. No architecture change was required.
2. `training/train_multilabel.py` already supports `--loss_weights` and saves the selected values in `config_used.json`.
3. Four positive-weight loss-weight models were trained and evaluated.
4. Stronger loss weighting improves intermediate exits, especially Exit 2.
5. Exit 1 remains weak and should not yet be used as an independent stopping point.
6. The best 3-exit dynamic result is `3exit_lw060_posweight` under Policy 002: macro-F1 `0.6579`, estimated depth-compute saved `3.56%`.
7. The best 5-exit practical result is `5exit_lw080_posweight` with `min_exit=3, stable_k=2`: macro-F1 `0.6504`, samples-F1 `0.6693`, exact match `0.3174`, hamming loss `0.1253`, estimated depth-compute saved `8.03%`.
8. Compared with the previous 5-exit greedy-EE baseline, the new best model improves prediction quality but saves slightly less estimated depth compute.

---

## Research question

> Can stronger early-exit supervision improve Exit 1 and Exit 2 enough to increase compute savings without major macro-F1 degradation?

Answer:

> Loss weighting improves Exit 2 and improves several dynamic policy trade-offs. However, Exit 1 remains weak, so loss weighting alone is not sufficient to make the earliest head reliable.

---

## Loss-weight variants

| Model | Tap blocks | Exits | Pos-weight | `pos_weight_max` | Loss weights |
|---|---|---:|---|---:|---|
| `3exit_lw060_posweight` | `1,3` | 3 | Yes | 20.0 | `[0.6, 0.6, 1.0]` |
| `3exit_lw080_posweight` | `1,3` | 3 | Yes | 20.0 | `[0.8, 0.6, 1.0]` |
| `5exit_lw060_posweight` | `1,2,3,4` | 5 | Yes | 20.0 | `[0.6, 0.6, 0.7, 0.9, 1.0]` |
| `5exit_lw080_posweight` | `1,2,3,4` | 5 | Yes | 20.0 | `[0.8, 0.7, 0.7, 0.9, 1.0]` |

Baseline comparison:

| Baseline model | Previous loss weights | Key previous result |
|---|---|---|
| `3exit_posweight` | `[0.3, 0.3, 1.0]` | Policy 002 macro-F1 `0.6293`, saved `2.53%` |
| `5exit_posweight` | `[0.3, 0.3, 0.6, 0.8, 1.0]` | Best 5-exit macro-F1 `0.6449`, saved `9.33%` |

---

## Experiment store

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

---

## Static threshold-tuned results

| Model | Final Exit | Threshold | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `3exit_lw060_posweight` | 3 | fixed 0.5 | 0.6446 | 0.6413 | **0.6740** | 0.2725 | 0.1351 | 2.2051 |
| `3exit_lw060_posweight` | 3 | tuned | **0.6671** | 0.6382 | 0.6628 | 0.2416 | 0.1382 | 2.2584 |
| `3exit_lw080_posweight` | 3 | tuned | 0.6496 | 0.6344 | 0.6563 | 0.2809 | 0.1295 | 1.9803 |
| `5exit_lw060_posweight` | 5 | tuned | 0.6370 | 0.6285 | 0.6519 | 0.2809 | 0.1348 | 2.0674 |
| `5exit_lw080_posweight` | 5 | tuned | 0.6519 | **0.6443** | 0.6680 | **0.3034** | **0.1278** | 2.0309 |

### Best static exit per model

| Model | Best Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| `3exit_lw060_posweight` | 3 | **0.6671** | 0.6382 | 0.6628 | 0.2416 | 0.1382 |
| `3exit_lw080_posweight` | 3 | 0.6496 | 0.6344 | 0.6563 | 0.2809 | 0.1295 |
| `5exit_lw060_posweight` | 4 | 0.6508 | **0.6485** | **0.6732** | **0.3174** | **0.1197** |
| `5exit_lw080_posweight` | 5 | 0.6519 | 0.6443 | 0.6680 | 0.3034 | 0.1278 |

---

## Did early exits improve?

### Exit 1 remains weak

| Model | Exit 1 Macro-F1 | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|
| `3exit_posweight` baseline | 0.4268 | 0.3508 | 4.3230 |
| `3exit_lw060_posweight` | 0.4220 | 0.3140 | 3.6629 |
| `3exit_lw080_posweight` | 0.4195 | 0.3025 | 3.4691 |
| `5exit_posweight` baseline | 0.4116 | 0.3191 | 3.7191 |
| `5exit_lw060_posweight` | 0.4148 | 0.3379 | 4.0028 |
| `5exit_lw080_posweight` | 0.4146 | 0.3312 | 3.8848 |

### Exit 2 improves clearly

| Model | Exit 2 Macro-F1 | Change vs baseline | Hamming Loss | Change vs baseline |
|---|---:|---:|---:|---:|
| `3exit_posweight` baseline | 0.5727 | - | 0.1705 | - |
| `3exit_lw060_posweight` | 0.5884 | +0.0157 | 0.1756 | +0.0051 |
| `3exit_lw080_posweight` | **0.5909** | **+0.0182** | **0.1621** | **-0.0084** |
| `5exit_posweight` baseline | 0.4777 | - | 0.2455 | - |
| `5exit_lw060_posweight` | 0.5030 | +0.0253 | 0.2312 | -0.0143 |
| `5exit_lw080_posweight` | **0.5067** | **+0.0290** | **0.2146** | **-0.0309** |

---

## Dynamic policy results

### Policy 001: `min_exit=2, stable_k=2`

| Model | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Exit Depth | Compute Saved |
|---|---:|---:|---:|---:|---:|---:|
| `3exit_lw060_posweight` | **0.6671** | 0.6628 | 0.2416 | 0.1382 | 3.0000 / 3 | 0.00% |
| `3exit_lw080_posweight` | 0.6496 | 0.6563 | 0.2809 | 0.1295 | 3.0000 / 3 | 0.00% |
| `5exit_lw060_posweight` | 0.6251 | 0.6479 | 0.2837 | 0.1396 | 4.2219 / 5 | 15.56% |
| `5exit_lw080_posweight` | **0.6418** | **0.6631** | **0.3062** | **0.1315** | 4.2275 / 5 | 15.45% |

### Policy 002: `min_exit=1, stable_k=2`

| Model | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Exit Depth | Compute Saved |
|---|---:|---:|---:|---:|---:|---:|
| `3exit_lw060_posweight` | **0.6579** | 0.6616 | 0.2388 | 0.1503 | 2.8933 / 3 | 3.56% |
| `3exit_lw080_posweight` | 0.6475 | 0.6579 | **0.2781** | **0.1362** | 2.8961 / 3 | 3.46% |
| `5exit_lw060_posweight` | 0.6225 | 0.6478 | 0.2809 | 0.1480 | 3.9522 / 5 | **20.96%** |
| `5exit_lw080_posweight` | **0.6305** | **0.6579** | **0.2949** | **0.1427** | 3.9551 / 5 | 20.90% |

### Best 5-exit target: `min_exit=3, stable_k=2`

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Exit Depth | Compute Saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5exit_lw060_posweight` | 0.6345 | 0.6277 | 0.6543 | 0.2949 | 0.1323 | 4.5815 / 5 | 8.37% |
| `5exit_lw080_posweight` | **0.6504** | **0.6443** | **0.6693** | **0.3174** | **0.1253** | 4.5983 / 5 | 8.03% |

Compared with previous best 5-exit greedy-EE baseline:

| Model | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss | Compute Saved |
|---|---:|---:|---:|---:|---:|
| Previous `5exit_posweight`, `min_exit=3, stable_k=2` | 0.6449 | 0.6690 | 0.2921 | 0.1346 | **9.33%** |
| New `5exit_lw080_posweight`, `min_exit=3, stable_k=2` | **0.6504** | **0.6693** | **0.3174** | **0.1253** | 8.03% |

---

## Main findings

1. Loss weighting helps, mainly from Exit 2 onward.
2. Exit 1 remains weak and should not be used for immediate stopping.
3. Exit 2 improves clearly in both 3-exit and 5-exit models.
4. The compact 3-exit model benefits under Policy 002.
5. The best practical 5-exit result is `5exit_lw080_posweight` with `min_exit=3, stable_k=2`.
6. The branch improves the quality-focused trade-off, but does not fully solve earliest-exit reliability.

---

## Recommended branch conclusion

> The early-exit loss-weighting experiment shows that stronger supervision of intermediate exits improves the multi-label early-exit trade-off. Exit 2 improves consistently across both 3-exit and 5-exit models. For the compact 3-exit model, enabling Exit 1 under a label-set stability policy now achieves higher macro-F1 and slightly greater estimated compute saving than the previous greedy-EE baseline. However, Exit 1 remains weak and over-predictive, indicating that loss weighting alone is insufficient to make the earliest head reliable. The strongest practical result is obtained by the 5-exit loss-weighted model with `min_exit=3, stable_k=2`, which achieves macro-F1 `0.6504` with `8.03%` estimated depth-compute saving.

---

## Suggested next step

The next controlled research step should be one of:

1. sigmoid-aware hint passing;
2. explicit exit-to-exit consistency or distillation loss;
3. a milder positive-weight cap such as `5.0` or `10.0` under the same loss-weight structure;
4. FLOPs/latency profiling to support the depth-unit saving estimate.

