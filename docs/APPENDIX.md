# Appendix — `kexit_human_talk_incremental_eval`

This appendix records the reproducibility protocol and extended result summary for the human-talk incremental evaluation branch.

```text
Branch: kexit_human_talk_incremental_eval
Base branch: kexit_multi-label_EE_lossweight
Task: clean human-talk speaker classification
Stage: clean2_balanced
Main idea: test whether the K-exit audio model generalises to human-talk speaker classification and whether additional exits improve dynamic inference flexibility
```

## A1. Current branch status

| Item | Value |
|---|---|
| Active branch | `kexit_human_talk_incremental_eval` |
| Base branch | `kexit_multi-label_EE_lossweight` |
| Stage completed | `clean2_balanced` |
| Classes | `Les_Brown`, `Simon_Sinek` |
| Compared models | `human_talk_clean2_3exit_nohint`, `human_talk_clean2_5exit_nohint` |
| Thresholding | fixed sigmoid threshold `0.5` |
| Policy | greedy label-set stability |
| Main metric | macro-F1 with hamming loss and estimated depth-compute saving |
| Main caveat | renamed-format parser issue in metadata |

## A2. Reproducibility commands

### Stage preparation

```powershell
powershell -ExecutionPolicy Bypass -File .\scriptsun_human_talk_stage_prepare.ps1 `
  -Stage clean2_balanced `
  -RawRoot human_talk_dataset `
  -WorkspaceRoot human_talk_workspace `
  -Clean
```

### Feature extraction

```powershell
python .\scripts\extract_multilabel_features.py `
  --manifest "human_talk_workspace\stages\clean2_balanced\data\metadata\multilabel_train_manifest.csv" `
  --labels_json "human_talk_workspace\stages\clean2_balanced\data\metadata\labels.json" `
  --out_cache "human_talk_workspace\stages\clean2_balanced\cache" `
  --sample_rate 16000 `
  --clip_sec 1.0 `
  --n_mels 64 `
  --n_fft 1024 `
  --win_ms 25 `
  --hop_ms 10 `
  --cmvn
```

### 3-exit training

```powershell
python -m training.train_multilabel `
  --manifest "human_talk_workspace\stages\clean2_balanced\cache\metadata\multilabel_features_manifest.csv" `
  --features_root "human_talk_workspace\stages\clean2_balanced\cacheeatures" `
  --labels_json "human_talk_workspace\stages\clean2_balanced\data\metadata\labels.json" `
  --runs_root "human_talk_workspace\stages\clean2_balanceduns" `
  --variant "human_talk_clean2_3exit_nohint" `
  --tap_blocks "1,3" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --device cpu
```

### 5-exit training

```powershell
python -m training.train_multilabel `
  --manifest "human_talk_workspace\stages\clean2_balanced\cache\metadata\multilabel_features_manifest.csv" `
  --features_root "human_talk_workspace\stages\clean2_balanced\cacheeatures" `
  --labels_json "human_talk_workspace\stages\clean2_balanced\data\metadata\labels.json" `
  --runs_root "human_talk_workspace\stages\clean2_balanceduns" `
  --variant "human_talk_clean2_5exit_nohint" `
  --tap_blocks "1,2,3,4" `
  --epochs 40 `
  --batch_size 64 `
  --lr 0.001 `
  --device cpu
```

### 5-exit greedy-policy evaluation

```powershell
python .\scripts\multilabel_greedy_policy.py `
  --run_dir "human_talk_workspace\stages\clean2_balanceduns\human_talk_clean2_5exit_nohint_20260517_123828" `
  --name "human_talk_clean2_5exit_nohint" `
  --device cpu `
  --split test `
  --threshold_mode fixed_0p5 `
  --min_exit 3 `
  --stable_k 2 `
  --sweep_min_exits "1,2,3" `
  --sweep_stable_k "1,2,3"
```

## A3. Dataset summary

| Split | Parent clips | Segments |
|---|---:|---:|
| Train | 660 | 5,940 |
| Val | 142 | 1,278 |
| Test | 142 | 1,278 |
| Total | 944 | 8,496 |

## A4. Static per-exit results

### 3-exit model

| Exit | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8765 | 0.8732 | 0.8701 | 0.1232 |
| 2 | 0.9765 | 0.9765 | 0.9765 | 0.0235 |
| 3 | 0.9926 | 0.9922 | 0.9922 | 0.0074 |

### 5-exit model

| Exit | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8799 | 0.8785 | 0.8748 | 0.1201 |
| 2 | 0.9259 | 0.9244 | 0.9233 | 0.0739 |
| 3 | 0.9703 | 0.9700 | 0.9695 | 0.0297 |
| 4 | 0.9883 | 0.9883 | 0.9883 | 0.0117 |
| 5 | 0.9930 | 0.9930 | 0.9930 | 0.0070 |

## A5. Dynamic policy results

| Model | Selected policy | Macro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Exit Depth | Compute Saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3exit_nohint` | `min_exit=2, stable_k=2` | 0.9926 | 0.9922 | 0.9922 | 0.0074 | 3.0000 / 3 | 0.00% |
| `5exit_nohint` | `min_exit=3, stable_k=2` | 0.9883 | 0.9883 | 0.9883 | 0.0117 | 4.0219 / 5 | 19.56% |

## A6. Policy sweep highlights

| Model | Policy | Macro-F1 | Samples-F1 | Hamming Loss | Avg Exit Depth | Compute Saved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3exit_nohint` | `min_exit=1, stable_k=2` | 0.9836 | 0.9836 | 0.0164 | 2.1252 / 3 | 29.16% |
| `3exit_nohint` | `min_exit=2, stable_k=1` | 0.9765 | 0.9765 | 0.0235 | 2.0000 / 3 | 33.33% |
| `3exit_nohint` | `min_exit=2, stable_k=2` | 0.9926 | 0.9922 | 0.0074 | 3.0000 / 3 | 0.00% |
| `5exit_nohint` | `min_exit=1, stable_k=3` | 0.9765 | 0.9765 | 0.0235 | 3.1854 / 5 | 36.29% |
| `5exit_nohint` | `min_exit=2, stable_k=3` | 0.9898 | 0.9898 | 0.0102 | 4.0806 / 5 | 18.39% |
| `5exit_nohint` | `min_exit=3, stable_k=2` | 0.9883 | 0.9883 | 0.0117 | 4.0219 / 5 | 19.56% |
| `5exit_nohint` | `min_exit=3, stable_k=3` | 0.9930 | 0.9930 | 0.0070 | 5.0000 / 5 | 0.00% |

## A7. Main appendix conclusion for human-talk Stage 1

The human-talk Stage 1 branch is a successful controlled generalisation check. It shows that the K-exit audio model can separate two clean speakers with near-perfect final-exit performance. More importantly, the 5-exit model provides a useful dynamic inference option: macro-F1 `0.9883` with `19.56%` estimated depth-compute saving under `min_exit=3, stable_k=2`. The strongest accuracy/saving policy from the sweep is `min_exit=2, stable_k=3`, with macro-F1 `0.9898` and `18.39%` saving. Exit 1 remains unreliable, and the renamed-format parser should be fixed before Stage 2.

---

# Previous loss-weight appendix retained below

The previous appendix is retained below so the original result tables and reproducibility record are not lost.

---

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

