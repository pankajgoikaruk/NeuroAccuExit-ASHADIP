# Experiment Log — agentic_data_preprocessing_v0.6

This log records the active **`agentic_data_preprocessing_v0.6`** branch after completion of TATA routing, main-model training, and final raw holdout evaluation.

```text
Branch: agentic_data_preprocessing_v0.6
Core task: multi-label human-talk audio triage and manifest generation
Final recommended model: main_v06_expanded_3exit + fixed threshold 0.5 + parent-level mean aggregation
Final raw holdout: 867 parent clips / 4,335 one-second segments
```

## Completed chronology

| Step | Output | Status |
|---|---|---|
| Create v0.6 branch | `agentic_data_preprocessing_v0.6` | Completed |
| Move from 12 labels to 10 labels | merged applause/laughter/crowd_cheer into `audience_reaction_present` | Completed |
| Train TATA v0.6 on reviewed seed data | 3-exit TATA scratch run | Completed |
| Validate TATA fixed/tuned/policy | TATA v0.6 3-exit fixed/tuned results | Completed |
| Split raw dataset | raw pseudo-pool + final raw holdout | Completed |
| Build raw pseudo-pool segments/features | 4,910 parent clips / 24,550 segments | Completed |
| Run TATA on raw pseudo-pool | fixed and hybrid routing CSVs | Completed |
| Choose routing policy | hybrid selected as safer | Completed |
| Manual correction of raw `needs_review` | 3,171 rows corrected | Completed |
| Non-target speaker handling | Les/Mel/Oprah/Rabin/Simon mapped to `other_speaker_present` | Completed |
| Build final expanded training manifest | 34,794 segment rows | Completed |
| Train main 3-exit model | `main_v06_expanded_3exit_20260603_194435` | Completed |
| Train main 5-exit model | `main_v06_expanded_5exit_20260603_210324` | Completed |
| Threshold tuning | 3-exit and 5-exit | Completed |
| Dynamic policy evaluation | 3-exit and 5-exit | Completed |
| Manual final raw holdout labels | 867 clips | Completed |
| Segment-level holdout evaluation | 4,335 segments | Completed |
| Parent-level max evaluation | 867 clips | Completed |
| Parent-level mean evaluation | 867 clips | Completed |

## TATA seed validation

| model                  |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |
|:-----------------------|-----------:|-----------:|-------------:|--------------:|---------------:|
| TATA v0.6 3-exit fixed |     0.8164 |     0.8272 |       0.8264 |        0.6594 |         0.0474 |
| TATA v0.6 3-exit tuned |     0.8291 |     0.8121 |       0.8075 |        0.5977 |         0.0543 |

## Raw routing counts

| mode      |   accepted |   accepted_with_warning |   needs_review |   rejected |   accepted_plus_warning |
|:----------|-----------:|------------------------:|---------------:|-----------:|------------------------:|
| fixed_0p5 |        537 |                    1189 |           2711 |        473 |                    1726 |
| hybrid    |        369 |                     925 |           3171 |        445 |                    1294 |

## Internal main-model comparison

| model            |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   compute_saved |
|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|
| 3-exit fixed 0.5 |     0.8225 |     0.8219 |       0.824  |        0.6221 |         0.0502 |            0    |
| 3-exit tuned     |     0.8217 |     0.8154 |       0.823  |        0.5875 |         0.0542 |            0    |
| 3-exit dynamic   |     0.8217 |     0.8154 |       0.823  |        0.5875 |         0.0542 |            0    |
| 5-exit fixed 0.5 |     0.812  |     0.7999 |       0.7892 |        0.5767 |         0.0562 |            0    |
| 5-exit tuned     |     0.8217 |     0.8079 |       0.8101 |        0.5752 |         0.0562 |            0    |
| 5-exit dynamic   |     0.7764 |     0.761  |       0.7624 |        0.4656 |         0.0727 |           19.75 |

## Final raw holdout results

### Segment-level

| setting          |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   compute_saved |
|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|
| 3-exit fixed 0.5 |     0.729  |     0.8476 |       0.8507 |        0.7301 |         0.0409 |            0    |
| 3-exit tuned     |     0.726  |     0.8489 |       0.8639 |        0.7165 |         0.0417 |            0    |
| 5-exit tuned     |     0.725  |     0.8396 |       0.8518 |        0.6932 |         0.0441 |            0    |
| 5-exit dynamic   |     0.6766 |     0.7912 |       0.8092 |        0.6028 |         0.0597 |           24.01 |

### Parent-level max aggregation

| setting          |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_pred_labels   |   compute_saved |
|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|----------------:|
| 3-exit fixed 0.5 |     0.7337 |     0.8073 | 0.8427       |        0.5767 |         0.0619 | 1.8720            |            0    |
| 3-exit tuned     |     0.7009 |     0.7858 | 0.8227       |        0.5063 |         0.0712 | 1.9804            |            0    |
| 5-exit tuned     |     0.7275 |     0.7844 | 0.8134       |        0.4787 |         0.0721 | 2.0012            |            0    |
| 5-exit dynamic   |     0.6892 |     0.7451 |              |        0.391  |         0.0893 |                   |           18.69 |

### Parent-level mean aggregation

| setting          |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_pred_labels   |   compute_saved |
|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|----------------:|
| 3-exit fixed 0.5 |     0.7598 |     0.8976 | 0.9048       |        0.8155 |         0.0271 | 1.3045            |             0   |
| 3-exit tuned     |     0.7615 |     0.8937 | 0.9115       |        0.7785 |         0.0292 | 1.4014            |             0   |
| 5-exit tuned     |     0.77   |     0.8866 | 0.9032       |        0.7439 |         0.0311 | 1.4025            |             0   |
| 5-exit dynamic   |     0.7186 |     0.8283 |              |        0.6332 |         0.0498 |                   |            28.6 |

## Aggregation analysis

| evaluation    |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_pred_labels |
|:--------------|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|
| Segment-level |     0.729  |     0.8476 |       0.8507 |        0.7301 |         0.0409 |            1.3409 |
| Parent max    |     0.7337 |     0.8073 |       0.8427 |        0.5767 |         0.0619 |            1.872  |
| Parent mean   |     0.7598 |     0.8976 |       0.9048 |        0.8155 |         0.0271 |            1.3045 |

Mean aggregation is the best final evaluation strategy because it reduces parent-level false positives.

## Dynamic policy analysis

| Setting | Macro-F1 | Micro-F1 | Exact Match | Hamming Loss | Compute Saved |
|---|---:|---:|---:|---:|---:|
| 5-exit dynamic, segment-level | 0.6766 | 0.7912 | 0.6028 | 0.0597 | 24.01% |
| 5-exit dynamic, parent max | 0.6892 | 0.7451 | 0.3910 | 0.0893 | 18.69% |
| 5-exit dynamic, parent mean | 0.7186 | 0.8283 | 0.6332 | 0.0498 | 28.60% |

Dynamic exits work as an efficiency ablation, but they are not the final recommended configuration because the quality drop is still substantial.

## Per-label final holdout observation

The 3-exit fixed model performs strongly on major speaker/background labels at segment level, but low-support labels remain weak.

| label                     |     f1 |   support |   predicted_positive |
|:--------------------------|-------:|----------:|---------------------:|
| Brene_Brown               | 0.865  |       365 |                  287 |
| Eckhart_Tolle             | 0.9782 |       420 |                  406 |
| Eric_Thomas               | 0.7002 |       340 |                  514 |
| Gary_Vee                  | 0.9184 |       340 |                  346 |
| Jay_Shetty                | 0.9436 |       450 |                  455 |
| Nick_Vujicic              | 0.8243 |       245 |                  199 |
| other_speaker_present     | 0.8722 |      2300 |                 2222 |
| music_present             | 0.82   |      1185 |                 1298 |
| audience_reaction_present | 0.2784 |       120 |                   74 |
| silence_present           | 0.0896 |        55 |                   12 |

## Key research findings

1. **TATA routing is useful**: high-confidence pseudo labels plus human-corrected `needs_review` rows produced a training manifest that generalised strongly to raw holdout data.
2. **Hybrid routing is safer than fixed routing**: it sends more uncertain examples to human review and reduces the risk of blindly trusting ambiguous raw clips.
3. **Parent-level mean aggregation is essential**: it produced the best final reliability metrics by smoothing noisy 1-sec segment predictions.
4. **3-exit fixed is the final reliable model**: it outperforms 5-exit in exact match and hamming loss under parent-level mean evaluation.
5. **5-exit dynamic provides compute saving**: it saved up to 28.60% compute under parent-level mean aggregation, but with reduced accuracy.
6. **Fine-grained event labels remain difficult**: `audience_reaction_present` and `silence_present` require more targeted data or better temporal annotation.

## Final paper-safe conclusion

```text
The v0.6 experiments show that TinyAudioTriageAgent can act as a human-in-the-loop pseudo-manifest generator for raw human-talk audio. The final expanded manifest, combining reviewed seed labels, TATA high-confidence pseudo labels, and human-corrected uncertain clips, trained a main 3-exit model that achieved 0.7598 Macro-F1, 0.8976 Micro-F1, 0.9048 Samples-F1, 0.8155 Exact Match, and 0.0271 Hamming Loss on a manually labelled raw holdout set using parent-level mean aggregation. The 5-exit dynamic policy offers compute saving but remains an efficiency/accuracy trade-off rather than the best final model.
```

## Known limitations to carry forward

- The final training manifest may still contain incomplete background/event labels for some non-target speaker clips.
- Segment-level labels are weak inherited labels from parent clips.
- Mean aggregation works best now, but it should be compared with calibrated pooling in future work.
- Dynamic early exit needs stronger intermediate exits or improved policy design before becoming the final deployment mode.


# Agentic Data Preprocessing v0.7 — Filtered Target-Speaker Ablation

## Motivation

v0.6 included five non-target source-speaker folders as `other_speaker_present`. During analysis, we identified that some non-target rows may have incomplete background/event labels. v0.7 therefore removes these source folders from the CSV manifests to test a cleaner target-speaker-focused setting.

Removed source classes:

```text
Les_Brown
Mel_Robbins
Oprah_Winfrey
Rabin_Sharma
Simon_Sinek
```

## v0.7 filtered dataset

| Item | Count |
|---|---:|
| Final filtered holdout parent clips | 441 |
| Final filtered holdout segments | 2,205 |
| Final combined training segments | 24,619 |
| Seed reviewed segments | 12,469 |
| Filtered raw expanded training segments | 12,150 |

## v0.7 internal test result

| Model | Best epoch | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| main_v07_filtered_3exit | 34 | 0.8296 | 0.8343 | 0.8347 | 0.6670 | 0.0453 |

## v0.7 filtered holdout: fixed vs tuned

| Setting | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0.7446 | 0.8983 | 0.9041 | 0.7596 | 0.0317 | 1.4467 |
| Tuned per-exit | 0.7468 | 0.8953 | 0.9013 | 0.7347 | 0.0345 | 1.6190 |

Decision: fixed threshold 0.5 remains the best practical v0.7 setting because tuned thresholds slightly increase Macro-F1 but reduce reliability metrics.

## Fair comparison on the same filtered holdout

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v0.6 model on v0.7 filtered holdout | 0.7308 | 0.9001 | 0.8986 | 0.7483 | 0.0320 |
| v0.7 filtered model on v0.7 filtered holdout | 0.7446 | 0.8983 | 0.9041 | 0.7596 | 0.0317 |

Interpretation: v0.7 improves Macro-F1, Samples-F1, Exact Match, and Hamming Loss on the filtered holdout. Micro-F1 is almost unchanged and slightly favours the v0.6 model. Overall, v0.7 is a useful target-speaker-focused ablation but not a dramatic improvement.

## Per-label finding

Target-speaker labels are strong under v0.7 fixed threshold:

| Label | F1 |
|---|---:|
| Brene_Brown | 0.9793 |
| Eckhart_Tolle | 0.9940 |
| Eric_Thomas | 0.9134 |
| Gary_Vee | 0.9926 |
| Jay_Shetty | 0.9836 |
| Nick_Vujicic | 0.9231 |

Weak labels:

| Label | Fixed F1 | Tuned F1 | Finding |
|---|---:|---:|---|
| other_speaker_present | 0.3200 | 0.5133 | Filtering removed many non-target examples; tuned threshold recovers recall but hurts exact match. |
| audience_reaction_present | 0.4500 | 0.2759 | Low support and ambiguous acoustic mixtures remain. |
| silence_present | 0.0000 | 0.0000 | Needs targeted silence examples and threshold analysis. |

## v0.7 conclusion

The v0.7 filtered experiment confirms that removing the five non-target source-speaker folders creates a cleaner target-speaker-focused setting. However, it does not solve weak background/event labels. The next branch should focus on weak-label repair for `other_speaker_present`, `audience_reaction_present`, and `silence_present`.
