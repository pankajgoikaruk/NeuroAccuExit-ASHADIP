# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing v0.7

This README documents the active **`agentic_data_preprocessing_v0.7`** branch. The branch extends the earlier TinyAudioTriageAgent work from a 12-label weakclip baseline into a complete **TATA-assisted human-in-the-loop raw-data preprocessing pipeline**.

```text
Branch: agentic_data_preprocessing_v0.7
Agenda: TATA-assisted pseudo-manifest generation for human-talk multi-label audio
Core idea: train TATA on reviewed seed data, route raw clips, correct uncertain clips, and train main multi-label models
Label schema: 10 labels = 6 target speakers + other_speaker_present + music_present + audience_reaction_present + silence_present
Audience label design: applause/laughter/crowd_cheer are merged into audience_reaction_present
Best v0.6 broad model: main_v06_expanded_3exit + fixed threshold 0.5 + parent-level mean aggregation
Best v0.6 broad raw holdout: Macro-F1 0.7598, Micro-F1 0.8976, Samples-F1 0.9048, Exact Match 0.8155, Hamming Loss 0.0271
Best v0.7 filtered model: main_v07_filtered_3exit + fixed threshold 0.5 + parent-level mean aggregation
Best v0.7 filtered holdout: Macro-F1 0.7446, Micro-F1 0.8983, Samples-F1 0.9041, Exact Match 0.7596, Hamming Loss 0.0317
```

## Branch agenda

The v0.6 branch tests whether a compact multi-label audio triage model can act as an **agentic preprocessing assistant**. TATA is trained on a manually reviewed seed dataset, then used to predict and route a larger raw dataset. High-confidence clips become pseudo-labelled training data, while uncertain clips are sent to a human reviewer. The corrected manifest is then used to train main 3-exit and 5-exit multi-label models.

The final v0.6 workflow is:

```text
Reviewed seed manifest
  -> train TATA v0.6
  -> split raw dataset into pseudo-pool and final raw holdout
  -> run TATA on raw pseudo-pool only
  -> hybrid routing: accepted / accepted_with_warning / needs_review / rejected
  -> manually correct needs_review
  -> combine seed + accepted + warning + corrected needs_review
  -> train main 3-exit and 5-exit models
  -> manually label final raw holdout
  -> evaluate segment-level and parent/clip-level generalisation
```


## Agentic Data Preprocessing v0.7 filtered ablation

The v0.7 branch is a controlled ablation built from `agentic_data_preprocessing_v0.6`. It removes five non-target source-speaker folders from the raw pseudo/corrected training manifests and from the final raw holdout evaluation set:

```text
Les_Brown
Mel_Robbins
Oprah_Winfrey
Rabin_Sharma
Simon_Sinek
```

The original v0.6 data and audio files were not deleted. v0.7 filters rows from CSV manifests only, creating a cleaner six-target-speaker experiment.

### v0.7 filtered dataset summary

| Item | Count / description |
|---|---:|
| Final filtered holdout parent clips | 441 |
| Final filtered holdout 1-second segments | 2,205 |
| v0.7 final expanded training segments | 24,619 |
| Seed reviewed segments retained | 12,469 |
| Filtered raw expanded training segments | 12,150 |
| Removed source folders | Les/Mel/Oprah/Rabin/Simon |

### v0.7 internal training result

| Model | Best epoch | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|---:|
| v0.7 filtered 3-exit, internal test | 34 | 0.8296 | 0.8343 | 0.8347 | 0.6670 | 0.0453 |

### v0.7 final filtered holdout result

Evaluation setting:

```text
Model: main_v07_filtered_3exit
Evaluation: parent/clip-level mean aggregation
Holdout: filtered target-focused raw holdout, 441 clips
```

| Setting | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| v0.7 fixed 0.5 | 0.7446 | 0.8983 | 0.9041 | 0.7596 | 0.0317 | 1.4467 |
| v0.7 tuned per-exit | 0.7468 | 0.8953 | 0.9013 | 0.7347 | 0.0345 | 1.6190 |

Fixed threshold 0.5 remains the best practical v0.7 configuration because tuned thresholds provide only a tiny Macro-F1 gain but reduce Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

### Fair comparison: v0.6 vs v0.7 on the same filtered holdout

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v0.6 main 3-exit on v0.7 filtered holdout | 0.7308 | 0.9001 | 0.8986 | 0.7483 | 0.0320 |
| v0.7 filtered 3-exit on v0.7 filtered holdout | 0.7446 | 0.8983 | 0.9041 | 0.7596 | 0.0317 |

Interpretation: v0.7 is slightly better overall on the filtered target-focused holdout, improving Macro-F1, Samples-F1, Exact Match, and Hamming Loss. The Micro-F1 difference is negligible and slightly favours v0.6.

### v0.7 per-label observation

The six target speakers are very strong under v0.7 fixed-threshold parent-level mean evaluation:

| Label | F1 |
|---|---:|
| Brene_Brown | 0.9793 |
| Eckhart_Tolle | 0.9940 |
| Eric_Thomas | 0.9134 |
| Gary_Vee | 0.9926 |
| Jay_Shetty | 0.9836 |
| Nick_Vujicic | 0.9231 |

Weak labels remain:

| Label | F1 | Main issue |
|---|---:|---|
| other_speaker_present | 0.3200 fixed / 0.5133 tuned | Filtering removes many non-target examples, creating a precision/recall trade-off. |
| audience_reaction_present | 0.4500 fixed / 0.2759 tuned | Low support and ambiguous audience/music/speech mixtures. |
| silence_present | 0.0000 | Very low support; no positives predicted in the filtered holdout. |

### v0.7 research conclusion

The v0.7 filtered experiment confirms that removing five noisy non-target source-speaker folders creates a cleaner target-speaker-focused setting. It slightly improves target-focused holdout behaviour compared with v0.6 on the same filtered holdout, but it does not solve weak background/event labels. The next strategy should therefore focus on targeted weak-label repair and augmentation rather than additional source filtering alone.


## Label schema

The v0.6 label schema is deliberately coarser than the earlier 12-label schema. The previous `applause_present`, `laughter_present`, and `crowd_cheer_present` labels were difficult to separate consistently, even manually. They were merged into one robust label:

```text
audience_reaction_present
```

Active labels:

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
audience_reaction_present
silence_present
```

Non-target speakers such as `Les_Brown`, `Mel_Robbins`, `Oprah_Winfrey`, `Rabin_Sharma`, and `Simon_Sinek` are not separate output classes in v0.6. They are mapped to `other_speaker_present = 1`.

## Research questions

| ID | Research question | v0.6 answer |
|---|---|---|
| RQ1 | Can a 10-label coarse audience schema reduce ambiguity from applause/laughter/cheer overlap? | Yes. The merged `audience_reaction_present` label made the pipeline more stable than the earlier fine-grained 12-label audience schema. |
| RQ2 | Can TATA route a raw dataset into useful pseudo-label groups? | Yes. Hybrid routing produced 369 accepted, 925 accepted-with-warning, 3,171 needs-review, and 445 rejected raw parent clips. |
| RQ3 | Does human-in-the-loop correction protect the final manifest from uncertain pseudo labels? | Yes. The highest-risk raw clips were routed to `needs_review` and corrected before inclusion in training. |
| RQ4 | Does the final expanded manifest train a main model that generalises to manually labelled raw holdout clips? | Yes. The best parent-level mean result achieved Micro-F1 0.8976, Samples-F1 0.9048, Exact Match 0.8155, and Hamming Loss 0.0271. |
| RQ5 | Does threshold tuning improve final holdout reliability? | Not overall. Tuned thresholds slightly improved Macro-F1 in some settings, but fixed 0.5 gave the best exact match and hamming loss for the recommended 3-exit model. |
| RQ6 | Is parent-level max or mean aggregation better for clip-level evaluation? | Mean aggregation is clearly better. Max aggregation over-predicts labels and increases false positives. |
| RQ7 | Does a 5-exit model provide a useful early-exit trade-off? | Yes as an ablation. 5-exit dynamic policy saved 28.60% compute under parent-level mean aggregation, but quality dropped too much to be the final model. |

## TATA v0.6 seed-model validation

TATA was first validated on the reviewed seed data before being used for raw pseudo-routing.

| model                  |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |
|:-----------------------|-----------:|-----------:|-------------:|--------------:|---------------:|
| TATA v0.6 3-exit fixed |     0.8164 |     0.8272 |       0.8264 |        0.6594 |         0.0474 |
| TATA v0.6 3-exit tuned |     0.8291 |     0.8121 |       0.8075 |        0.5977 |         0.0543 |

Interpretation: TATA v0.6 is a strong triage model. Fixed threshold is better for conservative pseudo-label reliability, while tuned threshold gives the best seed Macro-F1.

## Raw pseudo-routing outcome

TATA was applied to the raw pseudo-pool only, not to the final holdout split.

| mode      |   accepted |   accepted_with_warning |   needs_review |   rejected |   accepted_plus_warning |
|:----------|-----------:|------------------------:|---------------:|-----------:|------------------------:|
| fixed_0p5 |        537 |                    1189 |           2711 |        473 |                    1726 |
| hybrid    |        369 |                     925 |           3171 |        445 |                    1294 |

Hybrid routing was selected as the safer pseudo-label route because it sends more uncertain clips to human review.

## Final expanded training dataset

The main-model training manifest combines:

```text
reviewed seed dataset
+ raw hybrid accepted pseudo labels
+ raw hybrid accepted_with_warning pseudo labels
+ corrected raw hybrid needs_review labels
```

The final raw holdout split remains separate and is used only for external evaluation.

| Component | Rows / clips |
|---|---:|
| Reviewed seed segments | 12,469 |
| Raw pseudo/corrected parent clips used for training | 4,465 |
| Raw pseudo/corrected segments used for training | 22,325 |
| Final combined segment manifest | 34,794 |
| Internal train rows | 30,950 |
| Internal validation rows | 1,883 |
| Internal test rows | 1,961 |
| Final raw holdout parent clips | 867 |
| Final raw holdout segments | 4,335 |

## Internal main-model comparison

These are internal validation/test-style results from the expanded training manifest, not the final raw holdout.

| model            |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   compute_saved |
|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|
| 3-exit fixed 0.5 |     0.8225 |     0.8219 |       0.824  |        0.6221 |         0.0502 |            0    |
| 3-exit tuned     |     0.8217 |     0.8154 |       0.823  |        0.5875 |         0.0542 |            0    |
| 3-exit dynamic   |     0.8217 |     0.8154 |       0.823  |        0.5875 |         0.0542 |            0    |
| 5-exit fixed 0.5 |     0.812  |     0.7999 |       0.7892 |        0.5767 |         0.0562 |            0    |
| 5-exit tuned     |     0.8217 |     0.8079 |       0.8101 |        0.5752 |         0.0562 |            0    |
| 5-exit dynamic   |     0.7764 |     0.761  |       0.7624 |        0.4656 |         0.0727 |           19.75 |

The 3-exit fixed-threshold model is the best internal reliability configuration. The 5-exit dynamic model saves compute but loses too much quality.

## Final raw holdout: segment-level result

| setting          |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   compute_saved |
|:-----------------|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|
| 3-exit fixed 0.5 |     0.729  |     0.8476 |       0.8507 |        0.7301 |         0.0409 |            0    |
| 3-exit tuned     |     0.726  |     0.8489 |       0.8639 |        0.7165 |         0.0417 |            0    |
| 5-exit tuned     |     0.725  |     0.8396 |       0.8518 |        0.6932 |         0.0441 |            0    |
| 5-exit dynamic   |     0.6766 |     0.7912 |       0.8092 |        0.6028 |         0.0597 |           24.01 |

Segment-level evaluation is useful, but the final manual labels are clip-level labels. Therefore, parent/clip-level evaluation is the main result.

## Final raw holdout: parent-level max aggregation

| setting          |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_pred_labels   |   compute_saved |
|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|----------------:|
| 3-exit fixed 0.5 |     0.7337 |     0.8073 | 0.8427       |        0.5767 |         0.0619 | 1.8720            |            0    |
| 3-exit tuned     |     0.7009 |     0.7858 | 0.8227       |        0.5063 |         0.0712 | 1.9804            |            0    |
| 5-exit tuned     |     0.7275 |     0.7844 | 0.8134       |        0.4787 |         0.0721 | 2.0012            |            0    |
| 5-exit dynamic   |     0.6892 |     0.7451 |              |        0.391  |         0.0893 |                   |           18.69 |

Max aggregation means a label is predicted present if any one-second segment fires. This is sensitive but over-predicts labels.

## Final raw holdout: parent-level mean aggregation

| setting          |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_pred_labels   |   compute_saved |
|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|----------------:|
| 3-exit fixed 0.5 |     0.7598 |     0.8976 | 0.9048       |        0.8155 |         0.0271 | 1.3045            |             0   |
| 3-exit tuned     |     0.7615 |     0.8937 | 0.9115       |        0.7785 |         0.0292 | 1.4014            |             0   |
| 5-exit tuned     |     0.77   |     0.8866 | 0.9032       |        0.7439 |         0.0311 | 1.4025            |             0   |
| 5-exit dynamic   |     0.7186 |     0.8283 |              |        0.6332 |         0.0498 |                   |            28.6 |

Mean aggregation gives the cleanest final result because it smooths noisy one-second predictions and keeps the predicted label cardinality close to the ground truth.

## Final recommended result

```text
Best model: main_v06_expanded_3exit
Threshold: fixed 0.5
Evaluation: parent/clip-level raw holdout
Aggregation: mean over segment probabilities
Macro-F1: 0.7598
Micro-F1: 0.8976
Samples-F1: 0.9048
Exact Match: 0.8155
Hamming Loss: 0.0271
```

## Figures

![Final parent mean comparison](figures/human_talk/agentic_data_preprocessing_v0.6/final_holdout_parent_mean_comparison.png)

![Aggregation comparison](figures/human_talk/agentic_data_preprocessing_v0.6/aggregation_comparison_3exit_fixed.png)

![Hamming loss comparison](figures/human_talk/agentic_data_preprocessing_v0.6/final_holdout_hamming_loss_comparison.png)

![Dynamic policy tradeoff](figures/human_talk/agentic_data_preprocessing_v0.6/dynamic_policy_tradeoff_parent_level.png)

![Raw routing counts](figures/human_talk/agentic_data_preprocessing_v0.6/raw_pseudo_routing_counts.png)

![Per-label F1](figures/human_talk/agentic_data_preprocessing_v0.6/per_label_f1_segment_3exit_fixed.png)

## Theoretical notes

### Multi-label BCE/sigmoid formulation

Each clip can contain multiple simultaneous labels, so v0.6 uses independent sigmoid outputs rather than softmax. For label `c`, the model predicts probability `p_c = sigmoid(z_c)`. Training uses binary cross entropy over the label vector.

### Human-in-the-loop pseudo-manifest generation

The raw data is not blindly pseudo-labelled. TATA produces predictions and the routing layer assigns one decision:

```text
accepted
accepted_with_warning
needs_review
rejected
```

The final training manifest uses high-confidence pseudo rows plus human-corrected uncertain rows. This prevents low-confidence clips from being treated as trusted labels.

### Parent-level aggregation

For a parent clip with segments `s = 1...S`, parent-level probability is computed as either:

```text
max aggregation:  p_c(parent) = max_s p_c(segment_s)
mean aggregation: p_c(parent) = (1/S) * sum_s p_c(segment_s)
```

In v0.6, mean aggregation is preferred because max aggregation produced too many extra labels.

### Dynamic exit policy

The dynamic policy uses label-set stability. It exits when the predicted label set remains unchanged for a required number of exits. Compute saving is estimated by:

```text
compute_saved = (1 - average_exit_depth / final_exit_depth) * 100
```

The 5-exit dynamic policy achieved compute saving, but its predictive quality was below the best static 3-exit model.

## Research conclusion

The v0.6 experiments support the central claim that **TATA-assisted human-in-the-loop preprocessing can generate a high-quality expanded multi-label manifest from raw audio**. The final model trained from this manifest generalised strongly to a manually labelled raw holdout set, especially under parent-level mean aggregation.

Paper-safe statement:

```text
The agentic_data_preprocessing_v0.6 branch demonstrates that TinyAudioTriageAgent can serve as a reliable human-in-the-loop pseudo-manifest generator. By combining reviewed seed labels, conservative TATA routing, and human correction of uncertain clips, the system produced a main model that achieved 0.8976 Micro-F1, 0.9048 Samples-F1, 0.8155 Exact Match, and 0.0271 Hamming Loss on an unseen manually labelled raw holdout set. The best reliable configuration was a 3-exit fixed-threshold model with parent-level mean aggregation. A 5-exit dynamic policy provided compute saving but with reduced accuracy, making it an efficiency/accuracy ablation rather than the final recommended model.
```

## Current limitations

- Some non-target speaker training rows may have incomplete background/event annotations. This is a known limitation and should be revisited if final event-label performance becomes the main focus.
- `audience_reaction_present` and `silence_present` remain difficult low-support/event labels.
- The final holdout is manually labelled at clip level; segment labels are weak inherited labels.
- Dynamic early exit is not yet the best accuracy option.

## Next strategy

1. Commit and push the v0.6 docs, scripts, and final result tables.
2. Keep **3-exit fixed + parent mean** as the final reliable result.
3. Keep **5-exit dynamic mean** as the efficiency/accuracy trade-off baseline.
4. For future improvement, repair or sample-check non-target background labels and add targeted augmentation for `audience_reaction_present` and `silence_present`.


## v0.7 final recommendation

Use v0.6 as the broader realistic setting and v0.7 as the cleaner target-focused ablation:

| Version | Role | Best result |
|---|---|---|
| v0.6 | Broad raw-world setting with non-target source speakers | Parent mean: Macro-F1 0.7598, Micro-F1 0.8976, Exact Match 0.8155, Hamming Loss 0.0271 |
| v0.7 | Filtered target-focused setting without Les/Mel/Oprah/Rabin/Simon | Parent mean: Macro-F1 0.7446, Micro-F1 0.8983, Exact Match 0.7596, Hamming Loss 0.0317 |

The next research direction is **weak-label improvement** for `other_speaker_present`, `audience_reaction_present`, and `silence_present`.
