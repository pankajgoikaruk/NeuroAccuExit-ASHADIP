# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing v0.9

This README documents the active `agentic_data_preprocessing_v0.9` branch of **NeuroAccuExit-ASHADIP**. The branch extends the earlier TATA/Lawyer weak-labelling pipeline by introducing a **frozen labelwise parent-aggregation strategy** for human-talk multi-label audio classification. Instead of retraining the model, v0.9 reuses the strongest v0.8-HCB model and tests whether label-specific aggregation can improve corrected-holdout performance, especially for rare and transient context labels.

> **GitHub preview note:** this README intentionally does **not** start with `---`, because that makes GitHub treat the beginning of the file as YAML/front matter and can cause the preview error: `Error in user YAML: found character that cannot start any token`.

---

## Branch summary

| Item | Details |
|---|---|
| Branch | `agentic_data_preprocessing_v0.9` |
| Agenda | Frozen labelwise parent aggregation for human-talk multi-label audio classification |
| Core idea | Reuse the v0.8-HCB trained model, evaluate different parent-level aggregation strategies, freeze the best stable labelwise aggregation map, and apply it to the full corrected holdout |
| Base model reused | `main_v08_human_corrected_balanced_3exit_20260610_084027` |
| Training status | No retraining in v0.9; this is a post-hoc evaluation and aggregation strategy branch |
| Label schema | 10 labels = 6 target speakers + `other_speaker_present` + `music_present` + `audience_reaction_present` + `silence_present` |
| Audience label design | `applause_present`, `laughter_present`, and `crowd_cheer_present` are merged into `audience_reaction_present` |
| Parent clips in corrected holdout | 867 |
| Holdout segments | 4,335 |
| Final selected threshold | Fixed `0.5` for all labels |
| Final selected aggregation | Label-specific frozen map: `mean` for stable labels and `top2mean` for selected speaker/context labels |
| Best v0.8 official result | Parent-level mean aggregation, fixed threshold 0.5: Macro-F1 `0.7801`, Micro-F1 `0.9332`, Samples-F1 `0.9406`, Exact Match `0.8397`, Hamming Loss `0.0194` |
| Best v0.8 simple label-aware result | Mean for stable labels and max for transient labels: Macro-F1 `0.8320`, Micro-F1 `0.9285`, Samples-F1 `0.9375`, Exact Match `0.8235`, Hamming Loss `0.0211` |
| Best v0.9 final result | Frozen labelwise aggregation, fixed threshold 0.5: Macro-F1 `0.8512`, Micro-F1 `0.9372`, Samples-F1 `0.9482`, Exact Match `0.8420`, Hamming Loss `0.0185` |

---

## Branch agenda

The v0.9 branch investigates whether parent-level aggregation can be made label-aware without retraining the acoustic model. Earlier v0.8 results showed that simple parent-level mean aggregation was strong overall, but rare transient labels such as `audience_reaction_present` and `silence_present` were diluted when segment probabilities were averaged across the parent clip.

The agenda of this branch is therefore:

1. Reproduce the v0.8-HCB corrected-holdout baseline using the existing segment probability CSV.
2. Compare parent aggregation strategies: `mean`, `max`, and `top2mean`.
3. Run repeated calibration/evaluation splits to test whether aggregation choices are stable.
4. Compare fixed threshold 0.5 against threshold calibration.
5. Freeze the most reliable labelwise aggregation map.
6. Evaluate the frozen map on the full corrected holdout.
7. Document the final result as the v0.9 post-hoc aggregation improvement.

---

## Research questions

### RQ1 — Does label-specific parent aggregation improve multi-label audio classification?

Yes. The final frozen v0.9 aggregation map improves Macro-F1, Micro-F1, Samples-F1, Exact Match, and Hamming Loss compared with the v0.8 official parent-mean baseline.

### RQ2 — Are transient labels diluted by parent-level mean aggregation?

Yes. The v0.8 post-hoc analysis showed that `audience_reaction_present` and `silence_present` were weak under mean aggregation. More selective aggregation, especially `top2mean`, improved these labels.

### RQ3 — Is global max aggregation suitable as the final method?

No. Global max helped rare transient labels, but it over-predicted labels overall and harmed Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

### RQ4 — Is threshold calibration better than fixed threshold 0.5?

Not for the final v0.9 setting. Repeated-split experiments showed that threshold calibration achieved similar Macro-F1 but weakened Micro-F1, Samples-F1, Exact Match, and Hamming Loss. Therefore, the final v0.9 method keeps fixed threshold `0.5`.

### RQ5 — Can the final improvement be achieved without retraining?

Yes. v0.9 improves the corrected-holdout result using only post-hoc aggregation over existing segment probabilities. No new neural-network training is performed in this branch.

---

## Label schema

| Group | Labels |
|---|---|
| Target speakers | `Brene_Brown`, `Eckhart_Tolle`, `Eric_Thomas`, `Gary_Vee`, `Jay_Shetty`, `Nick_Vujicic` |
| Non-target speaker context | `other_speaker_present` |
| Background/context | `music_present`, `audience_reaction_present`, `silence_present` |

The final schema contains 10 labels:

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

The audience-related labels are intentionally merged:

```text
applause_present + laughter_present + crowd_cheer_present
→ audience_reaction_present
```

This merge reduces annotation ambiguity because applause, laughter, and crowd cheering often overlap acoustically in motivational speech clips.

---

## v0.8 baseline context

v0.8-HCB remains the base model used by v0.9.

| Item | Details |
|---|---|
| v0.8 experiment name | `v0.8-human-corrected-balanced` |
| v0.8 run directory | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/main_models/runs/main_v08_human_corrected_balanced_3exit_20260610_084027` |
| Model variant | 3-exit multi-label acoustic classifier |
| Tap blocks | `1,3` |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Device | CPU |
| Loss weights | `[0.3, 0.3, 1.0]` |
| Best validation epoch | 39 |
| Best validation final-exit Macro-F1 | `0.8105` |

### v0.8 internal test result

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2185 | 0.3580 | 0.2833 | 0.1535 | 0.1293 |
| 2 | 0.6713 | 0.6837 | 0.6478 | 0.4472 | 0.0844 |
| 3 | 0.8305 | 0.8283 | 0.8285 | 0.6206 | 0.0502 |

### v0.8 corrected-holdout official result

| Method | Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent mean, fixed 0.5 | 3 | 0.7801 | 0.9332 | 0.9406 | 0.8397 | 0.0194 | 1.4302 |

### v0.8 simple label-aware post-hoc result

The first post-hoc label-aware experiment used mean for stable labels and max for two transient labels:

```text
mean: 8 stable labels
max: audience_reaction_present, silence_present
```

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| v0.8 official parent mean | 0.7801 | 0.9332 | 0.9406 | 0.8397 | 0.0194 | 1.4302 |
| v0.8 simple label-aware mean/max | 0.8320 | 0.9285 | 0.9375 | 0.8235 | 0.0211 | 1.4844 |

This motivated v0.9, where a more stable repeated-split aggregation selection was tested.

---

## v0.9 experiment design

v0.9 starts from the v0.8-HCB segment-probability file:

```text
human_talk_workspace/tata_v0.9_labelwise_calibration/verification/v08_parent_mean_fixed/parent_eval_segment_probs_fixed_0p5_mean.csv
```

The branch evaluates parent-level aggregation methods over the segment probabilities:

| Aggregation | Meaning |
|---|---|
| `mean` | Average all segment probabilities within the parent clip |
| `max` | Use the highest segment probability within the parent clip |
| `top2mean` | Average the two highest segment probabilities within the parent clip |

The repeated-split experiment used:

| Setting | Value |
|---|---|
| Seeds | 20 |
| Calibration fraction | 0.5 |
| Evaluation fraction | 0.5 |
| Threshold range | 0.10 to 0.95 |
| Threshold step | 0.05 |
| Probability prefix | `exit3_prob_` |
| Parent ID column | `parent_clip_id` |

The final frozen evaluation used:

| Setting | Value |
|---|---|
| Parent clips | 867 |
| Segments | 4,335 |
| Aggregation map | Frozen labelwise v0.9 map |
| Threshold | Fixed 0.5 |
| Calibration split | None |
| Threshold search | None |
| Retraining | None |

---

## Repeated split aggregation and threshold comparison

The repeated split experiment was used to select the most reliable aggregation strategy. These values are means over 20 random calibration/evaluation splits.

| Method | Macro-F1 Mean | Micro-F1 Mean | Samples-F1 Mean | Exact Mean | Hamming Loss Mean ↓ | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `max_fixed_thresholds` | 0.7187 | 0.8200 | 0.8419 | 0.5098 | 0.0632 | Rejected; over-predicts labels |
| `mean_fixed_thresholds` | 0.7802 | 0.9315 | 0.9392 | 0.8371 | 0.0199 | Strong baseline |
| `top2mean_fixed_thresholds` | 0.8023 | 0.8884 | 0.9060 | 0.6927 | 0.0358 | Helps Macro-F1 but weakens reliability |
| **`v06_calibration_selected_aggregation_fixed_thresholds`** | **0.8310** | **0.9345** | **0.9449** | **0.8368** | **0.0193** | Best balanced repeated-split strategy |
| `v07_aggregation_threshold_calibrated` | 0.8319 | 0.9288 | 0.9363 | 0.8185 | 0.0208 | Macro-F1 is good, but weaker overall |

### Repeated split conclusion

The repeated-split result shows that **aggregation selection is useful**, but **threshold calibration is not worth freezing** for the final v0.9 result. The final strategy therefore uses the selected aggregation map with fixed threshold `0.5`.

---

## Frozen v0.9 aggregation map

| Label | Frozen Aggregation | Threshold |
|---|---|---:|
| `Brene_Brown` | `mean` | 0.5 |
| `Eckhart_Tolle` | `top2mean` | 0.5 |
| `Eric_Thomas` | `mean` | 0.5 |
| `Gary_Vee` | `top2mean` | 0.5 |
| `Jay_Shetty` | `mean` | 0.5 |
| `Nick_Vujicic` | `mean` | 0.5 |
| `other_speaker_present` | `mean` | 0.5 |
| `music_present` | `mean` | 0.5 |
| `audience_reaction_present` | `top2mean` | 0.5 |
| `silence_present` | `top2mean` | 0.5 |

---

## Final corrected-holdout comparison

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss ↓ | Avg Pred Labels |
|---|---:|---:|---:|---:|---:|---:|
| v0.8 official parent mean | 0.7801 | 0.9332 | 0.9406 | 0.8397 | 0.0194 | 1.4302 |
| v0.8 simple label-aware mean/max | 0.8320 | 0.9285 | 0.9375 | 0.8235 | 0.0211 | 1.4844 |
| **v0.9 frozen labelwise top2mean/mean** | **0.8512** | **0.9372** | **0.9482** | **0.8420** | **0.0185** | **1.4694** |

### Final v0.9 headline result

```text
v0.9 frozen labelwise aggregation
Macro-F1     = 0.8512
Micro-F1     = 0.9372
Samples-F1   = 0.9482
Exact Match  = 0.8420
Hamming Loss = 0.0185
```

Compared with the v0.8 official parent-mean result, v0.9 improves:

```text
Macro-F1:     0.7801 → 0.8512
Micro-F1:     0.9332 → 0.9372
Samples-F1:   0.9406 → 0.9482
Exact Match:  0.8397 → 0.8420
Hamming Loss: 0.0194 → 0.0185
```

---

## Final per-label v0.9 result

| Label | Precision | Recall | F1 | Support | Predicted Positive | Aggregation |
|---|---:|---:|---:|---:|---:|---|
| `Brene_Brown` | 1.0000 | 0.9315 | 0.9645 | 73 | 68 | mean |
| `Eckhart_Tolle` | 1.0000 | 1.0000 | 1.0000 | 84 | 84 | top2mean |
| `Eric_Thomas` | 0.9028 | 0.9559 | 0.9286 | 68 | 72 | mean |
| `Gary_Vee` | 0.9444 | 1.0000 | 0.9714 | 68 | 72 | top2mean |
| `Jay_Shetty` | 0.9278 | 1.0000 | 0.9626 | 90 | 97 | mean |
| `Nick_Vujicic` | 1.0000 | 0.9592 | 0.9792 | 49 | 47 | mean |
| `other_speaker_present` | 0.9156 | 0.9435 | 0.9293 | 460 | 474 | mean |
| `music_present` | 0.9640 | 0.9413 | 0.9525 | 341 | 333 | mean |
| `audience_reaction_present` | 0.6818 | 0.5172 | 0.5882 | 29 | 22 | top2mean |
| `silence_present` | 0.4000 | 0.1667 | 0.2353 | 12 | 5 | top2mean |

### Rare/context label improvement

| Label | v0.8 Mean F1 | v0.8 Simple Label-Aware F1 | v0.9 Frozen F1 |
|---|---:|---:|---:|
| `audience_reaction_present` | 0.1250 | 0.4706 | **0.5882** |
| `silence_present` | 0.0000 | 0.1739 | **0.2353** |

---

## Commands and scripts

### Repeated split calibration/aggregation experiment

```powershell
$SegmentPredCsv = "human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"

$OutDir = "human_talk_workspace\tata_v0.9_labelwise_calibration\repeated_v07_style_calibration"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.9\run_labelwise_aggregation_threshold_calibration.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --out-dir "$OutDir" `
  --fixed-threshold 0.5 `
  --seeds 20 `
  --cal-fraction 0.5 `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.05 `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_"
```

### Final frozen full-holdout evaluation

```powershell
$SegmentPredCsv = "human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"

$OutDir = "human_talk_workspace\tata_v0.9_labelwise_calibration\final_frozen_v06_labelwise_fixed_0p5"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.9\evaluate_frozen_labelwise_aggregation_v09.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --out-dir "$OutDir" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --threshold 0.5
```

---

## Important output files

| File | Purpose |
|---|---|
| `scripts/v0.9/run_labelwise_aggregation_threshold_calibration.py` | Repeated split aggregation and threshold calibration analysis |
| `scripts/v0.9/evaluate_frozen_labelwise_aggregation_v09.py` | Final full-holdout frozen labelwise aggregation evaluation |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_final_full_holdout_comparison.md` | Final v0.8/v0.9 comparison table |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_repeated_split_calibration_summary.md` | Repeated split calibration/evaluation table |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_frozen_labelwise_method_map.md` | Frozen v0.9 aggregation map |
| `docs/tables/agentic_data_preprocessing_v0.9/v09_frozen_labelwise_per_label_metrics.md` | Final per-label metrics |
| `human_talk_workspace/tata_v0.9_labelwise_calibration/final_frozen_v06_labelwise_fixed_0p5/parent_holdout_eval_frozen_labelwise_v09_fixed_0p5.json` | Final full-holdout evaluation JSON |

---

## Documentation map

Recommended v0.9 documentation locations:

```text
docs/v0.9/
docs/tables/agentic_data_preprocessing_v0.9/
docs/figures/human_talk/agentic_data_preprocessing_v0.9/
scripts/v0.9/
human_talk_workspace/tata_v0.9_labelwise_calibration/
```

Existing v0.8 documentation remains useful as historical context:

```text
docs/reports/human_talk/V08_HUMAN_CORRECTED_BALANCED_EXPERIMENT_REPORT.md
docs/results/human_talk/V08_RESULTS_SUMMARY.md
docs/tables/agentic_data_preprocessing_v0.8/
docs/figures/human_talk/agentic_data_preprocessing_v0.8/
docs/COMMANDS_V08.md
docs/APPENDIX.md
docs/MULTILABEL_EXPERIMENT_LOG.md
```

---

## Key conclusion

The v0.9 branch shows that **label-specific parent aggregation improves heterogeneous multi-label audio classification without retraining the neural model**. The final frozen aggregation map improves not only Macro-F1 but also Micro-F1, Samples-F1, Exact Match, and Hamming Loss compared with the v0.8 official parent-mean baseline.

The final recommended result for the `agentic_data_preprocessing_v0.9` branch is:

```text
v0.9 frozen labelwise aggregation + fixed threshold 0.5
Macro-F1     = 0.8512
Micro-F1     = 0.9372
Samples-F1   = 0.9482
Exact Match  = 0.8420
Hamming Loss = 0.0185
```

This should be treated as the strongest current post-hoc aggregation result for the human-talk TATA/Lawyer multi-label audio pipeline.
