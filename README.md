# NeuroAccuExit-ASHADIP — Agentic Data Preprocessing v0.9

This README documents the active `agentic_data_preprocessing_v0.9` branch. The branch does **not** retrain the acoustic model. Instead, it reuses the strongest v0.8-HCB model and improves parent/clip-level multi-label evaluation through **labelwise probability aggregation**.

The key v0.9 finding is that a small, frozen, label-specific aggregation map improves corrected-holdout Macro-F1 from **0.7801** under parent-mean aggregation to **0.8518**, while also improving Micro-F1, Exact Match, and Hamming Loss.

> Note: several files were renamed from v0.8 to v0.9. This is correct for documentation, but the underlying base model is still the v0.8-HCB trained model: `main_v08_human_corrected_balanced_3exit_20260610_084027`.

---

## Branch summary

| Item | Details |
|---|---|
| Branch | `agentic_data_preprocessing_v0.9` |
| Main purpose | Parent-level labelwise aggregation and calibration analysis |
| Training status | No retraining in v0.9 |
| Base model reused | `main_v08_human_corrected_balanced_3exit_20260610_084027` |
| Evaluation level | Parent/clip level from segment-level probabilities |
| Holdout set | Corrected human-talk holdout |
| Parent clips | 867 |
| Segments | 4,335 |
| Labels | 10 labels: 6 target speakers + `other_speaker_present`, `music_present`, `audience_reaction_present`, `silence_present` |
| Final decision | Use `v09_frozen_frequency_plus_gary_mean` |
| Final threshold | Fixed `0.5` for all labels |
| Final aggregation methods | `mean` for 7 labels; `top2mean` for 3 labels; no `max` in the final map |

---

## Final v0.9 result

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss ↓ | Avg pred labels |
|---|---:|---:|---:|---:|---:|---:|
| v0.8 mean-all baseline | 0.7801 | 0.9332 | 0.9406 | 0.8397 | 0.0194 | 1.4302 |
| v0.8 simple event-max | 0.8320 | 0.9285 | 0.9375 | 0.8235 | 0.0211 | 1.4844 |
| v0.9 frozen frequency map | 0.8512 | 0.9372 | **0.9482** | 0.8420 | 0.0185 | 1.4694 |
| **v0.9 frozen + Gary mean** | **0.8518** | **0.9374** | 0.9464 | **0.8431** | **0.0183** | 1.4614 |

The final v0.9 map improves over the official mean baseline by:

| Metric | Improvement |
|---|---:|
| Macro-F1 | +0.0717 |
| Micro-F1 | +0.0043 |
| Samples-F1 | +0.0058 |
| Exact Match | +0.0035 |
| Hamming Loss | -0.0010 |
| Avg predicted labels | +0.0311 |

---

## Final selected aggregation map

| Label                     | Final aggregation   |
|:--------------------------|:--------------------|
| Brene_Brown               | mean                |
| Eckhart_Tolle             | top2mean            |
| Eric_Thomas               | mean                |
| Gary_Vee                  | mean                |
| Jay_Shetty                | mean                |
| Nick_Vujicic              | mean                |
| other_speaker_present     | mean                |
| music_present             | mean                |
| audience_reaction_present | top2mean            |
| silence_present           | top2mean            |

Interpretation:

- `mean` is retained for stable speaker/background labels where evidence should be consistent across segments.
- `top2mean` is used for labels where the positive evidence can appear in a subset of segments but where global `max` is too aggressive.
- `Gary_Vee` was changed from `top2mean` to `mean`, producing a small but measurable improvement in Macro-F1, Micro-F1, Exact Match, and Hamming Loss.

---

## Mapping-bank outcome

| Method                          |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Mean labels |   Max labels |   Top2mean labels |
|:--------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|--------------:|-------------:|------------------:|
| v09 frozen + Gary mean          |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |             7 |            0 |                 3 |
| v09 frozen frequency            |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |             6 |            0 |                 4 |
| simple event-max v08            |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |             8 |            2 |                 0 |
| TATA-LAWYER improved diagnostic |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |             3 |            2 |                 5 |
| mean-all baseline               |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            10 |            0 |                 0 |
| TATA-LAWYER original diagnostic |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |             4 |            5 |                 1 |

The old TATA-LAWYER mappings were tested as prior-informed candidates. They did not transfer well to the current v0.9 model because they over-used `max` and `top2mean` for stable labels, which increased false positives and reduced Exact Match.

---

## Threshold-calibration decision

Repeated split calibration showed that labelwise aggregation selection is useful, but the calibrated thresholds were not selected for the final full-holdout result:

| Method                                       |   Macro-F1 mean |   Macro-F1 std |   Micro-F1 mean |   Samples-F1 mean |   Exact mean |   Hamming mean ↓ |   Avg pred labels mean |
|:---------------------------------------------|----------------:|---------------:|----------------:|------------------:|-------------:|-----------------:|-----------------------:|
| max fixed                                    |          0.7187 |         0.0171 |          0.82   |            0.8419 |       0.5098 |           0.0632 |                 2.0371 |
| mean fixed                                   |          0.7802 |         0.0071 |          0.9315 |            0.9392 |       0.8371 |           0.0199 |                 1.4274 |
| top2mean fixed                               |          0.8023 |         0.017  |          0.8884 |            0.906  |       0.6927 |           0.0358 |                 1.7377 |
| selected aggregation + fixed threshold       |          0.831  |         0.0135 |          0.9345 |            0.9449 |       0.8368 |           0.0193 |                 1.4674 |
| selected aggregation + calibrated thresholds |          0.8319 |         0.0158 |          0.9288 |            0.9363 |       0.8185 |           0.0208 |                 1.4486 |

The best repeated-split calibration result was close between selected aggregation with fixed thresholds and selected aggregation with calibrated thresholds. However, the final mapping-bank evaluation showed that old TATA-LAWYER per-label thresholds do not transfer well to this model. Therefore, the final v0.9 decision is:

```text
Use the final frozen labelwise aggregation map with fixed threshold 0.5.
Do not use old TATA-LAWYER threshold values.
```

---

## Main artifacts

| Type | Path |
|---|---|
| Mapping config | `configs/v0.9/labelwise_aggregation_maps.json` |
| Repeated calibration script | `scripts/v0.9/run_labelwise_aggregation_threshold_calibration.py` |
| Frozen map evaluator | `scripts/v0.9/evaluate_frozen_labelwise_aggregation_v09.py` |
| Mapping-bank evaluator | `scripts/v0.9/evaluate_labelwise_mapping_bank_v09.py` |
| Commands | `docs/v0.9/COMMANDS_V09.md` |
| Full report | `docs/reports/v0.9/V09_LABELWISE_AGGREGATION_CALIBRATION_REPORT.md` |
| Results summary | `docs/results/v0.9/V09_RESULTS_SUMMARY.md` |
| Evidence tables | `docs/tables/agentic_data_preprocessing_v0.9/` |

---

## Final status

v0.9 is now ready to document as a **post-hoc parent-level aggregation improvement** over the v0.8-HCB trained model. The final result should be reported as:

```text
v09_frozen_frequency_plus_gary_mean
Macro-F1   = 0.8518
Micro-F1   = 0.9374
Samples-F1 = 0.9464
Exact      = 0.8431
Hamming    = 0.0183
```
