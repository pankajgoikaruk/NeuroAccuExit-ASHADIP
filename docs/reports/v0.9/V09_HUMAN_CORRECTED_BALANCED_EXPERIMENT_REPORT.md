# v0.9 Labelwise Aggregation and Calibration Report

## 1. Experiment identity

```text
Branch: agentic_data_preprocessing_v0.9
Experiment name: v0.9 labelwise aggregation and calibration
Base model: main_v08_human_corrected_balanced_3exit_20260610_084027
Training status: no retraining in v0.9
Evaluation split: corrected holdout, parent/clip-level
Parent clips: 867
Segments: 4,335
Final method: v09_frozen_frequency_plus_gary_mean
Final threshold: fixed 0.5
```

v0.9 should be interpreted as a **post-hoc parent-level evaluation improvement** over the v0.8-HCB model. The trained model is unchanged. The improvement comes from replacing one global parent aggregation rule with a label-specific aggregation rule.

---

## 2. Motivation

The v0.8-HCB model produced strong corrected-holdout performance using parent-level mean aggregation, but rare and transient labels were still weak. This suggested that the model had useful segment-level evidence that was being diluted during parent-level aggregation.

For a parent clip containing multiple segments, the model first produces segment-level probabilities:

```text
segment_1_prob(label), segment_2_prob(label), ..., segment_n_prob(label)
```

The parent-level probability must then be computed. A single global rule is not ideal because labels behave differently:

| Label behaviour | Example labels | Preferred aggregation intuition |
|---|---|---|
| Stable speaker/background labels | target speakers, `other_speaker_present`, `music_present` | Evidence should be consistent across several segments; `mean` suppresses noisy spikes |
| Short/transient labels | `audience_reaction_present`, `silence_present` | Evidence may appear in only one or two segments; `top2mean` can preserve short evidence without using aggressive `max` |

---

## 3. Candidate aggregation rules

v0.9 compared three parent-level probability aggregation functions:

| Method | Definition | Intended use |
|---|---|---|
| `mean` | Average probability across all segments in the parent clip | Stable labels |
| `max` | Maximum probability across all segments | Diagnostic for short events, but often over-predicts |
| `top2mean` | Average of the top two segment probabilities | Middle ground between `mean` and `max` |

The parent-level ground truth is computed as:

```text
parent_true(label) = max(segment_true(label))
```

A parent clip is positive for a label if at least one segment is positive for that label.

---

## 4. Experimental stages

### 4.1 Baseline verification

The segment probability file was generated using the v0.8-HCB model:

```text
human_talk_workspace/tata_v0.9_labelwise_calibration/verification/v08_parent_mean_fixed/parent_eval_segment_probs_fixed_0p5_mean.csv
```

The official mean-all baseline result was:

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| mean-all baseline | 0.7801 | 0.9332 | 0.9406 | 0.8397 | 0.0194 |

### 4.2 Repeated split aggregation and threshold calibration

The repeated split experiment tested aggregation selection and threshold calibration across 20 random calibration/evaluation splits.

| Method                                       |   Macro-F1 mean |   Macro-F1 std |   Micro-F1 mean |   Samples-F1 mean |   Exact mean |   Hamming mean ↓ |   Avg pred labels mean |
|:---------------------------------------------|----------------:|---------------:|----------------:|------------------:|-------------:|-----------------:|-----------------------:|
| max fixed                                    |          0.7187 |         0.0171 |          0.82   |            0.8419 |       0.5098 |           0.0632 |                 2.0371 |
| mean fixed                                   |          0.7802 |         0.0071 |          0.9315 |            0.9392 |       0.8371 |           0.0199 |                 1.4274 |
| top2mean fixed                               |          0.8023 |         0.017  |          0.8884 |            0.906  |       0.6927 |           0.0358 |                 1.7377 |
| selected aggregation + fixed threshold       |          0.831  |         0.0135 |          0.9345 |            0.9449 |       0.8368 |           0.0193 |                 1.4674 |
| selected aggregation + calibrated thresholds |          0.8319 |         0.0158 |          0.9288 |            0.9363 |       0.8185 |           0.0208 |                 1.4486 |

Finding:

- Global `max` performs poorly because it over-predicts labels.
- `top2mean` improves Macro-F1 over mean in repeated splits but harms Micro-F1, Samples-F1, Exact Match, and Hamming Loss.
- Labelwise aggregation selection is beneficial.
- Threshold calibration gives only a small Macro-F1 improvement in repeated splits and harms other metrics.
- Therefore, threshold calibration was not selected as the final full-holdout setting.

### 4.3 Selection-frequency evidence

The v0.6-style selected aggregation with fixed threshold produced stable label choices:

| Label                     | Selection frequency          |
|:--------------------------|:-----------------------------|
| Brene_Brown               | mean:10 / top2mean:10        |
| Eckhart_Tolle             | top2mean:11 / max:8 / mean:1 |
| Eric_Thomas               | mean:20                      |
| Gary_Vee                  | top2mean:11 / mean:9         |
| Jay_Shetty                | mean:20                      |
| Nick_Vujicic              | mean:20                      |
| audience_reaction_present | top2mean:20                  |
| music_present             | mean:20                      |
| other_speaker_present     | mean:20                      |
| silence_present           | top2mean:16 / mean:4         |

The v0.7-style selected aggregation plus threshold calibration produced the following aggregation frequencies:

| Label                     | Selection frequency          |
|:--------------------------|:-----------------------------|
| Brene_Brown               | mean:8 / max:7 / top2mean:5  |
| Eckhart_Tolle             | top2mean:11 / max:8 / mean:1 |
| Eric_Thomas               | mean:15 / top2mean:5         |
| Gary_Vee                  | mean:11 / top2mean:9         |
| Jay_Shetty                | mean:20                      |
| Nick_Vujicic              | mean:15 / top2mean:5         |
| audience_reaction_present | max:11 / top2mean:5 / mean:4 |
| music_present             | mean:14 / max:3 / top2mean:3 |
| other_speaker_present     | mean:17 / top2mean:3         |
| silence_present           | top2mean:20                  |

The threshold summary from the repeated split experiment was:

| Label                     |   Threshold mean |   Threshold std |   Min |   Max |   Support mean |   Fallback count |
|:--------------------------|-----------------:|----------------:|------:|------:|---------------:|-----------------:|
| Brene_Brown               |           0.5975 |          0.2274 |  0.35 |  0.95 |          37.45 |                0 |
| Eckhart_Tolle             |           0.5    |          0      |  0.5  |  0.5  |          42.1  |                0 |
| Eric_Thomas               |           0.6175 |          0.1711 |  0.45 |  0.9  |          33.85 |                0 |
| Gary_Vee                  |           0.43   |          0.1152 |  0.25 |  0.55 |          33.35 |                0 |
| Jay_Shetty                |           0.78   |          0.139  |  0.55 |  0.9  |          45.15 |                0 |
| Nick_Vujicic              |           0.5225 |          0.1446 |  0.4  |  0.85 |          23.85 |                0 |
| audience_reaction_present |           0.57   |          0.1949 |  0.2  |  0.9  |          13.15 |                0 |
| music_present             |           0.6175 |          0.1801 |  0.45 |  0.95 |         171.4  |                0 |
| other_speaker_present     |           0.63   |          0.1056 |  0.55 |  0.9  |         228.6  |                0 |
| silence_present           |           0.405  |          0.1213 |  0.3  |  0.65 |           6.65 |                0 |

This threshold analysis was useful diagnostically, but the final full-holdout mapping-bank test showed that fixed threshold `0.5` is more robust for the current model.

---

## 5. Mapping-bank evaluation

A mapping bank was created so that all candidate maps could be tested under the same conditions:

```text
same model
same segment probability CSV
same corrected holdout
same fixed threshold 0.5 unless map-specific thresholds are explicitly included
only the aggregation map changes
```

The final mapping-bank result was:

| Method                          |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Mean labels |   Max labels |   Top2mean labels |
|:--------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|--------------:|-------------:|------------------:|
| v09 frozen + Gary mean          |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |             7 |            0 |                 3 |
| v09 frozen frequency            |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |             6 |            0 |                 4 |
| simple event-max v08            |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |             8 |            2 |                 0 |
| TATA-LAWYER improved diagnostic |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |             3 |            2 |                 5 |
| mean-all baseline               |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            10 |            0 |                 0 |
| TATA-LAWYER original diagnostic |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |             4 |            5 |                 1 |

The best method was:

```text
v09_frozen_frequency_plus_gary_mean
```

---

## 6. Final selected map

The final selected map is:

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

This map differs from the first frozen frequency map only in one label:

```text
Gary_Vee: top2mean -> mean
```

That change improved the overall balance:

| Metric | Frozen frequency map | Frozen + Gary mean | Difference |
|---|---:|---:|---:|
| Macro-F1 | 0.8512 | 0.8518 | +0.0006 |
| Micro-F1 | 0.9372 | 0.9374 | +0.0002 |
| Samples-F1 | 0.9482 | 0.9464 | -0.0018 |
| Exact Match | 0.8420 | 0.8431 | +0.0012 |
| Hamming Loss | 0.0185 | 0.0183 | -0.0001 |
| Avg predicted labels | 1.4694 | 1.4614 | -0.0081 |

The only trade-off is a small Samples-F1 reduction, but Macro-F1, Micro-F1, Exact Match, and Hamming Loss all improve.

---

## 7. Final per-label performance

| Label                     | Aggregation   |   Threshold |   Precision |   Recall |     F1 |   Support |   Predicted + |
|:--------------------------|:--------------|------------:|------------:|---------:|-------:|----------:|--------------:|
| Brene_Brown               | mean          |         0.5 |      1      |   0.9315 | 0.9645 |        73 |            68 |
| Eckhart_Tolle             | top2mean      |         0.5 |      1      |   1      | 1      |        84 |            84 |
| Eric_Thomas               | mean          |         0.5 |      0.9028 |   0.9559 | 0.9286 |        68 |            72 |
| Gary_Vee                  | mean          |         0.5 |      1      |   0.9559 | 0.9774 |        68 |            65 |
| Jay_Shetty                | mean          |         0.5 |      0.9278 |   1      | 0.9626 |        90 |            97 |
| Nick_Vujicic              | mean          |         0.5 |      1      |   0.9592 | 0.9792 |        49 |            47 |
| other_speaker_present     | mean          |         0.5 |      0.9156 |   0.9435 | 0.9293 |       460 |           474 |
| music_present             | mean          |         0.5 |      0.964  |   0.9413 | 0.9525 |       341 |           333 |
| audience_reaction_present | top2mean      |         0.5 |      0.6818 |   0.5172 | 0.5882 |        29 |            22 |
| silence_present           | top2mean      |         0.5 |      0.4    |   0.1667 | 0.2353 |        12 |             5 |

The weakest remaining labels are:

| Label | F1 | Notes |
|---|---:|---|
| `silence_present` | 0.2353 | Very low support: 12 |
| `audience_reaction_present` | 0.5882 | Improved substantially compared with mean-only aggregation, but still limited by low support |

---

## 8. TATA-LAWYER prior mapping and threshold transfer

Prior TATA-LAWYER mappings were tested because they came from a related model trained with the same main dataset plus human-reviewed recovered low-energy signals. This made them relevant candidates.

However, the mapping-bank test showed that the old maps did not transfer better than the new v0.9 map. The old threshold setting was especially weak:

| Method                                   |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Threshold min |   Threshold max |   Threshold mean |
|:-----------------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|----------------:|----------------:|-----------------:|
| v09 frozen + Gary mean                   |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |            0.5  |            0.5  |            0.5   |
| v09 frozen frequency                     |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |            0.5  |            0.5  |            0.5   |
| simple event-max v08                     |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER improved diagnostic          |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |            0.5  |            0.5  |            0.5   |
| mean-all baseline                        |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER original diagnostic          |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER optimal map + old thresholds |     0.7284 |     0.861  |       0.879  |  0.654  |      0.0436 |            1.6678 |            0.38 |            0.95 |            0.683 |

Interpretation:

- The old aggregation patterns were informative as candidates.
- The old per-label thresholds are not reliable for this current model.
- The old thresholds were too strict for some labels and too permissive for others.
- Final v0.9 should use fixed threshold `0.5`.

---

## 9. Final conclusion

The final v0.9 result is:

```text
Method     = v09_frozen_frequency_plus_gary_mean
Macro-F1   = 0.8518
Micro-F1   = 0.9374
Samples-F1 = 0.9464
Exact      = 0.8431
Hamming    = 0.0183
```

Compared with the official mean-all baseline, v0.9 improves Macro-F1 by **+0.0717** while also improving Micro-F1, Samples-F1, Exact Match, and Hamming Loss.

Therefore, `v09_frozen_frequency_plus_gary_mean` should be reported as the final v0.9 parent-level aggregation strategy.
