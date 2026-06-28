# Multilabel Experiment Log — v0.9

## 1. Branch objective

v0.9 investigates whether the v0.8-HCB trained model can be improved at parent/clip level through label-specific probability aggregation.

```text
Branch: agentic_data_preprocessing_v0.9
Base model: main_v08_human_corrected_balanced_3exit_20260610_084027
Training: no new training
Evaluation: corrected holdout
Parent clips: 867
Segments: 4,335
```

---

## 2. Official mean-all baseline

The v0.8-HCB model was first evaluated with parent-level mean aggregation for all labels.

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| mean-all baseline | 0.7801 | 0.9332 | 0.9406 | 0.8397 | 0.0194 |

Finding:

```text
Mean aggregation is strong overall but dilutes rare transient evidence.
```

---

## 3. Simple event-max diagnostic

A simple rule was tested:

```text
mean for 8 stable labels
max for audience_reaction_present and silence_present
```

Result:

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| simple event-max v08 | 0.8320 | 0.9285 | 0.9375 | 0.8235 | 0.0211 |

Finding:

```text
Macro-F1 improves, confirming that rare/transient labels need different aggregation.
```

---

## 4. Repeated split labelwise aggregation and threshold calibration

Repeated split calibration tested mean, max, top2mean, selected aggregation, and selected aggregation with threshold calibration.

| Method                                       |   Macro-F1 mean |   Macro-F1 std |   Micro-F1 mean |   Samples-F1 mean |   Exact mean |   Hamming mean ↓ |   Avg pred labels mean |
|:---------------------------------------------|----------------:|---------------:|----------------:|------------------:|-------------:|-----------------:|-----------------------:|
| max fixed                                    |          0.7187 |         0.0171 |          0.82   |            0.8419 |       0.5098 |           0.0632 |                 2.0371 |
| mean fixed                                   |          0.7802 |         0.0071 |          0.9315 |            0.9392 |       0.8371 |           0.0199 |                 1.4274 |
| top2mean fixed                               |          0.8023 |         0.017  |          0.8884 |            0.906  |       0.6927 |           0.0358 |                 1.7377 |
| selected aggregation + fixed threshold       |          0.831  |         0.0135 |          0.9345 |            0.9449 |       0.8368 |           0.0193 |                 1.4674 |
| selected aggregation + calibrated thresholds |          0.8319 |         0.0158 |          0.9288 |            0.9363 |       0.8185 |           0.0208 |                 1.4486 |

Finding:

```text
Labelwise aggregation selection helps. Threshold calibration is useful diagnostically but not selected as final.
```

---

## 5. Frozen frequency map

The first frozen v0.9 map was built from stable selection frequencies.

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v09 frozen frequency map | 0.8512 | 0.9372 | 0.9482 | 0.8420 | 0.0185 |

Map:

```text
mean:
  Brene_Brown
  Eric_Thomas
  Jay_Shetty
  Nick_Vujicic
  other_speaker_present
  music_present

top2mean:
  Eckhart_Tolle
  Gary_Vee
  audience_reaction_present
  silence_present
```

---

## 6. Mapping-bank evaluation

The mapping-bank evaluator was introduced to avoid confusion between old maps, frozen maps, and threshold-calibrated maps.

| Method                          |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Mean labels |   Max labels |   Top2mean labels |
|:--------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|--------------:|-------------:|------------------:|
| v09 frozen + Gary mean          |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |             7 |            0 |                 3 |
| v09 frozen frequency            |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |             6 |            0 |                 4 |
| simple event-max v08            |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |             8 |            2 |                 0 |
| TATA-LAWYER improved diagnostic |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |             3 |            2 |                 5 |
| mean-all baseline               |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            10 |            0 |                 0 |
| TATA-LAWYER original diagnostic |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |             4 |            5 |                 1 |

Finding:

```text
The best method is v09_frozen_frequency_plus_gary_mean.
```

---

## 7. Gary_Vee correction

The per-label comparison showed that `Gary_Vee` performed slightly better with mean aggregation than with top2mean. Therefore, one final hybrid map was tested:

```text
v09_frozen_frequency_map
but Gary_Vee: top2mean -> mean
```

Result:

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| v09 frozen + Gary mean | 0.8518 | 0.9374 | 0.9464 | 0.8431 | 0.0183 |

Decision:

```text
Select v09_frozen_frequency_plus_gary_mean as final v0.9.
```

---

## 8. TATA-LAWYER mappings and thresholds

Old TATA-LAWYER maps were tested as relevant prior candidates. They did not outperform the current v0.9-derived map.

The old threshold map was also tested:

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| TATA-LAWYER optimal map + old thresholds | 0.7284 | 0.8610 | 0.8790 | 0.6540 | 0.0436 |

Decision:

```text
Reject old TATA-LAWYER thresholds for this model.
```

---

## 9. Final v0.9 status

```text
Final method: v09_frozen_frequency_plus_gary_mean
Macro-F1:   0.8518
Micro-F1:   0.9374
Samples-F1: 0.9464
Exact:      0.8431
Hamming:    0.0183
```

v0.9 is ready to document as the final parent-level labelwise aggregation branch.
