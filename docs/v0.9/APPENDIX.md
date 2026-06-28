# Appendix — v0.9 Labelwise Aggregation and Calibration

## A. Scope of v0.9

v0.9 is a post-hoc evaluation branch. It reuses the v0.8-HCB model and changes only the parent-level aggregation strategy.

```text
Base model: main_v08_human_corrected_balanced_3exit_20260610_084027
Parent clips: 867
Segments: 4,335
Threshold: fixed 0.5
Final map: v09_frozen_frequency_plus_gary_mean
```

---

## B. Parent-level aggregation formulation

Let a parent clip contain `S` segments. For label `l`, the model produces segment probabilities:

```text
p_1(l), p_2(l), ..., p_S(l)
```

The parent probability is computed by one of:

```text
mean(l)     = average(p_1(l), ..., p_S(l))
max(l)      = max(p_1(l), ..., p_S(l))
top2mean(l) = average(top two probabilities among p_1(l), ..., p_S(l))
```

The final parent-level prediction is:

```text
y_hat_parent(l) = 1 if p_parent(l) >= 0.5 else 0
```

The parent-level truth is:

```text
y_parent(l) = max(y_segment_1(l), ..., y_segment_S(l))
```

---

## C. Final v0.9 aggregation map

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

---

## D. Mapping-bank comparison

| Method                          |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Mean labels |   Max labels |   Top2mean labels |
|:--------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|--------------:|-------------:|------------------:|
| v09 frozen + Gary mean          |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |             7 |            0 |                 3 |
| v09 frozen frequency            |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |             6 |            0 |                 4 |
| simple event-max v08            |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |             8 |            2 |                 0 |
| TATA-LAWYER improved diagnostic |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |             3 |            2 |                 5 |
| mean-all baseline               |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            10 |            0 |                 0 |
| TATA-LAWYER original diagnostic |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |             4 |            5 |                 1 |

The best map is `v09_frozen_frequency_plus_gary_mean`.

---

## E. Final per-label result

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

---

## F. Repeated split calibration summary

| Method                                       |   Macro-F1 mean |   Macro-F1 std |   Micro-F1 mean |   Samples-F1 mean |   Exact mean |   Hamming mean ↓ |   Avg pred labels mean |
|:---------------------------------------------|----------------:|---------------:|----------------:|------------------:|-------------:|-----------------:|-----------------------:|
| max fixed                                    |          0.7187 |         0.0171 |          0.82   |            0.8419 |       0.5098 |           0.0632 |                 2.0371 |
| mean fixed                                   |          0.7802 |         0.0071 |          0.9315 |            0.9392 |       0.8371 |           0.0199 |                 1.4274 |
| top2mean fixed                               |          0.8023 |         0.017  |          0.8884 |            0.906  |       0.6927 |           0.0358 |                 1.7377 |
| selected aggregation + fixed threshold       |          0.831  |         0.0135 |          0.9345 |            0.9449 |       0.8368 |           0.0193 |                 1.4674 |
| selected aggregation + calibrated thresholds |          0.8319 |         0.0158 |          0.9288 |            0.9363 |       0.8185 |           0.0208 |                 1.4486 |

Repeated split calibration helped identify which labels prefer `mean` or `top2mean`. However, calibrated thresholds were not selected for the final result because they did not show a robust full-holdout advantage.

---

## G. Selection-frequency details

### G.1 v0.6-style fixed-threshold selection

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

### G.2 v0.7-style threshold-calibrated selection

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

### G.3 Threshold summary

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

---

## H. TATA-LAWYER threshold-transfer diagnostic

Old TATA-LAWYER thresholds were tested because that experiment used the same main dataset plus recovered low-energy signals reviewed by humans. The thresholds did not transfer to the current v0.9 model.

| Method                                   |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Threshold min |   Threshold max |   Threshold mean |
|:-----------------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|----------------:|----------------:|-----------------:|
| v09 frozen + Gary mean                   |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |            0.5  |            0.5  |            0.5   |
| v09 frozen frequency                     |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |            0.5  |            0.5  |            0.5   |
| simple event-max v08                     |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER improved diagnostic          |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |            0.5  |            0.5  |            0.5   |
| mean-all baseline                        |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER original diagnostic          |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER optimal map + old thresholds |     0.7284 |     0.861  |       0.879  |  0.654  |      0.0436 |            1.6678 |            0.38 |            0.95 |            0.683 |

Decision:

```text
Reject old threshold values.
Use fixed threshold 0.5.
```

---

## I. Final reporting statement

```text
The v0.9 branch introduces a frozen labelwise parent aggregation strategy for the corrected human-talk holdout. 
Using the v0.8-HCB model without retraining, labelwise aggregation improves Macro-F1 from 0.7801 under mean-all aggregation to 0.8518. 
The final map uses mean aggregation for seven labels and top2mean aggregation for Eckhart_Tolle, audience_reaction_present, and silence_present. 
The final fixed threshold is 0.5 for all labels.
```
