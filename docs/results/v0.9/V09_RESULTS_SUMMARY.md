# v0.9 Results Summary

## Experiment identity

```text
Branch: agentic_data_preprocessing_v0.9
Experiment family: v0.9 labelwise parent aggregation and calibration
Base model: main_v08_human_corrected_balanced_3exit_20260610_084027
Training: no retraining in v0.9
Evaluation: corrected holdout, parent/clip level
Parent clips: 867
Segments: 4,335
Labels: 10
Final threshold: fixed 0.5
Final selected method: v09_frozen_frequency_plus_gary_mean
```

---

## Final result

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss ↓ | Avg true labels | Avg pred labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v09 frozen + Gary mean** | **0.8518** | **0.9374** | 0.9464 | **0.8431** | **0.0183** | 1.4694 | 1.4614 |

---

## Comparison against baselines

| Method                          |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Mean labels |   Max labels |   Top2mean labels |
|:--------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|--------------:|-------------:|------------------:|
| v09 frozen + Gary mean          |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |             7 |            0 |                 3 |
| v09 frozen frequency            |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |             6 |            0 |                 4 |
| simple event-max v08            |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |             8 |            2 |                 0 |
| TATA-LAWYER improved diagnostic |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |             3 |            2 |                 5 |
| mean-all baseline               |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            10 |            0 |                 0 |
| TATA-LAWYER original diagnostic |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |             4 |            5 |                 1 |

Decision:

```text
Use v09_frozen_frequency_plus_gary_mean as the final v0.9 method.
```

---

## Final aggregation map

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

## Final per-label result

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

## Improvement over official mean baseline

| Metric | Mean-all baseline | Final v0.9 | Difference |
|---|---:|---:|---:|
| Macro-F1 | 0.7801 | 0.8518 | +0.0717 |
| Micro-F1 | 0.9332 | 0.9374 | +0.0043 |
| Samples-F1 | 0.9406 | 0.9464 | +0.0058 |
| Exact Match | 0.8397 | 0.8431 | +0.0035 |
| Hamming Loss | 0.0194 | 0.0183 | -0.0010 |
| Avg pred labels | 1.4302 | 1.4614 | +0.0311 |

---

## Threshold-transfer diagnostic

The old TATA-LAWYER optimal map with old per-label thresholds was tested and rejected:

| Method                                   |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact |   Hamming ↓ |   Avg pred labels |   Threshold min |   Threshold max |   Threshold mean |
|:-----------------------------------------|-----------:|-----------:|-------------:|--------:|------------:|------------------:|----------------:|----------------:|-----------------:|
| v09 frozen + Gary mean                   |     0.8518 |     0.9374 |       0.9464 |  0.8431 |      0.0183 |            1.4614 |            0.5  |            0.5  |            0.5   |
| v09 frozen frequency                     |     0.8512 |     0.9372 |       0.9482 |  0.842  |      0.0185 |            1.4694 |            0.5  |            0.5  |            0.5   |
| simple event-max v08                     |     0.832  |     0.9285 |       0.9375 |  0.8235 |      0.0211 |            1.4844 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER improved diagnostic          |     0.7995 |     0.8706 |       0.8865 |  0.6436 |      0.0422 |            1.7924 |            0.5  |            0.5  |            0.5   |
| mean-all baseline                        |     0.7801 |     0.9332 |       0.9406 |  0.8397 |      0.0194 |            1.4302 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER original diagnostic          |     0.7617 |     0.8531 |       0.874  |  0.6032 |      0.0483 |            1.8212 |            0.5  |            0.5  |            0.5   |
| TATA-LAWYER optimal map + old thresholds |     0.7284 |     0.861  |       0.879  |  0.654  |      0.0436 |            1.6678 |            0.38 |            0.95 |            0.683 |

The old threshold values did not transfer to the current v0.9 model. Therefore, the final result uses fixed threshold `0.5`.

---

## Evidence files

| Evidence | Path |
|---|---|
| Mapping-bank summary | `docs/tables/agentic_data_preprocessing_v0.9/v09_mapping_bank_summary_fixed_0p5_plus_gary_mean.csv` |
| Mapping-bank per-label | `docs/tables/agentic_data_preprocessing_v0.9/v09_mapping_bank_per_label_fixed_0p5_plus_gary_mean.csv` |
| Repeated split summary | `docs/tables/agentic_data_preprocessing_v0.9/v09_repeated_eval_summary.csv` |
| Threshold diagnostic | `docs/tables/agentic_data_preprocessing_v0.9/v09_mapping_bank_summary_with_tata_lawyer_thresholds.csv` |
