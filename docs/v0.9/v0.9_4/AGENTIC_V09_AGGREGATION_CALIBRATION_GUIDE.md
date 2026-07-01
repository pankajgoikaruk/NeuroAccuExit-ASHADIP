# v0.9_4 LATS-v2 Label-wise Aggregation and Threshold Calibration Report

## 1. Experiment identity

```text
Branch: agentic_data_preprocessing_v0.9_4
Base model: main_v08_human_corrected_balanced_3exit_20260610_084027
Training status: no retraining
Evaluation level: parent/clip-level
Parent clips: 867
Segments: 4,335
Labels: 10
Previous best: LATS-v1 final frozen config
New method: LATS-v2 metric-aware coordinate search
```

## 2. Motivation

LATS-v1 improved parent-level prediction by selecting label-specific aggregation--threshold pairs. However, its selection was mostly label-wise. In multi-label inference, the best rule for one label can harm complete prediction-set quality by increasing false positives or causing label-count mismatch. LATS-v2 therefore evaluates each candidate label update using a full multi-label objective.

## 3. Method

LATS-v2 starts from the LATS-v1 frozen configuration and tries to update one label at a time. A candidate update is accepted only if the complete multi-label objective improves. This makes LATS-v2 a metric-aware inference-time optimisation policy rather than a simple independent label-wise calibration procedure.

## 4. Global result

| Method                      |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact Match |   Hamming Loss ↓ |   Avg pred labels |
|:----------------------------|-----------:|-----------:|-------------:|--------------:|-----------------:|------------------:|
| v0.8 mean-all baseline      |     0.7801 |     0.9332 |       0.9406 |        0.8397 |           0.0194 |            1.4302 |
| v0.8 simple event-max       |     0.832  |     0.9285 |       0.9375 |        0.8235 |           0.0211 |            1.4844 |
| v0.9 frozen frequency map   |     0.8512 |     0.9372 |       0.9482 |        0.842  |           0.0185 |            1.4694 |
| v0.9 frozen + Gary mean     |     0.8518 |     0.9374 |       0.9464 |        0.8431 |           0.0183 |            1.4614 |
| LATS-v1 final frozen config |     0.8667 |     0.9436 |       0.9495 |        0.8524 |           0.0165 |            1.4544 |
| LATS-v2 metric-aware config |     0.8673 |     0.9458 |       0.9517 |        0.8604 |           0.0158 |            1.4452 |

## 5. LATS-v1 vs LATS-v2

| Metric                |   LATS-v1 |   LATS-v2 |   Difference |
|:----------------------|----------:|----------:|-------------:|
| Macro-F1              |    0.8667 |    0.8673 |       0.0005 |
| Micro-F1              |    0.9436 |    0.9458 |       0.0022 |
| Samples-F1            |    0.9495 |    0.9517 |       0.0022 |
| Exact Match           |    0.8524 |    0.8604 |       0.0081 |
| Hamming Loss ↓        |    0.0165 |    0.0158 |      -0.0007 |
| Jaccard               |    0.9274 |    0.9309 |       0.0035 |
| Avg true labels       |    1.4694 |    1.4694 |       0      |
| Avg pred labels       |    1.4544 |    1.4452 |      -0.0092 |
| Label-count abs error |    0.015  |    0.0242 |       0.0092 |

## 6. Pair changes

| Label                     | LATS-v1 aggregation   |   LATS-v1 threshold | LATS-v2 aggregation   |   LATS-v2 threshold | Changed                |
|:--------------------------|:----------------------|--------------------:|:----------------------|--------------------:|:-----------------------|
| Brene_Brown               | top3mean              |                0.5  | top3mean              |                0.5  | No                     |
| Eckhart_Tolle             | top2mean              |                0.5  | top2mean              |                0.5  | No                     |
| Eric_Thomas               | mean                  |                0.53 | mean                  |                0.54 | threshold              |
| Gary_Vee                  | top3mean              |                0.5  | top3mean              |                0.5  | No                     |
| Jay_Shetty                | mean                  |                0.8  | mean                  |                0.82 | threshold              |
| Nick_Vujicic              | mean                  |                0.43 | mean                  |                0.43 | No                     |
| other_speaker_present     | top3mean              |                0.76 | top3mean              |                0.76 | No                     |
| music_present             | mean                  |                0.49 | mean                  |                0.49 | No                     |
| audience_reaction_present | max                   |                0.68 | max                   |                0.68 | No                     |
| silence_present           | top2mean              |                0.38 | p75                   |                0.34 | aggregation, threshold |

## 7. Label-wise changes

| Label                     |   LATS-v1 F1 |   LATS-v2 F1 |   F1 change |   LATS-v1 errors |   LATS-v2 errors |   Error change |   LATS-v1 pred+ |   LATS-v2 pred+ |
|:--------------------------|-------------:|-------------:|------------:|-----------------:|-----------------:|---------------:|----------------:|----------------:|
| Brene_Brown               |       0.979  |       0.979  |      0      |                3 |                3 |              0 |              70 |              70 |
| Eckhart_Tolle             |       1      |       1      |      0      |                0 |                0 |              0 |              84 |              84 |
| Eric_Thomas               |       0.9343 |       0.9412 |      0.0069 |                9 |                8 |             -1 |              69 |              68 |
| Gary_Vee                  |       0.9853 |       0.9853 |      0      |                2 |                2 |              0 |              68 |              68 |
| Jay_Shetty                |       0.967  |       0.9724 |      0.0053 |                6 |                5 |             -1 |              92 |              91 |
| Nick_Vujicic              |       0.9897 |       0.9897 |      0      |                1 |                1 |              0 |              48 |              48 |
| other_speaker_present     |       0.9444 |       0.9444 |      0      |               51 |               51 |              0 |             457 |             457 |
| music_present             |       0.9556 |       0.9556 |      0      |               30 |               30 |              0 |             335 |             335 |
| audience_reaction_present |       0.6087 |       0.6087 |      0      |               18 |               18 |              0 |              17 |              17 |
| silence_present           |       0.303  |       0.2963 |     -0.0067 |               23 |               19 |             -4 |              21 |              15 |

## 8. Repeated split comparison

| Metric          |   LATS-v1 mean |   LATS-v1 std |   LATS-v2 mean |   LATS-v2 std |   Mean difference |
|:----------------|---------------:|--------------:|---------------:|--------------:|------------------:|
| Macro-F1        |         0.8309 |        0.0154 |         0.8304 |        0.017  |           -0.0006 |
| Micro-F1        |         0.9293 |        0.0067 |         0.9326 |        0.0065 |            0.0033 |
| Samples-F1      |         0.9369 |        0.0073 |         0.9402 |        0.0066 |            0.0032 |
| Exact Match     |         0.8179 |        0.0182 |         0.8275 |        0.0147 |            0.0096 |
| Hamming Loss ↓  |         0.0207 |        0.0021 |         0.0197 |        0.002  |           -0.001  |
| Avg pred labels |         1.4606 |        0.032  |         1.4502 |        0.0319 |           -0.0104 |
| Objective score |       nan      |      nan      |         0.8283 |        0.0075 |          nan      |

## 9. Stability check: LATS-v1

| Version   | Label                     | Final aggregation   |   Final threshold | Selection count   |   Selection fraction | Stability   |
|:----------|:--------------------------|:--------------------|------------------:|:------------------|---------------------:|:------------|
| LATS-v1   | Brene_Brown               | top3mean            |              0.5  | 8/20              |                 0.4  | Low         |
| LATS-v1   | Eckhart_Tolle             | top2mean            |              0.5  | 19/20             |                 0.95 | High        |
| LATS-v1   | Eric_Thomas               | mean                |              0.53 | 10/20             |                 0.5  | Moderate    |
| LATS-v1   | Gary_Vee                  | top3mean            |              0.5  | 11/20             |                 0.55 | Moderate    |
| LATS-v1   | Jay_Shetty                | mean                |              0.8  | 14/20             |                 0.7  | Moderate    |
| LATS-v1   | Nick_Vujicic              | mean                |              0.43 | 10/20             |                 0.5  | Moderate    |
| LATS-v1   | other_speaker_present     | top3mean            |              0.76 | 10/20             |                 0.5  | Moderate    |
| LATS-v1   | music_present             | mean                |              0.49 | 14/20             |                 0.7  | Moderate    |
| LATS-v1   | audience_reaction_present | max                 |              0.68 | 11/20             |                 0.55 | Moderate    |
| LATS-v1   | silence_present           | top2mean            |              0.38 | 15/20             |                 0.75 | High        |

## 10. Stability check: LATS-v2

| Version   | Label                     | Final aggregation   |   Final threshold | Selection count   |   Selection fraction | Stability   |
|:----------|:--------------------------|:--------------------|------------------:|:------------------|---------------------:|:------------|
| LATS-v2   | Brene_Brown               | top3mean            |              0.5  | 11/20             |                 0.55 | Moderate    |
| LATS-v2   | Eckhart_Tolle             | top2mean            |              0.5  | 20/20             |                 1    | High        |
| LATS-v2   | Eric_Thomas               | mean                |              0.54 | 8/20              |                 0.4  | Low         |
| LATS-v2   | Gary_Vee                  | top3mean            |              0.5  | 9/20              |                 0.45 | Low         |
| LATS-v2   | Jay_Shetty                | mean                |              0.82 | 11/20             |                 0.55 | Moderate    |
| LATS-v2   | Nick_Vujicic              | mean                |              0.43 | 11/20             |                 0.55 | Moderate    |
| LATS-v2   | other_speaker_present     | top3mean            |              0.76 | 6/20              |                 0.3  | Low         |
| LATS-v2   | music_present             | mean                |              0.49 | 12/20             |                 0.6  | Moderate    |
| LATS-v2   | audience_reaction_present | max                 |              0.68 | 9/20              |                 0.45 | Low         |
| LATS-v2   | silence_present           | p75                 |              0.34 | 10/20             |                 0.5  | Moderate    |

## 11. Research findings

1. **Frozen model probabilities still contain exploitable parent-level information.** LATS-v2 improves the result without retraining the base model.
2. **Global multi-label optimisation is more appropriate than independent label-wise optimisation.** LATS-v2 improves Exact Match and Hamming Loss by optimising the full prediction set.
3. **Most of the improvement comes from targeted changes, not a full rewrite of the policy.** Only `Eric_Thomas`, `Jay_Shetty`, and `silence_present` changed from LATS-v1.
4. **The silence label shows a useful trade-off.** LATS-v2 slightly lowers `silence_present` F1 but reduces false positives, improving the overall multi-label output.
5. **Stability differs by label.** `Eckhart_Tolle` is highly stable, while several labels show lower selection stability, so the final claim should be framed as the best observed frozen configuration under repeated metric-aware coordinate search.

## 12. Final conclusion

Use LATS-v2 as the new best v0.9_4 result. It improves every major full-holdout global metric over LATS-v1 while keeping the model frozen.
