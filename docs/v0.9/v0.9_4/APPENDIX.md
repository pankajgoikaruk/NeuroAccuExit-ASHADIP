# Appendix — v0.9_4 LATS-v2

## A. Final global comparison

| Method                      |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact Match |   Hamming Loss ↓ |   Avg pred labels |
|:----------------------------|-----------:|-----------:|-------------:|--------------:|-----------------:|------------------:|
| v0.8 mean-all baseline      |     0.7801 |     0.9332 |       0.9406 |        0.8397 |           0.0194 |            1.4302 |
| v0.8 simple event-max       |     0.832  |     0.9285 |       0.9375 |        0.8235 |           0.0211 |            1.4844 |
| v0.9 frozen frequency map   |     0.8512 |     0.9372 |       0.9482 |        0.842  |           0.0185 |            1.4694 |
| v0.9 frozen + Gary mean     |     0.8518 |     0.9374 |       0.9464 |        0.8431 |           0.0183 |            1.4614 |
| LATS-v1 final frozen config |     0.8667 |     0.9436 |       0.9495 |        0.8524 |           0.0165 |            1.4544 |
| LATS-v2 metric-aware config |     0.8673 |     0.9458 |       0.9517 |        0.8604 |           0.0158 |            1.4452 |

## B. Pair changes

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

## C. Final per-label performance

| Label                     | Aggregation   |   Threshold |   Precision |   Recall |     F1 |   Support |   Pred + |   Errors |
|:--------------------------|:--------------|------------:|------------:|---------:|-------:|----------:|---------:|---------:|
| Brene_Brown               | top3mean      |        0.5  |      1      |   0.9589 | 0.979  |        73 |       70 |        3 |
| Eckhart_Tolle             | top2mean      |        0.5  |      1      |   1      | 1      |        84 |       84 |        0 |
| Eric_Thomas               | mean          |        0.54 |      0.9412 |   0.9412 | 0.9412 |        68 |       68 |        8 |
| Gary_Vee                  | top3mean      |        0.5  |      0.9853 |   0.9853 | 0.9853 |        68 |       68 |        2 |
| Jay_Shetty                | mean          |        0.82 |      0.967  |   0.9778 | 0.9724 |        90 |       91 |        5 |
| Nick_Vujicic              | mean          |        0.43 |      1      |   0.9796 | 0.9897 |        49 |       48 |        1 |
| other_speaker_present     | top3mean      |        0.76 |      0.9475 |   0.9413 | 0.9444 |       460 |      457 |       51 |
| music_present             | mean          |        0.49 |      0.9642 |   0.9472 | 0.9556 |       341 |      335 |       30 |
| audience_reaction_present | max           |        0.68 |      0.8235 |   0.4828 | 0.6087 |        29 |       17 |       18 |
| silence_present           | p75           |        0.34 |      0.2667 |   0.3333 | 0.2963 |        12 |       15 |       19 |

## D. Repeated split stability

| Metric          |   LATS-v1 mean |   LATS-v1 std |   LATS-v2 mean |   LATS-v2 std |   Mean difference |
|:----------------|---------------:|--------------:|---------------:|--------------:|------------------:|
| Macro-F1        |         0.8309 |        0.0154 |         0.8304 |        0.017  |           -0.0006 |
| Micro-F1        |         0.9293 |        0.0067 |         0.9326 |        0.0065 |            0.0033 |
| Samples-F1      |         0.9369 |        0.0073 |         0.9402 |        0.0066 |            0.0032 |
| Exact Match     |         0.8179 |        0.0182 |         0.8275 |        0.0147 |            0.0096 |
| Hamming Loss ↓  |         0.0207 |        0.0021 |         0.0197 |        0.002  |           -0.001  |
| Avg pred labels |         1.4606 |        0.032  |         1.4502 |        0.0319 |           -0.0104 |
| Objective score |       nan      |      nan      |         0.8283 |        0.0075 |          nan      |

## E. Stability definitions

| Category | Definition |
|---|---|
| High | Selected in at least 15 out of 20 splits |
| Moderate | Selected in 10 to 14 out of 20 splits |
| Low | Selected in fewer than 10 out of 20 splits |

## F. Stability tables

### LATS-v1

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

### LATS-v2

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
