# v0.9_4 Results Summary — LATS-v2 Metric-Aware Coordinate Search

## Final decision

LATS-v2 is the new recommended v0.9_4 result.

## Global comparison

| Method                      |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact Match |   Hamming Loss ↓ |   Avg pred labels |
|:----------------------------|-----------:|-----------:|-------------:|--------------:|-----------------:|------------------:|
| v0.8 mean-all baseline      |     0.7801 |     0.9332 |       0.9406 |        0.8397 |           0.0194 |            1.4302 |
| v0.8 simple event-max       |     0.832  |     0.9285 |       0.9375 |        0.8235 |           0.0211 |            1.4844 |
| v0.9 frozen frequency map   |     0.8512 |     0.9372 |       0.9482 |        0.842  |           0.0185 |            1.4694 |
| v0.9 frozen + Gary mean     |     0.8518 |     0.9374 |       0.9464 |        0.8431 |           0.0183 |            1.4614 |
| LATS-v1 final frozen config |     0.8667 |     0.9436 |       0.9495 |        0.8524 |           0.0165 |            1.4544 |
| LATS-v2 metric-aware config |     0.8673 |     0.9458 |       0.9517 |        0.8604 |           0.0158 |            1.4452 |

## LATS-v1 vs LATS-v2 difference

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

## Pair changes

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

## Stability tables

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

## Conclusion

LATS-v2 should replace LATS-v1 as the best v0.9 result because it improves Macro-F1, Micro-F1, Samples-F1, Exact Match, Hamming Loss, and Jaccard on the corrected holdout. The result is especially meaningful because the base model is unchanged.
