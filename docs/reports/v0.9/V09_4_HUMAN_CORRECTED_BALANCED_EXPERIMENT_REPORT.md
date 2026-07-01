# v0.9_4 Human-Corrected Balanced Experiment Report

This report updates the human-corrected balanced v0.9 line with LATS-v2 results. The underlying neural model remains unchanged: `main_v08_human_corrected_balanced_3exit_20260610_084027`. The v0.9_4 contribution is an inference-time optimisation layer.

## Final result comparison

| Method                      |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact Match |   Hamming Loss ↓ |   Avg pred labels |
|:----------------------------|-----------:|-----------:|-------------:|--------------:|-----------------:|------------------:|
| v0.8 mean-all baseline      |     0.7801 |     0.9332 |       0.9406 |        0.8397 |           0.0194 |            1.4302 |
| v0.8 simple event-max       |     0.832  |     0.9285 |       0.9375 |        0.8235 |           0.0211 |            1.4844 |
| v0.9 frozen frequency map   |     0.8512 |     0.9372 |       0.9482 |        0.842  |           0.0185 |            1.4694 |
| v0.9 frozen + Gary mean     |     0.8518 |     0.9374 |       0.9464 |        0.8431 |           0.0183 |            1.4614 |
| LATS-v1 final frozen config |     0.8667 |     0.9436 |       0.9495 |        0.8524 |           0.0165 |            1.4544 |
| LATS-v2 metric-aware config |     0.8673 |     0.9458 |       0.9517 |        0.8604 |           0.0158 |            1.4452 |

## Final LATS-v2 configuration

| Label                     | Aggregation   |   Threshold |
|:--------------------------|:--------------|------------:|
| Brene_Brown               | top3mean      |        0.5  |
| Eckhart_Tolle             | top2mean      |        0.5  |
| Eric_Thomas               | mean          |        0.54 |
| Gary_Vee                  | top3mean      |        0.5  |
| Jay_Shetty                | mean          |        0.82 |
| Nick_Vujicic              | mean          |        0.43 |
| other_speaker_present     | top3mean      |        0.76 |
| music_present             | mean          |        0.49 |
| audience_reaction_present | max           |        0.68 |
| silence_present           | p75           |        0.34 |

## Per-label final performance

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

## Main interpretation

LATS-v2 improves the global parent-level multi-label result without retraining. The strongest global improvement is Exact Match, which increases from 0.8524 to 0.8604 compared with LATS-v1. Hamming Loss decreases from 0.0165 to 0.0158. These changes indicate that LATS-v2 produces a cleaner complete prediction vector.

## Reporting caution

The corrected holdout is not an independent external test set. The repeated split results should be used as stability evidence, while the final full-holdout result should be reported as a frozen corrected-holdout evaluation.
