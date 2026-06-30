# Appendix — v0.9_3 LATS Final Frozen Result

## A. Scope

```text
Branch: agentic_data_preprocessing_v0.9
Subbranch: agentic_data_preprocessing_v0.9_3
Base model: main_v08_human_corrected_balanced_3exit_20260610_084027
Training: no retraining
Final method: lats_final_frozen_config_v09
```

---

## B. Mathematical formulation

Let a parent clip contain `S` segments. For label `l`, the frozen model produces probabilities:

```text
p_1(l), p_2(l), ..., p_S(l)
```

LATS chooses one aggregation function `A_l` and one threshold `τ_l` per label:

```text
p_parent(l) = A_l(p_1(l), ..., p_S(l))
y_hat(l) = 1 if p_parent(l) >= τ_l else 0
```

The final decision rule is therefore label-specific:

```text
(A_l, τ_l)
```

---

## C. Final LATS configuration

| Label                     | Aggregation   |   Threshold |   Selected count |   Selection fraction |
|:--------------------------|:--------------|------------:|-----------------:|---------------------:|
| Brene_Brown               | top3mean      |        0.5  |                8 |                 0.4  |
| Eckhart_Tolle             | top2mean      |        0.5  |               19 |                 0.95 |
| Eric_Thomas               | mean          |        0.53 |               10 |                 0.5  |
| Gary_Vee                  | top3mean      |        0.5  |               11 |                 0.55 |
| Jay_Shetty                | mean          |        0.8  |               14 |                 0.7  |
| Nick_Vujicic              | mean          |        0.43 |               10 |                 0.5  |
| other_speaker_present     | top3mean      |        0.76 |               10 |                 0.5  |
| music_present             | mean          |        0.49 |               14 |                 0.7  |
| audience_reaction_present | max           |        0.68 |               11 |                 0.55 |
| silence_present           | top2mean      |        0.38 |               15 |                 0.75 |

---

## D. Final full-holdout comparison

| Method                        |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact Match |   Hamming Loss ↓ |   Avg pred labels |
|:------------------------------|-----------:|-----------:|-------------:|--------------:|-----------------:|------------------:|
| v0.8 mean-all baseline        |     0.7801 |     0.9332 |       0.9406 |        0.8397 |           0.0194 |            1.4302 |
| v0.8 simple event-max         |     0.832  |     0.9285 |       0.9375 |        0.8235 |           0.0211 |            1.4844 |
| v0.9 frozen frequency map     |     0.8512 |     0.9372 |       0.9482 |        0.842  |           0.0185 |            1.4694 |
| v0.9 frozen + Gary mean       |     0.8518 |     0.9374 |       0.9464 |        0.8431 |           0.0183 |            1.4614 |
| LATS-v0.9 final frozen config |     0.8667 |     0.9436 |       0.9495 |        0.8524 |           0.0165 |            1.4544 |

---

## E. Improvement summary

| Metric         |   vs mean-all baseline |   vs previous v0.9 best |
|:---------------|-----------------------:|------------------------:|
| Macro-F1       |                 0.0866 |                  0.0149 |
| Micro-F1       |                 0.0104 |                  0.0062 |
| Samples-F1     |                 0.0089 |                  0.0031 |
| Exact Match    |                 0.0127 |                  0.0092 |
| Hamming Loss ↓ |                -0.0029 |                 -0.0018 |

---

## F. Repeated split standard deviation table

| Metric               |   Mean |    Std |    Min |    Max |
|:---------------------|-------:|-------:|-------:|-------:|
| Macro-F1             | 0.8309 | 0.0154 | 0.8082 | 0.8593 |
| Micro-F1             | 0.9293 | 0.0067 | 0.9193 | 0.9431 |
| Samples-F1           | 0.9369 | 0.0073 | 0.9273 | 0.9532 |
| Exact Match          | 0.8179 | 0.0182 | 0.7875 | 0.8499 |
| Hamming Loss ↓       | 0.0207 | 0.0021 | 0.0166 | 0.024  |
| Avg predicted labels | 1.4606 | 0.032  | 1.3972 | 1.5381 |

---

## G. Final per-label metrics

| Label                     | Aggregation   |   Threshold |   Precision |   Recall |     F1 |   Support |   Pred + |   Errors |
|:--------------------------|:--------------|------------:|------------:|---------:|-------:|----------:|---------:|---------:|
| Brene_Brown               | top3mean      |        0.5  |      1      |   0.9589 | 0.979  |        73 |       70 |        3 |
| Eckhart_Tolle             | top2mean      |        0.5  |      1      |   1      | 1      |        84 |       84 |        0 |
| Eric_Thomas               | mean          |        0.53 |      0.9275 |   0.9412 | 0.9343 |        68 |       69 |        9 |
| Gary_Vee                  | top3mean      |        0.5  |      0.9853 |   0.9853 | 0.9853 |        68 |       68 |        2 |
| Jay_Shetty                | mean          |        0.8  |      0.9565 |   0.9778 | 0.967  |        90 |       92 |        6 |
| Nick_Vujicic              | mean          |        0.43 |      1      |   0.9796 | 0.9897 |        49 |       48 |        1 |
| other_speaker_present     | top3mean      |        0.76 |      0.9475 |   0.9413 | 0.9444 |       460 |      457 |       51 |
| music_present             | mean          |        0.49 |      0.9642 |   0.9472 | 0.9556 |       341 |      335 |       30 |
| audience_reaction_present | max           |        0.68 |      0.8235 |   0.4828 | 0.6087 |        29 |       17 |       18 |
| silence_present           | top2mean      |        0.38 |      0.2381 |   0.4167 | 0.303  |        12 |       21 |       23 |

---

## H. Selection stability

| Label                     | Aggregation   |   Threshold |   Selected count |   Selection fraction |   Threshold mean |   Threshold std | Method counts                                        |
|:--------------------------|:--------------|------------:|-----------------:|---------------------:|-----------------:|----------------:|:-----------------------------------------------------|
| Brene_Brown               | top3mean      |        0.5  |                8 |                 0.4  |           0.5    |          0      | {"max": 8, "mean": 2, "top2mean": 2, "top3mean": 8}  |
| Eckhart_Tolle             | top2mean      |        0.5  |               19 |                 0.95 |           0.5    |          0      | {"mean": 1, "top2mean": 19}                          |
| Eric_Thomas               | mean          |        0.53 |               10 |                 0.5  |           0.52   |          0.0226 | {"mean": 10, "top2mean": 4, "top3mean": 6}           |
| Gary_Vee                  | top3mean      |        0.5  |               11 |                 0.55 |           0.51   |          0.0341 | {"mean": 3, "top2mean": 6, "top3mean": 11}           |
| Jay_Shetty                | mean          |        0.8  |               14 |                 0.7  |           0.7407 |          0.1508 | {"mean": 14, "top3mean": 6}                          |
| Nick_Vujicic              | mean          |        0.43 |               10 |                 0.5  |           0.458  |          0.0361 | {"mean": 10, "top3mean": 10}                         |
| other_speaker_present     | top3mean      |        0.76 |               10 |                 0.5  |           0.769  |          0.0233 | {"mean": 6, "top2mean": 4, "top3mean": 10}           |
| music_present             | mean          |        0.49 |               14 |                 0.7  |           0.5057 |          0.0454 | {"max": 2, "mean": 14, "top2mean": 3, "top3mean": 1} |
| audience_reaction_present | max           |        0.68 |               11 |                 0.55 |           0.6845 |          0.068  | {"max": 11, "mean": 3, "top2mean": 3, "top3mean": 3} |
| silence_present           | top2mean      |        0.38 |               15 |                 0.75 |           0.36   |          0.052  | {"max": 2, "top2mean": 15, "top3mean": 3}            |

---

## I. Final statement for thesis/report

```text
The v0.9_3 sub-branch freezes LATS-v0.9 as an inference-time optimisation layer. 
The method performs joint label-wise parent aggregation and threshold search over frozen segment probabilities, without retraining the base audio model. 
The final frozen configuration improves corrected-holdout Macro-F1 from 0.7801 to 0.8667, Micro-F1 from 0.9332 to 0.9436, Exact Match from 0.8397 to 0.8524, and reduces Hamming Loss from 0.0194 to 0.0165.
```
