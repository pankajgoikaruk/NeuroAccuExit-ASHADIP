# v0.9_3 Results Summary — LATS Final Frozen Config

## Experiment identity

```text
Branch: agentic_data_preprocessing_v0.9
Subbranch: agentic_data_preprocessing_v0.9_3
Experiment family: label-wise parent aggregation and threshold search
Final method: lats_final_frozen_config_v09
Base model: main_v08_human_corrected_balanced_3exit_20260610_084027
Training: no retraining in v0.9_3
Evaluation: corrected holdout, parent/clip level
Parent clips: 867
Segments: 4,335
Labels: 10
Search methods: mean, max, top2mean, top3mean
Threshold search: 0.10 to 0.95, step 0.01
Repeated splits: 20
Calibration fraction: 0.5
```

---

## Research questions

This v0.9_3 sub-branch freezes the novelty work around the following research questions:

| ID | Research question | Answer from current evidence |
|---|---|---|
| RQ1 | Can parent-level multi-label audio performance be improved without retraining the base model? | Yes. LATS improves Macro-F1 from 0.7801 to 0.8667 using frozen segment probabilities. |
| RQ2 | Is one global parent aggregation rule enough for all labels? | No. The final rules use `mean`, `max`, `top2mean`, and `top3mean` depending on the label. |
| RQ3 | Can joint label-wise aggregation and threshold search outperform manually/frequency-selected maps? | Yes. LATS improves over `v09_frozen_frequency_plus_gary_mean` on Macro-F1, Micro-F1, Samples-F1, Exact Match, and Hamming Loss. |
| RQ4 | Which labels require non-mean aggregation? | `Brene_Brown`, `Eckhart_Tolle`, `Gary_Vee`, `other_speaker_present`, `audience_reaction_present`, and `silence_present` use non-mean aggregation in the final LATS config. |
| RQ5 | Is the final method fully model-training dependent? | No. The novelty is an inference-time decision layer over frozen model probabilities. |


---

## Final result

| Method                        |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact Match |   Hamming Loss ↓ |   Avg pred labels |
|:------------------------------|-----------:|-----------:|-------------:|--------------:|-----------------:|------------------:|
| v0.8 mean-all baseline        |     0.7801 |     0.9332 |       0.9406 |        0.8397 |           0.0194 |            1.4302 |
| v0.8 simple event-max         |     0.832  |     0.9285 |       0.9375 |        0.8235 |           0.0211 |            1.4844 |
| v0.9 frozen frequency map     |     0.8512 |     0.9372 |       0.9482 |        0.842  |           0.0185 |            1.4694 |
| v0.9 frozen + Gary mean       |     0.8518 |     0.9374 |       0.9464 |        0.8431 |           0.0183 |            1.4614 |
| LATS-v0.9 final frozen config |     0.8667 |     0.9436 |       0.9495 |        0.8524 |           0.0165 |            1.4544 |

## Improvement summary

| Metric         |   vs mean-all baseline |   vs previous v0.9 best |
|:---------------|-----------------------:|------------------------:|
| Macro-F1       |                 0.0866 |                  0.0149 |
| Micro-F1       |                 0.0104 |                  0.0062 |
| Samples-F1     |                 0.0089 |                  0.0031 |
| Exact Match    |                 0.0127 |                  0.0092 |
| Hamming Loss ↓ |                -0.0029 |                 -0.0018 |

Final frozen result:

```text
Method     = lats_final_frozen_config_v09
Macro-F1   = 0.8667
Micro-F1   = 0.9436
Samples-F1 = 0.9495
Exact      = 0.8524
Hamming    = 0.0165
```

---

## Final LATS configuration

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

## Repeated-split standard deviation table

| Metric               |   Mean |    Std |    Min |    Max |
|:---------------------|-------:|-------:|-------:|-------:|
| Macro-F1             | 0.8309 | 0.0154 | 0.8082 | 0.8593 |
| Micro-F1             | 0.9293 | 0.0067 | 0.9193 | 0.9431 |
| Samples-F1           | 0.9369 | 0.0073 | 0.9273 | 0.9532 |
| Exact Match          | 0.8179 | 0.0182 | 0.7875 | 0.8499 |
| Hamming Loss ↓       | 0.0207 | 0.0021 | 0.0166 | 0.024  |
| Avg predicted labels | 1.4606 | 0.032  | 1.3972 | 1.5381 |

---

## Final per-label metrics

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


---

## Repeated-seed label-wise stability

To avoid relying on a single calibration split, LATS-v0.9 was repeated across **20 random calibration/evaluation splits**. For each label, the aggregation method and threshold were selected on the calibration parents. The table below reports how stable the final selected rule was across seeds.

| Label                       | Most selected aggregation   | Selection count   |   Selection fraction |   Final threshold |   Mean threshold |   Median threshold |   Std threshold |
|:----------------------------|:----------------------------|:------------------|---------------------:|------------------:|-----------------:|-------------------:|----------------:|
| `Brene_Brown`               | `top3mean`                  | 8/20              |                 0.4  |              0.5  |           0.5    |              0.5   |          0      |
| `Eckhart_Tolle`             | `top2mean`                  | 19/20             |                 0.95 |              0.5  |           0.5    |              0.5   |          0      |
| `Eric_Thomas`               | `mean`                      | 10/20             |                 0.5  |              0.53 |           0.52   |              0.53  |          0.0226 |
| `Gary_Vee`                  | `top3mean`                  | 11/20             |                 0.55 |              0.5  |           0.51   |              0.5   |          0.0341 |
| `Jay_Shetty`                | `mean`                      | 14/20             |                 0.7  |              0.8  |           0.7407 |              0.805 |          0.1508 |
| `Nick_Vujicic`              | `mean`                      | 10/20             |                 0.5  |              0.43 |           0.458  |              0.43  |          0.0361 |
| `other_speaker_present`     | `top3mean`                  | 10/20             |                 0.5  |              0.76 |           0.769  |              0.76  |          0.0233 |
| `music_present`             | `mean`                      | 14/20             |                 0.7  |              0.49 |           0.5057 |              0.49  |          0.0454 |
| `audience_reaction_present` | `max`                       | 11/20             |                 0.55 |              0.68 |           0.6845 |              0.68  |          0.068  |
| `silence_present`           | `top2mean`                  | 15/20             |                 0.75 |              0.38 |           0.36   |              0.38  |          0.052  |

Interpretation:

- `Eckhart_Tolle` and `silence_present` show strong aggregation stability.
- `Jay_Shetty` and `music_present` also show good stability, both selecting `mean` in 14/20 seeds.
- Some labels, such as `Brene_Brown`, `Eric_Thomas`, `Nick_Vujicic`, and `other_speaker_present`, have moderate aggregation stability; this should be reported transparently.
- The final frozen LATS configuration was selected using the most frequent aggregation method per label and the median threshold for that selected method.

---

## Final decision

```text
Freeze v0.9_3 with LATS-v0.9 as the final novelty method.
```

The previous best `v09_frozen_frequency_plus_gary_mean` remains useful as an intermediate ablation, but the final method is now `lats_final_frozen_config_v09`.
