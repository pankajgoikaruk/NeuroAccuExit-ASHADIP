# Agentic v0.9_3 Guide — LATS Label-wise Aggregation and Threshold Search

This guide explains the final frozen v0.9_3 method: **LATS-v0.9**.

LATS uses frozen segment-level probabilities from the v0.8-HCB audio model and learns a parent-level inference rule per label. It does not retrain the model.

---

## 1. Core pipeline

```text
segment-level probabilities
        ↓
parent-level aggregation per label
        ↓
label-specific threshold
        ↓
parent-level multi-label prediction
```

---

## 2. Research questions

This v0.9_3 sub-branch freezes the novelty work around the following research questions:

| ID | Research question | Answer from current evidence |
|---|---|---|
| RQ1 | Can parent-level multi-label audio performance be improved without retraining the base model? | Yes. LATS improves Macro-F1 from 0.7801 to 0.8667 using frozen segment probabilities. |
| RQ2 | Is one global parent aggregation rule enough for all labels? | No. The final rules use `mean`, `max`, `top2mean`, and `top3mean` depending on the label. |
| RQ3 | Can joint label-wise aggregation and threshold search outperform manually/frequency-selected maps? | Yes. LATS improves over `v09_frozen_frequency_plus_gary_mean` on Macro-F1, Micro-F1, Samples-F1, Exact Match, and Hamming Loss. |
| RQ4 | Which labels require non-mean aggregation? | `Brene_Brown`, `Eckhart_Tolle`, `Gary_Vee`, `other_speaker_present`, `audience_reaction_present`, and `silence_present` use non-mean aggregation in the final LATS config. |
| RQ5 | Is the final method fully model-training dependent? | No. The novelty is an inference-time decision layer over frozen model probabilities. |


---

## 3. Inputs

Main input CSV:

```text
human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv
```

The CSV contains:

```text
parent_clip_id
<label columns>
exit3_prob_<label columns>
```

Parent-level truth is computed as:

```python
parent_true[label] = max(segment_true[label])
```

---

## 4. Aggregation functions

| Aggregation | Meaning | Role |
|---|---|---|
| `mean` | Average across all parent segments | Stable label evidence |
| `max` | Maximum segment probability | Very short/bursty evidence |
| `top2mean` | Average of top 2 segment probabilities | Sparse evidence without full max aggression |
| `top3mean` | Average of top 3 segment probabilities | Smoother sparse evidence |

---

## 5. Search strategy

For each label:

```text
for aggregation in [mean, max, top2mean, top3mean]:
    for threshold in [0.10, 0.11, ..., 0.95]:
        evaluate label F1 on calibration parents
        keep best aggregation + threshold
```

Tie-breaking is conservative:

```text
1. Higher F1
2. Lower hamming errors
3. Fewer false positives
4. Threshold closer to 0.5
5. Simpler aggregation preference
```

---

## 6. Final frozen configuration

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

The final config is saved in:

```text
human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v09_search\lats_final_frozen_config.json
```

It has also been added to:

```text
configs\v0.9\labelwise_aggregation_maps.json
```

as:

```text
lats_final_frozen_config_v09
```

---

## 7. Final result

| Method                        |   Macro-F1 |   Micro-F1 |   Samples-F1 |   Exact Match |   Hamming Loss ↓ |   Avg pred labels |
|:------------------------------|-----------:|-----------:|-------------:|--------------:|-----------------:|------------------:|
| v0.8 mean-all baseline        |     0.7801 |     0.9332 |       0.9406 |        0.8397 |           0.0194 |            1.4302 |
| v0.8 simple event-max         |     0.832  |     0.9285 |       0.9375 |        0.8235 |           0.0211 |            1.4844 |
| v0.9 frozen frequency map     |     0.8512 |     0.9372 |       0.9482 |        0.842  |           0.0185 |            1.4694 |
| v0.9 frozen + Gary mean       |     0.8518 |     0.9374 |       0.9464 |        0.8431 |           0.0183 |            1.4614 |
| LATS-v0.9 final frozen config |     0.8667 |     0.9436 |       0.9495 |        0.8524 |           0.0165 |            1.4544 |

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

## 8. Repeated split standard deviation table

| Metric               |   Mean |    Std |    Min |    Max |
|:---------------------|-------:|-------:|-------:|-------:|
| Macro-F1             | 0.8309 | 0.0154 | 0.8082 | 0.8593 |
| Micro-F1             | 0.9293 | 0.0067 | 0.9193 | 0.9431 |
| Samples-F1           | 0.9369 | 0.0073 | 0.9273 | 0.9532 |
| Exact Match          | 0.8179 | 0.0182 | 0.7875 | 0.8499 |
| Hamming Loss ↓       | 0.0207 | 0.0021 | 0.0166 | 0.024  |
| Avg predicted labels | 1.4606 | 0.032  | 1.3972 | 1.5381 |

---

## 9. Recommended reporting text

```text
We introduce LATS-v0.9, a label-wise aggregation and threshold-search procedure for parent-level multi-label audio prediction. 
Using frozen segment probabilities from the v0.8-HCB model, LATS selects a parent aggregation function and decision threshold independently for each label across repeated calibration splits. 
The final frozen configuration improves corrected-holdout Macro-F1 from 0.7801 to 0.8667 without retraining the base model.
```
