# Experiment Setup — v0.9_4 LATS-v2 Metric-Aware Coordinate Search

## Experiment identity

| Item | Value |
|---|---|
| Branch | `agentic_data_preprocessing_v0.9_4` |
| Parent branch | `agentic_data_preprocessing_v0.9_3` |
| Experiment | LATS-v2 metric-aware coordinate search |
| Base model | `main_v08_human_corrected_balanced_3exit_20260610_084027` |
| Training status | No retraining |
| Input | Frozen segment-level probabilities |
| Segment prediction CSV | `human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv` |
| Start config | `human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v09_search\lats_final_frozen_config.json` |
| Output directory | `human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v2_metric_coordinate_search` |
| Parent clips | 867 |
| Segments | 4,335 |
| Labels | 10 |

## Objective

LATS-v2 optimises the full multi-label objective instead of selecting labels independently:

```text
Score =
  0.40 * Macro-F1
+ 0.20 * Micro-F1
+ 0.20 * Samples-F1
+ 0.15 * Exact Match
- 0.05 * Hamming Loss
- 0.05 * |avg_pred_labels - avg_true_labels|
```

## Search space

| Component | Values |
|---|---|
| Aggregation methods | `mean`, `max`, `top2mean`, `top3mean`, `top4mean`, `top5mean`, `median`, `p75`, `p90`, `noisy_or` |
| Threshold range | 0.10 to 0.95 |
| Threshold step | 0.01 |
| Repeated splits | 20 |
| Calibration fraction | 0.5 |
| Maximum coordinate iterations | 5 |

## Coordinate-search rule

```text
Start from LATS-v1 frozen config.
For each seed:
    split parent clips into calibration and evaluation subsets
    for each coordinate iteration:
        for each label:
            test candidate aggregation-threshold pairs
            accept candidate only if full multi-label objective improves
Freeze final configuration by method frequency and median threshold.
Evaluate frozen configuration on full corrected holdout.
```

## Final result

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
