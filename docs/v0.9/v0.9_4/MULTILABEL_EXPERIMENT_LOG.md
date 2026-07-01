# Multilabel Experiment Log — v0.9_4 LATS-v2

## Entry: LATS-v2 metric-aware coordinate search

| Item | Value |
|---|---|
| Date | 2026-07-01 |
| Branch | `agentic_data_preprocessing_v0.9_4` |
| Parent branch | `agentic_data_preprocessing_v0.9_3` |
| Base model | `main_v08_human_corrected_balanced_3exit_20260610_084027` |
| Training | No retraining |
| Parent clips | 867 |
| Segments | 4,335 |
| Start point | LATS-v1 final frozen config |
| New method | LATS-v2 metric-aware coordinate search |

## Result

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

## Research finding

LATS-v2 shows that full multi-label objective optimisation can improve global prediction-set quality even when an individual rare-label F1 score slightly decreases. The `silence_present` rule is the key example: its F1 decreases slightly, but false positives decrease and the overall Exact Match and Hamming Loss improve.
