# Agentic Data Preprocessing v0.9 Tables

This folder is the cleaned table/result hierarchy for the v0.9 line. Files are separated by experiment stage so that baseline, LATS-v1, LATS-v2, and comparison artefacts do not mix at the parent level.

## Folder layout

| Folder | Purpose |
|---|---|
| `v0.9_baselines/` | v0.8/v0.9 baseline parent aggregation results, mapping-bank outputs, repeated calibration summaries, and older threshold diagnostics. |
| `v0.9_lats_v1/` | LATS-v1 repeated split outputs, frozen configuration, final holdout metrics, per-label metrics, and LATS-v1 stability tables. |
| `v0.9_lats_v2/` | LATS-v2 metric-aware coordinate search outputs, final frozen config, full-holdout metrics, per-label metrics, coordinate-search logs, and research LaTeX. |
| `v0.9_comparisons/` | Direct LATS-v1 vs LATS-v2 comparison tables: global metrics, pair changes, label-wise changes, repeated split comparison, and stability comparison. |

## Current final result

The recommended v0.9_4 result is **LATS-v2 metric-aware coordinate search**.

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss ↓ |
|---|---:|---:|---:|---:|---:|
| LATS-v1 final frozen config | 0.8667 | 0.9436 | 0.9495 | 0.8524 | 0.0165 |
| **LATS-v2 metric-aware config** | **0.8673** | **0.9458** | **0.9517** | **0.8604** | **0.0158** |

## Important files

- `v0.9_lats_v2/lats_v2_final_full_holdout_eval.csv`
- `v0.9_lats_v2/lats_v2_final_full_holdout_per_label.csv`
- `v0.9_lats_v2/lats_v2_final_frozen_config.json`
- `v0.9_lats_v2/v09_lats_v2_research_findings_latex.tex`
- `v0.9_comparisons/v09_lats_v2_global_result_comparison.md`
- `v0.9_comparisons/v09_lats_v2_pair_changes.md`
- `v0.9_comparisons/v09_lats_v2_labelwise_changes.md`
- `v0.9_comparisons/v09_lats_v1_vs_v2_stability.md`

## Reporting caution

The corrected holdout is not an independent external test set. The repeated split results should be used as stability evidence, while the full-holdout evaluation should be reported as a frozen corrected-holdout result.
