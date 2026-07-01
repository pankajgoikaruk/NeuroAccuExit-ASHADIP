# Final `docs/` Structure

This is the cleaned documentation tree for the ASHADIP / NeuroAccuExit Agentic Data Preprocessing work.

## Top-level folders

| Path | Purpose |
|---|---|
| `docs/v0.9/` | Versioned v0.9 experiment documentation. |
| `docs/v0.9/v0.9_4/` | Current LATS-v2 experiment setup, commands, appendix, and experiment log. |
| `docs/reports/v0.9/` | Narrative experiment reports for v0.9_2, v0.9_3, and v0.9_4. |
| `docs/results/v0.9/` | Result summaries for v0.9_2, v0.9_3, and v0.9_4. |
| `docs/tables/agentic_data_preprocessing_v0.9/` | Cleaned v0.9 table hierarchy. |
| `docs/figures/` | Figures from earlier experiment versions. |
| `docs/releases/` | Release-note documents. |

## Cleaned v0.9 table hierarchy

```text
docs/tables/agentic_data_preprocessing_v0.9/
├── README.md
├── INDEX.md
├── FILE_MANIFEST.csv
├── FILE_MANIFEST.md
├── v09_final_full_holdout_comparison.md
├── v0.9_baselines/
├── v0.9_lats_v1/
├── v0.9_lats_v2/
└── v0.9_comparisons/
```

## Removed duplicate/problem folders

The cleaned package removes these accidental folders:

```text
docs/v0.9/v0.9_4-
docs/v0.9/v0.9_4--
docs/tables/agentic_data_preprocessing_v0.9_
```

Their useful v0.9_4 content has been consolidated into:

```text
docs/v0.9/v0.9_4/
docs/tables/agentic_data_preprocessing_v0.9/v0.9_lats_v2/
docs/tables/agentic_data_preprocessing_v0.9/v0.9_comparisons/
```

## Current final recommendation

Use **LATS-v2 metric-aware coordinate search** as the v0.9_4 final documented result.
