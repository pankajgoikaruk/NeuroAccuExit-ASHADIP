# v0.10_1 Low-Energy Recovery Ablation: Results and Research Findings

Branch: `agentic_data_preprocessing_v0.10_1`

Experiment variant:

```text
main_v010_1_no_hint_low_energy_augmented_20260706_215519
```

This experiment tested whether adding manually reviewed TATA-LAWYER low-energy one-second samples improves the current NeuroAccuExit human-talk model.

## Research question

> Does adding linked, manually reviewed low-energy one-second evidence improve difficult labels such as `silence_present` and `audience_reaction_present` without hurting global multi-label consistency?

## Data and manifest construction

The v0.10_1 build followed a non-destructive manifest strategy:

1. Read original manifests only.
2. Copy them into a new v0.10_1 workspace.
3. Use manually labelled TATA-LAWYER low-energy one-second samples.
4. Link each low-energy sample back to `parent_clip_id`.
5. Build a new augmented manifest.
6. Train/evaluate from the copied augmented manifest only.

Feature usage:

```text
Original ASHADIP rows          -> existing ASHADIP .npy features
TATA-LAWYER low-energy rows    -> existing TATA-LAWYER low_energy_*.npy features
Training manifest              -> copied v0.10_1 augmented manifest
```

## Build summary

```text
base_rows                                      = 29363
reviewed_initial_rows                          = 1018
split_filtered_rows                            = 317
partial_mask_filtered_rows                     = 34
holdout_parent_overlap_rows_detected           = 0
selected_low_energy_rows_appended_before_dedupe = 667
duplicates_removed_after_concat                = 0
final_rows                                     = 30030
final_low_energy_added_rows                    = 667
feature_resolution_mode                        = feat_relpath
feature_source_used_counts                     = feat_relpath: 667
```

The build was valid from a leakage and feature-linking perspective: no corrected-holdout parent overlap was detected, and the low-energy rows were resolved through `feat_relpath` rather than legacy `feature_path` rows.

## Training setup

```text
Model variant  = main_v010_1_no_hint_low_energy_augmented
Exit hint      = disabled
Pos weight     = disabled
Epochs         = 40
Batch size     = 64
Learning rate  = 0.001
Device         = CPU
Tap blocks     = 1,3
Seed           = 42
```

## Fixed 0.5 parent-level holdout result

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0935 | 0.3027 | 0.1908 | 0.0150 | 0.1302 |
| 2 | 0.5227 | 0.6917 | 0.5813 | 0.4187 | 0.0719 |
| 3 | 0.7556 | 0.9219 | 0.9245 | 0.8212 | 0.0219 |

## LATS-v2 coordinate re-optimized result

```text
Macro-F1      = 0.858125
Micro-F1      = 0.944644
Samples-F1    = 0.951920
Exact Match   = 0.856978
Hamming Loss  = 0.016032
Parent clips  = 867
Avg true labels = 1.469435
Avg pred labels = 1.426759
```

## Comparison with selected v0.10 no-hint result

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Selected v0.10 no-hint + LATS-v2 | **0.8624** | **0.9531** | **0.9589** | **0.8766** | **0.0137** |
| v0.10_1 low-energy recovery + LATS-v2 | 0.8581 | 0.9446 | 0.9519 | 0.8570 | 0.0160 |
| Difference | -0.0043 | -0.0085 | -0.0070 | -0.0196 | +0.0023 |

Lower Hamming Loss is better, so v0.10_1 is worse on every final parent-level metric.

## Research finding

The low-energy recovery augmentation did **not** improve the final corrected-holdout parent-level result. Although it safely added 667 manually reviewed low-energy examples and targeted difficult labels, the final LATS-v2 result underperformed the selected v0.10 no-hint configuration.

## Interpretation

The likely explanation is that the recovered low-energy samples provided useful local evidence for silence/audience-like phenomena, but they also shifted the training distribution and slightly disturbed global multi-label calibration. As a result, Exact Match and Hamming Loss degraded, which is important for parent-level multi-label consistency.

## Decision

```text
Do not promote v0.10_1 as the final model.
Keep selected v0.10 no-hint + LATS-v2 as the current best outcome.
Document v0.10_1 as a valid negative/diagnostic ablation.
```

## Key points for paper/reporting

- The v0.10_1 pipeline is methodologically useful because it tests whether recovered low-energy evidence can improve bottleneck labels.
- The experiment is non-destructive: original ASHADIP and TATA-LAWYER manifests are not modified.
- The corrected holdout remains unchanged, enabling fair comparison.
- The negative result is informative: simply adding recovered low-energy rows is not enough; future work may need label-aware weighting, separate calibration, or a masked-loss training path.

## Reproducibility command used locally

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v0.10_1\run_v010_1_no_hint_low_energy_ablation_LOCAL_PATCH.ps1 `
  -LowEnergyMaskedManifest "C:\Users\wwwsa\PycharmProjects\TATA-LAWYER\human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\human_reviewed_masked_v09\feature_cache\metadata\multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv" `
  -LowEnergyFeaturesRoot "C:\Users\wwwsa\PycharmProjects\TATA-LAWYER\human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\feature_cache\features" `
  -LowEnergyFeatureMode feat_relpath `
  -Device cpu `
  -Epochs 40 `
  -Objective macro_priority
```
