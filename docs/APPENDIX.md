# Appendix — agentic_data_preprocessing_v0.6

This appendix contains reproducibility notes and commands for **`agentic_data_preprocessing_v0.6`**.

```text
Branch: agentic_data_preprocessing_v0.6
Goal: TATA-assisted human-in-the-loop raw pseudo-manifest generation
Recommended final model: main_v06_expanded_3exit, fixed threshold 0.5, parent-level mean aggregation
```

## A1. Branch setup

```powershell
git fetch origin
git switch agentic_data_preprocessing_v0.6
git pull origin agentic_data_preprocessing_v0.6
```

If creating from a previous branch:

```powershell
git switch -c agentic_data_preprocessing_v0.6
git push -u origin agentic_data_preprocessing_v0.6
```

## A2. v0.6 labels

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
audience_reaction_present
silence_present
```

## A3. Raw dataset split

```powershell
python scripts\split_tata_raw_dataset.py `
  --raw_root human_talk_dataset `
  --out_root human_talk_workspace	ata_v0.6_raw_pipeline `
  --holdout_frac 0.15 `
  --seed 42
```

Outputs:

```text
human_talk_workspace/tata_v0.6_raw_pipeline/metadata/raw_pseudo_pool_parent_manifest.csv
human_talk_workspace/tata_v0.6_raw_pipeline/metadata/raw_final_holdout_MANUAL_LABEL_TEMPLATE.csv
```

## A4. Build raw pseudo-pool segments

```powershell
$RawRoot = "human_talk_workspace	ata_v0.6_raw_pipeline"
$RawPseudoSegmentCache = "$RawRootaw_pseudo_pool_segment_cache"

python scriptsuild_tata_raw_pseudo_segments.py `
  --raw_parent_manifest "$RawRoot\metadataaw_pseudo_pool_parent_manifest.csv" `
  --out_dir "$RawPseudoSegmentCache" `
  --sample_rate 16000 `
  --segment_sec 1.0 `
  --hop_sec 1.0 `
  --include_tail
```

## A5. Extract raw pseudo-pool features

```powershell
$RawPseudoFeatureCache = "$RawRootaw_pseudo_pool_feature_cache"

python scripts\extract_multilabel_features.py `
  --manifest "$RawPseudoSegmentCache\metadataaw_pseudo_pool_segment_manifest.csv" `
  --labels_json "$RawPseudoSegmentCache\metadata	ata_v06_labels.json" `
  --out_cache "$RawPseudoFeatureCache" `
  --sample_rate 16000 `
  --clip_sec 1.0 `
  --n_mels 64 `
  --n_fft 1024 `
  --win_ms 25 `
  --hop_ms 10 `
  --cmvn
```

## A6. Run TATA raw inference and routing

```powershell
$TataRunDir = "human_talk_workspace	ata_v0.6_scratchuns	ata_v06_3exit_coarse_audience_scratch_retry_20260531_211047"
$RawRoutingOut = "$RawRootaw_tata_pseudo_routing"

python scripts\predict_tata_raw_pseudo_routing.py `
  --run_dir "$TataRunDir" `
  --features_manifest "$RawPseudoFeatureCache\metadata\multilabel_features_manifest.csv" `
  --features_root "$RawPseudoFeatureCacheeatures" `
  --out_dir "$RawRoutingOut" `
  --device cpu `
  --batch_size 128
```

Hybrid routing output used for final expansion:

```text
human_talk_workspace/tata_v0.6_raw_pipeline/raw_tata_pseudo_routing/hybrid/hybrid_accepted.csv
human_talk_workspace/tata_v0.6_raw_pipeline/raw_tata_pseudo_routing/hybrid/hybrid_accepted_with_warning.csv
human_talk_workspace/tata_v0.6_raw_pipeline/raw_tata_pseudo_routing/hybrid/hybrid_needs_review.csv
```

## A7. Manual correction

The raw `needs_review` file was manually corrected and saved as:

```text
human_talk_workspace/tata_v0.6_raw_pipeline/manual_review_queue/02_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv
```

The final holdout ground truth was manually labelled and saved as:

```text
human_talk_workspace/tata_v0.6_raw_pipeline/manual_review_queue/01_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv
```

## A8. Build final expanded training manifest

```powershell
$ScratchRoot = "human_talk_workspace	ata_v0.6_scratch"
$RawRoot = "human_talk_workspace	ata_v0.6_raw_pipeline"

python scriptsuild_tata_v06_final_expanded_training_manifest.py `
  --seed_feature_manifest "$ScratchRooteature_cache\metadata\multilabel_features_manifest.csv" `
  --seed_features_root "$ScratchRooteature_cacheeatures" `
  --raw_feature_manifest "$RawRootaw_pseudo_pool_feature_cache\metadata\multilabel_features_manifest.csv" `
  --raw_features_root "$RawRootaw_pseudo_pool_feature_cacheeatures" `
  --hybrid_accepted_csv "$RawRootaw_tata_pseudo_routing\hybrid\hybrid_accepted.csv" `
  --hybrid_warning_csv "$RawRootaw_tata_pseudo_routing\hybrid\hybrid_accepted_with_warning.csv" `
  --corrected_needs_review_csv "$RawRoot\manual_review_queue_raw_hybrid_needs_review_MANUAL_CORRECTION_FINAL_refreshed.csv" `
  --labels_json "$ScratchRoot\metadata	ata_v06_labels.json" `
  --out_root "$RawRootinal_expanded_training_dataset"
```

## A9. Train main 3-exit model

```powershell
powershell -ExecutionPolicy Bypass -File scriptsun_tata_weakclip_experiment.ps1 `
  -Manifest "$RawRootinal_expanded_training_dataset\metadata\multilabel_features_manifest.csv" `
  -FeaturesRoot "." `
  -LabelsJson "$RawRootinal_expanded_training_dataset\metadata	ata_v06_labels.json" `
  -WorkspaceRoot "$RawRoot\main_models" `
  -RunsRoot "$RawRoot\main_modelsuns" `
  -PackagesRoot "$RawRoot\main_models\packages" `
  -LogsRoot "$RawRoot\main_models\logs" `
  -Variant "main_v06_expanded_3exit" `
  -TapBlocks "1,3" `
  -Epochs 40 `
  -BatchSize 64 `
  -LogEvery 25 `
  -LR 0.001 `
  -Threshold 0.5 `
  -Device cpu
```

## A10. Train main 5-exit model

```powershell
powershell -ExecutionPolicy Bypass -File scriptsun_tata_weakclip_experiment.ps1 `
  -Manifest "$RawRootinal_expanded_training_dataset\metadata\multilabel_features_manifest.csv" `
  -FeaturesRoot "." `
  -LabelsJson "$RawRootinal_expanded_training_dataset\metadata	ata_v06_labels.json" `
  -WorkspaceRoot "$RawRoot\main_models" `
  -RunsRoot "$RawRoot\main_modelsuns" `
  -PackagesRoot "$RawRoot\main_models\packages" `
  -LogsRoot "$RawRoot\main_models\logs" `
  -Variant "main_v06_expanded_5exit" `
  -TapBlocks "1,2,3,4" `
  -Epochs 40 `
  -BatchSize 64 `
  -LogEvery 25 `
  -LR 0.001 `
  -Threshold 0.5 `
  -Device cpu
```

## A11. Threshold tuning and dynamic policy

```powershell
python scripts	une_multilabel_thresholds.py `
  --run_dir "$RunDir" `
  --device cpu

python scripts\multilabel_greedy_policy.py `
  --run_dir "$RunDir" `
  --threshold_mode tuned_per_exit `
  --device cpu
```

## A12. Build final holdout segments and features

```powershell
$HoldoutCsv = "$RawRoot\manual_review_queue_raw_final_holdout_GROUND_TRUTH_FINAL_refreshed.csv"
$HoldoutSegmentCache = "$RawRootinal_holdout_segment_cache"
$HoldoutFeatureCache = "$RawRootinal_holdout_feature_cache"

python scriptsuild_tata_holdout_segments.py `
  --holdout_csv "$HoldoutCsv" `
  --out_dir "$HoldoutSegmentCache" `
  --sample_rate 16000 `
  --segment_sec 1.0 `
  --hop_sec 1.0 `
  --include_tail

python scripts\extract_multilabel_features.py `
  --manifest "$HoldoutSegmentCache\metadatainal_holdout_segment_manifest.csv" `
  --labels_json "$HoldoutSegmentCache\metadata	ata_v06_labels.json" `
  --out_cache "$HoldoutFeatureCache" `
  --sample_rate 16000 `
  --clip_sec 1.0 `
  --n_mels 64 `
  --n_fft 1024 `
  --win_ms 25 `
  --hop_ms 10 `
  --cmvn
```

## A13. Segment-level final holdout evaluation

```powershell
$Run3 = "human_talk_workspace	ata_v0.6_raw_pipeline\main_modelsuns\main_v06_expanded_3exit_20260603_194435"
$HoldoutEvalRoot = "$RawRootinal_holdout_evaluation"

python scripts\evaluate_tata_final_holdout.py `
  --run_dir "$Run3" `
  --holdout_manifest "$HoldoutFeatureCache\metadata\multilabel_features_manifest.csv" `
  --features_root "$HoldoutFeatureCacheeatures" `
  --out_dir "$HoldoutEvalRoot\main_v06_expanded_3exit" `
  --threshold_mode fixed_0p5 `
  --device cpu `
  --batch_size 128
```

## A14. Parent-level final holdout evaluation

```powershell
python scripts\evaluate_tata_final_holdout_parent_level.py `
  --run_dir "$Run3" `
  --holdout_manifest "$HoldoutFeatureCache\metadata\multilabel_features_manifest.csv" `
  --features_root "$HoldoutFeatureCacheeatures" `
  --out_dir "$HoldoutEvalRoot\main_v06_expanded_3exit_parent_level" `
  --threshold_mode fixed_0p5 `
  --aggregation mean `
  --device cpu `
  --batch_size 128
```

## A15. Final result summary

| setting          |   macro_f1 |   micro_f1 | samples_f1   |   exact_match |   hamming_loss | avg_pred_labels   |   compute_saved |
|:-----------------|-----------:|-----------:|:-------------|--------------:|---------------:|:------------------|----------------:|
| 3-exit fixed 0.5 |     0.7598 |     0.8976 | 0.9048       |        0.8155 |         0.0271 | 1.3045            |             0   |
| 3-exit tuned     |     0.7615 |     0.8937 | 0.9115       |        0.7785 |         0.0292 | 1.4014            |             0   |
| 5-exit tuned     |     0.77   |     0.8866 | 0.9032       |        0.7439 |         0.0311 | 1.4025            |             0   |
| 5-exit dynamic   |     0.7186 |     0.8283 |              |        0.6332 |         0.0498 |                   |            28.6 |

Recommended final configuration:

```text
3-exit fixed threshold 0.5 + parent-level mean aggregation
```
