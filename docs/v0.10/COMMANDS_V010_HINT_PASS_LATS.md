# v0.10 PowerShell Commands — Human-Talk Hint-Pass + LATS

These commands replace the older moth-data commands. Do **not** use `data\moth_sounds`, `data_caches`, or `configs\audio_moth.yaml` for this human-talk speaker experiment.

---

## 1. Apply safe hint activation patch

```powershell
python scripts\v0.10\apply_v010_safe_hint_activation_patch.py

python -m py_compile `
  models\exit_net.py `
  utils\model_factory.py `
  training\train_multilabel.py `
  scripts\evaluate_tata_final_holdout_parent_level.py
```

Check changes:

```powershell
git diff -- `
  models\exit_net.py `
  utils\model_factory.py `
  training\train_multilabel.py `
  scripts\run_tata_weakclip_experiment.ps1 `
  scripts\evaluate_tata_final_holdout_parent_level.py
```

---

## 2. Set shared paths

```powershell
$TrainManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\final_expanded_training_dataset_balanced\metadata\multilabel_features_manifest_balanced.csv"
$TrainFeaturesRoot = "."
$LabelsJson = "configs\human_talk_10label_schema.json"

$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$HoldoutFeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"

$WorkspaceRoot = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline"
$RunsRoot = "$WorkspaceRoot\main_models\runs"
$PackagesRoot = "$WorkspaceRoot\packages"
$LogsRoot = "$WorkspaceRoot\logs"

Test-Path $TrainManifest
Test-Path $LabelsJson
Test-Path $TrainFeaturesRoot
```

Expected:

```text
True
True
True
```

---

## 3. Train v0.10 no-hint control

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tata_weakclip_experiment.ps1 `
  -Manifest "$TrainManifest" `
  -FeaturesRoot "$TrainFeaturesRoot" `
  -LabelsJson "$LabelsJson" `
  -WorkspaceRoot "$WorkspaceRoot" `
  -RunsRoot "$RunsRoot" `
  -PackagesRoot "$PackagesRoot" `
  -LogsRoot "$LogsRoot" `
  -Variant "main_v010_human_corrected_balanced_3exit_no_hint" `
  -TapBlocks "1,3" `
  -Epochs 40 `
  -BatchSize 64 `
  -LogEvery 0 `
  -LR 0.001 `
  -Threshold 0.5 `
  -Device "cpu"
```

---

## 4. Train v0.10 hint-pass model

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tata_weakclip_experiment.ps1 `
  -Manifest "$TrainManifest" `
  -FeaturesRoot "$TrainFeaturesRoot" `
  -LabelsJson "$LabelsJson" `
  -WorkspaceRoot "$WorkspaceRoot" `
  -RunsRoot "$RunsRoot" `
  -PackagesRoot "$PackagesRoot" `
  -LogsRoot "$LogsRoot" `
  -Variant "main_v010_human_corrected_balanced_3exit_hint_pass" `
  -TapBlocks "1,3" `
  -ExitHint `
  -HintDim 8 `
  -HintSource "probs" `
  -HintActivation "sigmoid" `
  -Epochs 40 `
  -BatchSize 64 `
  -LogEvery 0 `
  -LR 0.001 `
  -Threshold 0.5 `
  -Device "cpu"
```

Note: `HintDetach` and `HintUseStats` default to true. Omit them if PowerShell complains about Boolean conversion.

---

## 5. Evaluate a run on corrected holdout

Set `$RunDir` manually or select the latest matching run.

```powershell
$RunDir = Get-ChildItem $RunsRoot -Directory |
  Where-Object { $_.Name -like "main_v010_human_corrected_balanced_3exit_no_hint*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$EvalOut = "$WorkspaceRoot\evaluation\no_hint_parent_mean_fixed"
New-Item -ItemType Directory -Force -Path $EvalOut | Out-Null

python scripts\evaluate_tata_final_holdout_parent_level.py `
  --run_dir "$($RunDir.FullName)" `
  --holdout_manifest "$HoldoutManifest" `
  --features_root "$HoldoutFeaturesRoot" `
  --out_dir "$EvalOut" `
  --threshold_mode fixed_0p5 `
  --aggregation mean `
  --device cpu `
  --batch_size 128
```

The segment probability CSV is:

```text
$EvalOut\parent_eval_segment_probs_fixed_0p5_mean.csv
```

---

## 6. Apply frozen old v0.9_4 LATS-v2 transfer

```powershell
$SegmentPredCsv = "$EvalOut\parent_eval_segment_probs_fixed_0p5_mean.csv"
$FrozenOut = "$WorkspaceRoot\lats_v2_frozen\no_hint"
New-Item -ItemType Directory -Force -Path $FrozenOut | Out-Null

python scripts\v0.10\evaluate_frozen_lats_v2_baseline_recheck.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --labels-json "$LabelsJson" `
  --out-dir "$FrozenOut" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_"
```

---

## 7. Re-optimize LATS for v0.10 probabilities

Use the v0.9 LATS search if available:

```powershell
$SegmentPredCsv = "$EvalOut\parent_eval_segment_probs_fixed_0p5_mean.csv"
$LatsOut = "$WorkspaceRoot\lats_reoptimized\no_hint"
New-Item -ItemType Directory -Force -Path $LatsOut | Out-Null

python scripts\v0.9\run_lats_labelwise_aggregation_threshold_search_v09.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --labels-json "$LabelsJson" `
  --out-dir "$LatsOut" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --seeds 20 `
  --cal-fraction 0.5 `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.01 `
  --aggregation-methods "mean,max,top2mean,top3mean,p75,p90" `
  --model-name "$($RunDir.Name)"
```

For metric-aware coordinate LATS-v2, start from the LATS-v1 result and run the coordinate-search implementation used in v0.9_4/v0.10.

---

## 8. Seed-stability check for v0.10 no-hint

```powershell
$Seeds = @(101, 202, 303)

$WorkspaceRoot = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline"
$RunsRoot = "$WorkspaceRoot\main_models\runs_seed_stability"
$EvalRoot = "$WorkspaceRoot\seed_stability_eval"
$LatsRoot = "$WorkspaceRoot\seed_stability_lats_reoptimized"

New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LatsRoot | Out-Null

foreach ($Seed in $Seeds) {

    $Variant = "main_v010_no_hint_seed_$Seed"

    python -m training.train_multilabel `
      --manifest "$TrainManifest" `
      --features_root "$TrainFeaturesRoot" `
      --labels_json "$LabelsJson" `
      --runs_root "$RunsRoot" `
      --variant "$Variant" `
      --tap_blocks "1,3" `
      --epochs 40 `
      --batch_size 64 `
      --log_every 0 `
      --lr 0.001 `
      --threshold 0.5 `
      --seed $Seed `
      --device cpu

    $RunDir = Get-ChildItem $RunsRoot -Directory |
      Where-Object { $_.Name -like "$Variant*" } |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1

    $EvalOut = "$EvalRoot\seed_$Seed`_parent_mean_fixed"
    New-Item -ItemType Directory -Force -Path $EvalOut | Out-Null

    python scripts\evaluate_tata_final_holdout_parent_level.py `
      --run_dir "$($RunDir.FullName)" `
      --holdout_manifest "$HoldoutManifest" `
      --features_root "$HoldoutFeaturesRoot" `
      --out_dir "$EvalOut" `
      --threshold_mode fixed_0p5 `
      --aggregation mean `
      --device cpu `
      --batch_size 128

    $SegmentPredCsv = "$EvalOut\parent_eval_segment_probs_fixed_0p5_mean.csv"
    $LatsOut = "$LatsRoot\seed_$Seed`_lats_reoptimized"
    New-Item -ItemType Directory -Force -Path $LatsOut | Out-Null

    python scripts\v0.9\run_lats_labelwise_aggregation_threshold_search_v09.py `
      --segment-pred-csv "$SegmentPredCsv" `
      --labels-json "$LabelsJson" `
      --out-dir "$LatsOut" `
      --parent-id-col "parent_clip_id" `
      --prob-prefix "exit3_prob_" `
      --seeds 20 `
      --cal-fraction 0.5 `
      --threshold-min 0.10 `
      --threshold-max 0.95 `
      --threshold-step 0.01 `
      --aggregation-methods "mean,max,top2mean,top3mean,p75,p90" `
      --model-name "$($RunDir.Name)"
}
```

Summarize seeds:

```powershell
Get-ChildItem $LatsRoot -Recurse -Filter "lats_final_full_holdout_eval.csv" |
  ForEach-Object {
    $row = Import-Csv $_.FullName
    [PSCustomObject]@{
      Seed = ($_.Directory.Name -replace "seed_", "" -replace "_lats_reoptimized", "")
      MacroF1 = [double]$row.macro_f1
      MicroF1 = [double]$row.micro_f1
      SamplesF1 = [double]$row.samples_f1
      Exact = [double]$row.exact_match
      Hamming = [double]$row.hamming_loss
      AvgPredLabels = [double]$row.avg_pred_labels
    }
  } | Sort-Object Seed | Format-Table -AutoSize
```
