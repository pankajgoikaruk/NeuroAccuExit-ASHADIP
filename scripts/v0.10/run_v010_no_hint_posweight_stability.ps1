param(
  [int[]]$Seeds = @(101, 202, 303),
  [double]$PosWeightMax = 5.0,
  [string]$Device = "cpu",
  [int]$Epochs = 40,
  [int]$BatchSize = 64,
  [double]$LR = 0.001,
  [string]$Objective = "macro_priority"
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------
# v0.10 no-hint + capped pos_weight seed stability experiment
# Research question:
# Can label-imbalance-aware BCE improve rare-label Macro-F1 more
# reliably than direct hint-pass?
# ------------------------------------------------------------

$TrainManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\final_expanded_training_dataset_balanced\metadata\multilabel_features_manifest_balanced.csv"
$TrainFeaturesRoot = "."
$LabelsJson = "configs\human_talk_10label_schema.json"

$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$HoldoutFeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"

$WorkspaceRoot = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline"
$RunsRoot = "$WorkspaceRoot\main_models\runs_posweight_stability"
$PackagesRoot = "$WorkspaceRoot\packages_posweight_stability"
$LogsRoot = "$WorkspaceRoot\logs_posweight_stability"
$EvalRoot = "$WorkspaceRoot\posweight_stability_eval"
$LatsRoot = "$WorkspaceRoot\posweight_stability_lats_reoptimized"

New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $PackagesRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LatsRoot | Out-Null

$RequiredPaths = @(
  $TrainManifest,
  $LabelsJson,
  $TrainFeaturesRoot,
  $HoldoutManifest,
  $HoldoutFeaturesRoot,
  "scripts\run_tata_weakclip_experiment.ps1",
  "scripts\evaluate_tata_final_holdout_parent_level.py",
  "scripts\v0.10\run_v010_lats_v2_coordinate_reoptimize.py"
)

foreach ($Path in $RequiredPaths) {
  if (-not (Test-Path $Path)) {
    throw "Missing required path: $Path"
  }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "v0.10 no-hint + pos_weight stability"
Write-Host "Seeds        = $($Seeds -join ', ')"
Write-Host "PosWeightMax = $PosWeightMax"
Write-Host "Objective    = $Objective"
Write-Host "Device       = $Device"
Write-Host "============================================================"

foreach ($Seed in $Seeds) {

  $Variant = "main_v010_no_hint_posweight_cap$($PosWeightMax)_seed_$Seed"

  Write-Host ""
  Write-Host "============================================================"
  Write-Host "Training: $Variant"
  Write-Host "============================================================"

  powershell -ExecutionPolicy Bypass -File scripts\run_tata_weakclip_experiment.ps1 `
    -Manifest "$TrainManifest" `
    -FeaturesRoot "$TrainFeaturesRoot" `
    -LabelsJson "$LabelsJson" `
    -WorkspaceRoot "$WorkspaceRoot" `
    -RunsRoot "$RunsRoot" `
    -PackagesRoot "$PackagesRoot" `
    -LogsRoot "$LogsRoot" `
    -Variant "$Variant" `
    -TapBlocks "1,3" `
    -UsePosWeight `
    -PosWeightMax $PosWeightMax `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -LogEvery 0 `
    -LR $LR `
    -Threshold 0.5 `
    -Device "$Device"

  $RunDir = Get-ChildItem $RunsRoot -Directory |
    Where-Object { $_.Name -like "$Variant*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  if ($null -eq $RunDir) {
    throw "Could not find run directory for $Variant"
  }

  Write-Host "RunDir = $($RunDir.FullName)"

  $EvalOut = "$EvalRoot\seed_$Seed`_cap_$($PosWeightMax)_parent_mean_fixed"
  New-Item -ItemType Directory -Force -Path $EvalOut | Out-Null

  Write-Host ""
  Write-Host "Evaluating corrected holdout for seed $Seed"

  python scripts\evaluate_tata_final_holdout_parent_level.py `
    --run_dir "$($RunDir.FullName)" `
    --holdout_manifest "$HoldoutManifest" `
    --features_root "$HoldoutFeaturesRoot" `
    --out_dir "$EvalOut" `
    --threshold_mode fixed_0p5 `
    --aggregation mean `
    --device "$Device" `
    --batch_size 128

  $SegmentPredCsv = "$EvalOut\parent_eval_segment_probs_fixed_0p5_mean.csv"
  if (-not (Test-Path $SegmentPredCsv)) {
    throw "Missing segment probability CSV: $SegmentPredCsv"
  }

  $LatsOut = "$LatsRoot\seed_$Seed`_cap_$($PosWeightMax)_lats_v2_$Objective"
  New-Item -ItemType Directory -Force -Path $LatsOut | Out-Null

  Write-Host ""
  Write-Host "Running LATS-v2 coordinate re-optimization for seed $Seed"

  python scripts\v0.10\run_v010_lats_v2_coordinate_reoptimize.py `
    --segment-pred-csv "$SegmentPredCsv" `
    --labels-json "$LabelsJson" `
    --out-dir "$LatsOut" `
    --parent-id-col "parent_clip_id" `
    --prob-prefix "exit3_prob_" `
    --threshold-min 0.10 `
    --threshold-max 0.95 `
    --threshold-step 0.01 `
    --aggregation-methods "mean,max,top2mean,top3mean,p75,p90" `
    --objective "$Objective" `
    --max-iter 20 `
    --model-name "$($RunDir.Name)"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Summary table"
Write-Host "============================================================"

Get-ChildItem $LatsRoot -Recurse -Filter "lats_v2_coordinate_reoptimized_summary.csv" |
  Where-Object { $_.FullName -like "*cap_$($PosWeightMax)*" } |
  ForEach-Object {
    $row = Import-Csv $_.FullName
    [PSCustomObject]@{
      Run = $_.Directory.Name
      MacroF1 = [double]$row.macro_f1
      MicroF1 = [double]$row.micro_f1
      SamplesF1 = [double]$row.samples_f1
      Exact = [double]$row.exact_match
      Hamming = [double]$row.hamming_loss
      AvgPredLabels = [double]$row.avg_pred_labels
    }
  } | Sort-Object Run | Format-Table -AutoSize

Write-Host ""
Write-Host "Done. Compare against v0.9_4 baseline:"
Write-Host "Macro=0.867256 Micro=0.945786 Samples=0.951700 Exact=0.860438 Hamming=0.015802"
