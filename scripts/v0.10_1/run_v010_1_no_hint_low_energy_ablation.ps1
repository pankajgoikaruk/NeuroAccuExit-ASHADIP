param(
  [string]$BaseTrainManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\final_expanded_training_dataset_balanced\metadata\multilabel_features_manifest_balanced.csv",
  [string]$LowEnergyMaskedManifest = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\human_reviewed_masked_v09\feature_cache\metadata\multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv",
  [string]$TataLawyerRepoRoot = "",
  [string]$LowEnergyFeaturesRoot = "",
  [string]$CorrectedHoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv",
  [string]$HoldoutFeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features",
  [string]$LabelsJson = "configs\human_talk_10label_schema.json",

  [string]$WorkspaceRoot = "human_talk_workspace\tata_v0.10_1_low_energy_recovery_ablation",
  [string]$TrainFeaturesRoot = ".",
  [string]$Device = "cpu",
  [int]$Epochs = 40,
  [int]$BatchSize = 64,
  [double]$LR = 0.001,
  [string]$Objective = "macro_priority"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "== v0.10_1 no-hint + low-energy recovery ablation ==" -ForegroundColor Cyan
Write-Host "Safety: source manifests are read-only; outputs go to $WorkspaceRoot" -ForegroundColor Yellow

$RelativeLowEnergyMaskedManifest = "human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\human_reviewed_masked_v09\feature_cache\metadata\multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv"

function Resolve-LowEnergyManifest {
  param(
    [string]$RequestedPath,
    [string]$RepoRoot,
    [string]$RelativePath
  )

  if ($RequestedPath -and (Test-Path $RequestedPath)) {
    return (Resolve-Path $RequestedPath).Path
  }

  if ($RepoRoot) {
    $Candidate = Join-Path $RepoRoot $RelativePath
    if (Test-Path $Candidate) {
      return (Resolve-Path $Candidate).Path
    }
  }

  $SiblingCandidate = Join-Path (Split-Path (Get-Location).Path -Parent) (Join-Path "TATA-LAWYER" $RelativePath)
  if (Test-Path $SiblingCandidate) {
    return (Resolve-Path $SiblingCandidate).Path
  }

  $CurrentName = Split-Path (Get-Location).Path -Leaf
  if ($CurrentName -ne "TATA-LAWYER") {
    Write-Host "" -ForegroundColor Yellow
    Write-Host "Low-energy masked manifest was not found in the ASHADIP repo." -ForegroundColor Yellow
    Write-Host "This file belongs to your local TATA-LAWYER workspace." -ForegroundColor Yellow
    Write-Host "Pass either:" -ForegroundColor Yellow
    Write-Host "  -TataLawyerRepoRoot C:\path\to\TATA-LAWYER" -ForegroundColor Yellow
    Write-Host "or:" -ForegroundColor Yellow
    Write-Host "  -LowEnergyMaskedManifest C:\path\to\multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
  }

  throw "Required low-energy masked manifest not found: $RequestedPath"
}

$LowEnergyMaskedManifest = Resolve-LowEnergyManifest `
  -RequestedPath $LowEnergyMaskedManifest `
  -RepoRoot $TataLawyerRepoRoot `
  -RelativePath $RelativeLowEnergyMaskedManifest

if (-not $LowEnergyFeaturesRoot) {
  # Manifest layout: ...\feature_cache\metadata\manifest.csv
  # Possible feature layouts:
  #   ...\feature_cache\features\<feat_relpath>
  #   ...\feature_cache\<feat_relpath>
  $FeatureCacheRoot = Split-Path (Split-Path $LowEnergyMaskedManifest -Parent) -Parent
  $CandidateFeaturesDir = Join-Path $FeatureCacheRoot "features"
  if (Test-Path $CandidateFeaturesDir) {
    $LowEnergyFeaturesRoot = $CandidateFeaturesDir
  } else {
    $LowEnergyFeaturesRoot = $FeatureCacheRoot
  }
}

Write-Host "LowEnergyMaskedManifest = $LowEnergyMaskedManifest" -ForegroundColor DarkGray
Write-Host "LowEnergyFeaturesRoot  = $LowEnergyFeaturesRoot" -ForegroundColor DarkGray

foreach ($Path in @($BaseTrainManifest, $CorrectedHoldoutManifest, $LabelsJson, $TrainFeaturesRoot, $HoldoutFeaturesRoot, $LowEnergyFeaturesRoot)) {
  if (-not (Test-Path $Path)) {
    throw "Required path not found: $Path"
  }
}

New-Item -ItemType Directory -Force -Path $WorkspaceRoot | Out-Null

# 1) Audit reviewed low-energy rows and leakage risk.
python scripts\v0.10_1\audit_low_energy_recovered_manifest.py `
  --base-train-manifest "$BaseTrainManifest" `
  --low-energy-masked-manifest "$LowEnergyMaskedManifest" `
  --corrected-holdout-manifest "$CorrectedHoldoutManifest" `
  --labels-json "$LabelsJson" `
  --out-dir "$WorkspaceRoot\audit"

if ($LASTEXITCODE -ne 0) {
  throw "Low-energy audit failed with exit code $LASTEXITCODE"
}

# 2) Build copied/augmented training manifest.
# Default behavior: train split only, fully-known reviewed rows only, exclude corrected-holdout parent overlap.
python scripts\v0.10_1\build_v010_1_low_energy_augmented_manifest.py `
  --base-train-manifest "$BaseTrainManifest" `
  --low-energy-masked-manifest "$LowEnergyMaskedManifest" `
  --low-energy-features-root "$LowEnergyFeaturesRoot" `
  --corrected-holdout-manifest "$CorrectedHoldoutManifest" `
  --labels-json "$LabelsJson" `
  --workspace-root "$WorkspaceRoot" `
  --include-splits "train"

if ($LASTEXITCODE -ne 0) {
  throw "Augmented manifest build failed with exit code $LASTEXITCODE"
}

$AugmentedManifest = "$WorkspaceRoot\metadata\multilabel_features_manifest_v010_1_LOW_ENERGY_AUGMENTED.csv"
if (-not (Test-Path $AugmentedManifest)) {
  throw "Augmented manifest was not created: $AugmentedManifest"
}

# Quick manifest check: no low-energy train row should still point to recovered_low_energy as a relative loader path.
$AugRows = Import-Csv $AugmentedManifest
if (($AugRows | Select-Object -First 1).PSObject.Properties.Name -notcontains "feat_relpath") {
  throw "Augmented manifest is missing feat_relpath, which training requires."
}
$BadPathCount = @($AugRows | Where-Object { $_.feat_relpath -like "recovered_low_energy*" }).Count
if ($BadPathCount -gt 0) {
  throw "Augmented manifest still contains $BadPathCount relative recovered_low_energy feat_relpath rows. Pull latest branch and rebuild workspace."
}

# 3) Train no-hint model only.
$RunsRoot = "$WorkspaceRoot\main_models\runs"
$PackagesRoot = "$WorkspaceRoot\packages"
$LogsRoot = "$WorkspaceRoot\logs"
$Variant = "main_v010_1_no_hint_low_energy_augmented"

powershell -ExecutionPolicy Bypass -File scripts\run_tata_weakclip_experiment.ps1 `
  -Manifest "$AugmentedManifest" `
  -FeaturesRoot "$TrainFeaturesRoot" `
  -LabelsJson "$LabelsJson" `
  -WorkspaceRoot "$WorkspaceRoot" `
  -RunsRoot "$RunsRoot" `
  -PackagesRoot "$PackagesRoot" `
  -LogsRoot "$LogsRoot" `
  -Variant "$Variant" `
  -TapBlocks "1,3" `
  -Epochs $Epochs `
  -BatchSize $BatchSize `
  -LogEvery 0 `
  -LR $LR `
  -Threshold 0.5 `
  -Device "$Device"

if ($LASTEXITCODE -ne 0) {
  throw "Training failed with exit code $LASTEXITCODE. Evaluation/LATS will not run until training completes."
}

$RunDir = Get-ChildItem $RunsRoot -Directory |
  Where-Object { $_.Name -like "$Variant*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if ($null -eq $RunDir) {
  throw "No run directory found for $Variant under $RunsRoot"
}

$BestCkpt = Join-Path $RunDir.FullName "ckpt\best.pt"
if (-not (Test-Path $BestCkpt)) {
  throw "Training run finished without ckpt\best.pt: $BestCkpt"
}

Write-Host "RunDir = $($RunDir.FullName)" -ForegroundColor Green

# 4) Evaluate on the same corrected holdout as v0.10.
$EvalOut = "$WorkspaceRoot\evaluation\no_hint_low_energy_augmented_parent_mean_fixed"
New-Item -ItemType Directory -Force -Path $EvalOut | Out-Null

python scripts\evaluate_tata_final_holdout_parent_level.py `
  --run_dir "$($RunDir.FullName)" `
  --holdout_manifest "$CorrectedHoldoutManifest" `
  --features_root "$HoldoutFeaturesRoot" `
  --out_dir "$EvalOut" `
  --threshold_mode fixed_0p5 `
  --aggregation mean `
  --device "$Device" `
  --batch_size 128

if ($LASTEXITCODE -ne 0) {
  throw "Holdout evaluation failed with exit code $LASTEXITCODE"
}

$SegmentPredCsv = "$EvalOut\parent_eval_segment_probs_fixed_0p5_mean.csv"
if (-not (Test-Path $SegmentPredCsv)) {
  throw "Segment prediction CSV not found: $SegmentPredCsv"
}

# 5) Re-run v0.10 LATS-v2 coordinate optimization.
$LatsOut = "$WorkspaceRoot\lats_v2_reoptimized\no_hint_low_energy_augmented"
New-Item -ItemType Directory -Force -Path $LatsOut | Out-Null

python scripts\v0.10\run_v010_lats_v2_coordinate_reoptimize.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --labels-json "$LabelsJson" `
  --out-dir "$LatsOut" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --objective "$Objective" `
  --model-name "$($RunDir.Name)"

if ($LASTEXITCODE -ne 0) {
  throw "LATS-v2 reoptimization failed with exit code $LASTEXITCODE"
}

$SummaryCsv = "$LatsOut\lats_v2_coordinate_reoptimized_summary.csv"
Write-Host ""
Write-Host "== v0.10_1 LATS-v2 summary ==" -ForegroundColor Cyan
Import-Csv $SummaryCsv | Format-Table -AutoSize

Write-Host ""
Write-Host "Compare against selected v0.10 no-hint target:" -ForegroundColor Yellow
Write-Host "Macro-F1   >= 0.8624"
Write-Host "Micro-F1   >= 0.9531"
Write-Host "Samples-F1 >= 0.9589"
Write-Host "Exact      >= 0.8766"
Write-Host "Hamming    <= 0.0137"
