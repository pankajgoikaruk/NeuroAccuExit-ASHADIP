param(
    [string]$RunDir = "",
    [string]$Device = "cpu",
    [int]$BatchSize = 128,
    [switch]$SkipUnitTests,
    [switch]$SkipParentEval
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Test-Path ".git")) {
    throw "Run this script from the NeuroAccuExit-ASHADIP repository root."
}

$ExpectedBranch = "active_budget_anytime_exit_v0.2"
$CurrentBranch = (git branch --show-current | Out-String).Trim()
if ($CurrentBranch -ne $ExpectedBranch) {
    throw "Current branch is '$CurrentBranch'. Switch to '$ExpectedBranch' before running v0.11_EE."
}

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RunName = "main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845"
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunMatches = @(
        Get-ChildItem "human_talk_workspace" `
            -Directory `
            -Recurse `
            -Filter $RunName `
            -ErrorAction SilentlyContinue
    )

    if ($RunMatches.Count -eq 0) {
        throw "Could not find canonical run '$RunName' under human_talk_workspace. Pass -RunDir explicitly."
    }
    if ($RunMatches.Count -gt 1) {
        $Paths = ($RunMatches.FullName -join "`n")
        throw "Multiple canonical run directories were found. Pass -RunDir explicitly:`n$Paths"
    }
    $RunDir = $RunMatches[0].FullName
}

$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$FeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"
$OutDir = "human_talk_workspace\active_budget_anytime_exit_v0.2\v0.11_EE\fixed_exit_audit"
$EquivalenceJson = "human_talk_workspace\active_budget_anytime_exit_v0.2\v0.11_EE\checkpoint_staged_equivalence.json"

$RequiredPaths = @(
    $RunDir,
    (Join-Path $RunDir "ckpt\best.pt"),
    $HoldoutManifest,
    $FeaturesRoot,
    $LabelsJson,
    $LatsConfig,
    "models\anytime_exit_net.py",
    "scripts\v0.11_EE\verify_checkpoint_equivalence_v011.py",
    "scripts\v0.11_EE\evaluate_fixed_exits_v011.py",
    "scripts\v0.10\evaluate_frozen_lats_config_v010.py"
)

foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $EquivalenceJson -Parent) | Out-Null

Write-Host "" 
Write-Host "=== NeuroAccuExit v0.11_EE ===" -ForegroundColor Cyan
Write-Host "Branch:            $CurrentBranch"
Write-Host "Run directory:     $RunDir"
Write-Host "Holdout manifest:  $HoldoutManifest"
Write-Host "Features root:     $FeaturesRoot"
Write-Host "Output directory:  $OutDir"
Write-Host "Device:            $Device"
Write-Host ""

if (-not $SkipUnitTests) {
    Write-Host "[1/3] Running staged-wrapper unit tests..." -ForegroundColor Yellow
    python -m unittest tests.test_anytime_exit_net -v
    if ($LASTEXITCODE -ne 0) {
        throw "Staged-wrapper unit tests failed."
    }
}
else {
    Write-Host "[1/3] Unit tests skipped by request." -ForegroundColor DarkYellow
}

Write-Host "[2/3] Verifying canonical checkpoint equivalence..." -ForegroundColor Yellow
$VerifyArgs = @(
    "scripts\v0.11_EE\verify_checkpoint_equivalence_v011.py",
    "--run_dir", $RunDir,
    "--labels_json", $LabelsJson,
    "--holdout_manifest", $HoldoutManifest,
    "--features_root", $FeaturesRoot,
    "--sample_count", "8",
    "--device", $Device,
    "--out_json", $EquivalenceJson
)
python @VerifyArgs
if ($LASTEXITCODE -ne 0) {
    throw "Canonical checkpoint staged-equivalence verification failed."
}

Write-Host "[3/3] Evaluating Always Exit 1 / Exit 2 / Exit 3..." -ForegroundColor Yellow
$EvalArgs = @(
    "scripts\v0.11_EE\evaluate_fixed_exits_v011.py",
    "--run_dir", $RunDir,
    "--holdout_manifest", $HoldoutManifest,
    "--features_root", $FeaturesRoot,
    "--labels_json", $LabelsJson,
    "--lats_config_json", $LatsConfig,
    "--out_dir", $OutDir,
    "--parent_id_col", "parent_clip_id",
    "--segment_threshold", "0.5",
    "--batch_size", "$BatchSize",
    "--device", $Device
)
if ($SkipParentEval) {
    $EvalArgs += "--skip_parent_eval"
}
python @EvalArgs
if ($LASTEXITCODE -ne 0) {
    throw "V0.11 fixed-exit evaluation failed."
}

Write-Host ""
Write-Host "V0.11_EE completed successfully." -ForegroundColor Green
Write-Host "Equivalence report: $EquivalenceJson"
Write-Host "Fixed-exit outputs: $OutDir"
Write-Host ""
Write-Host "Important: Exit 1 and Exit 2 parent metrics are frozen LATS-v2 policy-transfer results, not exit-specific calibrated optima." -ForegroundColor DarkYellow
