param(
    [string]$RunDir = "",
    [string]$Device = "cpu",
    [int]$BatchSize = 128,
    [double]$MaxMacroF1Drop = 0.02,
    [double]$MinExit2Fraction = 0.05,
    [ValidateSet("auto", "tuned_per_exit", "final_exit_tuned", "fixed_0p5")]
    [string]$ThresholdMode = "auto",
    [switch]$SkipPrechecks,
    [switch]$SkipTuning,
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
    throw "Current branch is '$CurrentBranch'. Switch to '$ExpectedBranch'."
}

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$RunName = "main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845"
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $Matches = @(
        Get-ChildItem "human_talk_workspace" `
            -Directory `
            -Recurse `
            -Filter $RunName `
            -ErrorAction SilentlyContinue
    )

    if ($Matches.Count -eq 0) {
        throw "Could not find canonical run '$RunName'. Pass -RunDir explicitly."
    }
    if ($Matches.Count -gt 1) {
        $Paths = ($Matches.FullName -join "`n")
        throw "Multiple canonical run directories found. Pass -RunDir explicitly:`n$Paths"
    }
    $RunDir = $Matches[0].FullName
}

$FixedPolicyScriptRoot = "scripts\v0.11_EE\fixed_policy"
$ThresholdComparison = Join-Path $RunDir "threshold_tuning\threshold_comparison.json"
if ($ThresholdMode -eq "auto") {
    if (Test-Path $ThresholdComparison) {
        $ResolvedThresholdMode = "tuned_per_exit"
    }
    else {
        $ResolvedThresholdMode = "fixed_0p5"
        Write-Host "[INFO] Per-exit tuned threshold file was not found." -ForegroundColor DarkYellow
        Write-Host "[INFO] Using fixed_0p5 for Exit 1/2 label-set and reliability decisions." -ForegroundColor DarkYellow
        Write-Host "[INFO] Policy selection still uses frozen LATS-v2 parent-level metrics." -ForegroundColor DarkYellow
    }
}
else {
    $ResolvedThresholdMode = $ThresholdMode
}

if ($ResolvedThresholdMode -eq "tuned_per_exit" -and -not (Test-Path $ThresholdComparison)) {
    throw "ThresholdMode=tuned_per_exit requires: $ThresholdComparison"
}

$DynamicScriptRoot = "scripts\v0.11_EE\dynamic_policy"

$VerifyScript = "$FixedPolicyScriptRoot\verify_checkpoint_equivalence_v011.py"
$TuneScript = "$DynamicScriptRoot\tune_dynamic_policy_v011.py"
$EvaluateScript = "$DynamicScriptRoot\evaluate_dynamic_early_exit_v011.py"

$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$FeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"

$OutputRoot = "human_talk_workspace\active_budget_anytime_exit_v0.2\v0.11_EE\dynamic_policy"
$TuningOut = "$OutputRoot\validation_tuning"
$HoldoutOut = "$OutputRoot\corrected_holdout_evaluation"
$FrozenPolicy = "$TuningOut\frozen_dynamic_policy_v011.json"
$EquivalenceJson = "$OutputRoot\checkpoint_staged_equivalence_dynamic_precheck.json"

$RequiredPaths = @(
    $RunDir,
    (Join-Path $RunDir "ckpt\best.pt"),
    $HoldoutManifest,
    $FeaturesRoot,
    $LabelsJson,
    $LatsConfig,
    "models\anytime_exit_net.py",
    "tests\test_anytime_exit_net.py",
    $VerifyScript,
    "scripts\v0.10\evaluate_frozen_lats_config_v010.py",
    $TuneScript,
    $EvaluateScript
)

foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $TuningOut, $HoldoutOut | Out-Null

Write-Host ""
Write-Host "=== NeuroAccuExit v0.11 Genuine Dynamic Early-Exit ===" -ForegroundColor Cyan
Write-Host "Branch:              $CurrentBranch"
Write-Host "Canonical run:       $RunDir"
Write-Host "Device:              $Device"
Write-Host "Threshold mode:      $ResolvedThresholdMode"
Write-Host "Validation tuning:   $TuningOut"
Write-Host "Frozen policy:       $FrozenPolicy"
Write-Host "Holdout evaluation:  $HoldoutOut"
Write-Host ""

if (-not $SkipPrechecks) {
    Write-Host "[1/4] Running staged-wrapper unit tests..." -ForegroundColor Yellow
    python -m unittest tests.test_anytime_exit_net -v
    if ($LASTEXITCODE -ne 0) {
        throw "Staged-wrapper unit tests failed."
    }

    Write-Host "[2/4] Rechecking checkpoint staged equivalence..." -ForegroundColor Yellow
    $VerifyArgs = @(
        $VerifyScript,
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
        throw "Checkpoint staged-equivalence verification failed."
    }
}
else {
    Write-Host "[1/4] Unit tests skipped." -ForegroundColor DarkYellow
    Write-Host "[2/4] Checkpoint equivalence skipped." -ForegroundColor DarkYellow
}

if (-not $SkipTuning) {
    Write-Host "[3/4] Tuning and freezing policy on validation data..." -ForegroundColor Yellow

    $TuneArgs = @(
        $TuneScript,
        "--run_dir", $RunDir,
        "--labels_json", $LabelsJson,
        "--split", "val",
        "--lats_config_json", $LatsConfig,
        "--parent_id_col", "parent_clip_id",
        "--threshold_mode", $ResolvedThresholdMode,
        "--confidence_grid", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        "--margin_grid", "0.00,0.02,0.05,0.08,0.10,0.15",
        "--max_macro_f1_drop", "$MaxMacroF1Drop",
        "--min_exit2_fraction", "$MinExit2Fraction",
        "--batch_size", "$BatchSize",
        "--device", $Device,
        "--out_dir", $TuningOut
    )

    python @TuneArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Dynamic policy tuning failed."
    }
}
else {
    Write-Host "[3/4] Validation tuning skipped; reusing frozen policy." -ForegroundColor DarkYellow
    if (-not (Test-Path $FrozenPolicy)) {
        throw "SkipTuning was used, but frozen policy does not exist: $FrozenPolicy"
    }
}

Write-Host "[4/4] Running genuine staged policy on corrected holdout..." -ForegroundColor Yellow

$EvaluateArgs = @(
    $EvaluateScript,
    "--run_dir", $RunDir,
    "--policy_json", $FrozenPolicy,
    "--holdout_manifest", $HoldoutManifest,
    "--features_root", $FeaturesRoot,
    "--labels_json", $LabelsJson,
    "--lats_config_json", $LatsConfig,
    "--parent_id_col", "parent_clip_id",
    "--batch_size", "$BatchSize",
    "--device", $Device,
    "--out_dir", $HoldoutOut
)

if ($SkipParentEval) {
    $EvaluateArgs += "--skip_parent_eval"
}

python @EvaluateArgs
if ($LASTEXITCODE -ne 0) {
    throw "Genuine Dynamic Early-Exit evaluation failed."
}

Write-Host ""
Write-Host "V0.11 genuine Dynamic Early-Exit completed." -ForegroundColor Green
Write-Host "Frozen policy:  $FrozenPolicy"
Write-Host "Validation:     $TuningOut"
Write-Host "Holdout:        $HoldoutOut"
Write-Host ""
Write-Host "The corrected holdout used the frozen validation policy; no holdout retuning occurred." -ForegroundColor DarkYellow
