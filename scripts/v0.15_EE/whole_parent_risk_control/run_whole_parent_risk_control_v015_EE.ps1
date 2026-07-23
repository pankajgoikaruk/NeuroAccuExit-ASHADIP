param(
    [string]$RunDir = "",
    [string]$Device = "cpu",
    [int]$BatchSize = 128,
    [int]$CvFolds = 5,
    [int]$TimingRepeats = 10,
    [int]$TorchThreads = 1,
    [double]$MaxMacroF1Drop = 0.005,
    [double]$MaxMicroF1Drop = 0.005,
    [double]$MaxExactMatchDrop = 0.01,
    [double]$MaxOverallHarmFraction = 0.01,
    [double]$MinParentStopFraction = 0.02,
    [ValidateSet("auto", "tuned_per_exit", "final_exit_tuned", "fixed_0p5")]
    [string]$ThresholdMode = "auto",
    [switch]$SkipPrechecks,
    [switch]$SkipTuning
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Test-Path ".git")) {
    throw "Run this script from the NeuroAccuExit-ASHADIP repository root."
}

$ExpectedBranch = "active_budget_anytime_exit_v0.3"
$CurrentBranch = (git branch --show-current | Out-String).Trim()
if ($CurrentBranch -ne $ExpectedBranch) {
    throw "Current branch is '$CurrentBranch'. Switch to '$ExpectedBranch'."
}

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "$TorchThreads"
$env:MKL_NUM_THREADS = "$TorchThreads"
$env:OPENBLAS_NUM_THREADS = "$TorchThreads"
$env:NUMEXPR_NUM_THREADS = "$TorchThreads"

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
        throw "Multiple canonical runs found. Pass -RunDir explicitly:`n$Paths"
    }
    $RunDir = $Matches[0].FullName
}

$ThresholdComparison = Join-Path $RunDir "threshold_tuning\threshold_comparison.json"
if ($ThresholdMode -eq "auto") {
    if (Test-Path $ThresholdComparison) {
        $ResolvedThresholdMode = "tuned_per_exit"
    }
    else {
        $ResolvedThresholdMode = "fixed_0p5"
        Write-Host "[INFO] Per-exit tuned thresholds were not found." -ForegroundColor DarkYellow
        Write-Host "[INFO] Using fixed_0p5 for segment decisions." -ForegroundColor DarkYellow
        Write-Host "[INFO] Whole-parent targets and selection use frozen LATS-v2." -ForegroundColor DarkYellow
    }
}
else {
    $ResolvedThresholdMode = $ThresholdMode
}

if ($ResolvedThresholdMode -eq "tuned_per_exit" -and -not (Test-Path $ThresholdComparison)) {
    throw "ThresholdMode=tuned_per_exit requires: $ThresholdComparison"
}

$ScriptRoot = "scripts\v0.15_EE\whole_parent_risk_control"
$VerifyScript = "scripts\v0.11_EE\fixed_policy\verify_checkpoint_equivalence_v011.py"
$TuneScript = "$ScriptRoot\tune_whole_parent_risk_control_v015.py"
$EvaluateScript = "$ScriptRoot\evaluate_whole_parent_risk_control_v015.py"

$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$FeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"

$OutputRoot = "human_talk_workspace\active_budget_anytime_exit_v0.3\v0.15_EE\whole_parent_risk_control"
$TuningOut = "$OutputRoot\validation_tuning"
$HoldoutOut = "$OutputRoot\corrected_holdout_evaluation"
$FrozenPolicy = "$TuningOut\frozen_whole_parent_policy_v015.json"
$EquivalenceJson = "$OutputRoot\checkpoint_staged_equivalence_precheck.json"

$RequiredPaths = @(
    $RunDir,
    (Join-Path $RunDir "ckpt\best.pt"),
    $HoldoutManifest,
    $FeaturesRoot,
    $LabelsJson,
    $LatsConfig,
    "models\anytime_exit_net.py",
    "policies\whole_parent_selective_exit.py",
    "tests\test_whole_parent_selective_exit.py",
    $VerifyScript,
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
Write-Host "=== NeuroAccuExit v0.15 Whole-Parent Selective Risk Control ===" -ForegroundColor Cyan
Write-Host "Branch:                     $CurrentBranch"
Write-Host "Canonical run:              $RunDir"
Write-Host "Device:                     $Device"
Write-Host "Threshold mode:             $ResolvedThresholdMode"
Write-Host "Grouped CV folds:           $CvFolds"
Write-Host "Parent Macro-F1 drop limit: $MaxMacroF1Drop"
Write-Host "Parent Micro-F1 drop limit: $MaxMicroF1Drop"
Write-Host "Overall harm limit:         $MaxOverallHarmFraction"
Write-Host "Timing repeats:             $TimingRepeats"
Write-Host "Torch/BLAS threads:         $TorchThreads"
Write-Host "Validation output:          $TuningOut"
Write-Host "Holdout output:             $HoldoutOut"
Write-Host ""

if (-not $SkipPrechecks) {
    Write-Host "[1/4] Running staged and whole-parent policy tests..." -ForegroundColor Yellow
    python -m unittest `
        tests.test_anytime_exit_net `
        tests.test_whole_parent_selective_exit `
        -v
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed."
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
    Write-Host "[3/4] Training OOF whole-parent risk controllers..." -ForegroundColor Yellow
    $TuneArgs = @(
        $TuneScript,
        "--run_dir", $RunDir,
        "--labels_json", $LabelsJson,
        "--lats_config_json", $LatsConfig,
        "--parent_id_col", "parent_clip_id",
        "--threshold_mode", $ResolvedThresholdMode,
        "--cv_folds", "$CvFolds",
        "--target_recall_grid", "0.80,0.90,0.95,0.98,1.00",
        "--expected_harm_grid", "none,0.005,0.01,0.02,0.05,0.10",
        "--max_macro_f1_drop", "$MaxMacroF1Drop",
        "--max_micro_f1_drop", "$MaxMicroF1Drop",
        "--max_exact_match_drop", "$MaxExactMatchDrop",
        "--max_overall_harm_fraction", "$MaxOverallHarmFraction",
        "--min_parent_stop_fraction", "$MinParentStopFraction",
        "--batch_size", "$BatchSize",
        "--device", $Device,
        "--out_dir", $TuningOut
    )
    python @TuneArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Whole-parent policy tuning failed."
    }
}
else {
    Write-Host "[3/4] Validation tuning skipped; reusing frozen policy." -ForegroundColor DarkYellow
    if (-not (Test-Path $FrozenPolicy)) {
        throw "SkipTuning was used, but frozen policy does not exist: $FrozenPolicy"
    }
}

Write-Host "[4/4] Running genuine whole-parent holdout evaluation..." -ForegroundColor Yellow
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
    "--timing_repeats", "$TimingRepeats",
    "--torch_threads", "$TorchThreads",
    "--device", $Device,
    "--out_dir", $HoldoutOut
)
python @EvaluateArgs
if ($LASTEXITCODE -ne 0) {
    throw "Whole-parent holdout evaluation failed."
}

Write-Host ""
Write-Host "V0.15 whole-parent risk-control experiment completed." -ForegroundColor Green
Write-Host "Frozen policy: $FrozenPolicy"
Write-Host "Validation:    $TuningOut"
Write-Host "Holdout:       $HoldoutOut"
Write-Host ""
Write-Host "The corrected holdout used frozen validation-only OOF policies; no holdout retuning occurred." -ForegroundColor DarkYellow
Write-Host "For publication timing, rerun with -TimingRepeats 30." -ForegroundColor DarkYellow
