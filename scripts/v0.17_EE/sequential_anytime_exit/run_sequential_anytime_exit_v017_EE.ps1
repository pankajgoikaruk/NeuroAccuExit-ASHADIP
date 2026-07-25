param(
    [string]$RunDir3 = "",
    [string]$RunDir5 = "",
    [string]$Device = "cpu",
    [int]$BatchSize = 128,
    [int]$PopulationSize = 96,
    [int]$Generations = 60,
    [int]$CvFolds = 5,
    [int]$TimingRepeats = 10,
    [int]$TorchThreads = 1,
    [double]$MaxMacroF1Drop = 0.01,
    [double]$MaxMicroF1Drop = 0.005,
    [double]$MaxExactMatchDrop = 0.01,
    [double]$MaxHammingIncrease = 0.002,
    [double]$MinTotalEarlyFraction = 0.02,
    [double]$MinExit1Fraction = 0.005,
    [double]$SafetyFraction = 0.75,
    [ValidateSet("auto", "tuned_per_exit", "final_exit_tuned", "fixed_0p5")]
    [string]$ThresholdMode = "auto",
    [switch]$Run3Only,
    [switch]$SkipPrechecks,
    [switch]$SkipTuning
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Test-Path ".git")) {
    throw "Run this script from the NeuroAccuExit-ASHADIP repository root."
}
$ExpectedBranch = "active_budget_anytime_exit_v0.4"
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

function Resolve-SingleRun([string]$Provided, [string]$Pattern, [string]$Description) {
    if (-not [string]::IsNullOrWhiteSpace($Provided)) {
        if (-not (Test-Path $Provided)) {
            throw "$Description run directory was not found: $Provided"
        }
        return (Resolve-Path $Provided).Path
    }
    $Matches = @(
        Get-ChildItem "human_talk_workspace" -Directory -Recurse -Filter $Pattern `
            -ErrorAction SilentlyContinue
    )
    if ($Matches.Count -eq 1) {
        return $Matches[0].FullName
    }
    if ($Matches.Count -gt 1) {
        $Paths = ($Matches.FullName -join "`n")
        throw "Multiple $Description runs found. Pass the run explicitly:`n$Paths"
    }
    return ""
}

$RunDir3 = Resolve-SingleRun `
    $RunDir3 `
    "main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845" `
    "3-exit"
if ([string]::IsNullOrWhiteSpace($RunDir3)) {
    throw "Could not find the canonical 3-exit run. Pass -RunDir3 explicitly."
}

if (-not $Run3Only) {
    $RunDir5 = Resolve-SingleRun `
        $RunDir5 `
        "main_v010_human_corrected_balanced_5exit_no_hint_*" `
        "fair 5-exit"
    if ([string]::IsNullOrWhiteSpace($RunDir5)) {
        throw @"
A fair 5-exit checkpoint was not found.
Pass -RunDir5 pointing to a 5-exit checkpoint trained with the SAME 10-label dataset,
manifest, feature cache and preprocessing as the canonical 3-exit model.
The older tata_2_5exit_weakclip checkpoint uses a different 12-label problem and is
therefore not accepted for the primary architecture comparison.
Use -Run3Only to run the 3-exit experiment first.
"@
    }
}

$ThresholdComparison3 = Join-Path $RunDir3 "threshold_tuning\threshold_comparison.json"
$ThresholdComparison5 = if ($Run3Only) { "" } else { Join-Path $RunDir5 "threshold_tuning\threshold_comparison.json" }
if ($ThresholdMode -eq "auto") {
    $ResolvedThresholdMode = if (
        (Test-Path $ThresholdComparison3) -and
        ($Run3Only -or (Test-Path $ThresholdComparison5))
    ) { "tuned_per_exit" } else { "fixed_0p5" }
}
else {
    $ResolvedThresholdMode = $ThresholdMode
}
if ($ResolvedThresholdMode -eq "tuned_per_exit") {
    if (-not (Test-Path $ThresholdComparison3)) {
        throw "3-exit tuned thresholds were not found: $ThresholdComparison3"
    }
    if (-not $Run3Only -and -not (Test-Path $ThresholdComparison5)) {
        throw "5-exit tuned thresholds were not found: $ThresholdComparison5"
    }
}

$ScriptRoot = "scripts\v0.17_EE\sequential_anytime_exit"
$VerifyScript = "scripts\v0.11_EE\fixed_policy\verify_checkpoint_equivalence_v011.py"
$TuneScript = "$ScriptRoot\tune_sequential_anytime_exit_v017.py"
$EvaluateScript = "$ScriptRoot\evaluate_sequential_anytime_exit_v017.py"
$CompareScript = "$ScriptRoot\compare_sequential_architectures_v017.py"
$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$FeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"
$OutputRoot = "human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit"
$Out3Tune = "$OutputRoot\3exit\validation_tuning"
$Out3Holdout = "$OutputRoot\3exit\corrected_holdout_evaluation"
$Policy3 = "$Out3Tune\frozen_sequential_policy_3exit_v017.json"
$Out5Tune = "$OutputRoot\5exit\validation_tuning"
$Out5Holdout = "$OutputRoot\5exit\corrected_holdout_evaluation"
$Policy5 = "$Out5Tune\frozen_sequential_policy_5exit_v017.json"
$CombinedOut = "$OutputRoot\architecture_comparison"

$RequiredPaths = @(
    $RunDir3,
    (Join-Path $RunDir3 "ckpt\best.pt"),
    $HoldoutManifest,
    $FeaturesRoot,
    $LabelsJson,
    $LatsConfig,
    "models\anytime_exit_net.py",
    "policies\sequential_anytime_exit.py",
    "tests\test_sequential_anytime_exit.py",
    $VerifyScript,
    $TuneScript,
    $EvaluateScript,
    $CompareScript
)
if (-not $Run3Only) {
    $RequiredPaths += @($RunDir5, (Join-Path $RunDir5 "ckpt\best.pt"))
}
foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}
New-Item -ItemType Directory -Force -Path $Out3Tune, $Out3Holdout | Out-Null
if (-not $Run3Only) {
    New-Item -ItemType Directory -Force -Path $Out5Tune, $Out5Holdout, $CombinedOut | Out-Null
}

Write-Host ""
Write-Host "=== NeuroAccuExit v0.17 Sequential Active-Budget Anytime Exit ===" -ForegroundColor Cyan
Write-Host "Branch:                       $CurrentBranch"
Write-Host "3-exit run:                   $RunDir3"
Write-Host "5-exit run:                   $(if ($Run3Only) { 'skipped' } else { $RunDir5 })"
Write-Host "Sequential routes:            1->2->3 and 1->2->3->4->5"
Write-Host "Threshold mode:               $ResolvedThresholdMode"
Write-Host "Population / generations:     $PopulationSize / $Generations"
Write-Host "Safety-buffered Pareto ratio: $SafetyFraction"
Write-Host "Minimum Exit-1 fraction:      $MinExit1Fraction"
Write-Host "Timing repeats:               $TimingRepeats"
Write-Host ""

if (-not $SkipPrechecks) {
    Write-Host "[1/5] Running staged and sequential policy tests..." -ForegroundColor Yellow
    python -m unittest `
        tests.test_anytime_exit_net `
        tests.test_sequential_anytime_exit `
        -v
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }

    Write-Host "[2/5] Verifying staged equivalence for each architecture..." -ForegroundColor Yellow
    $Verify3 = @(
        $VerifyScript,
        "--run_dir", $RunDir3,
        "--labels_json", $LabelsJson,
        "--holdout_manifest", $HoldoutManifest,
        "--features_root", $FeaturesRoot,
        "--sample_count", "8",
        "--device", $Device,
        "--out_json", "$OutputRoot\3exit\checkpoint_staged_equivalence.json"
    )
    python @Verify3
    if ($LASTEXITCODE -ne 0) { throw "3-exit staged equivalence failed." }
    if (-not $Run3Only) {
        $Verify5 = @(
            $VerifyScript,
            "--run_dir", $RunDir5,
            "--labels_json", $LabelsJson,
            "--holdout_manifest", $HoldoutManifest,
            "--features_root", $FeaturesRoot,
            "--sample_count", "8",
            "--device", $Device,
            "--out_json", "$OutputRoot\5exit\checkpoint_staged_equivalence.json"
        )
        python @Verify5
        if ($LASTEXITCODE -ne 0) { throw "5-exit staged equivalence failed." }
    }
}
else {
    Write-Host "[1/5] Unit tests skipped." -ForegroundColor DarkYellow
    Write-Host "[2/5] Staged equivalence skipped." -ForegroundColor DarkYellow
}

function Invoke-Tuning(
    [string]$RunDir,
    [string]$Output,
    [string]$FrozenPolicy
) {
    if ($SkipTuning) {
        if (-not (Test-Path $FrozenPolicy)) {
            throw "SkipTuning was used, but frozen policy does not exist: $FrozenPolicy"
        }
        return
    }
    $TuneArgs = @(
        $TuneScript,
        "--run_dir", $RunDir,
        "--labels_json", $LabelsJson,
        "--lats_config_json", $LatsConfig,
        "--parent_id_col", "parent_clip_id",
        "--threshold_mode", $ResolvedThresholdMode,
        "--population_size", "$PopulationSize",
        "--generations", "$Generations",
        "--cv_folds", "$CvFolds",
        "--max_macro_f1_drop", "$MaxMacroF1Drop",
        "--max_micro_f1_drop", "$MaxMicroF1Drop",
        "--max_exact_match_drop", "$MaxExactMatchDrop",
        "--max_hamming_increase", "$MaxHammingIncrease",
        "--min_total_early_fraction", "$MinTotalEarlyFraction",
        "--min_exit1_fraction", "$MinExit1Fraction",
        "--safety_fraction", "$SafetyFraction",
        "--batch_size", "$BatchSize",
        "--device", $Device,
        "--out_dir", $Output
    )
    python @TuneArgs
    if ($LASTEXITCODE -ne 0) { throw "Sequential policy tuning failed for $RunDir" }
}

Write-Host "[3/5] Tuning sequential policies on validation only..." -ForegroundColor Yellow
Invoke-Tuning $RunDir3 $Out3Tune $Policy3
if (-not $Run3Only) { Invoke-Tuning $RunDir5 $Out5Tune $Policy5 }

function Invoke-Holdout(
    [string]$RunDir,
    [string]$FrozenPolicy,
    [string]$Output
) {
    $EvalArgs = @(
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
        "--out_dir", $Output
    )
    python @EvalArgs
    if ($LASTEXITCODE -ne 0) { throw "Sequential holdout evaluation failed for $RunDir" }
}

Write-Host "[4/5] Running genuine sequential holdout evaluation and ablations..." -ForegroundColor Yellow
Invoke-Holdout $RunDir3 $Policy3 $Out3Holdout
if (-not $Run3Only) { Invoke-Holdout $RunDir5 $Policy5 $Out5Holdout }

if (-not $Run3Only) {
    Write-Host "[5/5] Auditing fairness and comparing 3 exits versus 5 exits..." -ForegroundColor Yellow
    $CompareArgs = @(
        $CompareScript,
        "--policy_3", $Policy3,
        "--comparison_3", "$Out3Holdout\v017_3exit_holdout_comparison.csv",
        "--policy_5", $Policy5,
        "--comparison_5", "$Out5Holdout\v017_5exit_holdout_comparison.csv",
        "--out_dir", $CombinedOut
    )
    python @CompareArgs
    if ($LASTEXITCODE -ne 0) { throw "Cross-architecture comparison failed." }
}
else {
    Write-Host "[5/5] Cross-architecture comparison skipped by -Run3Only." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "V0.17 sequential anytime-exit experiment completed." -ForegroundColor Green
Write-Host "3-exit outputs: $OutputRoot\3exit"
if (-not $Run3Only) {
    Write-Host "5-exit outputs: $OutputRoot\5exit"
    Write-Host "Combined tables: $CombinedOut"
}
Write-Host "The primary policy evaluates Exit 1 and every later non-final exit." -ForegroundColor DarkYellow
Write-Host "For publication timing, rerun with -TimingRepeats 30." -ForegroundColor DarkYellow
