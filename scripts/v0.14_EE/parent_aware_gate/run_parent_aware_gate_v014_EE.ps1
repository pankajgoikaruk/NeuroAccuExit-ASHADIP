param(
    [string]$RunDir = "",
    [string]$Device = "cpu",
    [int]$BatchSize = 128,
    [int]$CvFolds = 5,
    [int]$TimingRepeats = 10,
    [int]$TorchThreads = 1,
    [double]$MaxMacroF1Drop = 0.01,
    [double]$MinSourceFraction = 0.01,
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

$RunName = "main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845"
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $Matches = @(
        Get-ChildItem "human_talk_workspace" -Directory -Recurse `
            -Filter $RunName -ErrorAction SilentlyContinue
    )
    if ($Matches.Count -eq 0) {
        throw "Could not find canonical run '$RunName'. Pass -RunDir explicitly."
    }
    if ($Matches.Count -gt 1) {
        throw "Multiple canonical runs found. Pass -RunDir explicitly."
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
        Write-Host "[INFO] Parent-aware targets and selection use frozen LATS-v2." -ForegroundColor DarkYellow
    }
}
else {
    $ResolvedThresholdMode = $ThresholdMode
}

$Root = "scripts\v0.14_EE\parent_aware_gate"
$TuneScript = "$Root\tune_parent_aware_gate_v014.py"
$EvaluateScript = "$Root\evaluate_parent_aware_gate_v014.py"
$VerifyScript = "scripts\v0.11_EE\fixed_policy\verify_checkpoint_equivalence_v011.py"

$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$FeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"

$OutputRoot = "human_talk_workspace\active_budget_anytime_exit_v0.3\v0.14_EE\parent_aware_gate"
$TuningOut = "$OutputRoot\validation_tuning"
$HoldoutOut = "$OutputRoot\corrected_holdout_evaluation"
$FrozenPolicy = "$TuningOut\frozen_parent_aware_gate_v014.json"
$EquivalenceJson = "$OutputRoot\checkpoint_staged_equivalence_precheck.json"

$Required = @(
    $RunDir,
    (Join-Path $RunDir "ckpt\best.pt"),
    $HoldoutManifest,
    $FeaturesRoot,
    $LabelsJson,
    $LatsConfig,
    "models\anytime_exit_net.py",
    "policies\parent_aware_adaptive_gate.py",
    "tests\test_parent_aware_adaptive_gate.py",
    $TuneScript,
    $EvaluateScript,
    $VerifyScript
)
foreach ($Path in $Required) {
    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $TuningOut, $HoldoutOut | Out-Null

Write-Host ""
Write-Host "=== NeuroAccuExit v0.14 Parent-Aware Adaptive Gate ===" -ForegroundColor Cyan
Write-Host "Branch:                 $CurrentBranch"
Write-Host "Canonical run:          $RunDir"
Write-Host "Device:                 $Device"
Write-Host "Threshold mode:         $ResolvedThresholdMode"
Write-Host "Grouped CV folds:       $CvFolds"
Write-Host "Macro-F1 drop limit:    $MaxMacroF1Drop"
Write-Host "Timing repeats:         $TimingRepeats"
Write-Host "Torch/BLAS threads:     $TorchThreads"
Write-Host "Validation output:      $TuningOut"
Write-Host "Holdout output:         $HoldoutOut"
Write-Host ""

if (-not $SkipPrechecks) {
    Write-Host "[1/4] Running staged and parent-aware gate tests..." -ForegroundColor Yellow
    python -m unittest `
        tests.test_anytime_exit_net `
        tests.test_parent_aware_adaptive_gate `
        -v
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed."
    }

    Write-Host "[2/4] Rechecking checkpoint staged equivalence..." -ForegroundColor Yellow
    python $VerifyScript `
        --run_dir $RunDir `
        --labels_json $LabelsJson `
        --holdout_manifest $HoldoutManifest `
        --features_root $FeaturesRoot `
        --sample_count 8 `
        --device $Device `
        --out_json $EquivalenceJson
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint staged-equivalence verification failed."
    }
}
else {
    Write-Host "[1/4] Unit tests skipped." -ForegroundColor DarkYellow
    Write-Host "[2/4] Checkpoint equivalence skipped." -ForegroundColor DarkYellow
}

if (-not $SkipTuning) {
    Write-Host "[3/4] Training grouped-OOF parent-aware gates..." -ForegroundColor Yellow
    python $TuneScript `
        --run_dir $RunDir `
        --labels_json $LabelsJson `
        --lats_config_json $LatsConfig `
        --parent_id_col parent_clip_id `
        --threshold_mode $ResolvedThresholdMode `
        --cv_folds $CvFolds `
        --unsafe_recall_grid "0.80,0.90,0.95,0.98" `
        --threshold_scale_grid "0.75,1.00,1.25" `
        --expected_harm_grid "0.10,0.20,0.30,0.50,1.01" `
        --minimum_unsafe_examples 3 `
        --max_macro_f1_drop $MaxMacroF1Drop `
        --min_source_fraction $MinSourceFraction `
        --one_sided_z 1.645 `
        --batch_size $BatchSize `
        --device $Device `
        --out_dir $TuningOut
    if ($LASTEXITCODE -ne 0) {
        throw "Parent-aware gate tuning failed."
    }
}
else {
    Write-Host "[3/4] Tuning skipped; reusing frozen policy." -ForegroundColor DarkYellow
    if (-not (Test-Path $FrozenPolicy)) {
        throw "Frozen policy not found: $FrozenPolicy"
    }
}

Write-Host "[4/4] Running genuine staged holdout evaluation..." -ForegroundColor Yellow
python $EvaluateScript `
    --run_dir $RunDir `
    --policy_json $FrozenPolicy `
    --holdout_manifest $HoldoutManifest `
    --features_root $FeaturesRoot `
    --labels_json $LabelsJson `
    --lats_config_json $LatsConfig `
    --parent_id_col parent_clip_id `
    --batch_size $BatchSize `
    --timing_repeats $TimingRepeats `
    --torch_threads $TorchThreads `
    --device $Device `
    --out_dir $HoldoutOut
if ($LASTEXITCODE -ne 0) {
    throw "Parent-aware gate holdout evaluation failed."
}

Write-Host ""
Write-Host "V0.14 parent-aware gate experiment completed." -ForegroundColor Green
Write-Host "Frozen policy: $FrozenPolicy"
Write-Host "Validation:    $TuningOut"
Write-Host "Holdout:       $HoldoutOut"
Write-Host ""
Write-Host "The corrected holdout used frozen OOF-selected label-specific thresholds; no holdout retuning occurred." -ForegroundColor DarkYellow
Write-Host "For publication timing, rerun with -TimingRepeats 30." -ForegroundColor DarkYellow
