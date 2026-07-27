param(
    [string]$RunDir5 = "",
    [string]$Device = "cpu",
    [int]$PopulationSize = 128,
    [int]$Generations = 80,
    [int]$CvFolds = 5,
    [int]$TimingRepeats = 30,
    [int]$TorchThreads = 1,
    [double]$TargetFlopsSavedPct = 7.0,
    [double]$InternalMaxMacroDrop = 0.005,
    [double]$InternalMaxMicroDrop = 0.0025,
    [double]$InternalMaxExactDrop = 0.005,
    [double]$InternalMaxHammingIncrease = 0.001,
    [double]$DeploymentMaxMacroDrop = 0.01,
    [double]$DeploymentMaxMicroDrop = 0.005,
    [double]$DeploymentMaxExactDrop = 0.01,
    [double]$DeploymentMaxHammingIncrease = 0.002,
    [double]$MinExit3Fraction = 0.02,
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

function Test-Fair5Run([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate)) { return $false }
    $ConfigPath = Join-Path $Candidate "config_used.json"
    $CheckpointPath = Join-Path $Candidate "ckpt\best.pt"
    if (-not (Test-Path $ConfigPath) -or -not (Test-Path $CheckpointPath)) { return $false }
    try {
        $Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
        $Taps = @($Config.tap_blocks | ForEach-Object { [int]$_ })
        return ([int]$Config.num_exits -eq 5 -and ($Taps -join ",") -eq "1,2,3,4")
    }
    catch { return $false }
}

function Resolve-Fair5([string]$Provided) {
    if (-not [string]::IsNullOrWhiteSpace($Provided)) {
        if (-not (Test-Fair5Run $Provided)) {
            throw "The supplied run is not a valid fair five-exit checkpoint: $Provided"
        }
        return (Resolve-Path $Provided).Path
    }

    $Candidates = New-Object System.Collections.Generic.List[string]
    $StandardRoot = "human_talk_workspace\active_budget_anytime_exit_v0.4\v0.18_EE\fair_sequential_anytime_exit\fair_5exit_training\runs"
    if (Test-Path $StandardRoot) {
        Get-ChildItem $StandardRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            ForEach-Object { $Candidates.Add($_.FullName) }
    }
    Get-ChildItem "." -Directory -Filter "v018_fair5_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { $Candidates.Add($_.FullName) }

    foreach ($Candidate in $Candidates) {
        if (Test-Fair5Run $Candidate) { return (Resolve-Path $Candidate).Path }
    }
    throw "A fair v0.18 five-exit run could not be resolved. Pass -RunDir5 explicitly."
}

$RunDir5 = Resolve-Fair5 $RunDir5
$ScriptRoot = "scripts\v0.19_EE\targeted_exit3_to_exit5"
$OutputRoot = "human_talk_workspace\active_budget_anytime_exit_v0.4\v0.19_EE\targeted_exit3_to_exit5"
$TuneOut = "$OutputRoot\validation_tuning"
$HoldoutOut = "$OutputRoot\corrected_holdout_evaluation"
$FinalOut = "$OutputRoot\final_comparison"
$Policy = "$TuneOut\frozen_targeted_exit3_to_exit5_policy_v019.json"
$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$HoldoutFeatures = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"
$VerifyScript = "scripts\v0.11_EE\fixed_policy\verify_checkpoint_equivalence_v011.py"
$TuneScript = "$ScriptRoot\tune_targeted_exit3_to_exit5_v019.py"
$EvaluateScript = "$ScriptRoot\evaluate_targeted_exit3_to_exit5_v019.py"
$CompareScript = "$ScriptRoot\compare_v019.py"
$PreviousV018 = "human_talk_workspace\active_budget_anytime_exit_v0.4\v0.18_EE\fair_sequential_anytime_exit\5exit\corrected_holdout_evaluation\v018_5exit_holdout_comparison.csv"

$RequiredPaths = @(
    $RunDir5, (Join-Path $RunDir5 "ckpt\best.pt"),
    $HoldoutManifest, $HoldoutFeatures, $LabelsJson, $LatsConfig,
    "models\anytime_exit_net.py",
    "policies\targeted_exit3_to_exit5_v019.py",
    "tests\test_targeted_exit3_to_exit5_v019.py",
    $VerifyScript, $TuneScript, $EvaluateScript, $CompareScript
)
foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) { throw "Required path not found: $Path" }
}
New-Item -ItemType Directory -Force -Path $TuneOut, $HoldoutOut, $FinalOut | Out-Null

Write-Host ""
Write-Host "=== NeuroAccuExit v0.19 Final Targeted Exit 3 -> Exit 5 ===" -ForegroundColor Cyan
Write-Host "Branch:                         $CurrentBranch"
Write-Host "Fair five-exit run:             $RunDir5"
Write-Host "Decision route:                 Exit 3 -> Exit 5"
Write-Host "Target FLOPs saved:             $TargetFlopsSavedPct%"
Write-Host "Internal Macro/Micro limits:    $InternalMaxMacroDrop / $InternalMaxMicroDrop"
Write-Host "Internal Exact/Hamming limits:  $InternalMaxExactDrop / $InternalMaxHammingIncrease"
Write-Host "Deployment Macro/Micro limits:  $DeploymentMaxMacroDrop / $DeploymentMaxMicroDrop"
Write-Host "Deployment Exact/Hamming limits:$DeploymentMaxExactDrop / $DeploymentMaxHammingIncrease"
Write-Host "Population / generations:       $PopulationSize / $Generations"
Write-Host "Timing repeats:                 $TimingRepeats"
Write-Host ""

if (-not $SkipPrechecks) {
    Write-Host "[1/5] Running staged and targeted-policy tests..." -ForegroundColor Yellow
    python -m unittest `
        tests.test_anytime_exit_net `
        tests.test_targeted_exit3_to_exit5_v019 `
        -v
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }

    Write-Host "[2/5] Verifying five-exit staged equivalence..." -ForegroundColor Yellow
    python $VerifyScript `
        --run_dir $RunDir5 `
        --labels_json $LabelsJson `
        --holdout_manifest $HoldoutManifest `
        --features_root $HoldoutFeatures `
        --sample_count 8 `
        --device $Device `
        --out_json "$OutputRoot\checkpoint_staged_equivalence.json"
    if ($LASTEXITCODE -ne 0) { throw "Five-exit staged equivalence failed." }
}
else {
    Write-Host "[1/5] Unit tests skipped." -ForegroundColor DarkYellow
    Write-Host "[2/5] Staged equivalence skipped." -ForegroundColor DarkYellow
}

if (-not $SkipTuning) {
    Write-Host "[3/5] Tuning the final targeted policy on validation only..." -ForegroundColor Yellow
    python $TuneScript `
        --run_dir $RunDir5 `
        --labels_json $LabelsJson `
        --lats_config_json $LatsConfig `
        --threshold_mode fixed_0p5 `
        --population_size $PopulationSize `
        --generations $Generations `
        --cv_folds $CvFolds `
        --internal_max_macro_drop $InternalMaxMacroDrop `
        --internal_max_micro_drop $InternalMaxMicroDrop `
        --internal_max_exact_drop $InternalMaxExactDrop `
        --internal_max_hamming_increase $InternalMaxHammingIncrease `
        --deployment_max_macro_drop $DeploymentMaxMacroDrop `
        --deployment_max_micro_drop $DeploymentMaxMicroDrop `
        --deployment_max_exact_drop $DeploymentMaxExactDrop `
        --deployment_max_hamming_increase $DeploymentMaxHammingIncrease `
        --min_exit3_fraction $MinExit3Fraction `
        --target_flops_saved_pct $TargetFlopsSavedPct `
        --device $Device `
        --out_dir $TuneOut
    if ($LASTEXITCODE -ne 0) { throw "V0.19 targeted policy tuning failed." }
}
else {
    Write-Host "[3/5] Reusing frozen targeted policy." -ForegroundColor DarkYellow
    if (-not (Test-Path $Policy)) { throw "Frozen policy not found: $Policy" }
}

Write-Host "[4/5] Running genuine targeted corrected-holdout evaluation..." -ForegroundColor Yellow
python $EvaluateScript `
    --run_dir $RunDir5 `
    --policy_json $Policy `
    --holdout_manifest $HoldoutManifest `
    --features_root $HoldoutFeatures `
    --labels_json $LabelsJson `
    --lats_config_json $LatsConfig `
    --batch_size 128 `
    --timing_repeats $TimingRepeats `
    --torch_threads $TorchThreads `
    --device $Device `
    --out_dir $HoldoutOut
if ($LASTEXITCODE -ne 0) { throw "V0.19 targeted holdout evaluation failed." }

Write-Host "[5/5] Creating final EE decision and paper tables..." -ForegroundColor Yellow
$CompareArgs = @(
    $CompareScript,
    "--comparison", "$HoldoutOut\v019_targeted_holdout_comparison.csv",
    "--out_dir", $FinalOut
)
if (Test-Path $PreviousV018) { $CompareArgs += @("--previous_v018", $PreviousV018) }
python @CompareArgs
if ($LASTEXITCODE -ne 0) { throw "V0.19 final comparison generation failed." }

Write-Host ""
Write-Host "V0.19 final targeted experiment completed." -ForegroundColor Green
Write-Host "Frozen policy: $Policy"
Write-Host "Holdout:       $HoldoutOut"
Write-Host "Final decision:$FinalOut\v019_final_decision.json"
Write-Host ""
Write-Host "This is the final targeted EE experiment. Use the generated decision JSON to finalise or stop EE development." -ForegroundColor DarkYellow
