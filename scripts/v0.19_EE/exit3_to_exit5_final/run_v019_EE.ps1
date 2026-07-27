param(
    [string]$RunDir5 = "",
    [string]$PolicyDevice = "cpu",
    [int]$PopulationSize = 160,
    [int]$Generations = 100,
    [int]$CvFolds = 5,
    [int]$TimingRepeats = 30,
    [int]$TorchThreads = 1,
    [double]$MaxMacroF1Drop = 0.01,
    [double]$MaxMicroF1Drop = 0.005,
    [double]$MaxExactMatchDrop = 0.01,
    [double]$MaxHammingIncrease = 0.002,
    [double]$MinEarlyFraction = 0.05,
    [double]$SafetyFraction = 0.35,
    [double]$MinFlopsSavedPct = 5.0,
    [switch]$SkipPrechecks,
    [switch]$SkipPolicyTuning
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not (Test-Path ".git")) { throw "Run this script from the repository root." }
$ExpectedBranch = "active_budget_anytime_exit_v0.4"
$CurrentBranch = (git branch --show-current | Out-String).Trim()
if ($CurrentBranch -ne $ExpectedBranch) { throw "Switch to '$ExpectedBranch' before running v0.19." }

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "$TorchThreads"
$env:MKL_NUM_THREADS = "$TorchThreads"
$env:OPENBLAS_NUM_THREADS = "$TorchThreads"
$env:NUMEXPR_NUM_THREADS = "$TorchThreads"

function Resolve-Fair5([string]$Provided) {
    if (-not [string]::IsNullOrWhiteSpace($Provided)) {
        if (-not (Test-Path $Provided)) { throw "Fair five-exit run not found: $Provided" }
        return (Resolve-Path $Provided).Path
    }
    $matches = @(Get-ChildItem . -Directory -Recurse -Filter "main_v018_human_corrected_balanced_5exit_no_hint_auxmatched_*" -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "ckpt\best.pt") } |
        Sort-Object LastWriteTime -Descending)
    if ($matches.Count -lt 1) { throw "No fair v0.18 five-exit run found. Pass -RunDir5 explicitly." }
    return $matches[0].FullName
}

$RunDir5 = Resolve-Fair5 $RunDir5
$Root = "human_talk_workspace\active_budget_anytime_exit_v0.4\v0.19_EE\exit3_to_exit5_final"
$TuneDir = "$Root\validation_tuning"
$EvalDir = "$Root\corrected_holdout_evaluation"
$DecisionDir = "$Root\final_decision"
$ScriptRoot = "scripts\v0.19_EE\exit3_to_exit5_final"
$Policy = "$TuneDir\frozen_exit3_to_exit5_policy_v019.json"
$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$HoldoutFeatures = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"
$VerifyScript = "scripts\v0.11_EE\fixed_policy\verify_checkpoint_equivalence_v011.py"
$TuneScript = "$ScriptRoot\tune_exit3_to_exit5_v019.py"
$EvaluateScript = "scripts\v0.18_EE\fair_sequential_anytime_exit\evaluate_strict_sequential_v018.py"
$FinaliseScript = "$ScriptRoot\finalise_v019.py"

$Required = @($RunDir5, (Join-Path $RunDir5 "ckpt\best.pt"), $HoldoutManifest, $HoldoutFeatures, $LabelsJson, $LatsConfig,
    $TuneScript, $EvaluateScript, $FinaliseScript, "policies\exit3_to_exit5_v019.py", "tests\test_exit3_to_exit5_v019.py")
foreach ($Path in $Required) { if (-not (Test-Path $Path)) { throw "Required path not found: $Path" } }
New-Item -ItemType Directory -Force -Path $TuneDir, $EvalDir, $DecisionDir | Out-Null

Write-Host ""
Write-Host "=== NeuroAccuExit v0.19 Final Targeted Exit 3 -> Exit 5 ===" -ForegroundColor Cyan
Write-Host "Fair five-exit run: $RunDir5"
Write-Host "Stopping exits:      Exit 3 only"
Write-Host "Fallback:            Exit 5"
Write-Host "Population/gens:     $PopulationSize / $Generations"
Write-Host "Safety fraction:     $SafetyFraction"
Write-Host "Timing repeats:      $TimingRepeats"

if (-not $SkipPrechecks) {
    Write-Host "[1/4] Running focused policy tests and staged equivalence..." -ForegroundColor Yellow
    python -m unittest tests.test_exit3_to_exit5_v019 tests.test_anytime_exit_net -v
    if ($LASTEXITCODE -ne 0) { throw "v0.19 tests failed." }
    python $VerifyScript --run_dir $RunDir5 --labels_json $LabelsJson --holdout_manifest $HoldoutManifest `
        --features_root $HoldoutFeatures --sample_count 8 --device $PolicyDevice --out_json "$Root\checkpoint_staged_equivalence.json"
    if ($LASTEXITCODE -ne 0) { throw "Five-exit staged equivalence failed." }
} else { Write-Host "[1/4] Prechecks skipped." -ForegroundColor DarkYellow }

Write-Host "[2/4] Tuning only the Exit-3 stopping gate on validation..." -ForegroundColor Yellow
if (-not $SkipPolicyTuning) {
    python $TuneScript --run_dir $RunDir5 --labels_json $LabelsJson --lats_config_json $LatsConfig `
        --threshold_mode fixed_0p5 --population_size $PopulationSize --generations $Generations --cv_folds $CvFolds `
        --max_macro_f1_drop $MaxMacroF1Drop --max_micro_f1_drop $MaxMicroF1Drop `
        --max_exact_match_drop $MaxExactMatchDrop --max_hamming_increase $MaxHammingIncrease `
        --min_total_early_fraction $MinEarlyFraction --safety_fraction $SafetyFraction `
        --device $PolicyDevice --out_dir $TuneDir
    if ($LASTEXITCODE -ne 0) { throw "v0.19 tuning failed." }
} elseif (-not (Test-Path $Policy)) { throw "Frozen v0.19 policy not found: $Policy" }

Write-Host "[3/4] Running genuine corrected-holdout Exit-3/Exit-5 evaluation..." -ForegroundColor Yellow
python $EvaluateScript --run_dir $RunDir5 --policy_json $Policy --holdout_manifest $HoldoutManifest `
    --features_root $HoldoutFeatures --labels_json $LabelsJson --lats_config_json $LatsConfig `
    --batch_size 128 --timing_repeats $TimingRepeats --torch_threads $TorchThreads `
    --device $PolicyDevice --out_dir $EvalDir
if ($LASTEXITCODE -ne 0) { throw "v0.19 holdout evaluation failed." }

Write-Host "[4/4] Applying the final stop-or-finalise rule..." -ForegroundColor Yellow
python $FinaliseScript --comparison_csv "$EvalDir\v018_5exit_holdout_comparison.csv" `
    --policy_json $Policy --out_dir $DecisionDir --min_flops_saved_pct $MinFlopsSavedPct
if ($LASTEXITCODE -ne 0) { throw "v0.19 final decision generation failed." }

Write-Host ""
Write-Host "V0.19 completed. Read: $DecisionDir\V019_FINAL_DECISION.md" -ForegroundColor Green
