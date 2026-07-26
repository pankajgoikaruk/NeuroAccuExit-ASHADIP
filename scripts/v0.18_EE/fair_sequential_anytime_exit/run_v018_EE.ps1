param(
    [string]$RunDir3 = "",
    [string]$RunDir5 = "",
    [string]$TrainingDevice = "cpu",
    [string]$PolicyDevice = "cpu",
    [int]$PopulationSize = 112,
    [int]$Generations = 70,
    [int]$CvFolds = 5,
    [int]$TimingRepeats = 30,
    [int]$TorchThreads = 1,
    [double]$MaxMacroF1Drop = 0.01,
    [double]$MaxMicroF1Drop = 0.005,
    [double]$MaxExactMatchDrop = 0.01,
    [double]$MaxHammingIncrease = 0.002,
    [double]$MinTotalEarlyFraction = 0.02,
    [double]$MinExit1Fraction = 0.0025,
    [double]$SafetyFraction = 0.50,
    [switch]$ForceRetrain5,
    [switch]$SkipTraining5,
    [switch]$SkipPrechecks,
    [switch]$SkipPolicyTuning
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

function Resolve-Canonical3([string]$Provided) {
    if (-not [string]::IsNullOrWhiteSpace($Provided)) {
        if (-not (Test-Path $Provided)) { throw "3-exit run was not found: $Provided" }
        return (Resolve-Path $Provided).Path
    }
    $matches = @(Get-ChildItem "human_talk_workspace" -Directory -Recurse `
        -Filter "main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845" `
        -ErrorAction SilentlyContinue)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one canonical 3-exit run; found $($matches.Count). Pass -RunDir3 explicitly."
    }
    return $matches[0].FullName
}

$RunDir3 = Resolve-Canonical3 $RunDir3
$Config3Path = Join-Path $RunDir3 "config_used.json"
if (-not (Test-Path $Config3Path)) { throw "Canonical config not found: $Config3Path" }
$Config3 = Get-Content $Config3Path -Raw | ConvertFrom-Json

$ScriptRoot = "scripts\v0.18_EE\fair_sequential_anytime_exit"
$OutputRoot = "human_talk_workspace\active_budget_anytime_exit_v0.4\v0.18_EE\fair_sequential_anytime_exit"
$TrainingRoot = "$OutputRoot\fair_5exit_training"
$Runs5Root = "$TrainingRoot\runs"
$TrainingAudit = "$OutputRoot\fair_training_audit.json"
$HoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$HoldoutFeatures = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$LabelsJson = "configs\human_talk_10label_schema.json"
$LatsConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"
$VerifyScript = "scripts\v0.11_EE\fixed_policy\verify_checkpoint_equivalence_v011.py"
$AuditScript = "$ScriptRoot\audit_fair_training_v018.py"
$TuneScript = "$ScriptRoot\tune_strict_sequential_v018.py"
$EvaluateScript = "$ScriptRoot\evaluate_strict_sequential_v018.py"
$CompareScript = "$ScriptRoot\compare_v018.py"

New-Item -ItemType Directory -Force -Path $TrainingRoot, $Runs5Root | Out-Null

function Find-Fair5Run {
    $matches = @(Get-ChildItem $Runs5Root -Directory `
        -Filter "main_v018_human_corrected_balanced_5exit_no_hint_auxmatched_*" `
        -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    if ($matches.Count -gt 0) { return $matches[0].FullName }
    return ""
}

if (-not [string]::IsNullOrWhiteSpace($RunDir5)) {
    if (-not (Test-Path $RunDir5)) { throw "5-exit run was not found: $RunDir5" }
    $RunDir5 = (Resolve-Path $RunDir5).Path
}
elseif (-not $ForceRetrain5) {
    $RunDir5 = Find-Fair5Run
}

if ([string]::IsNullOrWhiteSpace($RunDir5)) {
    if ($SkipTraining5) {
        throw "No fair v0.18 five-exit run exists and -SkipTraining5 was supplied."
    }

    $Loss3 = @($Config3.loss_weights | ForEach-Object { [double]$_ })
    if ($Loss3.Count -ne 3) {
        throw "Canonical run must contain exactly three loss weights; found $($Loss3.Count)."
    }
    $AuxiliaryBudget = [double]$Loss3[0] + [double]$Loss3[1]
    $AuxiliaryEach = $AuxiliaryBudget / 4.0
    $FinalWeight = [double]$Loss3[2]
    $Invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $Loss5 = (@($AuxiliaryEach, $AuxiliaryEach, $AuxiliaryEach, $AuxiliaryEach, $FinalWeight) |
        ForEach-Object { $_.ToString("0.############", $Invariant) }) -join ","

    Write-Host ""
    Write-Host "[1/7] Training fair five-exit checkpoint..." -ForegroundColor Yellow
    Write-Host "Canonical auxiliary-loss budget: $AuxiliaryBudget"
    Write-Host "Five-exit loss weights:          $Loss5"

    $TrainArgs = @(
        "-m", "training.train_multilabel",
        "--manifest", [string]$Config3.manifest,
        "--features_root", [string]$Config3.features_root,
        "--labels_json", [string]$Config3.labels_json,
        "--runs_root", $Runs5Root,
        "--variant", "main_v018_human_corrected_balanced_5exit_no_hint_auxmatched",
        "--tap_blocks", "1,2,3,4",
        "--n_mels", [string]$Config3.n_mels,
        "--epochs", [string]$Config3.epochs,
        "--batch_size", [string]$Config3.batch_size,
        "--num_workers", [string]$Config3.num_workers,
        "--log_every", [string]$Config3.log_every,
        "--lr", [string]$Config3.lr,
        "--weight_decay", [string]$Config3.weight_decay,
        "--seed", [string]$Config3.seed,
        "--threshold", [string]$Config3.threshold,
        "--loss_weights", $Loss5,
        "--label_balance_power", [string]$Config3.label_balance_power,
        "--synthetic_balance_power", [string]$Config3.synthetic_balance_power,
        "--device", $TrainingDevice
    )
    if ([bool]$Config3.use_pos_weight) {
        $TrainArgs += @("--use_pos_weight", "--pos_weight_max", [string]$Config3.pos_weight_max)
    }
    python @TrainArgs
    if ($LASTEXITCODE -ne 0) { throw "Fair five-exit training failed." }
    $RunDir5 = Find-Fair5Run
    if ([string]::IsNullOrWhiteSpace($RunDir5)) {
        throw "Training completed but the new five-exit run could not be resolved."
    }
}
else {
    Write-Host "[1/7] Reusing fair five-exit run: $RunDir5" -ForegroundColor DarkYellow
}

$RequiredPaths = @(
    $RunDir3, (Join-Path $RunDir3 "ckpt\best.pt"),
    $RunDir5, (Join-Path $RunDir5 "ckpt\best.pt"),
    $HoldoutManifest, $HoldoutFeatures, $LabelsJson, $LatsConfig,
    "models\anytime_exit_net.py",
    "policies\strict_sequential_anytime_exit_v018.py",
    "tests\test_strict_sequential_anytime_exit_v018.py",
    $VerifyScript, $AuditScript, $TuneScript, $EvaluateScript, $CompareScript
)
foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) { throw "Required path not found: $Path" }
}

Write-Host "[2/7] Auditing fair training configuration..." -ForegroundColor Yellow
python $AuditScript --run3 $RunDir3 --run5 $RunDir5 --out_json $TrainingAudit
if ($LASTEXITCODE -ne 0) { throw "Fair training audit failed." }

if (-not $SkipPrechecks) {
    Write-Host "[3/7] Running staged and strict-policy tests..." -ForegroundColor Yellow
    python -m unittest `
        tests.test_anytime_exit_net `
        tests.test_strict_sequential_anytime_exit_v018 `
        -v
    if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }

    Write-Host "[4/7] Verifying staged equivalence for both architectures..." -ForegroundColor Yellow
    foreach ($item in @(
        @{ Name = "3exit"; Run = $RunDir3 },
        @{ Name = "5exit"; Run = $RunDir5 }
    )) {
        python $VerifyScript `
            --run_dir $item.Run `
            --labels_json $LabelsJson `
            --holdout_manifest $HoldoutManifest `
            --features_root $HoldoutFeatures `
            --sample_count 8 `
            --device $PolicyDevice `
            --out_json "$OutputRoot\$($item.Name)_checkpoint_staged_equivalence.json"
        if ($LASTEXITCODE -ne 0) { throw "$($item.Name) staged equivalence failed." }
    }
}
else {
    Write-Host "[3/7] Unit tests skipped." -ForegroundColor DarkYellow
    Write-Host "[4/7] Staged equivalence skipped." -ForegroundColor DarkYellow
}

$ThresholdMode = "fixed_0p5"
$Tune3 = "$OutputRoot\3exit\validation_tuning"
$Tune5 = "$OutputRoot\5exit\validation_tuning"
$Holdout3 = "$OutputRoot\3exit\corrected_holdout_evaluation"
$Holdout5 = "$OutputRoot\5exit\corrected_holdout_evaluation"
$Policy3 = "$Tune3\frozen_strict_sequential_policy_3exit_v018.json"
$Policy5 = "$Tune5\frozen_strict_sequential_policy_5exit_v018.json"
$Combined = "$OutputRoot\fair_architecture_comparison"
New-Item -ItemType Directory -Force -Path $Tune3, $Tune5, $Holdout3, $Holdout5, $Combined | Out-Null

function Invoke-Tune([string]$RunDir, [string]$Output, [string]$PolicyPath) {
    if ($SkipPolicyTuning) {
        if (-not (Test-Path $PolicyPath)) { throw "Frozen policy not found: $PolicyPath" }
        return
    }
    $Args = @(
        $TuneScript,
        "--run_dir", $RunDir,
        "--labels_json", $LabelsJson,
        "--lats_config_json", $LatsConfig,
        "--threshold_mode", $ThresholdMode,
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
        "--device", $PolicyDevice,
        "--out_dir", $Output
    )
    python @Args
    if ($LASTEXITCODE -ne 0) { throw "Strict sequential tuning failed for $RunDir" }
}

Write-Host "[5/7] Tuning safety-buffered strict policies..." -ForegroundColor Yellow
Invoke-Tune $RunDir3 $Tune3 $Policy3
Invoke-Tune $RunDir5 $Tune5 $Policy5

function Invoke-Evaluate([string]$RunDir, [string]$Policy, [string]$Output) {
    python $EvaluateScript `
        --run_dir $RunDir `
        --policy_json $Policy `
        --holdout_manifest $HoldoutManifest `
        --features_root $HoldoutFeatures `
        --labels_json $LabelsJson `
        --lats_config_json $LatsConfig `
        --batch_size 128 `
        --timing_repeats $TimingRepeats `
        --torch_threads $TorchThreads `
        --device $PolicyDevice `
        --out_dir $Output
    if ($LASTEXITCODE -ne 0) { throw "Strict sequential holdout evaluation failed for $RunDir" }
}

Write-Host "[6/7] Running genuine staged holdout evaluation and ablations..." -ForegroundColor Yellow
Invoke-Evaluate $RunDir3 $Policy3 $Holdout3
Invoke-Evaluate $RunDir5 $Policy5 $Holdout5

Write-Host "[7/7] Creating fair architecture and policy tables..." -ForegroundColor Yellow
python $CompareScript `
    --training_audit $TrainingAudit `
    --policy3 $Policy3 `
    --comparison3 "$Holdout3\v018_3exit_holdout_comparison.csv" `
    --policy5 $Policy5 `
    --comparison5 "$Holdout5\v018_5exit_holdout_comparison.csv" `
    --out_dir $Combined
if ($LASTEXITCODE -ne 0) { throw "V0.18 comparison generation failed." }

$RunDir5 | Set-Content "$TrainingRoot\latest_fair_5exit_run.txt" -Encoding UTF8
Write-Host ""
Write-Host "V0.18 completed." -ForegroundColor Green
Write-Host "Fair 5-exit run: $RunDir5"
Write-Host "Training audit:  $TrainingAudit"
Write-Host "3-exit policy:   $Policy3"
Write-Host "5-exit policy:   $Policy5"
Write-Host "Combined tables: $Combined"
