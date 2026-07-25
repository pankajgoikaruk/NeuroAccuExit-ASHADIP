param(
    [int]$TimingRepeats = 30,
    [string]$RunDir5 = ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324",
    [switch]$ReuseFrozenPolicies
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Test-Path ".git")) {
    throw "Run this script from the NeuroAccuExit-ASHADIP repository root."
}

$Runner = ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1"
if (-not (Test-Path $Runner)) {
    throw "Runner not found: $Runner"
}
if (-not (Test-Path $RunDir5)) {
    throw "5-exit run directory not found: $RunDir5"
}

$Arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $Runner,
    "-RunDir5", $RunDir5,
    "-TimingRepeats", "$TimingRepeats"
)
if ($ReuseFrozenPolicies) {
    $Arguments += @("-SkipPrechecks", "-SkipTuning")
}

powershell @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "v0.17 reproduction failed."
}
