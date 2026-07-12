param(
    [string]$Device = "cpu",
    [int]$BatchSize = 128,
    [switch]$SkipPrechecks
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

Write-Host ""
Write-Host "=== Reproduce NeuroAccuExit v0.11_EE ===" -ForegroundColor Cyan

$FixedArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", ".\scripts\v0.11_EE\fixed_policy\run_v011_EE.ps1",
    "-Device", $Device,
    "-BatchSize", "$BatchSize"
)
if ($SkipPrechecks) {
    $FixedArgs += "-SkipUnitTests"
}

powershell @FixedArgs
if ($LASTEXITCODE -ne 0) {
    throw "Fixed-exit reproduction failed."
}

$DynamicArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1",
    "-Device", $Device,
    "-BatchSize", "$BatchSize"
)
if ($SkipPrechecks) {
    $DynamicArgs += "-SkipPrechecks"
}

powershell @DynamicArgs
if ($LASTEXITCODE -ne 0) {
    throw "Dynamic Early-Exit reproduction failed."
}

Write-Host ""
Write-Host "v0.11_EE reproduction completed." -ForegroundColor Green
Write-Host "Review compact expected results under:"
Write-Host "docs\tables\active_budget_anytime_exit_v0.2\v0.11_EE"
Write-Host ""
Write-Host "Runtime outputs are under human_talk_workspace and may vary slightly in measured latency." -ForegroundColor DarkYellow
