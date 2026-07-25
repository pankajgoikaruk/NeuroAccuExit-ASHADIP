param(
    [int]$TimingRepeats = 30,
    [switch]$SkipTuning
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Runner = ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1"
if (-not (Test-Path $Runner)) {
    throw "Run from the NeuroAccuExit-ASHADIP repository root. Missing: $Runner"
}

$Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $Runner,
    "-TimingRepeats", "$TimingRepeats"
)
if ($SkipTuning) {
    $Args += @("-SkipPrechecks", "-SkipTuning")
}

powershell @Args
if ($LASTEXITCODE -ne 0) {
    throw "v0.16 reproduction failed."
}
