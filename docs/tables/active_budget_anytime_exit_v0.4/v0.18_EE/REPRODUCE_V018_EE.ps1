param(
    [string]$RunDir5 = "",
    [int]$TimingRepeats = 30,
    [switch]$ForceRetrain5,
    [switch]$SkipPrechecks,
    [switch]$SkipPolicyTuning
)

$Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1",
    "-TimingRepeats", "$TimingRepeats"
)
if (-not [string]::IsNullOrWhiteSpace($RunDir5)) { $Args += @("-RunDir5", $RunDir5) }
if ($ForceRetrain5) { $Args += "-ForceRetrain5" }
if ($SkipPrechecks) { $Args += "-SkipPrechecks" }
if ($SkipPolicyTuning) { $Args += "-SkipPolicyTuning" }
powershell @Args
exit $LASTEXITCODE
