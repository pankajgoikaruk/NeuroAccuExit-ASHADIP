# scripts/run_human_talk_clean_stage_experiment_logged.ps1
#
# Full-console logging wrapper for the human-talk clean-stage experiment runner.
#
# Why this wrapper exists:
#   The main runner uses Start-Transcript, which is useful but can miss or
#   compress some child-process output from Python/PowerShell commands. This
#   wrapper captures every PowerShell output stream with *>&1 and writes it to
#   a full console log through Tee-Object, while still showing progress in the
#   terminal.
#
# Use this wrapper for future experiments instead of calling
# run_human_talk_clean_stage_experiment.ps1 directly.
#
# Example:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment_logged.ps1 `
#     -Stage clean5_balanced `
#     -RawRoot human_talk_dataset `
#     -WorkspaceRoot human_talk_workspace `
#     -Device cpu `
#     -Clean `
#     -ZipResults
#
# Output:
#   human_talk_workspace\logs\<stage>_full_console_<timestamp>.txt
#
# Notes:
#   - All arguments are forwarded unchanged to run_human_talk_clean_stage_experiment.ps1.
#   - The existing runner transcript is still created as before.
#   - The full console log should be shared alongside the ZIP package.

$ErrorActionPreference = "Stop"

function Get-TimeStamp {
    return (Get-Date).ToString("yyyyMMdd_HHmmss")
}

function Get-ArgValue {
    param(
        [object[]]$ArgsList,
        [string]$Name,
        [string]$DefaultValue
    )

    $Flag = "-$Name"
    for ($i = 0; $i -lt $ArgsList.Count; $i++) {
        if ([string]$ArgsList[$i] -ieq $Flag) {
            if (($i + 1) -lt $ArgsList.Count) {
                $NextValue = [string]$ArgsList[$i + 1]
                if (-not $NextValue.StartsWith("-")) {
                    return $NextValue
                }
            }
        }
    }

    return $DefaultValue
}

function Get-VariantPrefix {
    param([string]$StageName)

    if ($StageName -eq "clean2_balanced") { return "human_talk_clean2" }
    if ($StageName -eq "clean3_balanced") { return "human_talk_clean3" }
    if ($StageName -eq "clean4_balanced") { return "human_talk_clean4" }
    if ($StageName -eq "clean5_balanced") { return "human_talk_clean5" }

    $Safe = $StageName -replace "[^A-Za-z0-9_]", "_"
    return "human_talk_$Safe"
}

$Stage = Get-ArgValue -ArgsList $args -Name "Stage" -DefaultValue "clean2_balanced"
$WorkspaceRoot = Get-ArgValue -ArgsList $args -Name "WorkspaceRoot" -DefaultValue "human_talk_workspace"
$VariantPrefix = Get-VariantPrefix -StageName $Stage
$Timestamp = Get-TimeStamp

$LogsRoot = Join-Path $WorkspaceRoot "logs"
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null

$FullConsoleLogPath = Join-Path $LogsRoot ("${VariantPrefix}_full_console_${Timestamp}.txt")
$RunnerPath = Join-Path $PSScriptRoot "run_human_talk_clean_stage_experiment.ps1"

if (!(Test-Path $RunnerPath)) {
    throw "Main runner not found: $RunnerPath"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Human-talk full-console logging wrapper"
Write-Host "============================================================"
Write-Host "Stage:            $Stage"
Write-Host "WorkspaceRoot:    $WorkspaceRoot"
Write-Host "Main runner:      $RunnerPath"
Write-Host "Full console log: $FullConsoleLogPath"
Write-Host "============================================================"
Write-Host ""

$ForwardArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $RunnerPath
) + $args

# Capture all PowerShell streams from the child runner and keep live terminal output.
& powershell @ForwardArgs *>&1 | Tee-Object -FilePath $FullConsoleLogPath
$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================================"
Write-Host " Full console log saved"
Write-Host "============================================================"
Write-Host "  $FullConsoleLogPath"
Write-Host "Exit code: $ExitCode"
Write-Host "============================================================"

exit $ExitCode
