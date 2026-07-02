<#
Run v0.10 hint-pass ablation
============================

This wrapper runs a controlled no-hint vs hint-pass comparison using the existing
scripts/run_full.ps1 pipeline. It keeps the same cache/data settings and changes
only the ExitHint flag and Variant name.

Recommended first test:
  - 3 exits: TapBlocks "1,3"
  - no-hint vs hint enabled
  - RunClipPolicy enabled if your dataset/pipeline supports it

Run from repo root.
#>

param(
  [string]$DataRoot = "data\moth_sounds",
  [string]$CacheRoot = "data_caches",
  [string]$RunsRoot = "runs_v0.10_hint_pass",
  [string]$Config = "configs\audio_moth.yaml",
  [string]$Device = "cpu",
  [string]$TapBlocks = "1,3",
  [string]$InputMode = "segment",
  [string]$Labels = "",
  [double]$SegmentSec = 1.0,
  [double]$HopSec = 0.5,
  [string]$Bandpass = "100,3000",
  [int]$NMels = 64,
  [int]$SampleRate = 16000,
  [double]$SilenceDbfs = -40,
  [int]$MaxSegmentsPerFileDefault = 0,
  [string]$MaxSegmentsPerLabelJson = "",
  [string]$SplitUnit = "file",
  [string]$GroupRegex = "",
  [switch]$RunClipPolicy,
  [switch]$ForceRebuild
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Invoke-RunFullVariant {
  param(
    [string]$Variant,
    [string]$ExitHint
  )

  Write-Host "`n============================================================" -ForegroundColor Cyan
  Write-Host "Running v0.10 variant: $Variant | ExitHint=$ExitHint | TapBlocks=$TapBlocks" -ForegroundColor Cyan
  Write-Host "============================================================" -ForegroundColor Cyan

  $args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\run_full.ps1",
    "-DataRoot", $DataRoot,
    "-CacheRoot", $CacheRoot,
    "-Config", $Config,
    "-RunsRoot", $RunsRoot,
    "-Variant", $Variant,
    "-Policy", "greedy",
    "-Device", $Device,
    "-TapBlocks", $TapBlocks,
    "-ExitHint", $ExitHint,
    "-InputMode", $InputMode,
    "-SegmentSec", ([string]$SegmentSec),
    "-HopSec", ([string]$HopSec),
    "-Bandpass", $Bandpass,
    "-NMels", ([string]$NMels),
    "-SampleRate", ([string]$SampleRate),
    "-SilenceDbfs", ([string]$SilenceDbfs),
    "-MaxSegmentsPerFileDefault", ([string]$MaxSegmentsPerFileDefault),
    "-SplitUnit", $SplitUnit
  )

  if ($Labels.Trim() -ne "") {
    $args += @("-Labels", $Labels)
  }
  if ($MaxSegmentsPerLabelJson.Trim() -ne "") {
    $args += @("-MaxSegmentsPerLabelJson", $MaxSegmentsPerLabelJson)
  }
  if ($GroupRegex.Trim() -ne "") {
    $args += @("-GroupRegex", $GroupRegex)
  }
  if ($RunClipPolicy) {
    $args += "-RunClipPolicy"
  }
  if ($ForceRebuild) {
    $args += "-ForceRebuild"
  }

  & powershell @args
  if ($LASTEXITCODE -ne 0) {
    throw "Variant failed: $Variant"
  }
}

$tapId = (($TapBlocks -replace '\s+', '') -replace ',', '-')
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$variantNoHint = "v010_${tapId}_no_hint_$stamp"
$variantHint   = "v010_${tapId}_hint_pass_$stamp"

Invoke-RunFullVariant -Variant $variantNoHint -ExitHint "false"
Invoke-RunFullVariant -Variant $variantHint   -ExitHint "true"

Write-Host "`nV0.10 hint-pass ablation complete." -ForegroundColor Green
Write-Host "RunsRoot: $RunsRoot" -ForegroundColor Green
Write-Host "No-hint variant: $variantNoHint" -ForegroundColor Green
Write-Host "Hint-pass variant: $variantHint" -ForegroundColor Green
Write-Host "Next: compare summaries/reports, then apply LATS-v2 to parent-level segment-probability CSVs if available." -ForegroundColor Yellow
