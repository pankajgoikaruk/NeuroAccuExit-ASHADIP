param(
  [string]$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline",
  [string]$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline",

  [string]$Config = "configs\lawyer_v08_human_talk.json",
  [string]$SegmentPredictionsCsv = "",
  [string]$ParentCsv = "",

  [string]$ModeName = "lawyer_v08"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

if ([string]::IsNullOrWhiteSpace($SegmentPredictionsCsv)) {
  $SegmentPredictionsCsv = Join-Path $V06Root "raw_tata_pseudo_routing\raw_segment_predictions.csv"
}

if ([string]::IsNullOrWhiteSpace($ParentCsv)) {
  $ParentCsv = Join-Path $V06Root "raw_tata_pseudo_routing\hybrid\hybrid_parent_predictions_all.csv"
}

$OutDir = Join-Path $V08Root "raw_tata_pseudo_routing\$ModeName"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ""
Write-Host "== LAWYER v0.8 config-driven refinement ==" -ForegroundColor Cyan
Write-Host "V08Root               = $V08Root"
Write-Host "Config                = $Config"
Write-Host "SegmentPredictionsCsv = $SegmentPredictionsCsv"
Write-Host "ParentCsv             = $ParentCsv"
Write-Host "OutDir                = $OutDir"
Write-Host "ModeName              = $ModeName"
Write-Host ""

if (-not (Test-Path $Config)) {
  throw "Config not found: $Config"
}

if (-not (Test-Path $SegmentPredictionsCsv)) {
  throw "SegmentPredictionsCsv not found: $SegmentPredictionsCsv"
}

$ArgsList = @(
  "scripts\lawyer_refine_weak_labels_v08.py",
  "--config", $Config,
  "--segment_predictions_csv", $SegmentPredictionsCsv,
  "--out_dir", $OutDir,
  "--mode_name", $ModeName
)

if ((Test-Path $ParentCsv)) {
  $ArgsList += "--parent_csv"
  $ArgsList += $ParentCsv
} else {
  Write-Host "ParentCsv not found, continuing without parent context: $ParentCsv" -ForegroundColor Yellow
}

Write-Host "Command:" -ForegroundColor Yellow
Write-Host ("python " + ($ArgsList -join " "))

& python @ArgsList

if ($LASTEXITCODE -ne 0) {
  throw "LAWYER refinement failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "LAWYER v0.8 outputs:" -ForegroundColor Green
Write-Host $OutDir
