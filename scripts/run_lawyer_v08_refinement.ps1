param(
  [string]$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline",
  [string]$V07Root = "human_talk_workspace\tata_v0.7_raw_pipeline",
  [string]$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline",

  [string]$SegmentPredictionsCsv = "",
  [string]$ParentCsv = "",

  [string]$ModeName = "lawyer_v08",

  [double]$SpeakerAlpha = 0.70,
  [double]$TargetThreshold = 0.50,
  [double]$TargetMarginThreshold = 0.10,

  [double]$OtherDirectThreshold = 0.55,
  [double]$OtherSpeechThreshold = 0.55,
  [double]$OtherKnownMaxThreshold = 0.35,

  [double]$MusicThreshold = 0.50,

  [int]$AudienceTopK = 2,
  [double]$AudienceThreshold = 0.50,
  [double]$AudienceLow = 0.35,
  [double]$AudienceHigh = 0.65,

  [double]$SilenceThreshold = 0.50,
  [double]$SilenceLow = 0.35,
  [double]$SilenceHigh = 0.65,

  [string]$SilenceEnergyCol = "",
  [double]$SilenceEnergyThreshold = -45.0,
  [string]$SilenceVadCol = "",
  [double]$SilenceVadThreshold = 0.15
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
Write-Host "== LAWYER v0.8 refinement ==" -ForegroundColor Cyan
Write-Host "V08Root               = $V08Root"
Write-Host "SegmentPredictionsCsv = $SegmentPredictionsCsv"
Write-Host "ParentCsv             = $ParentCsv"
Write-Host "OutDir                = $OutDir"
Write-Host ""

if (-not (Test-Path $SegmentPredictionsCsv)) {
  throw "SegmentPredictionsCsv not found: $SegmentPredictionsCsv"
}

$ArgsList = @(
  "scripts\lawyer_refine_weak_labels_v08.py",
  "--segment_predictions_csv", $SegmentPredictionsCsv,
  "--out_dir", $OutDir,
  "--mode_name", $ModeName,
  "--speaker_alpha", "$SpeakerAlpha",
  "--target_threshold", "$TargetThreshold",
  "--target_margin_threshold", "$TargetMarginThreshold",
  "--other_direct_threshold", "$OtherDirectThreshold",
  "--other_speech_threshold", "$OtherSpeechThreshold",
  "--other_known_max_threshold", "$OtherKnownMaxThreshold",
  "--music_threshold", "$MusicThreshold",
  "--audience_top_k", "$AudienceTopK",
  "--audience_threshold", "$AudienceThreshold",
  "--audience_low", "$AudienceLow",
  "--audience_high", "$AudienceHigh",
  "--silence_threshold", "$SilenceThreshold",
  "--silence_low", "$SilenceLow",
  "--silence_high", "$SilenceHigh",
  "--silence_energy_threshold", "$SilenceEnergyThreshold",
  "--silence_vad_threshold", "$SilenceVadThreshold"
)

if ((Test-Path $ParentCsv)) {
  $ArgsList += "--parent_csv"
  $ArgsList += $ParentCsv
} else {
  Write-Host "ParentCsv not found, continuing without parent context: $ParentCsv" -ForegroundColor Yellow
}

if (-not [string]::IsNullOrWhiteSpace($SilenceEnergyCol)) {
  $ArgsList += "--silence_energy_col"
  $ArgsList += $SilenceEnergyCol
}

if (-not [string]::IsNullOrWhiteSpace($SilenceVadCol)) {
  $ArgsList += "--silence_vad_col"
  $ArgsList += $SilenceVadCol
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
