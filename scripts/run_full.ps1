param(
  [string]$DataRoot   = "data2",
  [string]$CacheRoot  = "data_caches",
  [string]$CacheId    = "",
  [string]$Config     = "configs\audio_moth.yaml",
  [string]$RunsRoot   = "runs",
  [string]$Variant    = "gunshot_greedy_hint",
  [string]$Policy     = "greedy",
  [string]$Device     = "cpu",
  [double]$SegmentSec = 1.0,
  [double]$HopSec     = 0.5,
  [int]$NMels         = 64,
  [string]$TapBlocks  = "1,2,3,4",

  [switch]$RunClipPolicy,
  [double]$TimeConf   = 0.95,
  [int]$TimeStableK   = 2,
  [int]$TimeMinWindows = 2,
  [int]$EvalFixedKWindows = 3,
  [double]$TimeMargin = 0.0,

  [string]$InputMode = "segment",
  [double]$MinKeepSec = 0.25,
  [int]$MaxSegmentsPerFileDefault = 0,
  [string]$MaxSegmentsPerLabelJson = "",
  [string]$SplitUnit = "file",
  [string]$GroupMode = "none",
  [string]$GroupRegex = "",
  [switch]$StrictReadyLength,
  [double]$ReadyLengthToleranceSec = 0.05,
  [switch]$ExportSegmentWavs,
  [string]$ExportRoot = "",
  [switch]$SkipIfSegmentsExist,
  [switch]$ForceRebuild,
  [string[]]$Labels = @(),

  [Nullable[int]]$MaxSegmentsPerFileGunshot = $null,
  [Nullable[int]]$MaxSegmentsPerFileNonGunshot = $null,

  [string]$ExitHint = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function ConvertTo-Id([double]$x) {
  return ($x.ToString("0.###") -replace '\.', 'p')
}

function Quote-ForDisplay([string]$s) {
  if ($null -eq $s) { return "''" }
  if ($s -notmatch '[\s\"]' -and $s -notmatch "'") { return $s }
  $escaped = $s -replace "'", "''"
  return "'$escaped'"
}

function Invoke-PythonArgs {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Args,
    [string]$Label = ""
  )

  $shown = ($Args | ForEach-Object { Quote-ForDisplay $_ }) -join ' '
  if ($Label -ne "") {
    Write-Host "  python $shown" -ForegroundColor DarkGray
  } else {
    Write-Host "  python $shown" -ForegroundColor DarkGray
  }

  & python @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed ($LASTEXITCODE): python $shown"
  }
}

function Normalize-JsonString([string]$Text) {
  if ([string]::IsNullOrWhiteSpace($Text)) {
    return ""
  }
  try {
    $obj = $Text | ConvertFrom-Json
    return ($obj | ConvertTo-Json -Compress)
  }
  catch {
    throw 'MaxSegmentsPerLabelJson is not valid JSON. Example: {"non_gunshot":5,"fireworks":8}'
  }
}

if ($InputMode -notin @("segment", "ready")) {
  throw "InputMode must be either 'segment' or 'ready'."
}
if ($SplitUnit -notin @("file", "group")) {
  throw "SplitUnit must be either 'file' or 'group'."
}
if ($GroupMode -notin @("none", "parent", "stem", "regex")) {
  throw "GroupMode must be one of: none, parent, stem, regex."
}
if ($GroupMode -eq "regex" -and [string]::IsNullOrWhiteSpace($GroupRegex)) {
  throw "GroupRegex must be provided when GroupMode=regex."
}
if ($ExitHint -ne "") {
  $eh = $ExitHint.Trim().ToLower()
  if ($eh -notin @("true", "false")) {
    throw "ExitHint must be either 'true' or 'false'."
  }
  $ExitHint = $eh
}

$MaxSegmentsPerLabelJsonNormalized = Normalize-JsonString $MaxSegmentsPerLabelJson

$VariantSafe = ($Variant -replace '[^A-Za-z0-9_-]', '_')
$variantRunDir = Join-Path $RunsRoot $VariantSafe

New-Item -ItemType Directory -Path $RunsRoot -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantRunDir -ErrorAction SilentlyContinue | Out-Null

$variantEsc = [regex]::Escape($VariantSafe)
$maxN = 0
Get-ChildItem -Path $variantRunDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  if ($_.Name -match "^$variantEsc`_(\d+)$") {
    $n = [int]$Matches[1]
    if ($n -gt $maxN) { $maxN = $n }
  }
}
$nextN = $maxN + 1
$runId = "{0}_{1:000}" -f $VariantSafe, $nextN
$runPath = Join-Path $variantRunDir $runId
if (Test-Path $runPath) { throw "Target run folder already exists: $runPath" }

if ([string]::IsNullOrWhiteSpace($CacheId)) {
  $segId = ConvertTo-Id $SegmentSec
  $hopId = ConvertTo-Id $HopSec
  $tapId = (($TapBlocks -replace '\s+', '') -replace ',', '-')
  $modeId = $InputMode
  $splitId = if ($SplitUnit -eq "group") { "splitgrp-$GroupMode" } else { "splitfile" }
  $capId = if ($MaxSegmentsPerLabelJsonNormalized -ne "") { "capsCustom" } elseif ($MaxSegmentsPerFileDefault -gt 0) { "cap$MaxSegmentsPerFileDefault" } else { "capAll" }

  $legacyCapId = ""
  if ($null -ne $MaxSegmentsPerFileGunshot -or $null -ne $MaxSegmentsPerFileNonGunshot) {
    $gCapId  = if ($null -ne $MaxSegmentsPerFileGunshot -and $MaxSegmentsPerFileGunshot -gt 0) { "gcap$MaxSegmentsPerFileGunshot" } elseif ($null -ne $MaxSegmentsPerFileGunshot) { "gcapAll" } else { "" }
    $ngCapId = if ($null -ne $MaxSegmentsPerFileNonGunshot -and $MaxSegmentsPerFileNonGunshot -gt 0) { "ngcap$MaxSegmentsPerFileNonGunshot" } elseif ($null -ne $MaxSegmentsPerFileNonGunshot) { "ngcapAll" } else { "" }
    $legacyCapId = ((@($gCapId, $ngCapId) | Where-Object { $_ -ne "" }) -join "_")
  }

  $parts = @($modeId, "seg$segId", "hop$hopId", "bp100-3000", "mels$NMels", "tap$tapId", $splitId, $capId)
  if ($legacyCapId -ne "") { $parts += $legacyCapId }
  if ($StrictReadyLength) { $parts += "strictReady" }
  if ($ExportSegmentWavs) { $parts += "exportWavs" }
  $CacheId = (($parts | Where-Object { $_ -and $_.Trim() -ne "" }) -join "_")
}

$CacheIdSafe = ($CacheId -replace '[^A-Za-z0-9_-]', '_')
$variantCacheDir = Join-Path (Join-Path $CacheRoot $VariantSafe) $CacheIdSafe

New-Item -ItemType Directory -Path $CacheRoot -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path (Join-Path $CacheRoot $VariantSafe) -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantCacheDir -ErrorAction SilentlyContinue | Out-Null

if ([string]::IsNullOrWhiteSpace($ExportRoot)) {
  $ExportRootResolved = Join-Path $variantCacheDir "exported_segments"
} else {
  $ExportRootResolved = $ExportRoot
}

$SegCsv = Join-Path $variantCacheDir "segments.csv"
$FeatRoot = Join-Path $variantCacheDir "features"
$pipelineStart = Get-Date

$MaxSegLabelJsonDisplay = if ([string]::IsNullOrWhiteSpace($MaxSegmentsPerLabelJsonNormalized)) { '<none>' } else { $MaxSegmentsPerLabelJsonNormalized }
$GroupRegexDisplay = if ([string]::IsNullOrWhiteSpace($GroupRegex)) { '<none>' } else { $GroupRegex }

Write-Host "== ASHADIP: full pipeline run ==" -ForegroundColor Cyan
Write-Host "  DataRoot              = $DataRoot" -ForegroundColor DarkGray
Write-Host "  CacheRoot             = $CacheRoot" -ForegroundColor DarkGray
Write-Host "  CacheId               = $CacheIdSafe" -ForegroundColor DarkGray
Write-Host "  CacheDir              = $variantCacheDir" -ForegroundColor DarkGray
Write-Host "  Config                = $Config" -ForegroundColor DarkGray
Write-Host "  RunsRoot              = $RunsRoot" -ForegroundColor DarkGray
Write-Host "  Variant               = $Variant" -ForegroundColor DarkGray
Write-Host "  Policy                = $Policy" -ForegroundColor DarkGray
Write-Host "  Device                = $Device" -ForegroundColor DarkGray
Write-Host "  SegmentSec            = $SegmentSec" -ForegroundColor DarkGray
Write-Host "  HopSec                = $HopSec" -ForegroundColor DarkGray
Write-Host "  NMels                 = $NMels" -ForegroundColor DarkGray
Write-Host "  TapBlocks             = $TapBlocks" -ForegroundColor DarkGray
Write-Host "  InputMode             = $InputMode" -ForegroundColor DarkGray
Write-Host "  MinKeepSec            = $MinKeepSec" -ForegroundColor DarkGray
Write-Host "  MaxSeg/File Default   = $MaxSegmentsPerFileDefault" -ForegroundColor DarkGray
Write-Host "  MaxSeg/Label JSON     = $MaxSegLabelJsonDisplay" -ForegroundColor DarkGray
Write-Host "  SplitUnit             = $SplitUnit" -ForegroundColor DarkGray
Write-Host "  GroupMode             = $GroupMode" -ForegroundColor DarkGray
Write-Host "  GroupRegex            = $GroupRegexDisplay" -ForegroundColor DarkGray
Write-Host "  StrictReadyLength     = $StrictReadyLength" -ForegroundColor DarkGray
Write-Host "  ReadyLenToleranceSec  = $ReadyLengthToleranceSec" -ForegroundColor DarkGray
Write-Host "  ExportSegmentWavs     = $ExportSegmentWavs" -ForegroundColor DarkGray
Write-Host "  ExportRoot            = $ExportRootResolved" -ForegroundColor DarkGray
Write-Host "  SkipIfSegmentsExist   = $SkipIfSegmentsExist" -ForegroundColor DarkGray
Write-Host "  ForceRebuild          = $ForceRebuild" -ForegroundColor DarkGray
Write-Host "  Legacy GCap/File      = $(if ($null -ne $MaxSegmentsPerFileGunshot) { $MaxSegmentsPerFileGunshot } else { '<unset>' })" -ForegroundColor DarkGray
Write-Host "  Legacy NGCap/File     = $(if ($null -ne $MaxSegmentsPerFileNonGunshot) { $MaxSegmentsPerFileNonGunshot } else { '<unset>' })" -ForegroundColor DarkGray
if ($Labels.Count -gt 0) {
  Write-Host "  Labels                = $($Labels -join ', ')" -ForegroundColor DarkGray
} else {
  Write-Host "  Labels                = auto-discover" -ForegroundColor DarkGray
}
if ($ExitHint -ne "") {
  Write-Host "  ExitHint              = $ExitHint (CLI override)" -ForegroundColor DarkGray
} else {
  Write-Host "  ExitHint              = YAML default" -ForegroundColor DarkGray
}
Write-Host "  RunDir                = $runPath" -ForegroundColor DarkGray
Write-Host "  RunClipPolicy         = $RunClipPolicy" -ForegroundColor DarkGray

$cacheReady = (Test-Path $SegCsv) -and (Test-Path $FeatRoot) -and (-not $ForceRebuild)

if ($cacheReady) {
  Write-Host "`n[cache] Reusing existing cache: $variantCacheDir" -ForegroundColor Green
}
else {
  Write-Host "`n[1/10] Prep segments..." -ForegroundColor Yellow
  $prepArgs = [System.Collections.Generic.List[string]]::new()
  $prepArgs.Add('-m')
  $prepArgs.Add('scripts.prep_segments')
  $prepArgs.Add('--root')
  $prepArgs.Add($DataRoot)
  $prepArgs.Add('--cache')
  $prepArgs.Add($variantCacheDir)
  $prepArgs.Add('--sr')
  $prepArgs.Add('16000')
  $prepArgs.Add('--segment_sec')
  $prepArgs.Add($SegmentSec.ToString())
  $prepArgs.Add('--hop')
  $prepArgs.Add($HopSec.ToString())
  $prepArgs.Add('--silence_dbfs')
  $prepArgs.Add('-40')
  $prepArgs.Add('--bandpass')
  $prepArgs.Add('100')
  $prepArgs.Add('3000')
  $prepArgs.Add('--min_keep_sec')
  $prepArgs.Add($MinKeepSec.ToString())
  $prepArgs.Add('--config')
  $prepArgs.Add($Config)
  $prepArgs.Add('--input_mode')
  $prepArgs.Add($InputMode)
  $prepArgs.Add('--split_unit')
  $prepArgs.Add($SplitUnit)
  $prepArgs.Add('--group_mode')
  $prepArgs.Add($GroupMode)
  if ($GroupRegex -ne '') {
    $prepArgs.Add('--group_regex')
    $prepArgs.Add($GroupRegex)
  }
  $prepArgs.Add('--ready_length_tolerance_sec')
  $prepArgs.Add($ReadyLengthToleranceSec.ToString())
  $prepArgs.Add('--max_segments_per_file_default')
  $prepArgs.Add($MaxSegmentsPerFileDefault.ToString())
  if ($MaxSegmentsPerLabelJsonNormalized -ne '') {
    $prepArgs.Add('--max_segments_per_label_json')
    $prepArgs.Add($MaxSegmentsPerLabelJsonNormalized)
  }
  if ($null -ne $MaxSegmentsPerFileGunshot) {
    $prepArgs.Add('--max_segments_per_file_gunshot')
    $prepArgs.Add($MaxSegmentsPerFileGunshot.ToString())
  }
  if ($null -ne $MaxSegmentsPerFileNonGunshot) {
    $prepArgs.Add('--max_segments_per_file_non_gunshot')
    $prepArgs.Add($MaxSegmentsPerFileNonGunshot.ToString())
  }
  if ($StrictReadyLength)    { $prepArgs.Add('--strict_ready_length') }
  if ($ExportSegmentWavs)    { $prepArgs.Add('--export_segment_wavs') }
  if ($SkipIfSegmentsExist)  { $prepArgs.Add('--skip_if_segments_exist') }
  if ($ForceRebuild)         { $prepArgs.Add('--force_rebuild') }
  if ($ExportRootResolved -ne '') {
    $prepArgs.Add('--export_root')
    $prepArgs.Add($ExportRootResolved)
  }
  if ($Labels.Count -gt 0) {
    $prepArgs.Add('--labels')
    foreach ($lab in $Labels) { $prepArgs.Add($lab) }
  }
  Invoke-PythonArgs -Args $prepArgs.ToArray() -Label 'prep_segments'

  Write-Host "`n[2/10] Extract features..." -ForegroundColor Yellow
  Invoke-PythonArgs -Args @(
    '-m', 'scripts.extract_features',
    '--cache', $variantCacheDir,
    '--n_mels', $NMels.ToString(),
    '--n_fft', '1024',
    '--win_ms', '25',
    '--hop_ms', '10',
    '--cmvn',
    '--pad_short'
  ) -Label 'extract_features'
}

$trainArgs = [System.Collections.Generic.List[string]]::new()
@(
  '-m', 'training.train',
  '--config', $Config,
  '--run_dir', $runPath,
  '--cache_dir', $variantCacheDir,
  '--device', $Device,
  '--segment_sec', $SegmentSec.ToString(),
  '--hop_sec', $HopSec.ToString(),
  '--variant', $Variant,
  '--tap_blocks', $TapBlocks
) | ForEach-Object { $trainArgs.Add($_) }
if ($ExitHint -ne '') {
  $trainArgs.Add('--exit_hint_enable')
  $trainArgs.Add($ExitHint)
}

Write-Host "`n[3/10] Train ExitNet..." -ForegroundColor Yellow
Invoke-PythonArgs -Args $trainArgs.ToArray() -Label 'training.train'
Write-Host "Using run: $runPath" -ForegroundColor Green

$createdAtIso = Get-Date -Format o
$meta = @{
  run_id                           = $runId
  variant                          = $Variant
  variant_safe                     = $VariantSafe
  created_at                       = $createdAtIso
  runs_root                        = $RunsRoot
  variant_dir                      = $variantRunDir
  cache_root                       = $CacheRoot
  cache_id                         = $CacheIdSafe
  cache_dir                        = $variantCacheDir
  data_root                        = $DataRoot
  device                           = $Device
  policy                           = $Policy
  segment_sec                      = $SegmentSec
  hop_sec                          = $HopSec
  n_mels                           = $NMels
  tap_blocks                       = $TapBlocks
  input_mode                       = $InputMode
  min_keep_sec                     = $MinKeepSec
  max_segments_per_file_default    = $MaxSegmentsPerFileDefault
  max_segments_per_label_json      = $MaxSegmentsPerLabelJsonNormalized
  split_unit                       = $SplitUnit
  group_mode                       = $GroupMode
  group_regex                      = $GroupRegex
  strict_ready_length              = [bool]$StrictReadyLength
  ready_length_tolerance_sec       = $ReadyLengthToleranceSec
  export_segment_wavs              = [bool]$ExportSegmentWavs
  export_root                      = $ExportRootResolved
  skip_if_segments_exist           = [bool]$SkipIfSegmentsExist
  force_rebuild                    = [bool]$ForceRebuild
  labels                           = @($Labels)
  max_segments_gunshot             = $(if ($null -ne $MaxSegmentsPerFileGunshot) { $MaxSegmentsPerFileGunshot } else { $null })
  max_segments_non_gunshot         = $(if ($null -ne $MaxSegmentsPerFileNonGunshot) { $MaxSegmentsPerFileNonGunshot } else { $null })
  exit_hint_override               = $(if ($ExitHint -ne '') { $ExitHint } else { 'yaml_default' })
  run_clip_policy                  = [bool]$RunClipPolicy
  time_conf                        = $TimeConf
  time_stable_k                    = $TimeStableK
  time_min_windows                 = $TimeMinWindows
  eval_fixed_k_windows             = $EvalFixedKWindows
  time_margin                      = $TimeMargin
}
New-Item -ItemType Directory -Path $runPath -ErrorAction SilentlyContinue | Out-Null
$meta | ConvertTo-Json -Depth 8 | Out-File -FilePath (Join-Path $runPath 'meta.json') -Encoding UTF8

Write-Host "`n[4/10] Calibrate temperatures..." -ForegroundColor Yellow
Invoke-PythonArgs -Args @(
  '-m', 'training.calibrate',
  '--run_dir', $runPath,
  '--segments_csv', $SegCsv,
  '--features_root', $FeatRoot,
  '--tap_blocks', $TapBlocks,
  '--n_mels', $NMels.ToString()
) -Label 'training.calibrate'

Write-Host "`n[5/10] Select threshold (tau)..." -ForegroundColor Yellow
Invoke-PythonArgs -Args @(
  '-m', 'training.thresholds_offline',
  '--run_dir', $runPath,
  '--segments_csv', $SegCsv,
  '--features_root', $FeatRoot,
  '--tap_blocks', $TapBlocks,
  '--n_mels', $NMels.ToString()
) -Label 'training.thresholds_offline'

Write-Host "`n[6/10] Segment policy test..." -ForegroundColor Yellow
Invoke-PythonArgs -Args @(
  '-m', 'scripts.policy_test',
  '--run_dir', $runPath,
  '--segments_csv', $SegCsv,
  '--features_root', $FeatRoot,
  '--tap_blocks', $TapBlocks,
  '--n_mels', $NMels.ToString()
) -Label 'scripts.policy_test'

if ($RunClipPolicy -and $Policy -ne 'greedy') {
  throw 'Current scripts.clip_policy_test.py is greedy-only. Use -Policy greedy, or create an EA-compatible clip_policy_test.py first.'
}

if ($RunClipPolicy) {
  Write-Host "`n[6b/10] Clip policy test..." -ForegroundColor Yellow
  Invoke-PythonArgs -Args @(
    '-m', 'scripts.clip_policy_test',
    '--run_dir', $runPath,
    '--segments_csv', $SegCsv,
    '--features_root', $FeatRoot,
    '--device', $Device,
    '--tap_blocks', $TapBlocks,
    '--n_mels', $NMels.ToString(),
    '--time_conf', $TimeConf.ToString(),
    '--time_stable_k', $TimeStableK.ToString(),
    '--time_min_windows', $TimeMinWindows.ToString(),
    '--fixed_k_windows', $EvalFixedKWindows.ToString(),
    '--time_margin', $TimeMargin.ToString()
  ) -Label 'scripts.clip_policy_test'
}

Write-Host "`n[7/10] Summarise run..." -ForegroundColor Yellow
Invoke-PythonArgs -Args @(
  '-m', 'scripts.summarize_run',
  '--run_dir', $runPath,
  '--segments_csv', $SegCsv,
  '--features_root', $FeatRoot,
  '--tap_blocks', $TapBlocks,
  '--n_mels', $NMels.ToString()
) -Label 'scripts.summarize_run'

Write-Host "`n[8/10] Analyse run..." -ForegroundColor Yellow
Invoke-PythonArgs -Args @(
  '-m', 'scripts.analyse_run',
  '--run_dir', $runPath,
  '--segments_csv', $SegCsv,
  '--features_root', $FeatRoot,
  '--tap_blocks', $TapBlocks,
  '--n_mels', $NMels.ToString()
) -Label 'scripts.analyse_run'

Write-Host "`n[9/10] Profile latency..." -ForegroundColor Yellow
Invoke-PythonArgs -Args @(
  '-m', 'scripts.profile_latency',
  '--run_dir', $runPath,
  '--segments_csv', $SegCsv,
  '--features_root', $FeatRoot,
  '--variant', $Variant,
  '--device', $Device,
  '--tap_blocks', $TapBlocks,
  '--n_mels', $NMels.ToString()
) -Label 'scripts.profile_latency'

$pipelineEnd = Get-Date
$elapsed = $pipelineEnd - $pipelineStart
$totalSeconds = [Math]::Round($elapsed.TotalSeconds, 2)
$totalMinutes = [Math]::Round($elapsed.TotalMinutes, 2)
$timestampIso = Get-Date -Format o

Write-Host ""
Write-Host (("Total wall-clock time: {0} seconds (~{1} minutes)" -f $totalSeconds, $totalMinutes)) -ForegroundColor Cyan

$analysisDir = 'analysis'
New-Item -ItemType Directory -Path $analysisDir -ErrorAction SilentlyContinue | Out-Null
$runtimeCsv = Join-Path $analysisDir 'pipeline_runtime.csv'

if (-not (Test-Path $runtimeCsv)) {
  'timestamp,variant,policy,segment_sec,hop_sec,device,cache_dir,runs_root,run_id,total_seconds,total_minutes,run_clip_policy,tap_blocks,exit_hint_override,input_mode,split_unit,group_mode,export_segment_wavs' | Out-File $runtimeCsv -Encoding UTF8
}

$csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}" -f `
  $timestampIso, `
  $Variant, `
  $Policy, `
  $SegmentSec, `
  $HopSec, `
  $Device, `
  $variantCacheDir, `
  $RunsRoot, `
  $runId, `
  $totalSeconds, `
  $totalMinutes, `
  [bool]$RunClipPolicy, `
  $TapBlocks, `
  $(if ($ExitHint -ne '') { $ExitHint } else { 'yaml_default' }), `
  $InputMode, `
  $SplitUnit, `
  $GroupMode, `
  [bool]$ExportSegmentWavs

Add-Content -Path $runtimeCsv -Value $csvLine
Write-Host "Pipeline runtime logged to: $runtimeCsv" -ForegroundColor DarkGray

Write-Host "`n[10/10] Generate reports & LaTeX tables..." -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File scripts\run_reports.ps1 `
  -RunDir $runPath `
  -Variant $Variant `
  -DeviceFilter $Device `
  -SegmentsCsv $SegCsv `
  -FeaturesRoot $FeatRoot `
  -RunsRoot $RunsRoot

Write-Host "`n== Done. Artifacts at: $runPath ==" -ForegroundColor Cyan
Write-Host "Cache used: $variantCacheDir" -ForegroundColor DarkGray
