param(
  [string]$DataRoot   = "data\moth_sounds",
  [string]$CacheRoot  = "data_caches",
  [string]$CacheId    = "",
  [string]$Config     = "configs\audio_moth.yaml",
  [string]$RunsRoot   = "runs",
  [string]$Variant    = "V0",
  [string]$Policy     = "greedy",
  [string]$Device     = "cpu",
  [double]$SegmentSec = 1.0,
  [double]$HopSec     = 0.5,
  [int]$NMels         = 64,

  [switch]$RunClipPolicy,
  [double]$TimeConf   = 0.95,
  [int]$TimeStableK   = 2,
  [int]$TimeMinWindows = 2,
  [int]$EvalFixedKWindows = 3,
  [double]$TimeMargin = 0.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Invoke-Python([string]$cmd) {
  Write-Host "  $cmd" -ForegroundColor DarkGray
  Invoke-Expression $cmd
  if ($LASTEXITCODE -ne 0) { throw "Python command failed ($LASTEXITCODE): $cmd" }
}

function ConvertTo-Id([double]$x) {
  return ($x.ToString("0.###") -replace '\.', 'p')
}

# --------------------- Run directory scheme: runs/<Variant>/<Variant>_### ---------------------
$VariantSafe = ($Variant -replace '[^A-Za-z0-9_-]', '_')
$variantRunDir = Join-Path $RunsRoot $VariantSafe

New-Item -ItemType Directory -Path $RunsRoot      -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantRunDir -ErrorAction SilentlyContinue | Out-Null

$variantEsc = [regex]::Escape($VariantSafe)
$maxN = 0
Get-ChildItem -Path $variantRunDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  if ($_.Name -match "^$variantEsc`_(\d+)$") {
    $n = [int]$Matches[1]
    if ($n -gt $maxN) { $maxN = $n }
  }
}
$nextN  = $maxN + 1
$runId  = "{0}_{1:000}" -f $VariantSafe, $nextN
$runPath = Join-Path $variantRunDir $runId
if (Test-Path $runPath) { throw "Target run folder already exists: $runPath" }

# --------------------- Cache directory scheme ---------------------
if ([string]::IsNullOrWhiteSpace($CacheId)) {
  $segId = ConvertTo-Id $SegmentSec
  $hopId = ConvertTo-Id $HopSec
  $CacheId = "seg$segId" + "_hop$hopId" + "_bp100-3000" + "_mels$NMels"
}

$CacheIdSafe     = ($CacheId -replace '[^A-Za-z0-9_-]', '_')
$variantCacheDir = Join-Path (Join-Path $CacheRoot $VariantSafe) $CacheIdSafe

New-Item -ItemType Directory -Path $CacheRoot -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path (Join-Path $CacheRoot $VariantSafe) -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantCacheDir -ErrorAction SilentlyContinue | Out-Null

$SegCsv   = Join-Path $variantCacheDir "segments.csv"
$FeatRoot = Join-Path $variantCacheDir "features"

$pipelineStart = Get-Date

Write-Host "== ASHADIP: full pipeline run ==" -ForegroundColor Cyan
Write-Host "  DataRoot        = $DataRoot"        -ForegroundColor DarkGray
Write-Host "  CacheRoot       = $CacheRoot"       -ForegroundColor DarkGray
Write-Host "  CacheId         = $CacheIdSafe"     -ForegroundColor DarkGray
Write-Host "  CacheDir        = $variantCacheDir" -ForegroundColor DarkGray
Write-Host "  Config          = $Config"          -ForegroundColor DarkGray
Write-Host "  RunsRoot        = $RunsRoot"        -ForegroundColor DarkGray
Write-Host "  Variant         = $Variant"         -ForegroundColor DarkGray
Write-Host "  Policy          = $Policy"          -ForegroundColor DarkGray
Write-Host "  Device          = $Device"          -ForegroundColor DarkGray
Write-Host "  SegmentSec      = $SegmentSec"      -ForegroundColor DarkGray
Write-Host "  HopSec          = $HopSec"          -ForegroundColor DarkGray
Write-Host "  NMels           = $NMels"           -ForegroundColor DarkGray
Write-Host "  RunDir          = $runPath"         -ForegroundColor DarkGray
Write-Host "  RunClipPolicy   = $RunClipPolicy"   -ForegroundColor DarkGray

$cacheReady = (Test-Path $SegCsv) -and (Test-Path $FeatRoot)

if ($cacheReady) {
  Write-Host "`n[cache] Reusing existing cache: $variantCacheDir" -ForegroundColor Green
}
else {
  # --------------------- 1/10) Prep segments ---------------------
  Write-Host "`n[1/10] Prep segments..." -ForegroundColor Yellow
  Invoke-Python ('python -m scripts.prep_segments --root "{0}" --cache "{1}" --sr 16000 --segment_sec {2} --hop {3} --silence_dbfs -40 --bandpass 100 3000 --config "{4}"' -f `
    $DataRoot, $variantCacheDir, $SegmentSec, $HopSec, $Config)

  # --------------------- 2/10) Extract features ---------------------
  Write-Host "`n[2/10] Extract features..." -ForegroundColor Yellow
  Invoke-Python ('python -m scripts.extract_features --cache "{0}" --n_mels {1} --n_fft 1024 --win_ms 25 --hop_ms 10 --cmvn' -f `
    $variantCacheDir, $NMels)
}

# --------------------- 3/10) Train ExitNet ---------------------
Write-Host "`n[3/10] Train ExitNet..." -ForegroundColor Yellow
Invoke-Python ('python -m training.train --config "{0}" --run_dir "{1}" --cache_dir "{2}" --device "{3}" --segment_sec {4} --hop_sec {5} --variant "{6}"' -f `
  $Config, $runPath, $variantCacheDir, $Device, $SegmentSec, $HopSec, $Variant)

Write-Host "Using run: $runPath" -ForegroundColor Green

# Save meta.json for traceability
$createdAtIso = Get-Date -Format o
$meta = @{
  run_id        = $runId
  variant       = $Variant
  variant_safe  = $VariantSafe
  created_at    = $createdAtIso
  runs_root     = $RunsRoot
  variant_dir   = $variantRunDir
  cache_root    = $CacheRoot
  cache_id      = $CacheIdSafe
  cache_dir     = $variantCacheDir
  data_root     = $DataRoot
  device        = $Device
  policy        = $Policy
  segment_sec   = $SegmentSec
  hop_sec       = $HopSec
  n_mels        = $NMels
  run_clip_policy = [bool]$RunClipPolicy
  time_conf     = $TimeConf
  time_stable_k = $TimeStableK
  time_min_windows = $TimeMinWindows
  eval_fixed_k_windows = $EvalFixedKWindows
  time_margin   = $TimeMargin
}
New-Item -ItemType Directory -Path $runPath -ErrorAction SilentlyContinue | Out-Null
$meta | ConvertTo-Json -Depth 8 | Out-File -FilePath (Join-Path $runPath "meta.json") -Encoding UTF8

# --------------------- 4/10) Calibrate temperatures ---------------------
Write-Host "`n[4/10] Calibrate temperatures..." -ForegroundColor Yellow
Invoke-Python ('python -m training.calibrate --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

# --------------------- 5/10) Select threshold (greedy path) ---------------------
Write-Host "`n[5/10] Select threshold (tau)..." -ForegroundColor Yellow
Invoke-Python ('python -m training.thresholds_offline --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

# --------------------- 6/10) Segment policy test ---------------------
Write-Host "`n[6/10] Segment policy test..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.policy_test --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

# --------------------- Guard: current clip tester is greedy-only ---------------------
if ($RunClipPolicy -and $Policy -ne "greedy") {
  throw "Current scripts.clip_policy_test.py is greedy-only. Use -Policy greedy, or create an EA-compatible clip_policy_test.py first."
}

# --------------------- 6b/10) Clip policy test (optional) ---------------------
if ($RunClipPolicy) {
  Write-Host "`n[6b/10] Clip policy test..." -ForegroundColor Yellow
  Invoke-Python ('python -m scripts.clip_policy_test --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --device "{3}" --time_conf {4} --time_stable_k {5} --time_min_windows {6} --fixed_k_windows {7} --time_margin {8}' -f `
    $runPath, $SegCsv, $FeatRoot, $Device, $TimeConf, $TimeStableK, $TimeMinWindows, $EvalFixedKWindows, $TimeMargin)
}

# --------------------- 7/10) Summarise run ---------------------
Write-Host "`n[7/10] Summarise run..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.summarize_run --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

# --------------------- 8/10) Analyse run ---------------------
Write-Host "`n[8/10] Analyse run..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.analyse_run --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

# --------------------- 9/10) Profile latency ---------------------
Write-Host "`n[9/10] Profile latency..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.profile_latency --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --variant "{3}" --device "{4}"' -f `
  $runPath, $SegCsv, $FeatRoot, $Variant, $Device)

# --------------------- Timing & logging ---------------------
$pipelineEnd   = Get-Date
$elapsed       = $pipelineEnd - $pipelineStart
$totalSeconds  = [Math]::Round($elapsed.TotalSeconds, 2)
$totalMinutes  = [Math]::Round($elapsed.TotalMinutes, 2)
$timestampIso  = Get-Date -Format o

Write-Host ""
Write-Host ("Total wall-clock time: {0} seconds (~{1} minutes)" -f $totalSeconds, $totalMinutes) -ForegroundColor Cyan

$analysisDir = "analysis"
New-Item -ItemType Directory -Path $analysisDir -ErrorAction SilentlyContinue | Out-Null
$runtimeCsv = Join-Path $analysisDir "pipeline_runtime.csv"

if (-not (Test-Path $runtimeCsv)) {
  "timestamp,variant,policy,segment_sec,hop_sec,device,cache_dir,runs_root,run_id,total_seconds,total_minutes,run_clip_policy" | Out-File $runtimeCsv -Encoding UTF8
}

$csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}" -f `
  $timestampIso, $Variant, $Policy, $SegmentSec, $HopSec, $Device, $variantCacheDir, $RunsRoot, $runId, $totalSeconds, $totalMinutes, [bool]$RunClipPolicy

Add-Content -Path $runtimeCsv -Value $csvLine
Write-Host "Pipeline runtime logged to: $runtimeCsv" -ForegroundColor DarkGray

# --------------------- 10/10) Reports & LaTeX ---------------------
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