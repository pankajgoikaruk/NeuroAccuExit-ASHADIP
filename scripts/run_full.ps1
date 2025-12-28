# scripts/run_full.ps1

param(
  [string]$DataRoot   = "data\moth_sounds",

  # Cache root
  [string]$CacheRoot  = "data_caches",

  # Optional cache id (if not provided, it is auto-generated from params)
  [string]$CacheId    = "",

  [string]$Config     = "configs\audio_moth.yaml",
  [string]$RunsRoot   = "runs",
  [string]$Variant    = "V0",
  [string]$Device     = "cpu",
  [double]$SegmentSec = 1.0,
  [double]$HopSec     = 0.5
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
# Avoid OpenMP duplicate runtime crash on Windows (Intel MKL / matplotlib / numpy combos)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Invoke-Python([string]$cmd) {
  Write-Host "  $cmd" -ForegroundColor DarkGray
  iex $cmd
  if ($LASTEXITCODE -ne 0) { throw "Python command failed ($LASTEXITCODE): $cmd" }
}

function Num-ToId([double]$x) {
  # 1.0 -> 1p0, 0.5 -> 0p5
  return ($x.ToString("0.###") -replace '\.', 'p')
}

# --------------------- Run directory scheme: runs/<Variant>/<Variant>_### ---------------------
$VariantSafe = ($Variant -replace '[^A-Za-z0-9_-]', '_')
$variantRunDir = Join-Path $RunsRoot $VariantSafe

New-Item -ItemType Directory -Path $RunsRoot       -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantRunDir  -ErrorAction SilentlyContinue | Out-Null

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

# --------------------- Cache directory scheme: data_caches/<Variant>/<cache_id>/... ---------------------
# Auto CacheId if not provided (encode the things that change the cache)
if ([string]::IsNullOrWhiteSpace($CacheId)) {
  $segId = Num-ToId $SegmentSec
  $hopId = Num-ToId $HopSec
  # Keep it short but informative; extend later if you change feature params
  $CacheId = "seg$segId" + "_hop$hopId" + "_bp100-3000" + "_mels64"
}

$CacheIdSafe = ($CacheId -replace '[^A-Za-z0-9_-]', '_')
$variantCacheDir = Join-Path (Join-Path $CacheRoot $VariantSafe) $CacheIdSafe

New-Item -ItemType Directory -Path $CacheRoot -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path (Join-Path $CacheRoot $VariantSafe) -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantCacheDir -ErrorAction SilentlyContinue | Out-Null

# Common cache paths
$SegCsv   = Join-Path $variantCacheDir "segments.csv"
$FeatRoot = Join-Path $variantCacheDir "features"

# Start wall-clock timer
$pipelineStart = Get-Date

Write-Host "== ASHADIP: full pipeline run ==" -ForegroundColor Cyan
Write-Host "  DataRoot   = $DataRoot"        -ForegroundColor DarkGray
Write-Host "  CacheRoot  = $CacheRoot"       -ForegroundColor DarkGray
Write-Host "  CacheId    = $CacheIdSafe"     -ForegroundColor DarkGray
Write-Host "  CacheDir   = $variantCacheDir" -ForegroundColor DarkGray
Write-Host "  Config     = $Config"          -ForegroundColor DarkGray
Write-Host "  RunsRoot   = $RunsRoot"        -ForegroundColor DarkGray
Write-Host "  Variant    = $Variant"         -ForegroundColor DarkGray
Write-Host "  Device     = $Device"          -ForegroundColor DarkGray
Write-Host "  SegmentSec = $SegmentSec"      -ForegroundColor DarkGray
Write-Host "  HopSec     = $HopSec"          -ForegroundColor DarkGray
Write-Host "  RunDir     = $runPath"         -ForegroundColor DarkGray

$cacheReady = (Test-Path $SegCsv) -and (Test-Path $FeatRoot)

if ($cacheReady) {
  Write-Host "`n[cache] Reusing existing cache: $variantCacheDir" -ForegroundColor Green
} else {
  # run step 1 + 2
}


# --------------------- 1/10) Prep segments ---------------------
Write-Host "`n[1/10] Prep segments..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.prep_segments --root "{0}" --cache "{1}" --sr 16000 --segment_sec {2} --hop {3} --silence_dbfs -40 --bandpass 100 3000' -f `
  $DataRoot, $variantCacheDir, $SegmentSec, $HopSec)

# --------------------- 2/10) Extract features ---------------------
Write-Host "`n[2/10] Extract features..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.extract_features --cache "{0}" --n_mels 64 --n_fft 1024 --win_ms 25 --hop_ms 10 --cmvn' -f `
  $variantCacheDir)

# --------------------- 3/10) Train ExitNet ---------------------
Write-Host "`n[3/10] Train ExitNet..." -ForegroundColor Yellow
Invoke-Python ('python -m training.train --config "{0}" --run_dir "{1}" --cache_dir "{2}" --device "{3}" --segment_sec {4} --hop_sec {5} --variant "{6}"' -f `
  $Config, $runPath, $variantCacheDir, $Device, $SegmentSec, $HopSec, $Variant)

Write-Host "Using run: $runPath" -ForegroundColor Green

# Save meta.json for traceability
$createdAtIso = Get-Date -Format o
$meta = @{
  run_id       = $runId
  variant      = $Variant
  variant_safe = $VariantSafe
  created_at   = $createdAtIso
  runs_root    = $RunsRoot
  variant_dir  = $variantRunDir

  cache_root   = $CacheRoot
  cache_id     = $CacheIdSafe
  cache_dir    = $variantCacheDir

  data_root    = $DataRoot
  device       = $Device
  segment_sec  = $SegmentSec
  hop_sec      = $HopSec
}
New-Item -ItemType Directory -Path $runPath -ErrorAction SilentlyContinue | Out-Null
$meta | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $runPath "meta.json") -Encoding UTF8

# --------------------- 4/10) Calibrate temperatures ---------------------
Write-Host "`n[4/10] Calibrate temperatures..." -ForegroundColor Yellow
Invoke-Python ('python -m training.calibrate --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

# --------------------- 5/10) Select threshold (tau) ---------------------
Write-Host "`n[5/10] Select threshold (tau)..." -ForegroundColor Yellow
Invoke-Python ('python -m training.thresholds_offline --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

# --------------------- 6/10) Policy test ---------------------
Write-Host "`n[6/10] Policy test..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.policy_test --run_dir "{0}" --segments_csv "{1}" --features_root "{2}"' -f `
  $runPath, $SegCsv, $FeatRoot)

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
  "timestamp,variant,segment_sec,hop_sec,device,cache_dir,runs_root,run_id,total_seconds,total_minutes" | Out-File $runtimeCsv -Encoding UTF8
}

$csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f `
  $timestampIso, $Variant, $SegmentSec, $HopSec, $Device, $variantCacheDir, $RunsRoot, $runId, $totalSeconds, $totalMinutes

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
