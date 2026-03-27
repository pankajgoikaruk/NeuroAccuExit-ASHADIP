# scripts/run_full.ps1
param(
  [string]$DataRoot = "data\moth_sounds",
  [string]$CacheRoot = "data_caches",
  [string]$CacheId = "",
  [string]$Config = "configs\audio_moth.yaml",
  [string]$RunsRoot = "runs",
  [string]$Variant = "v0.5",

  # Policy override:
  # auto -> follow Variant rule (EA for EA/v0.2, otherwise greedy)
  # greedy-> force greedy tau
  # ea    -> force Depth-EA
  [ValidateSet("auto","greedy","ea")]
  [string]$Policy = "auto",

  [string]$Device = "cpu",
  [double]$SegmentSec = 1.0,
  [double]$HopSec = 0.5,

  # Depth-EA selector depth penalty (used only in training.ea_thresholds_offline)
  [double]$LambdaDepth = 0.08,

  # Auto lambda_depth for EA sweep (recommended): if enabled (or LambdaDepth<0),
  # uses 0.02 for K>=5 else 0.08.
  [switch]$AutoLambdaDepth,

  # Depth-EA minimum exit (0-indexed) used in training.ea_thresholds_offline:
  #   -1 = auto (K>=5 -> 2, else -> 0)
  #    0 = allow exit1
  #    1 = force at least exit2
  #    2 = force at least exit3 (recommended for K=5 to avoid exit1-heavy policies)
  [int]$EAMinExit = -1,

  # Step 0: K-exit + C-class knobs
  # K = len(tap_blocks) + 1
  [string]$TapBlocks = "1,3",
  [int]$NMels = 64,

  # Optional: run Depth×Time clip policy tests
  [switch]$RunClipPolicy,

  # Time-exit params (used only if -RunClipPolicy)
  [int]$TimeMinWindows = 2,
  [int]$TimeStableK = 2,
  [double]$TimeConf = 0.95,
  [double]$TimeMargin = 0.0,
  [int]$EvalFixedKWindows = 3,
  [switch]$PrintClipWindows
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Invoke-Python([string]$cmd) {
  Write-Host " $cmd" -ForegroundColor DarkGray
  iex $cmd
  if ($LASTEXITCODE -ne 0) { throw "Python command failed ($LASTEXITCODE): $cmd" }
}

function Num-ToId([double]$x) {
  return ($x.ToString("0.###") -replace '\.', 'p')  # 1.0->1p0
}

# --------------------- Run directory scheme: runs/<Variant>/<Variant>_### ---------------------
$VariantSafe = ($Variant -replace '[^A-Za-z0-9_-]', '_')
$variantRunDir = Join-Path $RunsRoot $VariantSafe

# --------------------- Decide whether to run Depth-EA ---------------------
$VariantLooksEA = ($Variant -match '^(?i:EA|v0\.?2|v0_?2)$')
$IsDepthEA = $false
if ($Policy -eq "ea") { $IsDepthEA = $true }
elseif ($Policy -eq "greedy") { $IsDepthEA = $false }
else { $IsDepthEA = $VariantLooksEA }  # auto

# --------------------- K exits + EA min-exit resolution ---------------------
$K = ($TapBlocks -split ',').Count + 1

# Resolve EA minimum-exit only when Depth-EA is active.
# Auto mode (-1): if K>=5 use ea_min_exit=2 (force exit3+), else allow exit1.
$EAMinExitEff = 0
if ($IsDepthEA) {
  $EAMinExitEff = $EAMinExit
  if ($EAMinExitEff -lt 0) {
    if ($K -ge 5) { $EAMinExitEff = 2 } else { $EAMinExitEff = 0 }
  }
  # Clamp to valid range [0..K-1]
  $maxExit = [Math]::Max(0, $K - 1)
  if ($EAMinExitEff -gt $maxExit) { $EAMinExitEff = $maxExit }
}

# Auto mode for EA sweep depth penalty:
# - if -AutoLambdaDepth is set OR LambdaDepth < 0, choose a safe default by K
#   (K>=5 -> 0.02; else -> 0.08).
$LambdaDepthEff = $LambdaDepth
$LambdaDepthAuto = $false
if ($AutoLambdaDepth -or $LambdaDepthEff -lt 0) {
  $LambdaDepthAuto = $true
  if ($K -ge 5) { $LambdaDepthEff = 0.02 } else { $LambdaDepthEff = 0.08 }
}

Write-Host "== ASHADIP: full pipeline run ==" -ForegroundColor Cyan
Write-Host " Variant = $Variant" -ForegroundColor DarkGray
Write-Host " Policy(param) = $Policy" -ForegroundColor DarkGray
Write-Host " Depth-EA active = $IsDepthEA" -ForegroundColor DarkGray
Write-Host " TapBlocks = $TapBlocks  (K = $K)" -ForegroundColor DarkGray
Write-Host " n_mels = $NMels" -ForegroundColor DarkGray
if ($IsDepthEA) {
  $ld_note = ""
  if ($LambdaDepthAuto) { $ld_note = " (auto)" }
  Write-Host " LambdaDepth = $LambdaDepthEff$ld_note" -ForegroundColor DarkGray
  Write-Host " EA min exit (0-index) = $EAMinExitEff" -ForegroundColor DarkGray
}

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

# --------------------- Cache directory scheme: data_caches/<Variant>/<CacheId> ---------------------
if ([string]::IsNullOrWhiteSpace($CacheId)) {
  $segId = Num-ToId $SegmentSec
  $hopId = Num-ToId $HopSec
  $CacheId = "seg$segId" + "_hop$hopId" + "_bp100-3000" + "_mels$NMels"
}
$CacheIdSafe = ($CacheId -replace '[^A-Za-z0-9_-]', '_')

New-Item -ItemType Directory -Path $CacheRoot -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path (Join-Path $CacheRoot $VariantSafe) -ErrorAction SilentlyContinue | Out-Null
$variantCacheDir = Join-Path (Join-Path $CacheRoot $VariantSafe) $CacheIdSafe
New-Item -ItemType Directory -Path $variantCacheDir -ErrorAction SilentlyContinue | Out-Null

$SegCsv = Join-Path $variantCacheDir "segments.csv"
$FeatRoot = Join-Path $variantCacheDir "features"

$pipelineStart = Get-Date

Write-Host " DataRoot = $DataRoot" -ForegroundColor DarkGray
Write-Host " CacheDir = $variantCacheDir" -ForegroundColor DarkGray
Write-Host " Config = $Config" -ForegroundColor DarkGray
Write-Host " Device = $Device" -ForegroundColor DarkGray
Write-Host " SegmentSec = $SegmentSec | HopSec = $HopSec" -ForegroundColor DarkGray
Write-Host " RunDir = $runPath" -ForegroundColor DarkGray

$cacheReady = (Test-Path $SegCsv) -and (Test-Path $FeatRoot)
if ($cacheReady) {
  Write-Host "`n[cache] Reusing existing cache: $variantCacheDir" -ForegroundColor Green
} else {
  Write-Host "`n[cache] Cache not ready; will generate segments + features." -ForegroundColor DarkYellow
}

# --------------------- 1/11) Prep segments ---------------------
if (-not $cacheReady) {
  Write-Host "`n[1/11] Prep segments..." -ForegroundColor Yellow
  Invoke-Python ('python -m scripts.prep_segments --root "{0}" --cache "{1}" --sr 16000 --segment_sec {2} --hop {3} --silence_dbfs -40 --bandpass 100 3000 --config "{4}"' -f `
    $DataRoot, $variantCacheDir, $SegmentSec, $HopSec, $Config)

  # --------------------- 2/11) Extract features ---------------------
  Write-Host "`n[2/11] Extract features..." -ForegroundColor Yellow
  Invoke-Python ('python -m scripts.extract_features --cache "{0}" --n_mels {1} --n_fft 1024 --win_ms 25 --hop_ms 10 --cmvn' -f `
    $variantCacheDir, $NMels)
} else {
  Write-Host "`n[1/11] Prep segments... (skip; cache ready)" -ForegroundColor DarkGray
  Write-Host "[2/11] Extract features... (skip; cache ready)" -ForegroundColor DarkGray
}

# --------------------- 3/11) Train ExitNet ---------------------
Write-Host "`n[3/11] Train ExitNet..." -ForegroundColor Yellow
Invoke-Python ('python -m training.train --config "{0}" --run_dir "{1}" --cache_dir "{2}" --device "{3}" --segment_sec {4} --hop_sec {5} --variant "{6}"' -f `
  $Config, $runPath, $variantCacheDir, $Device, $SegmentSec, $HopSec, $Variant)

Write-Host "Using run: $runPath" -ForegroundColor Green

# --------------------- 4/11) Calibrate temperatures ---------------------
Write-Host "`n[4/11] Calibrate temperatures..." -ForegroundColor Yellow
Invoke-Python ('python -m training.calibrate --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4}' -f `
  $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels)

# --------------------- 5/11) Select thresholds ---------------------
if ($IsDepthEA) {
  Write-Host "`n[5/11] Select EA thresholds (Depth-EA)..." -ForegroundColor Yellow
  Invoke-Python ('python -m training.ea_thresholds_offline --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4} --ea_min_exit {5} --lambda_depth {6}' -f `
    $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels, $EAMinExitEff, $LambdaDepthEff)
} else {
  Write-Host "`n[5/11] Select greedy threshold (tau)..." -ForegroundColor Yellow
  Invoke-Python ('python -m training.thresholds_offline --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4}' -f `
    $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels)
}

# --------------------- 6/11) Segment policy test ---------------------
Write-Host "`n[6/11] Segment policy test..." -ForegroundColor Yellow
if ($IsDepthEA) {
  Invoke-Python ('python -m scripts.policy_test --policy ea --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4}' -f `
    $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels)
} else {
  Invoke-Python ('python -m scripts.policy_test --policy greedy --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4}' -f `
    $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels)
}

# --------------------- 7/11) Clip policy test (optional) ---------------------
if ($RunClipPolicy) {
  Write-Host "`n[7/11] Clip policy test (FULL baseline)..." -ForegroundColor Yellow
  $fixedK = [Math]::Max(0, [int]$EvalFixedKWindows)

  $printFlag = ""
  if ($PrintClipWindows) { $printFlag = "--print_clip_windows" }

  Invoke-Python ('python -m scripts.clip_policy_test --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4} --disable_time_exit --eval_fixed_k_windows {5} {6}' -f `
    $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels, $fixedK, $printFlag)

  Write-Host "`n[7/11] Clip policy test (Depth×Time)..." -ForegroundColor Yellow
  Invoke-Python ('python -m scripts.clip_policy_test --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4} --time_conf {5} --time_stable_k {6} --time_min_windows {7} --time_margin {8} --eval_fixed_k_windows {9} --full_baseline_json "{10}" {11}' -f `
    $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels, $TimeConf, $TimeStableK, $TimeMinWindows, $TimeMargin, $fixedK, (Join-Path $runPath "clip_policy_results_full.json"), $printFlag)
} else {
  Write-Host "`n[7/11] Clip policy test... (skip; pass -RunClipPolicy to enable)" -ForegroundColor DarkGray
}

# --------------------- 8/11) Summarise run ---------------------
Write-Host "`n[8/11] Summarise run..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.summarize_run --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4}' -f `
  $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels)

# --------------------- 9/11) Analyse run ---------------------
Write-Host "`n[9/11] Analyse run..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.analyse_run --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --tap_blocks {3} --n_mels {4}' -f `
  $runPath, $SegCsv, $FeatRoot, $TapBlocks, $NMels)

# --------------------- 10/11) Profile latency ---------------------
Write-Host "`n[10/11] Profile latency..." -ForegroundColor Yellow
Invoke-Python ('python -m scripts.profile_latency --run_dir "{0}" --segments_csv "{1}" --features_root "{2}" --variant "{3}" --device "{4}" --tap_blocks {5} --n_mels {6}' -f `
  $runPath, $SegCsv, $FeatRoot, $Variant, $Device, $TapBlocks, $NMels)

# --------------------- Optional) W&B post-hoc logging ---------------------
if (-not [string]::IsNullOrWhiteSpace($env:ENABLE_WANDB)) {
  Write-Host "`n[W&B] Logging run artifacts to Weights & Biases..." -ForegroundColor Yellow
  Invoke-Python ('python -m scripts.wandb_log_run --run_dir "{0}" --log_plots' -f $runPath)
} else {
  Write-Host "`n[W&B] Skipped (set ENABLE_WANDB=1 to enable)." -ForegroundColor DarkGray
}

# --------------------- Timing & runtime log ---------------------
$pipelineEnd = Get-Date
$elapsed = $pipelineEnd - $pipelineStart
$totalSeconds = [Math]::Round($elapsed.TotalSeconds, 2)
$totalMinutes = [Math]::Round($elapsed.TotalMinutes, 2)
$timestampIso = Get-Date -Format o

Write-Host ""
Write-Host ("Total wall-clock time: {0} seconds (~{1} minutes)" -f $totalSeconds, $totalMinutes) -ForegroundColor Cyan

$analysisDir = "analysis"
New-Item -ItemType Directory -Path $analysisDir -ErrorAction SilentlyContinue | Out-Null
$runtimeCsv = Join-Path $analysisDir "pipeline_runtime.csv"
if (-not (Test-Path $runtimeCsv)) {
  "timestamp,variant,policy,depth_ea,lambda_depth,segment_sec,hop_sec,device,cache_dir,run_id,total_seconds,total_minutes,tap_blocks,n_mels,clip_policy" | Out-File $runtimeCsv -Encoding UTF8
}
$csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}" -f `
  $timestampIso, $Variant, $Policy, $IsDepthEA, $LambdaDepthEff, $SegmentSec, $HopSec, $Device, $variantCacheDir, $runId, $totalSeconds, $totalMinutes, $TapBlocks, $NMels, ([int]$RunClipPolicy.IsPresent)
Add-Content -Path $runtimeCsv -Value $csvLine
Write-Host "Pipeline runtime logged to: $runtimeCsv" -ForegroundColor DarkGray

Write-Host "`n== Done. Artifacts at: $runPath ==" -ForegroundColor Cyan
Write-Host "Cache used: $variantCacheDir" -ForegroundColor DarkGray