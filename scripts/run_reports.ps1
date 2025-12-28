# scripts/run_reports.ps1

param(
  [Parameter(Mandatory = $true)]
  [string]$RunDir,                # e.g. "runs\EA\EA_003" OR just "EA_003"

  [string]$Variant      = "V0",   # used only to help resolve RunDir when short id is provided
  [string]$DeviceFilter = "cpu",  # currently used only as a label in logs

  # NOTE: defaults; if not explicitly provided, we will override from meta.json.cache_dir
  [string]$SegmentsCsv  = "data_cache\segments.csv",
  [string]$FeaturesRoot = "data_cache\features",

  # runs root used to resolve short RunDir values (like "EA_003")
  [string]$RunsRoot     = "runs"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "== ASHADIP: generate reports & tables ==" -ForegroundColor Cyan
Write-Host "  RunDir       = $RunDir"
Write-Host "  Variant(arg) = $Variant"
Write-Host "  DeviceFilter = $DeviceFilter"
Write-Host "  SegmentsCsv  = $SegmentsCsv"
Write-Host "  FeaturesRoot = $FeaturesRoot"
Write-Host "  RunsRoot     = $RunsRoot"
Write-Host ""

# --------------------- Resolve RunDir robustly ---------------------
# Support:
#   1) RunDir is an existing path
#   2) RunDir is a short run id like "EA_003" -> runs\<VariantSafe>\EA_003
#   3) Try auto-find: runs\*\EA_003  (helps if Variant arg is wrong)
#   4) Legacy fallback: runs\<RunDir> (likely rejected by strict meta requirement)
$VariantSafe = ($Variant -replace '[^A-Za-z0-9_-]', '_')

function Resolve-RunPath([string]$InputRunDir) {
  # Case 1: direct path exists
  if (Test-Path $InputRunDir) {
    return (Resolve-Path $InputRunDir).Path
  }

  # Case 2: runs\<VariantSafe>\<InputRunDir>
  $cand2 = Join-Path (Join-Path $RunsRoot $VariantSafe) $InputRunDir
  if (Test-Path $cand2) {
    return (Resolve-Path $cand2).Path
  }

  # Case 3: auto-find under runs\*\InputRunDir
  $matches = @()
  if (Test-Path $RunsRoot) {
    Get-ChildItem -Path $RunsRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
      $cand = Join-Path $_.FullName $InputRunDir
      if (Test-Path $cand) { $matches += (Resolve-Path $cand).Path }
    }
  }
  if ($matches.Count -eq 1) { return $matches[0] }
  if ($matches.Count -gt 1) {
    throw "Ambiguous RunDir '$InputRunDir'. Found multiple matches under '$RunsRoot': `n - " + ($matches -join "`n - ")
  }

  # Case 4: legacy runs\<InputRunDir>
  $cand4 = Join-Path $RunsRoot $InputRunDir
  if (Test-Path $cand4) {
    return (Resolve-Path $cand4).Path
  }

  throw "Run directory not found. Tried: '$InputRunDir', '$cand2', '$RunsRoot\*\$InputRunDir', '$cand4'"
}

$runPath = Resolve-RunPath $RunDir
Write-Host "Resolved RunDir -> $runPath" -ForegroundColor Green

# --------------------- STRICT: require meta.json ---------------------
$metaPath = Join-Path $runPath "meta.json"
if (-not (Test-Path $metaPath)) {
  throw "STRICT mode: meta.json not found at $metaPath. This script only supports NEW runs (runs/<Variant>/<RunId>/...)."
}

# Read meta.json for canonical run_id + variant + cache_dir
$meta = Get-Content -Raw -Path $metaPath | ConvertFrom-Json

$RunIdEffective = $meta.run_id
if (-not $RunIdEffective -or "$RunIdEffective".Trim().Length -eq 0) {
  $RunIdEffective = Split-Path $runPath -Leaf
}

$VariantEffective = $meta.variant
if (-not $VariantEffective -or "$VariantEffective".Trim().Length -eq 0) {
  if ($meta.variant_safe -and "$($meta.variant_safe)".Trim().Length -gt 0) {
    $VariantEffective = $meta.variant_safe
  } else {
    $VariantEffective = $Variant
  }
}

Write-Host "Canonical (from meta.json): Variant=$VariantEffective, RunId=$RunIdEffective" -ForegroundColor Green

# --------------------- Auto-resolve cache paths from meta.json ---------------------
$defaultSeg  = "data_cache\segments.csv"
$defaultFeat = "data_cache\features"

$metaCacheDir = $meta.cache_dir
if ($metaCacheDir -and "$metaCacheDir".Trim().Length -gt 0) {
  $metaCacheDirPath = $metaCacheDir

  # If meta.cache_dir is relative, make it relative to project root (cwd)
  if (-not [System.IO.Path]::IsPathRooted($metaCacheDirPath)) {
    $metaCacheDirPath = Join-Path (Get-Location).Path $metaCacheDirPath
  }

  # Override only if user didn't explicitly pass custom values (still default)
  if (($SegmentsCsv -eq $defaultSeg) -or [string]::IsNullOrWhiteSpace($SegmentsCsv)) {
    $SegmentsCsv = Join-Path $metaCacheDirPath "segments.csv"
  }
  if (($FeaturesRoot -eq $defaultFeat) -or [string]::IsNullOrWhiteSpace($FeaturesRoot)) {
    $FeaturesRoot = Join-Path $metaCacheDirPath "features"
  }

  Write-Host "Cache (from meta.json): $metaCacheDir" -ForegroundColor DarkGray
  Write-Host "Using SegmentsCsv  -> $SegmentsCsv" -ForegroundColor DarkGray
  Write-Host "Using FeaturesRoot -> $FeaturesRoot" -ForegroundColor DarkGray
} else {
  Write-Host "[warn] meta.json has no cache_dir. Using provided SegmentsCsv/FeaturesRoot as-is." -ForegroundColor DarkYellow
}

# Validate cache paths exist (fail early)
if (-not (Test-Path $SegmentsCsv)) { throw "segments.csv not found: $SegmentsCsv" }
if (-not (Test-Path $FeaturesRoot)) { throw "features root not found: $FeaturesRoot" }

# Ensure analysis folders exist (global + per-run)
New-Item -ItemType Directory -Path "analysis" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "analysis\tables" -ErrorAction SilentlyContinue | Out-Null

# Per-run analysis dir to avoid overwriting tables
$VariantEffectiveSafe = ($VariantEffective -replace '[^A-Za-z0-9_-]', '_')
$runAnalysisDir = Join-Path (Join-Path "analysis\runs" $VariantEffectiveSafe) $RunIdEffective
New-Item -ItemType Directory -Path $runAnalysisDir -ErrorAction SilentlyContinue | Out-Null

# ---------------------------------------------------------------------------
# [1/5] Ensure report.json exists (test-set classification per exit)
# ---------------------------------------------------------------------------
Write-Host "[1/5] Evaluating test set (report.json)..." -ForegroundColor Yellow

python -m training.eval `
  --run_dir "$runPath" `
  --segments_csv "$SegmentsCsv" `
  --features_root "$FeaturesRoot"

Write-Host "  -> training.eval finished." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [2/5] Summarise run (summary.json + calibration plots)
# ---------------------------------------------------------------------------
Write-Host "`n[2/5] Summarising run (summary.json, calibration plots)..." -ForegroundColor Yellow

python -m scripts.summarize_run `
  --run_dir "$runPath" `
  --segments_csv "$SegmentsCsv" `
  --features_root "$FeaturesRoot"

Write-Host "  -> summarize_run finished." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [3/5] Global variants summary CSV (analysis/all_runs_summary.csv)
# ---------------------------------------------------------------------------
Write-Host "`n[3/5] Updating global variants summary (analysis/all_runs_summary.csv)..." -ForegroundColor Yellow

python -m scripts.compare_variants --root .

Write-Host "  -> compare_variants finished." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [4/5] Per-run classification LaTeX table (no overwrites)
#     analysis/runs/<Variant>/<RunId>/classification_table.*
# ---------------------------------------------------------------------------
Write-Host "`n[4/5] Generating classification table for this run..." -ForegroundColor Yellow

$analysisJson = Join-Path $runPath "analysis_run.json"
if (-not (Test-Path $analysisJson)) {
  throw "analysis_run.json not found at $analysisJson. Did analyse_run.py run?"
}

$outClsTex = Join-Path $runAnalysisDir "classification_table.tex"
$runLabel  = $RunIdEffective

python -m scripts.analysis_to_latex `
  --analysis_json "$analysisJson" `
  --out_tex "$outClsTex" `
  --run_label "$runLabel"

Write-Host "  -> Wrote per-run classification table: $outClsTex" -ForegroundColor Green

# ---------------------------------------------------------------------------
# [5/5] Cross-variant summary + on-device performance tables (global)
# ---------------------------------------------------------------------------
Write-Host "`n[5/5] Generating cross-variant & on-device LaTeX tables..." -ForegroundColor Yellow

$allRunsCsv = "analysis\all_runs_summary.csv"
if (Test-Path $allRunsCsv) {
  $variantsTex = "analysis\tables\variants_avg_summary_table.tex"
  python -m scripts.variants_to_latex `
    --summary_csv "$allRunsCsv" `
    --out_tex "$variantsTex"
  Write-Host "  -> Wrote variants summary table: $variantsTex" -ForegroundColor Green
}
else {
  Write-Host "  (skip) analysis\all_runs_summary.csv not found; run compare_variants.py first." -ForegroundColor DarkYellow
}

$onDevCsv = "analysis\on_device_summary.csv"
if (Test-Path $onDevCsv) {
  $onDevTex = "analysis\tables\on_device_performance_table.tex"
  python -m scripts.ondevice_to_latex `
    --summary_csv "$onDevCsv" `
    --out_tex "$onDevTex"
  Write-Host "  -> Wrote on-device performance table: $onDevTex" -ForegroundColor Green
}
else {
  Write-Host "  (skip) analysis\on_device_summary.csv not found; run profile_latency.py / run_full.ps1 first." -ForegroundColor DarkYellow
}

Write-Host "`n== Done. Reports updated for $VariantEffective / $RunIdEffective ==" -ForegroundColor Cyan
Write-Host "Per-run tables at: $runAnalysisDir" -ForegroundColor DarkGray
Write-Host "Global tables at : analysis\tables" -ForegroundColor DarkGray
