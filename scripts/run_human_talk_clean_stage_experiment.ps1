# scripts/run_human_talk_clean_stage_experiment.ps1
#
# One-command experiment runner for the human-talk incremental evaluation branch.
#
# Default behaviour:
#   1. Prepare the selected clean stage
#   2. Extract log-mel features
#   3. Train 3-exit no-hint model
#   4. Run 3-exit greedy policy
#   5. Train 5-exit no-hint model
#   6. Run 5-exit greedy policy
#   7. Save complete CLI transcript
#   8. Create compact ZIP package for sharing when -ZipResults is used
#
# Default stage:
#   clean2_balanced = Les_Brown vs Simon_Sinek
#
# Example:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
#     -Stage clean2_balanced `
#     -RawRoot human_talk_dataset `
#     -WorkspaceRoot human_talk_workspace `
#     -Device cpu `
#     -Clean `
#     -ZipResults
#
# Resume examples:
#   # Only train/evaluate using existing prepared data/cache:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
#     -Stage clean2_balanced -SkipPrepare -SkipFeatures -ZipResults
#
#   # Only package existing outputs:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clean_stage_experiment.ps1 `
#     -Stage clean2_balanced -ZipOnly

param(
    [string]$Stage = "clean2_balanced",
    [string]$RawRoot = "human_talk_dataset",
    [string]$WorkspaceRoot = "human_talk_workspace",

    [string]$Device = "cpu",
    [int]$Epochs = 40,
    [int]$BatchSize = 64,
    [double]$LR = 0.001,

    [double]$SegmentSec = 1.0,
    [double]$HopSec = 0.5,
    [int]$SampleRate = 16000,
    [int]$Seed = 42,
    [string]$FilenameSeparator = "__",

    [string]$ThresholdMode = "fixed_0p5",

    [switch]$Clean,
    [switch]$SkipPrepare,
    [switch]$SkipFeatures,
    [switch]$SkipTrain3,
    [switch]$SkipTrain5,
    [switch]$SkipPolicy,
    [switch]$SkipPolicy3,
    [switch]$SkipPolicy5,
    [switch]$ZipResults,
    [switch]$IncludeAllRuns,
    [switch]$ZipOnly
)

$ErrorActionPreference = "Stop"

function Get-TimeStamp {
    return (Get-Date).ToString("yyyyMMdd_HHmmss")
}

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " $Title"
    Write-Host "============================================================"

    $Start = Get-Date
    & $Command
    $End = Get-Date
    $Duration = $End - $Start

    Write-Host ""
    Write-Host "Completed: $Title in $Duration"
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

function Get-LatestRunDir {
    param(
        [string]$RunsRoot,
        [string]$Pattern
    )

    $Run = Get-ChildItem $RunsRoot -Directory -Filter $Pattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $Run) {
        throw "No run directory found under $RunsRoot matching pattern: $Pattern"
    }

    return $Run.FullName
}

function Copy-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (Test-Path $Source) {
        Copy-Item $Source $Destination -Force -ErrorAction SilentlyContinue
    }
}

function New-ResultsPackage {
    param(
        [string]$StageRoot,
        [string]$WorkspaceRoot,
        [string]$Stage,
        [string]$TranscriptPath,
        [string]$VariantPrefix,
        [string]$ScriptPath,
        [switch]$IncludeAllRuns
    )

    $Timestamp = Get-TimeStamp
    $PackagesRoot = Join-Path $WorkspaceRoot "packages"
    New-Item -ItemType Directory -Force -Path $PackagesRoot | Out-Null

    $ZipPath = Join-Path $PackagesRoot ("human_talk_${Stage}_results_to_share_${Timestamp}.zip")

    # Use OS temp folder so the repo root does not become messy.
    $TempShare = Join-Path $env:TEMP ("human_talk_${Stage}_share_${Timestamp}")
    Remove-Item $TempShare -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $TempShare | Out-Null

    Write-Host ""
    Write-Host "Packaging results..."
    Write-Host "Temp staging: $TempShare"
    Write-Host "ZIP output:   $ZipPath"

    # Stage metadata
    $StageMetadata = Join-Path $StageRoot "data\metadata"
    $DestStageMetadata = Join-Path $TempShare "stage_metadata"
    New-Item -ItemType Directory -Force -Path $DestStageMetadata | Out-Null

    Copy-IfExists "$StageMetadata\*.csv"  $DestStageMetadata
    Copy-IfExists "$StageMetadata\*.md"   $DestStageMetadata
    Copy-IfExists "$StageMetadata\*.json" $DestStageMetadata

    # Cache metadata only, not .npy features
    $CacheMetadata = Join-Path $StageRoot "cache\metadata"
    $DestCacheMetadata = Join-Path $TempShare "cache_metadata"
    New-Item -ItemType Directory -Force -Path $DestCacheMetadata | Out-Null

    Copy-IfExists "$CacheMetadata\*.csv"  $DestCacheMetadata
    Copy-IfExists "$CacheMetadata\*.json" $DestCacheMetadata
    Copy-IfExists "$CacheMetadata\*.md"   $DestCacheMetadata

    # Run outputs
    $RunsRoot = Join-Path $StageRoot "runs"
    $DestRunsRoot = Join-Path $TempShare "runs"
    New-Item -ItemType Directory -Force -Path $DestRunsRoot | Out-Null

    $RunPatterns = @(
        "${VariantPrefix}_3exit_nohint_*",
        "${VariantPrefix}_5exit_nohint_*"
    )

    foreach ($Pattern in $RunPatterns) {
        $Runs = Get-ChildItem $RunsRoot -Directory -Filter $Pattern -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending

        if (-not $IncludeAllRuns) {
            $Runs = $Runs | Select-Object -First 1
        }

        foreach ($Run in $Runs) {
            $DestRun = Join-Path $DestRunsRoot $Run.Name
            New-Item -ItemType Directory -Force -Path $DestRun | Out-Null

            # Copy useful top-level text/table/config files only.
            Copy-IfExists "$($Run.FullName)\*.json" $DestRun
            Copy-IfExists "$($Run.FullName)\*.csv"  $DestRun
            Copy-IfExists "$($Run.FullName)\*.md"   $DestRun
            Copy-IfExists "$($Run.FullName)\*.txt"  $DestRun
            Copy-IfExists "$($Run.FullName)\*.yaml" $DestRun
            Copy-IfExists "$($Run.FullName)\*.yml"  $DestRun

            # Copy policy outputs if present.
            $PolicyDir = Join-Path $Run.FullName "multilabel_greedy_policy"
            if (Test-Path $PolicyDir) {
                Copy-Item $PolicyDir (Join-Path $DestRun "multilabel_greedy_policy") -Recurse -Force
            }

            # Copy threshold tuning outputs if present.
            $ThreshDir = Join-Path $Run.FullName "threshold_tuning"
            if (Test-Path $ThreshDir) {
                Copy-Item $ThreshDir (Join-Path $DestRun "threshold_tuning") -Recurse -Force
            }

            # Do NOT copy ckpt/, .pt, .npy, raw segments, or generated features.
        }
    }

    # Logs and runner script
    $DestLogs = Join-Path $TempShare "logs"
    New-Item -ItemType Directory -Force -Path $DestLogs | Out-Null

    if (Test-Path $TranscriptPath) {
        Copy-Item $TranscriptPath (Join-Path $DestLogs (Split-Path $TranscriptPath -Leaf)) -Force
    }

    if (Test-Path $ScriptPath) {
        Copy-Item $ScriptPath (Join-Path $TempShare "run_human_talk_clean_stage_experiment.ps1") -Force
    }

    # Package README
    @"
Human-talk $Stage results package

Included:
- Stage metadata CSV/MD/JSON
- Cache metadata only
- latest 3-exit and 5-exit run summaries/configs by default
- Greedy early-exit policy outputs
- Threshold tuning outputs if present
- Full CLI transcript
- Experiment runner script copy

Optional:
- Use -IncludeAllRuns to package all matching runs instead of only the latest matching 3-exit and 5-exit runs.

Excluded:
- raw dataset
- generated WAV segments
- log-mel .npy feature files
- model checkpoints
"@ | Out-File (Join-Path $TempShare "README_results_package.txt") -Encoding utf8

    Compress-Archive -Path (Join-Path $TempShare "*") -DestinationPath $ZipPath -Force

    # Remove temporary staging folder so only ZIP remains.
    Remove-Item $TempShare -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host ""
    Write-Host "Created ZIP:"
    Write-Host "  $ZipPath"

    return $ZipPath
}

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
$StageRoot = Join-Path $WorkspaceRoot ("stages\" + $Stage)
$DataRoot = Join-Path $StageRoot "data"
$CacheRoot = Join-Path $StageRoot "cache"
$RunsRoot = Join-Path $StageRoot "runs"
$LogsRoot = Join-Path $WorkspaceRoot "logs"
$VariantPrefix = Get-VariantPrefix -StageName $Stage

New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null

$Timestamp = Get-TimeStamp
$TranscriptPath = Join-Path $LogsRoot ("${VariantPrefix}_full_pipeline_${Timestamp}_cli_output.txt")
$ScriptPath = $MyInvocation.MyCommand.Path

Start-Transcript -Path $TranscriptPath -Force | Out-Null

try {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Human-talk clean-stage experiment runner"
    Write-Host "============================================================"
    Write-Host "Stage:          $Stage"
    Write-Host "RawRoot:        $RawRoot"
    Write-Host "WorkspaceRoot:  $WorkspaceRoot"
    Write-Host "StageRoot:      $StageRoot"
    Write-Host "DataRoot:       $DataRoot"
    Write-Host "CacheRoot:      $CacheRoot"
    Write-Host "RunsRoot:       $RunsRoot"
    Write-Host "Device:         $Device"
    Write-Host "Epochs:         $Epochs"
    Write-Host "BatchSize:      $BatchSize"
    Write-Host "LR:             $LR"
    Write-Host "ThresholdMode:  $ThresholdMode"
    Write-Host "FilenameSep:    $FilenameSeparator"
    Write-Host "IncludeAllRuns: $IncludeAllRuns"
    Write-Host "VariantPrefix:  $VariantPrefix"
    Write-Host "Transcript:     $TranscriptPath"
    Write-Host "============================================================"

    if ($ZipOnly) {
        Invoke-Step "Create ZIP package only" {
            New-ResultsPackage `
                -StageRoot $StageRoot `
                -WorkspaceRoot $WorkspaceRoot `
                -Stage $Stage `
                -TranscriptPath $TranscriptPath `
                -VariantPrefix $VariantPrefix `
                -ScriptPath $ScriptPath `
                -IncludeAllRuns:$IncludeAllRuns | Out-Null
        }
        return
    }

    # ---------------------------------------------------------------------
    # 1. Prepare stage
    # ---------------------------------------------------------------------
    if (-not $SkipPrepare) {
        Invoke-Step "Prepare $Stage data and metadata" {
            $PrepareCmd = @(
                "-ExecutionPolicy", "Bypass",
                "-File", ".\scripts\run_human_talk_stage_prepare.ps1",
                "-Stage", $Stage,
                "-RawRoot", $RawRoot,
                "-WorkspaceRoot", $WorkspaceRoot,
                "-SegmentSec", "$SegmentSec",
                "-HopSec", "$HopSec",
                "-SampleRate", "$SampleRate",
                "-Seed", "$Seed",
                "-FilenameSeparator", $FilenameSeparator
            )

            if ($Clean) {
                $PrepareCmd += "-Clean"
            }

            powershell @PrepareCmd
        }
    }
    else {
        Write-Host "Skipping preparation."
    }

    # ---------------------------------------------------------------------
    # 2. Extract features
    # ---------------------------------------------------------------------
    if (-not $SkipFeatures) {
        Invoke-Step "Extract log-mel features for $Stage" {
            if ($Clean -and (Test-Path $CacheRoot)) {
                Write-Host "Clean enabled: removing old feature cache:"
                Write-Host "  $CacheRoot"
                Remove-Item $CacheRoot -Recurse -Force -ErrorAction SilentlyContinue
            }

            python .\scripts\extract_multilabel_features.py `
                --manifest "$DataRoot\metadata\multilabel_train_manifest.csv" `
                --labels_json "$DataRoot\metadata\labels.json" `
                --out_cache "$CacheRoot" `
                --sample_rate $SampleRate `
                --clip_sec $SegmentSec `
                --n_mels 64 `
                --n_fft 1024 `
                --win_ms 25 `
                --hop_ms 10 `
                --cmvn
        }
    }
    else {
        Write-Host "Skipping feature extraction."
    }

    $Manifest = "$CacheRoot\metadata\multilabel_features_manifest.csv"
    $FeaturesRoot = "$CacheRoot\features"
    $LabelsJson = "$DataRoot\metadata\labels.json"

    foreach ($RequiredPath in @($Manifest, $FeaturesRoot, $LabelsJson)) {
        if (!(Test-Path $RequiredPath)) {
            throw "Required file/folder missing before training: $RequiredPath"
        }
    }

    # ---------------------------------------------------------------------
    # 3. Train 3-exit
    # ---------------------------------------------------------------------
    if (-not $SkipTrain3) {
        Invoke-Step "Train 3-exit no-hint model" {
            python -m training.train_multilabel `
                --manifest $Manifest `
                --features_root $FeaturesRoot `
                --labels_json $LabelsJson `
                --runs_root $RunsRoot `
                --variant "${VariantPrefix}_3exit_nohint" `
                --tap_blocks "1,3" `
                --epochs $Epochs `
                --batch_size $BatchSize `
                --lr $LR `
                --device $Device
        }
    }
    else {
        Write-Host "Skipping 3-exit training."
    }

    # ---------------------------------------------------------------------
    # 4. Policy 3-exit
    # ---------------------------------------------------------------------
    if ((-not $SkipPolicy) -and (-not $SkipPolicy3)) {
        Invoke-Step "Run 3-exit greedy early-exit policy" {
            $Run3 = Get-LatestRunDir -RunsRoot $RunsRoot -Pattern "${VariantPrefix}_3exit_nohint_*"

            python .\scripts\multilabel_greedy_policy.py `
                --run_dir $Run3 `
                --name "${VariantPrefix}_3exit_nohint" `
                --device $Device `
                --split test `
                --threshold_mode $ThresholdMode `
                --min_exit 2 `
                --stable_k 2 `
                --sweep_min_exits "1,2" `
                --sweep_stable_k "1,2,3"
        }
    }
    else {
        Write-Host "Skipping greedy policy evaluation."
    }

    # ---------------------------------------------------------------------
    # 5. Train 5-exit
    # ---------------------------------------------------------------------
    if (-not $SkipTrain5) {
        Invoke-Step "Train 5-exit no-hint model" {
            python -m training.train_multilabel `
                --manifest $Manifest `
                --features_root $FeaturesRoot `
                --labels_json $LabelsJson `
                --runs_root $RunsRoot `
                --variant "${VariantPrefix}_5exit_nohint" `
                --tap_blocks "1,2,3,4" `
                --epochs $Epochs `
                --batch_size $BatchSize `
                --lr $LR `
                --device $Device
        }
    }
    else {
        Write-Host "Skipping 5-exit training."
    }

    # ---------------------------------------------------------------------
    # 6. Policy 5-exit
    # ---------------------------------------------------------------------
    if ((-not $SkipPolicy) -and (-not $SkipPolicy5)) {
        Invoke-Step "Run 5-exit greedy early-exit policy" {
            $Run5 = Get-LatestRunDir -RunsRoot $RunsRoot -Pattern "${VariantPrefix}_5exit_nohint_*"

            python .\scripts\multilabel_greedy_policy.py `
                --run_dir $Run5 `
                --name "${VariantPrefix}_5exit_nohint" `
                --device $Device `
                --split test `
                --threshold_mode $ThresholdMode `
                --min_exit 3 `
                --stable_k 2 `
                --sweep_min_exits "1,2,3" `
                --sweep_stable_k "1,2,3"
        }
    }

    # ---------------------------------------------------------------------
    # 7. Package outputs
    # ---------------------------------------------------------------------
    if ($ZipResults) {
        Invoke-Step "Create compact results ZIP" {
            New-ResultsPackage `
                -StageRoot $StageRoot `
                -WorkspaceRoot $WorkspaceRoot `
                -Stage $Stage `
                -TranscriptPath $TranscriptPath `
                -VariantPrefix $VariantPrefix `
                -ScriptPath $ScriptPath `
                -IncludeAllRuns:$IncludeAllRuns | Out-Null
        }
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Finished human-talk clean-stage experiment runner"
    Write-Host "============================================================"
    Write-Host "Transcript:"
    Write-Host "  $TranscriptPath"
}
finally {
    Stop-Transcript | Out-Null
}
