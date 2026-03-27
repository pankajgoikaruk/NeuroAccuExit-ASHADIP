param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Device = "cpu",
    [double]$SegmentSec = 1.0,
    [double]$HopSec = 0.5,
    [int]$NMels = 64,
    [double]$TimeConf = 0.95,
    [int]$TimeStableK = 2,
    [int]$TimeMinWindows = 2,
    [int]$EvalFixedKWindows = 3,
    [double]$V051DefaultLambdaDepth = 0.08,
    [double]$V051FixedLambdaDepth = 0.02,
    [int]$V051EaMinExit = 2,
    [int]$TunedTimeStableK = 3,
    [double]$TunedTimeMargin = 0.08,
    [double]$TunedTimeConf = 0.95
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "`n==== $Message ====" -ForegroundColor Cyan
}

function Ensure-RepoRoot {
    param([string]$Path)
    if (-not (Test-Path (Join-Path $Path ".git"))) {
        throw "RepoRoot '$Path' does not look like a git repository root. Run this from your main repo folder or pass -RepoRoot."
    }
}

function New-DirectoryIfMissing {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Get-WorktreePath {
    param([string]$BaseRepoRoot, [string]$FolderName)
    $parent = Split-Path -Parent $BaseRepoRoot
    return (Join-Path $parent $FolderName)
}

function Ensure-Worktree {
    param(
        [string]$BaseRepoRoot,
        [string]$Tag,
        [string]$FolderName
    )

    $worktreePath = Get-WorktreePath -BaseRepoRoot $BaseRepoRoot -FolderName $FolderName
    if (Test-Path $worktreePath) {
        Write-Host "Worktree already exists: $worktreePath" -ForegroundColor Yellow
        return $worktreePath
    }

    Write-Step "Creating worktree for $Tag at $worktreePath"
    & git -C $BaseRepoRoot worktree add --detach $worktreePath $Tag
    return $worktreePath
}

function Invoke-NativeLogged {
    param(
        [string]$WorkingDirectory,
        [string[]]$Command,
        [string]$LogPath
    )

    New-DirectoryIfMissing -Path (Split-Path -Parent $LogPath)
    Write-Host ("Running: " + ($Command -join ' ')) -ForegroundColor DarkGray

    Push-Location $WorkingDirectory
    try {
        $output = & $Command[0] $Command[1..($Command.Length - 1)] 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    $output | Tee-Object -FilePath $LogPath

    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See log: $LogPath"
    }

    return @($output | ForEach-Object { $_.ToString() })
}

function Invoke-RunFullWithVariants {
    param(
        [string]$RepoPath,
        [string[]]$VariantCandidates,
        [string]$ScenarioLabel,
        [hashtable]$NamedArgs
    )

    $logDir = Join-Path $RepoPath "comparison_logs"
    New-DirectoryIfMissing -Path $logDir

    foreach ($variant in $VariantCandidates) {
        $safeLabel = ($ScenarioLabel -replace '[^A-Za-z0-9_-]', '_')
        $safeVariant = ($variant -replace '[^A-Za-z0-9_.-]', '_')
        $logPath = Join-Path $logDir ("{0}_{1}.log" -f $safeLabel, $safeVariant)

        $cmd = @(
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $RepoPath "scripts\run_full.ps1"),
            "-Variant", $variant
        )

        foreach ($k in $NamedArgs.Keys) {
            $v = $NamedArgs[$k]
            if ($v -is [System.Management.Automation.SwitchParameter]) {
                if ($v.IsPresent) { $cmd += "-$k" }
            }
            elseif ($v -is [bool]) {
                if ($v) { $cmd += "-$k" }
            }
            elseif ($null -ne $v -and $v -ne "") {
                $cmd += "-$k"
                $cmd += [string]$v
            }
        }

        try {
            Write-Step "Running $ScenarioLabel with -Variant '$variant'"
            $lines = Invoke-NativeLogged -WorkingDirectory $RepoPath -Command $cmd -LogPath $logPath

            $runDirLine = $lines | Select-String -Pattern '^Using run:\s*(.+)$' | Select-Object -Last 1
            if (-not $runDirLine) {
                $runDirLine = $lines | Select-String -Pattern '^RunDir\s*=\s*(.+)$' | Select-Object -Last 1
            }
            $cacheDirLine = $lines | Select-String -Pattern '^Cache used:\s*(.+)$' | Select-Object -Last 1
            if (-not $cacheDirLine) {
                $cacheDirLine = $lines | Select-String -Pattern '^CacheDir\s*=\s*(.+)$' | Select-Object -Last 1
            }

            $runDir = $null
            if ($runDirLine) {
                $runDir = $runDirLine.Matches[0].Groups[1].Value.Trim()
            }
            $cacheDir = $null
            if ($cacheDirLine) {
                $cacheDir = $cacheDirLine.Matches[0].Groups[1].Value.Trim()
            }

            if (-not $runDir) {
                throw "Could not parse run directory from log: $logPath"
            }
            if (-not $cacheDir) {
                throw "Could not parse cache directory from log: $logPath"
            }

            return [pscustomobject]@{
                VariantUsed = $variant
                RepoPath = $RepoPath
                RunDir = $runDir
                CacheDir = $cacheDir
                LogPath = $logPath
                ScenarioLabel = $ScenarioLabel
            }
        }
        catch {
            Write-Warning "Variant '$variant' failed for $ScenarioLabel. $_"
        }
    }

    throw "All variant candidates failed for scenario '$ScenarioLabel'."
}

function Invoke-PythonLogged {
    param(
        [string]$RepoPath,
        [string[]]$Args,
        [string]$LogName
    )

    $logDir = Join-Path $RepoPath "comparison_logs"
    New-DirectoryIfMissing -Path $logDir
    $logPath = Join-Path $logDir $LogName
    $cmd = @("python") + $Args
    Invoke-NativeLogged -WorkingDirectory $RepoPath -Command $cmd -LogPath $logPath | Out-Null
    return $logPath
}

function Copy-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path $Source) {
        Copy-Item -Path $Source -Destination $Destination -Force
    }
}

function Snapshot-RunArtifacts {
    param(
        [string]$RepoPath,
        [string]$RunDir,
        [string]$SnapshotName,
        [string]$SourceLogPath
    )

    $runPath = if ([System.IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $RepoPath $RunDir }
    if (-not (Test-Path $runPath)) {
        throw "Run directory not found: $runPath"
    }

    $snapRoot = Join-Path $runPath "compare_snapshots"
    $snapDir = Join-Path $snapRoot $SnapshotName
    New-DirectoryIfMissing -Path $snapDir

    $filesToCopy = @(
        "policy_results.json",
        "clip_policy_results_full.json",
        "clip_policy_results_time.json",
        "summary.json",
        "analysis_run.json",
        "profiling.json",
        "ea_thresholds.json",
        "ea_sweep_results.json",
        "temperature.json"
    )

    foreach ($file in $filesToCopy) {
        $src = Join-Path $runPath $file
        Copy-IfExists -Source $src -Destination (Join-Path $snapDir $file)
    }

    if ($SourceLogPath -and (Test-Path $SourceLogPath)) {
        Copy-Item -Path $SourceLogPath -Destination (Join-Path $snapDir (Split-Path $SourceLogPath -Leaf)) -Force
    }

    return $snapDir
}

function Add-ManifestRow {
    param(
        [System.Collections.Generic.List[object]]$Manifest,
        [string]$Label,
        [string]$Tag,
        [string]$RepoPath,
        [string]$VariantUsed,
        [string]$RunDir,
        [string]$CacheDir,
        [string]$SnapshotDir,
        [string]$LogPath,
        [string]$Notes
    )

    $Manifest.Add([pscustomobject]@{
        label = $Label
        tag = $Tag
        repo_path = $RepoPath
        variant_used = $VariantUsed
        run_dir = $RunDir
        cache_dir = $CacheDir
        snapshot_dir = $SnapshotDir
        log_path = $LogPath
        notes = $Notes
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("s")
    }) | Out-Null
}

Ensure-RepoRoot -Path $RepoRoot

$manifest = New-Object 'System.Collections.Generic.List[object]'
$rootLogDir = Join-Path $RepoRoot "comparison_logs"
New-DirectoryIfMissing -Path $rootLogDir

$commonArgs = [ordered]@{
    Policy = "ea"
    Device = $Device
    SegmentSec = $SegmentSec
    HopSec = $HopSec
    NMels = $NMels
    RunClipPolicy = $true
    TimeConf = $TimeConf
    TimeStableK = $TimeStableK
    TimeMinWindows = $TimeMinWindows
    EvalFixedKWindows = $EvalFixedKWindows
}

# ---------- v0.4.1 ----------
$repo041 = Ensure-Worktree -BaseRepoRoot $RepoRoot -Tag "v0.4.1" -FolderName "NeuroAccuExit_v0_4_1"
$res041 = Invoke-RunFullWithVariants -RepoPath $repo041 -VariantCandidates @("v0.4.1", "v0.4") -ScenarioLabel "v0.4.1" -NamedArgs $commonArgs
$snap041 = Snapshot-RunArtifacts -RepoPath $repo041 -RunDir $res041.RunDir -SnapshotName "v0_4_1" -SourceLogPath $res041.LogPath
Add-ManifestRow -Manifest $manifest -Label "v0.4.1" -Tag "v0.4.1" -RepoPath $repo041 -VariantUsed $res041.VariantUsed -RunDir $res041.RunDir -CacheDir $res041.CacheDir -SnapshotDir $snap041 -LogPath $res041.LogPath -Notes "Baseline tagged run"

# ---------- v0.4.2 ----------
$repo042 = Ensure-Worktree -BaseRepoRoot $RepoRoot -Tag "v0.4.2" -FolderName "NeuroAccuExit_v0_4_2"
$res042 = Invoke-RunFullWithVariants -RepoPath $repo042 -VariantCandidates @("v0.4.2", "v0.4") -ScenarioLabel "v0.4.2" -NamedArgs $commonArgs
$snap042 = Snapshot-RunArtifacts -RepoPath $repo042 -RunDir $res042.RunDir -SnapshotName "v0_4_2" -SourceLogPath $res042.LogPath
Add-ManifestRow -Manifest $manifest -Label "v0.4.2" -Tag "v0.4.2" -RepoPath $repo042 -VariantUsed $res042.VariantUsed -RunDir $res042.RunDir -CacheDir $res042.CacheDir -SnapshotDir $snap042 -LogPath $res042.LogPath -Notes "Baseline tagged run"

# ---------- v0.5.1 default ----------
$repo051 = Ensure-Worktree -BaseRepoRoot $RepoRoot -Tag "v0.5.1" -FolderName "NeuroAccuExit_v0_5_1"
$args051Default = [ordered]@{
    Policy = "ea"
    Device = $Device
    SegmentSec = $SegmentSec
    HopSec = $HopSec
    LambdaDepth = $V051DefaultLambdaDepth
    TapBlocks = "1,2,3,4"
    NMels = $NMels
    RunClipPolicy = $true
    TimeConf = $TimeConf
    TimeStableK = $TimeStableK
    TimeMinWindows = $TimeMinWindows
    EvalFixedKWindows = $EvalFixedKWindows
}
$res051Default = Invoke-RunFullWithVariants -RepoPath $repo051 -VariantCandidates @("v0.5.1", "v0.5") -ScenarioLabel "v0.5.1_default" -NamedArgs $args051Default
$snap051Default = Snapshot-RunArtifacts -RepoPath $repo051 -RunDir $res051Default.RunDir -SnapshotName "v0_5_1_default" -SourceLogPath $res051Default.LogPath
Add-ManifestRow -Manifest $manifest -Label "v0.5.1 default" -Tag "v0.5.1" -RepoPath $repo051 -VariantUsed $res051Default.VariantUsed -RunDir $res051Default.RunDir -CacheDir $res051Default.CacheDir -SnapshotDir $snap051Default -LogPath $res051Default.LogPath -Notes "Default EA sweep from run_full.ps1"

# ---------- v0.5.1 fixed EA ----------
Write-Step "v0.5.1 fixed EA recalibration"
$runDir051 = $res051Default.RunDir
$cacheDir051 = $res051Default.CacheDir
$segmentsCsv051 = Join-Path $repo051 (Join-Path $cacheDir051 "segments.csv")
$featuresRoot051 = Join-Path $repo051 (Join-Path $cacheDir051 "features")

$logEA = Invoke-PythonLogged -RepoPath $repo051 -LogName "v0_5_1_fixed_ea_thresholds.log" -Args @(
    "-m", "training.ea_thresholds_offline",
    "--run_dir", $runDir051,
    "--segments_csv", $segmentsCsv051,
    "--features_root", $featuresRoot051,
    "--tap_blocks", "1,2,3,4",
    "--n_mels", "$NMels",
    "--ea_min_exit", "$V051EaMinExit",
    "--lambda_depth", "$V051FixedLambdaDepth"
)

$logPolicy = Invoke-PythonLogged -RepoPath $repo051 -LogName "v0_5_1_fixed_ea_policy_test.log" -Args @(
    "-m", "scripts.policy_test",
    "--policy", "ea",
    "--run_dir", $runDir051,
    "--segments_csv", $segmentsCsv051,
    "--features_root", $featuresRoot051,
    "--tap_blocks", "1,2,3,4",
    "--n_mels", "$NMels"
)

$logClipFull = Invoke-PythonLogged -RepoPath $repo051 -LogName "v0_5_1_fixed_ea_clip_full.log" -Args @(
    "-m", "scripts.clip_policy_test",
    "--run_dir", $runDir051,
    "--segments_csv", $segmentsCsv051,
    "--features_root", $featuresRoot051,
    "--tap_blocks", "1,2,3,4",
    "--n_mels", "$NMels",
    "--disable_time_exit",
    "--eval_fixed_k_windows", "$EvalFixedKWindows"
)

$snap051Fixed = Snapshot-RunArtifacts -RepoPath $repo051 -RunDir $runDir051 -SnapshotName "v0_5_1_fixed_ea" -SourceLogPath $logClipFull
Copy-IfExists -Source $logEA -Destination (Join-Path $snap051Fixed (Split-Path $logEA -Leaf))
Copy-IfExists -Source $logPolicy -Destination (Join-Path $snap051Fixed (Split-Path $logPolicy -Leaf))
Add-ManifestRow -Manifest $manifest -Label "v0.5.1 fixed EA" -Tag "v0.5.1" -RepoPath $repo051 -VariantUsed $res051Default.VariantUsed -RunDir $runDir051 -CacheDir $cacheDir051 -SnapshotDir $snap051Fixed -LogPath $logClipFull -Notes "Re-swept EA with ea_min_exit=2 and lambda_depth=0.02; full baseline only"

# ---------- v0.5.1 fixed EA + tuned Depth×Time ----------
Write-Step "v0.5.1 fixed EA + tuned Depth×Time"
$logClipTuned = Invoke-PythonLogged -RepoPath $repo051 -LogName "v0_5_1_fixed_ea_tuned_depth_time.log" -Args @(
    "-m", "scripts.clip_policy_test",
    "--run_dir", $runDir051,
    "--segments_csv", $segmentsCsv051,
    "--features_root", $featuresRoot051,
    "--tap_blocks", "1,2,3,4",
    "--n_mels", "$NMels",
    "--time_conf", "$TunedTimeConf",
    "--time_stable_k", "$TunedTimeStableK",
    "--time_min_windows", "$TimeMinWindows",
    "--time_margin", "$TunedTimeMargin",
    "--eval_fixed_k_windows", "$EvalFixedKWindows",
    "--full_baseline_json", (Join-Path $runDir051 "clip_policy_results_full.json")
)

$runPath051 = Join-Path $repo051 $runDir051
$tunedSrc = Join-Path $runPath051 "clip_policy_results_time.json"
$tunedDst = Join-Path $runPath051 "clip_policy_results_time_tuned.json"
if (Test-Path $tunedSrc) {
    Copy-Item -Path $tunedSrc -Destination $tunedDst -Force
}

$snap051Tuned = Snapshot-RunArtifacts -RepoPath $repo051 -RunDir $runDir051 -SnapshotName "v0_5_1_fixed_ea_tuned_depth_time" -SourceLogPath $logClipTuned
Copy-IfExists -Source $tunedDst -Destination (Join-Path $snap051Tuned "clip_policy_results_time_tuned.json")
Add-ManifestRow -Manifest $manifest -Label "v0.5.1 fixed EA + tuned DepthxTime" -Tag "v0.5.1" -RepoPath $repo051 -VariantUsed $res051Default.VariantUsed -RunDir $runDir051 -CacheDir $cacheDir051 -SnapshotDir $snap051Tuned -LogPath $logClipTuned -Notes "Fixed EA plus tuned temporal stopping (stable_k=3, margin=0.08)"

# ---------- Write manifest ----------
$manifestPath = Join-Path $RepoRoot "comparison_run_manifest.csv"
$manifest | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8

Write-Step "Done"
Write-Host "Manifest written to: $manifestPath" -ForegroundColor Green
$manifest | Format-Table -AutoSize
