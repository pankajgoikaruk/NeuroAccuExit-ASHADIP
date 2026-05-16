# scripts/run_human_talk_stage_prepare.ps1
#
# Human-talk staged preparation driver.
#
# Generated files stay under ONE workspace:
#
#   human_talk_workspace/
#   └─ stages/
#      ├─ clean2_balanced/
#      │  ├─ data/
#      │  └─ cache/
#      ├─ clean3_balanced/
#      └─ ...
#
# Raw data should be separate:
#
#   human_talk_dataset/
#   ├─ Les_Brown/
#   ├─ Simon_Sinek/
#   └─ ...
#
# Do NOT use event_dataset if it contains environmental classes.

param(
    [string]$Stage = "clean2_balanced",
    [string]$RawRoot = "human_talk_dataset",
    [string]$WorkspaceRoot = "human_talk_workspace",
    [double]$SegmentSec = 1.0,
    [double]$HopSec = 0.5,
    [int]$SampleRate = 16000,
    [int]$Seed = 42,
    [string]$FilenameSeparator = "__",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$CleanClasses = "Les_Brown,Mel_Robbins,Oprah_Winfrey,Rabin_Sharma,Simon_Sinek"

if ($Stage -eq "clean2_balanced") {
    $Classes = "Les_Brown,Simon_Sinek"
}
elseif ($Stage -eq "clean3_balanced") {
    $Classes = "Les_Brown,Simon_Sinek,Rabin_Sharma"
}
elseif ($Stage -eq "clean4_balanced") {
    $Classes = "Les_Brown,Simon_Sinek,Rabin_Sharma,Oprah_Winfrey"
}
elseif ($Stage -eq "clean5_balanced") {
    $Classes = $CleanClasses
}
else {
    throw "Unknown Stage: $Stage. Use clean2_balanced, clean3_balanced, clean4_balanced, or clean5_balanced."
}

$StageRoot = Join-Path $WorkspaceRoot ("stages\" + $Stage)
$DataRoot = Join-Path $StageRoot "data"
$CacheRoot = Join-Path $StageRoot "cache"
$MetadataRoot = Join-Path $DataRoot "metadata"

Write-Host ""
Write-Host "========================================"
Write-Host " Human-talk stage preparation"
Write-Host "========================================"
Write-Host "Stage:             $Stage"
Write-Host "RawRoot:           $RawRoot"
Write-Host "WorkspaceRoot:     $WorkspaceRoot"
Write-Host "DataRoot:          $DataRoot"
Write-Host "CacheRoot:         $CacheRoot"
Write-Host "Classes:           $Classes"
Write-Host "FilenameSeparator: $FilenameSeparator"
Write-Host ""

if (!(Test-Path $RawRoot)) {
    throw "RawRoot does not exist: $RawRoot"
}

foreach ($ClassName in $Classes.Split(",")) {
    $ClassPath = Join-Path $RawRoot $ClassName
    if (!(Test-Path $ClassPath)) {
        $Available = Get-ChildItem $RawRoot -Directory | Select-Object -ExpandProperty Name
        throw @"
Required human-talk class folder not found: $ClassPath

Available folders under ${RawRoot}:
$($Available -join ", ")

You are probably pointing to the old environmental dataset.
Use:
  -RawRoot human_talk_dataset
"@
    }
}

python .\scripts\audit_human_talk_dataset.py `
    --raw_root $RawRoot `
    --out_dir $MetadataRoot `
    --classes $Classes `
    --clean_classes $CleanClasses `
    --filename_separator $FilenameSeparator

$PrepareArgs = @(
    ".\scripts\prepare_human_talk_segments.py",
    "--raw_root", $RawRoot,
    "--out_root", $DataRoot,
    "--classes", $Classes,
    "--segment_sec", "$SegmentSec",
    "--hop_sec", "$HopSec",
    "--sample_rate", "$SampleRate",
    "--seed", "$Seed",
    "--filename_separator", "$FilenameSeparator",
    "--balance_to_min"
)

if ($Clean) {
    $PrepareArgs += "--clean"
}

python @PrepareArgs

Write-Host ""
Write-Host "========================================"
Write-Host " Next command: extract log-mel features"
Write-Host "========================================"
Write-Host "python .\scripts\extract_multilabel_features.py ``"
Write-Host "  --manifest `"$DataRoot\metadata\multilabel_train_manifest.csv`" ``"
Write-Host "  --labels_json `"$DataRoot\metadata\labels.json`" ``"
Write-Host "  --out_cache `"$CacheRoot`" ``"
Write-Host "  --sample_rate $SampleRate ``"
Write-Host "  --clip_sec $SegmentSec ``"
Write-Host "  --n_mels 64 ``"
Write-Host "  --n_fft 1024 ``"
Write-Host "  --win_ms 25 ``"
Write-Host "  --hop_ms 10 ``"
Write-Host "  --cmvn"
