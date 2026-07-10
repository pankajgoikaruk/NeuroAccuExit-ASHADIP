param([switch]$SkipEnvironmentCapture)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    throw "Run this script from the NeuroAccuExit-ASHADIP repository root."
}

$FreezeRoot = "docs\tables\active_budget_anytime_exit_v0.1\full_depth_baselines"
$PrimaryFrozen = "$FreezeRoot\primary_v010_no_hint_historical_lats_v2"
$SecondaryFrozen = "$FreezeRoot\secondary_direct_coordinate_reoptimized"
$SharedDir = "$FreezeRoot\shared_reproducibility_inputs"

$PrimarySource = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline\baseline_reproduction\no_hint_lats_v2_exact_frozen"
$SecondarySource = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline\lats_v2_coordinate_reoptimized\no_hint_global_consistency_reproduction"
$SegmentPredCsv = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline\evaluation\no_hint_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"
$LabelsJson = "configs\human_talk_10label_schema.json"
$HistoricalConfig = "docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json"
$PrimaryEvaluator = "scripts\v0.10\evaluate_frozen_lats_config_v010.py"
$SecondarySearchScript = "scripts\v0.10\run_v010_lats_v2_coordinate_reoptimize.py"

$required = @($PrimarySource,$SecondarySource,$SegmentPredCsv,$LabelsJson,$HistoricalConfig,$PrimaryEvaluator,$SecondarySearchScript)
foreach ($path in $required) {
    if (-not (Test-Path $path)) { throw "Required path not found: $path" }
}

New-Item -ItemType Directory -Force -Path $PrimaryFrozen,$SecondaryFrozen,$SharedDir | Out-Null
Copy-Item "$PrimarySource\*" $PrimaryFrozen -Recurse -Force
Copy-Item "$SecondarySource\*" $SecondaryFrozen -Recurse -Force
Copy-Item $SegmentPredCsv "$SharedDir\no_hint_exit3_segment_probabilities.csv" -Force
Copy-Item $LabelsJson "$SharedDir\human_talk_10label_schema.json" -Force
Copy-Item $HistoricalConfig "$SharedDir\historical_no_hint_lats_v2_config.json" -Force
Copy-Item $PrimaryEvaluator "$SharedDir\evaluate_frozen_lats_config_v010_snapshot.py" -Force
Copy-Item $SecondarySearchScript "$SharedDir\run_v010_lats_v2_coordinate_reoptimize_snapshot.py" -Force

$PrimarySummary = "$PrimaryFrozen\v010_frozen_lats_eval.csv"
$result = Import-Csv $PrimarySummary | Select-Object -First 1
$expected = [ordered]@{
    macro_f1 = 0.8623815322333925
    micro_f1 = 0.9531311539976368
    samples_f1 = 0.9588894381281925
    exact_match = 0.8765859284890427
    hamming_loss = 0.013725490196078431
    avg_pred_labels = 1.4590542099192618
    n_parent_clips = 867
}
foreach ($metric in $expected.Keys) {
    if ([math]::Abs(([double]$result.$metric) - ([double]$expected[$metric])) -gt 1e-10) {
        throw "Canonical metric mismatch: $metric"
    }
}

$rows = Import-Csv $SegmentPredCsv
$branch = (git branch --show-current | Out-String).Trim()
$commit = (git rev-parse HEAD | Out-String).Trim()

$manifest = [ordered]@{
    experiment_name = "v0.10 no-hint + frozen historical LATS-v2"
    role = "canonical_primary_full_depth_baseline"
    branch = $branch
    git_commit_when_documented = $commit
    task = "Human-talk multi-label speaker and acoustic-context classification"
    dataset = [ordered]@{
        segments = $rows.Count
        parent_clips = @($rows.parent_clip_id | Sort-Object -Unique).Count
        labels = 10
        holdout_manifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
        feature_root = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
    }
    model = [ordered]@{
        run = "main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845"
        architecture = "Three-exit ExitNet / TinyAudioCNN"
        tap_blocks = @(1,3)
        full_depth_exit = 3
        hint_pass = $false
        probability_prefix = "exit3_prob_"
    }
    training = [ordered]@{
        epochs = 40
        batch_size = 64
        learning_rate = 0.001
        threshold = 0.5
        device = "cpu"
    }
    holdout_evaluation = [ordered]@{
        batch_size = 128
        device = "cpu"
        threshold_mode = "fixed_0p5"
        preliminary_aggregation = "mean"
        parent_id_column = "parent_clip_id"
    }
    canonical_metrics = $expected
    baseline_policy = [ordered]@{
        use_for_standard_early_exit = $true
        use_for_budget_aware_early_exit = $true
        use_for_anytime_inference = $true
        secondary_result_is_baseline = $false
    }
}
$manifest | ConvertTo-Json -Depth 10 | Set-Content "$FreezeRoot\reproducibility_manifest.json" -Encoding UTF8

$primaryRecord = @'
# Primary Full-Depth Baseline Experiment Record

## Identity

- Method: **v0.10 no-hint + frozen historical LATS-v2**
- Role: canonical primary full-depth baseline
- Task: human-talk multi-label speaker and acoustic-context classification
- Segments: 4,335
- Parent clips: 867
- Labels: 10
- Full-depth output: Exit 3
- Hint passing: disabled

This is the only baseline used for standard Early-Exit, budget-aware Early-Exit,
anytime-inference, quality-versus-cost, latency, compute-saving, and exit-depth comparisons.

## Model and training settings

Source run: `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845`

| Setting | Value |
|---|---|
| Architecture | Three-exit ExitNet / TinyAudioCNN |
| Tap blocks | 1 and 3 |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Training threshold | 0.5 |
| Device | CPU |
| Output used | Exit 3 |

## Holdout evaluation settings

| Setting | Value |
|---|---|
| Evaluation batch size | 128 |
| Device | CPU |
| Initial threshold mode | `fixed_0p5` |
| Preliminary aggregation | `mean` |
| Probability columns | `exit3_prob_<label>` |
| Parent ID column | `parent_clip_id` |

The initial evaluation generated final-exit probabilities for 4,335 segments.
The frozen LATS-v2 rules aggregated these into 867 parent-level predictions.

## Frozen LATS-v2 rules

| Label | Aggregation | Threshold |
|---|---|---:|
| Brene Brown | p75 | 0.54 |
| Eckhart Tolle | top3mean | 0.50 |
| Eric Thomas | top4mean | 0.62 |
| Gary Vee | mean | 0.50 |
| Jay Shetty | p75 | 0.91 |
| Nick Vujicic | p75 | 0.34 |
| Other speaker present | noisy_or | 0.94 |
| Music present | mean | 0.37 |
| Audience reaction present | top3mean | 0.23 |
| Silence present | p75 | 0.42 |

Frozen replay performs no retraining, no threshold search, and no aggregation search.

## Canonical result

| Metric | Exact value | Paper value |
|---|---:|---:|
| Macro-F1 | 0.8623815322 | 0.8624 |
| Micro-F1 | 0.9531311540 | 0.9531 |
| Samples-F1 | 0.9588894381 | 0.9589 |
| Exact Match | 0.8765859285 | 0.8766 |
| Hamming Loss | 0.0137254902 | 0.0137 |
| Average predicted labels | 1.4590542099 | 1.4591 |
| Parent clips | 867 | 867 |

`1.4591` is average predicted labels, not average exit depth.

## Reporting limitation

Report this as a frozen corrected-holdout result, not as performance on an
independent external unseen-test set.
'@
Set-Content "$PrimaryFrozen\EXPERIMENT_RECORD.md" $primaryRecord -Encoding UTF8

$secondaryRecord = @'
# Secondary Direct Coordinate Re-optimisation Record

This is a secondary post-hoc inference-policy result and is not the Early-Exit baseline.

| Setting | Value |
|---|---|
| Input | Same frozen Exit 3 probabilities as the primary result |
| Threshold range | 0.10 to 0.95 |
| Threshold step | 0.01 |
| Maximum iterations | 20 |
| Objective | `global_consistency` |
| Aggregations | mean, max, top2mean, top3mean, p75, p90 |

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8599 |
| Micro-F1 | 0.9547 |
| Samples-F1 | 0.9620 |
| Exact Match | 0.8800 |
| Hamming Loss | 0.0131 |
| Average predicted labels | 1.4348 |
'@
Set-Content "$SecondaryFrozen\EXPERIMENT_RECORD.md" $secondaryRecord -Encoding UTF8

$primaryReplay = @'
$ErrorActionPreference = "Stop"
$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SharedDir = "$PackageRoot\shared_reproducibility_inputs"
$OutDir = "$PackageRoot\reproduced_outputs\primary_historical_lats_v2"
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
python "$SharedDir\evaluate_frozen_lats_config_v010_snapshot.py" `
  --segment-pred-csv "$SharedDir\no_hint_exit3_segment_probabilities.csv" `
  --labels-json "$SharedDir\human_talk_10label_schema.json" `
  --config-json "$SharedDir\historical_no_hint_lats_v2_config.json" `
  --out-dir "$OutDir" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --model-name "main_v010_no_hint_historical_lats_v2"
if ($LASTEXITCODE -ne 0) { throw "Primary replay failed." }
Import-Csv "$OutDir\v010_frozen_lats_eval.csv" |
  Select-Object macro_f1,micro_f1,samples_f1,exact_match,hamming_loss,avg_pred_labels,n_parent_clips |
  Format-List
'@
Set-Content "$PrimaryFrozen\REPRODUCE_PRIMARY.ps1" $primaryReplay -Encoding UTF8

$secondaryReplay = @'
$ErrorActionPreference = "Stop"
$PackageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SharedDir = "$PackageRoot\shared_reproducibility_inputs"
$OutDir = "$PackageRoot\reproduced_outputs\secondary_direct_coordinate_reoptimized"
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
python "$SharedDir\run_v010_lats_v2_coordinate_reoptimize_snapshot.py" `
  --segment-pred-csv "$SharedDir\no_hint_exit3_segment_probabilities.csv" `
  --labels-json "$SharedDir\human_talk_10label_schema.json" `
  --out-dir "$OutDir" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.01 `
  --aggregation-methods "mean,max,top2mean,top3mean,p75,p90" `
  --objective global_consistency `
  --max-iter 20 `
  --model-name "main_v010_no_hint_direct_coordinate_reoptimized"
if ($LASTEXITCODE -ne 0) { throw "Secondary replay failed." }
Import-Csv "$OutDir\lats_v2_coordinate_reoptimized_summary.csv" |
  Select-Object macro_f1,micro_f1,samples_f1,exact_match,hamming_loss,avg_pred_labels,parent_clips |
  Format-List
'@
Set-Content "$SecondaryFrozen\REPRODUCE_SECONDARY.ps1" $secondaryReplay -Encoding UTF8

$paperSummary = @'
# Paper-Ready Full-Depth Baseline Summary

The full-computation reference used the final exit of the three-exit no-hint
model followed by the frozen historical LATS-v2 parent-level inference policy.
Across 867 parent clips and 10 labels, it achieved Macro-F1 0.8624, Micro-F1
0.9531, Samples-F1 0.9589, Exact Match 0.8766, and Hamming Loss 0.0137.
This frozen result is the canonical quality reference for all subsequent
standard Early-Exit, budget-aware Early-Exit, and anytime-inference evaluations.

No model retraining, threshold search, or aggregation-method search occurs in
the deterministic replay.
'@
Set-Content "$FreezeRoot\PAPER_READY_BASELINE_SUMMARY.md" $paperSummary -Encoding UTF8

$readme = @'
# Frozen Full-Depth Baseline Package

Canonical baseline: **v0.10 no-hint + frozen historical LATS-v2**.

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8624 |
| Micro-F1 | 0.9531 |
| Samples-F1 | 0.9589 |
| Exact Match | 0.8766 |
| Hamming Loss | 0.0137 |

- `primary_v010_no_hint_historical_lats_v2/`: canonical result and replay.
- `secondary_direct_coordinate_reoptimized/`: secondary ablation.
- `shared_reproducibility_inputs/`: frozen inputs, configuration, and code snapshots.
- `artifact_hashes.csv`: SHA256 integrity manifest.
'@
Set-Content "$FreezeRoot\README.md" $readme -Encoding UTF8

if (-not $SkipEnvironmentCapture) {
    $pythonVersion = (python --version 2>&1 | Out-String).Trim()
    @"
Captured: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz")
Git branch: $branch
Git commit: $commit
Python: $pythonVersion
"@ | Set-Content "$FreezeRoot\environment_summary.txt" -Encoding UTF8
    python -m pip freeze | Set-Content "$FreezeRoot\environment_pip_freeze.txt" -Encoding UTF8
}

$rootResolved = (Resolve-Path $FreezeRoot).Path
Get-ChildItem $FreezeRoot -Recurse -File |
  Where-Object { $_.Name -ne "artifact_hashes.csv" } |
  ForEach-Object {
      [PSCustomObject]@{
          relative_path = $_.FullName.Substring($rootResolved.Length + 1)
          size_bytes = $_.Length
          sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
      }
  } |
  Sort-Object relative_path |
  Export-Csv "$FreezeRoot\artifact_hashes.csv" -NoTypeInformation -Encoding UTF8

Write-Host "`nReproducibility package completed." -ForegroundColor Green
Write-Host "Package: $FreezeRoot"
git status --short
