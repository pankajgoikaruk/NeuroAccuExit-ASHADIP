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
