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
