param(
  [string]$LatsRoot = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline\posweight_stability_lats_reoptimized"
)

$ErrorActionPreference = "Stop"

Get-ChildItem $LatsRoot -Recurse -Filter "lats_v2_coordinate_reoptimized_summary.csv" |
  ForEach-Object {
    $row = Import-Csv $_.FullName
    [PSCustomObject]@{
      Run = $_.Directory.Name
      MacroF1 = [double]$row.macro_f1
      MicroF1 = [double]$row.micro_f1
      SamplesF1 = [double]$row.samples_f1
      Exact = [double]$row.exact_match
      Hamming = [double]$row.hamming_loss
      AvgPredLabels = [double]$row.avg_pred_labels
    }
  } | Sort-Object Run | Format-Table -AutoSize
