# Commands — Agentic Data Preprocessing v0.9_4 LATS-v2

This file records the commands for creating the v0.9_4 branch, running LATS-v2, inspecting the outputs, and committing the documentation update.

## 1. Create v0.9_4 branch from v0.9_3

```powershell
git fetch origin

git switch agentic_data_preprocessing_v0.9_3

git pull origin agentic_data_preprocessing_v0.9_3

git switch -c agentic_data_preprocessing_v0.9_4

git push -u origin agentic_data_preprocessing_v0.9_4
```

Confirm:

```powershell
git branch --show-current
git status
```

Expected branch:

```text
agentic_data_preprocessing_v0.9_4
```

## 2. Run LATS-v2 metric-aware coordinate search

```powershell
$SegmentPredCsv = "human_talk_workspace\tata_v0.9_labelwise_calibration\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"

$StartConfigJson = "human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v09_search\lats_final_frozen_config.json"

$OutDir = "human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v2_metric_coordinate_search"

Remove-Item -Recurse -Force $OutDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.9\run_lats_v2_metric_aware_coordinate_search_v09.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --start-config-json "$StartConfigJson" `
  --out-dir "$OutDir" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --seeds 20 `
  --cal-fraction 0.5 `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.01 `
  --aggregation-methods "mean,max,top2mean,top3mean,top4mean,top5mean,median,p75,p90,noisy_or" `
  --objective-weights "macro_f1=0.40,micro_f1=0.20,samples_f1=0.20,exact_match=0.15,hamming_loss=-0.05,label_count_abs_error=-0.05" `
  --max-iterations 5 `
  --progress `
  --model-name "main_v08_human_corrected_balanced_3exit_20260610_084027"
```

## 3. Inspect final full-holdout result

```powershell
Import-Csv "$OutDir\lats_v2_final_full_holdout_eval.csv" |
  Format-Table method, macro_f1, micro_f1, samples_f1, exact_match, hamming_loss, avg_pred_labels, objective_score -AutoSize
```

Expected final result:

```text
Macro-F1   = 0.8673
Micro-F1   = 0.9458
Samples-F1 = 0.9517
Exact      = 0.8604
Hamming    = 0.0158
```

## 4. Compare against LATS-v1 start config

```powershell
Import-Csv "$OutDir\lats_v2_final_vs_init_comparison.csv" |
  Format-Table metric, initial_start_config, lats_v2_final_config, difference_final_minus_initial -AutoSize
```

## 5. Inspect label-wise changes

```powershell
Import-Csv "$OutDir\lats_v2_final_full_holdout_per_label.csv" |
  Format-Table label, aggregation, threshold, precision, recall, f1, support, predicted_positive, hamming_errors -AutoSize
```

## 6. Inspect stability

```powershell
Import-Csv "$OutDir\lats_v2_threshold_summary.csv" |
  Format-Table label, final_aggregation, final_threshold, selection_count, selection_fraction -AutoSize
```

## 7. Copy result files into organised docs table folders

The cleaned table hierarchy is:

```text
docs\tables\agentic_data_preprocessing_v0.9\
├── v0.9_baselines\
├── v0.9_lats_v1\
├── v0.9_lats_v2\
└── v0.9_comparisons\
```

Copy LATS-v2 raw result files into the LATS-v2 folder:

```powershell
$DocsTableDir = "docs\tables\agentic_data_preprocessing_v0.9\v0.9_lats_v2"
New-Item -ItemType Directory -Force -Path $DocsTableDir | Out-Null
Copy-Item -Force "$OutDir\lats_v2_*.csv" $DocsTableDir
Copy-Item -Force "$OutDir\lats_v2_*.json" $DocsTableDir
```

Derived comparison files should stay under:

```text
docs\tables\agentic_data_preprocessing_v0.9\v0.9_comparisons\
```

## 8. Commit v0.9_4 documentation update

```powershell
git status

git add docs

git commit -m "docs: document v0.9_4 LATS-v2 metric-aware results"

git push origin agentic_data_preprocessing_v0.9_4
```
