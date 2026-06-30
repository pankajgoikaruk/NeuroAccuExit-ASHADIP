# Commands — agentic_data_preprocessing_v0.9_3

Run commands from the repository root:

```powershell
C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP
```

---

## 1. Common paths

```powershell
$V09Root = "human_talk_workspace\tata_v0.9_labelwise_calibration"

$SegmentPredCsv = "$V09Root\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"

$LATSOutDir = "$V09Root\lats_v09_search"
```

---

## 2. Run final LATS-v0.9 search

```powershell
New-Item -ItemType Directory -Force -Path $LATSOutDir | Out-Null

python scripts\v0.9\run_lats_labelwise_aggregation_threshold_search_v09.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --out-dir "$LATSOutDir" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --seeds 20 `
  --cal-fraction 0.5 `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.01 `
  --aggregation-methods "mean,max,top2mean,top3mean" `
  --model-name "main_v08_human_corrected_balanced_3exit_20260610_084027"
```

Expected final full-holdout result:

```text
Method     = lats_final_frozen_config_v09
Macro-F1   = 0.8667
Micro-F1   = 0.9436
Samples-F1 = 0.9495
Exact      = 0.8524
Hamming    = 0.0165
```

---

## 3. Inspect final full-holdout result

```powershell
Import-Csv "$LATSOutDir\lats_final_full_holdout_eval.csv" |
  Format-Table method, macro_f1, micro_f1, samples_f1, exact_match, hamming_loss, avg_pred_labels -AutoSize
```

---

## 4. Inspect repeated split mean/std/min/max table

```powershell
@'
import pandas as pd
from pathlib import Path

out_dir = Path(r"human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v09_search")
df = pd.read_csv(out_dir / "lats_repeated_eval_summary.csv")

cols = ["macro_f1", "micro_f1", "samples_f1", "exact_match", "hamming_loss", "avg_pred_labels"]

print(df[cols].agg(["mean", "std", "min", "max"]).T)
'@ | python -
```

Expected table:

| Metric               |   Mean |    Std |    Min |    Max |
|:---------------------|-------:|-------:|-------:|-------:|
| Macro-F1             | 0.8309 | 0.0154 | 0.8082 | 0.8593 |
| Micro-F1             | 0.9293 | 0.0067 | 0.9193 | 0.9431 |
| Samples-F1           | 0.9369 | 0.0073 | 0.9273 | 0.9532 |
| Exact Match          | 0.8179 | 0.0182 | 0.7875 | 0.8499 |
| Hamming Loss ↓       | 0.0207 | 0.0021 | 0.0166 | 0.024  |
| Avg predicted labels | 1.4606 | 0.032  | 1.3972 | 1.5381 |

---

## 5. Inspect final selected label rules

```powershell
Import-Csv "$LATSOutDir\lats_threshold_summary.csv" |
  Format-Table label, final_aggregation, final_threshold, selection_count, selection_fraction -AutoSize
```

---

## 6. Inspect per-label final performance

```powershell
Import-Csv "$LATSOutDir\lats_final_full_holdout_per_label.csv" |
  Format-Table label, aggregation, threshold, precision, recall, f1, support, predicted_positive -AutoSize
```

---

## 7. Freeze/commit documentation

```powershell
git add README.md DOC_STRUCTURE.md `
  docs\v0.9 `
  docs\reports\v0.9 `
  docs\results\v0.9 `
  docs\tables\agentic_data_preprocessing_v0.9 `
  configs\v0.9 `
  scripts\v0.9 `
  human_talk_workspace\tata_v0.9_labelwise_calibration\lats_v09_search

git commit -m "freeze v0.9_3 LATS labelwise aggregation threshold search"
git push origin agentic_data_preprocessing_v0.9_3
```

---

## 8. Final reporting note

```text
Subbranch: agentic_data_preprocessing_v0.9_3
Final method: LATS-v0.9 final frozen config
Internal config: lats_final_frozen_config_v09
```
