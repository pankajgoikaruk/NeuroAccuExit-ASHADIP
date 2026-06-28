# Commands — agentic_data_preprocessing_v0.9

This file records the reproducible PowerShell commands for the v0.9 labelwise aggregation and calibration experiments.

Run commands from the repository root:

```powershell
C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP
```

---

## 1. Define common paths

```powershell
$RunDir = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\main_models\runs\main_v08_human_corrected_balanced_3exit_20260610_084027"
$CorrectedHoldoutManifest = "human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
$HoldoutFeaturesRoot = "human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features"
$V09Root = "human_talk_workspace\tata_v0.9_labelwise_calibration"
```

---

## 2. Generate parent mean verification and segment probabilities

This reproduces the official mean-all baseline and writes the segment probability CSV used by later v0.9 experiments.

```powershell
$MeanOutDir = "$V09Root\verification\v08_parent_mean_fixed"

New-Item -ItemType Directory -Force -Path $MeanOutDir | Out-Null

python scripts\evaluate_tata_final_holdout_parent_level.py `
  --run_dir "$RunDir" `
  --holdout_manifest "$CorrectedHoldoutManifest" `
  --features_root "$HoldoutFeaturesRoot" `
  --out_dir "$MeanOutDir" `
  --threshold_mode fixed_0p5 `
  --aggregation mean `
  --device cpu `
  --batch_size 128
```

Expected headline:

```text
Macro-F1   = 0.7801
Micro-F1   = 0.9332
Samples-F1 = 0.9406
Exact      = 0.8397
Hamming    = 0.0194
```

---

## 3. Run repeated labelwise aggregation and threshold calibration

```powershell
$SegmentPredCsv = "$V09Root\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"
$OutDir = "$V09Root\repeated_v07_style_calibration"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.9\run_labelwise_aggregation_threshold_calibration.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --out-dir "$OutDir" `
  --fixed-threshold 0.5 `
  --seeds 20 `
  --cal-fraction 0.5 `
  --threshold-min 0.10 `
  --threshold-max 0.95 `
  --threshold-step 0.05 `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_"
```

Inspect:

```powershell
Import-Csv "$OutDir\repeated_eval_summary.csv" |
  Sort-Object {[double]$_.macro_f1_mean} -Descending |
  Format-Table method, macro_f1_mean, micro_f1_mean, samples_f1_mean, exact_match_mean, hamming_loss_mean -AutoSize
```

---

## 4. Evaluate the original frozen frequency map

```powershell
$SegmentPredCsv = "$V09Root\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"
$OutDir = "$V09Root\final_frozen_v06_labelwise_fixed_0p5"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.9\evaluate_frozen_labelwise_aggregation_v09.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --out-dir "$OutDir" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --threshold 0.5
```

Expected headline:

```text
Macro-F1   = 0.8512
Micro-F1   = 0.9372
Samples-F1 = 0.9482
Exact      = 0.8420
Hamming    = 0.0185
```

---

## 5. Run the mapping-bank evaluator

```powershell
$SegmentPredCsv = "$V09Root\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"
$MapsJson = "configs\v0.9\labelwise_aggregation_maps.json"
$OutDir = "$V09Root\mapping_bank_fixed_0p5_plus_gary_mean"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.9\evaluate_labelwise_mapping_bank_v09.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --maps-json "$MapsJson" `
  --out-dir "$OutDir" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --threshold 0.5 `
  --model-name "main_v08_human_corrected_balanced_3exit_20260610_084027"
```

Inspect:

```powershell
Import-Csv "$OutDir\v09_mapping_bank_summary.csv" |
  Sort-Object {[double]$_.macro_f1} -Descending |
  Format-Table method, macro_f1, micro_f1, samples_f1, exact_match, hamming_loss, avg_pred_labels -AutoSize
```

Expected best:

```text
v09_frozen_frequency_plus_gary_mean
Macro-F1   = 0.8518
Micro-F1   = 0.9374
Samples-F1 = 0.9464
Exact      = 0.8431
Hamming    = 0.0183
```

---

## 6. Add the final Gary-mean map if missing

```powershell
$MapsJson = "configs\v0.9\labelwise_aggregation_maps.json"

@'
import json
from pathlib import Path

maps_json = Path(r"configs\v0.9\labelwise_aggregation_maps.json")

with maps_json.open("r", encoding="utf-8") as f:
    data = json.load(f)

maps = data["maps"]

maps["v09_frozen_frequency_plus_gary_mean"] = {
    "description": "v0.9 frozen frequency map with Gary_Vee changed from top2mean to mean.",
    "aggregation": {
        "Brene_Brown": "mean",
        "Eckhart_Tolle": "top2mean",
        "Eric_Thomas": "mean",
        "Gary_Vee": "mean",
        "Jay_Shetty": "mean",
        "Nick_Vujicic": "mean",
        "other_speaker_present": "mean",
        "music_present": "mean",
        "audience_reaction_present": "top2mean",
        "silence_present": "top2mean"
    }
}

with maps_json.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Added/updated: v09_frozen_frequency_plus_gary_mean")
'@ | python -
```

---

## 7. TATA-LAWYER threshold-transfer diagnostic

This diagnostic is included for completeness. It should **not** be selected as the final v0.9 result.

```powershell
$SegmentPredCsv = "$V09Root\verification\v08_parent_mean_fixed\parent_eval_segment_probs_fixed_0p5_mean.csv"
$MapsJson = "configs\v0.9\labelwise_aggregation_maps.json"
$OutDir = "$V09Root\mapping_bank_with_tata_lawyer_thresholds"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.9\evaluate_labelwise_mapping_bank_v09.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --maps-json "$MapsJson" `
  --out-dir "$OutDir" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --threshold 0.5 `
  --model-name "main_v08_human_corrected_balanced_3exit_20260610_084027"
```

Expected diagnostic result for `tata_lawyer_optimal_map_fixed_thresholds`:

```text
Macro-F1   = 0.7284
Micro-F1   = 0.8610
Exact      = 0.6540
Hamming    = 0.0436
```

Decision:

```text
Reject old threshold transfer. Keep fixed threshold 0.5.
```
