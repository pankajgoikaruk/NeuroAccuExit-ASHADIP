# Appendix — agentic_data_preprocessing_v0.5_tata_2

This appendix contains reproducibility notes and commands for the current **`agentic_data_preprocessing_v0.5_tata_2`** branch.

```text
Branch: agentic_data_preprocessing_v0.5_tata_2
Agenda: TinyAudioTriageAgent weak clip-level multi-label preprocessing for human-talk audio
Dataset stage: TATA reviewed 5-sec clip manifest -> weak 1-sec segment manifest
Task: multi-label detection of target speaker identity, non-target speech, and event/background audio
Model: TinyAudioCNN + ExitNet, 3-exit baseline
Labels: 12 labels = 6 target speakers + other_speaker_present + 5 event/background labels
Current status: first fixed-threshold TATA 3-exit baseline completed; threshold tuning not yet applied
```

## A1. Branch setup

```powershell
git fetch origin

git switch agentic_data_preprocessing_v0.4
git pull origin agentic_data_preprocessing_v0.4

git switch -c agentic_data_preprocessing_v0.5_tata_2
git push -u origin agentic_data_preprocessing_v0.5_tata_2
```

## A2. TATA folder and label design

Recommended seed root:

```text
human_talk_tata_seed_dataset/
├─ target_speaker/
│  ├─ Brene_Brown/
│  ├─ Eckhart_Tolle/
│  ├─ Eric_Thomas/
│  ├─ Gary_Vee/
│  ├─ Jay_Shetty/
│  └─ Nick_Vujicic/
├─ other_speaker/
└─ events/
   ├─ music/
   ├─ applause/
   ├─ laughter/
   ├─ crowd_cheer/
   └─ silence/
```

Active labels:

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
applause_present
laughter_present
crowd_cheer_present
silence_present
```

## A3. Rename audio before final manifest editing

```powershell
python scriptsename_wavs_by_class.py `
  --root human_talk_tata_seed_dataset	arget_speaker `
  --manifest human_talk_workspace	ata_2\metadataename_target_speaker_manifest.csv `
  --separator "__" `
  --preserve_case `
  --apply

python scriptsename_wavs_by_class.py `
  --root human_talk_tata_seed_dataset\other_speaker `
  --manifest human_talk_workspace	ata_2\metadataename_other_speaker_manifest.csv `
  --separator "__" `
  --preserve_case `
  --apply

python scriptsename_wavs_by_class.py `
  --root human_talk_tata_seed_dataset\events `
  --manifest human_talk_workspace	ata_2\metadataename_events_manifest.csv `
  --separator "__" `
  --preserve_case `
  --apply
```

## A4. Build 5-sec clip-level manifest

```powershell
python -m agentic_preprocessing.run_tata_clip_manifest_builder `
  --seed_root human_talk_tata_seed_dataset `
  --out_dir human_talk_workspace	ata_2\metadata
```

Manual editing is done on the clip-level manifest. The final training-ready file used in this run was:

```text
human_talk_workspace	ata_2\metadata	ata_clip_level_manifest_training_ready.csv
```

## A5. Build weak 1-sec segment manifest

```powershell
python -m agentic_preprocessing.run_tata_segment_manifest_builder `
  --clip_manifest human_talk_workspace	ata_2\metadata	ata_clip_level_manifest_training_ready.csv `
  --out_dir human_talk_workspace	ata_2\segment_cache `
  --sample_rate 16000 `
  --segment_sec 1.0 `
  --hop_sec 1.0 `
  --include_tail
```

Result:

| Item | Value |
| --- | --- |
| Reviewed clip-level training-ready rows | 2074 |
| Weak 1-sec segments created | 12469 |
| Segment build errors | 0 |
| Parent clips represented | 2074 |
| Parents split across train/val/test | 0 |
| Mean segments per parent clip | 6.01 |
| Min / max segments per parent clip | 2 / 109 |
| Mean active labels per segment | 1.6327 |
| Max active labels in a segment | 5 |

## A6. Extract log-mel features

```powershell
python scripts\extract_multilabel_features.py `
  --manifest human_talk_workspace	ata_2\segment_cache\metadata	ata_segment_manifest.csv `
  --labels_json human_talk_workspace	ata_2\segment_cache\metadata	ata_labels.json `
  --out_cache human_talk_workspace	ata_2eature_cache `
  --sample_rate 16000 `
  --clip_sec 1.0 `
  --n_mels 64 `
  --n_fft 1024 `
  --win_ms 25 `
  --hop_ms 10 `
  --cmvn
```

## A7. Run 3-exit TATA weakclip training with shareable ZIP output

```powershell
powershell -ExecutionPolicy Bypass -File scriptsun_tata_weakclip_experiment.ps1 `
  -Manifest "human_talk_workspace	ata_2eature_cache\metadata\multilabel_features_manifest.csv" `
  -FeaturesRoot "human_talk_workspace	ata_2eature_cacheeatures" `
  -LabelsJson "human_talk_workspace	ata_2\segment_cache\metadata	ata_labels.json" `
  -Variant "tata_2_3exit_weakclip" `
  -TapBlocks "1,3" `
  -Epochs 40 `
  -BatchSize 64 `
  -LR 0.001 `
  -Threshold 0.5 `
  -Device cpu
```

This script copies selected outputs into a ZIP for sharing and does not move or delete original run files.

## A8. Completed experiment settings

| Setting | Value |
| --- | --- |
| Branch | `agentic_data_preprocessing_v0.5_tata_2` |
| Run variant | `tata_2_3exit_weakclip` |
| Run directory | `human_talk_workspace\tata_2\runs\tata_2_3exit_weakclip_20260530_121030` |
| Task | `multi_label_audio` / TinyAudioTriageAgent |
| Model | TinyAudioCNN + ExitNet |
| Exits | 3 |
| Tap blocks | `1,3` |
| Labels | 12 |
| Loss / activation | BCEWithLogitsLoss + sigmoid |
| Threshold | 0.5 |
| Loss weights | `0.3, 0.3, 1.0` |
| Exit hint | `disabled` |
| Epochs | 40 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Device | `cpu` |
| Seed | 42 |
| Use positive class weighting | False |
| Runtime | 811.02 sec (~13.52 min) |

## A9. Completed experiment results

### A9.1 Test metrics by exit

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact match | Hamming loss | Avg predicted labels | Avg true labels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.1730 | 0.2890 | 0.2036 | 0.0673 | 0.1219 | 0.4707 | 1.5869 |
| 2 | 0.5468 | 0.6192 | 0.5304 | 0.2968 | 0.0838 | 1.0551 | 1.5869 |
| 3 | 0.7774 | 0.7656 | 0.7503 | 0.4895 | 0.0616 | 1.5650 | 1.5869 |

### A9.2 Final-exit per-label metrics

| Label | Precision | Recall | F1 | Support | Predicted positive |
| --- | --- | --- | --- | --- | --- |
| `Brene_Brown` | 0.7751 | 0.8733 | 0.8213 | 150 | 169 |
| `Eckhart_Tolle` | 0.9254 | 0.9185 | 0.9219 | 135 | 134 |
| `Eric_Thomas` | 0.8854 | 0.6296 | 0.7359 | 135 | 96 |
| `Gary_Vee` | 0.9804 | 0.7895 | 0.8746 | 190 | 153 |
| `Jay_Shetty` | 0.8622 | 0.8423 | 0.8521 | 260 | 254 |
| `Nick_Vujicic` | 0.8582 | 0.7667 | 0.8099 | 150 | 134 |
| `other_speaker_present` | 0.5654 | 0.6849 | 0.6195 | 511 | 619 |
| `music_present` | 0.9768 | 0.7342 | 0.8383 | 632 | 475 |
| `applause_present` | 0.9072 | 0.8151 | 0.8587 | 384 | 345 |
| `laughter_present` | 0.5609 | 0.7202 | 0.6306 | 243 | 312 |
| `crowd_cheer_present` | 0.6036 | 0.7976 | 0.6872 | 252 | 333 |
| `silence_present` | 0.8667 | 0.5571 | 0.6783 | 70 | 45 |

## A10. Commands for next threshold-tuning stage

Run this next, but it has not yet been incorporated into these results:

```powershell
python scripts	une_multilabel_thresholds.py `
  --run_dir "human_talk_workspace	ata_2uns	ata_2_3exit_weakclip_20260530_121030" `
  --device cpu

python scripts\multilabel_greedy_policy.py `
  --run_dir "human_talk_workspace	ata_2uns	ata_2_3exit_weakclip_20260530_121030" `
  --threshold_mode tuned_per_exit `
  --device cpu
```


## Figures

Generated figures for this branch are stored under `figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/`:

![Validation progression](figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_3exit_validation_progression.png)

![Training loss](figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_3exit_training_loss.png)

![Test metrics by exit](figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_test_metrics_by_exit.png)

![Per-label F1](figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_per_label_f1.png)

![Segment label distribution](figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_segment_label_distribution.png)

![Split distribution](figures/human_talk/agentic_data_preprocessing_v0.5_tata_2/tata_split_distribution.png)



## Paper-safe conclusion at this stage

The first TinyAudioTriageAgent experiment on `agentic_data_preprocessing_v0.5_tata_2` demonstrates that the NeuroAccuExit architecture can learn a 12-label multi-label audio triage task using BCE/sigmoid supervision and weak clip-level segment labels. The final exit achieved a fixed-threshold test Macro-F1 of **0.7774**, Micro-F1 of **0.7656**, Samples-F1 of **0.7503**, and Hamming loss of **0.0616**. This is a promising first baseline, but it is not yet an early-exit-ready TATA policy. The next required step is per-label threshold tuning, followed by multi-label greedy-policy testing.

