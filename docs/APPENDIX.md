# Appendix — Agentic Data Preprocessing Current Branch

This appendix contains reproducibility details for the current `agentic_data_preprocessing` branch only.

Older clean-stage branch commands and results are intentionally removed.

---

## A1. Current branch status

| Item | Value |
|---|---|
| Branch | `agentic_data_preprocessing` |
| Current run | `raw5_agentic_cleaned_3exit_greedy_final_001` |
| Dataset stage | `raw5_agentic_cleaned` |
| Task | Five-speaker human-talk classification |
| Model | TinyAudioCNN + ExitNet |
| Exits | 3 |
| Tap blocks | `[1, 3]` |
| Classes | 5 |
| Segment length | 1.0 s |
| Hop | 0.5 s |
| Feature type | 64-mel log-mel |
| Policy | greedy |
| Clip policies | full-clip aggregation, Depth×Time |
| Exit hint | disabled |
| Training epochs | 40 |
| Final cleaned files | 3,108 |

---

## A2. Current Raw5 classes

| Class | Raw accepted | Final training files |
|---|---:|---:|
| Brene_Brown | 595 | 595 |
| Eckhart_Tolle | 660 | 660 |
| Eric_Thomas | 594 | 593 |
| Gary_Vee | 642 | 642 |
| Jay_Shetty | 618 | 618 |
| **Total** | **3,109** | **3,108** |

---

## A3. Agentic preprocessing commands

### A3.1 Raw5 audit

```powershell
python -m agentic_preprocessing.run_agentic_preprocessing `
  --raw_root human_talk_dataset `
  --out_dir human_talk_workspace\agent_reports `
  --classes "Brene_Brown,Eckhart_Tolle,Eric_Thomas,Gary_Vee,Jay_Shetty" `
  --expected_sample_rate 16000 `
  --expected_duration_sec 5.0
```

### A3.2 Manifest builder

```powershell
python -m agentic_preprocessing.run_manifest_builder `
  --audit_csv human_talk_workspace\agent_reports\dataset_audit_agent_report.csv `
  --triage_seed_root human_talk_triage_seed_dataset `
  --out_dir human_talk_workspace\agent_reports
```

### A3.3 Dataset builder

```powershell
python -m agentic_preprocessing.run_dataset_builder `
  --accepted_manifest human_talk_workspace\agent_reports\accepted_manifest.csv `
  --raw_root human_talk_dataset `
  --out_root human_talk_workspace\datasets\raw5_agentic_cleaned `
  --apply
```

### A3.4 Final cleaned dataset audit

```powershell
python -m agentic_preprocessing.run_agentic_preprocessing `
  --raw_root human_talk_workspace\datasets\raw5_agentic_cleaned `
  --out_dir human_talk_workspace\agent_reports\raw5_agentic_cleaned_final_audit `
  --classes "Brene_Brown,Eckhart_Tolle,Eric_Thomas,Gary_Vee,Jay_Shetty" `
  --expected_sample_rate 16000 `
  --expected_duration_sec 5.0
```

---

## A4. Manual exclusion

The following generated cleaned file was excluded from the training root:

```text
human_talk_workspace\datasets\raw5_agentic_cleaned\Eric_Thomas\Eric_Thomas__0175.wav
```

Reason:

```text
pure_music_no_target_audio
```

Recommended traceable excluded location:

```text
human_talk_workspace\datasets\raw5_agentic_cleaned_excluded\Eric_Thomas\Eric_Thomas__0175.wav
```

PowerShell check:

```powershell
Test-Path "human_talk_workspace\datasets\raw5_agentic_cleaned\Eric_Thomas\Eric_Thomas__0175.wav"
Test-Path "human_talk_workspace\datasets\raw5_agentic_cleaned_excluded\Eric_Thomas\Eric_Thomas__0175.wav"
```

Expected:

```text
False
True
```

---

## A5. First experiment command

```powershell
.\scripts\run_full.ps1 `
  -DataRoot "human_talk_workspace\datasets\raw5_agentic_cleaned" `
  -Variant "raw5_agentic_cleaned_3exit_greedy_final" `
  -Policy greedy `
  -Device cpu `
  -InputMode segment `
  -Labels "Brene_Brown,Eckhart_Tolle,Eric_Thomas,Gary_Vee,Jay_Shetty" `
  -SegmentSec 1.0 `
  -HopSec 0.5 `
  -SampleRate 16000 `
  -Bandpass "50,7600" `
  -NMels 64 `
  -TapBlocks "1,3" `
  -SplitUnit file `
  -RunClipPolicy `
  -ForceRebuild
```

---

## A6. Run outputs used

| File | Purpose |
|---|---|
| `summary.json` | Aggregated train/test/clip metrics |
| `policy_results.json` | Segment greedy policy metrics |
| `config_used.yaml` | Saved run configuration |
| `dataset_audit_agent_report.csv/json/md` | Raw and cleaned audit outputs |
| `accepted_manifest.csv` | Accepted input for cleaned dataset build |
| `manual_exclusion_manifest.csv` | Trace of manually excluded file |

---

## A7. Per-exit test report

| Exit | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Support |
|---|---:|---:|---:|---:|---:|---:|
| Exit 1 | 65.62% | 63.74% | 65.24% | 64.04% | 64.36% | 4040 |
| Exit 2 | 92.40% | 92.54% | 92.48% | 92.29% | 92.37% | 4040 |
| Exit 3 | 97.60% | 97.56% | 97.59% | 97.56% | 97.59% | 4040 |

---

## A8. Main policy results

| Metric | Segment greedy | Full-clip greedy | Depth×Time greedy |
|---|---:|---:|---:|
| Accuracy | 96.83% | 99.57% | 98.93% |
| Windows used | — | 8.651 / 8.651 | 2.088 / 8.651 |
| Windows saved | — | 0.00% | 75.87% |
| Compute saved | 52.56% | 0.00% | 75.82% |
| Avg depth | 2.089 | 2.089 | 2.092 |
| Used windows | 4040 | 4040 | 975 |
| Clips | — | 467 | 467 |

---

## A9. Full-clip per-class report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Brene_Brown | 100.00% | 98.88% | 99.44% | 89 |
| Eckhart_Tolle | 100.00% | 100.00% | 100.00% | 99 |
| Eric_Thomas | 98.89% | 100.00% | 99.44% | 89 |
| Gary_Vee | 100.00% | 98.97% | 99.48% | 97 |
| Jay_Shetty | 98.94% | 100.00% | 99.47% | 93 |

---

## A10. Depth×Time per-class report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Brene_Brown | 100.00% | 96.63% | 98.29% | 89 |
| Eckhart_Tolle | 99.00% | 100.00% | 99.50% | 99 |
| Eric_Thomas | 100.00% | 98.88% | 99.44% | 89 |
| Gary_Vee | 96.97% | 98.97% | 97.96% | 97 |
| Jay_Shetty | 98.94% | 100.00% | 99.47% | 93 |

---

## A11. Clip confusion matrices

### A11.1 Full-clip greedy aggregation

Labels:

```text
Brene_Brown, Eckhart_Tolle, Eric_Thomas, Gary_Vee, Jay_Shetty
```

Matrix:

```text
[[88, 0, 1, 0, 0], [0, 99, 0, 0, 0], [0, 0, 89, 0, 0], [0, 0, 0, 96, 1], [0, 0, 0, 0, 93]]
```

Mistakes:

| True class | Predicted class | Count |
|---|---|---:|
| Brene_Brown | Eric_Thomas | 1 |
| Gary_Vee | Jay_Shetty | 1 |

### A11.2 Depth×Time greedy

Matrix:

```text
[[86, 0, 0, 3, 0], [0, 99, 0, 0, 0], [0, 0, 88, 0, 1], [0, 1, 0, 96, 0], [0, 0, 0, 0, 93]]
```

Mistakes:

| True class | Predicted class | Count |
|---|---|---:|
| Brene_Brown | Gary_Vee | 3 |
| Eric_Thomas | Jay_Shetty | 1 |
| Gary_Vee | Eckhart_Tolle | 1 |

---

## A12. TinyAudioTriageAgent seed status

The triage seed dataset is prepared for future detector work, but it was **not used** in the first Raw5 speaker-classification experiment.

| Triage label | Count |
|---|---:|
| target_speaker | 250 |
| other_speaker | 76 |
| music | 50 |
| applause | 49 |
| laughter | 44 |
| silence | 27 |

Total triage seed files: **496**.

---

## A13. Known reproducibility issue

The run command used `-Bandpass "50,7600"` and the cache path includes `bp50-7600`, but `config_used.yaml` still records `audio.bandpass: [100, 3000]`.

This should be fixed so future saved configs include CLI overrides exactly.

---

## A14. Next baseline command placeholder

The next experiment should be a matched uncleaned baseline:

```text
raw5_uncleaned_3exit_greedy
```

Use the same model and evaluation settings. The result should be added only after the run is completed.
