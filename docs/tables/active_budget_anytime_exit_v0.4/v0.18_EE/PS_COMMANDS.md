# v0.18 PowerShell Commands

Run from the repository root.

## Complete pipeline

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1" `
  -TimingRepeats 30
```

## Force fair five-exit retraining

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1" `
  -ForceRetrain5 `
  -TimingRepeats 30
```

## Reuse an existing fair five-exit run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1" `
  -RunDir5 "<FAIR_5EXIT_RUN_DIRECTORY>" `
  -TimingRepeats 30
```

## Fair-training audit only

```powershell
python ".\scripts\v0.18_EE\fair_sequential_anytime_exit\audit_fair_training_v018.py" `
  --run3 "<CANONICAL_3EXIT_RUN>" `
  --run5 "<FAIR_5EXIT_RUN>" `
  --out_json ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.18_EE\fair_sequential_anytime_exit\fair_training_audit.json"
```

## Strict policy tuning

```powershell
python ".\scripts\v0.18_EE\fair_sequential_anytime_exit\tune_strict_sequential_v018.py" `
  --run_dir "<RUN_DIRECTORY>" `
  --labels_json ".\configs\human_talk_10label_schema.json" `
  --lats_config_json ".\docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json" `
  --population_size 112 `
  --generations 70 `
  --cv_folds 5 `
  --safety_fraction 0.50 `
  --device cpu `
  --out_dir "<VALIDATION_OUTPUT_DIRECTORY>"
```

## Holdout evaluation

```powershell
python ".\scripts\v0.18_EE\fair_sequential_anytime_exit\evaluate_strict_sequential_v018.py" `
  --run_dir "<RUN_DIRECTORY>" `
  --policy_json "<FROZEN_POLICY_JSON>" `
  --holdout_manifest "<CORRECTED_HOLDOUT_MANIFEST>" `
  --features_root "<HOLDOUT_FEATURE_ROOT>" `
  --labels_json ".\configs\human_talk_10label_schema.json" `
  --lats_config_json ".\docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json" `
  --timing_repeats 30 `
  --torch_threads 1 `
  --device cpu `
  --out_dir "<HOLDOUT_OUTPUT_DIRECTORY>"
```

## Architecture comparison

```powershell
python ".\scripts\v0.18_EE\fair_sequential_anytime_exit\compare_v018.py" `
  --audit_json "<FAIR_TRAINING_AUDIT_JSON>" `
  --comparison_3 "<V018_3EXIT_COMPARISON_CSV>" `
  --comparison_5 "<V018_5EXIT_COMPARISON_CSV>" `
  --out_dir "<ARCHITECTURE_COMPARISON_DIRECTORY>"
```
