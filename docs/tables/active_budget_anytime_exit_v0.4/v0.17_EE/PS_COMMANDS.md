# v0.17_EE PowerShell Commands

Run all commands from the repository root:

```text
C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP
```

## Environment and branch

```powershell
conda activate ASHADIP_V0

git switch active_budget_anytime_exit_v0.4
git pull --ff-only origin active_budget_anytime_exit_v0.4

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## Training status

v0.17 does **not** train the CNN or exit heads. It reuses the canonical 3-exit run and the tested v0.6 5-exit run. No v0.17 backbone-training command exists. A future fair architecture comparison requires a new 5-exit checkpoint trained with the canonical 3-exit manifest and settings; that future training must be documented separately.

## Complete 3-exit run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -Run3Only
```

## Complete tested 3-exit and 5-exit run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324"
```

## Publication timing

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -TimingRepeats 30
```

## Reuse frozen policies

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -SkipPrechecks `
  -SkipTuning `
  -TimingRepeats 30
```

## Custom optimiser settings

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -PopulationSize 96 `
  -Generations 60 `
  -CvFolds 5 `
  -MaxMacroF1Drop 0.01 `
  -MaxMicroF1Drop 0.005 `
  -MaxExactMatchDrop 0.01 `
  -MaxHammingIncrease 0.002 `
  -MinTotalEarlyFraction 0.02 `
  -MinExit1Fraction 0.005 `
  -SafetyFraction 0.75 `
  -TimingRepeats 30 `
  -TorchThreads 1
```

## Policy and staged-wrapper tests

```powershell
python -m unittest `
  tests.test_anytime_exit_net `
  tests.test_sequential_anytime_exit `
  -v
```

## Direct tuner

The branch runner is recommended because it supplies the exact manifests and constraints. The direct entry point is:

```powershell
python ".\scripts\v0.17_EE\sequential_anytime_exit\tune_sequential_anytime_exit_v017.py" `
  --run_dir "<RUN_DIRECTORY>" `
  --labels_json ".\configs\human_talk_10label_schema.json" `
  --lats_config_json ".\docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json" `
  --parent_id_col "parent_clip_id" `
  --threshold_mode "fixed_0p5" `
  --population_size 96 `
  --generations 60 `
  --cv_folds 5 `
  --max_macro_f1_drop 0.01 `
  --max_micro_f1_drop 0.005 `
  --max_exact_match_drop 0.01 `
  --max_hamming_increase 0.002 `
  --min_total_early_fraction 0.02 `
  --min_exit1_fraction 0.005 `
  --safety_fraction 0.75 `
  --batch_size 128 `
  --device cpu `
  --out_dir "<VALIDATION_OUTPUT_DIRECTORY>"
```

## Direct holdout evaluator

```powershell
python ".\scripts\v0.17_EE\sequential_anytime_exit\evaluate_sequential_anytime_exit_v017.py" `
  --run_dir "<RUN_DIRECTORY>" `
  --policy_json "<FROZEN_POLICY_JSON>" `
  --holdout_manifest ".\human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv" `
  --features_root ".\human_talk_workspace\tata_v0.6_raw_pipeline\final_holdout_feature_cache\features" `
  --labels_json ".\configs\human_talk_10label_schema.json" `
  --lats_config_json ".\docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json" `
  --parent_id_col "parent_clip_id" `
  --batch_size 128 `
  --timing_repeats 30 `
  --torch_threads 1 `
  --device cpu `
  --out_dir "<HOLDOUT_OUTPUT_DIRECTORY>"
```

## Architecture reporting

```powershell
python ".\scripts\v0.17_EE\sequential_anytime_exit\compare_sequential_architectures_v017.py" `
  --policy_3 ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit\3exit\validation_tuning\frozen_sequential_policy_3exit_v017.json" `
  --comparison_3 ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit\3exit\corrected_holdout_evaluation\v017_3exit_holdout_comparison.csv" `
  --policy_5 ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit\5exit\validation_tuning\frozen_sequential_policy_5exit_v017.json" `
  --comparison_5 ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit\5exit\corrected_holdout_evaluation\v017_5exit_holdout_comparison.csv" `
  --out_dir ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit\architecture_comparison"
```

## Output roots

```text
human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit\
├── 3exit\
├── 5exit\
└── architecture_comparison\
```
