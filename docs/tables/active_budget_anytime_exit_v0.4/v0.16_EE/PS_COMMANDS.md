# v0.16_EE PowerShell Commands

Run every command from the repository root on branch `active_budget_anytime_exit_v0.4`.

## Environment

```powershell
conda activate ASHADIP_V0
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## Full experiment

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1"
```

This runs tests, checkpoint equivalence, validation optimisation, policy freezing, genuine holdout evaluation, and 10 timing repetitions.

## Publication timing

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -TimingRepeats 30
```

## Reuse frozen policy

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -SkipPrechecks `
  -SkipTuning `
  -TimingRepeats 30
```

## Custom search size

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -PopulationSize 120 `
  -Generations 100 `
  -CvFolds 5
```

A new search is a new experiment and must not overwrite the reported v0.16 interpretation without a separate version record.

## Custom validation constraints

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -MaxMacroF1Drop 0.005 `
  -MaxMicroF1Drop 0.003 `
  -MaxExactMatchDrop 0.005 `
  -MaxHammingIncrease 0.001 `
  -MinExit2Fraction 0.02
```

Do not choose new limits after inspecting their corrected-holdout outcome and then describe them as pre-specified.

## Direct validation tuning

The branch runner is preferred because it enforces paths and branch identity. For debugging, its underlying tuning stage calls:

```powershell
python ".\scripts\v0.16_EE\multiobjective_per_label_margin\tune_multiobjective_per_label_margin_v016.py" `
  --run_dir "<canonical-run-directory>" `
  --labels_json ".\configs\human_talk_10label_schema.json" `
  --lats_config_json ".\docs\tables\agentic_data_preprocessing_v0.10\no_hint_lats_v2_coordinate_reoptimized_config.json" `
  --parent_id_col "parent_clip_id" `
  --threshold_mode "fixed_0p5" `
  --population_size 80 `
  --generations 50 `
  --cv_folds 5 `
  --max_macro_f1_drop 0.01 `
  --max_micro_f1_drop 0.005 `
  --max_exact_match_drop 0.01 `
  --max_hamming_increase 0.002 `
  --min_exit2_fraction 0.02 `
  --batch_size 128 `
  --device cpu `
  --out_dir ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.16_EE\multiobjective_per_label_margin\validation_tuning"
```

## Reporting

```powershell
$Root = ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.16_EE\multiobjective_per_label_margin"

Import-Csv "$Root\validation_tuning\v016_selected_policy.csv" |
  Format-List

Import-Csv "$Root\validation_tuning\v016_pareto_front.csv" |
  Sort-Object {[double]$_.estimated_flops_saved_pct} -Descending |
  Select-Object -First 20 |
  Format-Table estimated_flops_saved_pct,parent_macro_f1,parent_micro_f1,parent_exact_match,parent_hamming_loss -AutoSize

Import-Csv "$Root\corrected_holdout_evaluation\v016_multiobjective_holdout_comparison.csv" |
  Format-Table method,exit2_fraction,estimated_flops_saved_pct,measured_speedup_vs_always_exit3,parent_macro_f1,parent_micro_f1,parent_exact_match,parent_hamming_loss -AutoSize
```

## Training

There is no v0.16 model-training command. The TinyAudioCNN checkpoint and all exit heads are frozen. v0.16 performs policy optimisation and staged inference only.
