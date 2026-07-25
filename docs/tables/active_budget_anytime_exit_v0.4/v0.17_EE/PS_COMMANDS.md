# v0.17_EE PowerShell Commands

Run all commands from:

```text
C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP
```

## Environment and branch

```powershell
conda activate ASHADIP_V0

$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

git fetch origin
git switch active_budget_anytime_exit_v0.4
git pull --ff-only origin active_budget_anytime_exit_v0.4
```

## Complete three-exit experiment

This command runs tests, staged equivalence, validation tuning, frozen-policy corrected-holdout evaluation, ablations, and timing.

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -Run3Only
```

## Complete combined three-exit and five-exit experiment

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324"
```

## Publication-quality timing

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -TimingRepeats 30
```

## Re-evaluate frozen policies without retuning

Use this after validation tuning has already produced both frozen policy JSON files.

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -SkipTuning `
  -TimingRepeats 30
```

To reuse both the frozen policies and previous precheck evidence:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -SkipPrechecks `
  -SkipTuning `
  -TimingRepeats 30
```

Do not use `-SkipTuning` unless the matching frozen policy files already exist under the corresponding `validation_tuning` directories.

## Re-run with explicit experimental settings

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
  -BatchSize 128 `
  -TorchThreads 1 `
  -ThresholdMode fixed_0p5 `
  -TimingRepeats 30
```

## Find five-exit checkpoints

```powershell
Get-ChildItem ".\human_talk_workspace" `
  -File `
  -Recurse `
  -Filter "best.pt" |
Where-Object { $_.FullName -match "5exit" } |
Select-Object -ExpandProperty DirectoryName
```

## Tests only

```powershell
python -m unittest `
  tests.test_anytime_exit_net `
  tests.test_sequential_anytime_exit `
  -v
```

## Direct tuning, evaluation, and reporting entrypoints

Inspect the accepted arguments before direct use:

```powershell
python ".\scripts\v0.17_EE\sequential_anytime_exit\tune_sequential_anytime_exit_v017.py" --help
python ".\scripts\v0.17_EE\sequential_anytime_exit\evaluate_sequential_anytime_exit_v017.py" --help
python ".\scripts\v0.17_EE\sequential_anytime_exit\compare_sequential_architectures_v017.py" --help
```

The runner is the authoritative command because it supplies the canonical manifests, features, label schema, LATS-v2 configuration, output paths, constraints, and architecture-specific policy paths.

## Output and reporting locations

```powershell
$V017Root = ".\human_talk_workspace\active_budget_anytime_exit_v0.4\v0.17_EE\sequential_anytime_exit"

Get-ChildItem "$V017Root\3exit\validation_tuning" -File
Get-ChildItem "$V017Root\3exit\corrected_holdout_evaluation" -File -Recurse
Get-ChildItem "$V017Root\5exit\validation_tuning" -File
Get-ChildItem "$V017Root\5exit\corrected_holdout_evaluation" -File -Recurse
Get-ChildItem "$V017Root\architecture_comparison" -File
```

Important reports:

```text
3exit/corrected_holdout_evaluation/v017_3exit_holdout_comparison.csv
3exit/corrected_holdout_evaluation/v017_3exit_ablation_table.csv
5exit/corrected_holdout_evaluation/v017_5exit_holdout_comparison.csv
5exit/corrected_holdout_evaluation/v017_5exit_ablation_table.csv
architecture_comparison/v017_3exit_vs_5exit_headline.csv
architecture_comparison/v017_combined_ablation_table.csv
architecture_comparison/v017_exit_distribution_comparison.csv
architecture_comparison/v017_fairness_audit.json
```

## Training status

No new backbone or exit-head training was performed for the completed v0.17 experiment. Both checkpoints were frozen and reused; v0.17 trained/optimised only the stopping-policy parameters.

A future fair comparison must train a new five-exit checkpoint using the same canonical manifest, preprocessing, labels, seed policy, loss, and evaluation protocol as the three-exit model. A speculative training command is intentionally not recorded as a completed v0.17 command. Document that command only after the matching five-exit manifest and run configuration are finalised.

## Reproducibility caution

- Do not retune the policy on the corrected holdout.
- Keep validation eligibility separate from holdout constraint satisfaction.
- Use the frozen policy generated under the corresponding architecture's `validation_tuning` directory.
- Report estimated FLOPs and measured timing separately.
- The historical five-exit checkpoint is valid for a within-model result, not a fair architecture-superiority claim.