# v0.17_EE PowerShell Commands

Run all commands from:

```text
C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP
```

## Environment and branch

```powershell
conda activate ASHADIP_V0

git fetch origin
git switch active_budget_anytime_exit_v0.4
git pull --ff-only origin active_budget_anytime_exit_v0.4
```

## Three-exit full experiment

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -Run3Only
```

## Combined three-exit and five-exit experiment

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

## Direct tuning/evaluation help

```powershell
python ".\scripts\v0.17_EE\sequential_anytime_exit\tune_sequential_anytime_exit_v017.py" --help
python ".\scripts\v0.17_EE\sequential_anytime_exit\evaluate_sequential_anytime_exit_v017.py" --help
python ".\scripts\v0.17_EE\sequential_anytime_exit\compare_sequential_architectures_v017.py" --help
```

Use the frozen policy generated under the corresponding architecture's `validation_tuning` directory. Do not retune on the corrected holdout.

## Training status

No new backbone training was performed for the completed v0.17 experiment. The three-exit and historical five-exit checkpoints were frozen.

A future fair comparison must train a new five-exit checkpoint using the same canonical manifest, preprocessing, labels, seed policy, loss, and evaluation protocol as the three-exit model. That training command should be documented only after the matching manifest and run configuration are finalized.
