# v0.11_EE PowerShell Commands

Run all commands from the repository root.

## Environment

```powershell
conda activate ASHADIP_V0
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## Verify repository branch

```powershell
git switch active_budget_anytime_exit_v0.2
git pull origin active_budget_anytime_exit_v0.2
git status
```

## Unit tests only

```powershell
python -m unittest tests.test_anytime_exit_net -v
```

Expected:

```text
Ran 4 tests
OK
```

## Fixed-exit audit

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\fixed_policy\run_v011_EE.ps1"
```

Skip unit tests:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\fixed_policy\run_v011_EE.ps1" `
  -SkipUnitTests
```

Specify the canonical run explicitly:

```powershell
$RunDir = "human_talk_workspace\tata_v0.10_hint_pass_lats_pipeline\main_models\runs\main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845"

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\fixed_policy\run_v011_EE.ps1" `
  -RunDir "$RunDir"
```

## Full Dynamic Early-Exit pipeline

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1"
```

The runner:

1. runs staged-wrapper tests;
2. verifies the real checkpoint;
3. tunes and freezes the policy on validation data;
4. evaluates genuine staged inference on the corrected holdout.

## Force fixed 0.5 threshold mode

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -ThresholdMode fixed_0p5
```

The default `auto` mode already falls back to `fixed_0p5` when the canonical run has no per-exit threshold file.

## Reuse a frozen policy

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -SkipPrechecks `
  -SkipTuning
```

This requires:

```text
human_talk_workspace\active_budget_anytime_exit_v0.2\v0.11_EE\
dynamic_policy\validation_tuning\frozen_dynamic_policy_v011.json
```

## Stricter validation policy

Maximum Macro-F1 drop of 0.01:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -MaxMacroF1Drop 0.01
```

Maximum Macro-F1 drop of 0.005:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -MaxMacroF1Drop 0.005
```

Treat each constraint as a separately named validation-selected experiment. Do not modify its policy after holdout evaluation.

## GPU execution

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -Device cuda
```

Use only when the environment and checkpoint support CUDA.

## Combined reproduction

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\docs\tables\active_budget_anytime_exit_v0.2\v0.11_EE\REPRODUCE_V011_EE.ps1"
```

## Expected output roots

Fixed exits:

```text
human_talk_workspace\active_budget_anytime_exit_v0.2\v0.11_EE\fixed_exit_audit
```

Dynamic policy:

```text
human_talk_workspace\active_budget_anytime_exit_v0.2\v0.11_EE\dynamic_policy
```

## Important safeguards

- Tune on validation only.
- Never select a policy using corrected-holdout metrics.
- Do not call post-hoc policy selection genuine compute saving.
- Do not interpret `avg_pred_labels` as average exit depth.
- Do not claim measured speedup until Always Exit 3 is timed with the same procedure.
