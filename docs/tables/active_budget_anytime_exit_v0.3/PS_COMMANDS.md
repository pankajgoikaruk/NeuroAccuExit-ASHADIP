# PowerShell Commands — v0.12_EE to v0.15_EE

Run every command from the repository root on branch:

```text
active_budget_anytime_exit_v0.3
```

Activate the environment:

```powershell
conda activate ASHADIP_V0
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## Preflight

```powershell
git branch --show-current
git status
Test-Path ".\models\anytime_exit_net.py"
Test-Path ".\human_talk_workspace\tata_v0.8_human_corrected_balanced_pipeline\corrected_holdout\multilabel_features_manifest_CORRECTED_LABELS.csv"
```

## v0.12_EE — label-aware policy

Complete run:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.12_EE\label_aware_policy\run_label_aware_v012_EE.ps1"
```

Reuse an existing frozen policy:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.12_EE\label_aware_policy\run_label_aware_v012_EE.ps1" `
  -SkipPrechecks `
  -SkipTuning
```

Override the validation quality constraint:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.12_EE\label_aware_policy\run_label_aware_v012_EE.ps1" `
  -MaxMacroF1Drop 0.005 `
  -MinExit2Fraction 0.02
```

Output root:

```text
human_talk_workspace\active_budget_anytime_exit_v0.3\v0.12_EE\label_aware_policy\
```

## v0.13_EE — matched strategy comparison

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.13_EE\matched_policy_comparison\run_matched_policy_comparison_v013_EE.ps1"
```

Output root:

```text
human_talk_workspace\active_budget_anytime_exit_v0.3\v0.13_EE\matched_policy_comparison\
```

Key outputs:

```text
validation_tuning\frozen_matched_policy_comparison_v013.json
validation_tuning\v013_matched_policy_validation_sweep.csv
validation_tuning\v013_selected_policy_comparison.csv
corrected_holdout_evaluation\v013_matched_holdout_comparison.csv
```

## v0.14_EE — parent-aware gate

Initial run:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.14_EE\parent_aware_gate\run_parent_aware_gate_v014_EE.ps1"
```

Publication-style timing:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.14_EE\parent_aware_gate\run_parent_aware_gate_v014_EE.ps1" `
  -TimingRepeats 30
```

Output root:

```text
human_talk_workspace\active_budget_anytime_exit_v0.3\v0.14_EE\parent_aware_gate\
```

Key outputs:

```text
validation_tuning\frozen_parent_aware_gate_v014.json
validation_tuning\v014_parent_aware_gate_validation_sweep.csv
validation_tuning\v014_selected_parent_aware_policies.csv
corrected_holdout_evaluation\v014_parent_aware_holdout_comparison.csv
```

## v0.15_EE — whole-parent risk control

Initial run:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.15_EE\whole_parent_risk_control\run_whole_parent_risk_control_v015_EE.ps1"
```

Publication-style timing:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.15_EE\whole_parent_risk_control\run_whole_parent_risk_control_v015_EE.ps1" `
  -TimingRepeats 30
```

Output root:

```text
human_talk_workspace\active_budget_anytime_exit_v0.3\v0.15_EE\whole_parent_risk_control\
```

Key outputs:

```text
validation_tuning\frozen_whole_parent_policy_v015.json
validation_tuning\v015_whole_parent_validation_sweep.csv
validation_tuning\v015_selected_parent_policies.csv
corrected_holdout_evaluation\v015_whole_parent_holdout_comparison.csv
```

## Run policy tests directly

```powershell
python -m unittest `
  tests.test_anytime_exit_net `
  tests.test_label_aware_early_exit_policy `
  tests.test_early_exit_strategy_comparison `
  tests.test_parent_aware_adaptive_gate `
  tests.test_whole_parent_selective_exit `
  -v
```

## Canonical full-depth reproduction

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\docs\tables\active_budget_anytime_exit_v0.1\full_depth_baselines\primary_v010_no_hint_historical_lats_v2\REPRODUCE_PRIMARY.ps1"
```

## Important command cautions

- Do not use a missing `tuned_per_exit` threshold mode unless the canonical run contains `threshold_tuning\threshold_comparison.json`.
- Do not modify the corrected-holdout labels or LATS-v2 configuration between comparisons.
- Do not rerun tuning after inspecting holdout outcomes unless it is declared as a new experiment with a new untouched final test set.
- Use `-TimingRepeats 30` for final latency records; the default shorter run is a development check.
