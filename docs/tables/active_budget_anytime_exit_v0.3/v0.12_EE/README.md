# v0.12_EE — Validation-Derived Label-Aware Dynamic Early Exit

## Status

Complete. Genuine Exit-2/Exit-3 staged inference with validation-derived label risk.

## Research question

Can validation evidence about label difficulty prevent premature Exit-2 decisions for labels that benefit most from deeper processing?

## Implementation

```text
policies/label_aware_early_exit_policy.py
tests/test_label_aware_early_exit_policy.py
scripts/v0.12_EE/label_aware_policy/
```

The policy combines Exit 1–Exit 2 label-set agreement, non-empty prediction, mean binary confidence, threshold margin, inter-exit probability change, and validation-derived per-label risk. Risk is a runtime continuation score, not a training-loss penalty.

## Frozen settings

| Setting | Value |
|---|---:|
| Device / batch size | CPU / 128 |
| Segment thresholds | fixed 0.5 |
| Parent evaluation | frozen LATS-v2 |
| Maximum validation Macro-F1 drop | 0.01 |
| Minimum validation Exit-2 fraction | 0.02 |
| Selected confidence | 0.55 |
| Selected margin | 0.00 |
| Selected max delta | 1.00 |
| Selected label-risk threshold | 0.50 |
| Exit 1–Exit 2 agreement | Required |

## Validation result

| Exit-2 fraction | Parent Macro-F1 | Macro drop | FLOPs saved | Status |
|---:|---:|---:|---:|---|
| 22.73% | 0.894195 | 0.008422 | 14.61% | `quality_constraint_met` |

## Holdout result

| Exit-2 rate | Avg depth | FLOPs saved | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11.19% | 2.8881 | 7.19% | 0.843703 | 0.936689 | 0.944692 | 0.840830 | 0.018570 |

## Interpretation

The method demonstrated genuine label-aware compute skipping and slightly improved over the earlier permissive global rule, but the final quality loss remained too large. Validation-to-holdout stopping coverage also shifted substantially.

## Command

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.12_EE\label_aware_policy\run_label_aware_v012_EE.ps1"
```

## Outputs

```text
human_talk_workspace/active_budget_anytime_exit_v0.3/v0.12_EE/label_aware_policy/
├── checkpoint_staged_equivalence_precheck.json
├── validation_tuning/frozen_label_aware_policy_v012.json
├── validation_tuning/v012_label_aware_validation_sweep.csv
├── validation_tuning/v012_validation_label_risk_profile.csv
└── corrected_holdout_evaluation/v012_label_aware_runtime_summary.json
```

## Limitation

The reported v0.12 latency is model-only and lacks a controlled identical-protocol Always Exit 3 timing comparator.
