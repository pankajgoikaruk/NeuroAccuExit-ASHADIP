# Active Budget and Anytime Exit v0.3 — Documentation Package

This directory stores compact, version-controlled records for the completed v0.12–v0.15 Early-Exit experiments.

## Package index

| Path | Purpose |
|---|---|
| `CUMULATIVE_RESULTS.md` | Confirmed cross-version tables, ablations, figures and final method selection |
| `PS_COMMANDS.md` | Windows commands, optional flags and runtime output paths |
| `cumulative_results_v012_v015.csv` | Machine-readable headline metrics |
| `v0.12_EE/README.md` | Label-aware risk-policy setup and result |
| `v0.13_EE/README.md` | Matched policy and learned-gate comparison |
| `v0.14_EE/README.md` | Parent-aware gate and Exit-1 ablation |
| `v0.15_EE/README.md` | Whole-parent selective risk control |

## Baseline dependency

The canonical full-depth result is not duplicated:

```text
docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/
```

## Storage policy

Committed:

- experiment descriptions;
- exact frozen settings;
- headline and ablation tables;
- PowerShell commands;
- theoretical notes;
- limitations and paper-safe interpretation.

Local only under `human_talk_workspace`:

- checkpoints and features;
- full segment prediction matrices;
- fitted `.joblib` gates;
- full validation sweeps;
- parent score/prediction tables;
- OOF probability exports.

## Current conclusion

Always Exit 3 remains the full-quality and measured-latency reference. The v0.13 per-label margin rule is the current adaptive recommendation. v0.14 and v0.15 are preserved as informative negative/diagnostic experiments.
