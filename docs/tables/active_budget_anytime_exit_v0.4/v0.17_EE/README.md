# NeuroAccuExit v0.17_EE — Sequential Active-Budget Anytime Exit

## Status

**Complete implementation and full integration for both tested checkpoints.**

The v0.17 package records:

- genuine sequential staged inference for `Exit 1 → Exit 2 → Exit 3`;
- genuine sequential staged inference for `Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5`;
- exact staged/full checkpoint-equivalence checks for both architectures;
- validation-only, constraint-aware NSGA-II-style policy optimisation;
- safety-buffered Pareto-knee selection;
- corrected-holdout evaluation with frozen policies and no holdout retuning;
- six policy ablations per architecture;
- 30-repeat controlled CPU timing;
- per-label and parent-change analysis;
- an explicit cross-architecture fairness audit;
- cross-version corrected-holdout tables and plots through `v0.17_EE`.

## Headline decision

| Finding | Status |
|---|---|
| 5-exit full sequential policy | **Confirmed within-checkpoint success** |
| 3-exit full sequential policy | Compute-successful but **not holdout quality-safe** |
| Direct claim that 5 exits outperform 3 exits | **Not supported yet** because the training/validation manifests differ |
| Exit 1 | Useful for additional savings, but the riskiest stopping stage |
| Label-specific margins | Strongly supported by ablations |
| Previous-exit stability | Supported as a safety mechanism |
| Current risk term | Practically non-binding in the selected policies |

## Holdout headline

| Architecture | Route | Early-exit fraction | FLOPs saved | Speedup | Parent Macro-F1 | Parent Micro-F1 | Exact Match | Hamming ↓ | Holdout limits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3-exit | `1→2→3` | 10.40% | 8.64% | 1.037× | 0.840128 | 0.937549 | 0.840830 | 0.018224 | **Failed** |
| 5-exit | `1→2→3→4→5` | 52.94% | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.688581 | 0.039100 | **Passed** |

The 5-exit policy met every predefined holdout-quality threshold relative to its own Always Exit 5 baseline while saving `30.71%` estimated FLOPs and producing a `1.114×` median CPU speedup. Exact Match increased from `0.673587` to `0.688581`.

## Cross-version records

| File | Purpose |
|---|---|
| `cross_version_3exit_table.csv` | Fair canonical 3-exit corrected-holdout comparison through v0.17 |
| `v017_architecture_table.csv` | v0.17 3-exit/5-exit policies against their own full-depth references |
| `cross_version_3exit_flops.svg` | Cross-version compute-saving plot |
| `cross_version_3exit_quality.svg` | Cross-version Macro/Micro/Samples/Exact plot |
| `cross_version_3exit_hamming.svg` | Cross-version Hamming-loss plot |
| `v017_architecture_flops.svg` | Separate v0.17 architecture-extension FLOPs plot |
| `FIGURES.md` | Figure interpretation and fairness cautions |

## Main records

| File | Purpose |
|---|---|
| `EXPERIMENT_SETUP.md` | Checkpoints, data, architectures, optimiser settings, constraints, and protocol |
| `THEORY_AND_METHOD.md` | Sequential multi-label stopping theory and Pareto selection |
| `RESULTS_AND_ANALYSIS.md` | Validation, holdout, timing, parent-change, per-label, and fairness analysis |
| `ABLATIONS_AND_FINDINGS.md` | Exit-1, stability, risk, label-margin, and confidence-only ablations |
| `PAPER_READY_SUMMARY.md` | Paper/thesis-safe claims and non-claims |
| `PS_COMMANDS.md` | Full, frozen-reuse, direct tuning/evaluation, and reporting commands |
| `REPRODUCE_V017_EE.ps1` | Documentation entry point |
| `experiment_manifest.json` | Compact machine-readable experiment record |
| `holdout_constraint_check.csv` | Explicit holdout quality audit |
| `cross_architecture_headline.csv` | 3-exit and 5-exit headline summary |
| `ablation_3exit.csv`, `ablation_5exit.csv` | Architecture-specific ablations |
| `per_label_holdout_delta_*.csv` | Label-level changes |
| `fairness_audit.json` | Direct-comparison validity |

## Main command

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -TimingRepeats 30
```

## Research decision

The 5-exit result is the major successful outcome of v0.17, but it must be reported as a **within-architecture quality–efficiency result**. The fairness audit failed only `same_validation_manifest`; therefore, the current evidence does not establish that five exits are intrinsically superior to three exits.

The next fair architectural study must train a 5-exit checkpoint using the exact canonical v0.8/v0.10 training manifest and preprocessing used by the 3-exit checkpoint.
