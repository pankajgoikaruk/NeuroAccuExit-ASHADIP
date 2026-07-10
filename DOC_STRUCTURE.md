# Documentation Structure — Active Budget and Anytime Exit v0.1

This document indexes the branch-specific documentation for the next NeuroAccuExit research phase.

---

## Branch identity

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.1` |
| Documentation name | **NeuroAccuExit — Active Budget and Anytime Exit v0.1** |
| Source branch | `agentic_data_preprocessing_v0.10` |
| Current milestone | Full-depth v0.10 no-hint LATS-v2 baseline frozen and reproduced |
| Next milestone | Standard Early-Exit evaluation |

---

## Top-level files

| File | Purpose |
|---|---|
| `README.md` | Authoritative project summary for the active branch, canonical baseline, roadmap, evaluation protocol, and paper-ready statement |
| `DOC_STRUCTURE.md` | This documentation and artifact index |
| `docs/v0.10/` | Historical v0.10 experiment documentation |
| `docs/v0.10_1/` | Historical low-energy recovery ablation plan and final results |
| `docs/tables/agentic_data_preprocessing_v0.10/` | Historical v0.10 tables, configurations, and diagnostic outputs |
| `docs/archive/` | Archived documentation from earlier project states |

The root files now describe the active efficiency-research branch. Historical v0.10 files remain available and must not be overwritten because they preserve the development and negative-result record.

---

## Branch documentation root

```text
docs/tables/active_budget_anytime_exit_v0.1/
```

Current structure:

```text
docs/tables/active_budget_anytime_exit_v0.1/
└── full_depth_baselines/
    ├── README.md
    ├── PAPER_READY_BASELINE_SUMMARY.md
    ├── artifact_hashes.csv
    ├── environment_pip_freeze.txt
    ├── environment_summary.txt
    ├── reproducibility_manifest.json
    │
    ├── primary_v010_no_hint_historical_lats_v2/
    │   ├── EXPERIMENT_RECORD.md
    │   ├── REPRODUCE_PRIMARY.ps1
    │   └── frozen primary result artifacts
    │
    ├── secondary_direct_coordinate_reoptimized/
    │   ├── EXPERIMENT_RECORD.md
    │   ├── REPRODUCE_SECONDARY.ps1
    │   └── frozen secondary result artifacts
    │
    ├── shared_reproducibility_inputs/
    │   ├── no_hint_exit3_segment_probabilities.csv
    │   ├── historical_no_hint_lats_v2_config.json
    │   ├── human_talk_10label_schema.json
    │   ├── evaluate_frozen_lats_config_v010_snapshot.py
    │   └── run_v010_lats_v2_coordinate_reoptimize_snapshot.py
    │
    └── reproduced_outputs/
        └── primary_historical_lats_v2/
            ├── v010_frozen_lats_eval.csv
            ├── v010_frozen_lats_eval.json
            ├── v010_frozen_lats_per_label.csv
            ├── v010_frozen_lats_config_used.json
            ├── v010_parent_predictions.csv
            ├── v010_parent_scores.csv
            └── v010_parent_truth.csv
```

---

## Frozen package files

### Package-level documentation

| File | Purpose |
|---|---|
| `full_depth_baselines/README.md` | Declares the canonical full-depth baseline and explains package contents |
| `full_depth_baselines/PAPER_READY_BASELINE_SUMMARY.md` | Concise method and result wording for research-paper drafting |
| `full_depth_baselines/reproducibility_manifest.json` | Machine-readable model, data, evaluation, metric, and baseline-selection metadata |
| `full_depth_baselines/artifact_hashes.csv` | SHA256 integrity record for frozen files |
| `full_depth_baselines/environment_summary.txt` | Git branch/commit, Python, and core-package versions |
| `full_depth_baselines/environment_pip_freeze.txt` | Complete Python package snapshot |

### Primary baseline

| File/folder | Purpose |
|---|---|
| `primary_v010_no_hint_historical_lats_v2/EXPERIMENT_RECORD.md` | Detailed settings, model source, data, frozen rules, results, rationale, and limitations |
| `primary_v010_no_hint_historical_lats_v2/REPRODUCE_PRIMARY.ps1` | Deterministically regenerates and validates the canonical result |
| Primary result CSV/JSON files | Preserve the exact canonical output and per-label measurements |
| Parent prediction/score/truth files | Support auditing, error analysis, and paper tables |

### Secondary result

| File/folder | Purpose |
|---|---|
| `secondary_direct_coordinate_reoptimized/EXPERIMENT_RECORD.md` | Documents the new direct coordinate-search settings and result |
| `secondary_direct_coordinate_reoptimized/REPRODUCE_SECONDARY.ps1` | Regenerates the secondary post-hoc variant |
| Secondary search outputs | Preserve the alternative global-consistency trade-off |

### Shared reproducibility inputs

| File | Purpose |
|---|---|
| `no_hint_exit3_segment_probabilities.csv` | Exact final-exit probability input used for both frozen results |
| `historical_no_hint_lats_v2_config.json` | Canonical label-specific aggregation and threshold configuration |
| `human_talk_10label_schema.json` | Frozen label ordering and schema |
| `evaluate_frozen_lats_config_v010_snapshot.py` | Evaluator snapshot used for deterministic replay |
| `run_v010_lats_v2_coordinate_reoptimize_snapshot.py` | Search-script snapshot used for the secondary result |

### Reproduced outputs

| File | Purpose |
|---|---|
| `v010_frozen_lats_eval.csv` | Canonical summary metrics |
| `v010_frozen_lats_eval.json` | Canonical machine-readable evaluation |
| `v010_frozen_lats_per_label.csv` | Label-level precision, recall, and F1 |
| `v010_frozen_lats_config_used.json` | Configuration actually applied during replay |
| `v010_parent_predictions.csv` | Parent-level binary predictions |
| `v010_parent_scores.csv` | Parent-level aggregated probability scores |
| `v010_parent_truth.csv` | Parent-level ground-truth labels |

---

## Canonical result snapshot

### Primary branch baseline

```text
v0.10 no-hint + frozen historical LATS-v2
```

| Metric | Value |
|---|---:|
| Macro-F1 | **0.8623815322** |
| Micro-F1 | **0.9531311540** |
| Samples-F1 | **0.9588894381** |
| Exact Match | **0.8765859285** |
| Hamming Loss ↓ | **0.0137254902** |
| Average predicted labels | 1.4590542099 |
| Parent clips | 867 |

This is the sole full-depth quality reference for the active branch.

### Secondary frozen ablation

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8598605 |
| Micro-F1 | 0.9547260 |
| Samples-F1 | 0.9619926 |
| Exact Match | 0.8800461 |
| Hamming Loss ↓ | 0.0131488 |
| Average predicted labels | 1.4348328 |

The secondary result is retained for trade-off analysis and must not replace the canonical baseline.

---

## Historical documentation retained

### v0.10 files

| File | Purpose |
|---|---|
| `docs/v0.10/README.md` | Historical v0.10 branch summary |
| `docs/v0.10/EXPERIMENT_SETUP.md` | Dataset, paths, model, LATS, hint-pass, and weighting settings |
| `docs/v0.10/RESULTS_AND_ANALYSIS.md` | Full historical method comparison |
| `docs/v0.10/SEED_STABILITY_ANALYSIS.md` | No-hint seed-stability analysis |
| `docs/v0.10/POS_WEIGHT_EXPERIMENT_ANALYSIS.md` | `pos_weight cap5` diagnostic |
| `docs/v0.10/RESEARCH_FINDINGS.md` | Historical research questions and outcomes |
| `docs/v0.10/PS_COMMANDS.md` | Historical PowerShell commands |


### v0.10_1 files

| File | Purpose |
|---|---|
| `docs/v0.10_1/LOW_ENERGY_RECOVERY_ABLATION_PLAN.md` | Non-destructive plan for the reviewed low-energy recovery experiment |
| `docs/v0.10_1/LOW_ENERGY_RECOVERY_ABLATION_RESULTS.md` | Final metrics, comparison, interpretation, and rejection decision |

The v0.10_1 result was:

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8581 |
| Micro-F1 | 0.9446 |
| Samples-F1 | 0.9519 |
| Exact Match | 0.8570 |
| Hamming Loss ↓ | 0.0160 |

It is retained as a valid negative/diagnostic ablation and is not the
full-depth baseline for the active-budget branch.


### Historical tables

| Path | Purpose |
|---|---|
| `docs/tables/agentic_data_preprocessing_v0.10/` | v0.9_4/v0.10 comparisons, seed stability, hint-pass, weighting, and related diagnostic outputs |
| `docs/archive/` | Snapshots of earlier README and structure documents |

Historical records remain valid. Their previous overall decision is not silently rewritten. The active branch simply defines a new branch-specific comparator for Early-Exit and budget experiments.

---

## Planned branch directories

Create these directories only when their corresponding experiments begin:

```text
docs/tables/active_budget_anytime_exit_v0.1/
├── full_depth_baselines/          # complete
├── standard_early_exit/           # next
├── true_staged_inference/         # planned
├── budget_aware_early_exit/       # planned
└── anytime_inference/             # planned
```

Recommended purpose:

| Planned folder | Intended contents |
|---|---|
| `standard_early_exit/` | Policy configurations, per-exit results, exit distributions, average depth, quality-retention tables |
| `true_staged_inference/` | Runtime implementation records, FLOPs, latency, simulated-versus-realised savings |
| `budget_aware_early_exit/` | Budget definitions, controller settings, quality–budget results, violation analysis |
| `anytime_inference/` | Per-budget predictions, quality-versus-cost curves, anytime summaries |

Recommended future script root:

```text
scripts/active_budget_anytime_exit_v0.1/
```

This keeps new efficiency work separate from historical v0.10 evaluator scripts.

---

## Required files for each future experiment

Every completed experiment folder should contain:

| File | Purpose |
|---|---|
| `README.md` | Experiment objective, status, and principal conclusion |
| `EXPERIMENT_SETUP.md` | Data, model, policy, thresholds, cost model, and environment |
| `RESULTS_AND_ANALYSIS.md` | Main table, per-label analysis, quality–cost interpretation |
| `REPRODUCE.ps1` | One-command deterministic or best-effort reproduction |
| `results_summary.csv` | Machine-readable headline metrics |
| `per_label_metrics.csv` | Label-level performance |
| `policy_config.json` | Exact exit-policy settings |
| `environment_summary.txt` | Git and software snapshot |
| `artifact_hashes.csv` | Integrity hashes |

For true staged or budget-aware inference, also include:

| File | Purpose |
|---|---|
| `exit_assignments.csv` | Exit chosen for every sample or parent |
| `exit_distribution.csv` | Fraction using each exit |
| `compute_profile.csv` | Cost/FLOPs by exit |
| `latency_profile.csv` | Measured runtime |
| `quality_cost_curve.csv` | Quality at each budget or cost level |

---

## Documentation rules

1. The canonical baseline must always be identified as:

   ```text
   v0.10 no-hint + frozen historical LATS-v2
   ```

2. Future comparison tables must use:

   ```text
   Macro-F1    = 0.8623815322333925
   Micro-F1    = 0.9531311539976368
   Samples-F1  = 0.9588894381281925
   Exact Match = 0.8765859284890427
   Hamming     = 0.013725490196078431
   ```

3. `1.4590542099` means average predicted labels, not average exit depth.

4. The secondary direct coordinate result is an ablation, not the baseline.

5. Simulated exit-policy evaluation and true staged inference must be reported separately.

6. Full-depth, standard Early-Exit, budget-aware Early-Exit, and anytime inference must each have distinct result folders.

7. Historical v0.10 negative results must remain available for research transparency.

---

## Reproduction entry point

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass `
  -File "docs\tables\active_budget_anytime_exit_v0.1\full_depth_baselines\primary_v010_no_hint_historical_lats_v2\REPRODUCE_PRIMARY.ps1"
```

Successful replay must return:

```text
Macro-F1        = 0.8623815322333925
Micro-F1        = 0.9531311539976368
Samples-F1      = 0.9588894381281925
Exact Match     = 0.8765859284890427
Hamming Loss    = 0.013725490196078431
Avg pred labels = 1.4590542099192618
Parent clips    = 867
```

---

## Current documentation status

| Area | Status |
|---|---|
| Root branch README | Updated |
| Root documentation index | Updated |
| Primary baseline freeze | Complete |
| Secondary result freeze | Complete |
| Exact reproduction | Verified |
| Environment capture | Complete |
| Integrity manifest | Complete |
| Paper-ready baseline summary | Complete |
| Standard Early-Exit documentation | Not yet created |
| Budget-aware documentation | Not yet created |
| Anytime-inference documentation | Not yet created |

---

## Next documentation action

After the first standard Early-Exit experiment, create:

```text
docs/tables/active_budget_anytime_exit_v0.1/standard_early_exit/
```

and record:

- policy definition;
- threshold-selection procedure;
- exit distribution;
- average exit depth;
- quality relative to the canonical full-depth baseline;
- estimated and measured computation saving;
- per-label degradation;
- reproducibility commands and hashes.
