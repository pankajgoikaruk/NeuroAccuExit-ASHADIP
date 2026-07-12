# Documentation Structure — Active Budget and Anytime Exit v0.2

This document indexes the repository artifacts for the NeuroAccuExit computation-adaptive inference phase.

---

## Active branch

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.2` |
| Current completed milestone | `v0.11_EE` fixed-exit audit and genuine Dynamic Early-Exit |
| Canonical full-depth comparator | v0.10 no-hint + frozen historical LATS-v2 |
| Next main experiment | Budget-aware Early-Exit |
| Later experiment | Anytime inference |

---

## Top-level documentation

| Path | Purpose |
|---|---|
| `README.md` | Authoritative branch summary, headline results, reproduction commands, conclusions, and roadmap |
| `DOC_STRUCTURE.md` | This documentation and artifact index |
| `docs/active_budget_anytime_exit_v0.2/README.md` | Detailed implementation and experiment overview |
| `docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/` | Frozen canonical and secondary full-depth baseline package |
| `docs/tables/active_budget_anytime_exit_v0.2/` | Compact v0.2 result tables and paper-ready experiment records |
| `docs/v0.10/` | Historical hint-pass, weighting, calibration, and negative-result documentation |
| `docs/v0.10_1/` | Historical low-energy recovery ablation documentation |
| `docs/archive/` | Archived repository documentation states |

Historical records must remain available and must not be silently rewritten.

---

## v0.11 implementation structure

```text
models/
└── anytime_exit_net.py

tests/
├── __init__.py
└── test_anytime_exit_net.py

scripts/v0.11_EE/
├── fixed_policy/
│   ├── verify_checkpoint_equivalence_v011.py
│   ├── evaluate_fixed_exits_v011.py
│   └── run_v011_EE.ps1
└── dynamic_policy/
    ├── tune_dynamic_policy_v011.py
    ├── evaluate_dynamic_early_exit_v011.py
    └── run_dynamic_v011_EE.ps1
```

| File | Purpose |
|---|---|
| `models/anytime_exit_net.py` | Inference-only staged wrapper that executes only the blocks required to reach the next exit |
| `tests/test_anytime_exit_net.py` | Three-exit, five-exit, hint-compatible, and state-progression equivalence tests |
| `verify_checkpoint_equivalence_v011.py` | Verifies staged/full logits and probabilities on the real canonical checkpoint |
| `evaluate_fixed_exits_v011.py` | Evaluates Always Exit 1/2/3 at segment and parent level |
| `run_v011_EE.ps1` | One-command fixed-exit audit |
| `tune_dynamic_policy_v011.py` | Validation-only grid search and frozen-policy generation |
| `evaluate_dynamic_early_exit_v011.py` | Genuine active-batch staged evaluation with Blocks 4–5 skipped for Exit-2 samples |
| `run_dynamic_v011_EE.ps1` | Prechecks, validation tuning, policy freezing, and corrected-holdout evaluation |

---

## Branch documentation root

```text
docs/tables/active_budget_anytime_exit_v0.2/
```

Expected structure after the v0.11 documentation freeze:

```text
docs/tables/active_budget_anytime_exit_v0.2/
├── README.md
└── v0.11_EE/
    ├── README.md
    ├── EXPERIMENT_SETUP.md
    ├── RESULTS_AND_ANALYSIS.md
    ├── PAPER_READY_SUMMARY.md
    ├── PS_COMMANDS.md
    ├── REPRODUCE_V011_EE.ps1
    ├── experiment_manifest.json
    ├── checkpoint_staged_equivalence.json
    ├── fixed_exit_segment_summary.csv
    ├── fixed_exit_parent_summary.csv
    ├── fixed_exit_parent_per_label.csv
    ├── validation_policy_selection.csv
    ├── dynamic_exit_summary.csv
    └── cumulative_comparison.csv
```

---

## Artifact roles

### Human-readable records

| File | Purpose |
|---|---|
| `v0.11_EE/README.md` | Experiment status, headline conclusions, and package index |
| `EXPERIMENT_SETUP.md` | Model, data, thresholds, policy, cost model, and evaluation protocol |
| `RESULTS_AND_ANALYSIS.md` | Fixed-exit and dynamic results, quality–cost interpretation, per-label findings, and limitations |
| `PAPER_READY_SUMMARY.md` | Reusable method, result, contribution, limitation, and caption wording |
| `PS_COMMANDS.md` | Windows PowerShell commands for each experiment mode |
| `REPRODUCE_V011_EE.ps1` | Runs the fixed and dynamic experiment runners from the repository root |

### Machine-readable records

| File | Purpose |
|---|---|
| `experiment_manifest.json` | Branch, model, policy, canonical baseline, result, and limitation metadata |
| `checkpoint_staged_equivalence.json` | Exact staged/full checkpoint-equivalence report |
| `fixed_exit_segment_summary.csv` | Segment quality at Always Exit 1/2/3 |
| `fixed_exit_parent_summary.csv` | Parent quality under frozen LATS-v2 transfer |
| `fixed_exit_parent_per_label.csv` | Label-level parent precision, recall, F1, and error counts at every exit |
| `validation_policy_selection.csv` | Frozen validation-selected stopping policy and constraint outcome |
| `dynamic_exit_summary.csv` | Exit distribution, depth, compute, latency, and holdout metrics |
| `cumulative_comparison.csv` | Canonical, fixed-exit, and dynamic headline comparison |

---

## Non-committed runtime outputs

Large data and prediction artifacts remain under:

```text
human_talk_workspace/active_budget_anytime_exit_v0.2/v0.11_EE/
```

Typical contents include:

- full segment probability exports;
- segment-level prediction CSVs;
- parent-level truth, score, and prediction tables;
- validation sweep tables;
- frozen runtime policy JSON;
- corrected-holdout dynamic predictions.

These files can be regenerated locally and are intentionally not copied wholesale into Git. Compact summaries are committed under `docs/tables`.

---

## Canonical result

```text
v0.10 no-hint + frozen historical LATS-v2
```

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8623815322333925 |
| Micro-F1 | 0.9531311539976368 |
| Samples-F1 | 0.9588894381281924 |
| Exact Match | 0.8765859284890427 |
| Hamming Loss ↓ | 0.0137254901960784 |
| Average exit depth | 3.0 |
| Estimated compute saved | 0% |

This is the sole full-depth quality reference for v0.11 and all later budget/anytime experiments.

---

## v0.11 result snapshot

### Fixed exits, parent level

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|
| Always Exit 1 | 0.1626 | 0.3877 | 0.2902 | 0.0577 | 0.1355 |
| Always Exit 2 | 0.6923 | 0.7760 | 0.7652 | 0.5156 | 0.0655 |
| Always Exit 3 | 0.8624 | 0.9531 | 0.9589 | 0.8766 | 0.0137 |

Exit 1 and Exit 2 use transferred historical LATS-v2 rules and are diagnostic.

### Genuine Dynamic Exit 2/3

| Item | Value |
|---|---:|
| Exit-2 samples | 508 / 4,335 |
| Exit-2 fraction | 11.72% |
| Average exit depth | 2.8828 |
| Estimated FLOPs saved | 7.53% |
| Parent Macro-F1 | 0.842248 |
| Parent Micro-F1 | 0.935484 |
| Parent Samples-F1 | 0.943577 |
| Parent Exact Match | 0.838524 |
| Parent Hamming Loss | 0.018916 |

---

## Reproduction entry points

Fixed-exit audit:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\fixed_policy\run_v011_EE.ps1"
```

Genuine Dynamic Early-Exit:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1"
```

Combined documentation entry point:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\docs\tables\active_budget_anytime_exit_v0.2\v0.11_EE\REPRODUCE_V011_EE.ps1"
```

---

## Documentation rules

1. Always identify the canonical baseline as `v0.10 no-hint + frozen historical LATS-v2`.
2. Do not confuse average predicted labels with average exit depth.
3. Report post-hoc policy simulations separately from genuine staged inference.
4. State that Exit-1/Exit-2 parent results are frozen-policy transfer diagnostics.
5. Record whether thresholds were selected on validation, calibration, or holdout data.
6. Do not tune a policy after inspecting its corrected-holdout result.
7. Distinguish architecture-estimated FLOP saving from measured latency saving.
8. Do not claim measured speedup until a same-protocol Always Exit 3 timing baseline exists.
9. Preserve per-label results because rare/context labels can be harmed despite strong Micro-F1.
10. Keep standard, budget-aware, and anytime results in separate experiment folders.

---

## Current documentation status

| Area | Status |
|---|---|
| Root v0.2 README | Updated for v0.11 |
| Root documentation index | Updated for v0.11 |
| Frozen full-depth baseline package | Complete |
| Staged inference implementation record | Complete |
| Checkpoint equivalence record | Complete |
| Fixed-exit tables | Complete |
| Dynamic-policy validation record | Complete |
| Genuine staged holdout summary | Complete |
| Paper-ready v0.11 wording | Complete |
| Budget-aware documentation | Not yet created |
| Anytime-inference documentation | Not yet created |

---

## Next documentation directories

Create only when their experiments begin:

```text
docs/tables/active_budget_anytime_exit_v0.2/
├── v0.11_EE/                   # complete
├── v0.12_budget_aware_EE/      # next
└── v0.13_anytime_inference/    # planned
```

The v0.12 package should add explicit budget definitions, budget-forced exit reasons, cost constraints, same-protocol latency baselines, and quality-versus-budget tables.
