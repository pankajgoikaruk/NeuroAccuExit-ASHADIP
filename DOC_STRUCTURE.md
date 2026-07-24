# Documentation Structure — Active Budget and Anytime Exit v0.3

This file indexes the completed `active_budget_anytime_exit_v0.3` research and repository artifacts. Historical v0.1 and v0.2 records remain preserved.

---

## Active branch

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.3` |
| Completed experiment range | `v0.12_EE`–`v0.15_EE` |
| Canonical comparator | v0.10 no-hint + frozen historical LATS-v2 + Exit 3 |
| Current adaptive recommendation | v0.13 per-label margin |
| Full-quality reference | Always Exit 3 |
| Explicit budget/anytime status | Not implemented in v0.3 |
| Next branch | `active_budget_anytime_exit_v0.4` — intentionally not started here |

---

## Top-level documentation

| Path | Purpose |
|---|---|
| `README.md` | Authoritative branch overview, cumulative results, commands and current decision |
| `DOC_STRUCTURE.md` | This documentation and artifact index |
| `docs/active_budget_anytime_exit_v0.3/README.md` | Detailed research narrative, theoretical explanation, questions and findings |
| `docs/tables/active_budget_anytime_exit_v0.3/README.md` | Compact result-package index |
| `docs/tables/active_budget_anytime_exit_v0.3/CUMULATIVE_RESULTS.md` | Cross-version tables, figures, ablations and selection rationale |
| `docs/tables/active_budget_anytime_exit_v0.3/PS_COMMANDS.md` | Reproduction, retiming and diagnostic PowerShell commands |
| `docs/tables/active_budget_anytime_exit_v0.3/cumulative_results_v012_v015.csv` | Machine-readable headline comparison |
| `docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/` | Frozen full-depth reproducibility package |
| `docs/active_budget_anytime_exit_v0.2/` | Preserved v0.11 staged-inference documentation |
| `docs/tables/active_budget_anytime_exit_v0.2/v0.11_EE/` | Preserved v0.11 compact result package |

Historical documents must not be silently rewritten or removed.

---

## Implementation traceability

### Shared staged inference

```text
models/anytime_exit_net.py
tests/test_anytime_exit_net.py
scripts/v0.11_EE/fixed_policy/
```

The shared wrapper is reused by all later policies and was numerically equivalent to the original full forward path.

### v0.12_EE — validation-derived label risk

```text
policies/label_aware_early_exit_policy.py
tests/test_label_aware_early_exit_policy.py
scripts/v0.12_EE/label_aware_policy/
├── common_v012.py
├── tune_label_aware_policy_v012.py
├── evaluate_label_aware_early_exit_v012.py
└── run_label_aware_v012_EE.ps1
```

Documentation:

```text
docs/tables/active_budget_anytime_exit_v0.3/v0.12_EE/README.md
```

### v0.13_EE — matched strategy comparison

```text
policies/early_exit_strategy_comparison.py
tests/test_early_exit_strategy_comparison.py
scripts/v0.13_EE/matched_policy_comparison/
├── README.md
├── common_v013.py
├── tune_matched_policy_comparison_v013.py
├── evaluate_matched_policy_comparison_v013.py
└── run_matched_policy_comparison_v013_EE.ps1
```

Documentation:

```text
docs/tables/active_budget_anytime_exit_v0.3/v0.13_EE/README.md
```

### v0.14_EE — parent-aware adaptive gates

```text
policies/parent_aware_adaptive_gate.py
tests/test_parent_aware_adaptive_gate.py
scripts/v0.14_EE/parent_aware_gate/
├── README.md
├── common_v014.py
├── tune_parent_aware_gate_v014.py
├── evaluate_parent_aware_gate_v014.py
└── run_parent_aware_gate_v014_EE.ps1
```

Documentation:

```text
docs/tables/active_budget_anytime_exit_v0.3/v0.14_EE/README.md
```

### v0.15_EE — whole-parent selective risk control

```text
policies/whole_parent_selective_exit.py
tests/test_whole_parent_selective_exit.py
scripts/v0.15_EE/whole_parent_risk_control/
├── README.md
├── common_v015.py
├── tune_whole_parent_risk_control_v015.py
├── evaluate_whole_parent_risk_control_v015.py
└── run_whole_parent_risk_control_v015_EE.ps1
```

Documentation:

```text
docs/tables/active_budget_anytime_exit_v0.3/v0.15_EE/README.md
```

---

## Runtime output structure

Large experiment outputs remain local under:

```text
human_talk_workspace/active_budget_anytime_exit_v0.3/
├── v0.12_EE/label_aware_policy/
├── v0.13_EE/matched_policy_comparison/
├── v0.14_EE/parent_aware_gate/
└── v0.15_EE/whole_parent_risk_control/
```

Typical versioned contents:

- checkpoint staged-equivalence JSON;
- validation sweep CSV;
- frozen policy JSON;
- fitted gate model (`.joblib`) where applicable;
- OOF predictions and fold assignments;
- per-segment or per-parent decisions;
- parent truth, scores and predictions;
- per-label metrics;
- holdout comparison CSV/JSON;
- runtime summary JSON.

These large artifacts are not duplicated wholesale in Git. The committed documentation records their paths, selected settings and confirmed summary values.

---

## Experiment and result mapping

| Version | Primary implementation | Validation protocol | Holdout output | Scientific status |
|---|---|---|---|---|
| v0.12 | Label-risk rule | Full validation, max Macro-F1 drop 0.01 | Genuine segment stopping | Feasibility result; quality loss too large |
| v0.13 | Five matched policies | 70% parent derivation / 30% selection | Six same-checkpoint comparisons | Per-label margin selected as current adaptive winner |
| v0.14 | Per-label parent-harm gates | Five parent-grouped OOF folds + quality UCB | Exit-2 primary + Exit-1 ablation | No robust candidate; negative/diagnostic result |
| v0.15 | Whole-parent risk control | Five parent-grouped OOF folds + harm/quality confidence bounds | Nonparametric + shared logistic | Quality preserved but coverage and speed insufficient |

---

## Compact documentation package

```text
docs/tables/active_budget_anytime_exit_v0.3/
├── README.md
├── CUMULATIVE_RESULTS.md
├── PS_COMMANDS.md
├── cumulative_results_v012_v015.csv
├── v0.12_EE/
│   └── README.md
├── v0.13_EE/
│   └── README.md
├── v0.14_EE/
│   └── README.md
└── v0.15_EE/
    └── README.md
```

---

## Documentation rules

1. Always identify the canonical reference as v0.10 no-hint + frozen historical LATS-v2 + Exit 3.
2. Distinguish predicted-label count, exit depth, stop rate, estimated FLOPs and measured latency.
3. State whether the decision unit is a segment or a complete parent.
4. State whether a result met its predefined validation/deployment constraint.
5. Never promote a fallback candidate to a successful policy.
6. Never select or retune an operating point after inspecting corrected-holdout performance.
7. Treat fixed 0.5 segment thresholds and frozen LATS-v2 parent thresholds as different mechanisms.
8. Report Parent Macro-F1 and Parent Micro-F1 together.
9. Preserve Exact Match and Hamming Loss because global consistency can degrade while one F1 metric improves.
10. Distinguish architecture-estimated FLOPs from repeated measured latency.
11. Do not claim measured acceleration when the adaptive controller is slower.
12. Preserve negative results: v0.14 and v0.15 are part of the research trajectory.
13. Do not claim that v0.3 implemented explicit budget-aware or anytime inference.
14. Do not describe the corrected holdout as an independent external test set.
15. Keep confirmed measurements separate from interpretation and proposed future work.

---

## Documentation status

| Area | Status |
|---|---|
| Root branch README | Updated through v0.15 |
| Root documentation index | Updated through v0.15 |
| Detailed v0.3 narrative | Complete |
| v0.12 traceability | Complete |
| v0.13 traceability | Complete |
| v0.14 traceability | Complete |
| v0.15 traceability | Complete |
| Cumulative result table | Complete |
| PowerShell command record | Complete |
| Confirmed vs interpretation separation | Complete |
| Explicit budget-aware controller | Not part of v0.3 |
| Anytime budget sweep | Not part of v0.3 |
| v0.4 implementation | Not started |
