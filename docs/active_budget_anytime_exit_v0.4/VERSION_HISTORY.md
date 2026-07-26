# Active Budget and Anytime Exit — Version History

Parent-level metrics use the frozen LATS-v2 evaluator unless explicitly marked as validation or segment-level metrics.

## Summary traceability

| Version | Implementation | Core research question | Main confirmed finding |
|---|---|---|---|
| v0.11 | Staged wrapper + global Exit-2/3 policy | Can the checkpoint skip later blocks without changing logits? | Yes, but the first policy loses too much quality. |
| v0.12 | Validation-derived label risk | Should difficult labels receive continuation protection? | Slight gain over v0.11; risk alone is insufficient. |
| v0.13 | Matched rules and logistic gate | Which matched stopping strategy is strongest? | Per-label margins are the reliable adaptive baseline. |
| v0.14 | Parent-aware counterfactual gate | Can learned parent-harm targets improve safety? | Exit-2 gate unsafe; Exit-1 coverage too small. |
| v0.15 | Whole-parent risk control | Does parent-level stopping fix joint aggregation mismatch? | Quality preserved, but efficiency negligible. |
| v0.16 | NSGA-II Exit-2 margin optimisation | Can multi-objective search improve the frontier? | Real speedup, but all holdout limits failed. |
| v0.17 | Fully sequential 3-/5-exit policy | Can every exit participate? | Strong non-fair 5-exit result; fairness unresolved. |
| v0.18 | Fair five-exit retraining + strict risk-veto policy | Does the result reproduce fairly, and can stronger safeguards transfer? | Fairness solved; full policies unsafe; No-Exit-1 nearly feasible. |

## v0.11_EE — staged and global Dynamic Early-Exit

- **Implementation:** `models/anytime_exit_net.py`, `scripts/v0.11_EE/fixed_policy/`, `scripts/v0.11_EE/dynamic_policy/`.
- **Settings:** 3 exits, taps `(1,3)`, fixed 0.5 segment threshold, validation confidence 0.55.
- **Holdout:** 11.72% stopped at Exit 2; 7.53% estimated FLOPs saved; Macro-F1 0.842248; Micro-F1 0.935484; Exact 0.838524; Hamming 0.018916.
- **Finding:** Genuine skipping was established, but the rule was too permissive.

## v0.12_EE — validation-derived label risk

- **Implementation:** `policies/label_aware_early_exit_policy.py`, `scripts/v0.12_EE/label_aware_policy/`.
- **Settings:** risk from normalized per-label `F1_exit3 − F1_exit2`; risk threshold 0.50.
- **Holdout:** 7.19% FLOPs saved; Macro-F1 0.843703; Micro-F1 0.936689; Exact 0.840830; Hamming 0.018570.
- **Finding:** Small improvement, but no new robust frontier.

## v0.13_EE — matched policies and learned gate

- **Implementation:** `policies/early_exit_strategy_comparison.py`, `scripts/v0.13_EE/matched_policy_comparison/`.
- **Settings:** 70/30 parent-disjoint derivation/selection split; Macro-F1-drop limit 0.01; minimum Exit-2 rate 0.02.

| Strategy | FLOPs saved | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|
| Per-label margin | 1.44% | 0.858748 | 0.951556 | 0.874279 | 0.014187 |
| Logistic gate | 11.30% | 0.833034 | 0.943529 | 0.855825 | 0.016609 |

- **Finding:** Per-label margins became the quality-constrained baseline.

## v0.14_EE — parent-aware counterfactual gates

- **Implementation:** `policies/parent_aware_adaptive_gate.py`, `scripts/v0.14_EE/parent_aware_gate/`.
- **Settings:** five parent-grouped folds, per-label parent-harm logistic models, 30-repeat timing.
- **Results:** Exit-2 gate saved 13.05% but Macro-F1 fell to 0.840798; Exit-1 ablation saved 0.69% and preserved quality more closely.
- **Finding:** Counterfactual local safety did not model joint parent effects sufficiently.

## v0.15_EE — whole-parent selective risk control

- **Implementation:** `policies/whole_parent_selective_exit.py`, `scripts/v0.15_EE/whole_parent_risk_control/`.
- **Settings:** five parent folds; nonparametric and shared-logistic controllers; Macro/Micro drop limits 0.005.
- **Results:** Nonparametric risk saved 0.44% FLOPs; shared logistic saved 0%.
- **Finding:** Parent-level target alignment improved, but validation size limited useful coverage.

## v0.16_EE — multi-objective per-label margin optimisation

- **Implementation:** `policies/multiobjective_per_label_margin.py`, `scripts/v0.16_EE/multiobjective_per_label_margin/`.
- **Settings:** population 80, 50 generations, 4,078 candidates, 20 Pareto points, 30-repeat CPU timing.
- **Holdout:** 5.06% FLOPs saved, 1.015× speedup, Macro-F1 0.849203, Micro-F1 0.942474, Exact 0.854671, Hamming 0.016840.
- **Finding:** Real acceleration, but every holdout-quality limit failed.

## v0.17_EE — fully sequential anytime inference

- **Implementation:** `policies/sequential_anytime_exit.py`, `scripts/v0.17_EE/sequential_anytime_exit/`.
- **Settings:** population 96, 60 generations, safety fraction 0.75, 30-repeat timing.
- **3-exit:** 8.64% FLOPs, 1.037× speedup, Macro-F1 0.840128; holdout constraints failed.
- **5-exit:** 30.71% FLOPs, 1.114× speedup, Macro-F1 0.801356; within-checkpoint limits passed.
- **Finding:** Every exit can participate, but the 3-/5-exit training manifests differed; architecture superiority was not established.

## v0.18_EE — fair strict sequential anytime inference

### Implementation

- `policies/strict_sequential_anytime_exit_v018.py`
- `scripts/v0.18_EE/fair_sequential_anytime_exit/`
- `tests/test_strict_sequential_anytime_exit_v018.py`

### Research questions

1. Can a fair 5-exit checkpoint be trained with the canonical 3-exit data and optimisation settings?
2. Does v0.17's strong five-exit result reproduce under matched training?
3. Can an Exit-1 confidence boost and high-risk uncertainty veto improve safety?
4. Does the redesigned continuation-risk term become active?
5. Which ablations are responsible for quality preservation?
6. Can either full sequential policy meet all holdout limits?

### Training settings

| Setting | 3-exit | 5-exit |
|---|---:|---:|
| Rows train/val/test | 25,519 / 1,883 / 1,961 | 25,519 / 1,883 / 1,961 |
| Tap blocks | `1,3` | `1,2,3,4` |
| Epochs | 40 | 40 |
| Batch | 64 | 64 |
| LR | 0.001 | 0.001 |
| Seed | 42 | 42 |
| Hint | Off | Off |
| Loss weights | `0.3,0.3,1.0` | `0.15,0.15,0.15,0.15,1.0` |
| Auxiliary budget | 0.60 | 0.60 |

The fairness audit passed all checks.

### Policy settings

- population 112;
- 70 generations;
- seed 42;
- five parent-grouped folds;
- safety fraction 0.50;
- minimum early fraction 0.02;
- minimum Exit-1 fraction 0.0025;
- fixed 0.5 segment thresholds;
- 30-repeat CPU timing.

### Corrected-holdout results

| Architecture | Policy | FLOPs | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 3-exit | `full_strict` | 3.82% | 1.018× | 0.85285 | 0.94530 | 0.86159 | 0.01603 | Fails 3/4 |
| 5-exit | `full_strict` | 12.70% | 1.057× | 0.79813 | 0.89832 | 0.75779 | 0.03010 | Fails 4/4 |
| 5-exit | `no_exit1` | 9.18% | 1.037× | 0.81015 | 0.90375 | 0.77163 | 0.02872 | Fails Macro only |

### Confirmed findings

- Fair architecture comparison is now valid.
- Five exits offer greater compute-saving capacity.
- The strong v0.17 fair-quality result did not reproduce under matched training.
- Exit 1 remains the riskiest stage.
- The redesigned risk veto is active and protects quality.
- Label margins and stability remain essential.
- Validation-to-holdout transfer remains the main bottleneck.
- `Exit 3 → Exit 5` is the closest current operating point.

### Final verdict

v0.18 is a successful methodological study but not a successful deployment policy. Its main contribution is the fair comparison, the active continuation-risk mechanism, and the identification of a nearly feasible five-exit No-Exit-1 route.
