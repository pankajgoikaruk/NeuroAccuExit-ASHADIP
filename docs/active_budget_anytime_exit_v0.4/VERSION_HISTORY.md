# Active Budget and Anytime Exit — Version History

This record maps each Early-Exit experiment to its implementation, settings, research question, confirmed result, and interpretation. Parent metrics use the frozen LATS-v2 evaluation unless explicitly identified as validation results.

## v0.11_EE — staged and global Dynamic Early-Exit

**Implementation:** `models/anytime_exit_net.py`, `scripts/v0.11_EE/fixed_policy/`, and `scripts/v0.11_EE/dynamic_policy/`.

**Research questions:** Can the trained checkpoint execute incrementally without changing logits? What quality is available at each fixed exit? Can selected samples skip Blocks 4–5?

**Settings:** three exits; taps `(1,3)`; fixed 0.5 segment thresholds; validation-selected confidence `0.55`; Exit-1/Exit-2 set agreement required; Exit 2 or Exit 3 only.

**Confirmed holdout:** 11.72% stopped at Exit 2; average depth 2.8828; 7.53% estimated FLOPs saved; Macro-F1 0.842248; Micro-F1 0.935484; Exact Match 0.838524; Hamming 0.018916.

**Finding:** Genuine compute skipping was proven, but the first rule was too permissive.

## v0.12_EE — validation-derived label risk

**Implementation:** `policies/label_aware_early_exit_policy.py` and `scripts/v0.12_EE/label_aware_policy/`.

**Research question:** Should labels that improve more from Exit 2 to Exit 3 receive greater continuation protection?

**Settings:** risk weight derived from normalized per-label `F1_exit3 − F1_exit2`; selected risk threshold 0.50; confidence 0.55; fixed 0.5 segment decisions.

**Confirmed holdout:** 11.19% Exit-2 fraction; 7.19% estimated FLOPs saved; Macro-F1 0.843703; Micro-F1 0.936689; Exact Match 0.840830; Hamming 0.018570.

**Finding:** Small improvement over v0.11, but later matched analysis showed that the risk condition did not create a superior frontier.

## v0.13_EE — matched policies and learned gate

**Implementation:** `policies/early_exit_strategy_comparison.py` and `scripts/v0.13_EE/matched_policy_comparison/`.

**Research question:** Under matched constraints, which strategy is strongest: global confidence/margin, probability delta, label risk, per-label margins, or a logistic gate?

**Settings:** 70% parent-disjoint derivation subset and 30% selection subset; Macro-F1-drop limit 0.01; minimum Exit-2 rate 0.02; frozen corrected-holdout evaluation.

**Key holdout ablations:**

| Strategy | FLOPs saved | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|
| Global confidence + margin | 0.76% | 0.861433 | 0.952719 | 0.875433 | 0.013841 |
| Global + delta | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| Label risk | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| Per-label margin | 1.44% | 0.858748 | 0.951556 | 0.874279 | 0.014187 |
| Logistic gate | 11.30% | 0.833034 | 0.943529 | 0.855825 | 0.016609 |

**Finding:** Per-label margins became the best reliable adaptive baseline; the logistic gate found greater coverage but caused unacceptable quality loss.

## v0.14_EE — parent-aware counterfactual gates

**Implementation:** `policies/parent_aware_adaptive_gate.py` and `scripts/v0.14_EE/parent_aware_gate/`.

**Research questions:** Can parent-aware counterfactual targets predict whether Exit 3 improves a segment? Can Exit 1 safely terminate samples?

**Settings:** five parent-grouped folds; per-label parent-harm logistic models; robust Macro-F1-drop limit 0.01; 30-repeat timing.

**Results:** Exit-2 gate saved 13.05% estimated FLOPs but Macro-F1 fell to 0.840798 and speedup was 0.985837×. Exit-1 ablation preserved quality closely but saved 0.69% and speedup was 0.958221×.

**Finding:** Individual counterfactual safety did not model joint parent aggregation effects; neither policy was deployable.

## v0.15_EE — whole-parent selective risk control

**Implementation:** `policies/whole_parent_selective_exit.py` and `scripts/v0.15_EE/whole_parent_risk_control/`.

**Research question:** Does making one complete-parent Exit-2/Exit-3 decision eliminate the joint-substitution mismatch?

**Settings:** five parent-level folds; nonparametric risk calibrator and shared logistic parent-label gate; Macro/Micro drop limits 0.005; overall harm limit 0.01.

**Results:** Nonparametric risk preserved quality but stopped 0.692% of parents, saved 0.445% FLOPs, and ran at 0.934×. Shared logistic stopped none and ran at 0.923×.

**Finding:** The target/runtime unit was corrected, but 304 validation parents were insufficient for useful risk-controlled coverage.

## v0.16_EE — multi-objective per-label margin optimisation

**Implementation:** `policies/multiobjective_per_label_margin.py` and `scripts/v0.16_EE/multiobjective_per_label_margin/`.

**Research question:** Can an NSGA-II-style search optimise the lightweight v0.13 rule across compute, Macro-F1, Micro-F1, Exact Match, and Hamming simultaneously?

**Settings:** population 80; 50 generations; 4,078 unique candidates; 20 Pareto candidates; five parent-grouped folds; 12 policy genes; fixed 0.5 segment thresholds; 30-repeat CPU timing.

**Validation:** selected point was feasible and predicted 12.63% FLOP saving.

**Holdout:** 7.87% Exit-2 fraction; 5.06% FLOPs saved; 1.015× speedup; Macro-F1 0.849203; Micro-F1 0.942474; Exact Match 0.854671; Hamming 0.016840.

**Finding:** v0.16 combined meaningful estimated saving with a controlled speedup, but failed every predefined holdout-quality limit.

## v0.17_EE — sequential active-budget anytime exit

**Implementation:** `policies/sequential_anytime_exit.py`, `scripts/v0.17_EE/sequential_anytime_exit/`, and `tests/test_sequential_anytime_exit.py`.

**Research questions:**

1. Can every non-final exit make a genuine stop/continue decision?
2. Can easy samples exit at Exit 1, moderate samples at middle exits, and difficult samples at the final exit?
3. Can one multi-objective formulation support both 3-exit and 5-exit checkpoints?
4. Which policy terms are necessary for safe multi-label early exit?
5. Does a safety-buffered Pareto-knee selector improve the quality–compute trade-off?

**Settings:** population 96; 60 generations; seed 42; five parent-grouped folds; safety fraction 0.75; minimum total early fraction 0.02; minimum Exit-1 fraction 0.005; fixed 0.5 segment thresholds; 30-repeat CPU timing. Every early exit owns confidence, probability-delta, risk, and ten label-margin parameters.

### 3-exit confirmed holdout

| Item | Value |
|---|---:|
| Exit distribution | 6.07% / 4.34% / 89.60% |
| Total early fraction | 10.40% |
| Estimated FLOPs saved | 8.64% |
| Speedup | 1.037× |
| Parent Macro-F1 | 0.840128 |
| Parent Micro-F1 | 0.937549 |
| Parent Exact Match | 0.840830 |
| Parent Hamming Loss | 0.018224 |
| Holdout quality constraints | **Failed** |

**Finding:** The full three-exit policy is computationally successful but not quality-safe. Exit 1 provides substantial extra saving but causes disproportionate loss.

### 5-exit confirmed holdout

| Item | Value |
|---|---:|
| Exit distribution | 6.83% / 1.22% / 18.59% / 26.30% / 47.06% |
| Total early fraction | 52.94% |
| Estimated FLOPs saved | 30.71% |
| Speedup | 1.114× |
| Parent Macro-F1 | 0.801356 |
| Parent Micro-F1 | 0.868859 |
| Parent Exact Match | 0.688581 |
| Parent Hamming Loss | 0.039100 |
| Holdout quality constraints | **Passed** |

The Always Exit 5 reference was Macro-F1 0.810761, Micro-F1 0.869498, Exact Match 0.673587, and Hamming 0.038985.

**Finding:** The tested five-exit policy is the major successful v0.17 result. It saves substantial compute and improves Exact Match while remaining within all predefined quality thresholds relative to its own full-depth baseline.

### Ablation interpretation

- `No Exit 1` preserves quality more strongly but saves less compute.
- Removing stability produces extra saving at a quality cost.
- Removing label margins causes severe quality collapse.
- Confidence-only policies are unsafe.
- The current risk term is non-binding.
- Remaining difficult labels include `audience_reaction_present`, `Nick_Vujicic`, `Eric_Thomas`, and `other_speaker_present`.

### Fairness limitation

The 3-exit and 5-exit checkpoints use different training/validation manifests. The architecture fairness audit is therefore false. v0.17 supports a successful within-checkpoint five-exit policy, not a fair proof that five exits outperform three.
