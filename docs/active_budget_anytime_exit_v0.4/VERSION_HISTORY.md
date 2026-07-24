# Active Budget and Anytime Exit — Version History

This record maps each Early-Exit experiment to its implementation, settings, research question, confirmed result, and interpretation. All parent metrics use the canonical frozen LATS-v2 evaluation unless explicitly identified as a validation result.

## v0.11_EE — staged and global Dynamic Early-Exit

**Implementation:** `models/anytime_exit_net.py`, `scripts/v0.11_EE/fixed_policy/`, and `scripts/v0.11_EE/dynamic_policy/`.

**Research questions:** Can the trained checkpoint be executed incrementally without changing logits? What quality is available at each fixed exit? Can selected samples skip Blocks 4–5?

**Settings:** three exits; taps `(1,3)`; fixed 0.5 segment thresholds; validation-selected confidence `0.55`; Exit-1/Exit-2 set agreement required; Exit 2 or Exit 3 only.

**Confirmed holdout:** 11.72% stopped at Exit 2; average depth 2.8828; 7.53% estimated FLOPs saved; Macro-F1 0.842248; Micro-F1 0.935484; Exact Match 0.838524; Hamming 0.018916.

**Finding:** Genuine compute skipping was proven, but the first rule was too permissive.

## v0.12_EE — validation-derived label risk

**Implementation:** `policies/label_aware_early_exit_policy.py` and `scripts/v0.12_EE/label_aware_policy/`.

**Research question:** Should labels that improve more from Exit 2 to Exit 3 receive greater continuation protection?

**Settings:** risk weight derived from normalized per-label `F1_exit3 − F1_exit2`; selected risk threshold 0.50; confidence 0.55; fixed 0.5 segment decisions.

**Confirmed holdout:** 11.19% Exit-2 fraction; 7.19% estimated FLOPs saved; Macro-F1 0.843703; Micro-F1 0.936689; Exact Match 0.840830; Hamming 0.018570.

**Finding:** Small improvement over v0.11, but later matched analysis showed that the label-risk condition was not sufficient evidence of a superior frontier.

## v0.13_EE — matched policies and learned gate

**Implementation:** `policies/early_exit_strategy_comparison.py` and `scripts/v0.13_EE/matched_policy_comparison/`.

**Research question:** Under matched constraints, which strategy is strongest: global confidence/margin, probability delta, label risk, direct per-label margins, or a logistic gate?

**Settings:** 70% parent-disjoint derivation subset and 30% selection subset; Macro-F1 drop limit 0.01; minimum Exit-2 rate 0.02; same corrected holdout for frozen evaluation.

**Key holdout ablations:**

| Strategy | FLOPs saved | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|
| Global confidence + margin | 0.76% | 0.861433 | 0.952719 | 0.875433 | 0.013841 |
| Global + delta | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| Label risk | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| Per-label margin | 1.44% | 0.858748 | 0.951556 | 0.874279 | 0.014187 |
| Logistic gate | 11.30% | 0.833034 | 0.943529 | 0.855825 | 0.016609 |

**Finding:** Direct per-label margins became the best reliable adaptive baseline; the logistic gate found larger coverage but caused unacceptable quality loss.

## v0.14_EE — parent-aware counterfactual gates

**Implementation:** `policies/parent_aware_adaptive_gate.py` and `scripts/v0.14_EE/parent_aware_gate/`.

**Research questions:** Can parent-aware counterfactual targets predict whether Exit 3 improves a segment? Can Exit 1 safely terminate some samples?

**Settings:** five parent-grouped folds; per-label parent-harm logistic models; robust Macro-F1-drop limit 0.01; 30-repeat controlled timing.

**Results:** Exit-2 gate saved 13.05% estimated FLOPs but Macro-F1 fell to 0.840798 and speedup was 0.985837×. Exit-1 ablation preserved quality closely but saved only 0.69% and speedup was 0.958221×.

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

**Settings:** population 80; 50 generations; 4,078 unique candidates; 20 Pareto candidates; five parent-grouped folds; 12 policy genes; fixed 0.5 segment thresholds; 30-repeat controlled CPU timing.

**Validation:** selected point was feasible and predicted 12.63% FLOP saving.

**Holdout:** 7.87% Exit-2 fraction; 5.06% FLOPs saved; 1.015× speedup; Macro-F1 0.849203; Micro-F1 0.942474; Exact Match 0.854671; Hamming 0.016840.

**Finding:** v0.16 is the first experiment to combine meaningful estimated saving with a small controlled speedup, but it failed every predefined holdout quality constraint. It expands the experimental Pareto region but does not replace v0.13.
