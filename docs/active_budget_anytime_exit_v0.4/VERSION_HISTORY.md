# Active Budget and Anytime Exit — Version History

This record maps every Early-Exit experiment to its implementation, settings, research questions, confirmed outputs, and interpretation. Parent metrics use frozen LATS-v2 evaluation unless stated otherwise.

## v0.11_EE — staged and global Dynamic Early Exit

**Implementation:** `models/anytime_exit_net.py`, `scripts/v0.11_EE/fixed_policy/`, and `scripts/v0.11_EE/dynamic_policy/`.

**Research questions:** Can the trained checkpoint execute incrementally without changing logits? What quality is available at each fixed exit? Can selected samples skip Blocks 4–5?

**Settings:** three exits; taps `(1,3)`; fixed 0.5 segment thresholds; validation-selected confidence `0.55`; Exit-1/Exit-2 set agreement required; stopping decision at Exit 2 or final Exit 3.

**Confirmed holdout:** 11.72% stopped at Exit 2; average depth 2.8828; 7.53% estimated FLOPs saved; Macro-F1 0.842248; Micro-F1 0.935484; Exact Match 0.838524; Hamming 0.018916.

**Finding:** Genuine compute skipping was proven, but the first global rule was too permissive.

## v0.12_EE — validation-derived label risk

**Implementation:** `policies/label_aware_early_exit_policy.py` and `scripts/v0.12_EE/label_aware_policy/`.

**Research question:** Should labels that improve more from Exit 2 to Exit 3 receive greater continuation protection?

**Settings:** risk weight from normalized `F1_exit3 − F1_exit2`; selected risk threshold 0.50; confidence 0.55; fixed 0.5 segment decisions.

**Confirmed holdout:** 11.19% Exit-2 fraction; 7.19% estimated FLOPs saved; Macro-F1 0.843703; Micro-F1 0.936689; Exact Match 0.840830; Hamming 0.018570.

**Finding:** Small improvement over v0.11, but matched analysis later showed the label-risk condition did not create a distinct superior frontier.

## v0.13_EE — matched policies and learned gate

**Implementation:** `policies/early_exit_strategy_comparison.py` and `scripts/v0.13_EE/matched_policy_comparison/`.

**Research question:** Under matched constraints, which strategy is strongest: global confidence/margin, probability delta, label risk, direct per-label margins, or a logistic gate?

**Settings:** 70% parent-disjoint derivation subset and 30% selection subset; Macro-F1 drop limit 0.01; minimum Exit-2 rate 0.02; same frozen corrected holdout.

| Strategy | FLOPs saved | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|
| Global confidence + margin | 0.76% | 0.861433 | 0.952719 | 0.875433 | 0.013841 |
| Global + delta | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| Label risk | 1.56% | 0.858556 | 0.950845 | 0.869666 | 0.014418 |
| Per-label margin | 1.44% | 0.858748 | 0.951556 | 0.874279 | 0.014187 |
| Logistic gate | 11.30% | 0.833034 | 0.943529 | 0.855825 | 0.016609 |

**Finding:** Per-label margins became the strongest reliable three-exit adaptive baseline; the logistic gate found larger coverage but unacceptable quality loss.

## v0.14_EE — parent-aware counterfactual gates

**Implementation:** `policies/parent_aware_adaptive_gate.py` and `scripts/v0.14_EE/parent__aware_gate/`.

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

**Settings:** population 80; 50 generations; 4,078 unique candidates; 20 Pareto candidates; five parent-grouped folds; 12 policy genes; fixed 0.5 thresholds; 30-repeat controlled CPU timing.

**Validation:** selected point predicted 12.63% FLOP saving and satisfied validation limits.

**Holdout:** 7.87% Exit-2 fraction; 5.06% FLOPs saved; 1.015× speedup; Macro-F1 0.849203; Micro-F1 0.942474; Exact Match 0.854671; Hamming 0.016840.

**Finding:** v0.16 was the first to combine meaningful estimated saving with a small measured speedup, but it failed every predefined holdout quality limit.

## v0.17_EE — fully sequential active-budget anytime exit

**Implementation:** `policies/sequential_anytime_exit.py`, `policies/sequential_anytime_exit_optim.py`, `scripts/v0.17_EE/sequential_anytime_exit/`, and `tests/test_sequential_anytime_exit.py`.

### Research questions

1. Can the policy use the complete route `Exit 1 → Exit 2 → Exit 3` rather than treating the task as Exit-2 versus Exit-3 only?
2. Can the same optimiser support `Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5`?
3. Do more sequential decision opportunities produce a better quality–compute trade-off?
4. Is Exit 1 beneficial or too risky?
5. Which safety conditions are essential: confidence, label margins, stability, delta, or risk?
6. Is the available three-exit/five-exit comparison fair?

### Settings

| Setting | Value |
|---|---:|
| Population / generations | 96 / 60 |
| Random seed | 42 |
| Safety-buffered Pareto ratio | 0.75 |
| Minimum total early-exit fraction | 0.02 |
| Minimum Exit-1 fraction | 0.005 |
| Validation segments / parents | 1,883 / 304 |
| Holdout segments / parents | 4,335 / 867 |
| Three-exit candidates / Pareto points | 5,847 / 19 |
| Five-exit candidates / Pareto points | 5,856 / 86 |
| Segment threshold mode | fixed 0.5 |
| Parent Macro-F1 drop limit | 0.010 |
| Parent Micro-F1 drop limit | 0.005 |
| Exact-Match drop limit | 0.010 |
| Hamming increase limit | 0.002 |
| Publication timing | 30 CPU repetitions |

### Three-exit confirmed holdout

| Method | Exit 1 | Exit 2 | Exit 3 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | 0.00% | 0.00% | 100.00% | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.876586 | 0.013725 |
| Full sequential | 6.07% | 4.34% | 89.60% | 8.64% | 1.037× | 0.840128 | 0.937549 | 0.840830 | 0.018224 |
| No Exit 1 | 0.00% | 5.28% | 94.72% | 3.39% | 1.022× | 0.854086 | 0.946871 | 0.866205 | 0.015571 |

**Finding:** genuine sequential routing and speedup were achieved, but the full policy failed all holdout quality limits. Exit 1 contributed most of the additional saving and much of the quality loss.

### Five-exit confirmed holdout

| Method | Exit 1 | Exit 2 | Exit 3 | Exit 4 | Exit 5 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 5 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 1.000× | 0.810761 | 0.869498 | 0.673587 | 0.038985 |
| Full sequential | 6.83% | 1.22% | 18.59% | 26.30% | 47.06% | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.688581 | 0.039100 |
| No Exit 1 | 0.00% | — | — | — | — | 26.80% | 1.096× | 0.809541 | 0.870906 | 0.687428 | approximately preserved |

**Holdout audit:** Macro-F1 drop 0.009406; Micro-F1 drop 0.000639; Exact Match improved by 0.014994; Hamming increased by 0.000115. All predefined limits were met.

### Ablation findings

- **No Exit 1:** quality improves, confirming Exit 1 is useful for compute but risky.
- **No stability:** slightly more saving but worse quality, validating label-set stability.
- **No risk:** identical or nearly identical outcome, showing the current risk term is weakly active.
- **No label margins:** large compute gain with severe quality collapse.
- **Confidence only:** largest compute gain and worst multi-label consistency.

### Per-label findings

The five-exit policy improved `silence_present`, `music_present`, `Eckhart_Tolle`, and `Jay_Shetty`, but degraded `Nick_Vujicic`, `audience_reaction_present`, and `Eric_Thomas`. The three-exit model also exposed `audience_reaction_present`, `Eric_Thomas`, and `other_speaker_present` as recurring risk labels.

### Fairness limitation

The generated fairness audit failed because the three-exit and five-exit checkpoints use different training manifests and training-set sizes. The five-exit result is therefore valid relative to its own Always Exit 5 baseline, but it does not prove architectural superiority over the canonical three-exit model.

### Final finding

v0.17 is the first study in this branch to show a substantial within-model anytime result: the five-exit policy saved 30.71% estimated FLOPs, achieved 1.114× CPU speedup, and met all holdout quality limits. The corresponding three-exit policy remains an unsuccessful quality-transfer result.
