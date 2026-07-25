# v0.16_EE Paper-Ready Summary

## Method paragraph

We formulate Early-Exit policy selection as a constraint-aware multi-objective optimisation problem for a frozen three-exit TinyAudioCNN. The policy terminates a segment at Exit 2 only when the Exit-1 and Exit-2 label sets agree, the Exit-2 prediction is non-empty, mean binary confidence exceeds a global threshold, inter-exit probability change remains below a global limit, and every label is sufficiently far from its decision threshold according to a label-specific margin. A dependency-light NSGA-II-style search jointly maximises estimated computation saving and minimises robust degradation in parent-level Macro-F1, Micro-F1, Exact Match, and Hamming Loss using five parent-grouped validation folds. The selected policy is frozen before genuine staged evaluation, so accepted samples do not execute the final backbone blocks.

## Result paragraph

The optimiser evaluated 4,078 unique policies and retained 20 Pareto candidates. The maximum-saving validation-feasible point predicted a 19.65% Exit-2 rate and 12.63% estimated FLOP saving. On the corrected holdout, the frozen policy stopped 7.87% of 4,335 segments at Exit 2, reduced estimated computation by 5.06%, and achieved a 1.015× median CPU speedup over 30 timing repetitions. Parent Macro-F1, Micro-F1, Exact Match, and Hamming Loss were 0.8492, 0.9425, 0.8547, and 0.0168, respectively, compared with 0.8624, 0.9531, 0.8766, and 0.0137 at full depth.

## Interpretation paragraph

The experiment establishes that multi-objective optimisation can discover lightweight policies with greater genuine compute saving and measurable runtime acceleration than manually derived per-label margins. However, the validation-selected maximum-compute point did not satisfy the predefined holdout quality limits: Macro-F1 and Micro-F1 fell by 0.0132 and 0.0107, Exact Match fell by 0.0219, and Hamming Loss increased by 0.0031. We therefore treat v0.16 as a Pareto and optimisation ablation rather than the selected deployment policy; the more conservative v0.13 per-label margin policy remains the quality-constrained adaptive baseline.

## Contribution wording

> We introduce a constraint-aware multi-objective optimisation framework for selecting label-specific Early-Exit reliability margins in multi-label audio inference. The method jointly models computation saving and multiple parent-level quality criteria and executes the selected policy through genuine staged inference.

This wording claims the implemented contribution without claiming a new state-of-the-art deployment result.

## Recommended table

| Method | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | 0.00% | 1.000× | 0.8624 | 0.9531 | 0.8766 | 0.0137 |
| v0.13 per-label margin | 1.44% | 0.997× | 0.8587 | 0.9516 | 0.8743 | 0.0142 |
| v0.16 multi-objective | 5.06% | 1.015× | 0.8492 | 0.9425 | 0.8547 | 0.0168 |

## Figure captions

- **Validation Pareto frontier:** Parent-level Macro-F1 and Micro-F1 across the 20 non-dominated validation policies as estimated FLOP saving increases.
- **Optimisation progress:** Best validation-feasible estimated FLOP saving over 50 evolutionary generations.
- **Holdout quality-compute comparison:** Full depth, the v0.13 per-label margin baseline, and the v0.16 Pareto-selected policy under the same staged evaluator and timing protocol.

## Limitations wording

The selected policy was eligible according to validation constraints but did not satisfy the same quality bounds on the corrected holdout, indicating validation-to-holdout shift and optimistic maximum-compute Pareto selection. Segment decisions used fixed 0.5 thresholds because exit-specific calibrated thresholds were unavailable. Runtime gains are specific to the tested CPU, batch size, and threading configuration. The experiment evaluates one frozen operating point and should not be described as a complete budget-conditioned anytime controller. The historical LATS-v2 aggregation was frozen but was originally derived using calibration material related to the corrected-holdout dataset, so the evaluation is not an independent external test.

## What not to claim

Do not claim:

- that v0.16 fulfilled all deployment-quality requirements;
- that `deployment_eligible=true` in the comparison means holdout approval;
- that NSGA-II guarantees the globally optimal policy;
- that 1.015× speedup generalises to GPUs or other CPUs;
- that v0.16 implements label-wise asynchronous exit;
- that v0.16 is a full anytime budget controller;
- that the corrected holdout is a fully independent external test.
