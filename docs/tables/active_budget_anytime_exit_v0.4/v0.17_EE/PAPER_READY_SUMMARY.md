# v0.17_EE Paper-Ready Summary

## Method paragraph

We extend the active-budget NeuroAccuExit framework from a two-stage Exit-2/Exit-3 decision to a fully sequential anytime policy. For a network with K exits, every non-final exit evaluates whether the current sample should terminate or continue. The decision combines mean multi-label binary confidence, per-label distance from the decision threshold, inter-exit probability change, previous-exit label-set stability, and validation-derived label-risk evidence. A constraint-aware NSGA-II-style optimiser learns a separate parameter block for each non-final exit and jointly maximises estimated computation saving while limiting degradation in parent-level Macro-F1, Micro-F1, Exact Match, and Hamming Loss. To avoid selecting the most aggressive validation-boundary point, the policy is frozen at a safety-buffered Pareto knee before corrected-holdout evaluation.

## Confirmed three-exit result

> For the `Exit 1→Exit 2→Exit 3` configuration, the frozen sequential controller routed `6.07%` of holdout segments to Exit 1 and `4.34%` to Exit 2, saving `8.64%` estimated FLOPs and producing a `1.037×` median CPU speedup. However, Parent Macro-F1 decreased from `0.8624` to `0.8401`, Micro-F1 from `0.9531` to `0.9375`, and Exact Match from `0.8766` to `0.8408`. The three-exit operating point therefore demonstrates real acceleration but does not satisfy the predefined holdout-quality constraints.

## Confirmed five-exit result

> For the tested `Exit 1→Exit 2→Exit 3→Exit 4→Exit 5` checkpoint, the frozen sequential controller terminated `52.94%` of holdout segments before the final exit. It saved `30.71%` estimated FLOPs and achieved a `1.114×` median CPU speedup over 30 repetitions. Relative to Always Exit 5, Parent Macro-F1 decreased by `0.0094`, Micro-F1 decreased by `0.0006`, Exact Match improved by `0.0150`, and Hamming Loss increased by only `0.0001`. All predefined holdout-quality constraints were met.

## Ablation paragraph

> Ablation results show that label-specific decision margins and previous-exit label-set stability are essential for safe multi-label termination. Removing label margins increased early-exit coverage but reduced five-exit Parent Macro-F1 from `0.8014` to `0.7045` and Exact Match from `0.6886` to `0.6090`. Confidence-only stopping was still more aggressive and produced severe quality loss. Removing the current validation-derived risk term did not change parent-level performance, indicating that risk was non-binding under the selected v0.17 thresholds. Exit 1 contributed additional computation saving, but the No-Exit-1 ablation preserved quality more strongly, identifying the first exit as the riskiest decision stage.

## Scientific interpretation

> The results indicate that multiple sequential stopping opportunities can support a strong within-model quality–efficiency trade-off. In the tested five-exit checkpoint, intermediate exits distributed samples across all stages and produced substantial measured acceleration while preserving aggregate quality. The three-exit policy did not transfer safely, and difficult labels—particularly audience reaction, Nick Vujicic, and Eric Thomas—remained sensitive to early termination.

## Architecture-comparison limitation

Use this exact caution:

> The three-exit and five-exit checkpoints were not trained using the same manifest: the canonical three-exit model uses the human-corrected balanced v0.8/v0.10 pipeline, whereas the tested five-exit checkpoint uses the earlier v0.6 expanded pipeline. Although the label schema, holdout, feature cache, LATS-v2 evaluator, optimiser settings, constraints, and timing protocol were matched, the training-manifest difference prevents a causal claim that five exits are superior to three exits. The five-exit result should therefore be described as a successful within-checkpoint sequential policy rather than a fair architecture winner.

## Recommended claims

- v0.17 implements genuine sequential staged inference across every available exit.
- The tested five-exit policy meets the predefined within-checkpoint quality constraints while saving substantial compute.
- Multi-label label margins and inter-exit stability are strongly supported by ablation.
- Exit 1 contributes useful saving but requires stronger safeguards.
- The current validation-derived risk component is non-binding.
- A fair retrained five-exit checkpoint is required before claiming architecture superiority.

## What must not be claimed

- Do not claim that five exits are generally better than three exits.
- Do not report the 5-exit Macro-F1 as comparable to the canonical 3-exit Macro-F1 without stating the different training manifests and baselines.
- Do not call `validation_eligible=true` sufficient evidence of holdout safety; report the explicit holdout audit.
- Do not describe estimated FLOP saving as measured latency saving.
- Do not generalise CPU speedup to GPU, edge hardware, or other batch sizes without remeasurement.
- Do not call the corrected holdout an independent external test.
- Do not claim the risk mechanism improved v0.17; the ablation found no parent-level benefit.
- Do not claim label-wise asynchronous exit; the complete sample stops at one exit.
- Do not include evidence accumulation or distilled knowledge as part of the primary v0.17 policy.
