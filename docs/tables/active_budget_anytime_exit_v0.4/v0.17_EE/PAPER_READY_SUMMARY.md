# v0.17_EE Paper-Ready Summary

## Method paragraph

We implement a sequential active-budget anytime-inference policy for multi-label audio classification. Unlike two-stage policies that decide only between an intermediate and final exit, the controller evaluates every non-final exit. For a three-exit network, samples follow `Exit 1 → Exit 2 → Exit 3`; for a five-exit network, they follow `Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5`. At each stage, stopping depends on multi-label confidence, label-specific decision margins, inter-exit probability change, thresholded label-set stability, and a validation-derived label-risk budget. A constraint-aware multi-objective optimiser learns separate parameters for each non-final exit and selects a safety-buffered Pareto-knee policy under Parent Macro-F1, Micro-F1, Exact-Match, and Hamming-Loss constraints.

## Experimental protocol paragraph

Policies were selected using validation data only and frozen before corrected-holdout evaluation. Genuine staged inference removed stopped samples from the active batch so that later backbone blocks were not executed. Both three-exit and five-exit staged wrappers reproduced conventional forward-pass logits and probabilities exactly at every exit. The final evaluation used 4,335 segments grouped into 867 parent clips and included 30-repeat single-thread CPU timing.

## Main results paragraph

Within the historical five-exit checkpoint, the full sequential policy routed 6.83%, 1.22%, 18.59%, 26.30%, and 47.06% of samples to Exits 1–5. It saved 30.71% estimated FLOPs and achieved a 1.114× median CPU speedup. Parent Macro-F1 decreased from 0.810761 to 0.801356, while Micro-F1 remained nearly unchanged (0.869498 to 0.868859), Exact Match improved from 0.673587 to 0.688581, and Hamming Loss changed from 0.038985 to 0.039100. All predefined within-model holdout-quality limits were satisfied.

## Unsuccessful result paragraph

The same framework did not transfer safely to the canonical three-exit checkpoint. Although it saved 8.64% estimated FLOPs and achieved a 1.037× speedup, Parent Macro-F1 fell from 0.862382 to 0.840128, Micro-F1 from 0.953131 to 0.937549, Exact Match from 0.876586 to 0.840830, and Hamming Loss increased from 0.013725 to 0.018224. The three-exit full sequential policy therefore failed all predefined holdout-quality limits.

## Ablation paragraph

Ablations show that label-specific margins and inter-exit label stability are central to safe multi-label Early Exit. Removing label margins or retaining confidence alone yields substantially larger compute savings but severe degradation in Macro-F1, Exact Match, and Hamming Loss. Disabling Exit 1 improves quality while reducing savings, identifying Exit 1 as a useful but high-risk stage. Removing the current risk term produces identical or nearly identical outcomes, indicating that the present risk formulation is not materially active.

## Safe novelty wording

> We present a fully sequential, multi-objective anytime-exit evaluation for multi-label human-talk audio classification in which every non-final exit independently applies confidence, label-margin, stability, delta, and risk constraints. The study demonstrates a substantial within-model quality–computation trade-off for a five-exit checkpoint and provides matched ablations identifying label margins and inter-exit stability as essential safety components.

## Limitation wording

> The available three-exit and five-exit checkpoints were trained using different manifests and training-set sizes. Consequently, the results establish within-model trade-offs but do not support a causal claim that five-exit architectures are generally superior to three-exit architectures. A fair architecture comparison requires retraining the five-exit model using the canonical three-exit data and protocol.

## Claims to avoid

- “Five exits are universally better than three exits.”
- “The architecture comparison is fair.”
- “The method was evaluated on an independent external test set.”
- “The learned risk mechanism improved performance.”
- “The optimiser found the global optimum.”
- “Different labels independently stop at different depths.”

## Suggested captions

**Table:** Corrected-holdout quality and computational efficiency for full-depth and sequential three-exit/five-exit inference. Each adaptive policy was selected on validation data and frozen before holdout evaluation. The five-exit result is compared with its own full-depth baseline; direct architectural comparison is limited by different training manifests.

**Figure — exit distribution:** The five-exit policy uses all five exits and terminates 52.94% of segments before the final exit.

**Figure — quality–compute:** The five-exit sequential policy remains inside the predefined Macro-F1 limit, while the three-exit policy exceeds it.
