# v0.18 Paper-Ready Summary

## Method

We train a five-exit model under a matched protocol with the canonical three-exit network, holding the training manifest, features, label schema, optimiser settings, seed, final-exit objective weight, and total auxiliary-loss budget constant. We then optimise a sequential multi-label stopping policy with exit-specific confidence, stability, probability-change, label-margin, and validation-derived continuation-risk constraints.

## Result

The fair five-exit model supports greater computation reduction than the three-exit model, but neither selected full sequential policy satisfies all corrected-holdout quality constraints. The strongest operating point is a five-exit No-Exit-1 policy that stops 14.28% of samples at Exit 3, saves 9.18% estimated FLOPs, achieves a 1.037× CPU speedup, and satisfies Micro-F1, Exact Match, and Hamming constraints. Its Macro-F1 drop exceeds the predefined limit by 0.000819.

## Final scientific verdict

> v0.18 establishes a fair architecture comparison and demonstrates that deeper multi-exit networks offer additional computation-saving capacity. However, it also shows that increased exit count alone does not guarantee a safe quality–compute trade-off. The principal unresolved challenge is robust validation-to-holdout transfer, particularly for Exit 1 and transient or context-sensitive labels.

## Unsuccessful finding

> The selected three-exit and five-exit full strict policies are not deployment-ready. The three-exit policy fails three quality limits and the five-exit policy fails all four.

## Safe claims

- The fair training audit passed.
- Staged inference is numerically equivalent to full forward execution.
- The five-exit model offers greater compute-saving capacity in this experiment.
- The v0.18 risk veto is active and quality-protective.
- Label-specific margins and stability are necessary safeguards.
- The five-exit No-Exit-1 route is nearly feasible.

## Non-claims

- Do not claim an optimal trade-off.
- Do not claim deployment readiness.
- Do not claim universal five-exit superiority.
- Do not claim the risk score is a causal explanation.
- Do not call the method label-wise asynchronous exit.
- Do not retune on the corrected holdout and preserve an untouched-test claim.
