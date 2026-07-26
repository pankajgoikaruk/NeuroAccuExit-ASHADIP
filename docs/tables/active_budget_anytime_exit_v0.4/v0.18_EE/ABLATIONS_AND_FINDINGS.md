# v0.18 Ablations and Findings

## Exit 1

Disabling Exit 1 in the five-exit route reduces FLOPs saved from 12.70% to 9.18%, but changes the policy from failing all constraints to passing three of four. Exit 1 is useful but currently unsafe.

## Risk veto

Removing risk protection increases five-exit savings to about 22.78%, but Macro-F1 falls to about 0.78119 and Exact Match to about 0.73472. The redesigned risk veto is therefore active and quality-protective.

## Stability

Removing previous-exit label-set stability increases savings but worsens quality. Stability remains a justified continuation safeguard.

## Label margins

| Architecture | FLOPs saved | Macro-F1 | Exact |
|---|---:|---:|---:|
| 3-exit no margins | 11.97% | 0.82596 | 0.80738 |
| 5-exit no margins | 38.51% | 0.72721 | 0.62053 |

Label-specific margins are essential.

## Confidence only

| Architecture | FLOPs saved | Macro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|
| 3-exit | 67.58% | 0.57495 | 0.44175 | 0.07659 |
| 5-exit | 75.29% | 0.45477 | 0.35525 | 0.09469 |

Confidence-only stopping is unsuitable for the multi-label task.

## Confirmed findings

1. More exits increase potential compute saving.
2. Exit 1 is the dominant safety problem.
3. Risk, stability, and margins protect quality.
4. Validation eligibility does not guarantee holdout safety.
5. The No-Exit-1 five-exit route is the strongest next candidate.

## Unsuccessful finding

Neither full strict policy meets the complete corrected-holdout constraint set. v0.18 is not a final deployable solution.
