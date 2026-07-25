# v0.17 Ablations and Research Findings

## Ablation table

### 3-exit

| Method | Early fraction | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full sequential | 10.40% | 8.64% | 1.037× | 0.840128 | 0.937549 | 0.840830 | 0.018224 |
| No Exit 1 | 5.28% | 3.39% | 1.022× | 0.854086 | 0.946871 | 0.866205 | 0.015571 |
| No stability | 19.40% | 14.42% | 1.047× | 0.838162 | 0.936709 | 0.838524 | 0.018454 |
| No risk | 10.40% | 8.64% | 1.040× | 0.840128 | 0.937549 | 0.840830 | 0.018224 |
| No label margins | 32.46% | 30.72% | 1.191× | 0.740154 | 0.850119 | 0.652826 | 0.043599 |
| Confidence only | 84.94% | 67.58% | 1.381× | 0.574954 | 0.729642 | 0.441753 | 0.076586 |

### 5-exit

| Method | Early fraction | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full sequential | 52.94% | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.688581 | 0.039100 |
| No Exit 1 | 50.87% | 26.80% | 1.096× | 0.809541 | 0.870906 | 0.687428 | 0.038639 |
| No stability | 57.16% | 33.82% | 1.136× | 0.791590 | 0.865556 | 0.686275 | 0.040023 |
| No risk | 53.10% | 30.78% | 1.134× | 0.801356 | 0.868859 | 0.688581 | 0.039100 |
| No label margins | 74.49% | 54.81% | 1.342× | 0.704487 | 0.819711 | 0.608997 | 0.053172 |
| Confidence only | 98.59% | 79.00% | 1.597× | 0.475651 | 0.711332 | 0.474048 | 0.080507 |

## Finding 1 — the 5-exit policy is the major success

The full 5-exit route stops `52.94%` of segments before the final exit, saves `30.71%` estimated FLOPs, and runs at `1.114×`. It stays inside all four holdout limits.

## Finding 2 — the 3-exit policy is computationally successful but not quality-safe

The 3-exit policy saves `8.64%` and reaches `1.037×`, proving that the sequential controller creates real acceleration. However, Macro-F1, Micro-F1, Exact Match, and Hamming all exceed their permitted degradation.

## Finding 3 — Exit 1 is useful but currently the riskiest stage

Adding Exit 1 to the No-Exit-1 route increases saving by `5.24` percentage points in the 3-exit policy and `3.91` points in the 5-exit policy. The No-Exit-1 variants preserve Macro-F1 and Micro-F1 more strongly, so future Exit-1 calibration should be stricter.

## Finding 4 — label margins are essential

Removing margins increases theoretical saving but causes severe quality collapse:

| Architecture | Full Macro-F1 | No-margin Macro-F1 | Full Exact | No-margin Exact |
|---|---:|---:|---:|---:|
| 3-exit | 0.840128 | 0.740154 | 0.840830 | 0.652826 |
| 5-exit | 0.801356 | 0.704487 | 0.688581 | 0.608997 |

Confidence alone is even more aggressive and unsafe. A single global confidence score cannot protect every label.

## Finding 5 — stability adds useful protection

Removing label-set stability increases compute saving, especially in the five-exit model, but lowers Macro-F1 and increases Hamming Loss. Stability is therefore a useful safety term.

## Finding 6 — the current risk term is non-binding

For the 3-exit policy, `no_risk` and `full_sequential` produce identical exit distributions and parent metrics. For the five-exit policy, only a tiny Exit-4/Exit-5 routing difference occurs and parent metrics remain identical.

Safe wording:

> Validation-derived risk weighting was implemented, but the selected v0.17 thresholds made the risk condition practically non-binding; margins and stability supplied the measurable protection.

## Finding 7 — per-label behaviour identifies remaining problems

The recurring difficult labels are:

```text
audience_reaction_present
Nick_Vujicic
Eric_Thomas
other_speaker_present
```

The five-exit policy improves `silence_present` and `music_present`, while losing F1 primarily on `Nick_Vujicic` and `audience_reaction_present`.

## Architecture-comparison caution

The full 5-exit result is much stronger in compute saving, but the 5-exit model was trained on a different manifest and starts from a weaker full-depth baseline. The result supports a promising hypothesis about more granular exit opportunities; it is not a fair proof that five exits are better.
