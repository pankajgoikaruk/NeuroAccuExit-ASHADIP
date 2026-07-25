# v0.17_EE Results and Analysis

## Integration result

| Check | 3-exit | 5-exit |
|---|---|---|
| Sequential-policy tests | Passed | Passed |
| Staged-equivalence status | PASS | PASS |
| Maximum probability difference | 0.0 | 0.0 |
| Validation-only policy selection | Yes | Yes |
| Holdout retuning | No | No |
| Timing repetitions | 30 | 30 |

Full integration against the actual checkpoints and datasets is complete.

## Validation selection

| Item | 3-exit | 5-exit |
|---|---:|---:|
| Unique candidates | 5,847 | 5,856 |
| Pareto candidates | 19 | 86 |
| Selection | Safety-buffered Pareto knee | Safety-buffered Pareto knee |
| Validation total early fraction | 18.69% | 46.10% |
| Validation Exit-1 fraction | 6.11% | 10.52% |
| Validation FLOPs saved | 13.98% | 29.39% |
| Validation quality constraints | Passed | Passed |

## Corrected-holdout results

### 3-exit route

| Method | Exit 1 | Exit 2 | Exit 3 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | 0% | 0% | 100% | 0% | 1.000× | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| Full sequential | 6.07% | 4.34% | 89.60% | 8.64% | 1.037× | 0.840128 | 0.937549 | 0.945653 | 0.840830 | 0.018224 |

The 3-exit controller produced a real speedup and genuine Exit-1/Exit-2 usage, but failed all four holdout-quality limits.

### 5-exit route

| Method | E1 | E2 | E3 | E4 | E5 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 5 | 0% | 0% | 0% | 0% | 100% | 0% | 1.000× | 0.810761 | 0.869498 | 0.887906 | 0.673587 | 0.038985 |
| Full sequential | 6.83% | 1.22% | 18.59% | 26.30% | 47.06% | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.886945 | 0.688581 | 0.039100 |

The 5-exit policy is the main successful v0.17 result:

- `52.94%` of segments stopped before Exit 5;
- `30.71%` estimated FLOPs were saved;
- median latency improved from `1.734` to `1.557` ms/segment;
- measured speedup was `1.114×`;
- all four holdout-quality limits were met;
- Exact Match improved by `0.014994`.

## Holdout constraint audit

| Architecture | Macro drop / 0.010 | Micro drop / 0.005 | Exact drop / 0.010 | Hamming increase / 0.002 | All met? |
|---|---:|---:|---:|---:|---|
| 3-exit | 0.022254 | 0.015582 | 0.035755 | 0.004498 | **No** |
| 5-exit | 0.009406 | 0.000639 | -0.014994 | 0.000115 | **Yes** |

A negative Exact-Match drop means the sequential policy improved Exact Match.

## Parent-level change audit

| Architecture | Changed parents | Improved | Worsened | Net label-error change | New exact matches | Lost exact matches |
|---|---:|---:|---:|---:|---:|---:|
| 3-exit | 45 | 5 | 40 | +39 | 5 | 36 |
| 5-exit | 76 | 36 | 38 | +1 | 31 | 18 |

The 5-exit policy changed more parent predictions but added only one net parent-label error. This explains the nearly unchanged Micro-F1/Hamming and improved Exact Match.

## Per-label behaviour

### 3-exit largest losses

| Label | Always-final F1 | Sequential F1 | Change |
|---|---:|---:|---:|
| `audience_reaction_present` | 0.535714 | 0.417910 | -0.117804 |
| `Eric_Thomas` | 0.942029 | 0.883721 | -0.058308 |
| `other_speaker_present` | 0.958696 | 0.944018 | -0.014678 |

### 5-exit improvements

| Label | Always-final F1 | Sequential F1 | Change |
|---|---:|---:|---:|
| `silence_present` | 0.186047 | 0.258065 | +0.072018 |
| `music_present` | 0.824324 | 0.840000 | +0.015676 |
| `Eckhart_Tolle` | 0.994012 | 1.000000 | +0.005988 |
| `Jay_Shetty` | 0.972376 | 0.977778 | +0.005402 |

### 5-exit remaining weaknesses

| Label | Always-final F1 | Sequential F1 | Change |
|---|---:|---:|---:|
| `Nick_Vujicic` | 0.968421 | 0.873563 | -0.094858 |
| `audience_reaction_present` | 0.461538 | 0.376471 | -0.085068 |
| `Eric_Thomas` | 0.872180 | 0.861538 | -0.010642 |

## Timing

| Architecture | Always-final ms/segment | Sequential ms/segment | Speedup |
|---|---:|---:|---:|
| 3-exit | 1.572119 | 1.515521 | 1.037345× |
| 5-exit | 1.733976 | 1.556767 | 1.113832× |

The 5-exit speedup is stronger because enough computation is skipped to exceed controller and active-batch overhead.

## Fairness audit

| Check | Outcome |
|---|---|
| Same labels | Pass |
| Same validation feature root | Pass |
| Same LATS-v2 configuration | Pass |
| Same threshold mode | Pass |
| Same optimiser budget | Pass |
| Same quality constraints | Pass |
| Same validation/training manifest | **Fail** |

Therefore:

> The tested 5-exit checkpoint has a stronger within-model trade-off, but v0.17 does not establish that a five-exit architecture is intrinsically superior to the canonical three-exit architecture.

## Final scientific verdict

### Successful finding

The tested five-exit sequential policy met the predefined quality constraints while producing substantial estimated saving and a measurable CPU speedup.

### Unsuccessful finding

The three-exit full sequential policy did not transfer safely from validation to holdout.

### Interpretation

More exit opportunities may make it easier to allocate intermediate computation gradually. However, the present checkpoints were trained on different manifests, so this remains a hypothesis rather than a fair causal architecture conclusion.
