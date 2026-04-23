# Appendix: Additional Tables for `kexit-greedy-gunshot-segment`

This appendix provides compact tables for the current **generic segmentation + multiclass greedy** branch-level evaluation on `kexit-greedy-gunshot-segment`.

The current validated stable run is:

- `kexit_greedy_gunshot_segment_002`

This appendix now includes:

- dataset inventory
- segmentation statistics
- per-exit test accuracy
- segment-level greedy policy
- full-clip sequential greedy
- Depth×Time clip greedy
- current research findings

---

## Table A. Inventory summary of the current validated dataset

| Class | Files | Min sec | Median sec | Mean sec | Max sec |
|---|---:|---:|---:|---:|---:|
| car_crash | 92 | 1.0000 | 2.3064 | 2.9281 | 10.8639 |
| conversation | 81 | 1.4800 | 3.0000 | 14.6656 | 994.3688 |
| engine_idling | 65 | 1.0376 | 8.0000 | 11.3280 | 36.0000 |
| fireworks | 14 | 1.3095 | 21.0000 | 108.0361 | 770.2427 |
| gun_shot | 187 | 0.4988 | 1.5084 | 1.9196 | 44.0000 |
| rain | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |
| road_traffic | 121 | 3.9998 | 4.9999 | 5.4463 | 60.0000 |
| scream | 151 | 0.5151 | 1.5020 | 1.7642 | 6.5912 |
| thunderstorm | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |
| wind | 100 | 4.9995 | 5.0000 | 5.0000 | 5.0005 |

**Interpretation.**  
The current dataset is clearly heterogeneous. Some classes contain short event-like sounds (`car_crash`, `gun_shot`, `scream`), while others contain much longer ambience/background recordings (`conversation`, `engine_idling`, `fireworks`). This is exactly why a generic segmentation branch was needed.

---

## Table B. File and segment split summary

### B1. File-level split

| Split | Files |
|---|---:|
| train | 707 |
| val | 152 |
| test | 152 |

### B2. Segment-level split

| Split | Segments |
|---|---:|
| train | 2536 |
| val | 551 |
| test | 555 |

### B3. Test split by label

| Label | Test segments |
|---|---:|
| car_crash | 58 |
| conversation | 43 |
| engine_idling | 49 |
| fireworks | 6 |
| gun_shot | 40 |
| rain | 75 |
| road_traffic | 90 |
| scream | 44 |
| thunderstorm | 75 |
| wind | 75 |

**Interpretation.**  
The split procedure is functioning, but class balance is still uneven. `fireworks` is especially small, which makes it an important rare-class stress point in the current branch.

---

## Table C. Rejected segment summary

| Reason | Count |
|---|---:|
| `cap_dropped` | 6249 |
| `silent_window` | 1786 |

**Interpretation.**  
This table captures one of the most important findings of the branch. The current hard cap successfully prevents segment explosion, but it discards a very large number of non-silent candidate segments. So the branch now works technically, but the current capping policy is likely too aggressive for long informative recordings.

---

## Table D. Per-exit test accuracy

| Exit | Accuracy |
|---|---:|
| Exit1 | 0.2613 |
| Exit2 | 0.4162 |
| Exit3 | 0.5946 |
| Exit4 | 0.7027 |
| Exit5 | 0.6739 |

**Interpretation.**  
The most important result here is that **Exit4 is stronger than Exit5**. This means the deepest exit is not currently the strongest classifier. It suggests that the final stage may be overfitting, under-calibrated for this data regime, or simply not benefiting from the current training balance.

---

## Table E. Segment-level greedy policy

| Metric | Value |
|---|---:|
| Policy accuracy | 0.6739 |
| Avg exit depth | 4.589 |
| Flip-any rate | 0.8198 |
| Avg flip count | 1.2793 |
| Exit consistency | 1.0000 |

### Exit mix

| Exit | Usage |
|---|---:|
| e1 | 0.0000 |
| e2 | 0.0649 |
| e3 | 0.0721 |
| e4 | 0.0721 |
| e5 | 0.7910 |

**Interpretation.**  
The current greedy policy is strongly biased toward the deepest exit. That matches the current multiclass difficulty: the system is effectively refusing to exit early most of the time. This is consistent with the weaker early-exit accuracies and with the general difficulty of the 10-class task.

---

## Table F. Threshold selection summary

| Metric | Value |
|---|---:|
| Best tau | 0.92 |
| Validation macro-F1 | 0.6708 |
| Validation accuracy | 0.7169 |

**Interpretation.**  
The selected threshold is relatively conservative, which again aligns with the routing behavior: the model is preferring deeper decisions rather than early exits for the current multiclass dataset.

---

## Table G. Full-clip sequential greedy

| Metric | Value |
|---|---:|
| Clip accuracy | 0.7829 |
| Segment acc over processed windows | 0.6739 |
| Avg windows used | 3.651 / 3.651 |
| Windows saved | 0.00% |
| Avg compute units | 16.757 |
| Compute saved | 0.00% |
| Avg depth per used window | 4.589 |
| Flip-rate (used windows) | 0.8198 |
| Exit-consistency | 1.0000 |

**Interpretation.**  
This is the no-saving clip baseline: it uses all windows and therefore establishes the reference compute cost for the clip-level evaluation on this dataset.

---

## Table H. Depth×Time clip greedy

| Metric | Value |
|---|---:|
| Clip accuracy | 0.7829 |
| Segment acc over used windows | 0.6209 |
| Avg windows used | 2.638 / 3.651 |
| Windows saved | 27.75% |
| Avg compute units | 11.993 |
| Compute saved | 28.43% |
| Avg depth per used window | 4.546 |
| Flip-rate (used windows) | 0.8254 |
| Exit-consistency | 1.0000 |

**Interpretation.**  
Depth×Time is already useful in this branch. It preserves the same clip accuracy as the full-clip baseline while reducing both windows and compute. This is one of the most important positive results of the current validated run.

---

## Table I. Current branch findings

| Finding | Status |
|---|---|
| Generic segmentation pipeline works end to end | Yes |
| File-level split and feature extraction are stable | Yes |
| Dynamic class-count override works in practice | Yes |
| Multiclass run is stable | Yes |
| Full-clip sequential greedy results validated in this run | Yes |
| Depth×Time clip-greedy results validated in this run | Yes |
| Current capping strategy is satisfactory | No |
| Current class balance is satisfactory | No |
| Rare-class performance is satisfactory | No |

**Interpretation.**  
This table summarizes the branch state clearly: the engineering objective has largely been achieved, and clip-policy evaluation is now validated. The main open experimental problems are class imbalance, aggressive capping, and poor rare-class performance.

---

## Table J. Rare-class warning

The clip-level per-class report shows that `fireworks` currently has:

- precision = **0.0**
- recall = **0.0**
- f1 = **0.0**
- support = **2**

**Interpretation.**  
This is a strong warning that the current global cap and class imbalance are harming rare classes. Any next-stage balancing strategy should explicitly target this weakness.

---

## Overall appendix takeaway

These appendix tables support the main conclusion of the updated branch documentation:

- the branch now has a **stable generic segmentation pipeline**
- the current validated run is the first stable **10-class greedy baseline with clip-policy results**
- the most positive clip-level finding is that **Depth×Time saves 27.75% windows and 28.43% compute with no clip-accuracy loss**
- the main remaining issues are:
  - class imbalance
  - heavy `cap_dropped` loss
  - weaker early-exit performance
  - very poor rare-class behavior for `fireworks`

Therefore, the current evidence suggests that the next stage of work should focus on **data balance, softer capping, and rare-class handling**, not on basic preprocessing stability.
