# Appendix: Additional Comparison Tables for Greedy No-Hint vs Hint Passing

This appendix provides five compact comparison tables derived from the corrected four-run workbook:
- `3exit_greedy`
- `5exit_greedy`
- `3exit_greedy_hint`
- `5exit_greedy_hint`

Throughout Tables A–D, the difference column is defined as:

- **Δ = Hint − No-Hint**

For Table E, the difference column is defined as:

- **Δ = Depth×Time − Full-Clip**

---

## Table A. Segment-level greedy policy comparison

| Exit setting | No-hint policy acc | Hint policy acc | Δ policy acc | No-hint avg exit depth | Hint avg exit depth | Δ avg exit depth | No-hint flip-any | Hint flip-any | Δ flip-any | No-hint exit consistency | Hint exit consistency | Δ exit consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-exit | 0.9754 | 0.9908 | +0.0154 | 1.982 | 1.895 | -0.087 | 0.1785 | 0.1908 | +0.0123 | 1.0000 | 1.0000 | +0.0000 |
| 5-exit | 0.9908 | 0.9723 | -0.0185 | 2.637 | 2.465 | -0.172 | 0.1908 | 0.2123 | +0.0215 | 1.0000 | 0.9908 | -0.0092 |

**Interpretation.**  
Hint passing clearly helps the compact 3-exit setting at the segment level, improving policy accuracy by **+0.0154** while slightly reducing average exit depth. In contrast, the current 5-exit hinted model loses **-0.0185** policy accuracy relative to the corrected 5-exit no-hint baseline, even though it also becomes slightly shallower. This indicates that the present hint mechanism is beneficial in the compact regime but not yet effective in the deeper greedy hierarchy.

---

## Table B. Per-exit test accuracy comparison

| Setting | Exit1 | Exit2 | Exit3 | Exit4 | Exit5 |
|---|---:|---:|---:|---:|---:|
| 3exit_greedy | 0.8338 | 0.9385 | 0.9754 | — | — |
| 3exit_greedy_hint | 0.8369 | 0.9662 | 0.9908 | — | — |
| Δ (Hint − No-Hint) | +0.0031 | +0.0277 | +0.0154 | — | — |
| 5exit_greedy | 0.8369 | 0.8892 | 0.9723 | 0.9754 | 0.9908 |
| 5exit_greedy_hint | 0.8308 | 0.8646 | 0.9231 | 0.9538 | 0.9692 |
| Δ (Hint − No-Hint) | -0.0061 | -0.0246 | -0.0492 | -0.0216 | -0.0216 |

**Interpretation.**  
The per-exit comparison reinforces the main result. In the 3-exit setting, hint passing improves every exit, especially **Exit2 (+0.0277)** and **Exit3 (+0.0154)**. In the 5-exit setting, however, the hinted model underperforms the corrected no-hint baseline at every exit, with the largest drop at **Exit3 (-0.0492)**. This shows that the current 5-exit hint formulation is not yet strengthening the deeper classifier chain.

---

## Table C. Exit mix comparison (segment policy and Depth×Time)

| Setting | Segment exit mix | Depth×Time exit mix |
|---|---|---|
| 3exit_greedy | e1=0.3631, e2=0.2923, e3=0.3446 | e1=0.0889, e2=0.2000, e3=0.7111 |
| 3exit_greedy_hint | e1=0.3846, e2=0.3354, e3=0.2800 | e1=0.0909, e2=0.3409, e3=0.5680 |
| 5exit_greedy | e1=0.3569, e2=0.0308, e3=0.3538, e4=0.1354, e5=0.1231 | e1=0.0889, e2=0.0000, e3=0.3778, e4=0.3111, e5=0.2222 |
| 5exit_greedy_hint | e1=0.3723, e2=0.1354, e3=0.2677, e4=0.1046, e5=0.1200 | e1=0.0870, e2=0.0652, e3=0.3043, e4=0.2826, e5=0.2609 |

**Interpretation.**  
The exit-mix view helps explain how hinting changes routing behavior. In 3-exit, the hinted model shifts more usage toward **Exit2** and away from the final exit, which is consistent with its improved intermediate quality and stronger efficiency. In 5-exit, hinting alters routing but does not translate into better overall performance; the hinted model still uses substantial depth under Depth×Time and does not achieve the same accuracy as the no-hint 5-exit baseline.

---

## Table D. Depth×Time comparison: no-hint vs hint passing

| Exit setting | No-hint used-win acc | Hint used-win acc | Δ used-win acc | No-hint avg windows | Hint avg windows | Δ avg windows | No-hint avg compute | Hint avg compute | Δ avg compute | No-hint compute saved | Hint compute saved | Δ compute saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-exit | 0.9778 | 1.0000 | +0.0222 | 2.045 / 14.773 | 2.000 / 14.773 | -0.045 | 5.364 | 4.955 | -0.409 | 81.68% | 82.31% | +0.63% |
| 5-exit | 0.9778 | 0.9783 | +0.0005 | 2.045 / 14.773 | 2.091 / 14.773 | +0.046 | 7.318 | 7.455 | +0.137 | 80.53% | 79.53% | -1.00% |

**Interpretation.**  
The Depth×Time comparison is the clearest deployment-oriented test. In the 3-exit setting, hinting improves used-window accuracy to **1.0000**, uses slightly fewer windows, lowers average compute, and increases compute saving. In the 5-exit setting, hinting barely changes used-window accuracy, but it uses slightly more windows and more compute, leading to worse compute saving. This is why `3exit_greedy_hint` is the best efficiency-quality result, whereas `5exit_greedy_hint` is not yet beneficial.

---

## Table E. Full-clip vs Depth×Time accuracy-efficiency tradeoff

| Setting | Full-clip acc | Full-clip avg windows | Full-clip avg compute | Depth×Time acc | Depth×Time avg windows | Depth×Time avg compute | Δ windows (D×T − Full) | Δ compute (D×T − Full) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3exit_greedy | 1.0000 | 14.773 / 14.773 | 29.273 | 0.9778 | 2.045 / 14.773 | 5.364 | -12.728 | -23.909 |
| 5exit_greedy | 1.0000 | 14.773 / 14.773 | 38.955 | 0.9778 | 2.045 / 14.773 | 7.318 | -12.728 | -31.637 |
| 3exit_greedy_hint | 1.0000 | 14.773 / 14.773 | 28.000 | 1.0000 | 2.000 / 14.773 | 4.955 | -12.773 | -23.045 |
| 5exit_greedy_hint | 1.0000 | 14.773 / 14.773 | 36.409 | 0.9783 | 2.091 / 14.773 | 7.455 | -12.682 | -28.954 |

**Interpretation.**  
All four models retain perfect full-clip accuracy, but Depth×Time reveals how efficiently that accuracy can be approximated with early stopping. The strongest tradeoff is achieved by `3exit_greedy_hint`, which preserves **1.0000** accuracy even under Depth×Time while reducing average windows from **14.773** to **2.000** and average compute from **28.000** to **4.955**. The 5-exit models remain more compute-expensive, and the 5-exit hinted variant does not surpass the corrected 5-exit no-hint baseline.

---

## Overall appendix takeaway

These appendix tables support the same main conclusion as the corrected branch documentation:

- **`3exit_greedy_hint`** is the best overall **efficiency-quality tradeoff**
- **`5exit_greedy`** is the best **deep-capacity no-hint baseline**
- **`5exit_greedy_hint`** is still not beneficial under the current design

Therefore, the current evidence suggests that sequential hint passing is highly effective in the compact 3-exit setting, but does not yet improve the deeper 5-exit greedy pipeline.
