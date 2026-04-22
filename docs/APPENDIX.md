# Appendix: Additional Comparison Tables for `kexit-greedy-hint-gunshot`

This appendix provides compact comparison tables for the current **gunshot vs non-gunshot** branch-level evaluation on `kexit-greedy-hint-gunshot`.

The four validated greedy runs are:
- `gs3`
- `gs3_hint`
- `gs5`
- `gs5_hint`

Throughout Tables A–D, the difference column is defined as:

- **Δ = Hint − No-Hint**

For Table E, the difference column is defined as:

- **Δ = Depth×Time − Full-Clip**

A key note for this appendix is that the current gunshot dataset has a much smaller average clip window budget than the earlier moth setting. In this branch, the full-clip denominator is **3.012** windows per clip, not **14.773**. Therefore, all accuracy-efficiency comparisons below must be interpreted in the context of the **current dataset and current preprocessing**, not forced to match the earlier branch.

---

## Table A. Segment-level greedy policy comparison

| Exit setting | No-hint policy acc | Hint policy acc | Δ policy acc | No-hint avg exit depth | Hint avg exit depth | Δ avg exit depth | No-hint flip-any | Hint flip-any | Δ flip-any | No-hint avg flip count | Hint avg flip count | Δ avg flip count | No-hint exit consistency | Hint exit consistency | Δ exit consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-exit | 0.9548 | 0.9535 | -0.0013 | 1.801 | 1.810 | +0.009 | 0.1628 | 0.1809 | +0.0181 | 0.1731 | 0.2003 | +0.0272 | 0.9935 | 0.9987 | +0.0052 |
| 5-exit | 0.9587 | 0.9561 | -0.0026 | 2.265 | 2.288 | +0.023 | 0.2080 | 0.1744 | -0.0336 | 0.2532 | 0.2054 | -0.0478 | 0.9987 | 0.9910 | -0.0077 |

**Interpretation.**  
Hint passing does **not** improve segment-level greedy policy accuracy in either 3-exit or 5-exit on this gunshot dataset. In 3-exit, hint is slightly worse and slightly deeper. In 5-exit, hint is also slightly worse in policy accuracy, although it reduces flip-any rate. So the current hinted models alter routing behavior, but they do not improve the main segment-policy decision quality.

---

## Table B. Per-exit test accuracy comparison

| Setting | Exit1 | Exit2 | Exit3 | Exit4 | Exit5 |
|---|---:|---:|---:|---:|---:|
| `gs3` | 0.8269 | 0.9380 | 0.9587 | — | — |
| `gs3_hint` | 0.8165 | 0.9432 | 0.9548 | — | — |
| Δ (Hint − No-Hint) | -0.0104 | +0.0052 | -0.0039 | — | — |
| `gs5` | 0.8049 | 0.9057 | 0.9509 | 0.9651 | 0.9599 |
| `gs5_hint` | 0.8140 | 0.8915 | 0.9496 | 0.9561 | 0.9625 |
| Δ (Hint − No-Hint) | +0.0091 | -0.0142 | -0.0013 | -0.0090 | +0.0026 |

**Interpretation.**  
The per-exit comparison shows that hint has **mixed local effects**, not a universal benefit. In the 3-exit case, hint improves only Exit2 and slightly harms Exit1 and Exit3. In the 5-exit case, hint helps Exit1 and Exit5 slightly, but hurts Exit2–Exit4. This supports the conclusion that the current hint mechanism is **task-dependent** and not consistently strengthening all intermediate stages on this binary gunshot dataset.

---

## Table C. Exit mix comparison (segment policy and Depth×Time)

| Setting | Segment exit mix | Depth×Time exit mix |
|---|---|---|
| `gs3` | e1=0.4070, e2=0.3850, e3=0.2080 | e1=0.3252, e2=0.4513, e3=0.2235 |
| `gs3_hint` | e1=0.3837, e2=0.4225, e3=0.1938 | e1=0.2683, e2=0.5255, e3=0.2062 |
| `gs5` | e1=0.3760, e2=0.2468, e3=0.2132, e4=0.0646, e5=0.0995 | e1=0.2667, e2=0.3244, e3=0.2422, e4=0.0667, e5=0.1000 |
| `gs5_hint` | e1=0.3540, e2=0.2946, e3=0.1693, e4=0.0736, e5=0.1085 | e1=0.2899, e2=0.3371, e3=0.1865, e4=0.0719, e5=0.1146 |

**Interpretation.**  
Hint clearly changes routing behavior. In 3-exit, hint shifts more decisions toward **Exit2** in both segment policy and Depth×Time, but this routing change does not improve the final clip result. In 5-exit, hint slightly increases usage of deeper exits under both modes, but again the routing change is not enough to surpass the no-hint baseline on overall clip accuracy.

---

## Table D. Depth×Time comparison: no-hint vs hint passing

| Exit setting | No-hint used-win acc | Hint used-win acc | Δ used-win acc | No-hint avg windows | Hint avg windows | Δ avg windows | No-hint avg compute | Hint avg compute | Δ avg compute | No-hint compute saved | Hint compute saved | Δ compute saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-exit | 0.9558 | 0.9512 | -0.0046 | 1.759 / 3.012 | 1.755 / 3.012 | -0.004 | 3.339 | 3.401 | +0.062 | 38.45% | 37.62% | -0.83 pp |
| 5-exit | 0.9578 | 0.9551 | -0.0027 | 1.751 / 3.012 | 1.732 / 3.012 | -0.019 | 4.218 | 4.128 | -0.090 | 38.16% | 40.09% | +1.93 pp |

**Interpretation.**  
Depth×Time again shows a mixed story. In 3-exit, hint slightly reduces windows used, but it also lowers used-window accuracy and worsens compute saving. In 5-exit, hint slightly reduces used windows and compute, and improves compute saved, but it still does **not** recover the loss in clip accuracy relative to the 5-exit no-hint baseline. Therefore, the current 5-exit hint result may be viewed as a **small efficiency trade-off**, not a clear accuracy improvement.

---

## Table E. Full-clip vs Depth×Time accuracy-efficiency tradeoff

| Setting | Full-clip acc | Full-clip avg windows | Full-clip avg compute | Depth×Time acc | Depth×Time avg windows | Depth×Time avg compute | Δ windows (D×T − Full) | Δ compute (D×T − Full) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gs3` | 0.9844 | 3.012 / 3.012 | 5.424 | 0.9844 | 1.759 / 3.012 | 3.339 | -1.253 | -2.085 |
| `gs5` | 0.9767 | 3.012 / 3.012 | 6.821 | 0.9728 | 1.751 / 3.012 | 4.218 | -1.261 | -2.603 |
| `gs3_hint` | 0.9650 | 3.012 / 3.012 | 5.451 | 0.9650 | 1.755 / 3.012 | 3.401 | -1.257 | -2.050 |
| `gs5_hint` | 0.9689 | 3.012 / 3.012 | 6.891 | 0.9650 | 1.732 / 3.012 | 4.128 | -1.280 | -2.763 |

**Interpretation.**  
All four models benefit from Depth×Time by reducing the average number of used windows and compute relative to the full-clip baseline. However, the strongest **overall practical result** is still `gs3`, because it preserves **0.9844** clip accuracy under both full-clip and Depth×Time while also using lower compute than the deeper 5-exit models. The hinted runs do not beat their no-hint counterparts on this dataset in final clip accuracy, even when they provide modest efficiency gains.

---

## Overall appendix takeaway

These appendix tables support the same main conclusion as the updated gunshot branch documentation:

- **`gs3`** is the best overall **practical model** on the current gunshot dataset.
- **`gs5`** is the best **deeper no-hint reference**.
- **`gs3_hint`** does not improve the compact 3-exit system on this dataset.
- **`gs5_hint`** is mixed: it shows slight efficiency advantages in some metrics, but it does not beat `gs5` or `gs3` on overall clip accuracy.

Therefore, the current evidence suggests that sequential hint passing is **not useless**, but it is **not universally beneficial**. In this branch, its effect is **task-dependent** and currently weaker than on the earlier moth setting.
