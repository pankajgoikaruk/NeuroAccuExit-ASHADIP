# NeuroAccuExit-ASHADIP — Active Budget and Anytime Exit v0.4

This branch studies **genuine computation-adaptive inference** for the frozen human-talk multi-label NeuroAccuExit model. It inherits the staged-inference work from v0.2/v0.3 and completes the v0.16 multi-objective optimisation milestone.

The branch does not retrain the TinyAudioCNN backbone or exit heads. It reuses the canonical three-exit checkpoint and changes only inference-time stopping policies, validation-time controller fitting, and policy optimisation.

---

## Branch identity

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.4` |
| Source branch | `active_budget_anytime_exit_v0.3` |
| Task | 10-label human-talk speaker/context classification |
| Canonical checkpoint | `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845` |
| Current completed milestone | `v0.16_EE` multi-objective per-label margin optimisation |
| Current adaptive baseline | `v0.13_EE` per-label margin policy |
| Full-quality reference | Always Exit 3 + frozen historical LATS-v2 |
| Full integration status | Complete on the real checkpoint, validation data, corrected holdout, and 30-repeat CPU timing |

---

## Canonical full-depth reference

All adaptive experiments are compared with:

```text
v0.10 no-hint + frozen historical LATS-v2 + Exit 3 probabilities
```

| Metric | Value |
|---|---:|
| Parent Macro-F1 | 0.862382 |
| Parent Micro-F1 | 0.953131 |
| Parent Samples-F1 | 0.958889 |
| Parent Exact Match | 0.876586 |
| Parent Hamming Loss ↓ | 0.013725 |
| Average exit depth | 3.000000 |
| Estimated FLOPs saved | 0% |
| 30-repeat CPU latency | 1.522200 ms/segment |
| Parent clips | 867 |

The frozen reproducibility package remains under:

```text
docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/
```

---

## Architecture and genuine staged inference

The canonical model is one five-block TinyAudioCNN with intermediate heads after Blocks 1 and 3:

| Exit | Cumulative backbone | Increment from previous exit |
|---|---|---|
| Exit 1 | Block 1 | Block 1 |
| Exit 2 | Blocks 1–3 | Blocks 2–3 |
| Exit 3 | Blocks 1–5 | Blocks 4–5 |

`models/anytime_exit_net.py` executes the network incrementally. A sample accepted at Exit 2 does not execute Blocks 4–5. Checkpoint equivalence passed at all exits with maximum absolute logit and probability differences equal to `0.0`.

Recorded cumulative FLOPs:

| Exit | Cumulative FLOPs | Normalized cost |
|---|---:|---:|
| Exit 1 | 1,861,952 | 0.0361 |
| Exit 2 | 18,451,072 | 0.3574 |
| Exit 3 | 51,629,312 | 1.0000 |

For an Exit-2 stopping fraction `q`, the architecture-based saving is:

```text
saving(q) = q × (1 − FLOPs_exit2 / FLOPs_exit3)
          = q × 64.2624%
```

---

## Version traceability

| Version | Implementation | Research question | Main outcome |
|---|---|---|---|
| `v0.11_EE` | Staged wrapper, fixed exits, global Exit-2/3 rule | Can the trained model genuinely skip deeper blocks without changing exit outputs? | Yes; 7.53% estimated FLOPs saved, but substantial quality loss. |
| `v0.12_EE` | Validation-derived label-risk continuation rule | Do labels that benefit more from Exit 3 require stronger continuation protection? | Slightly improved v0.11 quality; matched-policy advantage not established. |
| `v0.13_EE` | Matched global, delta, label-risk, per-label-margin, and logistic policies | Which lightweight or learned policy gives the best matched quality-compute trade-off? | Per-label margin became the strongest quality-constrained adaptive baseline. |
| `v0.14_EE` | Parent-aware counterfactual gates; Exit-1 ablation | Can parent-aware harm targets improve stopping safety, and is Exit 1 useful? | No robust Exit-2 gate; Exit 1 preserved quality but stopped too rarely and slowed inference. |
| `v0.15_EE` | Whole-parent nonparametric and shared-logistic risk control | Does one joint parent decision remove multi-segment interaction errors? | Quality was preserved, but coverage and compute saving were negligible; controllers were slower. |
| `v0.16_EE` | Constraint-aware NSGA-II-style optimisation of per-label margins | Can multi-objective search improve the lightweight policy and produce real speedup while preserving quality? | Meaningful compute and 1.015× CPU speedup achieved, but holdout quality constraints failed. |

Detailed version history: `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md`.

---

## v0.16 method

The chromosome contains 12 parameters:

```text
θ = [confidence threshold,
     maximum Exit-1→Exit-2 probability delta,
     ten label-specific Exit-2 decision margins]
```

For Exit `e` and label `l`:

```text
binary confidence = max(p_e,l, 1 − p_e,l)
margin_e,l        = |p_e,l − threshold_e,l|
delta_l           = |p_2,l − p_1,l|
```

A segment stops at Exit 2 only when:

1. Exit 1 and Exit 2 thresholded label sets agree;
2. the Exit-2 prediction is non-empty;
3. mean binary confidence exceeds the optimised threshold;
4. maximum Exit-1→Exit-2 probability change is below the optimised limit;
5. every label-specific margin exceeds its own optimised value.

Otherwise Blocks 4–5 run and Exit 3 is used.

The optimiser simultaneously maximises estimated FLOP saving and minimises robust degradation in Parent Macro-F1, Micro-F1, Exact Match, and Hamming Loss. Candidate quality is evaluated with five parent-grouped folds and one-sided upper confidence bounds (`z=1.645`).

---

## v0.16 settings

| Setting | Value |
|---|---:|
| Population size | 80 |
| Generations | 50 |
| Random seed | 42 |
| Unique candidates evaluated | 4,078 |
| Pareto candidates | 20 |
| Parent-grouped validation folds | 5 |
| Validation segments / parents | 1,883 / 304 |
| Holdout segments / parents | 4,335 / 867 |
| Segment threshold mode | `fixed_0p5` |
| Batch size | 128 |
| Device | CPU |
| Torch/BLAS threads | 1 |
| Publication timing repetitions | 30 |
| Maximum validation Macro-F1 drop | 0.010 |
| Maximum validation Micro-F1 drop | 0.005 |
| Maximum validation Exact-Match drop | 0.010 |
| Maximum validation Hamming increase | 0.002 |
| Minimum validation Exit-2 fraction | 0.020 |

No new model training was performed. The optimiser searched policy parameters over frozen Exit-1/2/3 predictions and then evaluated the frozen selected policy using genuine staged inference.

---

## Selected v0.16 policy

| Parameter | Value |
|---|---:|
| Mean confidence threshold | 0.678603 |
| Maximum probability delta | 0.937178 |
| Label-set agreement | required |
| Empty Exit-2 prediction | cannot stop |

| Label | Minimum Exit-2 margin |
|---|---:|
| `Brene_Brown` | 0.004844 |
| `Eckhart_Tolle` | 0.003097 |
| `Eric_Thomas` | 0.024152 |
| `Gary_Vee` | 0.240167 |
| `Jay_Shetty` | 0.025998 |
| `Nick_Vujicic` | 0.181231 |
| `other_speaker_present` | 0.115068 |
| `music_present` | 0.005948 |
| `audience_reaction_present` | 0.122540 |
| `silence_present` | 0.009552 |

Validation selected this point because it was the feasible Pareto candidate with maximum estimated compute saving.

| Validation result | Value |
|---|---:|
| Exit-2 fraction | 19.6495% |
| Average exit depth | 2.803505 |
| Estimated FLOPs saved | 12.6272% |
| Parent Macro-F1 drop | 0.000389 |
| Parent Micro-F1 drop | 0.000940 |
| Parent Exact-Match drop | 0.000000 |
| Parent Hamming increase | 0.000329 |
| Validation status | `quality_constraints_met=true` |

---

## Corrected-holdout comparison

| Method | Exit-2 fraction | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | 0.00% | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| v0.13 per-label margin | 2.24% | 1.44% | 0.997× | 0.858748 | 0.951556 | 0.957198 | 0.874279 | 0.014187 |
| v0.16 multi-objective margin | 7.87% | 5.06% | 1.015× | 0.849203 | 0.942474 | 0.950266 | 0.854671 | 0.016840 |

### Holdout constraint audit

| Constraint | Limit | Observed | Met? |
|---|---:|---:|---|
| Macro-F1 drop | ≤ 0.010 | 0.013178 | **No** |
| Micro-F1 drop | ≤ 0.005 | 0.010657 | **No** |
| Exact-Match drop | ≤ 0.010 | 0.021915 | **No** |
| Hamming increase | ≤ 0.002 | 0.003114 | **No** |

The `deployment_eligible=true` field in the runtime comparison records **validation eligibility**, not post-hoc holdout approval. The correct status is:

```text
validation_eligible = true
holdout_constraints_met = false
```

---

## Confirmed findings

1. Genuine staged inference remains numerically equivalent to conventional full-forward inference at every exit.
2. The dependency-light NSGA-II-style search successfully explored 4,078 unique policies and retained a 20-point validation Pareto front.
3. v0.16 increased holdout Exit-2 coverage to 7.87%, saved 5.06% estimated FLOPs, and achieved a stable 1.015× median CPU speedup over 30 repetitions.
4. v0.16 expanded the observed compute-saving region beyond v0.13, but did not satisfy the predefined holdout quality constraints.
5. `audience_reaction_present` and `other_speaker_present` accounted for important holdout degradation; per-label results must accompany aggregate metrics.
6. v0.13 per-label margin remains the recommended quality-constrained adaptive baseline.
7. Always Exit 3 remains the deployment-quality reference.

---

## Reproduction commands

Complete v0.16 run:

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1"
```

Publication timing:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -TimingRepeats 30
```

Reuse the frozen policy without repeating optimisation:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.16_EE\multiobjective_per_label_margin\run_multiobjective_per_label_margin_v016_EE.ps1" `
  -SkipPrechecks `
  -SkipTuning `
  -TimingRepeats 30
```

No v0.16 training command exists because the backbone and exit heads are frozen.

---

## Documentation and result package

```text
docs/active_budget_anytime_exit_v0.4/
docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/
```

The compact package includes exact setup, validation/holdout analysis, paper-ready language, commands, selected policy, Pareto/history tables, per-label comparisons, cumulative ablations, and SVG figures.

---

## Limitations and cautions

- v0.16 is a validation-selected quality-constrained policy, but it failed the same thresholds on the corrected holdout.
- The holdout must not be used to retune v0.16 and then be reported as an untouched test result.
- The frozen historical LATS-v2 configuration was derived using calibration material related to the corrected-holdout dataset; this is a frozen corrected-holdout evaluation, not an independent external test.
- Segment thresholds were fixed at 0.5 because no per-exit tuned threshold artifact existed.
- The 1.015× speedup is CPU-, hardware-, batch-, and implementation-specific and should not be generalized to other devices without remeasurement.
- Estimated FLOP saving and measured latency are related but not interchangeable.
- v0.16 is not yet an explicit budget-conditioned anytime controller and does not produce a budget-quality curve.
- v0.16 is sample-wise; it is not label-wise asynchronous inference where different labels terminate at different depths.
- Do not claim that multi-objective optimisation guarantees holdout-optimal or globally Pareto-optimal behavior.
- Do not claim v0.16 as the deployment winner; the selected adaptive baseline remains v0.13 per-label margin.

---

## Current research decision

| Role | Method |
|---|---|
| Deployment-quality reference | Always Exit 3 + frozen LATS-v2 |
| Recommended adaptive baseline | v0.13 per-label margin |
| Compute-forward Pareto ablation | v0.16 multi-objective per-label margin |
| Negative/diagnostic gate studies | v0.14 and v0.15 |
| Next methodological direction | safety-buffered Pareto-knee selection or a genuinely budget-conditioned anytime evaluation using a newly reserved calibration/evaluation protocol |
