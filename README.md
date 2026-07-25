# NeuroAccuExit-ASHADIP — Active Budget and Anytime Exit v0.4

This branch studies **genuine computation-adaptive inference** for multi-label human-talk audio classification. It preserves the completed v0.11–v0.16 policy studies and adds v0.17, the first fully sequential anytime-exit experiment that evaluates every available exit in both three-exit and five-exit models.

The branch does not retrain the canonical three-exit TinyAudioCNN for policy experiments. v0.17 also evaluates an available historical five-exit checkpoint. Stopping policies are selected on validation data, frozen, and then evaluated with genuine staged execution on the corrected holdout.

---

## Branch identity

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.4` |
| Source branch | `active_budget_anytime_exit_v0.3` |
| Task | 10-label human-talk speaker/context classification |
| Current completed milestone | `v0.17_EE` sequential active-budget anytime exit |
| Three-exit route | `Exit 1 → Exit 2 → Exit 3` |
| Five-exit route | `Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5` |
| Three-exit checkpoint | `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845` |
| Five-exit checkpoint used | `main_v06_expanded_5exit_20260603_210324` |
| Full-quality three-exit reference | Always Exit 3 + frozen historical LATS-v2 |
| Current fair-comparison status | **Not fair across architectures** because training manifests differ |
| Full integration status | Complete for both checkpoints, validation tuning, corrected holdout, ablations, and 30-repeat CPU timing |

---

## Canonical three-exit full-depth reference

All three-exit adaptive experiments are compared with:

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
| Parent clips | 867 |

The frozen baseline package is stored under `docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/`.

---

## Architecture and genuine staged inference

The staged wrapper executes one backbone increment at a time and removes stopped samples from the active batch. Returning an intermediate prediction after computing the full network is **not** counted as genuine Early Exit.

### Three exits

| Exit | Cumulative backbone |
|---|---|
| Exit 1 | Block 1 |
| Exit 2 | Blocks 1–3 |
| Exit 3 | Blocks 1–5 |

### Five exits

| Exit | Cumulative backbone |
|---|---|
| Exit 1 | Block 1 |
| Exit 2 | Blocks 1–2 |
| Exit 3 | Blocks 1–3 |
| Exit 4 | Blocks 1–4 |
| Exit 5 | Blocks 1–5 |

Checkpoint-equivalence tests passed at every exit for both checkpoints with maximum absolute logit and probability differences equal to `0.0`.

---

## Version traceability

| Version | Main implementation | Research question | Confirmed outcome |
|---|---|---|---|
| `v0.11_EE` | Staged wrapper, fixed exits, global Exit-2/3 rule | Can the frozen model genuinely skip deeper blocks? | Yes; 7.53% estimated FLOPs saved, but substantial quality loss. |
| `v0.12_EE` | Validation-derived label-risk rule | Should labels benefiting from Exit 3 receive stronger continuation protection? | Small improvement over v0.11; risk rule alone did not establish a better frontier. |
| `v0.13_EE` | Matched rules and logistic gate | Which lightweight or learned policy gives the best matched trade-off? | Per-label margin became the strongest reliable three-exit adaptive baseline. |
| `v0.14_EE` | Parent-aware segment counterfactual gates | Can predicted parent harm make Exit-2 stopping safer, and is Exit 1 useful? | Exit-2 gate failed quality; Exit 1 preserved quality but stopped too rarely and slowed inference. |
| `v0.15_EE` | Whole-parent nonparametric and shared-logistic risk control | Does one parent-level decision remove joint-substitution errors? | Quality preserved, but useful stopping coverage and speedup were not achieved. |
| `v0.16_EE` | NSGA-II-style per-label margin optimisation | Can multi-objective search produce meaningful saving and speedup? | 5.06% FLOPs and 1.015× speedup, but all holdout quality limits failed. |
| `v0.17_EE` | Fully sequential 3-exit and 5-exit optimisation | Does evaluating every exit improve the quality–computation trade-off? | Five-exit sequential policy met all within-model holdout limits with 30.71% FLOPs saved and 1.114× speedup; three-exit policy was faster but not quality-safe. |

Detailed history: `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md`.

---

## v0.17 theoretical method

For exit `e`, label `l`, probability `p(e,l)`, and classification threshold `t(e,l)`:

```text
binary confidence(e,l) = max(p(e,l), 1 − p(e,l))
margin(e,l)            = |p(e,l) − t(e,l)|
delta(e,l)             = |p(e,l) − p(e−1,l)|
```

The policy evaluates each non-final exit sequentially. A sample stops at the first exit satisfying all active conditions:

1. mean binary confidence is above the stage-specific threshold;
2. every label-specific margin is above its stage-specific minimum;
3. maximum probability movement from the previous exit is below the stage limit;
4. from Exit 2 onward, the thresholded label set is stable relative to the previous exit;
5. the validation-derived maximum label-risk score remains below the stage risk budget;
6. the predicted label set is non-empty.

Otherwise, only unresolved active samples continue to the next block and exit. Exit 1 is included in the primary policy; disabling it is reported only as an ablation.

### Multi-objective optimiser

The chromosome contains one parameter block per non-final exit:

```text
[confidence threshold,
 maximum inter-exit probability delta,
 maximum label-risk budget,
 ten label-specific margins]
```

The constraint-aware NSGA-II-style search maximises estimated FLOP saving while minimising robust degradation in Parent Macro-F1, Parent Micro-F1, Exact Match, and Hamming Loss. v0.17 uses a safety-buffered Pareto-knee selection rather than selecting the most aggressive feasible validation point.

---

## v0.17 experimental settings

| Setting | Value |
|---|---:|
| Segment labels | 10 |
| Segment threshold mode | `fixed_0p5` |
| Population size | 96 |
| Generations | 60 |
| Random seed | 42 |
| Safety-buffered Pareto ratio | 0.75 |
| Minimum total early-exit fraction | 0.02 |
| Minimum Exit-1 fraction | 0.005 |
| Validation segments / parents | 1,883 / 304 |
| Holdout segments / parents | 4,335 / 867 |
| Three-exit candidates / Pareto points | 5,847 / 19 |
| Five-exit candidates / Pareto points | 5,856 / 86 |
| Maximum robust Macro-F1 drop | 0.010 |
| Maximum robust Micro-F1 drop | 0.005 |
| Maximum robust Exact-Match drop | 0.010 |
| Maximum robust Hamming increase | 0.002 |
| Device | CPU |
| Batch size | 128 |
| Torch/BLAS threads | 1 |
| Publication timing repetitions | 30 |

No policy was retuned on the corrected holdout.

---

## v0.17 headline results

### Three-exit model

| Method | Exit 1 | Exit 2 | Exit 3 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | 0.00% | 0.00% | 100.00% | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.876586 | 0.013725 |
| Full sequential | 6.07% | 4.34% | 89.60% | 8.64% | 1.037× | 0.840128 | 0.937549 | 0.840830 | 0.018224 |
| No Exit 1 | 0.00% | 5.28% | 94.72% | 3.39% | 1.022× | 0.854086 | 0.946871 | 0.866205 | 0.015571 |

**Confirmed status:** computationally successful, but the full sequential policy failed every predefined holdout quality limit.

### Five-exit model

| Method | Exit 1 | Exit 2 | Exit 3 | Exit 4 | Exit 5 | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always Exit 5 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 1.000× | 0.810761 | 0.869498 | 0.673587 | 0.038985 |
| Full sequential | 6.83% | 1.22% | 18.59% | 26.30% | 47.06% | 30.71% | 1.114× | 0.801356 | 0.868859 | 0.688581 | 0.039100 |
| No Exit 1 | 0.00% | — | — | — | — | 26.80% | 1.096× | 0.809541 | 0.870906 | 0.687428 | approximately preserved |

### Five-exit quality-limit audit

| Constraint | Limit | Observed change | Met? |
|---|---:|---:|---|
| Macro-F1 drop | ≤ 0.010 | 0.009406 | **Yes** |
| Micro-F1 drop | ≤ 0.005 | 0.000639 | **Yes** |
| Exact-Match drop | ≤ 0.010 | −0.014994 (improvement) | **Yes** |
| Hamming increase | ≤ 0.002 | 0.000115 | **Yes** |

Within this five-exit checkpoint, the sequential policy is the first result in the programme to combine substantial genuine compute reduction, repeatable measured speedup, and compliance with all predefined holdout-quality limits.

---

## Confirmed research findings

1. **The five-exit result is the major success.** More than half of holdout segments stopped before Exit 5, saving 30.71% estimated FLOPs and producing a repeatable 1.114× CPU speedup while meeting all four quality limits.
2. **The three-exit policy is computationally successful but not quality-safe.** It saved 8.64% FLOPs and achieved 1.037× speedup, but Macro-F1, Micro-F1, Exact Match, and Hamming all exceeded the permitted degradation.
3. **Exit 1 is useful but risky.** It adds substantial compute saving in both architectures; removing Exit 1 improves quality, especially in the three-exit model.
4. **Label margins and stability are essential.** Removing label margins or using confidence alone creates much larger savings but severe multi-label quality collapse. Removing stability gives limited additional saving with worse quality.
5. **The current risk term is weakly active.** `No risk` produced identical or nearly identical outcomes to the full policy; risk formulation needs refinement.
6. **Per-label failures are concentrated.** `Nick_Vujicic`, `audience_reaction_present`, and `Eric_Thomas` are major remaining risk labels in the five-exit model; `audience_reaction_present`, `Eric_Thomas`, and `other_speaker_present` are important risks in the three-exit model.
7. **Intermediate exits can occasionally improve complete label sets.** In the five-exit run, Exact Match improved despite a small Macro-F1 decline, showing that Exit 5 is not always superior for every parent.
8. **The architecture comparison is not yet fair.** The three-exit and five-exit checkpoints used different training manifests and training-set sizes, so direct architectural superiority must not be claimed.

---

## Cumulative ablation interpretation

| Ablation | Confirmed interpretation |
|---|---|
| Full sequential | Primary multi-stage controller; uses every available exit. |
| No Exit 1 | Quantifies Exit-1 compute benefit and safety cost. |
| No stability | Demonstrates that previous-exit label agreement is a useful safety constraint. |
| No risk | Shows the current risk score adds little beyond margins and stability. |
| No label margins | Demonstrates that label-specific margins are critical for multi-label consistency. |
| Confidence only | Demonstrates that one global confidence signal is unsafe for this task. |

Extreme ablations are intentionally retained as negative evidence: large compute savings alone are not sufficient when multi-label prediction consistency collapses.

---

## Reproduction commands

### Three-exit sequential experiment

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -Run3Only
```

### Combined three-exit and five-exit experiment

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324"
```

### Publication timing

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.17_EE\sequential_anytime_exit\run_sequential_anytime_exit_v017_EE.ps1" `
  -RunDir5 ".\human_talk_workspace\tata_v0.6_raw_pipeline\main_models\runs\main_v06_expanded_5exit_20260603_210324" `
  -TimingRepeats 30
```

No v0.17 backbone-training command exists. The current policy experiments reuse frozen checkpoints. A future fair architecture comparison requires training a new five-exit checkpoint using the same canonical training manifest, preprocessing, seed protocol, and evaluation setup as the three-exit checkpoint.

---

## Documentation and results

```text
docs/active_budget_anytime_exit_v0.4/
docs/tables/active_budget_anytime_exit_v0.4/v0.16_EE/
docs/tables/active_budget_anytime_exit_v0.4/v0.17_EE/
```

The v0.17 package contains experiment setup, result analysis, paper-ready wording, commands, compact tables, and SVG comparison figures.

---

## Limitations and cautions

- Do not claim the historical five-exit architecture is superior to the canonical three-exit architecture; their training manifests differ.
- The five-exit result is valid **within its checkpoint** and relative to its own Always Exit 5 baseline.
- The three-exit and five-exit validation sets match, but their training data do not.
- The frozen historical LATS-v2 configuration is a corrected-holdout evaluation policy, not an independent external-test protocol.
- Segment decisions use fixed `0.5` thresholds because per-exit tuned threshold artifacts were unavailable.
- Validation eligibility is not identical to holdout approval; both must be reported separately.
- Estimated FLOPs and measured latency are not interchangeable.
- CPU speedups are hardware-, implementation-, thread-, and batch-size-specific.
- The current risk-score ablation indicates that the risk component is not yet demonstrably useful.
- Do not omit the unsuccessful three-exit result or the negative ablations.
- Do not describe v0.17 as label-wise asynchronous exit; it still selects one exit depth per sample.
- Do not claim global optimality from one NSGA-II run or one random seed.

---

## Final scientific verdict

### Successful finding

The five-exit sequential controller is a strong within-model result: it achieved genuine routing across all exits, 30.71% estimated FLOP saving, 1.114× measured CPU speedup, and compliance with all predefined holdout quality limits.

### Unsuccessful finding

The three-exit full sequential policy did not transfer safely to the corrected holdout, and the current risk term did not provide measurable protection.

### Current decision

| Role | Method |
|---|---|
| Canonical deployment-quality reference | Three-exit Always Exit 3 + frozen LATS-v2 |
| Strongest fair three-exit adaptive baseline | v0.13 per-label margin |
| Successful within-model anytime result | v0.17 five-exit full sequential |
| Compute-successful but quality-unsafe result | v0.17 three-exit full sequential |
| Required next fairness experiment | Train a canonical five-exit model on the same data and protocol as the three-exit checkpoint |
