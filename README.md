# NeuroAccuExit-ASHADIP — Active Budget and Anytime Exit v0.4

This branch studies **genuine computation-adaptive inference** for ten-label, multi-label human-talk audio classification. It contains the complete Early-Exit progression from `v0.11_EE` through `v0.18_EE`, including fixed-exit baselines, label-aware rules, learned controllers, parent-level risk control, multi-objective optimisation, fully sequential anytime inference, and a fair 3-exit versus 5-exit retraining study.

The current completed milestone is:

```text
v0.18_EE — fair sequential anytime exit with matched 3-exit/5-exit training,
           stricter Exit-1 protection, and validation-derived continuation risk
```

The primary sequential routes are:

```text
3-exit: Exit 1 → Exit 2 → Exit 3
5-exit: Exit 1 → Exit 2 → Exit 3 → Exit 4 → Exit 5
```

Samples that stop are removed from the active batch and do not execute later CNN blocks.

---

## Branch identity

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.4` |
| Source branch | `active_budget_anytime_exit_v0.3` |
| Task | 10-label human-talk speaker/context classification |
| Current milestone | `v0.18_EE` fair strict sequential anytime exit |
| Previous milestones | `v0.16_EE` multi-objective Exit-2 rule; `v0.17_EE` sequential 3-/5-exit inference |
| Canonical 3-exit checkpoint | `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845` |
| Fair 5-exit checkpoint | `main_v018_human_corrected_balanced_5exit_no_hint_auxmatched_20260725_231353` |
| Canonical quality reference | Always Exit 3 + frozen historical LATS-v2 |
| Evaluation unit | 4,335 segments aggregated into 867 parent clips |
| Timing protocol | CPU, one Torch/BLAS thread, batch 128, 30 repetitions |
| Full integration | Complete: training, fairness audit, equivalence, tuning, holdout, timing, ablations, and reporting |

---

## Current scientific verdict

| Finding | Confirmed decision |
|---|---|
| Fair 3-exit/5-exit training | **Established in v0.18**: same manifest, features, labels, seed, optimiser settings, no-hint setting, final loss weight, and total auxiliary-loss budget |
| 3-exit `full_strict` policy | Saves compute, but fails three of four holdout-quality constraints |
| 5-exit `full_strict` policy | Saves more compute and gives a stronger speedup, but fails all four holdout-quality constraints |
| 5-exit `no_exit1` ablation | **Closest feasible point**: passes Micro-F1, Exact Match, and Hamming limits; Macro-F1 drop exceeds the limit by only `0.000819` |
| Exit 1 | Useful for additional saving, but remains the highest-risk stage |
| Label-specific margins | Essential; removing them causes major quality collapse |
| Previous-exit stability | Useful safety mechanism |
| v0.18 continuation-risk veto | Now demonstrably active and quality-protective, unlike the non-binding v0.17 risk term |
| Validation-to-holdout transfer | Main unresolved bottleneck |
| Final deployable policy | **Not established by v0.18** |

### Final scientific verdict

> v0.18 resolves the fairness limitation of v0.17 and confirms that a deeper five-exit network offers greater computation-saving capacity under matched training. However, more exits do not automatically yield a quality-safe policy. The best current candidate is the five-exit `Exit 3 → Exit 5` (`no_exit1`) route, which is extremely close to satisfying all four predefined holdout constraints.

### Unsuccessful finding

> Neither selected `full_strict` policy is deployment-ready. The 3-exit policy fails Micro-F1, Exact Match, and Hamming constraints; the 5-exit policy fails all four constraints. Therefore, v0.18 must not be presented as a completed optimal quality–compute solution.

---

## Canonical full-depth references

### Always Exit 3

| Metric | Value |
|---|---:|
| Parent Macro-F1 | 0.862382 |
| Parent Micro-F1 | 0.953131 |
| Parent Samples-F1 | 0.958889 |
| Parent Exact Match | 0.876586 |
| Parent Hamming Loss ↓ | 0.013725 |
| Estimated FLOPs saved | 0% |
| Parent clips | 867 |

### Fair Always Exit 5 from v0.18

| Metric | Value |
|---|---:|
| Parent Macro-F1 | 0.820970 |
| Parent Micro-F1 | 0.907340 |
| Parent Samples-F1 | 0.923620 |
| Parent Exact Match | 0.779700 |
| Parent Hamming Loss ↓ | 0.027800 |
| Estimated FLOPs saved | 0% |
| Parent clips | 867 |

The fair five-exit model is evaluated with the same frozen LATS-v2 protocol for controlled comparison. Its parent-level probability distribution is not independently recalibrated in v0.18.

---

## Version traceability

| Version | Main implementation | Research question | Confirmed outcome |
|---|---|---|---|
| `v0.11_EE` | Staged wrapper, fixed exits, global Exit-2/3 rule | Can frozen checkpoints genuinely skip later blocks? | Genuine skipping established; 7.53% estimated saving, but substantial quality loss. |
| `v0.12_EE` | Validation-derived label-risk continuation | Should labels benefiting from the final exit receive protection? | Small improvement over v0.11; risk alone did not form a stronger frontier. |
| `v0.13_EE` | Matched global, delta, risk, per-label-margin, and logistic policies | Which matched policy best balances quality and compute? | Per-label margin became the current quality-constrained 3-exit baseline. |
| `v0.14_EE` | Parent-aware counterfactual gates and Exit-1 ablation | Can parent-harm targets improve safety, and can Exit 1 help? | Exit-2 gate unsafe; Exit-1 route preserved quality but stopped too rarely and slowed inference. |
| `v0.15_EE` | Whole-parent nonparametric and shared-logistic risk control | Does one parent-level stopping decision fix joint aggregation mismatch? | Quality preserved, but stopping coverage and real efficiency were negligible. |
| `v0.16_EE` | NSGA-II-style optimisation of the Exit-2 per-label rule | Can multi-objective search find a useful Pareto point? | 5.06% FLOPs and 1.015× speedup; all holdout-quality limits failed. |
| `v0.17_EE` | Fully sequential 3-/5-exit optimisation | Can every exit participate, and can extra exits improve the trade-off? | Strong 5-exit within-checkpoint result, but architecture comparison was not fair. |
| `v0.18_EE` | Fair 5-exit retraining + strict sequential risk-veto policy | Does the v0.17 result reproduce under matched training, and can stronger safeguards transfer? | Fair comparison established; full policies remain unsafe; 5-exit No-Exit-1 is closest to feasible. |

Detailed history: `docs/active_budget_anytime_exit_v0.4/VERSION_HISTORY.md`.

---

## v0.18 fair training protocol

The five-exit checkpoint was retrained with the canonical data and optimisation setup.

| Setting | 3-exit | 5-exit |
|---|---:|---:|
| Training rows | 25,519 | 25,519 |
| Validation rows | 1,883 | 1,883 |
| Test rows | 1,961 | 1,961 |
| Labels | 10 | 10 |
| Tap blocks | `1,3` | `1,2,3,4` |
| Exits | 3 | 5 |
| Epochs | 40 | 40 |
| Batch size | 64 | 64 |
| Learning rate | 0.001 | 0.001 |
| Seed | 42 | 42 |
| Hint passing | Disabled | Disabled |
| Final-exit loss weight | 1.0 | 1.0 |
| Auxiliary-loss weights | `0.3,0.3` | `0.15,0.15,0.15,0.15` |
| Total auxiliary budget | 0.60 | 0.60 |

The fairness audit passed every required check. Only exit topology and the distribution of the matched auxiliary-loss budget differ.

### Five-exit training behaviour

| Exit | Test Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|
| Exit 1 | 0.2106 | 0.3472 | 0.2704 | 0.1469 | 0.1296 |
| Exit 2 | 0.3795 | 0.5075 | 0.4506 | 0.2902 | 0.1109 |
| Exit 3 | 0.6112 | 0.6542 | 0.6196 | 0.4182 | 0.0898 |
| Exit 4 | 0.7419 | 0.7356 | 0.7163 | 0.4946 | 0.0727 |
| Exit 5 | **0.8320** | **0.8216** | **0.8204** | **0.6206** | **0.0502** |

Quality increases consistently with depth, explaining why Exit 1 must be highly selective.

---

## v0.18 strict sequential policy

At each non-final exit, the policy uses mean multi-label binary confidence, ten label-specific decision margins, probability change from the preceding exit, previous-exit label-set stability, validation-derived continuation-risk scores, a risk-weighted margin multiplier, a high-risk uncertainty veto, an extra Exit-1 confidence boost, and a non-empty prediction condition.

For label `l` at exit `e`:

```text
binary confidence = max(p[e,l], 1 − p[e,l])
decision margin   = |p[e,l] − threshold[e,l]|
probability delta = |p[e,l] − p[e−1,l]|
```

Validation-only continuation risk combines how often the final exit corrects an earlier label and how much label F1 improves at deeper exits. It is an inference-time stopping safeguard, not a training-loss penalty.

A sample stops at the first exit satisfying all active conditions. Otherwise, it continues. The entire sample stops together; v0.18 is not label-wise asynchronous inference.

---

## v0.18 optimisation settings

| Setting | Value |
|---|---:|
| Algorithm | Constraint-aware NSGA-II-style search |
| Population | 112 |
| Generations | 70 |
| Seed | 42 |
| Parent-grouped folds | 5 |
| Safety fraction | 0.50 |
| Minimum total early fraction | 0.020 |
| Minimum Exit-1 fraction | 0.0025 |
| Segment threshold mode | `fixed_0p5` |
| Validation segments / parents | 1,883 / 304 |
| Holdout segments / parents | 4,335 / 867 |
| Batch size | 128 |
| Device | CPU |
| Torch/BLAS threads | 1 |
| Timing repetitions | 30 |

### Holdout quality limits

| Metric | Maximum permitted degradation |
|---|---:|
| Parent Macro-F1 drop | 0.010 |
| Parent Micro-F1 drop | 0.005 |
| Parent Exact-Match drop | 0.010 |
| Parent Hamming increase | 0.002 |

Validation eligibility and corrected-holdout compliance are reported separately.

---

## v0.18 corrected-holdout headline

| Architecture | Method | Exit distribution | FLOPs saved | Speedup | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ | All limits met? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3-exit | Always final | `0/0/100%` | 0.00% | 1.000× | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 | Reference |
| 3-exit | `full_strict` | `0.48/5.21/94.30%` | 3.82% | 1.018× | 0.85285 | 0.94530 | — | 0.86159 | 0.01603 | **No** |
| 5-exit | Always final | `0/0/0/0/100%` | 0.00% | 1.000× | 0.82097 | 0.90734 | 0.92362 | 0.77970 | 0.02780 | Reference |
| 5-exit | `full_strict` | `4.71/0/12.71/0/82.58%` | 12.70% | 1.057× | 0.79813 | 0.89832 | — | 0.75779 | 0.03010 | **No** |
| 5-exit | `no_exit1` | `0/0/14.28/0/85.72%` | **9.18%** | **1.037×** | **0.81015** | **0.90375** | — | **0.77163** | **0.02872** | 3/4 limits |

`—` indicates that the compact execution summary did not provide a rounded parent Samples-F1 value for that row; the full CSV/JSON output remains authoritative.

### Constraint audit

| Policy | Macro drop | Micro drop | Exact drop | Hamming increase | Status |
|---|---:|---:|---:|---:|---|
| 3-exit `full_strict` | 0.00953 | 0.00783 | 0.01499 | 0.00231 | Fails 3/4 |
| 5-exit `full_strict` | 0.02284 | 0.00902 | 0.02191 | 0.00231 | Fails 4/4 |
| 5-exit `no_exit1` | **0.010819** | **0.003594** | **0.008074** | **0.000923** | Fails Macro only by **0.000819** |

---

## Canonical cross-version corrected-holdout table

| Method | FLOPs saved | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Always Exit 3 | 0.00% | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| v0.13 per-label margin | 1.44% | 0.858748 | 0.951556 | 0.957198 | 0.874279 | 0.014187 |
| v0.13 logistic gate | 11.30% | 0.833034 | 0.943529 | 0.949750 | 0.855825 | 0.016609 |
| v0.14 Exit 1–3 ablation | 0.69% | 0.861442 | 0.952756 | 0.958697 | 0.876586 | 0.013841 |
| v0.14 Exit 2–3 gate | 13.05% | 0.840798 | 0.933966 | 0.942473 | 0.835063 | 0.019262 |
| v0.15 nonparametric parent risk | 0.44% | 0.863129 | 0.952681 | 0.958505 | 0.875433 | 0.013841 |
| v0.15 shared logistic parent gate | 0.00% | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 |
| v0.16 multi-objective margin | 5.06% | 0.849203 | 0.942474 | 0.950266 | 0.854671 | 0.016840 |
| v0.17 sequential anytime (3-exit) | 8.64% | 0.840128 | 0.937549 | 0.945653 | 0.840830 | 0.018224 |
| **v0.18 strict sequential (3-exit)** | **3.82%** | **0.85285** | **0.94530** | — | **0.86159** | **0.01603** |

---

## Canonical policy-structure comparison

| Method | Stop unit | Decision route | FLOPs saved | Parent Macro-F1 | Parent Micro-F1 | Exact Match | Hamming ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| Always Exit 3 | None | Exit 3 only | 0.00% | 0.86238 | 0.95313 | 0.87659 | 0.01373 |
| v0.13 per-label margin | Segment | Exit 2 → Exit 3 | 1.44% | 0.85875 | 0.95156 | 0.87428 | 0.01419 |
| v0.14 Exit-1 ablation | Segment | Exit 1 → Exit 3 | 0.69% | 0.86144 | 0.95276 | 0.87659 | 0.01384 |
| v0.15 nonparametric parent | Parent | Exit 2 → Exit 3 | 0.44% | 0.86313 | 0.95268 | 0.87543 | 0.01384 |
| v0.15 shared logistic | Parent | Exit 2 → Exit 3 | 0.00% | 0.86238 | 0.95313 | 0.87659 | 0.01373 |
| v0.16 multi-objective margin | Segment | Exit 2 → Exit 3 | 5.06% | 0.84920 | 0.94247 | 0.85467 | 0.01684 |
| v0.17 sequential anytime | Segment, sequential | Exit 1 → Exit 2 → Exit 3 | 8.64% | 0.84013 | 0.93755 | 0.84083 | 0.01822 |
| **v0.18 strict sequential** | Segment, sequential | Exit 1 → Exit 2 → Exit 3 | **3.82%** | **0.85285** | **0.94530** | **0.86159** | **0.01603** |
| v0.13 logistic gate | Segment | Exit 2 → Exit 3 | 11.30% | 0.83303 | 0.94353 | 0.85582 | 0.01661 |
| v0.14 Exit-2 parent-aware | Segment | Exit 2 → Exit 3 | 13.05% | 0.84080 | 0.93397 | 0.83506 | 0.01926 |

The five-exit v0.18 results are maintained separately because they use their own Always-Exit-5 quality reference.

---

## Ablation findings

- Disabling Exit 1 in the five-exit study reduces saving from 12.70% to 9.18% but makes the policy nearly feasible.
- Removing the risk veto increases five-exit savings to about 22.78%, while Macro-F1 falls to about 0.78119 and Exact Match to about 0.73472.
- Removing label margins causes major quality collapse.
- Removing stability increases savings but reduces quality.
- Confidence-only stopping is unsafe for this multi-label problem.

---

## Reproduction commands

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1" `
  -TimingRepeats 30
```

Force retraining:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1" `
  -ForceRetrain5 `
  -TimingRepeats 30
```

Full command documentation: `docs/tables/active_budget_anytime_exit_v0.4/v0.18_EE/PS_COMMANDS.md`.

---

## Limitations and non-claims

- Do not claim that v0.18 found the optimal quality–compute trade-off.
- Do not call either `full_strict` policy deployment-ready.
- Do not claim universal five-exit superiority.
- Do not use the non-fair v0.17 five-exit result as the principal fair conclusion.
- The frozen LATS-v2 evaluator may favour the canonical 3-exit probability distribution.
- Validation eligibility does not imply holdout compliance.
- Estimated FLOPs and measured latency are different quantities.
- v0.18 is sample-wise, not label-wise asynchronous inference.
- Evidence accumulation and distilled knowledge are not completed v0.18 components.

---

## Current research decision

| Role | Method |
|---|---|
| Canonical full-quality reference | Always Exit 3 + frozen LATS-v2 |
| Quality-constrained 3-exit adaptive baseline | v0.13 per-label margin |
| Fair architecture study | v0.18 |
| Closest-to-feasible policy | v0.18 five-exit `no_exit1` (`Exit 3 → Exit 5`) |
| Unsuccessful v0.18 result | Both selected `full_strict` policies |
| Next experiment | Dedicated No-Exit-1 optimisation with stronger transient-label safeguards |
