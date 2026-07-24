# NeuroAccuExit-ASHADIP — Active Budget and Anytime Exit v0.3

This branch extends the genuine staged-inference baseline with label-aware, learned, parent-aware, and risk-controlled Early-Exit policies for the human-talk multi-label NeuroAccuExit model.

The completed v0.3 experiment sequence is:

1. `v0.12_EE` — validation-derived label-risk stopping;
2. `v0.13_EE` — matched rule/gate comparison;
3. `v0.14_EE` — parent-aware segment counterfactual gates;
4. `v0.15_EE` — whole-parent selective risk control.

The branch is documentation-complete. It does **not** implement `active_budget_anytime_exit_v0.4`, an explicit budget controller, or an anytime budget sweep.

---

## Branch identity

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.3` |
| Source branch | `active_budget_anytime_exit_v0.2` |
| Task | Ten-label human-talk speaker/context classification |
| Backbone | Five-block TinyAudioCNN / ExitNet |
| Exits | Block 1, Block 3, Block 5 |
| Canonical checkpoint | `main_v010_human_corrected_balanced_3exit_no_hint_20260703_201845/ckpt/best.pt` |
| Canonical comparator | v0.10 no-hint + frozen historical LATS-v2 + Exit 3 |
| Current adaptive recommendation | v0.13 per-label margin policy |
| Deployment-quality reference | Always Exit 3 + frozen LATS-v2 |
| Next branch | `active_budget_anytime_exit_v0.4` — not implemented here |

---

## Canonical full-depth reference

All v0.12–v0.15 comparisons use the same frozen full-depth result:

| Metric | Value |
|---|---:|
| Parent Macro-F1 | **0.8623815322** |
| Parent Micro-F1 | **0.9531311540** |
| Parent Samples-F1 | **0.9588894381** |
| Parent Exact Match | **0.8765859285** |
| Parent Hamming Loss ↓ | **0.0137254902** |
| Average predicted labels | 1.4590542099 |
| Average exit depth | 3.0 |
| Estimated compute saved | 0% |
| Corrected-holdout parents | 867 |
| Corrected-holdout segments | 4,335 |

`1.4591` is the average number of positive labels predicted per parent clip. It is not average exit depth.

The frozen baseline package remains at:

```text
docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/
```

---

## Architecture and genuine staged inference

The model is one shared five-block CNN, not three separate networks.

| Exit | Cumulative backbone | Deeper work skipped when accepted |
|---|---|---|
| Exit 1 | Block 1 | Blocks 2–5 |
| Exit 2 | Blocks 1–3 | Blocks 4–5 |
| Exit 3 | Blocks 1–5 | None |

```text
Input
  └─ Block 1 ─ Exit 1
       └─ Blocks 2–3 ─ Exit 2
            └─ Blocks 4–5 ─ Exit 3
```

`models/anytime_exit_net.py` preserves the trained weights and exposes staged execution. Every v0.12–v0.15 run rechecked staged/full-forward equivalence on real holdout features. The maximum logit and probability differences were `0.0` at all three exits.

A method counts as genuine Early Exit only when stopped samples do not execute later backbone blocks.

---

## Common experimental protocol

| Setting | Value |
|---|---|
| Labels | 6 target speakers + `other_speaker_present`, `music_present`, `audience_reaction_present`, `silence_present` |
| Validation segments | 1,883 |
| Validation parents | 304 where parent-level protocols are used |
| Corrected holdout | 4,335 segments / 867 parents |
| Device | CPU |
| Segment threshold mode | `fixed_0p5` |
| Reason for fixed 0.5 | No per-exit `threshold_comparison.json` existed in the canonical run |
| Parent aggregation | Frozen historical LATS-v2 |
| Holdout tuning | Prohibited; every evaluated policy was frozen first |
| Compute metric | Architecture-estimated cumulative FLOPs |
| Final timing | Repeated same-protocol median/IQR where available |

The segment threshold of 0.5 is distinct from the label-specific frozen LATS-v2 parent thresholds.

---

## Version traceability

| Version | Implementation | Primary research question | Outcome |
|---|---|---|---|
| `v0.12_EE` | Validation-derived label-risk stopping | Can labels that benefit more from Exit 3 be protected from premature Exit 2? | Genuine skipping worked; 7.19% estimated FLOPs saved, but quality loss remained substantial. |
| `v0.13_EE` | Matched rules + logistic gate | Which rule/gate gives the best quality–compute point under identical selection constraints? | Per-label margin became the strongest reliable adaptive baseline; learned gate was too aggressive. |
| `v0.14_EE` | Parent-aware per-label unsafe gates | Can parent-aware counterfactual targets improve transfer and safely include Exit 1? | No robust candidate passed validation; Exit-2 gate harmed holdout quality, Exit-1 gate saved too little. |
| `v0.15_EE` | Whole-parent empirical/logistic risk control | Does making one decision for the entire parent solve multi-segment interaction errors? | Quality was preserved, but controllers stopped almost no parents and were slower due to overhead. |

Detailed records:

```text
docs/active_budget_anytime_exit_v0.3/README.md
docs/tables/active_budget_anytime_exit_v0.3/
```

---

## Cumulative corrected-holdout comparison

| Method | Early-stop unit | Stop/Exit-2 rate | FLOPs saved | Parent Macro-F1 | Parent Micro-F1 | Samples-F1 | Exact Match | Hamming ↓ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Always Exit 3 | None | 0.00% | 0.00% | **0.862382** | **0.953131** | **0.958889** | **0.876586** | **0.013725** | Full-quality reference |
| v0.12 label risk | Segment | 11.19% | 7.19% | 0.843703 | 0.936689 | 0.944692 | 0.840830 | 0.018570 | Feasibility baseline |
| v0.13 global confidence + margin | Segment | 1.18% | 0.76% | 0.861433 | 0.952719 | 0.958505 | 0.875433 | 0.013841 | Most conservative rule |
| v0.13 global + delta | Segment | 2.42% | 1.56% | 0.858556 | 0.950845 | 0.956736 | 0.869666 | 0.014418 | Rule ablation |
| v0.13 label risk | Segment | 2.42% | 1.56% | 0.858556 | 0.950845 | 0.956736 | 0.869666 | 0.014418 | Risk condition non-binding |
| **v0.13 per-label margin** | Segment | **2.24%** | **1.44%** | **0.858748** | **0.951556** | **0.957198** | **0.874279** | **0.014187** | **Current adaptive recommendation** |
| v0.13 logistic gate | Segment | 17.58% | 11.30% | 0.833034 | 0.943529 | 0.949750 | 0.855825 | 0.016609 | Aggressive; unsafe trade-off |
| v0.14 Exit 2→3 gate | Segment | 20.30% | 13.05% | 0.840798 | 0.933966 | 0.942473 | 0.835063 | 0.019262 | Failed robust validation |
| v0.14 Exit 1→3 ablation | Segment | 0.72% | 0.69% | 0.861442 | 0.952756 | 0.958697 | **0.876586** | 0.013841 | Quality preserved; negligible saving |
| v0.15 nonparametric parent risk | Parent | 0.69% | 0.44% | 0.863129 | 0.952681 | 0.958505 | 0.875433 | 0.013841 | Not deployment-eligible; slower |
| v0.15 shared logistic parent gate | Parent | 0.00% | 0.00% | 0.862382 | 0.953131 | 0.958889 | 0.876586 | 0.013725 | No stopping; controller overhead |

The v0.15 Macro-F1 increase does not establish general superiority: it reflects one rare-label correction offset by errors in `other_speaker_present`, while Micro-F1 and Exact Match decreased.

---

## Main research findings

1. **Genuine staged execution is correct.** Every experiment preserved exact checkpoint equivalence.
2. **Exit 2 is the useful intermediate depth.** Exit 1 can be safe for a very small subset but has not produced practical savings.
3. **Simple rules remain competitive.** The v0.13 per-label margin policy produced the most defensible quality–compute balance.
4. **Learned gates can identify more stopping opportunities, but transfer is weak.** The v0.13 logistic gate saved 11.30% estimated FLOPs but exceeded the accepted quality loss.
5. **Label-risk weighting alone was insufficient.** In v0.13, the selected label-risk policy made exactly the same decisions as the global confidence/delta rule.
6. **Parent-aware individual counterfactuals were misaligned with joint runtime substitutions.** v0.14 was a useful negative result.
7. **Whole-parent risk control fixed the conceptual mismatch but reduced sample size and coverage.** v0.15 preserved quality while stopping too few parents to offset controller overhead.
8. **Estimated FLOPs do not guarantee latency speedup.** The v0.14 and v0.15 repeated timing runs showed adaptive methods slower than Always Exit 3 when stopping coverage was small.
9. **The current optimisation problem is multi-objective.** The next strategy should search per-label rule parameters for a Pareto frontier under simultaneous Macro-F1, Micro-F1, Exact Match, Hamming and real-cost constraints.

---

## Recommended current selections

| Role | Selected method |
|---|---|
| Deployment-quality reference | Always Exit 3 + frozen LATS-v2 |
| Current adaptive Early-Exit baseline | v0.13 per-label margin |
| Aggressive compute ablation | v0.13 logistic gate (`0.75` selected on validation; unsafe on holdout) |
| Exit-1 feasibility ablation | v0.14 Exit 1→3 |
| Parent-level negative/diagnostic ablation | v0.15 whole-parent controllers |

No v0.14 or v0.15 controller should be described as deployment-eligible.

---

## Reproduction commands

```powershell
conda activate ASHADIP_V0
```

### v0.12 label-aware policy

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.12_EE\label_aware_policy\run_label_aware_v012_EE.ps1"
```

### v0.13 matched strategy comparison

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.13_EE\matched_policy_comparison\run_matched_policy_comparison_v013_EE.ps1"
```

### v0.14 parent-aware gate

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.14_EE\parent_aware_gate\run_parent_aware_gate_v014_EE.ps1" `
  -TimingRepeats 30
```

### v0.15 whole-parent risk control

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.15_EE\whole_parent_risk_control\run_whole_parent_risk_control_v015_EE.ps1" `
  -TimingRepeats 30
```

Complete commands and output locations are indexed in:

```text
docs/tables/active_budget_anytime_exit_v0.3/PS_COMMANDS.md
```

---

## Repository organization

```text
models/
└── anytime_exit_net.py

policies/
├── label_aware_early_exit_policy.py
├── early_exit_strategy_comparison.py
├── parent_aware_adaptive_gate.py
└── whole_parent_selective_exit.py

scripts/
├── v0.12_EE/label_aware_policy/
├── v0.13_EE/matched_policy_comparison/
├── v0.14_EE/parent_aware_gate/
└── v0.15_EE/whole_parent_risk_control/

tests/
├── test_anytime_exit_net.py
├── test_label_aware_early_exit_policy.py
├── test_early_exit_strategy_comparison.py
├── test_parent_aware_adaptive_gate.py
└── test_whole_parent_selective_exit.py

docs/
├── active_budget_anytime_exit_v0.2/       # preserved historical v0.11 record
├── active_budget_anytime_exit_v0.3/       # completed v0.12–v0.15 narrative
└── tables/active_budget_anytime_exit_v0.3/
```

Large prediction matrices, fitted gate models, parent scores and sweep tables remain under `human_talk_workspace`. Compact documentation and comparison tables are committed under `docs/`.

---

## Theory: reliability, utility and cost

At an intermediate exit `e`, the controller observes an information state `z_e` derived from probabilities, margins, stability, label identity or parent aggregation. A generic stopping objective is:

\[
\text{stop at }e \quad \text{when} \quad
\widehat{R}(z_e) \leq \tau_R
\quad \text{and} \quad
\Delta C_{e\rightarrow e+1} \text{ is not justified by expected quality gain.}
\]

Where:

- `R` is estimated prediction risk;
- `τ_R` is a validation-frozen risk tolerance;
- `ΔC` is the incremental cost of executing the next backbone stage;
- quality is assessed at both segment and frozen-LATS parent level.

The experiments progressively changed the risk estimator:

```text
v0.12: validation-derived label risk
v0.13: matched hand-written rules and a learned sample gate
v0.14: per-label parent-harm counterfactual gates
v0.15: joint whole-parent selective risk control
```

An explicit external budget and anytime budget sweep were not implemented in v0.3.

---

## Important limitations and non-claims

- Do not claim that v0.3 found an optimal trade-off; it found several empirical operating points.
- Do not claim that a learned gate outperformed rules; the current evidence favours the simpler v0.13 per-label margin policy.
- Do not claim measured acceleration from estimated FLOPs alone.
- Do not report v0.12 model-only latency as a controlled speedup.
- Treat v0.13 timing as preliminary because its method ordering and common-stage timing were not sufficiently controlled.
- Treat v0.14 and v0.15 repeated CPU timing as evidence that controller overhead can exceed saved backbone work.
- Do not retune policies after inspecting corrected-holdout results.
- Do not describe the corrected holdout as a fully independent external test set; the historical LATS configuration was derived from calibration splits associated with that dataset.
- Do not call v0.14 or v0.15 controllers deployment-eligible.
- Do not state that budget-aware or anytime inference was completed in this branch.

---

## Documentation entry points

| Document | Purpose |
|---|---|
| `DOC_STRUCTURE.md` | Complete artifact and documentation index |
| `docs/active_budget_anytime_exit_v0.3/README.md` | Detailed research narrative and theory |
| `docs/tables/active_budget_anytime_exit_v0.3/CUMULATIVE_RESULTS.md` | Confirmed tables, ablations and figures |
| `docs/tables/active_budget_anytime_exit_v0.3/PS_COMMANDS.md` | Exact Windows commands and output roots |
| `docs/tables/active_budget_anytime_exit_v0.3/v0.12_EE/README.md` | v0.12 traceability |
| `docs/tables/active_budget_anytime_exit_v0.3/v0.13_EE/README.md` | v0.13 traceability |
| `docs/tables/active_budget_anytime_exit_v0.3/v0.14_EE/README.md` | v0.14 traceability |
| `docs/tables/active_budget_anytime_exit_v0.3/v0.15_EE/README.md` | v0.15 traceability |
