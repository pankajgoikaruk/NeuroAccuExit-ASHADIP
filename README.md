# NeuroAccuExit-ASHADIP — Active Budget and Anytime Exit v0.2

This branch develops computationally adaptive inference for the human-talk multi-label NeuroAccuExit model. The work is organized into three stages:

1. standard sample-wise Early-Exit;
2. budget-aware Early-Exit;
3. anytime inference across explicit computation budgets.

The **v0.11_EE milestone is complete**. It establishes numerically equivalent staged inference, audits the quality available at each fixed exit, and demonstrates genuine runtime compute skipping with a validation-frozen Exit-2/Exit-3 policy.

---

## Branch identity

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.2` |
| Documentation name | **NeuroAccuExit — Active Budget and Anytime Exit v0.2** |
| Source branch | `active_budget_anytime_exit_v0.1` |
| Task | Human-talk multi-label speaker/context detection |
| Current milestone | v0.11 fixed-exit audit and genuine Dynamic Early-Exit complete |
| Next milestone | Budget-aware Early-Exit |
| Later milestone | Anytime inference and quality-versus-cost curves |

This branch does not retrain the canonical model for v0.11. It reuses the trained three-exit network and changes only the inference path and stopping policy.

---

## Canonical full-depth reference

All Early-Exit experiments are compared against:

```text
v0.10 no-hint + frozen historical LATS-v2 + Exit 3 probabilities
```

| Metric | Exact value | Paper value |
|---|---:|---:|
| Macro-F1 | 0.8623815322333925 | **0.8624** |
| Micro-F1 | 0.9531311539976368 | **0.9531** |
| Samples-F1 | 0.9588894381281924 | **0.9589** |
| Exact Match | 0.8765859284890427 | **0.8766** |
| Hamming Loss ↓ | 0.0137254901960784 | **0.0137** |
| Average predicted labels per parent | 1.4590542099192618 | 1.4591 |
| Average exit depth | 3.0 | 3.0 |
| Compute saved | 0% | 0% |
| Parent clips | 867 | 867 |

`1.4591` is the average number of predicted positive labels per parent clip. It is not average exit depth.

The complete frozen baseline package remains at:

```text
docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/
```

---

## Model architecture used by v0.11

The canonical checkpoint uses one five-block TinyAudioCNN backbone with two intermediate taps:

```text
tap_blocks = (1, 3)
```

This produces three cumulative exits:

| Exit | Backbone computation reached | Additional blocks from previous exit |
|---|---|---|
| Exit 1 | Block 1 | Block 1 |
| Exit 2 | Blocks 1–3 | Blocks 2–3 |
| Exit 3 | Blocks 1–5 | Blocks 4–5 |

The model is not three independent CNNs. It is one shared backbone with intermediate heads.

---

## Genuine staged inference

`models/anytime_exit_net.py` wraps the existing trained `ExitNet` without adding parameters or changing weights.

```python
logits1, state = anytime_model.start(x)
logits2, state = anytime_model.continue_from(state)
logits3, state = anytime_model.continue_from(state)
```

A stopped sample does not execute later blocks. For example, a sample accepted at Exit 2 never executes Blocks 4–5.

The historical full-forward training and evaluation path remains unchanged.

### Numerical equivalence

The real canonical checkpoint was tested on eight corrected-holdout features with shape:

```text
[8, 1, 64, 101]
```

For all three exits:

```text
maximum absolute logit difference       = 0.0
mean absolute logit difference          = 0.0
maximum absolute probability difference = 0.0
```

All staged-wrapper unit tests also passed.

---

## v0.11 fixed-exit quality audit

The same 4,335 corrected-holdout segments were evaluated under three separate scenarios:

```text
Always Exit 1
Always Exit 2
Always Exit 3
```

### Segment-level results at threshold 0.5

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming ↓ | Avg predicted labels |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1208 | 0.2941 | 0.1915 | 0.0323 | 0.1334 | 0.4210 |
| 2 | 0.5978 | 0.7035 | 0.6328 | 0.4422 | 0.0782 | 1.1663 |
| 3 | 0.7377 | 0.8793 | 0.8713 | 0.7384 | 0.0341 | 1.3596 |

### Parent-level frozen LATS-v2 transfer

The historical Exit-3 LATS-v2 aggregation methods and thresholds were transferred unchanged to Exit 1 and Exit 2 for diagnostic comparison.

| Method | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming ↓ | Avg predicted labels |
|---|---:|---:|---:|---:|---:|---:|
| Always Exit 1 | 0.1626 | 0.3877 | 0.2902 | 0.0577 | 0.1355 | 0.7439 |
| Always Exit 2 | 0.6923 | 0.7760 | 0.7652 | 0.5156 | 0.0655 | 1.4556 |
| Always Exit 3 | **0.8624** | **0.9531** | **0.9589** | **0.8766** | **0.0137** | 1.4591 |

These Exit-1 and Exit-2 parent results are **policy-transfer diagnostics**, not exit-specific calibrated optima.

Main finding:

> Exit 1 is too weak for routine stopping. Exit 2 contains useful early decisions but still requires selective escalation to Exit 3.

---

## v0.11 genuine Dynamic Early-Exit

The first staged policy intentionally permits stopping only at Exit 2:

```text
Exit 1
  ↓
Exit 2
  ↓
reliable? ── yes → stop at Exit 2
         └─ no  → execute Blocks 4–5 and use Exit 3
```

Exit 2 is accepted when:

1. Exit 1 and Exit 2 produce the same thresholded label set;
2. the Exit 2 label set is non-empty;
3. Exit 2 mean binary confidence satisfies the frozen threshold;
4. the Exit 2 decision-margin condition is satisfied.

### Validation-selected policy

| Setting | Frozen value |
|---|---:|
| Segment threshold mode | `fixed_0p5` |
| Confidence threshold | 0.55 |
| Minimum decision margin | 0.00 |
| Exit 1–Exit 2 label-set agreement | required |
| Empty prediction allowed to stop | no |
| Maximum validation Macro-F1 drop | 0.02 |
| Minimum validation Exit-2 fraction | 0.05 |

The canonical run did not contain `threshold_tuning/threshold_comparison.json`; therefore the v0.11 runner automatically used fixed 0.5 segment thresholds. Policy selection still used frozen LATS-v2 parent-level validation performance.

Validation selection result:

| Metric | Value |
|---|---:|
| Validation samples | 1,883 |
| Exit-2 fraction | 23.31% |
| Parent Macro-F1 | 0.892317 |
| Absolute Macro-F1 drop | 0.0103 |
| Estimated FLOPs saved | 14.98% |
| Selection status | `quality_constraint_met` |

### Corrected-holdout result

| Metric | Dynamic v0.11 |
|---|---:|
| Segments | 4,335 |
| Parent clips | 867 |
| Exit-2 samples | 508 |
| Exit-3 samples | 3,827 |
| Exit-2 fraction | 11.72% |
| Average exit depth | 2.8828 |
| Estimated FLOPs saved | 7.53% |
| Model latency per segment | 0.8552 ms |
| Segment Macro-F1 | 0.721297 |
| Parent Macro-F1 | 0.842248 |
| Parent Micro-F1 | 0.935484 |
| Parent Samples-F1 | 0.943577 |
| Parent Exact Match | 0.838524 |
| Parent Hamming Loss ↓ | 0.018916 |
| Parent average predicted labels | 1.462514 |

Only the 3,827 samples assigned to Exit 3 executed Blocks 4–5.

### Change from the canonical full-depth reference

| Metric | Dynamic | Full depth | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.842248 | 0.862382 | −0.020134 |
| Micro-F1 | 0.935484 | 0.953131 | −0.017647 |
| Samples-F1 | 0.943577 | 0.958889 | −0.015312 |
| Exact Match | 0.838524 | 0.876586 | −0.038062 |
| Hamming Loss ↓ | 0.018916 | 0.013725 | +0.005191 |

This is a valid proof of real adaptive computation. It is not yet the final quality–efficiency operating point.

---

## Reproduction commands

### Fixed-exit audit

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\fixed_policy\run_v011_EE.ps1"
```

### Genuine Dynamic Early-Exit

```powershell
conda activate ASHADIP_V0

powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1"
```

Reuse an existing frozen policy and skip completed prechecks:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -SkipPrechecks `
  -SkipTuning
```

Detailed commands and expected outputs are documented under:

```text
docs/tables/active_budget_anytime_exit_v0.2/v0.11_EE/
```

---

## Repository organization

```text
models/
└── anytime_exit_net.py

tests/
├── __init__.py
└── test_anytime_exit_net.py

scripts/v0.11_EE/
├── fixed_policy/
│   ├── verify_checkpoint_equivalence_v011.py
│   ├── evaluate_fixed_exits_v011.py
│   └── run_v011_EE.ps1
└── dynamic_policy/
    ├── tune_dynamic_policy_v011.py
    ├── evaluate_dynamic_early_exit_v011.py
    └── run_dynamic_v011_EE.ps1

docs/
├── active_budget_anytime_exit_v0.2/
│   └── README.md
└── tables/active_budget_anytime_exit_v0.2/
    └── v0.11_EE/
```

Large per-segment probability and prediction files remain under `human_talk_workspace` and are not committed. Compact result tables and paper-ready summaries are committed under `docs/tables`.

---

## Current scientific conclusions

1. Staged inference exactly reproduces the trained model’s full-forward logits.
2. Exit 1 is not sufficiently reliable for general sample-wise stopping.
3. Exit 2 is the principal early-stop candidate.
4. Genuine staged inference can skip Blocks 4–5 for selected samples.
5. The first frozen policy saved 7.53% estimated FLOPs while retaining 97.67% of parent Macro-F1 and 98.15% of parent Micro-F1.
6. Validation selected substantially more Exit-2 samples than the corrected holdout, indicating a reliability-distribution shift.
7. Exit-specific calibration may improve Exit-2 performance because the current parent diagnostic transfers Exit-3 LATS rules unchanged.
8. Dynamic latency must still be compared with Always Exit 3 under the identical timing protocol before reporting measured speedup.

---

## Roadmap

| Stage | Status |
|---|---|
| Frozen full-depth reference | Complete |
| Staged inference implementation | Complete |
| Staged/full numerical equivalence | Complete |
| Always Exit 1/2/3 audit | Complete |
| Genuine standard Dynamic Early-Exit | Complete — first operating point |
| Stricter quality-constrained dynamic policies | Next ablation |
| Exit-specific calibration | Planned |
| Same-protocol fixed-exit latency profiling | Planned |
| Budget-aware Early-Exit | Next main milestone |
| Anytime inference and Pareto curves | Planned |

---

## Paper-ready v0.11 statement

> We implemented an inference-only staged wrapper for a trained three-exit TinyAudioCNN, enabling samples to terminate before executing deeper backbone blocks. Staged and conventional full-forward logits were numerically identical at all exits. A validation-frozen Exit-2/Exit-3 policy stopped 11.72% of corrected-holdout segments at Exit 2, reducing estimated computation by 7.53% and average exit depth from 3.0 to 2.8828. Parent-level Macro-F1 decreased from 0.8624 to 0.8422, while Micro-F1 decreased from 0.9531 to 0.9355. These results establish genuine compute-adaptive inference and define the baseline trade-off for subsequent budget-aware and anytime policies.

---

## Important limitations

- Exit 1 and Exit 2 parent metrics currently use transferred Exit-3 LATS-v2 rules.
- The first dynamic policy uses fixed 0.5 segment thresholds because no per-exit tuned threshold artifact existed.
- The corrected holdout was not used to select the policy.
- The reported 0.8552 ms dynamic latency lacks an identical-protocol Always Exit 3 comparator.
- Estimated FLOP saving is architecture-based and should be supplemented by repeated latency and memory profiling.
- The first policy is a standard sample-wise policy, not yet a budget-aware or label-wise asynchronous controller.
