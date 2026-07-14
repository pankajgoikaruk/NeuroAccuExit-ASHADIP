# Active Budget and Anytime Exit v0.2

## Scope

Branch:

```text
active_budget_anytime_exit_v0.2
```

This branch studies real adaptive inference for the trained human-talk multi-label NeuroAccuExit model:

1. fixed-exit quality across network depth;
2. standard sample-wise Dynamic Early-Exit;
3. budget-aware stopping;
4. anytime quality-versus-cost evaluation.

The v0.11 milestone is complete. No retraining was performed for this milestone.

---

## Canonical comparator

```text
v0.10 no-hint + frozen historical LATS-v2 + Exit 3
```

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8623815322333925 |
| Micro-F1 | 0.9531311539976368 |
| Samples-F1 | 0.9588894381281924 |
| Exact Match | 0.8765859284890427 |
| Hamming Loss ↓ | 0.0137254901960784 |
| Average predicted labels | 1.4590542099192618 |
| Average exit depth | 3.0 |
| Compute saved | 0% |

---

## Architecture

The canonical model is one five-block TinyAudioCNN with taps after Blocks 1 and 3.

| Exit | Cumulative backbone |
|---|---|
| Exit 1 | Block 1 |
| Exit 2 | Blocks 1–3 |
| Exit 3 | Blocks 1–5 |

`models/anytime_exit_net.py` exposes staged execution while preserving the original `ExitNet.forward()` path.

---

## Implemented components

| Component | Status |
|---|---|
| Staged wrapper | Complete |
| Three-exit no-hint equivalence tests | Complete |
| Five-exit no-hint equivalence tests | Complete |
| Hint-compatible equivalence tests | Complete |
| Real checkpoint equivalence | PASS |
| Always Exit 1/2/3 audit | Complete |
| Validation-only policy tuning | Complete |
| Genuine Exit-2/Exit-3 staged evaluation | Complete |
| Budget-aware controller | Next |
| Anytime evaluation | Planned |

### Exact checkpoint equivalence

Across eight real corrected-holdout features:

```text
max logit difference at Exit 1 = 0.0
max logit difference at Exit 2 = 0.0
max logit difference at Exit 3 = 0.0
```

---

## Fixed-exit findings

### Parent-level frozen LATS-v2 transfer

| Exit | Macro-F1 | Micro-F1 | Samples-F1 | Exact | Hamming ↓ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1626 | 0.3877 | 0.2902 | 0.0577 | 0.1355 |
| 2 | 0.6923 | 0.7760 | 0.7652 | 0.5156 | 0.0655 |
| 3 | 0.8624 | 0.9531 | 0.9589 | 0.8766 | 0.0137 |

Exit 1 is too weak for broad stopping. Exit 2 is the primary early-exit candidate.

The Exit-1 and Exit-2 parent results transfer the historical Exit-3 LATS-v2 rules unchanged. They should not be described as exit-specific calibrated optima.

---

## First genuine Dynamic Early-Exit policy

The first policy permits stopping only at Exit 2.

```text
Exit 1 → Exit 2
             ├── reliable → stop
             └── otherwise → Blocks 4–5 → Exit 3
```

Frozen validation rule:

```text
Exit 1 and Exit 2 label sets agree
AND Exit 2 prediction is non-empty
AND mean binary confidence >= 0.55
AND minimum threshold margin >= 0.00
```

Segment thresholds use fixed 0.5 because the canonical run had no per-exit threshold-comparison artifact.

### Validation selection

| Item | Value |
|---|---:|
| Samples | 1,883 |
| Exit-2 fraction | 23.31% |
| Parent Macro-F1 | 0.892317 |
| Macro-F1 drop | 0.0103 |
| Estimated FLOPs saved | 14.98% |

### Corrected holdout

| Item | Value |
|---|---:|
| Segments | 4,335 |
| Exit-2 samples | 508 |
| Exit-3 samples | 3,827 |
| Exit-2 fraction | 11.72% |
| Average exit depth | 2.8828 |
| Estimated FLOPs saved | 7.53% |
| Dynamic model latency | 0.8552 ms/segment |
| Parent Macro-F1 | 0.842248 |
| Parent Micro-F1 | 0.935484 |
| Parent Samples-F1 | 0.943577 |
| Parent Exact Match | 0.838524 |
| Parent Hamming Loss | 0.018916 |

This is genuine compute skipping: Exit-2 samples never execute Blocks 4–5.

---

## Interpretation

The first Dynamic Early-Exit policy demonstrates feasibility but is not the final operating point.

Advantages:

- real staged execution;
- validation-only policy selection;
- no holdout retuning;
- measurable exit distribution;
- reduced average depth;
- estimated computation saving.

Current limitations:

- the policy is permissive (`confidence=0.55`, `margin=0.00`);
- holdout Exit-2 coverage is lower than validation coverage;
- parent Exit-2 decisions use thresholds designed for Exit 3;
- same-protocol Always Exit 3 latency is still required;
- Exit 1 is not used as a stopping point.

---

## Commands

Fixed-exit audit:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\fixed_policy\run_v011_EE.ps1"
```

Dynamic policy:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1"
```

Use a stricter validation quality constraint:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -MaxMacroF1Drop 0.01
```

Use an already frozen policy:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.11_EE\dynamic_policy\run_dynamic_v011_EE.ps1" `
  -SkipPrechecks `
  -SkipTuning
```

---

## Documentation package

```text
docs/tables/active_budget_anytime_exit_v0.2/v0.11_EE/
```

This package contains experiment setup, paper-ready wording, exact compact result tables, checkpoint-equivalence evidence, PowerShell commands, and a machine-readable manifest.

---

## Next milestone

The next main experiment is budget-aware Early-Exit.

Required controller decisions:

```text
reliable_early_exit
budget_forced_exit
final_exit
```

The controller must compare the remaining budget with the incremental cost of the next stage. Anytime evaluation will then report quality at multiple normalized budgets.
