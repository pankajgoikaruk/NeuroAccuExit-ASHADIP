# v0.10 No-Hint + Capped `pos_weight` Experiment

## Research question

Can label-imbalance-aware BCE improve weak/rare-label Macro-F1 more reliably than direct exit-to-exit hint passing?

## Motivation

The previous v0.10 hint-pass run did not outperform the no-hint control after LATS recalibration. The next safer hypothesis is that weak labels such as `silence_present`, `audience_reaction_present`, and `other_speaker_present` may benefit from loss-level positive weighting rather than passing early-exit predictions forward.

## Controlled setting

- Dataset: same v0.8 human-corrected balanced training manifest.
- Model: same 3-exit multi-label model.
- Hint-pass: disabled.
- Loss: BCEWithLogitsLoss with capped `pos_weight`.
- Suggested cap: `PosWeightMax = 5.0`.
- Inference: corrected holdout, parent-level mean, then LATS-v2 coordinate re-optimization.
- Seeds: 101, 202, 303.

## Acceptance rule

Keep this direction only if the 3-seed average is competitive with v0.9_4 and improves Macro-F1 or rare-label F1 without damaging global metrics.

Reference v0.9_4 baseline:

```text
Macro-F1   = 0.867256
Micro-F1   = 0.945786
Samples-F1 = 0.951700
Exact      = 0.860438
Hamming    = 0.015802
```

## Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v0.10\run_v010_no_hint_posweight_stability.ps1 `
  -Seeds 101,202,303 `
  -PosWeightMax 5.0 `
  -Device cpu `
  -Epochs 40 `
  -Objective macro_priority
```
