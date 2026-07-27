# v0.19_EE — Final targeted Exit 3 → Exit 5 experiment

This is the final planned sample-wise Early-Exit experiment. It reuses the fair v0.18 five-exit checkpoint and optimises exactly one stopping decision:

```text
Exit 3 → stop
or
Exit 3 → continue to Exit 5
```

Exits 1, 2 and 4 are disabled as stopping points. This directly tests the closest-to-feasible v0.18 ablation without reopening broad EE experimentation.

## Selection protocol

- tune only on the existing validation split;
- use frozen fixed-0.5 segment thresholds and frozen LATS-v2 parent evaluation;
- grouped robustness checks remain active;
- use a stricter 0.35 safety fraction by default;
- evaluate the corrected holdout once after freezing;
- require all four quality constraints and at least 5% FLOP savings.

## Final rule

If every holdout constraint passes and FLOP savings are at least 5%, finalise v0.19. Otherwise stop sample-wise EE development and retain v0.13 per-label margin as the safest adaptive baseline.

## Run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.19_EE\exit3_to_exit5_final\run_v019_EE.ps1" `
  -RunDir5 "N:\v018_fair5_20260725_231353" `
  -TimingRepeats 30
```

Outputs are written to:

```text
human_talk_workspace\active_budget_anytime_exit_v0.4\v0.19_EE\exit3_to_exit5_final\
```
