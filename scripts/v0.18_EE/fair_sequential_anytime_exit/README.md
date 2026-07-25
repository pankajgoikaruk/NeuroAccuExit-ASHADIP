# v0.18 — Fair Sequential Anytime Exit with Strict Risk Safeguards

v0.18 addresses the two main limitations identified in v0.17:

1. the available five-exit checkpoint was trained from a different manifest, making the architecture comparison invalid;
2. Exit 1 was useful but disproportionately responsible for quality loss, while the original risk score was practically non-binding.

## Phase A — fair five-exit retraining

The runner reads the canonical three-exit `config_used.json` and trains a five-exit no-hint model with the same:

- ten-label manifest and feature cache;
- label order and preprocessing;
- epochs, batch size, optimiser, learning rate and seed;
- threshold, positive weighting and sampling configuration.

The only topology change is `tap_blocks=1,2,3,4`. To prevent extra auxiliary heads from receiving a larger total supervision budget, the canonical three-exit auxiliary-loss sum is divided equally across the four five-exit auxiliary heads. The final-exit loss weight remains unchanged.

`audit_fair_training_v018.py` fails the pipeline if any required fairness field differs.

## Phase B — stricter sequential policy

Each non-final exit searches:

- mean confidence;
- inter-exit probability delta;
- base label-risk budget;
- risk-score threshold;
- high-risk margin multiplier;
- high-risk uncertainty band;
- Exit-1 confidence boost;
- one margin threshold per label.

The continuation-risk profile is derived from validation only. A label receives higher risk when the final exit frequently corrects that exit and/or provides a positive label-wise F1 gain.

Exit 1 has an additional confidence boost because it has no preceding exit for stability evidence. Any high-risk label near its decision threshold vetoes an early stop.

## Selection

The multi-objective search maximises FLOPs saved while constraining robust parent-level degradation in:

- Macro-F1;
- Micro-F1;
- Exact Match;
- Hamming Loss.

The selected point uses a 50% safety-buffered Pareto knee by default, rather than the aggressive boundary solution.

## Reporting

v0.18 reports validation eligibility and corrected-holdout compliance separately. It generates:

- fair-training audit;
- strict-risk profiles;
- Pareto fronts;
- 3-exit and 5-exit holdout tables;
- combined ablation table;
- fair architecture headline table;
- policy-structure table;
- per-exit coverage, FLOPs and measured latency.

## Run

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\v0.18_EE\fair_sequential_anytime_exit\run_v018_EE.ps1" `
  -TimingRepeats 30
```

The first run trains the fair five-exit checkpoint. Later runs automatically reuse it unless `-ForceRetrain5` is supplied.
