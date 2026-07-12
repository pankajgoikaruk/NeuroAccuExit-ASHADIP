# Active Budget and Anytime Exit v0.2

## Branch

```text
active_budget_anytime_exit_v0.2
```

Source branch:

```text
active_budget_anytime_exit_v0.1
```

This branch is dedicated exclusively to:

1. standard multi-label Early-Exit;
2. budget-aware Early-Exit;
3. anytime inference evaluation.

The frozen v0.10 no-hint historical LATS-v2 result remains the sole canonical full-depth quality reference.

## Canonical baseline

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8623815322333925 |
| Micro-F1 | 0.9531311539976368 |
| Samples-F1 | 0.9588894381281925 |
| Exact Match | 0.8765859284890427 |
| Hamming Loss | 0.013725490196078431 |
| Average exit depth | 3.0 |
| Compute saved | 0% |

The historical value `1.4590542099192618` is average predicted labels per parent clip, not average exit depth.

## Repository audit

Most of the required foundation already exists and should be reused.

| Existing component | Reuse decision |
|---|---|
| `adapters/audio_adapter.py` | Reuse the five ordered CNN blocks and configurable tap blocks. |
| `models/exit_net.py` | Reuse trained early-exit heads, final head, and optional hint construction. Do not change the established training forward path. |
| `utils/model_factory.py` | Reuse existing checkpoint-compatible model construction. Load the checkpoint before wrapping the model for staged inference. |
| `scripts/multilabel_greedy_policy.py` | Reuse sigmoid probabilities, per-exit thresholds, multi-label metrics, fixed-exit metrics, label-set stability, exit distributions, and policy sweeps. Its current compute saving is a post-hoc depth proxy. |
| `utils/profiling.py` | Reuse cumulative TinyAudioCNN FLOP estimates and latency timing utilities. Extend later for staged exit-specific latency. |
| `policies/early_exit.py` | Legacy softmax/conformal policy; retain for historical compatibility but do not use directly as the main multi-label policy. |
| `scripts/policy_test.py` | Legacy single-label greedy evaluator; useful as a structural reference, but it computes every exit before choosing one. |
| Frozen v0.1 reproducibility package | Reuse unchanged as the full-depth baseline package. |

## Confirmed implementation gap

The existing backbone `forward()` executes all CNN blocks before returning the intermediate taps and final feature. Existing dynamic policy scripts therefore choose an exit after all exit outputs have already been computed.

That is valid for policy simulation but is not real computational Early-Exit.

The first v0.2 implementation milestone is a staged path that can stop before deeper blocks execute.

## Implemented in v0.2

### `models/anytime_exit_net.py`

Adds an inference-only wrapper around the existing `ExitNet`.

The wrapper:

- adds no trainable parameters;
- changes no trained weights;
- preserves the original `ExitNet.forward()` for training and historical evaluation;
- executes only the blocks needed to reach the next exit;
- carries the feature map and optional hint between exits;
- supports both the current three-exit model and configurable five-exit models;
- exposes:

```python
logits1, state = anytime_model.start(x)
logits2, state = anytime_model.continue_from(state)
logits3, state = anytime_model.continue_from(state)
```

For the canonical `tap_blocks=(1, 3)` model:

```text
Exit 1: block 1
Exit 2: blocks 2-3
Exit 3: blocks 4-5
```

A policy can now stop after Exit 1 or Exit 2 without invoking later blocks.

### `tests/test_anytime_exit_net.py`

Adds numerical equivalence tests for:

- three-exit no-hint execution;
- five-exit no-hint execution;
- three-exit hint-compatible execution;
- expected block and exit-state progression;
- rejection of continuation after the final exit.

Run from the repository root:

```powershell
conda activate ASHADIP_V0
$env:PYTHONPATH = (Get-Location).Path
python -m unittest tests.test_anytime_exit_net -v
```

Required result:

```text
OK
```

The tests compare staged logits with the unchanged full-forward logits using strict floating-point tolerances.

## Checkpoint usage

Build and load the existing model exactly as before, then wrap it:

```python
base_model = build_audio_exit_net(
    num_classes=num_classes,
    n_mels=n_mels,
    tap_blocks=(1, 3),
    model_cfg=run_model_cfg,
)
base_model.load_state_dict(checkpoint)
base_model.eval()

anytime_model = AnytimeExitNet(base_model)
anytime_model.eval()
```

Do not load the historical checkpoint directly into the wrapper, because the checkpoint belongs to the underlying `ExitNet` state dictionary.

## Next implementation sequence

### Milestone 1 — staged equivalence on the canonical checkpoint

Run the staged/full equivalence test with the actual canonical no-hint checkpoint and representative holdout batches.

Required checks:

```text
staged Exit 1 logits == full-forward Exit 1 logits
staged Exit 2 logits == full-forward Exit 2 logits
staged Exit 3 logits == full-forward Exit 3 logits
```

### Milestone 2 — fixed-exit quality audit

Export and evaluate:

```text
Always Exit 1
Always Exit 2
Always Exit 3
```

At segment and parent level.

Initially transfer the frozen historical LATS-v2 parent rules to all exits and label those results clearly as a frozen-policy transfer diagnostic. Do not claim that the historical Exit 3 thresholds are optimal for Exit 1 or Exit 2.

### Milestone 3 — standard Early-Exit

Reuse the existing multi-label policy utilities and add a staged stopping controller based on validation-selected combinations of:

- label-set stability;
- per-label threshold margin;
- confidence summaries;
- probability-vector change;
- empty-label-set safeguards.

### Milestone 4 — cost model

Extend the existing profiling code to report:

- cumulative FLOPs at each exit;
- incremental FLOPs between exits;
- batch-size-1 CPU latency by exit;
- normalized cost relative to Exit 3.

### Milestone 5 — budget-aware Early-Exit

Add explicit stop reasons:

```text
reliable_early_exit
budget_forced_exit
final_exit
```

The controller must compare the remaining budget with the incremental cost of reaching the next exit.

### Milestone 6 — anytime evaluation

Evaluate several normalized computation budgets and create quality-versus-cost Pareto curves.

## Rules for this branch

- Do not reopen hint-pass or low-energy recovery as primary experiments.
- Do not alter the canonical frozen baseline.
- Do not call post-hoc exit selection real compute saving.
- Tune policy thresholds on validation/calibration data only.
- Preserve per-label metrics, especially for rare/context labels.
- Keep the existing training path backward-compatible.
- Maintain Windows PowerShell commands and complete experiment records.
