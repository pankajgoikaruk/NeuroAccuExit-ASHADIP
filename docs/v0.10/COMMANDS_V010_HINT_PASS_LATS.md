# V0.10 Hint-Pass + LATS-v2 Commands

## Goal

V0.10 tests whether the old exit-to-exit hint-pass idea can improve the current ASHADIP/LABLEX pipeline when combined with the frozen LATS-v2 parent-level inference policy.

V0.9_4 baseline to beat:

| Metric | Value |
|---|---:|
| Macro-F1 | 0.8673 |
| Micro-F1 | 0.9458 |
| Samples-F1 | 0.9517 |
| Exact Match | 0.8604 |
| Hamming Loss | 0.0158 |

## Implementation status

The current `agentic_data_preprocessing_v0.10` branch already has the model-side hint path:

- `models/exit_net.py` supports `hint_dim`, `hint_source`, `hint_detach`, and optional hint statistics.
- `utils/model_factory.py` reads `model.exit_hint` from YAML and passes hint settings into `ExitNet`.
- `scripts/run_full.ps1` accepts `-ExitHint "true|false"` and forwards it to `training.train`.

Therefore v0.10 should focus on controlled ablation and parent-level evaluation, not rewriting the whole model.

---

## Step 1: Run 3-exit no-hint vs hint-pass ablation

Run from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v0.10\run_v010_hint_pass_ablation.ps1 `
  -DataRoot "data\moth_sounds" `
  -CacheRoot "data_caches" `
  -RunsRoot "runs_v0.10_hint_pass" `
  -Config "configs\audio_moth.yaml" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -InputMode "segment" `
  -RunClipPolicy
```

For a ready train/val/test layout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v0.10\run_v010_hint_pass_ablation.ps1 `
  -DataRoot "PATH_TO_READY_DATA" `
  -InputMode "ready" `
  -RunsRoot "runs_v0.10_hint_pass" `
  -Device "cpu" `
  -TapBlocks "1,3" `
  -RunClipPolicy
```

This creates two variants:

```text
v010_1-3_no_hint_<timestamp>
v010_1-3_hint_pass_<timestamp>
```

---

## Step 2: Compare model-side greedy/clip results

Inspect generated summaries under:

```text
runs_v0.10_hint_pass/<variant>/<variant>_###/
```

Compare no-hint and hint-pass on:

- final exit accuracy / macro-F1
- greedy policy accuracy
- average exit depth
- clip-level accuracy
- compute saved
- flip rate / consistency

---

## Step 3: Apply frozen LATS-v2 to a segment probability CSV

When you have a segment-level probability CSV for a v0.10 candidate in the same schema as v0.9:

```text
parent_clip_id
Brene_Brown
...
silence_present
exit3_prob_Brene_Brown
...
exit3_prob_silence_present
```

run:

```powershell
$SegmentPredCsv = "PATH_TO_V010_SEGMENT_PROBS.csv"
$ConfigJson = "docs\tables\agentic_data_preprocessing_v0.9\v0.9_lats_v2\lats_v2_final_frozen_config.json"
$OutDir = "human_talk_workspace\tata_v0.10_hint_pass\frozen_lats_v2_eval"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python scripts\v0.10\evaluate_frozen_lats_config_v010.py `
  --segment-pred-csv "$SegmentPredCsv" `
  --labels-json "configs\human_talk_10label_schema.json" `
  --config-json "$ConfigJson" `
  --out-dir "$OutDir" `
  --parent-id-col "parent_clip_id" `
  --prob-prefix "exit3_prob_" `
  --model-name "v0.10_hint_pass_candidate"
```

Inspect:

```powershell
Import-Csv "$OutDir\v010_frozen_lats_eval.csv" |
  Format-Table method, macro_f1, micro_f1, samples_f1, exact_match, hamming_loss, avg_pred_labels -AutoSize
```

---

## Acceptance rule

Accept a v0.10 hint-pass candidate only if it beats LATS-v2 clearly or gives similar quality with better compute:

```text
Macro-F1   > 0.8673
Micro-F1   >= 0.9458
Samples-F1 >= 0.9517
Exact      >= 0.8604
Hamming    <= 0.0158
```

If quality is similar, also consider:

```text
lower average exit depth
higher compute saved
lower flip rate
better exit consistency
```

---

## Interpretation

V0.9 improved the parent-level decision policy without retraining.

V0.10 tests whether architectural hint passing can improve the segment-level probabilities that are later consumed by the frozen LATS-v2 parent-level inference policy.
