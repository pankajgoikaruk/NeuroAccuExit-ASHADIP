# v0.10_1 Low-Energy Recovery Ablation

This branch tests whether the manually reviewed TATA-LAWYER low-energy one-second samples can improve the current NeuroAccuExit human-talk model without risking the original manifests.

## Safety policy

Original manifests are **read-only**.

The implementation only:

1. reads the original v0.10/v0.8-HCB training manifest;
2. reads the TATA-LAWYER human-reviewed low-energy masked manifest;
3. reads the corrected holdout manifest only for leakage checks;
4. writes copied inputs and a new augmented manifest under a new v0.10_1 workspace.

No source manifest is overwritten.

## Source manifests

Base training manifest:

```text
human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/final_expanded_training_dataset_balanced/metadata/multilabel_features_manifest_balanced.csv
```

TATA-LAWYER human-reviewed low-energy/masked manifest:

```text
human_talk_workspace/tata_v0.9_pipeline/tata_triage_model/silence_recovered_v09/human_reviewed_masked_v09/feature_cache/metadata/multilabel_features_manifest_v09_HUMAN_REVIEWED_MASKED.csv
```

Corrected holdout manifest:

```text
human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv
```

## Output workspace

```text
human_talk_workspace/tata_v0.10_1_low_energy_recovery_ablation/
```

Important outputs:

```text
metadata/multilabel_features_manifest_v010_1_LOW_ENERGY_AUGMENTED.csv
metadata/selected_low_energy_rows_used.csv
metadata/build_v010_1_low_energy_augmented_report.json
audit/low_energy_audit_report.json
```

## Default build rules

The default builder is conservative:

| Rule | Default |
|---|---|
| Use source manifests directly? | No, copy first |
| Append low-energy rows to training? | Yes |
| Touch corrected holdout? | No |
| Include low-energy split | `train` only |
| Include partially unknown/masked rows | No |
| Exclude corrected-holdout parent overlap | Yes |
| Use hint-pass | No |
| Use pos_weight | No |

Reason: the current `training.train_multilabel` path does not apply per-label masks, so the first safe ablation uses only fully known reviewed low-energy rows.

## One-command experiment

From repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v0.10_1\run_v010_1_no_hint_low_energy_ablation.ps1 `
  -Device cpu `
  -Epochs 40 `
  -Objective macro_priority
```

The command performs:

1. low-energy audit;
2. copied/augmented manifest build;
3. no-hint training;
4. corrected-holdout parent-level evaluation;
5. LATS-v2 coordinate re-optimization;
6. final metric summary.

## Acceptance target

Promote v0.10_1 only if it beats the selected v0.10 no-hint result:

| Metric | Target |
|---|---:|
| Macro-F1 | >= 0.8624 |
| Micro-F1 | >= 0.9531 |
| Samples-F1 | >= 0.9589 |
| Exact Match | >= 0.8766 |
| Hamming Loss | <= 0.0137 |

## Expected research question

> Does adding linked, manually reviewed low-energy one-second evidence improve difficult labels such as `silence_present` and `audience_reaction_present` without hurting global multi-label consistency?

## Decision rule

If v0.10_1 improves `silence_present` but hurts Exact/Hamming, document it as a useful low-energy diagnostic rather than a new final model.

If it improves global metrics and bottleneck labels, promote it as the new selected v0.10_1 result.
