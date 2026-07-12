# v0.10_1 Low-Energy Recovery Ablation

This note records the v0.10_1 ablation for testing whether manually reviewed TATA-LAWYER low-energy one-second samples can improve the current NeuroAccuExit human-talk model without risking the original manifests.

## Safety policy

Original manifests are **read-only**.

The experiment should only:

1. read the original v0.10/v0.8-HCB training manifest;
2. read the TATA-LAWYER human-reviewed low-energy masked manifest;
3. read the corrected holdout manifest only for leakage checks;
4. write copied inputs and a new augmented manifest under a new v0.10_1 workspace.

No source manifest should be overwritten.

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

The default builder should be conservative:

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

## Feature-path decision

The ASHADIP loader uses `feat_relpath`. Therefore, TATA-LAWYER low-energy rows must resolve through the true low-energy feature cache, not through legacy parent/segment `feature_path` rows.

The verified local feature root is expected to be:

```text
C:\Users\wwwsa\PycharmProjects\TATA-LAWYER\human_talk_workspace\tata_v0.9_pipeline\tata_triage_model\silence_recovered_v09\feature_cache\features
```

The experiment should use:

```text
LowEnergyFeatureMode = feat_relpath
```

## Final result

The v0.10_1 low-energy recovery ablation completed successfully but did **not** beat the selected v0.10 no-hint LATS-v2 result.

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Selected v0.10 no-hint + LATS-v2 | **0.8624** | **0.9531** | **0.9589** | **0.8766** | **0.0137** |
| v0.10_1 low-energy recovery + LATS-v2 | 0.8581 | 0.9446 | 0.9519 | 0.8570 | 0.0160 |

Lower Hamming Loss is better. v0.10_1 is worse on every final parent-level metric.

## Research question

> Does adding linked, manually reviewed low-energy one-second evidence improve difficult labels such as `silence_present` and `audience_reaction_present` without hurting global multi-label consistency?

## Research finding

The answer from this ablation is **no**. The low-energy rows were linked correctly and no corrected-holdout parent leakage was detected, but adding 667 recovered low-energy training rows did not improve the final corrected-holdout parent-level result.

## Decision

```text
Do not promote v0.10_1 as the final model.
Keep selected v0.10 no-hint + LATS-v2 as the current best outcome.
Document v0.10_1 as a valid negative/diagnostic ablation.
```

## Future direction

A stronger version may require a masked-loss training path, label-aware calibration, or a separate low-energy specialist rather than directly mixing recovered low-energy rows into the main training distribution.
