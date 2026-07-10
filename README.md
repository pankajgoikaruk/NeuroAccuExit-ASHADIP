# NeuroAccuExit-ASHADIP — Active Budget and Anytime Exit v0.1

This branch extends the human-talk multi-label NeuroAccuExit system from a verified **full-depth reference** toward **standard Early-Exit**, **budget-aware Early-Exit**, and **anytime inference**.

The previous v0.10 work established the strongest reproducible final-exit inference configuration. This branch freezes that result and uses it as the sole full-computation quality reference for all subsequent efficiency experiments.

---

## Branch identity

| Item | Value |
|---|---|
| Git branch | `active_budget_anytime_exit_v0.1` |
| Documentation name | **NeuroAccuExit — Active Budget and Anytime Exit v0.1** |
| Source branch | `agentic_data_preprocessing_v0.10` |
| Task | Human-talk multi-label speaker/context detection |
| Current phase | Full-depth baseline frozen; standard Early-Exit is the next experiment |
| Main purpose | Measure and control the quality–computation trade-off across exits |

This branch is intentionally restricted to:

1. standard sample-wise Early-Exit;
2. true staged inference that skips unnecessary deeper computation;
3. budget-aware Early-Exit;
4. anytime quality-versus-cost evaluation.

The branch does not reopen the completed hint-pass or `pos_weight cap5` investigations unless a new, clearly defined hypothesis requires them.

---

## Current branch decision

| Decision item | Outcome |
|---|---|
| Canonical full-depth baseline | `v0.10 no-hint + frozen historical LATS-v2` |
| Full-depth probability source | Exit 3, using columns prefixed by `exit3_prob_` |
| Parent-level policy | Frozen label-specific LATS-v2 aggregation and thresholds |
| Secondary frozen result | Direct coordinate re-optimisation; ablation only |
| Standard hint-pass | Not selected for the current multi-label dataset |
| `pos_weight cap5` | Not selected |
| Next experiment | Standard Early-Exit evaluation against the frozen baseline |
| Future experiments | Budget-aware Early-Exit and anytime inference |

---

## Dataset and evaluation setting

| Item | Value |
|---|---|
| Task | Human-talk multi-label speaker/context detection |
| Parent clips | 867 |
| Segments | 4,335 |
| Training rows | 25,519 |
| Validation rows | 1,883 |
| Test rows | 1,961 |
| Labels | 10 |
| Label schema | `configs/human_talk_10label_schema.json` |
| Training manifest | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/final_expanded_training_dataset_balanced/metadata/multilabel_features_manifest_balanced.csv` |
| Corrected holdout manifest | `human_talk_workspace/tata_v0.8_human_corrected_balanced_pipeline/corrected_holdout/multilabel_features_manifest_CORRECTED_LABELS.csv` |
| Parent identifier | `parent_clip_id` |
| Full-depth probability prefix | `exit3_prob_` |

Labels:

```text
Brene_Brown
Eckhart_Tolle
Eric_Thomas
Gary_Vee
Jay_Shetty
Nick_Vujicic
other_speaker_present
music_present
audience_reaction_present
silence_present
```

---

## Canonical full-depth baseline

The official quality reference for this branch is:

```text
v0.10 no-hint + frozen historical LATS-v2
```

It uses final-exit segment probabilities and the exact committed historical LATS-v2 parent-level configuration. Reproduction applies the frozen rules directly; it performs no neural-network retraining, threshold search, or aggregation-method search.

### Exact reproduced result

| Metric | Exact value | Paper value |
|---|---:|---:|
| Macro-F1 | 0.8623815322333925 | **0.8624** |
| Micro-F1 | 0.9531311539976368 | **0.9531** |
| Samples-F1 | 0.9588894381281925 | **0.9589** |
| Exact Match | 0.8765859284890427 | **0.8766** |
| Hamming Loss ↓ | 0.013725490196078431 | **0.0137** |
| Average predicted labels | 1.4590542099192618 | 1.4591 |
| Parent clips | 867 | 867 |

### Interpretation

- This is a **full-depth, final-exit** result.
- It is the only baseline used to calculate Early-Exit quality retention or degradation.
- `1.4591` is the average number of predicted positive labels per parent clip.
- `1.4591` is **not** average exit depth.
- Average exit depth, exit distribution, computation saving, and latency will be introduced by the Early-Exit experiments.

---

## Secondary frozen result

The direct coordinate re-optimisation result is retained as a post-hoc inference-policy ablation:

| Metric | Secondary result | Difference from canonical |
|---|---:|---:|
| Macro-F1 | 0.8598605 | −0.0025210 |
| Micro-F1 | 0.9547260 | +0.0015948 |
| Samples-F1 | 0.9619926 | +0.0031032 |
| Exact Match | 0.8800461 | +0.0034602 |
| Hamming Loss ↓ | 0.0131488 | −0.0005767 |
| Average predicted labels | 1.4348328 | −0.0242215 |

This variant improves several global/sample-level metrics but reduces Macro-F1 and does not reproduce the original historical LATS-v2 procedure. It is therefore frozen for analysis but must not replace the canonical baseline.

---

## Why this baseline was selected

The canonical result was selected because it:

1. exactly matches the committed historical v0.10 LATS-v2 configuration;
2. reproduces all metrics deterministically from the frozen probability CSV;
3. represents the complete final-exit computation path;
4. preserves the original label-specific aggregation and threshold choices;
5. provides a stable reference for standard, budget-aware, and anytime inference;
6. is fully packaged with inputs, outputs, code snapshots, environment information, and integrity hashes.

The earlier `v0.9_4 / LATS-v2` result remains an important historical stability result from the previous research phase. However, the experiments in `active_budget_anytime_exit_v0.1` use the verified v0.10 no-hint frozen result above as their branch-specific full-depth comparator.

---

## Frozen LATS-v2 parent-level rules

| Label | Aggregation | Threshold |
|---|---|---:|
| Brene Brown | `p75` | 0.54 |
| Eckhart Tolle | `top3mean` | 0.50 |
| Eric Thomas | `top4mean` | 0.62 |
| Gary Vee | `mean` | 0.50 |
| Jay Shetty | `p75` | 0.91 |
| Nick Vujicic | `p75` | 0.34 |
| Other speaker present | `noisy_or` | 0.94 |
| Music present | `mean` | 0.37 |
| Audience reaction present | `top3mean` | 0.23 |
| Silence present | `p75` | 0.42 |

These rules are applied to final-exit segment probabilities to obtain one prediction vector per parent clip.

---

## Reproducibility package

The complete frozen package is stored at:

```text
docs/tables/active_budget_anytime_exit_v0.1/full_depth_baselines/
```

Key contents:

| Path | Purpose |
|---|---|
| `README.md` | Package overview and canonical-baseline declaration |
| `PAPER_READY_BASELINE_SUMMARY.md` | Concise research-paper description |
| `reproducibility_manifest.json` | Machine-readable experiment settings and metrics |
| `artifact_hashes.csv` | SHA256 integrity manifest |
| `environment_summary.txt` | Git, Python, and core-package information |
| `environment_pip_freeze.txt` | Complete Python package snapshot |
| `primary_v010_no_hint_historical_lats_v2/` | Canonical baseline artifacts and detailed record |
| `secondary_direct_coordinate_reoptimized/` | Secondary ablation artifacts and record |
| `shared_reproducibility_inputs/` | Frozen probabilities, label schema, configuration, and script snapshots |
| `reproduced_outputs/primary_historical_lats_v2/` | Independently regenerated canonical outputs |

### Exact reproduction command

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass `
  -File "docs\tables\active_budget_anytime_exit_v0.1\full_depth_baselines\primary_v010_no_hint_historical_lats_v2\REPRODUCE_PRIMARY.ps1"
```

Expected metrics:

```text
Macro-F1        = 0.8623815322333925
Micro-F1        = 0.9531311539976368
Samples-F1      = 0.9588894381281925
Exact Match     = 0.8765859284890427
Hamming Loss    = 0.013725490196078431
Avg pred labels = 1.4590542099192618
Parent clips    = 867
```

---

## Historical v0.10 findings retained

The new branch does not invalidate the earlier v0.10 analysis.

| Finding | Decision |
|---|---|
| Frozen v0.9_4 LATS-v2 did not transfer reliably to new v0.10 probabilities | Retained |
| v0.10-specific LATS re-optimisation recovered performance | Retained |
| Standard hint-pass did not beat no-hint after recalibration | Retained as a negative result |
| `pos_weight cap5` did not improve the final outcome | Retained as a negative result |
| Label-specific parent-level inference was more useful than the tested architecture changes | Retained |
| No-hint v0.10 produced the strongest branch-specific full-depth reference | Adopted for this branch |

---

## Experiment roadmap

### Stage 0 — Full-depth reference

Status: **complete and frozen**

- Exit 3 segment probabilities;
- frozen parent-level LATS-v2;
- exact deterministic reproduction;
- primary and secondary results preserved;
- paper-ready documentation generated.

### Stage 1 — Standard Early-Exit

Status: **next**

Evaluate sample-wise exit decisions at Exit 1, Exit 2, or Exit 3.

Required outputs:

- Macro-F1, Micro-F1, Samples-F1, Exact Match, and Hamming Loss;
- exit distribution;
- average exit depth;
- full-depth agreement;
- quality loss relative to the canonical baseline;
- estimated computation saving.

### Stage 2 — True staged inference

Status: **planned**

Replace post-hoc policy selection with execution that genuinely stops computation and skips deeper CNN blocks.

Required outputs:

- measured latency;
- cumulative FLOPs or a validated compute proxy;
- realised computation saving;
- consistency between simulated and true staged decisions.

### Stage 3 — Budget-aware Early-Exit

Status: **planned**

Introduce explicit cost constraints, such as:

- maximum exit depth;
- per-sample compute budget;
- average dataset-level budget;
- target quality-retention constraint;
- dynamic allocation based on uncertainty.

### Stage 4 — Anytime inference

Status: **planned**

Produce predictions and quality measurements at increasing computation budgets.

Required outputs:

- quality-versus-cost curves;
- performance at each exit;
- area under the anytime curve where appropriate;
- per-label and global behaviour across budgets;
- comparison with full-depth and standard Early-Exit.

---

## Evaluation protocol

Every future result must be compared with the canonical full-depth baseline.

Recommended comparison columns:

| Category | Measures |
|---|---|
| Prediction quality | Macro-F1, Micro-F1, Samples-F1, Exact Match, Hamming Loss |
| Exit behaviour | Exit-1/Exit-2/Exit-3 fractions, average exit depth |
| Efficiency | Cumulative FLOPs, estimated saving, measured latency |
| Reliability | Agreement with full-depth prediction, flip rate, per-label degradation |
| Budget behaviour | Budget used, budget violations, quality at fixed budgets |
| Anytime behaviour | Quality at each computation point and quality-versus-cost curve |

For each quality metric \(M\), report both the absolute Early-Exit result and its change from the canonical reference:

```text
quality_change = early_exit_metric - full_depth_metric
```

For Hamming Loss and cost, lower values are better and the direction must be interpreted accordingly.

---

## Branch research questions

| ID | Research question |
|---|---|
| RQ1 | How much full-depth multi-label quality can standard Early-Exit retain? |
| RQ2 | Which confidence or stability policy gives the best quality–depth trade-off? |
| RQ3 | Do simulated Early-Exit savings remain valid under true staged execution? |
| RQ4 | How should a limited compute budget be allocated across samples? |
| RQ5 | How does prediction quality evolve as more computation becomes available? |
| RQ6 | Which labels are safe to decide early, and which require deeper evidence? |
| RQ7 | Can budget-aware or label-aware policies outperform one global exit rule? |

---

## Paper-ready baseline statement

> The full-computation reference used the final exit of the three-exit no-hint model followed by the frozen historical LATS-v2 parent-level inference policy. Across 867 parent clips and 10 labels, it achieved a Macro-F1 of 0.8624, Micro-F1 of 0.9531, Samples-F1 of 0.9589, Exact Match of 0.8766, and Hamming Loss of 0.0137. This deterministic frozen result was used as the canonical quality reference for all subsequent standard Early-Exit, budget-aware Early-Exit, and anytime-inference evaluations.

---

## Reporting cautions

- Call this a **frozen corrected-holdout evaluation**.
- Do not describe it as an independent external test set.
- Do not report average predicted labels as average exit depth.
- Do not compare Early-Exit policies against whichever historical row is most favourable; use the canonical branch baseline consistently.
- Distinguish simulated policy selection from true computation-saving staged inference.
- Preserve the secondary result as an ablation rather than silently replacing the baseline.

---

## Current next action

```text
Implement and evaluate the standard Early-Exit baseline on
active_budget_anytime_exit_v0.1, using the frozen v0.10 no-hint
historical LATS-v2 result as the sole full-depth quality reference.
```
