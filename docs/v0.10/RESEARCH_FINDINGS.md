# v0.10 Research Findings

## Research questions

| ID | Research question | Finding |
|---|---|---|
| RQ1 | Can standard exit-to-exit hint-pass improve the human-talk multi-label pipeline? | No. It did not outperform the no-hint control after recalibration. |
| RQ2 | Does frozen v0.9_4 LATS-v2 transfer to retrained v0.10 models? | No. Frozen transfer underperformed, showing that thresholds must be recalibrated when probabilities shift. |
| RQ3 | Does v0.10-specific LATS re-optimization help? | Yes. It strongly improves both no-hint and hint-pass outputs compared with frozen transfer. |
| RQ4 | Can v0.10 no-hint replace v0.9_4? | Not yet. It is promising in some seeds but not stable across the 3-seed check. |
| RQ5 | What is the strongest current contribution? | LATS-v2 metric-aware inference-policy optimization remains the strongest stable contribution. |

---

## Why hint-pass likely failed

The previous hint-pass idea was more suitable for simpler single-label/binary audio tasks. The human-talk task is multi-label: one parent clip can contain a target speaker, other speaker, music, audience reaction, and silence indicators together.

In this setting, early exits may produce incomplete label evidence. Passing those early probabilities forward can propagate bias into later exits. This is especially risky for rare or bursty labels such as:

```text
audience_reaction_present
silence_present
```

The result is that hint-pass may improve some raw label scores but hurt global multi-label consistency.

---

## Why no-hint v0.10 sometimes improved

v0.10 no-hint is a newly trained checkpoint. Even with the same architecture, the probabilities differ from v0.9_4 due to random seed, batch ordering, checkpoint selection, and validation dynamics. After LATS-v2 re-optimization, some seeds produced better global metrics.

Therefore, the v0.10 no-hint gain is best described as:

```text
retrained probability distribution + v0.10-specific LATS calibration
```

not as a hint-pass gain.

---

## Future work

Standard hint-pass should not be continued in its current form. Future work should test more selective hinting:

1. Hint only speaker labels, not context labels.
2. Use hints only from Exit 2 to Exit 3.
3. Add learnable label-wise hint gates.
4. Use separate speaker/context hint vectors.
5. Use consistency regularization instead of feeding raw previous-exit probabilities.
6. Combine LATS with uncertainty or calibration-aware thresholding.

---

## Paper/report wording

> The v0.10 hint-pass ablation shows that standard exit-to-exit probability hinting does not directly transfer from simpler audio classification settings to multi-label speaker/context detection. Re-optimized LATS substantially improves v0.10 outputs, confirming the importance of model-specific inference calibration. However, hint-pass remains weaker than the no-hint control, and the promising no-hint gains are not stable across seeds. This supports retaining v0.9_4 LATS-v2 as the stable final baseline while documenting v0.10 as a diagnostic ablation and negative hint-pass result.
