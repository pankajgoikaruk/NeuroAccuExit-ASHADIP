# Multi-Label Experiment Log

This document is a quick memory book for the multi-label NeuroAccuExit work. It records the main branches, methods, motivation, outcomes, and lessons learned so future experiments do not lose context.

```text
Project: NeuroAccuExit-ASHADIP
Scope: Multi-label K-exit audio classification and dynamic early-exit policy
Current file location: docs/MULTILABEL_EXPERIMENT_LOG.md
Reference baseline branch: kexit_multi-label_greedy_EE
Follow-up branch discussed: kexit_multi-label_EE_lossweight
```

---

## 1. Why we moved to multi-label

Earlier versions mainly treated the audio task as single-label or multi-class:

```text
one audio segment -> one class
```

However, real environmental audio can contain overlapping events:

```text
rain + thunderstorm
road_traffic + gun_shot
fireworks + conversation
wind + rain
```

Therefore, the task was reformulated as:

```text
one audio segment -> multiple possible labels
```

This required moving from:

```text
softmax + CrossEntropyLoss
```

to:

```text
sigmoid + BCEWithLogitsLoss
```

### Verdict

Helpful and necessary. Multi-label formulation is the correct direction for overlapping audio events.

---

## 2. Multi-label dataset and synthetic mixture stage

### Method

A multi-label data pipeline was introduced:

```text
clean class-wise audio
-> seed manifest
-> synthetic two-label mixtures
-> multi-hot labels
-> log-mel features
-> train/val/test manifest
```

Example target for `rain + thunderstorm`:

```text
[0,0,0,0,0,1,0,0,1,0]
```

### Why this was done

The project needed overlapping-label examples. Since naturally mixed audio was limited, synthetic two-label mixtures provided a controlled way to train and evaluate multi-label behaviour.

### Outcome

Helpful. It enabled a proper multi-label training and evaluation setup.

### Limitation

Synthetic mixtures are not a complete replacement for real overlapping audio scenes. A real mixed-audio test set is still needed later.

---

## 3. Multi-label training setup

### Likely branch family

```text
kexit_cclass_greedy_multi-label
```

### Main method

The model was adapted for multi-label prediction using:

```text
TinyAudioCNN + ExitNet
multi-label logits at every exit
sigmoid probabilities
BCEWithLogitsLoss
```

Important files/scripts include:

```text
training/train_multilabel.py
data/datasets_multilabel.py
scripts/tune_multilabel_thresholds.py
scripts/summarize_multilabel_threshold_runs.py
```

### Why this was done

Multi-label audio tagging requires independent prediction for each label rather than one mutually exclusive softmax class.

### Outcome

Helpful, but fixed threshold `0.5` was not enough.

### Main lesson

The model could perform multi-label prediction, but threshold selection became a critical part of the method.

---

## 4. Threshold tuning

### Method

Instead of using only:

```text
probability >= 0.5
```

we tuned thresholds per label and later per exit/per label:

```text
Exit 1 threshold for each label
Exit 2 threshold for each label
Exit 3 threshold for each label
...
```

### Why this was done

Different labels have different score distributions. Some labels require lower thresholds to recover recall; other labels require higher thresholds to reduce false positives.

### Outcome

Strongly helpful.

### Main lesson

Per-label and per-exit threshold tuning is essential for fair multi-label evaluation.

---

## 5. Positive-label weighting

### Likely branch family

```text
kexit_cclass_greedy_multi-label_pos-weight
```

### Method

Positive-label weighting was added through:

```powershell
--use_pos_weight
--pos_weight_max 20.0
```

### Why this was done

Some labels were harder or less frequent. Positive weighting was introduced to make BCE loss pay more attention to positive labels.

### Outcome

Mixed but useful.

It improved label-balanced behaviour and helped macro-F1, but it also increased the risk of over-predicting labels.

### Main lesson

Positive weighting improves recall-sensitive learning, but it can increase false positives and predicted-label count.

---

## 6. Static per-exit analysis

### Method

Each exit was evaluated independently:

```text
Exit 1 quality
Exit 2 quality
Exit 3 quality
Exit 4 quality
Exit 5 quality
```

Metrics:

```text
macro-F1
micro-F1
samples-F1
exact match
hamming loss
average predicted labels
```

### Why this was done

Before using dynamic early exit, we needed to know whether each exit was reliable enough to make predictions.

### Outcome

Very helpful.

### Key findings

```text
Exit 1 is weak.
Exit 2 is somewhat useful.
Exit 4 is the strongest practical intermediate exit.
The final exit is not always the best exit.
```

### Important result from greedy-EE baseline

```text
5exit_posweight Exit 4 macro-F1 = 0.6538
```

### Main lesson

A multi-exit network should not be judged only by its final head. Intermediate exits can be strong and sometimes better than the final exit for macro-F1.

---

## 7. Greedy label-set stability policy

### Branch family

```text
kexit_cclass_greedy_multi-label_policy
kexit_multi-label_greedy_EE
```

### Main script

```text
scripts/multilabel_greedy_policy.py
```

### Method

The policy works as follows:

```text
1. Run all exits.
2. Convert sigmoid probabilities into multi-label sets using tuned thresholds.
3. Start checking from min_exit.
4. Stop when the predicted label set is stable for stable_k consecutive considered exits.
5. If no stable decision is found, fall back to the final exit.
```

Example:

```text
Exit 1 predicts: rain + wind
Exit 2 predicts: rain + wind
If stable_k = 2, stop at Exit 2.
```

### Why this was done

In multi-label early exit, confidence alone is not enough. The whole predicted label set must become stable.

### Outcome

Helpful and necessary. It added dynamic-neural-network evidence:

```text
where samples exit
average exit depth
estimated compute saved
dynamic macro-F1 and hamming loss
```

### Main lesson

Label-set stability is a reasonable first dynamic policy for multi-label early-exit inference.

---

## 8. Policy 001: conservative policy

### Branch

```text
kexit_multi-label_greedy_EE
```

### Setting

```text
min_exit = 2
stable_k = 2
allow_empty_stop = False
```

### Why this was done

Exit 1 was weak, so this policy intentionally ignored Exit 1.

### Outcome

Helpful for 5-exit models, but not useful for 3-exit compute saving.

For 3-exit models:

```text
Exit 1 ignored.
Exit 2 is first considered.
stable_k=2 requires agreement between Exit 2 and Exit 3.
Earliest stable stop becomes final Exit 3.
Compute saved = 0%.
```

For 5-exit models, it produced useful dynamic behaviour:

```text
5exit_nohint saved about 16% depth compute.
5exit_posweight saved about 14% depth compute.
```

### Verdict

Good conservative policy, but not suitable for proving 3-exit Tiny early-exit behaviour.

---

## 9. Policy 002: Exit-1-enabled fair policy

### Branch

```text
kexit_multi-label_greedy_EE
```

### Setting

```text
min_exit = 1
stable_k = 2
allow_empty_stop = False
```

### Why this was done

This policy tested whether the compact 3-exit model could behave as a real early-exit model.

### Outcome

Helpful, but savings were small.

Results from greedy-EE baseline:

```text
3exit_nohint:    3.84% compute saved
3exit_posweight: 2.53% compute saved
```

Macro-F1 dropped slightly compared with the conservative/final-exit result.

### Verdict

Good fairness test. It proved that 3-exit can early-exit, but also showed that early exits were not strong enough yet.

---

## 10. Policy 003: aggressive early-exit ablation

### Branch

```text
kexit_multi-label_greedy_EE
```

### Setting

```text
min_exit = 1
stable_k = 1
allow_empty_stop = False
```

### Why this was done

This was an ablation to test what happens if the model stops immediately at Exit 1.

### Outcome

Strongly degraded prediction quality.

Compute saving was large:

```text
66% to 80%
```

but macro-F1 collapsed to roughly:

```text
0.38 to 0.43
```

### Verdict

Not a successful policy. Useful only as evidence that Exit 1 is not reliable enough for immediate stopping.

### Main lesson

High compute saving is not meaningful if the early prediction is poor.

---

## 11. Policy 004: allow empty-label stopping

### Branch

```text
kexit_multi-label_greedy_EE
```

### Setting

```text
min_exit = 1
stable_k = 2
allow_empty_stop = True
```

### Why this was done

To check whether allowing early stopping on an empty predicted label set improves efficiency.

### Outcome

Almost no useful benefit.

It gave tiny extra saving for some 5-exit models, but no meaningful accuracy advantage.

### Verdict

Appendix-only ablation, not a main policy.

---

## 12. Best greedy-EE baseline result

### Branch

```text
kexit_multi-label_greedy_EE
```

### Best practical result

```text
Model: 5exit_posweight
Policy: min_exit=3, stable_k=2
Macro-F1: 0.6449
Samples-F1: 0.6690
Average exit depth: 4.5337 / 5
Estimated depth-compute saved: 9.33%
```

### Why it worked

This policy avoids the weakest early exits and waits until later exits are more reliable.

```text
Exit 1: too weak
Exit 2: moderate
Exit 3 onwards: more stable
Exit 4: strongest static candidate
```

### Verdict

Strong locked baseline for dynamic multi-label early exit.

---

## 13. Loss-weight branch

### Branch

```text
kexit_multi-label_EE_lossweight
```

### Method

Only the exit loss weights were changed.

Previous likely weights:

```text
3-exit: [0.3, 0.3, 1.0]
5-exit: [0.3, 0.3, 0.6, 0.8, 1.0]
```

New tested weights:

```text
3-exit:
LW060 = [0.6, 0.6, 1.0]
LW080 = [0.8, 0.6, 1.0]

5-exit:
LW060 = [0.6, 0.6, 0.7, 0.9, 1.0]
LW080 = [0.8, 0.7, 0.7, 0.9, 1.0]
```

### Why this was done

The greedy-EE branch showed:

```text
Exit 1 is too weak.
Exit 2 is somewhat useful.
Exit 4 is strongest.
```

Therefore, stronger early-exit supervision was tested to improve Exit 1 and Exit 2.

---

## 14. Loss-weight results

### Exit 1

Exit 1 remained weak.

```text
Exit 1 macro-F1 stayed around 0.41 to 0.42.
```

Correct interpretation:

```text
Loss weighting alone does not fix Exit 1.
```

### Exit 2

Exit 2 improved clearly.

```text
3exit Exit 2:
0.5727 -> 0.5909

5exit Exit 2:
0.4777 -> 0.5067
```

### 3-exit dynamic improvement

Old baseline:

```text
3exit_posweight Policy 002
macro-F1 = 0.6293
compute saved = 2.53%
```

New loss-weight result:

```text
3exit_lw080 Policy 002
macro-F1 = 0.6475
compute saved = 3.46%
```

### 5-exit dynamic improvement

Old best:

```text
5exit_posweight
min_exit=3, stable_k=2
macro-F1 = 0.6449
compute saved = 9.33%
```

New best:

```text
5exit_lw080
min_exit=3, stable_k=2
macro-F1 = 0.6504
compute saved = 8.03%
```

### Verdict

Successful controlled ablation.

```text
Loss weighting improves Exit 2 and dynamic quality.
Loss weighting alone does not solve Exit 1.
```

---

## 15. Branch-by-branch memory table

| Branch / Stage | Main Method | Why We Did It | Helped or Degraded? | Main Lesson |
|---|---|---|---|---|
| `kexit_cclass_greedy_multi-label` | Multi-label BCE/sigmoid setup | Move from single-label to overlapping labels | Helped | Multi-label formulation is necessary |
| Dataset stage | Synthetic two-label mixtures | Create overlapping audio labels | Helped | Controlled test, but real mixed data still needed |
| Threshold tuning | Per-label/per-exit thresholds | Fixed 0.5 was unreliable | Strongly helped | Threshold tuning is essential |
| `kexit_cclass_greedy_multi-label_pos-weight` | Positive label weighting | Improve rare/difficult label recall | Mixed but useful | Improves macro-F1 but can increase false positives |
| Static exit analysis | Evaluate every exit alone | Diagnose exit reliability | Very helpful | Exit 1 weak, Exit 4 strong |
| `kexit_cclass_greedy_multi-label_policy` | Greedy label-set stability script | Add dynamic early-exit evidence | Helped | Measures exit depth and compute saving |
| `kexit_multi-label_greedy_EE` | Structured policy experiments | Compare conservative/fair/aggressive policies | Helped | Best baseline: 5exit posweight, min_exit=3, stable_k=2 |
| `kexit_multi-label_EE_lossweight` | Stronger early-exit loss weights | Strengthen Exit 1 and Exit 2 | Partially helped | Exit 2 improved; Exit 1 still weak |

---

## 16. What improved overall

### Helpful methods

```text
Multi-label formulation
Synthetic mixture setup
Per-label and per-exit threshold tuning
Positive-label weighting for macro-F1
Static per-exit diagnosis
Greedy label-set stability policy
Policy 002 for fair 3-exit early-exit testing
min_exit=3, stable_k=2 for 5-exit practical dynamic inference
Loss weighting for improving Exit 2
```

### Weak or degraded methods

```text
Fixed 0.5 threshold alone
Aggressive Exit 1 stopping with stable_k=1
Allow-empty-label stopping as a main policy
Loss weighting alone for fixing Exit 1
```

---

## 17. Current thesis story

```text
1. Reformulated audio classification as multi-label prediction.
2. Built synthetic two-label mixtures to simulate overlapping events.
3. Trained K-exit multi-label networks with BCE/sigmoid outputs.
4. Found that fixed 0.5 thresholding was insufficient.
5. Added per-label and per-exit threshold tuning.
6. Added positive-label weighting to improve macro-F1.
7. Diagnosed per-exit reliability and found Exit 1 weak, Exit 4 strong.
8. Added greedy label-set stability for dynamic early exit.
9. Showed conservative policy works for 5-exit but not 3-exit compute saving.
10. Showed Exit-1-enabled policy allows 3-exit early exit but with small savings.
11. Showed aggressive Exit 1 stopping fails.
12. Added stronger loss weighting to improve early exits.
13. Found that loss weighting improves Exit 2 and dynamic quality, but Exit 1 remains weak.
```

---

## 18. Current best conclusion

The multi-label NeuroAccuExit experiments show that dynamic early-exit inference is feasible for overlapping audio events, but exit reliability is critical. Threshold tuning and positive-label weighting are necessary to obtain fair multi-label performance. Greedy label-set stability provides a practical dynamic policy, but immediate Exit 1 stopping is unreliable. The strongest greedy-EE baseline came from the 5-exit positive-weighted model with `min_exit=3, stable_k=2`. The loss-weighting branch improved intermediate-exit quality, especially Exit 2, and produced a better quality-focused 5-exit trade-off, but it did not fully solve the weak Exit 1 problem.

---

## 19. Recommended next steps

| Next step | Priority | Reason |
|---|---:|---|
| Update `README.md`, `DOC_STRUCTURE.md`, and `docs/APPENDIX.md` for `kexit_multi-label_EE_lossweight` | High | Lock the new results |
| Keep this file updated after every branch | Very high | Prevent losing experimental memory |
| Try lower `pos_weight_max`, such as 5.0 or 10.0 | Medium | May reduce over-prediction |
| Add calibration/mAP/AUC diagnostics | Medium | Reviewer-proof multi-label evaluation |
| Add sigmoid-aware hint passing | Later | More novel, but should come after documenting loss weighting |
| Test real mixed audio | High for final paper | Synthetic mixtures are a limitation |

---

## 20. Rule for future branches

For every new branch, add a short entry here:

```text
Branch:
Method:
Why:
Commands:
Main result:
Compared with:
Helped/degraded:
Decision:
```

This file should be treated as the memory log for the multi-label research path.
