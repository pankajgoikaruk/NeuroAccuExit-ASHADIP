# Sequential Active-Budget Anytime Exit — Theory and Method

## From a two-exit rule to an anytime sequence

Earlier experiments mainly selected between Exit 2 and the final exit. v0.17 instead defines:

```text
x → E1 → E2 → ... → EK
```

At every non-final exit, the policy computes a stop decision. A sample stops at the **first** exit satisfying all enabled conditions. Samples that do not satisfy the stage rule continue to the next exit.

## Multi-label confidence

For label `l` at exit `e`, let `p[e,l]` be the sigmoid probability and `t[e,l]` the classification threshold:

```text
binary confidence c[e,l] = max(p[e,l], 1 − p[e,l])
stage confidence         = mean_l c[e,l]
```

This is not a softmax maximum; each label is an independent binary decision.

## Label-specific margin

```text
margin[e,l] = |p[e,l] − t[e,l]|
```

Every label must exceed its stage-specific optimised margin. The ablations show this condition is essential: removing margins caused large Macro-F1, Exact-Match, and Hamming degradation.

## Inter-exit stability and probability movement

From Exit 2 onward, the thresholded label set must match the preceding exit:

```text
y_hat[e] = y_hat[e−1]
```

The maximum probability movement is:

```text
delta[e] = max_l |p[e,l] − p[e−1,l]|
```

A stage may stop only when the movement is below its optimised limit. Exit 1 has no preceding vector, so its movement term is zero by construction.

## Validation-derived label risk

For every non-final exit and label, v0.17 counts how often the final exit corrects that early exit on validation data. Counts are normalised within the stage into risk weights. Boundary proximity and inter-exit movement form a per-label uncertainty score, and the stage requires the maximum score to remain under its risk budget.

The `no_risk` ablation produced the same parent metrics as the full policy. The selected risk condition was therefore non-binding; this does not establish that risk modelling is theoretically useless.

## Complete stopping rule

A live sample stops at non-final exit `e` only when all enabled conditions hold:

```text
non-empty prediction
AND mean confidence ≥ stage threshold
AND probability movement ≤ stage limit
AND maximum label risk ≤ stage risk budget
AND every label margin ≥ its label-specific threshold
AND previous-exit label set is stable (from Exit 2 onward)
```

## Multi-objective search

Every candidate jointly minimises:

```text
[-FLOP saving,
 robust Macro-F1 drop,
 robust Micro-F1 drop,
 robust Exact-Match drop,
 robust Hamming increase]
```

Feasible candidates must also satisfy minimum total early coverage and minimum Exit-1 coverage.

## Safety-buffered Pareto knee

v0.16 selected the feasible point with maximum compute saving. v0.17 instead:

1. constructs the feasible Pareto front;
2. measures each candidate's use of the permitted quality budget;
3. keeps candidates inside a 75% safety fraction;
4. selects a normalised knee balancing quality and saving.

## Estimated compute and measured latency

For selected exit `s[i]` and final exit `K`:

```text
saving = 1 − sum_i C[s[i]] / (N × C[K])
```

where `C[e]` is cumulative architecture FLOPs through exit `e`.

Estimated FLOPs and latency are reported separately. Runtime includes staged CNN execution, policy evaluation, active-batch filtering, and Python/PyTorch overhead. Practical acceleration requires measured speedup above `1.0×` under the controlled protocol.

## Meaning of active budget

The controller allocates more computation to samples that fail confidence, margin, stability, movement, or risk checks. v0.17 freezes one operating point per architecture. It does not yet expose a user-specified runtime budget or a complete budget–quality curve.

## What v0.17 is not

- It is not label-wise asynchronous execution; the complete sample stops together.
- It does not retrain the backbone or exit heads.
- It does not use evidence accumulation or knowledge distillation in the primary policy.
- It is not proof that five exits are universally better than three.
