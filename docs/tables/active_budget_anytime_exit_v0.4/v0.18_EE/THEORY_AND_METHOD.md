# v0.18 Theory and Method

## Sequential anytime inference

For a network with `E` exits, inference proceeds in order:

```text
Exit 1 → Exit 2 → ... → Exit E
```

A sample stopped at Exit `e` is removed from the active batch and does not execute blocks required only by exits `e+1,...,E`.

## Multi-label evidence

For probability `p[e,l]` and threshold `t[e,l]`:

```text
confidence[e,l] = max(p[e,l], 1 − p[e,l])
margin[e,l]     = |p[e,l] − t[e,l]|
delta[e,l]      = |p[e,l] − p[e−1,l]|
```

The stop decision combines mean binary confidence, all-label margin checks, maximum probability delta, previous-exit label-set stability, non-empty prediction, per-label continuation-risk scores, risk-weighted margin expansion, a high-risk uncertainty veto, and an extra Exit-1 confidence boost.

## Continuation risk

Risk is derived only from validation data. It combines:

1. how often the final exit corrects an earlier label;
2. how much per-label F1 improves at deeper exits.

The score is used at inference to make early stopping harder for labels that historically need deeper computation. It is not a loss penalty and does not retrain the CNN.

## Optimisation variables

Every non-final exit has independent parameters for confidence threshold, maximum probability delta, maximum label risk, risk-score threshold, risk-margin multiplier, risk uncertainty band, Exit-1 confidence boost, and ten per-label margins.

## Objectives

The optimiser maximises estimated FLOPs saved while minimising robust degradation in Parent Macro-F1, Parent Micro-F1, Parent Exact Match, and Parent Hamming Loss.

## Selection

The selected candidate is a safety-buffered Pareto point. Eligibility is determined on validation only. Corrected-holdout compliance is reported separately.

## Fair architecture comparison

The comparison is valid because both models share data, labels, preprocessing, optimiser settings, random seed, final-exit weight, and total auxiliary-loss budget. Only topology and auxiliary-weight distribution differ.
