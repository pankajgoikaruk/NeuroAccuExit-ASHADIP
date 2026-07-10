# Paper-Ready Full-Depth Baseline Summary

The full-computation reference used the final exit of the three-exit no-hint
model followed by the frozen historical LATS-v2 parent-level inference policy.
Across 867 parent clips and 10 labels, it achieved Macro-F1 0.8624, Micro-F1
0.9531, Samples-F1 0.9589, Exact Match 0.8766, and Hamming Loss 0.0137.
This frozen result is the canonical quality reference for all subsequent
standard Early-Exit, budget-aware Early-Exit, and anytime-inference evaluations.

No model retraining, threshold search, or aggregation-method search occurs in
the deterministic replay.
