# v0.11_EE Paper-Ready Summary

## Method paragraph

We converted the trained three-exit TinyAudioCNN into an anytime-compatible model through an inference-only staged wrapper. The wrapper preserves the original model parameters and full-forward path while exposing continuation states after each exit. For the canonical configuration with taps after backbone Blocks 1 and 3, Exit 1 executes Block 1, Exit 2 executes Blocks 1–3 cumulatively, and Exit 3 executes all five blocks. Consequently, samples accepted at Exit 2 do not execute Blocks 4–5. Staged logits and probabilities were numerically identical to conventional full-forward outputs at every exit.

## Fixed-exit diagnostic paragraph

We first evaluated the complete corrected-holdout set under Always Exit 1, Always Exit 2, and Always Exit 3 scenarios. At parent level, using the historical frozen LATS-v2 policy as a transfer diagnostic, Macro-F1 increased from 0.1626 at Exit 1 to 0.6923 at Exit 2 and 0.8624 at Exit 3. Exit 2 performed strongly for labels such as Eckhart Tolle, Jay Shetty, music, and Brene Brown, but remained substantially below full depth for Eric Thomas, Nick Vujicic, audience reaction, Gary Vee, and other-speaker detection. These results motivated a conservative policy that treats Exit 2 as the first practical stopping point.

## Dynamic-policy paragraph

The stopping policy was selected only on validation data and frozen before corrected-holdout evaluation. Every sample reached Exit 2. A sample stopped when the thresholded label sets at Exits 1 and 2 agreed, the Exit-2 set was non-empty, and Exit-2 mean binary confidence exceeded 0.55; otherwise, the sample continued to Exit 3. Because no per-exit tuned threshold artifact existed for the canonical run, the first policy used a fixed segment threshold of 0.5.

## Main result paragraph

On 4,335 corrected-holdout segments, the frozen policy stopped 508 samples (11.72%) at Exit 2 and sent 3,827 samples to Exit 3. Average exit depth decreased from 3.0 to 2.8828, corresponding to an estimated 7.53% reduction in computation. Parent-level Macro-F1 decreased from 0.8624 to 0.8422, Micro-F1 from 0.9531 to 0.9355, Samples-F1 from 0.9589 to 0.9436, and Exact Match from 0.8766 to 0.8385. Hamming Loss increased from 0.0137 to 0.0189. The result demonstrates genuine computation-adaptive inference while establishing the initial quality–efficiency trade-off for later budget-aware policies.

## Contribution wording

> We provide a checkpoint-preserving staged inference mechanism for multi-label audio Early-Exit and demonstrate real sample-dependent computation skipping rather than post-hoc exit selection. The staged implementation is numerically equivalent to the original model at every exit and supports validation-frozen stopping policies that avoid deeper backbone computation for accepted samples.

## Conservative claim

> The first policy establishes feasibility rather than an optimal deployment point: it saves 7.53% estimated computation while retaining 97.67% of full-depth Macro-F1 and 98.15% of Micro-F1.

## Limitation wording

> Parent-level diagnostics for the shallow exits reuse aggregation methods and thresholds optimized for the final exit; thus, they measure frozen-policy transfer rather than exit-specific calibrated performance. In addition, dynamic latency has not yet been compared with an Always Exit 3 baseline under an identical timing protocol.

## Suggested table caption

> **Fixed-depth and dynamic Early-Exit performance on the corrected human-talk holdout.** Always Exit 1 and Always Exit 2 use the frozen historical Exit-3 LATS-v2 parent policy as a transfer diagnostic. The dynamic policy is selected on validation data and performs genuine staged execution, such that samples accepted at Exit 2 do not execute backbone Blocks 4–5.

## Suggested implementation figure caption

> **Three-stage inference path for the five-block TinyAudioCNN.** Exit 1 follows Block 1, Exit 2 follows Block 3, and the final exit follows Block 5. The staged wrapper carries the intermediate feature map forward and invokes only the blocks required to reach the next exit.

## Suggested research questions answered

**RQ1: Can the existing trained network support genuine Early-Exit without retraining?**  
Yes. The staged wrapper exactly reproduces the original checkpoint outputs.

**RQ2: Which exit is the first viable early stopping point?**  
Exit 2. Exit 1 is substantially underperforming for the full multi-label task.

**RQ3: Does real compute skipping preserve useful quality?**  
Yes, for a conservative subset of samples, although the first policy incurs measurable quality loss and requires further calibration.

**RQ4: What is the next technical requirement?**  
An explicit cost-aware controller, same-protocol latency baselines, and quality evaluation at multiple budgets.
