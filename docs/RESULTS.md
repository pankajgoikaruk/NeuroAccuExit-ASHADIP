\# ASHADIP / NeuroAccuExit — Results Log



This file tracks key run outcomes (accuracy vs early-exit efficiency) so we can compare changes across variants.



\## Goal (current target)

\- \*\*Accuracy ≥ 0.94\*\*

\- \*\*Avg exit depth as low as possible\*\* (efficiency)

\- Track exit mix + flip-rate where available



---



\## Historical run summary



| Run ID | Variant | Policy | Policy test acc | Avg exit depth | Exit mix (e1/e2/e3) | Notes |

|---|---|---:|---:|---:|---|---|

| `runs\\v0\_1\\v0\_1\_003` | v0.1 | greedy | \*\*0.9569\*\* | \*\*2.102\*\* | 0.394 / 0.111 / 0.495 | Strong baseline accuracy; moderate depth savings |

| `runs\\v0\_2\\v0\_2\_002` | v0.2 | EA | \*\*0.8954\*\* | \*\*1.295\*\* | 0.797 / 0.111 / 0.092 | EA tuned for depth → big depth improvement, accuracy drop |

| `runs\\v0\_2\\v0\_2\_003` | v0.2 | EA | \*\*0.8554\*\* | \*\*1.172\*\* | 0.874 / 0.080 / 0.046 | Added \*\*KD + SpecAug\*\*; depth improved further, accuracy dropped more |

| `runs\\v0\_2\\v0\_2\_003` | v0.2 | greedy | \*\*0.9231\*\* | \*\*1.772\*\* | 0.529 / 0.169 / 0.302 | Greedy on same trained model; better than EA, still < 0.94 |



---



\## What changed \& observed impact



\### v0.1 → v0.2 (Depth-EA policy + EA threshold selection)

\- ✅ \*\*Efficiency improved\*\* (EA exits earlier; more e1/e2).

\- ❌ \*\*Accuracy dropped\*\* (EA exits too aggressively / early exits not reliable enough).



\### v0.2\_003: Added KD + SpecAug (training-side change)

\- ✅ Further shifted exit mix to exit1 (very early exits).

\- ❌ Accuracy dropped again → suggests \*\*exit1 became overconfident/less calibrated\*\* or training regularization harmed final decision quality.



---



\## Key takeaway right now

\- EA is achieving the desired \*\*low depth\*\*, but we need to recover accuracy.

\- Next iteration should enforce: \*\*accuracy safety constraint (≥ 0.94)\*\* while still optimizing depth.



---



\## Next planned step (v0.2.3 direction)

\- Keep EA policy logic, but adjust \*\*training\*\* to make exit1 trustworthy:

&nbsp; - Tune loss weights + KD strength + SpecAug intensity

&nbsp; - Add EA selection constraints (max allowed accuracy/F1 drop vs greedy baseline)


---



\## Config snapshot (v0.2.2)



This is the exact config used for the KD + SpecAug run (`v0\_2\_003`) so the results are reproducible.



File: `configs/audio\_moth.yaml`



```yaml

seed: 42



paths:

&nbsp; data\_root: data

&nbsp; cache\_root: data\_cache

&nbsp; runs\_root: runs



audio:

&nbsp; sample\_rate: 16000

&nbsp; bandpass: \[100, 3000]

&nbsp; segment\_sec: 1.0

&nbsp; segment\_hop: 0.5

&nbsp; silence\_dbfs: -40



features:

&nbsp; type: logmel

&nbsp; n\_mels: 64

&nbsp; n\_fft: 1024

&nbsp; win\_ms: 25

&nbsp; hop\_ms: 10

&nbsp; cmvn: true



train:

&nbsp; batch\_size: 64

&nbsp; epochs: 40

&nbsp; lr: 0.001

&nbsp; weight\_decay: 0.0

&nbsp; optimizer: adam

&nbsp; num\_workers: 0

&nbsp; val\_every: 1



&nbsp; # Make early exits stronger (important for EA depth~1.3)

&nbsp; loss\_weights: \[1.0, 0.5, 0.2]



&nbsp; # Knowledge Distillation: exit3 teaches exit1/exit2

&nbsp; kd:

&nbsp;   enable: true

&nbsp;   alpha: 0.6

&nbsp;   temp: 2.0

&nbsp;   weights: \[1.0, 0.7]



&nbsp; # SpecAugment for robustness (train only)

&nbsp; specaug:

&nbsp;   enable: true

&nbsp;   freq\_mask: 8

&nbsp;   time\_mask: 12

&nbsp;   num\_masks: 2



model:

&nbsp; num\_classes: 2

&nbsp; exits: 3



calibration:

&nbsp; temperature\_scaling: true



thresholds:

&nbsp; strategy: f1\_under\_budget

&nbsp; compute\_budget\_ms: 4.0

&nbsp; grid\_tau: \[0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]

&nbsp; grid\_alpha: \[0.10, 0.08, 0.05, 0.03, 0.02]



ea:

&nbsp; enabled: true

&nbsp; mode: logprob

&nbsp; threshold\_grid: \[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

&nbsp; min\_exit: 0

&nbsp; stable\_k: 1

&nbsp; flip\_penalty: 0.0



split:

&nbsp; train: 0.7

&nbsp; val: 0.15

&nbsp; test: 0.15

&nbsp; stratify: true






\### Run command snapshot (v0.2.2)



This is the exact command used to reproduce the full pipeline run for \*\*KD + SpecAug + EA\*\*.



```powershell

powershell -ExecutionPolicy Bypass -File scripts\\run\_full.ps1 `

&nbsp; -Variant "v0.2" `

&nbsp; -Policy "ea" `

&nbsp; -Device "cpu" `

&nbsp; -SegmentSec 1.0 `

&nbsp; -HopSec 0.5







