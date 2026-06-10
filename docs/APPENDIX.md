# Appendix — v0.8 Human-Corrected-Balanced Methodology and Results

## A. Research context

The ASHADIP/NeuroAccuExit audio pipeline studies early-exit multi-label classification for human-talk audio. The wider objective is to build a compact model that can classify speaker identity and background/context labels while enabling agentic preprocessing through a TinyAudioTriageAgent (TATA).

## B. Version history

| Version | Main idea | Key finding |
|---|---|---|
| v0.6 | Broad TATA-assisted pseudo-manifest + human correction | Strong first broad result, but old holdout labels were incomplete for non-target context. |
| v0.7 | Filter out five known non-target source folders | Target-speaker setting became cleaner, but event/background weaknesses remained. |
| v0.8 | Human-corrected-balanced delta repair | Best corrected-holdout result and fair improvement over v0.6. |

## C. Label schema

| label                     |
|:--------------------------|
| Brene_Brown               |
| Eckhart_Tolle             |
| Eric_Thomas               |
| Gary_Vee                  |
| Jay_Shetty                |
| Nick_Vujicic              |
| other_speaker_present     |
| music_present             |
| audience_reaction_present |
| silence_present           |

## D. Non-target identity rule

Known non-target source folders:

```text
Les_Brown
Mel_Robbins
Oprah_Winfrey
Rabin_Sharma
Simon_Sinek
```

Rule:

```text
all target speaker labels = 0
other_speaker_present = 1
music_present/audience_reaction_present/silence_present = manually reviewed context labels
```

## E. LAWYER v0.8 role

LAWYER is used as a label-specific weak-label refinement stage. It is not treated as ground truth. For v0.8-HCB, only reviewed LAWYER-new samples were added. The changed-label queue was preserved as a future ablation rather than blindly accepted.

## F. Training manifest

| item                                        |   count |
|:--------------------------------------------|--------:|
| seed_segment_rows                           |   12469 |
| raw_expanded_segment_rows                   |   23780 |
| final_combined_segment_rows                 |   36249 |
| raw_parent_labels_used                      |    4756 |
| zero_active_corrected_needs_review_excluded |       0 |
| missing_parent_segment_groups               |       0 |

Training groups:

| group                             |   count_before_balance |   count_after_balance |
|:----------------------------------|-----------------------:|----------------------:|
| raw_hybrid_needs_review_corrected |                  15855 |                 10132 |
| seed_reviewed                     |                  12469 |                 12469 |
| raw_hybrid_accepted_with_warning  |                   6080 |                  4941 |
| raw_hybrid_accepted               |                   1845 |                  1821 |

## G. Balancing

| label                     |   before_balance |   after_balance |
|:--------------------------|-----------------:|----------------:|
| Brene_Brown               |             2885 |            2885 |
| Eckhart_Tolle             |             3145 |            3145 |
| Eric_Thomas               |             2850 |            2850 |
| Gary_Vee                  |             3135 |            3135 |
| Jay_Shetty                |             4225 |            4225 |
| Nick_Vujicic              |             2425 |            2425 |
| other_speaker_present     |            15916 |            9030 |
| music_present             |            13045 |           11393 |
| audience_reaction_present |             5124 |            5124 |
| silence_present           |             1724 |            1724 |

![Balancing plot](figures/v08_label_counts_before_after_balance.png)

## H. Training settings

| Setting                  | Value                                   |
|:-------------------------|:----------------------------------------|
| run                      | main_v08_human_corrected_balanced_3exit |
| tap_blocks               | 1,3                                     |
| number of exits          | 3                                       |
| epochs                   | 40                                      |
| batch size               | 64                                      |
| learning rate            | 0.001                                   |
| loss weights             | [0.3, 0.3, 1.0]                         |
| fixed threshold          | 0.5                                     |
| device                   | cpu                                     |
| positive class weighting | False                                   |

## I. Internal test by exit

|   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
|      1 |     0.2185 |     0.358  |       0.2833 |        0.1535 |         0.1293 |            1.4493 |            0.565  |
|      2 |     0.6713 |     0.6837 |       0.6478 |        0.4472 |         0.0844 |            1.4493 |            1.2208 |
|      3 |     0.8305 |     0.8283 |       0.8285 |        0.6206 |         0.0502 |            1.4493 |            1.4737 |

## J. Corrected holdout by exit

| model           | threshold_mode   | aggregation   |   exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   jaccard_score |   avg_true_labels |   avg_pred_labels |
|:----------------|:-----------------|:--------------|-------:|-----------:|-----------:|-------------:|--------------:|---------------:|----------------:|------------------:|------------------:|
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      1 |     0.113  |     0.3166 |       0.204  |        0.0288 |         0.1275 |          0.1596 |            1.4694 |            0.3956 |
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      2 |     0.6315 |     0.7739 |       0.7197 |        0.5467 |         0.0591 |          0.6752 |            1.4694 |            1.1419 |
| v0.8-HCB 3-exit | fixed_0p5        | mean          |      3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |          0.9174 |            1.4694 |            1.4302 |

## K. Per-label corrected holdout behaviour

| label                     |   precision |   recall |     f1 |   support |   predicted_positive |
|:--------------------------|------------:|---------:|-------:|----------:|---------------------:|
| Brene_Brown               |      1      |   0.9315 | 0.9645 |        73 |                   68 |
| Eckhart_Tolle             |      1      |   0.9643 | 0.9818 |        84 |                   81 |
| Eric_Thomas               |      0.9028 |   0.9559 | 0.9286 |        68 |                   72 |
| Gary_Vee                  |      1      |   0.9559 | 0.9774 |        68 |                   65 |
| Jay_Shetty                |      0.9278 |   1      | 0.9626 |        90 |                   97 |
| Nick_Vujicic              |      1      |   0.9592 | 0.9792 |        49 |                   47 |
| other_speaker_present     |      0.9156 |   0.9435 | 0.9293 |       460 |                  474 |
| music_present             |      0.964  |   0.9413 | 0.9525 |       341 |                  333 |
| audience_reaction_present |      0.6667 |   0.069  | 0.125  |        29 |                    3 |
| silence_present           |      0      |   0      | 0      |        12 |                    0 |

## L. Fair comparison

| model           |   final_exit |   macro_f1 |   micro_f1 |   samples_f1 |   exact_match |   hamming_loss |   avg_true_labels |   avg_pred_labels |
|:----------------|-------------:|-----------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| v0.6 3-exit     |            3 |     0.7537 |     0.8865 |       0.8992 |        0.7497 |         0.0315 |            1.4694 |            1.3045 |
| v0.6 5-exit     |            5 |     0.746  |     0.8771 |       0.8881 |        0.7232 |         0.0338 |            1.4694 |            1.2814 |
| v0.8-HCB 3-exit |            3 |     0.7801 |     0.9332 |       0.9406 |        0.8397 |         0.0194 |            1.4694 |            1.4302 |

## M. Thesis-ready conclusion

On the corrected parent-level holdout set containing 867 parent clips and 4,335 one-second segments, the v0.8-human-corrected-balanced 3-exit model achieved the strongest final-exit performance under mean probability aggregation and a fixed 0.5 threshold. Compared with the previous v0.6 3-exit model re-evaluated on the same corrected holdout, it improved Macro-F1 from 0.7537 to 0.7801, Micro-F1 from 0.8865 to 0.9332, Samples-F1 from 0.8992 to 0.9406, and Exact Match from 0.7497 to 0.8397, while reducing Hamming Loss from 0.0315 to 0.0194. The model also predicted a more realistic number of labels per clip, increasing average predicted labels from 1.3045 to 1.4302 against a corrected ground-truth average of 1.4694.

## N. Recommended thesis figure set

| Figure | File | Purpose |
|---|---|---|
| Training validation curve | `docs/figures/v08_training_validation_curve.png` | Shows learning progression and late best epoch. |
| Label balance plot | `docs/figures/v08_label_counts_before_after_balance.png` | Shows reduction of `other_speaker_present` dominance. |
| Corrected holdout comparison | `docs/figures/v08_vs_v06_corrected_holdout_bar.png` | Shows v0.8 improvement over v0.6. |
| Hamming loss comparison | `docs/figures/v08_vs_v06_hamming_loss_bar.png` | Shows reduced label error rate. |
| Per-label F1 | `docs/figures/v08_corrected_holdout_per_label_f1_bar.png` | Shows strong labels and remaining rare-event weaknesses. |
