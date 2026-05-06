# Appendix Update — `kexit_cclass_greedy_v2`

This appendix update should be read before the older 8-run appendix tables.

---

# A0. Latest branch-level audit

| Item | Value |
|---|---|
| Branch | `kexit_cclass_greedy_v2` |
| Current valid setting | Prepared/grouped 10-class C-class |
| Intended next setting | Refined 11-class C-class |
| New class | `rain_thunderstorm` |
| Current issue | Refined logs still process only 10 effective labels |
| Action required | Fix label ingestion and rerun 4 refined variants |

---

# A1. Intended 11-class split counts

| Class | Train | Val | Test |
|---|---:|---:|---:|
| car_crash | 223 | 42 | 38 |
| conversation | 223 | 42 | 38 |
| engine_idling | 223 | 42 | 38 |
| fireworks | 223 | 42 | 38 |
| gun_shot | 223 | 42 | 38 |
| rain | 223 | 42 | 38 |
| rain_thunderstorm | 223 | 42 | 38 |
| road_traffic | 223 | 42 | 38 |
| scream | 223 | 42 | 38 |
| thunderstorm | 223 | 42 | 38 |
| wind | 223 | 42 | 38 |
| **Total** | **2453** | **462** | **418** |

---

# A2. Effective label audit from available refined logs

| Field | Expected | Observed | Interpretation |
|---|---:|---:|---|
| CLI labels | 11 | 11 | Command was intended correctly |
| Inventory labels | 11 | 10 | Effective preprocessing missed one class |
| Missing class | None | `rain_thunderstorm` | Must fix before final reporting |
| `num_classes` | 11 | 10 | Model/evaluation still effective-10 |
| Train segments | 2453 | 2230 | 10 × 223 |
| Val segments | 462 | 420 | 10 × 42 |
| Test segments | 418 | 380 | 10 × 38 |

---

# A3. Prepared/grouped 10-class result table

| Variant | Exits | Hint | Policy acc | Full clip acc | Depth×Time acc | Windows saved | Compute saved | Exit behaviour |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `3exit_cclass_greedy_prepared_grouped` | 3 | No | 55.79% | 66.44% | 66.44% | 22.63% | 22.01% | Uses e2/e3; final exit not strongest |
| `3exit_cclass_greedy_hint_prepared_grouped` | 3 | Yes | 68.16% | **84.56%** | **84.56%** | 22.37% | 22.40% | Best current valid prepared/grouped model |
| `5exit_cclass_greedy_prepared_grouped` | 5 | No | 61.58% | 75.17% | 75.17% | 22.11% | 21.99% | Mostly e5 |
| `5exit_cclass_greedy_hint_prepared_grouped` | 5 | Yes | 68.42% | 79.87% | 79.19% | **23.16%** | **23.53%** | Better efficiency, lower accuracy than 3-exit hint |

---

# A4. Segment-policy details

| Variant | Policy acc | Avg exit depth | Exit mix | Flip-any rate | Exit consistency |
|---|---:|---:|---|---:|---:|
| `3exit_cclass_greedy_prepared_grouped` | 55.79% | 2.511 | e1=8.68%, e2=31.58%, e3=59.74% | 70.00% | 93.42% |
| `3exit_cclass_greedy_hint_prepared_grouped` | 68.16% | 2.808 | e1=0.53%, e2=18.16%, e3=81.32% | 67.63% | 99.47% |
| `5exit_cclass_greedy_prepared_grouped` | 61.58% | 4.547 | e1=0.00%, e2=7.63%, e3=10.26%, e4=1.84%, e5=80.26% | 69.74% | 99.74% |
| `5exit_cclass_greedy_hint_prepared_grouped` | 68.42% | 4.474 | e1=0.79%, e2=8.42%, e3=10.00%, e4=4.21%, e5=76.58% | 66.32% | 99.47% |

---

# A5. Clip-policy details

| Variant | Mode | Clip acc | Segment acc | Avg windows | Windows saved | Avg compute | Compute saved | Flip rate | Exit consistency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `3exit_cclass_greedy_prepared_grouped` | Full | 66.44% | 55.79% | 2.550 / 2.550 | 0.00% | 6.403 | 0.00% | 70.00% | 93.42% |
| `3exit_cclass_greedy_prepared_grouped` | Depth×Time | 66.44% | 57.14% | 1.973 / 2.550 | 22.63% | 4.993 | 22.01% | 71.77% | 93.54% |
| `3exit_cclass_greedy_hint_prepared_grouped` | Full | 84.56% | 68.16% | 2.550 / 2.550 | 0.00% | 7.161 | 0.00% | 67.63% | 99.47% |
| `3exit_cclass_greedy_hint_prepared_grouped` | Depth×Time | 84.56% | 71.86% | 1.980 / 2.550 | 22.37% | 5.557 | 22.40% | 70.51% | 99.66% |
| `5exit_cclass_greedy_prepared_grouped` | Full | 75.17% | 61.58% | 2.550 / 2.550 | 0.00% | 11.597 | 0.00% | 69.74% | 99.74% |
| `5exit_cclass_greedy_prepared_grouped` | Depth×Time | 75.17% | — | 1.987 / 2.550 | 22.11% | 9.047 | 21.99% | 71.62% | 99.66% |
| `5exit_cclass_greedy_hint_prepared_grouped` | Full | 79.87% | 68.42% | 2.550 / 2.550 | 0.00% | 11.409 | 0.00% | 66.32% | 99.47% |
| `5exit_cclass_greedy_hint_prepared_grouped` | Depth×Time | 79.19% | — | 1.960 / 2.550 | 23.16% | 8.725 | 23.53% | 67.81% | 99.66% |

---

# A6. Hint vs no-hint deltas on prepared/grouped 10-class

| Exits | Metric | No hint | Hint | Delta |
|---:|---|---:|---:|---:|
| 3 | Segment policy acc | 55.79% | 68.16% | **+12.37pp** |
| 3 | Full-clip acc | 66.44% | 84.56% | **+18.12pp** |
| 3 | Depth×Time acc | 66.44% | 84.56% | **+18.12pp** |
| 3 | Compute saved | 22.01% | 22.40% | +0.39pp |
| 5 | Segment policy acc | 61.58% | 68.42% | **+6.84pp** |
| 5 | Full-clip acc | 75.17% | 79.87% | **+4.70pp** |
| 5 | Depth×Time acc | 75.17% | 79.19% | **+4.02pp** |
| 5 | Compute saved | 21.99% | 23.53% | +1.54pp |

---

# A7. 3-exit vs 5-exit deltas on prepared/grouped 10-class

| Hint | Metric | 3 exits | 5 exits | Delta 5-3 |
|---|---|---:|---:|---:|
| No | Segment policy acc | 55.79% | 61.58% | +5.79pp |
| No | Full-clip acc | 66.44% | 75.17% | +8.73pp |
| No | Depth×Time acc | 66.44% | 75.17% | +8.73pp |
| No | Compute saved | 22.01% | 21.99% | -0.02pp |
| Yes | Segment policy acc | 68.16% | 68.42% | +0.26pp |
| Yes | Full-clip acc | **84.56%** | 79.87% | -4.69pp |
| Yes | Depth×Time acc | **84.56%** | 79.19% | -5.37pp |
| Yes | Compute saved | 22.40% | **23.53%** | +1.13pp |

---

# A8. Diagnostic refined-run table, not final 11-class

| Variant | Policy acc | Full clip acc | Depth×Time acc | Note |
|---|---:|---:|---:|---|
| `3exit_cclass_greedy_refined11_grouped` | 71.32% | 83.33% | 83.97% | Effective 10-class only |
| `3exit_cclass_greedy_hint_refined11_grouped` | 73.68% | 80.77% | — | Effective 10-class only |
| `5exit_cclass_greedy_refined11_grouped` | 71.05% | 79.49% | 79.49% | Effective 10-class only |
| `5exit_cclass_greedy_hint_refined11_grouped` | 72.37% | 82.05% | — | Effective 10-class only |

These values can be used to debug the model behaviour, but not as final refined 11-class evidence.

---

# A9. Final 11-class reporting template

After the corrected rerun, fill this table.

| Variant | Exits | Hint | Effective classes | Policy acc | Full-clip acc | Depth×Time acc | Compute saved | Best? |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `3exit_cclass_greedy_refined11_grouped` | 3 | No | 11 | TBD | TBD | TBD | TBD | TBD |
| `3exit_cclass_greedy_hint_refined11_grouped` | 3 | Yes | 11 | TBD | TBD | TBD | TBD | TBD |
| `5exit_cclass_greedy_refined11_grouped` | 5 | No | 11 | TBD | TBD | TBD | TBD | TBD |
| `5exit_cclass_greedy_hint_refined11_grouped` | 5 | Yes | 11 | TBD | TBD | TBD | TBD | TBD |

---

# A10. Research-safe interpretation

The safest interpretation is:

> The prepared/grouped evaluation is now meaningful and shows that compact 3-exit hint passing currently gives the best valid prepared/grouped C-class clip accuracy. However, the newly intended 11-class refined experiment requires one more corrected rerun because the available logs show only 10 effective classes. The final claim should therefore be delayed until `rain_thunderstorm` is confirmed in inventory, `num_classes=11`, and the split counts match 2453/462/418.

---



---

# Previous Extended 8-Run Appendix

# Appendix — Extended Tables for 8-Run Generic K-Exit / C-Class Study

This appendix contains the extended result tables for the controlled 8-run study:

```text
3exit_2class_greedy
3exit_2class_greedy_hint
5exit_2class_greedy
5exit_2class_greedy_hint
3exit_cclass_greedy
3exit_cclass_greedy_hint
5exit_cclass_greedy
5exit_cclass_greedy_hint
```

The appendix should be used together with:

- `README.md` for the main repository summary
- `DOC_STRUCTURE.md` for thesis/report organization
- `ASHADIP_8_run_comparison_tables.xlsx` for spreadsheet inspection

---

# A. Overview table

| Key finding                                 | Variant                      | Value                         |
|:--------------------------------------------|:-----------------------------|:------------------------------|
| Best 2-class segment policy                 | 3exit_2class_greedy_hint     | 99.38%                        |
| Best 2-class clip accuracy                  | All four 2-class variants    | 100.00%                       |
| Best 10-class segment policy                | 3exit_cclass_greedy          | 69.90%                        |
| Best 10-class full/depth-time clip accuracy | 3exit_cclass_greedy          | 81.58%                        |
| Best 10-class compute saving                | 3exit_cclass_greedy          | 34.34%                        |
| Hint effect on C-class                      | Negative in all C-class runs | -8.22pp / -4.44pp segment acc |

---

# B. Dataset and segmentation summary

| Dataset        | Variant                  |   Classes |   Train files |   Val files |   Test files |   Train segs |   Val segs |   Test segs |   Total segs |   Test avg windows/clip |   Test median windows/clip |
|:---------------|:-------------------------|----------:|--------------:|------------:|-------------:|-------------:|-----------:|------------:|-------------:|------------------------:|---------------------------:|
| 2-class moth   | 3exit_2class_greedy      |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 2-class moth   | 3exit_2class_greedy_hint |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 2-class moth   | 5exit_2class_greedy      |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 2-class moth   | 5exit_2class_greedy_hint |         2 |            99 |          21 |           22 |         1646 |        246 |         325 |         2217 |                   14.77 |                        9.5 |
| 10-class audio | 3exit_cclass_greedy      |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |
| 10-class audio | 3exit_cclass_greedy_hint |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |
| 10-class audio | 5exit_cclass_greedy      |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |
| 10-class audio | 5exit_cclass_greedy_hint |        10 |           707 |         152 |          152 |         2713 |        566 |         608 |         3887 |                    4    |                        5   |

---

# C. Per-exit test accuracy and macro-F1

| Dataset        | Variant                  |   Exits | Hint   | Exit 1 acc   | Exit 1 macro F1   | Exit 2 acc   | Exit 2 macro F1   | Exit 3 acc   | Exit 3 macro F1   |
|:---------------|:-------------------------|--------:|:-------|:-------------|:------------------|:-------------|:------------------|:-------------|:------------------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | 84.00%       | 79.44%            | 93.54%       | 91.18%            | 98.46%       | 97.90%            |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | 84.62%       | 80.38%            | 94.46%       | 92.72%            | 99.38%       | 99.14%            |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | 83.38%       | 78.81%            | 85.54%       | 82.36%            | 96.00%       | 94.68%            |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | 83.38%       | 78.81%            | 86.15%       | 82.77%            | 94.77%       | 93.15%            |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | 29.28%       | 21.29%            | 61.51%       | 56.79%            | 69.74%       | 64.74%            |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | 22.70%       | 16.52%            | 53.29%       | 48.89%            | 61.68%       | 57.44%            |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | 21.38%       | 14.90%            | 42.60%       | 35.56%            | 58.39%       | 51.89%            |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | 21.05%       | 15.04%            | 40.13%       | 34.20%            | 56.74%       | 51.71%            |

---

# D. Segment-level greedy policy

| Dataset        | Variant                  |   Exits | Hint   | Policy acc   |   Avg exit depth | Flip-any rate   |   Avg flip count | Exit consistency   |   Tau |   N segments |
|:---------------|:-------------------------|--------:|:-------|:-------------|-----------------:|:----------------|-----------------:|:-------------------|------:|-------------:|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | 98.15%       |            1.862 | 17.23%          |            0.194 | 99.69%             |  0.9  |          325 |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | 99.38%       |            1.911 | 16.31%          |            0.172 | 100.00%            |  0.95 |          325 |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | 97.85%       |            2.462 | 20.31%          |            0.252 | 99.69%             |  0.95 |          325 |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | 96.92%       |            2.36  | 19.38%          |            0.243 | 99.69%             |  0.9  |          325 |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | 69.90%       |            2.873 | 71.38%          |            0.895 | 99.34%             |  0.92 |          608 |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | 61.68%       |            2.87  | 84.38%          |            1.086 | 100.00%            |  0.85 |          608 |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | 68.75%       |            4.612 | 81.91%          |            1.352 | 99.67%             |  0.9  |          608 |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | 64.31%       |            4.748 | 83.22%          |            1.339 | 100.00%            |  0.95 |          608 |

---

# E. Full-clip vs Depth×Time comparison

| Dataset        | Variant                  |   Exits | Hint   | Clip mode   | Clip acc   | Segment acc used   |   Avg windows used |   Avg windows total | Windows saved %   |   Avg compute units | Compute saved %   |   Avg depth/used window | Flip rate   | Exit consistency   |   N clips |
|:---------------|:-------------------------|--------:|:-------|:------------|:-----------|:-------------------|-------------------:|--------------------:|:------------------|--------------------:|:------------------|------------------------:|:------------|:-------------------|----------:|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Full clip   | 100.00%    | 98.15%             |             14.77  |               14.77 | 0.00%             |              27.5   | 0.00%             |                   1.862 | 17.23%      | 99.69%             |        22 |
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Depth×Time  | 100.00%    | 97.73%             |              2     |               14.77 | 86.46%            |               4.818 | 82.48%            |                   2.409 | 36.36%      | 100.00%            |        22 |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Full clip   | 100.00%    | 99.38%             |             14.77  |               14.77 | 0.00%             |              28.23  | 0.00%             |                   1.911 | 16.31%      | 100.00%            |        22 |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Depth×Time  | 100.00%    | 100.00%            |              2     |               14.77 | 86.46%            |               4.864 | 82.77%            |                   2.432 | 36.36%      | 100.00%            |        22 |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Full clip   | 100.00%    | 97.85%             |             14.77  |               14.77 | 0.00%             |              36.36  | 0.00%             |                   2.462 | 20.31%      | 99.69%             |        22 |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Depth×Time  | 100.00%    | 97.78%             |              2.045 |               14.77 | 86.15%            |               6.455 | 82.25%            |                   3.156 | 40.00%      | 97.78%             |        22 |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Full clip   | 100.00%    | 96.92%             |             14.77  |               14.77 | 0.00%             |              34.86  | 0.00%             |                   2.36  | 19.38%      | 99.69%             |        22 |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Depth×Time  | 100.00%    | 97.87%             |              2.136 |               14.77 | 85.54%            |               7.136 | 79.53%            |                   3.34  | 40.43%      | 100.00%            |        22 |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Full clip   | 81.58%     | 69.90%             |              4     |                4    | 0.00%             |              11.49  | 0.00%             |                   2.873 | 71.38%      | 99.34%             |       152 |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Depth×Time  | 81.58%     | 65.59%             |              2.638 |                4    | 34.05%            |               7.546 | 34.34%            |                   2.86  | 72.07%      | 99.50%             |       152 |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Full clip   | 78.29%     | 61.68%             |              4     |                4    | 0.00%             |              11.48  | 0.00%             |                   2.87  | 84.38%      | 100.00%            |       152 |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Depth×Time  | 78.29%     | 61.20%             |              2.967 |                4    | 25.82%            |               8.467 | 26.25%            |                   2.854 | 82.48%      | 100.00%            |       152 |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Full clip   | 78.95%     | 68.75%             |              4     |                4    | 0.00%             |              18.45  | 0.00%             |                   4.612 | 81.91%      | 99.67%             |       152 |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Depth×Time  | 78.95%     | 65.57%             |              2.809 |                4    | 29.77%            |              12.93  | 29.89%            |                   4.604 | 80.80%      | 99.53%             |       152 |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Full clip   | 73.68%     | 64.31%             |              4     |                4    | 0.00%             |              18.99  | 0.00%             |                   4.748 | 83.22%      | 100.00%            |       152 |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Depth×Time  | 73.03%     | 61.69%             |              2.73  |                4    | 31.74%            |              12.88  | 32.21%            |                   4.716 | 81.69%      | 100.00%            |       152 |

---

# F. Segment-policy exit mix

| Dataset        | Variant                  |   Exits | Hint   | e1     | e2     | e3     | e4     | e5     |
|:---------------|:-------------------------|--------:|:-------|:-------|:-------|:-------|:-------|:-------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | 39.38% | 35.08% | 25.54% | 0.00%  | 0.00%  |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | 36.31% | 36.31% | 27.38% | 0.00%  | 0.00%  |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | 36.31% | 6.46%  | 39.38% | 10.46% | 7.38%  |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | 38.46% | 13.85% | 29.23% | 10.15% | 8.31%  |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | 0.00%  | 12.66% | 87.34% | 0.00%  | 0.00%  |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | 0.00%  | 12.99% | 87.01% | 0.00%  | 0.00%  |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | 0.00%  | 2.96%  | 11.18% | 7.57%  | 78.29% |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | 0.00%  | 1.97%  | 8.55%  | 2.14%  | 87.34% |

---

# G. Full-clip exit mix

| Dataset        | Variant                  |   Exits | Hint   | Clip mode   | e1     | e2     | e3     | e4     | e5     |
|:---------------|:-------------------------|--------:|:-------|:------------|:-------|:-------|:-------|:-------|:-------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Full clip   | 39.38% | 35.08% | 25.54% | 0.00%  | 0.00%  |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Full clip   | 36.31% | 36.31% | 27.38% | 0.00%  | 0.00%  |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Full clip   | 36.31% | 6.46%  | 39.38% | 10.46% | 7.38%  |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Full clip   | 38.46% | 13.85% | 29.23% | 10.15% | 8.31%  |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Full clip   | 0.00%  | 12.66% | 87.34% | 0.00%  | 0.00%  |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Full clip   | 0.00%  | 12.99% | 87.01% | 0.00%  | 0.00%  |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Full clip   | 0.00%  | 2.96%  | 11.18% | 7.57%  | 78.29% |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Full clip   | 0.00%  | 1.97%  | 8.55%  | 2.14%  | 87.34% |

---

# H. Depth×Time exit mix

| Dataset        | Variant                  |   Exits | Hint   | Clip mode   | e1     | e2     | e3     | e4     | e5     |
|:---------------|:-------------------------|--------:|:-------|:------------|:-------|:-------|:-------|:-------|:-------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     | Depth×Time  | 11.36% | 36.36% | 52.27% | 0.00%  | 0.00%  |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    | Depth×Time  | 9.09%  | 38.64% | 52.27% | 0.00%  | 0.00%  |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     | Depth×Time  | 8.89%  | 6.67%  | 53.33% | 22.22% | 8.89%  |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    | Depth×Time  | 8.51%  | 10.64% | 36.17% | 27.66% | 17.02% |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     | Depth×Time  | 0.00%  | 13.97% | 86.03% | 0.00%  | 0.00%  |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    | Depth×Time  | 0.00%  | 14.63% | 85.37% | 0.00%  | 0.00%  |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     | Depth×Time  | 0.00%  | 4.22%  | 10.30% | 6.32%  | 79.16% |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    | Depth×Time  | 0.00%  | 2.89%  | 8.43%  | 2.89%  | 85.78% |

---

# I. Hint vs no-hint deltas

Positive delta means hint improved the metric; negative delta means hint reduced the metric.

| Dataset        |   Exits | Metric                     |   No hint |   Hint |   Delta Hint-No |
|:---------------|--------:|:---------------------------|----------:|-------:|----------------:|
| 2-class moth   |       3 | Segment policy acc         |     0.982 |  0.994 |           0.012 |
| 2-class moth   |       3 | Full-clip acc              |     1     |  1     |           0     |
| 2-class moth   |       3 | Depth×Time clip acc        |     1     |  1     |           0     |
| 2-class moth   |       3 | Depth×Time compute saved % |    82.48  | 82.77  |           0.29  |
| 2-class moth   |       3 | Avg exit depth             |     1.862 |  1.911 |           0.049 |
| 2-class moth   |       5 | Segment policy acc         |     0.978 |  0.969 |          -0.009 |
| 2-class moth   |       5 | Full-clip acc              |     1     |  1     |           0     |
| 2-class moth   |       5 | Depth×Time clip acc        |     1     |  1     |           0     |
| 2-class moth   |       5 | Depth×Time compute saved % |    82.25  | 79.53  |          -2.719 |
| 2-class moth   |       5 | Avg exit depth             |     2.462 |  2.36  |          -0.102 |
| 10-class audio |       3 | Segment policy acc         |     0.699 |  0.617 |          -0.082 |
| 10-class audio |       3 | Full-clip acc              |     0.816 |  0.783 |          -0.033 |
| 10-class audio |       3 | Depth×Time clip acc        |     0.816 |  0.783 |          -0.033 |
| 10-class audio |       3 | Depth×Time compute saved % |    34.34  | 26.25  |          -8.098 |
| 10-class audio |       3 | Avg exit depth             |     2.873 |  2.87  |          -0.003 |
| 10-class audio |       5 | Segment policy acc         |     0.688 |  0.643 |          -0.044 |
| 10-class audio |       5 | Full-clip acc              |     0.789 |  0.737 |          -0.053 |
| 10-class audio |       5 | Depth×Time clip acc        |     0.789 |  0.73  |          -0.059 |
| 10-class audio |       5 | Depth×Time compute saved % |    29.89  | 32.21  |           2.327 |
| 10-class audio |       5 | Avg exit depth             |     4.612 |  4.748 |           0.137 |

---

# J. 3-exit vs 5-exit deltas

Positive delta means 5 exits improved the metric relative to 3 exits.

| Dataset        | Hint   | Metric                     |   3 exits |   5 exits |   Delta 5-3 |
|:---------------|:-------|:---------------------------|----------:|----------:|------------:|
| 2-class moth   | No     | Segment policy acc         |     0.982 |     0.978 |      -0.003 |
| 2-class moth   | No     | Full-clip acc              |     1     |     1     |       0     |
| 2-class moth   | No     | Depth×Time clip acc        |     1     |     1     |       0     |
| 2-class moth   | No     | Depth×Time compute saved % |    82.48  |    82.25  |      -0.229 |
| 2-class moth   | No     | Avg exit depth             |     1.862 |     2.462 |       0.6   |
| 2-class moth   | Yes    | Segment policy acc         |     0.994 |     0.969 |      -0.025 |
| 2-class moth   | Yes    | Full-clip acc              |     1     |     1     |       0     |
| 2-class moth   | Yes    | Depth×Time clip acc        |     1     |     1     |       0     |
| 2-class moth   | Yes    | Depth×Time compute saved % |    82.77  |    79.53  |      -3.239 |
| 2-class moth   | Yes    | Avg exit depth             |     1.911 |     2.36  |       0.449 |
| 10-class audio | No     | Segment policy acc         |     0.699 |     0.688 |      -0.012 |
| 10-class audio | No     | Full-clip acc              |     0.816 |     0.789 |      -0.026 |
| 10-class audio | No     | Depth×Time clip acc        |     0.816 |     0.789 |      -0.026 |
| 10-class audio | No     | Depth×Time compute saved % |    34.34  |    29.89  |      -4.459 |
| 10-class audio | No     | Avg exit depth             |     2.873 |     4.612 |       1.738 |
| 10-class audio | Yes    | Segment policy acc         |     0.617 |     0.643 |       0.026 |
| 10-class audio | Yes    | Full-clip acc              |     0.783 |     0.737 |      -0.046 |
| 10-class audio | Yes    | Depth×Time clip acc        |     0.783 |     0.73  |      -0.053 |
| 10-class audio | Yes    | Depth×Time compute saved % |    26.25  |    32.21  |       5.967 |
| 10-class audio | Yes    | Avg exit depth             |     2.87  |     4.748 |       1.878 |

---

# K. C-class full-clip per-class F1

| Class         | 3exit_cclass_greedy F1   |   3exit_cclass_greedy support | 3exit_cclass_greedy_hint F1   |   3exit_cclass_greedy_hint support | 5exit_cclass_greedy F1   |   5exit_cclass_greedy support | 5exit_cclass_greedy_hint F1   |   5exit_cclass_greedy_hint support |
|:--------------|:-------------------------|------------------------------:|:------------------------------|-----------------------------------:|:-------------------------|------------------------------:|:------------------------------|-----------------------------------:|
| car_crash     | 70.59%                   |                            14 | 73.33%                        |                                 14 | 69.23%                   |                            14 | 75.86%                        |                                 14 |
| conversation  | 95.65%                   |                            12 | 95.65%                        |                                 12 | 95.65%                   |                            12 | 95.65%                        |                                 12 |
| engine_idling | 82.35%                   |                            10 | 46.15%                        |                                 10 | 46.15%                   |                            10 | 33.33%                        |                                 10 |
| fireworks     | 0.00%                    |                             2 | 0.00%                         |                                  2 | 0.00%                    |                             2 | 0.00%                         |                                  2 |
| gun_shot      | 90.00%                   |                            28 | 91.53%                        |                                 28 | 94.74%                   |                            28 | 90.57%                        |                                 28 |
| rain          | 66.67%                   |                            15 | 58.33%                        |                                 15 | 57.14%                   |                            15 | 60.87%                        |                                 15 |
| road_traffic  | 90.00%                   |                            18 | 84.21%                        |                                 18 | 85.71%                   |                            18 | 66.67%                        |                                 18 |
| scream        | 87.80%                   |                            23 | 85.00%                        |                                 23 | 93.33%                   |                            23 | 93.33%                        |                                 23 |
| thunderstorm  | 40.00%                   |                            15 | 66.67%                        |                                 15 | 51.61%                   |                            15 | 40.00%                        |                                 15 |
| wind          | 100.00%                  |                            15 | 85.71%                        |                                 15 | 89.66%                   |                            15 | 73.17%                        |                                 15 |

---

# L. C-class per-class precision, recall, F1, and support

| Variant                  | Mode       | Class         | Precision   | Recall   | F1      |   Support |
|:-------------------------|:-----------|:--------------|:------------|:---------|:--------|----------:|
| 3exit_cclass_greedy      | Full clip  | car_crash     | 60.00%      | 85.71%   | 70.59%  |        14 |
| 3exit_cclass_greedy      | Full clip  | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 3exit_cclass_greedy      | Full clip  | engine_idling | 100.00%     | 70.00%   | 82.35%  |        10 |
| 3exit_cclass_greedy      | Full clip  | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 3exit_cclass_greedy      | Full clip  | gun_shot      | 84.38%      | 96.43%   | 90.00%  |        28 |
| 3exit_cclass_greedy      | Full clip  | rain          | 57.14%      | 80.00%   | 66.67%  |        15 |
| 3exit_cclass_greedy      | Full clip  | road_traffic  | 81.82%      | 100.00%  | 90.00%  |        18 |
| 3exit_cclass_greedy      | Full clip  | scream        | 100.00%     | 78.26%   | 87.80%  |        23 |
| 3exit_cclass_greedy      | Full clip  | thunderstorm  | 80.00%      | 26.67%   | 40.00%  |        15 |
| 3exit_cclass_greedy      | Full clip  | wind          | 100.00%     | 100.00%  | 100.00% |        15 |
| 3exit_cclass_greedy      | Depth×Time | car_crash     | 60.00%      | 85.71%   | 70.59%  |        14 |
| 3exit_cclass_greedy      | Depth×Time | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 3exit_cclass_greedy      | Depth×Time | engine_idling | 100.00%     | 70.00%   | 82.35%  |        10 |
| 3exit_cclass_greedy      | Depth×Time | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 3exit_cclass_greedy      | Depth×Time | gun_shot      | 84.38%      | 96.43%   | 90.00%  |        28 |
| 3exit_cclass_greedy      | Depth×Time | rain          | 57.14%      | 80.00%   | 66.67%  |        15 |
| 3exit_cclass_greedy      | Depth×Time | road_traffic  | 81.82%      | 100.00%  | 90.00%  |        18 |
| 3exit_cclass_greedy      | Depth×Time | scream        | 100.00%     | 78.26%   | 87.80%  |        23 |
| 3exit_cclass_greedy      | Depth×Time | thunderstorm  | 80.00%      | 26.67%   | 40.00%  |        15 |
| 3exit_cclass_greedy      | Depth×Time | wind          | 100.00%     | 100.00%  | 100.00% |        15 |
| 3exit_cclass_greedy_hint | Full clip  | car_crash     | 68.75%      | 78.57%   | 73.33%  |        14 |
| 3exit_cclass_greedy_hint | Full clip  | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 3exit_cclass_greedy_hint | Full clip  | engine_idling | 100.00%     | 30.00%   | 46.15%  |        10 |
| 3exit_cclass_greedy_hint | Full clip  | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 3exit_cclass_greedy_hint | Full clip  | gun_shot      | 87.10%      | 96.43%   | 91.53%  |        28 |
| 3exit_cclass_greedy_hint | Full clip  | rain          | 77.78%      | 46.67%   | 58.33%  |        15 |
| 3exit_cclass_greedy_hint | Full clip  | road_traffic  | 80.00%      | 88.89%   | 84.21%  |        18 |
| 3exit_cclass_greedy_hint | Full clip  | scream        | 100.00%     | 73.91%   | 85.00%  |        23 |
| 3exit_cclass_greedy_hint | Full clip  | thunderstorm  | 50.00%      | 100.00%  | 66.67%  |        15 |
| 3exit_cclass_greedy_hint | Full clip  | wind          | 92.31%      | 80.00%   | 85.71%  |        15 |
| 3exit_cclass_greedy_hint | Depth×Time | car_crash     | 68.75%      | 78.57%   | 73.33%  |        14 |
| 3exit_cclass_greedy_hint | Depth×Time | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 3exit_cclass_greedy_hint | Depth×Time | engine_idling | 100.00%     | 30.00%   | 46.15%  |        10 |
| 3exit_cclass_greedy_hint | Depth×Time | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 3exit_cclass_greedy_hint | Depth×Time | gun_shot      | 87.10%      | 96.43%   | 91.53%  |        28 |
| 3exit_cclass_greedy_hint | Depth×Time | rain          | 77.78%      | 46.67%   | 58.33%  |        15 |
| 3exit_cclass_greedy_hint | Depth×Time | road_traffic  | 80.00%      | 88.89%   | 84.21%  |        18 |
| 3exit_cclass_greedy_hint | Depth×Time | scream        | 100.00%     | 73.91%   | 85.00%  |        23 |
| 3exit_cclass_greedy_hint | Depth×Time | thunderstorm  | 50.00%      | 100.00%  | 66.67%  |        15 |
| 3exit_cclass_greedy_hint | Depth×Time | wind          | 92.31%      | 80.00%   | 85.71%  |        15 |
| 5exit_cclass_greedy      | Full clip  | car_crash     | 75.00%      | 64.29%   | 69.23%  |        14 |
| 5exit_cclass_greedy      | Full clip  | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 5exit_cclass_greedy      | Full clip  | engine_idling | 100.00%     | 30.00%   | 46.15%  |        10 |
| 5exit_cclass_greedy      | Full clip  | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 5exit_cclass_greedy      | Full clip  | gun_shot      | 93.10%      | 96.43%   | 94.74%  |        28 |
| 5exit_cclass_greedy      | Full clip  | rain          | 50.00%      | 66.67%   | 57.14%  |        15 |
| 5exit_cclass_greedy      | Full clip  | road_traffic  | 75.00%      | 100.00%  | 85.71%  |        18 |
| 5exit_cclass_greedy      | Full clip  | scream        | 95.45%      | 91.30%   | 93.33%  |        23 |
| 5exit_cclass_greedy      | Full clip  | thunderstorm  | 50.00%      | 53.33%   | 51.61%  |        15 |
| 5exit_cclass_greedy      | Full clip  | wind          | 92.86%      | 86.67%   | 89.66%  |        15 |
| 5exit_cclass_greedy      | Depth×Time | car_crash     | 75.00%      | 64.29%   | 69.23%  |        14 |
| 5exit_cclass_greedy      | Depth×Time | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 5exit_cclass_greedy      | Depth×Time | engine_idling | 100.00%     | 30.00%   | 46.15%  |        10 |
| 5exit_cclass_greedy      | Depth×Time | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 5exit_cclass_greedy      | Depth×Time | gun_shot      | 93.10%      | 96.43%   | 94.74%  |        28 |
| 5exit_cclass_greedy      | Depth×Time | rain          | 50.00%      | 66.67%   | 57.14%  |        15 |
| 5exit_cclass_greedy      | Depth×Time | road_traffic  | 75.00%      | 100.00%  | 85.71%  |        18 |
| 5exit_cclass_greedy      | Depth×Time | scream        | 95.45%      | 91.30%   | 93.33%  |        23 |
| 5exit_cclass_greedy      | Depth×Time | thunderstorm  | 50.00%      | 53.33%   | 51.61%  |        15 |
| 5exit_cclass_greedy      | Depth×Time | wind          | 92.86%      | 86.67%   | 89.66%  |        15 |
| 5exit_cclass_greedy_hint | Full clip  | car_crash     | 73.33%      | 78.57%   | 75.86%  |        14 |
| 5exit_cclass_greedy_hint | Full clip  | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 5exit_cclass_greedy_hint | Full clip  | engine_idling | 100.00%     | 20.00%   | 33.33%  |        10 |
| 5exit_cclass_greedy_hint | Full clip  | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 5exit_cclass_greedy_hint | Full clip  | gun_shot      | 96.00%      | 85.71%   | 90.57%  |        28 |
| 5exit_cclass_greedy_hint | Full clip  | rain          | 45.16%      | 93.33%   | 60.87%  |        15 |
| 5exit_cclass_greedy_hint | Full clip  | road_traffic  | 100.00%     | 50.00%   | 66.67%  |        18 |
| 5exit_cclass_greedy_hint | Full clip  | scream        | 95.45%      | 91.30%   | 93.33%  |        23 |
| 5exit_cclass_greedy_hint | Full clip  | thunderstorm  | 50.00%      | 33.33%   | 40.00%  |        15 |
| 5exit_cclass_greedy_hint | Full clip  | wind          | 57.69%      | 100.00%  | 73.17%  |        15 |
| 5exit_cclass_greedy_hint | Depth×Time | car_crash     | 71.43%      | 71.43%   | 71.43%  |        14 |
| 5exit_cclass_greedy_hint | Depth×Time | conversation  | 100.00%     | 91.67%   | 95.65%  |        12 |
| 5exit_cclass_greedy_hint | Depth×Time | engine_idling | 100.00%     | 20.00%   | 33.33%  |        10 |
| 5exit_cclass_greedy_hint | Depth×Time | fireworks     | 0.00%       | 0.00%    | 0.00%   |         2 |
| 5exit_cclass_greedy_hint | Depth×Time | gun_shot      | 96.00%      | 85.71%   | 90.57%  |        28 |
| 5exit_cclass_greedy_hint | Depth×Time | rain          | 45.16%      | 93.33%   | 60.87%  |        15 |
| 5exit_cclass_greedy_hint | Depth×Time | road_traffic  | 100.00%     | 50.00%   | 66.67%  |        18 |
| 5exit_cclass_greedy_hint | Depth×Time | scream        | 95.45%      | 91.30%   | 93.33%  |        23 |
| 5exit_cclass_greedy_hint | Depth×Time | thunderstorm  | 45.45%      | 33.33%   | 38.46%  |        15 |
| 5exit_cclass_greedy_hint | Depth×Time | wind          | 57.69%      | 100.00%  | 73.17%  |        15 |

---

# M. Runtime and profiling summary

| Dataset        | Variant                  |   Exits | Hint   |   Expected MFLOPs |   Full MFLOPs | Compute saving %   | Latency mean ms   | Latency p50 ms   |
|:---------------|:-------------------------|--------:|:-------|------------------:|--------------:|:-------------------|:------------------|:-----------------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     |             20.39 |         51.63 | 60.51%             | —                 | —                |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    |             21.51 |         51.63 | 58.33%             | —                 | —                |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     |             15.68 |         51.63 | 69.63%             | —                 | —                |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    |             15.18 |         51.63 | 70.59%             | —                 | —                |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     |             47.43 |         51.63 | 8.14%              | —                 | —                |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    |             47.32 |         51.63 | 8.35%              | —                 | —                |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     |             45.1  |         51.63 | 12.65%             | —                 | —                |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    |             47.55 |         51.63 | 7.89%              | —                 | —                |

---

# N. Per-class split segment counts

| Dataset        | Variant                  | Class         |   Train |   Val |   Test |   Total |
|:---------------|:-------------------------|:--------------|--------:|------:|-------:|--------:|
| 2-class moth   | 3exit_2class_greedy      | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 3exit_2class_greedy      | male          |     502 |   141 |     76 |     719 |
| 2-class moth   | 3exit_2class_greedy_hint | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 3exit_2class_greedy_hint | male          |     502 |   141 |     76 |     719 |
| 2-class moth   | 5exit_2class_greedy      | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 5exit_2class_greedy      | male          |     502 |   141 |     76 |     719 |
| 2-class moth   | 5exit_2class_greedy_hint | female        |    1144 |   105 |    249 |    1498 |
| 2-class moth   | 5exit_2class_greedy_hint | male          |     502 |   141 |     76 |     719 |
| 10-class audio | 3exit_cclass_greedy      | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 3exit_cclass_greedy      | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 3exit_cclass_greedy      | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 3exit_cclass_greedy      | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 3exit_cclass_greedy      | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 3exit_cclass_greedy      | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 3exit_cclass_greedy      | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 3exit_cclass_greedy      | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 3exit_cclass_greedy      | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 3exit_cclass_greedy      | wind          |     350 |    75 |     75 |     500 |
| 10-class audio | 3exit_cclass_greedy_hint | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 3exit_cclass_greedy_hint | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 3exit_cclass_greedy_hint | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 3exit_cclass_greedy_hint | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 3exit_cclass_greedy_hint | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 3exit_cclass_greedy_hint | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 3exit_cclass_greedy_hint | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 3exit_cclass_greedy_hint | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 3exit_cclass_greedy_hint | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 3exit_cclass_greedy_hint | wind          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy      | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 5exit_cclass_greedy      | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 5exit_cclass_greedy      | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 5exit_cclass_greedy      | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 5exit_cclass_greedy      | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 5exit_cclass_greedy      | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy      | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 5exit_cclass_greedy      | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 5exit_cclass_greedy      | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 5exit_cclass_greedy      | wind          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy_hint | car_crash     |     256 |    55 |     99 |     410 |
| 10-class audio | 5exit_cclass_greedy_hint | conversation  |     192 |    47 |     43 |     282 |
| 10-class audio | 5exit_cclass_greedy_hint | engine_idling |     199 |    50 |     49 |     298 |
| 10-class audio | 5exit_cclass_greedy_hint | fireworks     |      59 |    11 |      9 |      79 |
| 10-class audio | 5exit_cclass_greedy_hint | gun_shot      |     290 |    42 |     40 |     372 |
| 10-class audio | 5exit_cclass_greedy_hint | rain          |     350 |    75 |     75 |     500 |
| 10-class audio | 5exit_cclass_greedy_hint | road_traffic  |     425 |    90 |     90 |     605 |
| 10-class audio | 5exit_cclass_greedy_hint | scream        |     242 |    47 |     53 |     342 |
| 10-class audio | 5exit_cclass_greedy_hint | thunderstorm  |     350 |    74 |     75 |     499 |
| 10-class audio | 5exit_cclass_greedy_hint | wind          |     350 |    75 |     75 |     500 |

---

# O. Threshold calibration summary

| Dataset        | Variant                  |   Exits | Hint   |   Tau | Val macro F1   | Val acc   |
|:---------------|:-------------------------|--------:|:-------|------:|:---------------|:----------|
| 2-class moth   | 3exit_2class_greedy      |       3 | No     |  0.9  | 97.51%         | 97.56%    |
| 2-class moth   | 3exit_2class_greedy_hint |       3 | Yes    |  0.95 | 98.35%         | 98.37%    |
| 2-class moth   | 5exit_2class_greedy      |       5 | No     |  0.95 | 98.34%         | 98.37%    |
| 2-class moth   | 5exit_2class_greedy_hint |       5 | Yes    |  0.9  | 96.67%         | 96.75%    |
| 10-class audio | 3exit_cclass_greedy      |       3 | No     |  0.92 | 67.65%         | 71.38%    |
| 10-class audio | 3exit_cclass_greedy_hint |       3 | Yes    |  0.85 | 70.53%         | 71.73%    |
| 10-class audio | 5exit_cclass_greedy      |       5 | No     |  0.9  | 67.59%         | 71.73%    |
| 10-class audio | 5exit_cclass_greedy_hint |       5 | Yes    |  0.95 | 66.28%         | 69.96%    |

---

# P. Appendix interpretation

## P.1 Why moth remains strong

The moth task is binary and acoustically narrower. All four moth variants reached:

```text
Full-clip accuracy:   100%
Depth×Time accuracy: 100%
```

This confirms that the new generic preprocessing/export pathway is not causing the lower C-class accuracy.

## P.2 Why C-class is harder

The C-class task is harder because:

- the number of classes increases from 2 to 10
- clips vary from very short to very long
- several classes are acoustically similar
- `fireworks` has very low support
- environmental classes may overlap in frequency structure
- event classes and background classes coexist in the same pipeline

## P.3 Why 3exit_cclass_greedy is the current C-class baseline

`3exit_cclass_greedy` has the best C-class balance:

```text
Segment policy accuracy: 69.90%
Full-clip accuracy:      81.58%
Depth×Time accuracy:    81.58%
Compute saved:          34.34%
```

It also avoids the excessive final-exit dependence seen in 5-exit C-class runs.

## P.4 Why hint passing is not yet C-class ready

Hint passing improves compact moth performance but hurts C-class. This may happen because early exits in a 10-class setting are much less reliable. If their uncertain predictions are passed forward as hints, the later exits may receive noisy guidance.

Future hint experiments should include confidence gating, entropy/margin checks, stronger regularization, or selective hint passing.

## P.5 Why 5 exits do not yet help C-class

The 5-exit C-class variants mostly exit at the final exit, which means the additional exits are not confidently solving the task early. This increases average exit depth and reduces the expected efficiency benefit.

## P.6 Recommended next experiments

1. Add class-balanced sampling.
2. Add source-file-balanced sampling.
3. Tune class caps and window length.
4. Try wider audio frequency ranges for non-moth data.
5. Train C-class models with stronger auxiliary exit supervision.
6. Re-test hint passing with confidence gating.
7. Add macro-F1-first threshold selection.
8. Run at least three seeds for the final selected C-class baseline.
