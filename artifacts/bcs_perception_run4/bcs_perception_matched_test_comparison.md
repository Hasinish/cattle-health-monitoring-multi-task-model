# Phase 3 BCS Fair Matched-Subset Test Comparison

> **Scientific Protocol Notice:** The only statistically and scientifically valid direct comparison
> is between **Run 1 matched subset** and **Run 4 matched subset** on the EXACT SAME image identities
> where perception succeeded. The original 8,040-image Run 1 metric is reported for reference only
> and is NOT directly comparable to the smaller filtered perception subset.

## 1. Test Population & Perception Coverage

- **Canonical Full Test Count**: 8,040 images
- **Evaluated Test Manifest Rows**: 8,040 images
- **Successful Perception Test Count**: 7,549 images
- **Excluded Detection Failures**: 489 images (RT-DETR found no cow)
- **Excluded SAM Failures**: 2 images (SAM returned no mask)
- **Perception Coverage**: 93.89%

## 2. Primary Performance Comparison Table

| Metric | (1) Run 1 Original Full Test (Ref Only, N=8,040) | (2) Run 1 Matched Subset | (3) Run 4 Matched Subset | Delta (Run 4 - Run 1 Matched) |
| :--- | :---: | :---: | :---: | :---: |
| **Real BCS MAE (Primary)** | **0.1848** | **0.1929** | **0.1709** | **-0.0220 BCS units** |
| **Acc@1 (+/- 0.25 units)** | 86.74% | 84.95% | 89.40% | +4.45% |
| **Acc@0 (Exact match)** | 41.63% | 40.84% | 43.57% | +2.73% |
| **Balanced Accuracy** | 40.41% | 40.23% | 39.70% | -0.53% |
| **Macro-F1** | 0.4110 | 0.4074 | 0.4039 | -0.0035 |

## 3. Test Split Integrity Statement

- Canonical test labels and data were NOT used for training or checkpoint selection.
- A small test-subset plumbing evaluation was performed during pipeline verification.
- Final full Run 4 test evaluation remains post-training only.
- Test metrics must never affect checkpoint or hyperparameter selection.
