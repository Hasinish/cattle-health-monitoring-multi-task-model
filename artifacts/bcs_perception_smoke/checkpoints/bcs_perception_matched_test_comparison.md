# Phase 3 BCS Fair Matched-Subset Test Comparison

> **Scientific Protocol Notice:** The only statistically and scientifically valid direct comparison
> is between **Run 1 matched subset** and **Run 4 matched subset** on the EXACT SAME image identities
> where perception succeeded. The original 8,040-image Run 1 metric is reported for reference only
> and is NOT directly comparable to the smaller filtered perception subset.

## 1. Test Population & Perception Coverage

- **Canonical Full Test Count**: 8,040 images
- **Evaluated Test Manifest Rows**: 10 images
- **Successful Perception Test Count**: 7 images
- **Excluded Detection Failures**: 3 images (RT-DETR found no cow)
- **Excluded SAM Failures**: 0 images (SAM returned no mask)
- **Perception Coverage**: 70.00%

## 2. Primary Performance Comparison Table

| Metric | (1) Run 1 Original Full Test (Ref Only, N=8,040) | (2) Run 1 Matched Subset | (3) Run 4 Matched Subset | Delta (Run 4 - Run 1 Matched) |
| :--- | :---: | :---: | :---: | :---: |
| **Real BCS MAE (Primary)** | **0.1848** | **0.3214** | **0.2857** | **-0.0357 BCS units** |
| **Acc@1 (+/- 0.25 units)** | 86.74% | 57.14% | 71.43% | +14.29% |
| **Acc@0 (Exact match)** | 41.63% | 14.29% | 14.29% | +0.00% |
| **Balanced Accuracy** | 40.41% | 12.50% | 25.00% | +12.50% |
| **Macro-F1** | 0.4110 | 0.1333 | 0.0833 | -0.0500 |

## 3. Test Split Integrity Statement

- Canonical test labels and data were NOT used for training or checkpoint selection.
- A small test-subset plumbing evaluation was performed during pipeline verification.
- Final full Run 4 test evaluation remains post-training only.
- Test metrics must never affect checkpoint or hyperparameter selection.
