# Phase 3 Run 5 vs Run 2 Matched Behavior Test Comparison

**Evaluation Date:** 2026-09-24 00:26:16  
**Matched Test Sample Count:** 780 (from 809 canonical candidates)

---

## 1. High-Level Performance Comparison

| Metric | Run 2 RGB Full (Historical) | Run 2 RGB Matched (N=780) | Run 5 Perception+TCN (N=780) | Delta (Run 5 - Run 2 Matched) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 88.88% | 88.46% | **87.44%** | **-1.02%** |
| **Balanced Accuracy** | 71.72% | 71.30% | **74.43%** | **+3.13%** |
| **Macro-F1** | 0.7413 | 0.7378 | **0.7397** | **+0.0019** |
| **Test Loss** | 0.5312 | 0.5505 | **0.4430** | **-0.1075** |

---

## 2. Per-Class F1 Score Comparison (Matched Subset)

| Behavior Class | Run 2 RGB Matched F1 | Run 5 Perception+TCN F1 | Delta (Run 5 - Run 2) | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Standing** | 0.7556 | **0.7215** | **-0.0341** | 108 |
| **Lying** | 0.9512 | **0.9169** | **-0.0343** | 237 |
| **Feeding** | 0.9218 | **0.9465** | **+0.0247** | 346 |
| **Drinking** | 0.8430 | **0.8682** | **+0.0252** | 63 |
| **Walking** *(CVB-only)* | 0.2174 | **0.2456** | **+0.0282** | 26 |

---

## 3. Dataset Source Breakdown (Matched Subset)

| Dataset | Metric | Run 2 RGB Matched | Run 5 Perception+TCN |
| :--- | :--- | :--- | :--- |
| **CVB (Barn CCTV, Multi-Cow)** | Accuracy | 84.12% | **80.09%** |
| | Macro-F1 | 0.6541 | **0.6188** |
| **Kaggle Beef (Single-Cow Clips)** | Accuracy | 93.58% | **96.09%** |
| | Macro-F1 | 0.9079 | **0.9414** |

---
*Generated automatically by Phase 3 Run 5 Test Evaluation Suite.*
