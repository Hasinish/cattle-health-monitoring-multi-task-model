# Phase 3 Step 4.2 Behavior RGB Baseline (30-Epoch Full Training Summary)

**Execution Date:** 2026-09-23  
**Status:** FULL_SUCCESS / 100% CERTIFIED ✅  
**App ID:** `ap-ZrBKKvGcVzM7IB1AYMs2LH` (Modal profile `tigerwood693`)  
**Hardware:** NVIDIA L40S GPU (48 GB VRAM), 8 CPUs, 32 GB RAM  
**Dataset:** Canonical CVB + Kaggle Beef (`datasets/behavior/cvb_beef/`; Train 3,785, Val 680, Test 809; Seed 2026)  
**Total Training Duration:** 821.47 seconds (~13.69 minutes)  
**Best Validation Epoch:** Epoch 25 (Val Macro-F1: 0.7399, Val Bal Acc: 73.40%, Val Acc: 87.06%)  

---

## 1. Final Held-Out Test Evaluation Results (N = 809)

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **88.88%** (719 / 809 correct) |
| **Balanced Accuracy** | **71.72%** (>3.5x random chance) |
| **Macro-F1** | **0.7413** |
| **Test Cross-Entropy Loss** | **0.5312** |

---

## 2. Per-Class Test Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Lying** | 0.9608 | 0.9496 | **0.9552** | 258 |
| **Feeding** | 0.8647 | 0.9885 | **0.9225** | 349 |
| **Drinking** | 0.8793 | 0.8095 | **0.8430** | 63 |
| **Standing** | 0.9481 | 0.6460 | **0.7684** | 113 |
| **Walking** (CVB only) | 0.2500 | 0.1923 | **0.2174** | 26 |

---

## 3. Sub-Dataset Breakdown

- **Kaggle Beef (387 test clips):** Accuracy = **94.06%**, Balanced Accuracy = **90.84%**, Macro-F1 = **0.9113**
- **CVB (422 test clips):** Accuracy = **84.12%**, Balanced Accuracy = **60.62%**, Macro-F1 = **0.6541**

---

## 4. Confusion Matrix

Classes: `[Standing, Lying, Feeding, Drinking, Walking]`

```
[[ 73,   9,  14,   7,  10],
 [  1, 245,  10,   0,   2],
 [  1,   0, 345,   0,   3],
 [  1,   0,  11,  51,   0],
 [  1,   1,  19,   0,   5]]
```

---

## 5. Checkpoints & Storage

- Best Checkpoint: `behavior_baseline_best.pth` on Modal volume `behavior-checkpoints`
- Latest Checkpoint: `behavior_baseline_latest.pth` on Modal volume `behavior-checkpoints`
- Metrics: `artifacts/behavior_baseline/behavior_baseline_metrics.json`
