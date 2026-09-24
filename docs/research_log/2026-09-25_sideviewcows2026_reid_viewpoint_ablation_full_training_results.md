# SideViewCows2026 Re-ID + Viewpoint Ablation Full 30-Epoch Training Results & Protocol A Evaluation

**Date**: 2026-09-25  
**Author**: Hasin Ishrak  
**Execution Environment**: Modal Cloud Profile `dryousufmozumder`  
**Hardware Tier**: NVIDIA L40S GPU (48 GB VRAM, 8 vCPUs, 32 GB RAM)  
**Modal App ID**: `ap-AeQjQdRmqaL05QDCRVtGJi`  
**Target Milestone**: Controlled Representation Ablation — Re-ID + Viewpoint Prior vs Run 6 GT Mask Baseline

---

## 1. Executive Summary

Full 30-epoch training and official Protocol A retrieval evaluation of the **SideViewCows2026 Re-ID + Viewpoint Ablation** were completed on Modal profile `dryousufmozumder` using an NVIDIA L40S GPU.

The architecture combines the Run 6 4-channel `[R, G, B, Mask]` ResNet-18 spatial trunk with a frozen real-cattle viewpoint classification prior (`viewpoint_resnet18_real_best.pth`, 3-class front/side/rear probabilities transformed via a 16-D MLP) into a 528-D normalized metric embedding. Trainable parameters are 11,201,737 (+1,056 params / +0.0094% vs Run 6).

On the held-out Protocol A evaluation across 69 unseen cows against 36,811 parlor gallery images, the empirical retrieval results are:
* **Query Snapshots -> Gallery Parlor** (607 queries across 63 cows):
  * **Rank-1 Accuracy: 65.40%** (vs 62.93% in Run 6 and 38.88% in Run 3 RGB)
  * **Rank-5 Accuracy: 78.42%** (vs 75.29% in Run 6 and 57.17% in Run 3 RGB)
  * **Rank-10 Accuracy: 83.03%** (vs 81.05% in Run 6 and 64.58% in Run 3 RGB)
  * **Mean Average Precision (mAP): 41.17%** (vs 40.42% in Run 6 and 27.05% in Run 3 RGB)
* **Query Barn -> Gallery Parlor** (25,260 queries across 69 cows):
  * **Rank-1 Accuracy: 63.41%** (vs 63.90% in Run 6 and 58.64% in Run 3 RGB)
  * **Rank-5 Accuracy: 77.29%** (vs 77.10% in Run 6 and 78.19% in Run 3 RGB)
  * **Rank-10 Accuracy: 83.02%** (vs 82.58% in Run 6 and 83.72% in Run 3 RGB)
  * **Mean Average Precision (mAP): 38.32%** (vs 40.68% in Run 6 and 38.32% in Run 3 RGB)

Comparing against Run 6: Snapshots metrics improved across all retrieval thresholds and mAP, while Barn retrieval results were mixed (Rank-1 and mAP slightly lower, Rank-5 and Rank-10 slightly higher).

---

## 2. Protocol A Retrieval Comparison Table

| Metric | Run 3 (RGB Baseline) | Run 6 (GT Mask Baseline) | Run 6 + Viewpoint Prior (Ablation) | Delta vs Run 6 | Delta vs Run 3 RGB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Snapshots -> Parlor Rank-1** | 38.88% | 62.93% | **65.40%** | +2.47% | +26.52% |
| Snapshots -> Parlor Rank-5 | 57.17% | 75.29% | **78.42%** | +3.13% | +21.25% |
| Snapshots -> Parlor Rank-10 | 64.58% | 81.05% | **83.03%** | +1.98% | +18.45% |
| **Snapshots -> Parlor mAP** | 27.05% | 40.42% | **41.17%** | +0.75% | +14.12% |
| **Barn -> Parlor Rank-1** | 58.64% | **63.90%** | 63.41% | -0.49% | +4.77% |
| Barn -> Parlor Rank-5 | 78.19% | 77.10% | **77.29%** | +0.19% | -0.90% |
| Barn -> Parlor Rank-10 | **83.72%** | 82.58% | 83.02% | +0.44% | -0.70% |
| **Barn -> Parlor mAP** | 38.32% | **40.68%** | 38.32% | -2.36% | 0.00% |

---

## 3. Training & Validation Convergence

* **Best Validation Epoch**: Epoch 27
  * Validation Loss: 0.0548
  * Validation Top-1 Accuracy: 98.47%
  * Validation Balanced Accuracy: 98.63%
  * Validation Macro-F1: 0.9858
* **Final Epoch (Epoch 30)**:
  * Train Loss: 0.0002, Train Top-1 Accuracy: 100.0%
  * Validation Loss: 0.0616, Validation Top-1 Accuracy: 98.21%, Validation Macro-F1: 0.9828
* **Protocol A Evaluation Runtime**: 653.73 seconds across 62,678 held-out images.

---

## 4. Artifact Registry

* Model Checkpoints (Remote Volume `reid-checkpoints` on `dryousufmozumder`):
  * `/checkpoints/sideview_reid_viewpoint_ablation/reid_viewpoint_best.pth`
  * `/checkpoints/sideview_reid_viewpoint_ablation/reid_viewpoint_latest.pth`
* Local Artifacts:
  * `artifacts/reid_viewpoint_ablation/reid_viewpoint_metrics.json`
  * `artifacts/reid_viewpoint_ablation/reid_viewpoint_smoke_metrics.json`
  * `artifacts/reid_viewpoint_ablation/viewpoint_transfer_sanity_metrics.json`
  * `artifacts/reid_viewpoint_ablation/viewpoint_transfer_samples.csv`
  * `artifacts/reid_viewpoint_ablation/viewpoint_transfer_contact_sheet.jpg`
