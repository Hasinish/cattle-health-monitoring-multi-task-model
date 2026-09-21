# Research Log: MOO ResNet-18 Full Directional Training & Real Cattle Diagnostic Evaluation

**Date**: 2026-09-21  
**Phase**: Phase 3 (Step 2.4 — Cattle Viewpoint Perception Feasibility)  
**Author / Operator**: Hasin Ishrak (in pair programming with Antigravity)  
**Status**: COMPLETED / AUDITED (STEP 2.4 IN PROGRESS; OPERATIONAL GENERATOR PENDING)  

---

## 1. Executive Summary

Following the initial frozen linear probe investigation (2026-09-20), we executed full fine-tuning of an ImageNet-pretrained ResNet-18 on the canonical identity-disjoint 8-direction MOO synthetic split (76,800 train images from 800 cows, 9,600 validation images from 100 cows, 9,600 test images from 100 cows; seed 2026) using an NVIDIA L4 GPU on Modal. The fine-tuned model achieved near-complete convergence in the synthetic domain, reaching **99.59% validation accuracy** (Macro-F1 0.9959) at Best Epoch 14 and **99.44% test accuracy** (Macro-F1 0.9944) on the held-out 9,600 synthetic test images across 100 cow identities.

Evaluation of this fine-tuned checkpoint on the 100-sample real cattle diagnostic benchmark (N=95 non-ambiguous samples; ScienceDB, MmCows, SideViewCows2026) yielded a **clean physical accuracy of 27.37% (26/95)** and **Macro-F1 of 0.1564**. False-front predictions decreased sharply from 40 (in the frozen linear probe) to **1**. However, the model exhibited a strong rear/rear-oblique prediction bias on real cattle (91/95 predictions mapped to `back`, `back-left`, or `back-right`), misclassifying 47 of 54 broadside lateral cows as rear-oblique or rear. While full fine-tuning substantially improves over the frozen linear probe baseline (12.63%), zero-shot synthetic-only transfer remains insufficient as a standalone operational viewpoint generator under this diagnostic setup.

---

## 2. Context & Motivation

In Step 2.4, earlier audits established:
1. Coarse cattle viewpoint taxonomy (`rear`, `rear-oblique`, `side`, `front-oblique`, `front`, `unknown / ambiguous`) is visually definable and practically necessary across Phase 3 datasets (`docs/audits/phase3_perception_feasibility.md`).
2. Frozen zero-shot VLMs (OpenAI CLIP, OpenCLIP, SigLIP) failed under tested setups (12.6%–33.7% accuracy, high false-front rates up to 66/95).
3. The preliminary frozen linear probe trained on 2,500 MOO synthetic images achieved 93.2% synthetic validation accuracy but collapsed to 12.63% accuracy with 40 false fronts on the 95 non-ambiguous real benchmark crops (`docs/research_log/2026-09-20_moo_viewpoint_synthetic_transfer.md`).
4. An apparent 63.16% post-hoc remapping result was formally audited and classified as an `exploratory post-hoc benchmark fit; invalid as final held-out evaluation evidence`, while independent visual inspection confirmed MOO's 3D coordinate system is anatomically aligned.

To determine whether full end-to-end representation learning on the complete MOO dataset could bridge the domain gap without post-hoc remapping, we implemented and executed the full fine-tuning pipeline on Modal.

---

## 3. Methodology & Execution

### 3.1 Training Pipeline Configuration
- **Script**: `scripts/modal_moo_pipeline.py::train_full_directional_classifier`
- **Hardware**: Modal Cloud Container with NVIDIA L4 GPU (24 GB VRAM), 8 vCPUs, 32 GB RAM.
- **Git Commit Provenance**: `d2c6578eaea060c4aeae9300d95ce5c77bbf25a7`
- **Dataset**: CEA Kalisteo MOO (Multi-view Oriented Observations), 8 canonical non-top directional classes (`front`, `front-left`, `left`, `back-left`, `back`, `back-right`, `right`, `front-right`).
- **Splits**:
  - `train.csv`: 800 cows, 76,800 images (SHA-256: `596a1a49c985223202bdf04b6d2b20d6544ab01d7683827fe65b70f6fc521e61`)
  - `val.csv`: 100 cows, 9,600 images (SHA-256: `7995c1e357cc33ccc17c9d70f0a210d1035839bfdd4ff26d7179201fdf42b261`)
  - `test.csv`: 100 cows, 9,600 images (SHA-256: `276603b9589f66d6d66f19140889d2e5ca369e69493815a23b47e13fb0ab4a8d`)
  - Identity-disjoint: 0 cow overlap, 0 image overlap across splits.
- **Hyperparameters**:
  - Architecture: `torchvision.models.resnet18(weights="IMAGENET1K_V1")` with trainable backbone and head (`fc = nn.Linear(512, 8)`).
  - Epochs: 15 (600 batches per epoch, batch size 128).
  - Optimizer: AdamW (`lr=3e-4`, `weight_decay=1e-4`).
  - Scheduler: CosineAnnealingLR (`T_max=15`, `eta_min=1e-6`).
  - Mixed Precision: PyTorch AMP (`GradScaler`, `autocast`).
  - Transforms: `RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))`, `ColorJitter(0.1, 0.1, 0.1)`, `Normalize` (viewpoint-preserving; strictly no horizontal flipping).
  - Checkpoint: `moo_resnet18_viewpoint8_full_l4.pth` (44.8 MB).

### 3.2 Real Cattle Diagnostic Evaluation Setup
- **Benchmark Manifest**: `artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv` (100 samples: ScienceDB: 34, MmCows: 33, SideViewCows2026: 33).
- **Benchmark Role**: Reused diagnostic/development benchmark, **not thesis-final held-out evidence**.
- **Important Benchmark Characteristic**: The 100-sample benchmark contains **zero true `front` samples** in its ground truth. Therefore, front recall and successful front recognition cannot be assessed on this set. False-front predictions can still be quantitatively tracked.
- **Input Crops**: Step 2.1 RT-DETR-L target bounding boxes recovered from `artifacts/perception_audit/localization_detections_expanded.csv` (86 crops, 14 full-image fallbacks; exact same input as zero-shot and linear probe audits).
- **Directional Mapping**: 8 directional classes mapped to 5 coarse evaluation classes:
  - `front` -> `front`
  - `front-left`, `front-right` -> `front-oblique`
  - `left`, `right` -> `side`
  - `back-left`, `back-right` -> `rear-oblique`
  - `back` -> `rear`
- **Evaluation Script**: `scripts/evaluate_moo_viewpoint.py`.

---

## 4. Empirical Findings & Results

### 4.1 Synthetic In-Domain Performance (MOO)

- **Best Validation Checkpoint**: Epoch 14 / 15
  - Best Val Accuracy: **99.59%**
  - Best Val Macro-F1: **0.9959**
- **Synthetic Test Evaluation (Best Epoch 14 Checkpoint)**:
  - Synthetic Test Accuracy: **99.44%**
  - Synthetic Test Macro-F1: **0.9944**
  - Evaluated on **9,600 held-out synthetic images across 100 unseen cow identities**.

### 4.2 Real Diagnostic Benchmark Performance (N=95 Non-Ambiguous Samples)

| Model / Approach | Supervision | Accuracy (N=95) | Macro-F1 | False Fronts (GT=0) |
| :--- | :--- | :--- | :--- | :--- |
| **Google SigLIP** (ViT-B/16-224) | Zero-Shot VLM | 12.63% | 0.0958 | 66 / 95 |
| **OpenCLIP** (LAION-2B ViT-B/32) | Zero-Shot VLM | 15.79% | 0.1149 | 13 / 95 |
| **MOO ResNet-18 (Linear Probe)** | Synthetic Linear Head (CPU) | 12.63% | 0.0674 | 40 / 95 |
| **OpenAI CLIP** (ViT-B/32) | Zero-Shot VLM | 33.68% | 0.2711 | 19 / 95 |
| **MOO ResNet-18 (Full L4 Fine-Tuned)** | **Full Synthetic Fine-Tuning** | **27.37% (26/95)** | **0.1564** | **1 / 95** |

### 4.3 Per-Dataset Diagnostic Accuracy

- **ScienceDB (BCS — rear-dominated chute)**: **58.82%** (20 / 34 correct)
- **MmCows (Behavior — CCTV indoor loose housing)**: **10.71%** (3 / 28 correct)
- **SideViewCows2026 (Re-ID — milking parlor side-views)**: **9.09%** (3 / 33 correct)

### 4.4 Diagnostic Coarse Confusion Matrix (N=95)

```text
                    Pred_front  Pred_front-oblique  Pred_side  Pred_rear-oblique  Pred_rear
True_front                   0                   0          0                  0          0
True_front-oblique           0                   0          0                  1          1
True_side                    1                   3          3                 30         17
True_rear-oblique            0                   0          0                  4          8
True_rear                    0                   1          1                  6         19
```

### 4.5 8-Direction Direct Prediction Distribution on Real Cattle

```text
back           : 49
back-left      : 35
back-right     :  7
right          :  4
front-right    :  3
front-left     :  1
front          :  1
```

---

## 5. Scientific Analysis & Observations

1. **Substantial Reduction in False Front Predictions**:
   In the frozen linear probe, the model produced 40 false-front predictions out of 95 samples. In the full fine-tuned model, false-front predictions dropped to **1**. Because the benchmark contains zero true `front` examples, this improvement demonstrates reduced false-positive frontal triggering, but does not establish whether true frontal views can be reliably recognized.

2. **Strong Rear / Rear-Oblique Prediction Bias**:
   91 of the 95 real predictions (95.8%) fall into `back`, `back-left`, or `back-right`.
   On real side-view cattle (`True_side`, N=54), 47 samples were predicted as rear-oblique (30) or rear (17), with only 3 predicted as side.
   On ScienceDB (where cattle are oriented rear-facing in the chute), this bias aligns with ground truth, yielding 58.82% accuracy and 70.4% recall on pure rear views (19/27).
   On lateral views (SideViewCows2026 and MmCows), this bias produces widespread misclassification.

3. **Domain Gap vs. Coordinate System**:
   Independent visual inspection had already confirmed that MOO coordinates are anatomically standard. The high synthetic accuracy (99.44%) juxtaposed with 27.37% real accuracy confirms that the performance limitation is a genuine sim-to-real visual domain gap (texture, lighting, background, and camera perspective differences between Blender renders and physical farm sensors), not a coordinate error.

4. **Status of Operational Viewpoint Generator**:
   Full synthetic MOO fine-tuning provides a substantially more discriminative feature backbone than the frozen linear probe, but zero-shot synthetic transfer remains insufficient as a standalone operational real-cattle viewpoint generator under this diagnostic setup.
   The operational viewpoint generator remains: **NOT YET SELECTED**.

---

## 6. Artifacts & File Registry

- **Modal Training Pipeline**: `scripts/modal_moo_pipeline.py` (includes `train_full_directional_classifier`, `benchmark_h200_throughput`, `benchmark_l4_throughput`)
- **Canonical Evaluation Script**: `scripts/evaluate_moo_viewpoint.py`
- **Model Checkpoint**: `artifacts/checkpoints/moo_resnet18_viewpoint8_full_l4.pth` (tracked on Modal volume `moo-data`; local copy gitignored per repository rules)
- **Evaluation Results**: `artifacts/perception_audit/viewpoint_moo_resnet18_l4_evaluation.csv`
- **Benchmark Manifest**: `artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv`

---

## 7. Next Steps

1. Maintain Step 2.4 as **REOPENED / IN PROGRESS**; do not freeze an operational viewpoint generator prematurely.
2. Evaluate candidate paths forward for Step 2.4:
   - **Path A**: Use the full fine-tuned ResNet-18 as an upstream cattle representation / feature extractor without hard thresholding.
   - **Path B**: Supervised cattle-specific viewpoint classification or calibration using real cattle training imagery outside this diagnostic benchmark (never train on the diagnostic benchmark).
   - **Path C**: Formulate viewpoint as an auxiliary multi-task head in Step 9 rather than an upstream hard filter for Step 3 caching.
