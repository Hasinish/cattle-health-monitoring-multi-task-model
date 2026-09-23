# Research Log: MOO-to-Real Viewpoint 3-Class Fine-Tuning Preparation & Modal Cloud Staging

**Date:** 2026-09-23  
**Status:** COMPLETE & AUDITED (TRAINING PIPELINE READY; RUN AWAITING MANUAL LAUNCH)  
**Topic:** Real Viewpoint Dataset Staging on Modal (`tigerwood693`), Leakage-Safe Stratified Group Split Generation, and MOO-to-Real ResNet-18 3-Class Transfer Pipeline Design  
**Source Dataset:** `datasets/viewpoint/self_clean_v1_rtdetr_crop/` (879 crops)  
**Modal Profile:** `tigerwood693`  
**Modal Persistent Volumes:**
  - `viewpoint-real-data`: Source cropped dataset (`/data/self_clean_v1_rtdetr_crop/`)
  - `moo-data`: Pretrained MOO synthetic checkpoint (`/moo_data/moo_resnet18_viewpoint8_full_l4.pth`)
  - `viewpoint-checkpoints`: Output directory for trained checkpoints and metrics (`/checkpoints/viewpoint_real_finetune/`)  
**Target Hardware for Future Training:** NVIDIA Tesla T4 (Modest cost, ~$0.15/hr)  

---

## 1. Executive Summary

Following the generation of the 879-image RT-DETR-L cropped real cattle viewpoint dataset (`self_clean_v1_rtdetr_crop`), we prepared the entire end-to-end cloud training and evaluation stack for transfer learning from synthetic MOO pretraining to authentic farm cattle. Under Modal profile `tigerwood693`, we verified that `moo-data` contains the canonical 8-direction synthetic checkpoint (`moo_resnet18_viewpoint8_full_l4.pth`), created the new persistent volume `viewpoint-real-data`, and implemented a live-progress resumable upload tool (`scripts/upload_viewpoint_crops_modal.py`) utilizing minimal CPU resources.

We constructed deterministic (Seed 2026) group-stratified train/val/test splits strictly partitioned by `duplicate_group_id` (`train`: 616 samples, 70.1%; `val`: 132 samples, 15.0%; `test`: 131 samples, 14.9%) with **0 group overlap across all partitions**. We implemented the PyTorch training engine (`scripts/train_moo_real_viewpoint.py`) which initializes a ResNet-18 from the MOO backbone weights, replaces the old 8-direction classification head with a fresh `Linear(512, 3)` head (`front`, `side`, `rear`), applies label-preserving bilateral augmentations (`RandomHorizontalFlip`, `RandomResizedCrop`, `ColorJitter`), and tracks `val_macro_f1` as the primary model selection criterion. A minimal local smoke test verified 100% path resolution, parameter transfer, forward/backward pass, and checkpoint serialization without ever loading the frozen test set. We built the Modal cloud wrapper (`scripts/modal_train_moo_real_viewpoint.py`) ready for manual execution on an NVIDIA Tesla T4 GPU. In strict accordance with user constraints, **paid full training was NOT launched**.

---

## 2. Context & Motivation

In earlier Step 2.4 audits, zero-shot Vision-Language Models (CLIP, SigLIP) collapsed on real cattle viewpoints (12.6%–33.7% accuracy), and zero-shot transfer of a synthetic-only MOO classifier achieved only 27.37% clean accuracy due to a severe sim-to-real visual domain gap (texture, lighting, background, and camera angle shifts). While synthetic data alone cannot serve as an operational viewpoint estimator, its learned feature representations (dorsal/lateral/caudal animal contours) provide a strong visual prior for fine-tuning.

With the finalized real-viewpoint collection cleaned and cropped to 879 single-cow images, fine-tuning the MOO-pretrained ResNet-18 on real data tests whether pretraining on synthetic cattle geometry accelerates convergence and improves real-world viewpoint classification compared to training from scratch or frozen probing.

---

## 3. Dataset Splitting & Provenance Audit

### 3.1 Stratified Group Partitioning (Seed 2026)

To prevent train/test leakage arising from multi-angle shots, burst captures, or web duplicates, the dataset was partitioned strictly by `duplicate_group_id` using `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=2026)` (14 train folds, 3 val folds, 3 test folds).

| Split | Images | % of Total | Duplicate Groups | `front` | `side` | `rear` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **616** | 70.1% | 616 | 276 (44.8%) | 155 (25.2%) | 185 (30.0%) |
| **`val`** | **132** | 15.0% | 132 | 57 (43.2%) | 34 (25.8%) | 41 (31.1%) |
| **`test`** | **131** | 14.9% | 131 | 59 (45.0%) | 33 (25.2%) | 39 (29.8%) |
| **TOTAL** | **879** | **100.0%** | **879** | **392 (44.6%)** | **222 (25.3%)** | **265 (30.1%)** |

### 3.2 Leakage & Integrity Proof

- **Train vs. Val Group Overlap:** `0` (0.00%)
- **Train vs. Test Group Overlap:** `0` (0.00%)
- **Val vs. Test Group Overlap:** `0` (0.00%)
- **Quarantined Failure (`rear_0003`):** Verified excluded from all partitions.

### 3.3 Cryptographic Checksums (SHA-256)

- `datasets/viewpoint/self_clean_v1_rtdetr_crop/train.csv`: `26a8a6924a15dd0d346ab13bdbf3e9746c2216693d2ff2a6c1c785c631ab04af`
- `datasets/viewpoint/self_clean_v1_rtdetr_crop/val.csv`: `07cda3d88c24b40842c9e9e7dfa9e3ed147274b1b4084ec898e36aab1685cea2`
- `datasets/viewpoint/self_clean_v1_rtdetr_crop/test.csv`: `7dd3202dd9ff8421eb6df7541c7170ac13d40d8d3b1f96058edae9f1a33c2a41`

---

## 4. Architectural & Training Design

### 4.1 Backbone Weight Transfer & Head Reconfiguration
- **Pretrained Checkpoint:** `moo_resnet18_viewpoint8_full_l4.pth` (trained on 76.8k synthetic MOO renders, Best Val Macro-F1 0.9959).
- **Weight Transfer:** The ResNet-18 trunk (`conv1`, `bn1`, `layer1`–`layer4`, 512-D pooling) is initialized from the MOO checkpoint.
- **Head Replacement:** The 8-direction synthetic head (`fc.weight`: `[8, 512]`) is discarded and replaced with a fresh linear layer `nn.Linear(512, 3)` mapped to `["front", "side", "rear"]`.

### 4.2 Bilateral Symmetry Justification for Horizontal Flip
In fine-grained 8-direction classification, horizontal flipping changes `left` to `right` and is prohibited. However, in our coarse 3-class taxonomy (`front`, `side`, `rear`):
- A cow facing front remains facing front when flipped horizontally.
- A cow facing rear remains facing rear when flipped horizontally.
- A cow viewed from the left flank flips to the right flank, but **both belong to the single canonical class `side`**.
Therefore, `RandomHorizontalFlip(p=0.5)` is 100% label-preserving, provides valuable data augmentation, and eliminates arbitrary left/right bias.

### 4.3 Training Hyperparameters
- **Input Size:** 224x224 RGB (ImageNet mean/std normalization)
- **Batch Size:** 32 (Train batches per epoch: 20; Val batches: 5)
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- **Learning Rate Schedule:** CosineAnnealingLR (`T_max=20`, `eta_min=1e-6`)
- **Loss:** CrossEntropyLoss
- **Model Selection:** Best checkpoint chosen strictly by **Validation Macro-F1** (`val_macro_f1`).
- **Test Set Protection:** `test.csv` is NEVER loaded during training, early stopping, or hyperparameter validation.

---

## 5. Artifacts & File Registry

| File / Path | Description |
| :--- | :--- |
| `scripts/build_viewpoint_crop_splits.py` | Standalone script generating deterministic Seed-2026 group-stratified splits. |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/train.csv` | Canonical 616-sample train split manifest. |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/val.csv` | Canonical 132-sample validation split manifest. |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/test.csv` | Frozen 131-sample test split manifest. |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/split_report.md` | Comprehensive split audit report and distribution proofs. |
| `scripts/upload_viewpoint_crops_modal.py` | Live-progress resumable volume uploader and zero-GPU verification script. |
| `scripts/train_moo_real_viewpoint.py` | Core PyTorch fine-tuning engine with live tqdm progress bars and metrics calculation. |
| `scripts/modal_train_moo_real_viewpoint.py` | Modal cloud wrapper mounting all 3 volumes on `tigerwood693` with T4 GPU configuration. |

---

## 6. Exact Next Steps & Execution Command

The entire pipeline is ready and verified. To launch the full 20-epoch training run on Modal profile `tigerwood693`, run:

```bash
modal run --profile tigerwood693 scripts/modal_train_moo_real_viewpoint.py::main --epochs 20 --batch-size 32
```
