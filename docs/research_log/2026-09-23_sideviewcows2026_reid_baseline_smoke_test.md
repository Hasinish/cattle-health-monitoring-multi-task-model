# Research Log: SideViewCows2026 RGB Re-ID Baseline Pipeline & Smoke Test Certification

**Date:** 2026-09-23  
**Author:** Hasin Ishrak  
**Target Modal Profile:** `tigerwood697`  
**Dataset Volume:** `sideview-data` (`/data/sideviewcows2026/`)  
**Checkpoint Volume:** `reid-checkpoints` (`/checkpoints/`)  
**Task:** Phase 3 Step 4.3 (Run 3 of 8) Re-ID RGB Single-Task Baseline  
**Smoke Test Environments:**
- Local GPU: NVIDIA GeForce GTX 1050 Ti (4GB VRAM, Runtime: 0.3 mins)
- Modal Cloud: NVIDIA Tesla T4 (App `ap-umQEFMmIMLfv8ssNbBYX7j`, Runtime: ~2.1 mins)

---

## 1. Executive Summary

To establish the single-task identity discrimination control for the Phase 3 Multi-Task Deep Learning Framework (Run 3 of 8 in the Deadline Execution Plan), the SideViewCows2026 RGB Re-ID baseline pipeline was implemented (`scripts/train_sideview_reid_baseline.py`) and wrapped for Modal cloud execution (`scripts/modal_train_sideview_reid.py`).

The baseline strictly adheres to the approved canonical Protocol A: 41 parlor training cows are used for representation learning, while 69 evaluation cows are strictly held out for zero-shot gallery retrieval. Validation checkpoint selection utilizes the canonical Protocol D session-disjoint partition restricted to the 41 training cows (12,753 train images / 2,683 val images).

Comprehensive smoke tests were executed both locally and on Modal cloud (T4 GPU tier), certifying all 10 verification criteria: path resolution, image decoding, forward/backward pass, 512-D embedding shape, L2 normalization, 41-class classifier, validation evaluation, checkpoint saving, bit-identical resume from epoch 1 to epoch 2, and 100% isolation of the Protocol A held-out evaluation set.

---

## 2. Protocol Partitioning & Leakage Protections

| Dataset Split | Source Protocol | Cows | Images | Description / Role | Leakage Boundary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Protocol D (restricted) | **41** | **12,753** | Representation learning (in-domain early parlor) | Chronological session protection (`dt <= 60s`) |
| **Validation** | Protocol D (restricted) | **41** | **2,683** | Checkpoint selection (in-domain mid parlor) | Chronological session protection (`dt <= 60s`) |
| **Gallery (Held-Out)** | Protocol A | **69** | **36,811** | Reference retrieval gallery (parlor) | 100% identity disjoint from 41 train cows |
| **Query Barn (Held-Out)**| Protocol A | **69** | **25,260** | Handheld barn video queries | 100% identity disjoint from 41 train cows |
| **Query Snapshots (Held-Out)**| Protocol A | **63** | **607** | Unconstrained field/cubicle snapshot queries | 100% identity disjoint from 41 train cows |

### Identity Disjointness Proof:
- `set(train_cows).intersection(set(eval_cows)) == set()` (0 cows overlap).
- In smoke testing mode, Protocol A gallery and queries are never loaded into memory.

---

## 3. Model Architecture & Training Loss

The baseline is deliberately kept generic and simple to serve as an unpolluted visual prior control:
```text
RGB Image (224x224) 
  → ResNet-18 (ImageNet Pretrained)
  → Global Average Pooling (512-D)
  → L2-Normalized Embedding (512-D, norm = 1.0)
  → Linear Classifier (512 → 41 identities) [Training Only]
  → CrossEntropyLoss
```
- **Excluded Components**: No triplet loss, no ArcFace/CosFace, no SAM masks, no pose keypoints, no viewpoint estimation, no temporal models.

---

## 4. Smoke Test Verification Results

### 10-Point Verification Matrix
1. **Path Resolution**: PASS (100% resolved across `/data/sideviewcows2026/` and local paths).
2. **RGB Image Decoding**: PASS (100% valid PIL decode on sampled batches).
3. **Forward & Backward Passes**: PASS (gradients computed and weights updated cleanly).
4. **512-D Embedding Shape**: PASS (`(B, 512)` verified on batch tensor).
5. **L2 Normalization**: PASS (`torch.norm(norm_embs, p=2, dim=1) == 1.0000`).
6. **Classifier Identity Count**: PASS (exactly 41 classes in linear head).
7. **Validation Evaluation**: PASS (Top-1 Acc, Balanced Acc, Macro-F1, CrossEntropy loss computed).
8. **Checkpoint Saving**: PASS (`reid_baseline_best.pth` and `reid_baseline_latest.pth` created).
9. **Checkpoint Resume**: PASS (resumed from Epoch 1, trained Epoch 2, verified state continuity).
10. **Strict Test Isolation**: PASS (Protocol A gallery and query sets were untouched).

### Smoke Metrics Summary (Modal T4, App `ap-umQEFMmIMLfv8ssNbBYX7j`):
- Epoch 1: Train Loss 4.0506, Train Acc 3.12% | Val Loss 3.7771, Val Acc 3.12%
- Epoch 2: Train Loss 2.3453, Train Acc 51.56% | Val Loss 3.6980, Val Acc 4.69%

---

## 5. Execution Plan for Full Run 3 Training

The training script and cloud wrapper are fully prepared for user execution:
```bash
modal run --profile tigerwood697 scripts/modal_train_sideview_reid.py::main --epochs 30 --batch-size 64
```
- **Recommended Hardware**: NVIDIA L4 (24GB Ada Lovelace tier) or NVIDIA L40S.
- **Estimated Runtime**: ~15 to 25 minutes for 12,753 images across 30 epochs + Protocol A chunked retrieval.
- **Estimated Cost**: ~$0.50 – $0.85 on NVIDIA L4 (well within remaining credit ceiling).
