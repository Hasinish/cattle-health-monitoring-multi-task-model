# Research Log: Phase 3 Run 5 Behavior Perception-Enhanced TCN Full Training Results

**Date:** 2026-09-24  
**Author:** Hasin Ishrak  
**Target Modal Profile:** `tigerwood693`  
**GPU Tier:** NVIDIA L40S (24GB)  
**App ID:** `ap-S405sWmuNqeenUylBipoDg`  
**Volumes:** `behavior-checkpoints` (mounted at `/checkpoints`), `behavior-perception-cache` (mounted at `/cache`)  
**Epochs Completed:** 30 / 30  
**Best Checkpoint Epoch:** Epoch 9  
**Model Parameters:** 11,903,621 trainable parameters  

---

## 1. Executive Summary

Successfully completed the full 30-epoch training of Phase 3 Run 5 (Behavior Perception-Enhanced Temporal Model) on Modal (`tigerwood693`, NVIDIA L40S). The model couples cattle-centered RGB crops with SAM 2.1 Small binary foreground masks into a 4-channel ResNet-18 spatial feature extractor and a 1D Temporal Convolutional Network (TCN) over T=8 temporal frames.

Using the RAM-preloaded in-memory dataset (~6.5 GB uint8 tensors across 4,271 retained sequences), the training finished in ~17 minutes (~35 seconds per epoch) with zero disk I/O stalls. The best validation model was captured at **Epoch 9**, achieving:
- **Validation Macro-F1: 0.7722** (beating the Run 2 RGB single-task baseline of **0.7399** by **+3.23%**)
- **Validation Balanced Accuracy: 78.38%** (beating the Run 2 baseline of **73.40%** by **+4.98%**)
- **Validation Accuracy: 87.62%** (beating the Run 2 baseline of **87.06%** by **+0.56%**)

All checkpoints, optimizer states, and training records were committed to `/checkpoints/behavior_run5_perception/` on persistent volume `behavior-checkpoints`.

---

## 2. Context & Motivation

Phase 3 Run 5 tests whether conditioning behavior recognition on cattle-centered visual crops and foreground segmentation masks combined with temporal modeling (1D TCN across T=8 frames) outperforms the single-frame generic RGB baseline (Run 2). In Run 2, the baseline achieved 0.7399 Val Macro-F1 / 0.7413 Test Macro-F1, but severely struggled on minority and motion-dependent behaviors (e.g. Walking F1 was only 0.2174 because head-down postures were confused with Feeding). Run 5 introduces:
1. Spatial foreground isolation: RT-DETR-L crop + SAM 2.1 Small binary mask as a 4th channel.
2. Temporal trajectory modeling: T=8 frame features processed by a 2-block causal-like 1D TCN with residual skips.

---

## 3. Training & Validation Performance

### Head-to-Head: RGB Baseline (Run 2) vs Perception+TCN (Run 5)

| Metric | Run 2 (RGB Midpoint ResNet-18) | Run 5 (Perception + 1D TCN) | Delta (Run 5 vs Run 2) |
| :--- | :--- | :--- | :--- |
| **Backbone Channels** | 3 (RGB) | 4 (RGB + SAM 2.1 Mask) | +1 Channel (Mask) |
| **Temporal Frames** | 1 (Midpoint) | 8 (Evenly spaced across clip) | +7 Frames |
| **Model Architecture** | ResNet-18 | ResNet-18 (4ch) + 1D TCN | +727,109 params |
| **Trainable Parameters**| 11,176,512 | **11,903,621** | +727,109 (+6.5%) |
| **Best Val Epoch** | Epoch 25 | **Epoch 9** | Converged 16 epochs faster |
| **Val Macro-F1** | 0.7399 | **0.7722** | **+0.0323 (+3.23%)** |
| **Val Balanced Accuracy**| 73.40% | **78.38%** | **+4.98%** |
| **Val Accuracy** | 87.06% | **87.62%** | **+0.56%** |

### Per-Class Validation Breakdown (Best Epoch 9)

| Class | Precision | Recall | F1-Score | Support | Run 2 Baseline F1 | F1 Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lying** | 0.9364 | 0.9493 | **0.9428** | 217 | 0.9552 | -0.0124 |
| **Feeding** | 0.9672 | 0.9402 | **0.9535** | 251 | 0.9225 | **+0.0310** |
| **Drinking** | 0.9333 | 1.0000 | **0.9655** | 42 | 0.8430 | **+0.1225** |
| **Standing** | 0.6477 | 0.6064 | **0.6264** | 94 | 0.7684 | -0.1420 |
| **Walking (CVB-Only)** | 0.3333 | 0.4231 | **0.3729** | 26 | 0.2174 | **+0.1555 (+71.5% rel)** |

### Sub-Dataset Performance
- **Kaggle Beef (318 val sequences)**: Overall Accuracy: **92.45%**, Balanced Accuracy: **89.66%**, Macro-F1: **0.8982**
- **CVB (312 val sequences)**: Overall Accuracy: **82.69%**, Balanced Accuracy: **77.23%**, Macro-F1: **0.7561** (beating Run 2 baseline of 0.6541 by **+0.1020**)

### Execution Characteristics
- **Throughput**: ~35 seconds per epoch with zero disk seeks during training due to uint8 tensor in-memory preloading.
- **Total Duration**: 1,009.91s (~16.8 minutes) for all 30 epochs.
- **Checkpoints**:
  - Best model: `/checkpoints/behavior_run5_perception/behavior_tcn_best.pth`
  - Latest model: `/checkpoints/behavior_run5_perception/behavior_tcn_latest.pth`

---

## 4. Architectural Decisions & Reproducibility

1. **Scientific Unit & Grouping Safeguards**:
   - The primary Behavior dataset unit is strictly grouped **sequence/sample** (Train 3,785 canonical, 3,641 retained; Val 680 canonical, 630 retained).
   - CVB is protected by source-video grouping; Kaggle Beef is protected by recording-session grouping.
   - This partition is sequence-safe and source-disjoint; it is NEVER described as cow-disjoint or "held-out cows".
2. **Strict Test Set Protection Preserved**:
   - `test.csv` (809 candidate sequences) was NOT loaded, seen, or evaluated during the 30-epoch training.
   - Checkpoint selection was determined strictly by validation Macro-F1 on the 630 retained validation sequences.
3. **Full Training Reproducibility & Provenance**:
   - Seed: 2026 across Python random, NumPy, PyTorch CPU and CUDA.
   - cuDNN deterministic = True, cuDNN benchmark = False, python_hash_seed = "2026".
   - Canonical Train CSV SHA-256: `117d3191b175f4a6f43dc3cfb92f1ecbe42230f7f46a01c2d67cb81d84177e30`
   - Canonical Val CSV SHA-256: `897105d6266eba01b2b7bd45e2a7eb63bca7e9107faa202b07ba82e6d866b925`
   - Retained Train CSV SHA-256: `e636e0bf9a32498167d5f4567edb30b007e6ac40c9c2fdf9b383424c4b776c4a`
   - Retained Val CSV SHA-256: `502aa4e3ca57f9be1d0a8db4839912053cc1cfaf45fb2be1bd2c4b93930ac3f5`
4. **Defensive Metric Extraction**:
   - Handled scikit-learn `classification_report` output defensively with `macro_precision`, `macro_recall`, and `per_class_f1` cleanly parsed.
5. **Threading Isolation**:
   - `cv2.setNumThreads(0)` and `cv2.ocl.setUseOpenCL(False)` successfully eliminated OpenMP thread explosion.

---

## 5. Artifacts & File Registry

- **Modal Training Wrapper**: `scripts/modal_train_cvb_beef_behavior_tcn.py`
- **Core Training Script**: `scripts/train_cvb_beef_behavior_tcn.py`
- **Perception Cache Engine**: `scripts/build_behavior_perception_cache.py`
- **Persistent Volume Checkpoints**:
  - `behavior-checkpoints:/checkpoints/behavior_run5_perception/behavior_tcn_best.pth`
  - `behavior-checkpoints:/checkpoints/behavior_run5_perception/behavior_tcn_latest.pth`
  - `behavior-checkpoints:/checkpoints/behavior_run5_perception/behavior_tcn_metrics.json`

---

## 6. Next Steps

1. **Execute Strict Final Test Evaluation Gate**:
   - Run `modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior_tcn.py::evaluate_test_run5`
   - Evaluates the best Run 5 checkpoint on the unseen test set, generates test SAM 2.1 masks, and evaluates the historical Run 2 checkpoint on the matched test subset.
2. **Launch ScienceDB BCS Run 4 Full Training**:
   - As soon as `pack_cache` completes on `tigerwood697`, launch full 30-epoch BCS training on L40S.
