# Phase 3 Run 6: SideViewCows2026 GT-Mask Perception-Enhanced Re-ID Modal Cloud Wrapper Preparation

**Date:** 2026-09-24  
**Status:** MODAL CLOUD WRAPPER PREPARED AND VERIFIED; FULL TRAINING AND PROTOCOL A EVALUATION PENDING MANUAL LAUNCH  
**Target Profile:** `dryousufmozumder`  
**Dataset Volume:** `sideview-data` (mounted at `/data`; dataset root `/data/sideviewcows2026`)  
**Checkpoint Volume:** `reid-checkpoints` (mounted at `/checkpoints`)  
**Core Trainer:** `scripts/train_sideview_reid_perception.py`  
**Modal Wrapper:** `scripts/modal_train_sideview_reid_perception.py`  

---

## 1. Executive Summary

The Modal cloud training and evaluation wrapper for **Phase 3 Run 6 (SideViewCows2026 GT/Oracle Segmentation-Guided Perception-Enhanced Re-ID)** has been designed, implemented, and locally verified in `scripts/modal_train_sideview_reid_perception.py`. The wrapper targets the verified Modal workspace `dryousufmozumder`, attaching the existing certified dataset volume `sideview-data` (80,260 RGB images, 80,260 masks, 110 biological cows) and persistent checkpoint volume `reid-checkpoints`. 

The wrapper strictly exposes three separate entrypoints: (1) `verify_readiness` (cheap, non-training dataset and protocol integrity check), (2) `smoke_test` (cheap 2-epoch verification on low-cost T4 GPU in `smoke=True` mode, train/val only, zero Protocol A evaluation), and (3) `main` (prepared 30-epoch full launch on NVIDIA L40S, deferred for user manual execution). Neither full training nor Protocol A final evaluation was launched.

---

## 2. Context & Scientific Condition

### 2.1 Scientific Representation (Inviolable Specification)
Run 6 represents the official thesis deadline **GT/oracle segmentation-guided Re-ID condition**:
- **Source**: SideViewCows2026 official target-cow ground-truth binary masks (Zenodo record 21605650).
- **Bounding Box Derivation**: Target cow bounding box derived directly from the positive mask pixels.
- **Margin**: Deterministic 5% proportional margin expansion (`cx1, cy1, cx2, cy2`).
- **Synchronized Crop & Resize**: Exact identical crop geometry applied to both RGB and mask, resized to 224x224.
- **Channel Concatenation**: ImageNet-normalized RGB (`float32`, 3 channels) concatenated with unnormalized binary mask (`{0.0, 1.0}`, 1 channel) into `[R, G, B, Mask]` (`[B, 4, 224, 224]`).
- **No RGB Masking**: RGB pixel values are NOT multiplied or zeroed by the mask.
- **Model Capacity**: ResNet-18 backbone with `conv1` expanded from 3 to 4 channels (`nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)`). Channels 0-2 copy ImageNet weights; channel 3 is initialized from the channel mean. Exactly 11,200,681 trainable parameters (+3,136 over Run 3 RGB baseline's 11,197,545).
- **Feature & Head**: 512-D raw feature vector -> L2-normalized unit embedding -> `Linear(512, 41)` training classification head.

**Strict Scientific Boundary**: This pipeline is an oracle segmentation-guided representation experiment. It is NOT an automated SAM deployment pipeline, pose, viewpoint, or temporal model.

### 2.2 Canonical Protocols & Evaluation Isolation
- **Representation Learning**: 41 canonical cows (`setting_role == "train"` in `protocol_cross_setting.csv`).
- **Protocol D Closed-Set Training Partition**: 12,753 images across the 41 train cows.
- **Protocol D Closed-Set Validation Partition**: 2,683 images across the same 41 train cows.
- **Protocol A Held-Out Evaluation**: 69 completely held-out biological cows (`setting_role != "train"`), evaluated across 36,811 Parlor gallery images, 25,260 Barn query images, and 607 Handheld Snapshot query images.
- **Strict Isolation Rule**: The 69 held-out evaluation cows must NEVER be loaded, sampled, or accessed during training, validation, or checkpoint selection.

---

## 3. Modal Cloud Configuration & Volumes

### 3.1 Workspace Separation
- **Target Profile**: `dryousufmozumder`.
- **Dataset Volume**: `sideview-data` at `/data` (already hydrated and forensically certified: 80,260 images, 80,260 masks, 110 cow identities).
- **Checkpoint Volume**: `reid-checkpoints` at `/checkpoints` (created fresh on `dryousufmozumder` via `modal.Volume.from_name("reid-checkpoints", create_if_missing=True)`).
- **Run 6 Directory**: `/checkpoints/sideview_reid_perception_run6` (completely isolated from Run 3 baseline).
- **Smoke Directory**: `/checkpoints/sideview_reid_perception_smoke`.

### 3.2 Container Image Specification
```python
reid_perception_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy",
        "pandas",
        "pillow",
        "scikit-learn",
        "tqdm",
    )
    .add_local_dir(
        str(REPO_ROOT / "datasets" / "id" / "sideviewcows2026"),
        remote_path="/root/datasets/id/sideviewcows2026",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_baseline.py"),
        remote_path="/root/scripts/train_sideview_reid_baseline.py",
    )
    .add_local_file(
        str(REPO_ROOT / "scripts" / "train_sideview_reid_perception.py"),
        remote_path="/root/scripts/train_sideview_reid_perception.py",
    )
)
```

---

## 4. Entrypoint Specifications

### 4.1 Entrypoint 1: `verify_readiness`
- **Cost**: Cheap (~$0.01 on T4, 300s timeout).
- **Functionality**:
  1. CUDA availability and GPU properties check.
  2. Dataset root presence on `/data/sideviewcows2026`.
  3. Exact image counts: Parlor (54,393), Barn (25,260), Snapshots (607) -> 80,260 total.
  4. Exact mask counts: Parlor (54,393), Barn (25,260), Snapshots (607) -> 80,260 total.
  5. 110 unique cow identities.
  6. Canonical protocol resolution: 41 train cows, 69 held-out cows, 0 overlap. Protocol D train = 12,753, val = 2,683.
  7. Checkpoint volume write/delete test on `/checkpoints/sideview_reid_perception_run6`.
  8. Trainer script (`scripts.train_sideview_reid_perception`) import check in remote container.
  9. Sample RGB-mask pair decoding and crop box derivation across Parlor, Barn, and Snapshots.

### 4.2 Entrypoint 2: `smoke_test`
- **Cost**: Low-cost (~$0.03 on T4, 600s timeout).
- **Functionality**:
  1. Executes `train_sideview_reid_perception(...)` in `smoke=True` mode with `smoke_samples=64` on T4 GPU.
  2. Asserts authentic paired RGB/mask inputs with 4-channel tensor `[B, 4, 224, 224]`.
  3. Asserts nearest-neighbor binary mask values `{0.0, 1.0}`.
  4. Asserts 512-D features, unit L2-norm embeddings, and 41-class logits.
  5. Runs 2 epochs with checkpoint save at Epoch 1 and resume from checkpoint at Epoch 2.
  6. Asserts bit-identical checkpoint reload: `max_logit_difference == 0.0`.
  7. Asserts Protocol A held-out gallery/query images loaded = 0 (`status == "NOT_LOADED_OR_EVALUATED_IN_SMOKE_MODE"`).

### 4.3 Entrypoint 3: `main`
- **Execution State**: Prepared and frozen. **NOT executed.**
- **Default Arguments**: `epochs=30`, `batch_size=64`, `lr=1e-4`, `workers=8`, `seed=2026`, `gpu="L40S"`.
- **Checkpointing**: Saves `reid_perception_latest.pth` and `reid_perception_best.pth` to `/checkpoints/sideview_reid_perception_run6`.
- **Post-Training Evaluation**: Protocol A retrieval (Barn query and Snapshot query against Parlor gallery) will execute only upon completion of the 30 training epochs.

---

## 5. Local Verification Results

1. **Python Compilation**: `py_compile.compile('scripts/modal_train_sideview_reid_perception.py')` passed with exit code 0.
2. **Local Import Verification**: Trainer function `train_sideview_reid_perception` imported cleanly with exit code 0.
3. **AST Structure Verification**: Confirmed presence of `verify_readiness_remote`, `smoke_test_remote`, `train_full_remote`, `verify_readiness`, `smoke_test`, and `main`.
4. **Volume Target Verification**: Verified `sideview-data` at `/data` and `reid-checkpoints` at `/checkpoints`.

---

## 6. Execution Commands for User

All cloud executions are deferred to the user:

```powershell
# 1. Readiness Verification (No-training dataset and protocol audit)
modal run --profile dryousufmozumder scripts/modal_train_sideview_reid_perception.py::verify_readiness

# 2. Cheap T4 Cloud Smoke Test (2 epochs, train/val only, held-out untouched)
modal run --profile dryousufmozumder scripts/modal_train_sideview_reid_perception.py::smoke_test
```
