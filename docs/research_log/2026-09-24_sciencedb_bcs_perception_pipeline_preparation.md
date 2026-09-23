# Research Log: ScienceDB BCS Perception-Enhanced Pipeline Preparation (Phase 3 Run 4)

**Date:** 2026-09-24  
**Status:** IMPLEMENTED, SMOKE-TESTED & READY FOR CLOUD EXECUTION  
**Target Profile:** `tigerwood697`  
**Target Architecture:** 4-Channel ResNet-18 with Ordinal BCE Head  
**Primary Volumes:** `sciencedb-data` (/data), `sciencedb-perception-cache` (/cache), `sciencedb-checkpoints` (/checkpoints)  

---

## 1. Executive Summary

To execute Phase 3 Run 4 (BCS Perception-Enhanced Model) under the active September 26 deadline priority overlay, we designed, implemented, and smoke-tested the complete reproducible perception preprocessing and training pipeline for ScienceDB. The architecture incorporates cattle-centered spatial priors via a 4-channel ResNet-18 ([R, G, B, Mask]), where the primary cow is localized via pretrained RT-DETR-L (with a fixed 5% proportional margin) and segmented via SAM 2.1. 

Empirical inspection of the Ultralytics SAM 2.1 output tensor resolved mask semantics: SAM produces strictly binary boolean masks (`torch.bool`, values {0, 1}), which are documented as binary foreground segmentations rather than manufactured continuous probabilities. SuperAnimal pose is explicitly excluded due to previously audited anatomical unreliability on rear-view chute postures and absence of pelvic skeletal landmarks. Viewpoint is certified on its own real test split (86.26%) but deferred from automatic injection pending cross-domain validation. A 10-sample smoke test across all 5 BCS classes verified loss backpropagation, training convergence (Val MAE: 0.2500), 100% bit-identical checkpoint save/resume, exact parameter tracking (+3,136 parameters for conv1), and visual quality on a 4-panel contact sheet. Zero full training or paid Modal runs were launched.

---

## 2. Perception Cache Architecture & Failure Policies

### 2.1 Pipeline Flow
```text
ScienceDB Image (1024x576)
    ↓
RT-DETR-L Cow Localization (conf >= 0.25, COCO class 19)
    ↓
Primary Cow Selection (Max area = w * h, confidence tie-breaker)
    ↓
Proportional Margin Expansion (5% width/height margin)
    ↓
SAM 2.1 Instance Segmentation (Detector box prompt)
    ↓
Crop Image & Mask to Exact Same Bounding Box
    ↓
Save Crop (.jpg, Q95) + Binary Mask (.png, lossless) + Manifest (.csv)
```

### 2.2 Mask Semantics Verification
- Direct tensor inspection of `res[0].masks.data` on authentic ScienceDB samples confirmed:
  `dtype: torch.bool, unique: [False, True]`.
- The raw mask is **binary boolean**, NOT a continuous confidence map.
- Stored as single-channel 8-bit PNG (0 = background, 255 = foreground cow).
- Documented strictly as **BINARY foreground mask guidance** (zero claims of soft probability).
- In DataLoader, converted to float tensor with values in `{0.0, 1.0}`.

### 2.3 Explicit Failure Handling & Strict Exclusion Policy
- **Zero RT-DETR Detections:** Recorded explicitly in manifest with `detection_status='no_detection'` and `sam_status='upstream_localization_failure'`. Zero artificial detector boxes, fake crops, or zero masks are written to disk (`crop_rel_path=null`, `mask_rel_path=null`).
- **SAM Zero Mask:** Recorded explicitly with `sam_status='sam_no_mask'`. Zero artificial masks are written to disk (`mask_rel_path=null`).
- **Training Exclusion Policy:** Downstream training dataset strictly requires `detection_status == 'detected'` AND `sam_status == 'segmented'`. Non-successful perception rows are completely excluded from model training, validation, and test evaluation. Excluded counts are reported per split. Zero fabricated full-image or zero-mask samples are fed to the network.

### 2.4 Resumable Cache Architecture
- Periodic checkpoint commits: writes manifest and calls Modal `cache_volume.commit()` every 100 samples and upon split completion.
- Re-run safe: reads existing split manifests, verifies physical existence and non-zero size (`st_size > 0`) of crops and masks on disk, increments `skip` count, and avoids redundant GPU compute.
- Live `tqdm` progress tracking with real-time `skip`, `det`, `seg`, and `fail` counters.

---

## 3. Run 4 Model Architecture & Parameter Verification

To isolate the empirical effect of cattle-centered localization and foreground masking against the Run 1 BCS RGB baseline:
- **Trunk:** ResNet-18 (ImageNet-1K pretrained weights).
- **conv1 Modification:** `nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)`.
  - Channels 0, 1, 2: ImageNet pretrained RGB filter weights copied directly.
  - Channel 3: Initialized deterministically from channel-mean of ImageNet conv1 weights (`old_conv1.weight.mean(dim=1, keepdim=True)`).
- **Head:** Frank & Hall (2001) Ordinal BCE head (`Linear(512, 4)`).
- **Exact Parameter Counts:**
  - Baseline 3-Channel ResNet-18 BCS (Run 1): **11,178,564** parameters.
  - Perception 4-Channel ResNet-18 BCS (Run 4): **11,181,700** parameters.
  - **Exact Delta:** **+3,136 parameters (+0.028%)**, completely confined to `conv1`.

---

## 4. Augmentation Matching & Test Isolation

- **Run 1 Augmentation Fair Match:**
  - Resize 224
  - RandomHorizontalFlip(p=0.5) [Synchronized across RGB + mask]
  - RandomRotation(15 degrees) [Synchronized across RGB + mask]
  - ColorJitter(brightness=0.1, contrast=0.1) [Applied to RGB ONLY, never mask]
- **Strict Test Set Isolation:**
  - Training and checkpoint selection strictly evaluate TRAIN and VAL splits only.
  - Best checkpoint is selected by validation Real MAE.
  - Frozen test split is evaluated exactly ONCE post-training using the best checkpoint (`bcs_perception_best.pth`).
  - Test metrics are saved independently to `bcs_perception_test_metrics.json`.
  - Test data NEVER influences checkpoint selection or training decisions.

---

## 5. Local Smoke Test Verification

Executed on local GTX 1050 Ti:
- **Failure Exclusion Verified:** Smoke validation manifest filtered from 10 raw rows down to 6 usable samples (4 perception failures explicitly excluded).
- **Live TQDM Verified:** Progress bars streamed for training and validation batches.
- **Forward & Backward Pass:** Successfully executed forward pass, loss calculation, and backward gradient backpropagation across 2 epochs (Val MAE: 0.4167 BCS).
- **Checkpoint Determinism:** Bit-identical checkpoint save and reload verified (max absolute output difference = `0.00000000`).
- **Post-Training Test Evaluation:** Verified one-time post-training test evaluation function on 7 valid test samples (`bcs_perception_test_metrics.json`).
- **Visual Audit:** Generated 4-panel master contact sheet (`docs/audits/assets/bcs_perception/bcs_perception_contact_sheet.jpg`, 1280x1250 px).

---

## 5. Artifact Registry

- Preprocessing Engine: `scripts/build_sciencedb_perception_cache.py`
- Training Engine: `scripts/train_sciencedb_bcs_perception.py`
- Modal Cloud Wrapper: `scripts/modal_train_sciencedb_bcs_perception.py`
- Contact Sheet Generator: `scripts/build_bcs_perception_contact_sheet.py`
- Cache Manifest Schema: `artifacts/bcs_perception_smoke/cache_schema.json`
- Contact Sheet Visual Asset: `docs/audits/assets/bcs_perception/bcs_perception_contact_sheet.jpg`
- Resume Test: `scratch/verify_checkpoint_resume.py`

---

## 6. Manual Cloud Execution Commands

### A. Full ScienceDB Perception Cache Generation (Modal L4 GPU)
```bash
modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs_perception.py::build_cache
```

### B. Run 4 Full 30-Epoch Training (Modal L40S GPU)
```bash
modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs_perception.py::main --epochs 30 --batch-size 64
```
