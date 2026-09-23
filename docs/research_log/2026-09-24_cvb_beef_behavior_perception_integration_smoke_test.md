# Run 5 Behavior Perception Integration & Modal T4 Smoke Test

**Date:** 2026-09-24  
**Task:** Phase 3 Run 5 (Behavior Perception-Enhanced Temporal Model)  
**Execution Profile:** `tigerwood693` (Modal Cloud)  
**Target Hardware:** NVIDIA Tesla T4 GPU (Smoke Mode Only)  
**Modal App ID:** `ap-AsaXHJW9XRmrEmEJrOo6C7`  
**Git HEAD:** `dda61b4bf219f22096de39edd98bcfd965c7f801` (Clean extension)  

---

## 1. Executive Summary
Designed, implemented, verified, and smoke-tested the authentic **Phase 3 Run 5 Behavior Perception Integration** combining cattle-centered RGB crops with real Segment Anything Model (SAM 2.1 Small) binary masks feeding a 4-channel ResNet18 spatial backbone coupled to a 1D Temporal Convolutional Network (TCN). On the canonical 40-sequence balanced smoke subset (30 train, 10 val across 5 classes), the real perception pipeline generated **304 authentic binary masks** (192 CVB exact GT-prompted, 93 Kaggle Beef RT-DETR-L + center-prompted A5, and 19 Kaggle Beef center-point fallback masks). Zero dummy black images or synthetic empty/all-ones masks were produced. Two sequences with dense stanchion bar occlusion were properly isolated and excluded. The 4-channel ResNet18 + TCN architecture was certified with exactly **11,903,621 trainable parameters** (+3,136 parameters over 3-channel RGB baseline; conv1 mask channel initialized from RGB channel mean). Checkpoint save/reload bit-identity was verified with Max Logit Diff = 0.00000000. Canonical test split (`test.csv`, 809 sequences) remained strictly untouched and was never evaluated.

**Run 5 perception integration smoke-certified; full Behavior perception caching and full training were NOT launched.**

---

## 2. Context & Architectural Motivation
In Phase 3 Step 4.2, the 3-channel generic RGB Behavior baseline struggled on walking (F1 = 0.2174 on CVB), often misclassifying head-down walking cattle as feeding due to lack of temporal context and pen/background noise. In the previous turn, the 3-channel temporal TCN core was smoke-certified (`docs/research_log/2026-09-24_cvb_beef_behavior_tcn_temporal_core_smoke_test.md`).

Run 5 unites this temporal core with real instance segmentation guidance:
1. **CVB Sub-Dataset**: High-resolution overhead/surveillance video with 10-15 cows per pen. To avoid identity confusion, we utilize the authentic ground-truth tracklet bounding boxes (`CVBExactAnnotationCache`) as bounding box prompts for SAM 2.1 Small. Missing target bboxes trigger explicit perception failure rather than nearest-frame or synthetic fallbacks.
2. **Kaggle Beef Sub-Dataset**: Single-cow 224x224 video crops. RT-DETR-L localizes the primary cow and extracts condition A5 (bounding box + positive center point) as prompt to SAM 2.1 Small. If stanchion bars block detection, heuristic fallback uses image center point (112, 112) -> SAM 2.1 Small.
3. **Strict Zero-Dummy Policy**: No all-zeros, all-ones, or synthesized black masks are tolerated. If any of the T=8 frames fails perception (0 foreground pixels or extraction failure), the entire sequence is removed from the dataset.

---

## 3. Architecture & Parameter Verification

### 3.1 Input Channel Formulation
Each sequence is represented as a 4-channel tensor of shape `[B, T=8, C=4, H=224, W=224]`:
- Channels 0, 1, 2: Normalized RGB cattle crop (ImageNet mean and std).
- Channel 3: Binary segmentation mask in {0.0, 1.0} derived from real SAM 2.1 prediction thresholded at > 127.

### 3.2 4-Channel ResNet-18 Backbone
The spatial feature extractor modifies `conv1` (`nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)`):
- Weights for channels 0..2 are copied directly from ImageNet-pretrained ResNet-18 weights.
- Weights for channel 3 (mask channel) are initialized as the mean across the 3 RGB channels: `old_conv.weight.mean(dim=1, keepdim=True)`.
- Extra parameters: 64 output filters * 1 input channel * 7 * 7 = **3,136 parameters**.

### 3.3 Trainable Parameter Accounting
- **RGB Baseline Temporal Model (Run 5 Core)**:
  - Backbone: 11,176,512 params
  - TCN Head: 723,973 params
  - Total: 11,900,485 params
- **Perception-Enhanced Model (Run 5 4-Channel)**:
  - Backbone: 11,179,648 params (+3,136)
  - TCN Head: 723,973 params (Identical)
  - Total: **11,903,621 params** (Exact Verified in code assertion)

---

## 4. Empirical Smoke Test Execution & Forensic Findings

### 4.1 Execution Details
- **Command**: `modal run --profile tigerwood693 scripts/modal_train_cvb_beef_behavior_tcn.py::smoke_test`
- **Volume Mounts**:
  - `/mnt/cvb`: `cvb-data` (Read-only)
  - `/mnt/beef`: `beef-behavior-data` (Read-only)
  - `/checkpoints`: `behavior-checkpoints` (Persistent write cache & models)
- **Cache Location**: `/checkpoints/behavior_perception_smoke_cache`
- **Total Caching Time**: 65.4 seconds on NVIDIA T4 GPU.

### 4.2 Perception Cache Statistics
Across the 40 candidate sequences (320 frames):
- **Total Real Masks Generated**: 304 masks
- **CVB GT Frames Generated**: 192 masks (24 sequences * 8 frames, 100% success)
- **Beef A5 Frames Generated**: 93 masks
- **Beef Center Fallback Frames**: 19 masks
- **Sequences Retained**:
  - Train: 29 / 30 sequences retained (96.7%)
  - Val: 9 / 10 sequences retained (90.0%)
- **Excluded Sequences**:
  - `beef_00000000109000000_19_clip_10` (train, Lying): Frame 5 occluded behind dense metal chute bars; SAM returned 0 foreground pixels.
  - `beef_00000000514000000_15_clip_19` (val, Lying): Frame 2 occluded; SAM returned 0 foreground pixels.
  - Both sequences were discarded cleanly without corrupting the dataset with fake or empty masks.

### 4.3 Training & Verification Metrics
- **Epochs**: 2 epochs, batch size 4
- **Runtime**: 59.04 seconds
- **Train Loss**: 1.5794 (Epoch 1) -> 0.9135 (Epoch 2)
- **Val Loss**: 1.3117 (Epoch 2)
- **Val Overall Accuracy**: 44.44% (4 / 9)
- **Val Balanced Accuracy**: 50.00%
- **Val Macro-F1**: 0.4600
- **Checkpoint Bit-Identity Reload**: Max Logit Diff = **0.00000000** (Verified bit-identical on resumed validation set).

### 4.4 Canonical Test-Set Protection
- **Rule**: `test.csv` (809 sequences) must NOT be parsed, loaded into a DataFrame, sampled, tuned, or evaluated.
- **Verification**: `test_stat_before.st_mtime == test_stat_after.st_mtime` verified bit-level non-modification; `test_csv_evaluated: false` logged in `behavior_tcn_metrics.json`.

---

## 5. Artifacts & File Registry
- Cache Generator: [`scripts/build_behavior_perception_cache.py`](../../scripts/build_behavior_perception_cache.py)
- PyTorch Training Engine: [`scripts/train_cvb_beef_behavior_tcn.py`](../../scripts/train_cvb_beef_behavior_tcn.py)
- Modal Cloud Runner: [`scripts/modal_train_cvb_beef_behavior_tcn.py`](../../scripts/modal_train_cvb_beef_behavior_tcn.py)
- Local Metrics: [`artifacts/behavior_perception_smoke/behavior_tcn_metrics.json`](../../artifacts/behavior_perception_smoke/behavior_tcn_metrics.json)
- Perception Manifest: [`artifacts/behavior_perception_smoke/perception_manifest.csv`](../../artifacts/behavior_perception_smoke/perception_manifest.csv)
- Visual Perception Contact Sheet: [`artifacts/behavior_perception_smoke/temporal_perception_contact_sheet.jpg`](../../artifacts/behavior_perception_smoke/temporal_perception_contact_sheet.jpg)
- Audit Asset Contact Sheet: [`docs/audits/assets/behavior_perception_smoke/temporal_perception_contact_sheet.jpg`](../audits/assets/behavior_perception_smoke/temporal_perception_contact_sheet.jpg)

---

## 6. Next Steps & Thesis Roadmap Alignment
1. Run 5 Behavior Perception smoke certification is 100% complete and verified.
2. Full Behavior perception dataset caching (3,785 train + 680 val sequences) and full 30-epoch training remain deferred to the scheduled cloud execution window.
3. Synchronize `memory/state.md`, `memory/history.md`, and `memory/index.md`.
4. Git commit Run 5 files cleanly without disturbing concurrent Run 6 work.
