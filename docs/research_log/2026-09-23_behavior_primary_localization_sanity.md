# Research Log: Primary Behavior Stack RT-DETR-L Localization Sanity Check

**Date:** 2026-09-23  
**Status:** COMPLETED / VERIFIED  
**Task:** Small RT-DETR-L cattle localization sanity check directly on the new primary Behavior stack (CVB + Kaggle Beef).  
**Execution Environment:** Modal cloud (profile `tigerwood693`, GPU Tier `T4`, CPU 2.0, RAM 4096MB).  
**Git Commit:** `9f1142dc65d427cb7964068d435ca2d3ebf9f36c`  
**Deterministic Seed:** 2026  
**Artifacts Generated:**
- `scripts/audit_behavior_primary_localization.py`
- `artifacts/perception_audit/behavior_primary_localization_sanity.csv`
- `docs/audits/assets/behavior_primary_localization_sanity/cvb_localization_sanity_contact_sheet.jpg`
- `docs/audits/assets/behavior_primary_localization_sanity/beef_localization_sanity_contact_sheet.jpg`
- 45 individual visual overlay files under `docs/audits/assets/behavior_primary_localization_sanity/`

---

## 1. Executive Summary

We executed an empirical localization sanity check using pretrained RT-DETR-L (COCO class 19 `cow`, confidence threshold 0.25) directly on the newly approved primary Behavior training partition (`datasets/behavior/cvb_beef/train.csv`). Testing 45 deterministic midpoint frames across 5 canonical behavior classes (25 CVB track segments, 20 Kaggle Beef clips) revealed a decisive operational divergence:
1. **CVB (Full 1080p Pasture Video):** RT-DETR-L achieved a **100.0% target-cow overlap rate (25/25)**, **96.0% localization hit rate at IoU >= 0.50 (24/25)**, and a **mean target IoU of 0.8227 (median 0.8671)** against authentic CVB ground truth annotations (`instances_default.json`). CVB frames contain an average of 13.04 cows per frame; upstream cattle localization and tracklet association are mandatory and highly effective.
2. **Kaggle Beef (224x224 Pre-Cropped Clips):** RT-DETR-L achieved a **95.0% raw detection rate (19/20)**, with 1 failure on a recumbent cow heavily occluded by stall pipes (`beef_00000000580000000_2_clip_3`, 0 detections). However, because Kaggle Beef clips are *already single-cow crops*, RT-DETR-L frequently produced multiple fragmented detections per frame (mean 4.35 boxes per 224x224 image, capturing cow heads, rumps, and neighboring stall cattle).
3. **Operational Decision:** Upstream RT-DETR-L localization is **operationally validated and mandatory for CVB**, but **unnecessary and counterproductive for Kaggle Beef**, which should be ingested directly as pre-cropped cow clips.

---

## 2. Experimental Setup & Sample Design

### Sampling Protocol (Deterministic Seed 2026)
Samples were selected strictly from `datasets/behavior/cvb_beef/train.csv` (val/test sets 100% untouched):
- **CVB (25 samples):** Exactly 5 samples per canonical class (`Standing`, `Lying`, `Feeding`, `Drinking`, `Walking`). Maximized source diversity to **25 unique source videos** (100% video-disjoint across sampled items). Representative midpoint frames were calculated deterministically as `(start_frame + end_frame) // 2`. Target ground-truth bounding boxes were extracted from authentic CVB COCO annotations (`instances_default.json`) for the exact track ID.
- **Kaggle Beef (20 samples):** Exactly 5 samples per canonical class (`Standing`, `Lying`, `Feeding`, `Drinking`). Maximized session diversity to **19 unique recording sessions** across the 20 samples (all 4 unique Lying sessions present in training were sampled). Midpoint frames were extracted at `n_frames // 2 = 125` from continuous 25.0 FPS MP4 clips.

### Physical Volumes on Modal
- CVB Volume: `cvb-data` mounted at `/data/cvb/000058916v001`
- Kaggle Beef Volume: `beef-behavior-data` mounted at `/data/beef_behavior`
- Model: `RTDETR("rtdetr-l.pt")` via `ultralytics` on NVIDIA Tesla T4 GPU. Average inference latency: **62.5 ms/frame** on 1080p CVB frames; **57.1 ms/frame** on 224x224 Beef crops.

---

## 3. Quantitative Results

### 3.1 CVB Target Localization (Ground Truth Available: 25/25 = 100%)

| Canonical Behavior | Sample Count | Unique Source Videos | Target Overlap (IoU > 0.0) | Hit Rate (IoU >= 0.50) | Mean Target IoU | Median Target IoU | Min IoU | Max IoU | Mean Cattle Dets / Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Feeding** | 5 | 5 | 5 / 5 (100%) | 5 / 5 (100%) | **0.9105** | 0.9602 | 0.7861 | 0.9749 | 11.0 |
| **Lying** | 5 | 5 | 5 / 5 (100%) | 5 / 5 (100%) | **0.9031** | 0.9252 | 0.8062 | 0.9604 | 15.0 |
| **Standing** | 5 | 5 | 5 / 5 (100%) | 5 / 5 (100%) | **0.8020** | 0.8427 | 0.5881 | 0.9806 | 12.8 |
| **Drinking** | 5 | 5 | 5 / 5 (100%) | 5 / 5 (100%) | **0.7998** | 0.7671 | 0.7042 | 0.9419 | 10.4 |
| **Walking** | 5 | 5 | 5 / 5 (100%) | 4 / 5 (80%) | **0.6980** | 0.7180 | 0.4404 | 0.9234 | 16.0 |
| **OVERALL CVB** | **25** | **25** | **25 / 25 (100%)** | **24 / 25 (96.0%)** | **0.8227** | **0.8671** | **0.4404** | **0.9806** | **13.04** |

### 3.2 Kaggle Beef Detection (Pre-Cropped 224x224 Single-Cow Clips)

| Canonical Behavior | Sample Count | Unique Sessions | Detection Rate (>=1 cow) | No-Detection Count | Mean Max Box Area Ratio | Min Area Ratio | Max Area Ratio | Mean Detections / Crop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drinking** | 5 | 5 | 5 / 5 (100%) | 0 / 5 (0%) | 0.8210 | 0.7300 | 0.8906 | 3.8 |
| **Feeding** | 5 | 5 | 5 / 5 (100%) | 0 / 5 (0%) | 0.7595 | 0.7314 | 0.8624 | 7.4 |
| **Standing** | 5 | 5 | 5 / 5 (100%) | 0 / 5 (0%) | 0.7352 | 0.6300 | 0.8580 | 4.2 |
| **Lying** | 5 | 4 | 4 / 5 (80%) | 1 / 5 (20%) | 0.3557 | 0.0000 | 0.5982 | 2.6 |
| **OVERALL BEEF** | **20** | **19** | **19 / 20 (95.0%)** | **1 / 20 (5.0%)** | **0.6678** | **0.0000** | **0.8906** | **4.35** |

*(Note: Box area ratio represents the fraction of the 224x224 crop covered by the largest detected bounding box; it is an area coverage index, not a localization accuracy metric, as Kaggle Beef lacks target bounding box annotations in this audit).*

---

## 4. Qualitative Failure Modes & Edge Case Analysis

### CVB Edge Cases
1. **Sub-0.50 IoU Case (`cvb_0400..._tr9_seg0`, Walking, IoU = 0.4404):**
   - In frame 203 of cut `0400_arm01_gopro1_20200324_005347_beh7_ani2_ins1_cut_3`, the target walking cow (Track 9) is immediately adjacent to another black cow eating at the water trough. Because both animals have dark coats and overlap physically in 2D projection, the RT-DETR-L bounding box encompasses parts of both cows, resulting in a lateral box shift.
   - Despite the shift, overlap remains substantial (`overlap = True`, confidence 0.84), and 15 total cows in the wide pasture were cleanly detected without false positives on pasture background.
2. **Distant Small Cattle:**
   - In cuts recorded from `gopro4` (e.g., `cvb_1333..._tr9_seg0`, Walking, IoU = 0.5826), cattle in the far background appear at resolutions under 65x45 px. RT-DETR-L successfully detected them with confidence >= 0.78, demonstrating high sensitivity on small pasture targets.

### Kaggle Beef Edge Cases
1. **Zero-Detection Failure (`beef_00000000580000000_2_clip_3`, Lying, Dets = 0):**
   - The cow is in sternal recumbency viewed through heavy, intersecting horizontal and vertical metal stanchion pipes. The bars break up the cow's contour, and the animal fills 100% of the frame without visible pasture background. Zero-shot COCO RT-DETR-L produced 0 detections.
2. **Partial Head Detection (`beef_00000000109000000_4_clip_105`, Lying, Area Ratio = 0.093):**
   - The detector triggered only on the cow's white face and ear tag (confidence 0.36), failing to identify the occluded brown body behind the stall dividers.
3. **Severe Over-Detection / Fragmentation:**
   - On Feeding clips (mean 7.4 detections per 224x224 image), RT-DETR-L frequently placed 3-5 overlapping bounding boxes on different segments of the same cow (e.g., head, neck, dorsal line) and placed extra boxes on neighboring cows visible through pen partitions.

---

## 5. Architectural & Pipeline Recommendations

1. **CVB Ingestion Strategy:**
   - Upstream RT-DETR-L localization is **operationally verified and essential**. Because CVB raw frames are 1080p wide-angle recordings with multiple grazing animals, extracting crops centered on the target cow tracklet is required.
   - For Step 3 caching, the project can safely run RT-DETR-L on CVB, using the authentic tracklet annotations to guide target association.
2. **Kaggle Beef Ingestion Strategy:**
   - Upstream RT-DETR-L localization is **not necessary and should be bypassed**. Kaggle Beef clips are already single-cow centered crops. Applying an object detector risks dropped frames on occluded lying cattle and spurious box jitter.
   - Kaggle Beef clips should feed directly into the downstream feature extractor (or SAM 2.1 mask refinement if soft foreground masking is applied).
3. **Readiness for Next Step:**
   - The evidence strongly supports proceeding to the **SAM 2.1 cattle segmentation sanity check** on the primary Behavior stack.
   - For CVB: Test RT-DETR-L box -> SAM 2.1 soft mask.
   - For Kaggle Beef: Test direct whole-crop prompt -> SAM 2.1 soft mask to assess whether background stall bars can be separated from the cow without an upstream detector box.

---

## 6. Verification Asset Index

| Asset Type | File Path | Dimensions / Rows | Key Features |
| :--- | :--- | :--- | :--- |
| **Audit CSV** | `artifacts/perception_audit/behavior_primary_localization_sanity.csv` | 45 rows, 18 columns | Full per-frame metrics, IoUs, detection counts, inference times |
| **CVB Contact Sheet** | `docs/audits/assets/behavior_primary_localization_sanity/cvb_localization_sanity_contact_sheet.jpg` | 2400 x 1405 px | 5x5 grid (25 frames): Target GT (green), matched RT-DETR (cyan), other dets (orange) |
| **Beef Contact Sheet** | `docs/audits/assets/behavior_primary_localization_sanity/beef_localization_sanity_contact_sheet.jpg` | 1600 x 1335 px | 4x5 grid (20 frames): Yellow RT-DETR boxes on 224x224 crops |
| **Execution Script** | `scripts/audit_behavior_primary_localization.py` | 440 lines | Deterministic sampling + Modal T4 distributed execution |
