# Forensic Assessment: MmCows vs. CBVD-5 for Primary Behavior Recognition

**Date**: 2026-09-20  
**Auditors**: Hasin Ishrak & Antigravity  
**Status**: COMPLETE / PROPOSAL ONLY (No roadmap changes implemented)  
**Investigation Trigger**: Manual visual inspection during Step 1 identified that certain MmCows behavior crops exhibit CCTV compression artifacts and motion blur, prompting a rigorous audit of whether CBVD-5 should replace MmCows as Primary Behavior dataset.  

---

## 1. Executive Summary
A forensic audit comparing **MmCows** (213,686 crops, 16 cows) and **CBVD-5** (887 videos, 25,324 bboxes, 107 herd cattle) was conducted across 10 scientific dimensions. Although CBVD-5 raw video frames are Full HD 1080p (giving the initial visual impression of high sharpness), individual cow bounding boxes in CBVD-5 are actually **2.5x smaller in area** (median 156 x 167 px) than MmCows crops (median 390 x 370 px). Crucially, CBVD-5 contains **zero biological cow ID annotations** (actor ID is hardcoded to `1` across all annotations), rendering identity-disjoint train/test splitting physically impossible and precluding shortcut-learning evaluation. Furthermore, CBVD-5 lacks walking and grooming behaviors, uses multi-label rumination/posture combinations, and exhibits 100% video overlap between its official validation and test splits. Promoting CBVD-5 to primary would consume the project's only external behavior benchmark. We decisively recommend **Strategy A: Retain MmCows as PRIMARY and CBVD-5 as EXTERNAL VALIDATION**.

---

## 2. Head-to-Head Forensic Findings across 10 Dimensions

### 1. Visual Quality & Crop Resolution
- **MmCows**:
  - Source: 4 fixed commercial CCTV ceiling cameras (Hikvision).
  - Crop Dimensions: Mean **409 x 386 px**, Median **390 x 370 px** (Max 1152 x 936 px).
  - Visual Artifacts: Visible CCTV compression noise and motion blur on dynamic movements (e.g. walking hooves, licking tongue), exacerbated because crops are tightly zoomed. Metal stall bars create real-world partial occlusions.
- **CBVD-5**:
  - Source: Ceiling cameras recording wide pen overviews (1920 x 1080 @ 25 fps).
  - Crop Dimensions: Mean **177 x 217 px**, Median **156 x 167 px** (Min 1.9 px!).
  - Visual Artifacts: Full frames look crisp, but individual cattle occupy only ~1.5% of the total frame area. Cropping cows results in small, low-detail patches where subtle actions like rumination (jaw movement) are barely discernible without multi-frame temporal video playback.

### 2. Dataset Scale & Labeled Volume
- **MmCows**: **213,686 labeled instances** across 21.0 hours of continuous CCTV recording. Average 13,355 samples per cow.
- **CBVD-5**: **25,324 labeled bounding boxes** across 5,322 annotated frames (extracted from 887 10-second videos = 2.46 hours total video). MmCows contains **8.4x more annotated instances** and **8.5x more temporal recording hours**.

### 3. Behavior Taxonomy & Class Alignment
- **MmCows (7 Single-Label Classes)**:
  - `Walking`: 4,118 (1.93%)
  - `Standing`: 70,107 (32.81%)
  - `Feeding_head_up`: 19,080 (8.93%)
  - `Feeding_head_down`: 31,255 (14.63%)
  - `Licking`: 2,009 (0.94%)
  - `Drinking`: 3,311 (1.55%)
  - `Lying`: 83,806 (39.22%)
- **CBVD-5 (5 Multi-Label Attributes)**:
  - `stand`: 15,823
  - `lying down`: 9,506
  - `rumination`: 6,079
  - `foraging`: 5,711
  - `drinking water`: 744
- **Taxonomy Mismatch**:
  - CBVD-5 has **NO Walking** and **NO Licking**.
  - CBVD-5 merges feeding posture into generic `foraging`.
  - CBVD-5 treats `rumination` as a secondary attribute co-occurring with `lying down` (4,955) and `stand` (1,124).
  - Converting CBVD-5 into single-label classification requires arbitrary priority heuristics (e.g. is a cow standing and foraging labeled "Stand" or "Feeding"?).

### 4. Biological Identity Validity
- **MmCows**: **100% verified biological cow IDs** (Cows 1 to 16) verified via ear-tags in a controlled research barn (NeurIPS 2024 Spotlight).
- **CBVD-5**: **ZERO biological cow IDs.** All annotation CSVs (`CBVD-5.csv`, `ava_train_v2.1.csv`, `ava_val_v2.1.csv`) assign dummy actor ID = `1`. Video IDs are clip indices (`1` to `899`). It is scientifically impossible to perform cow-disjoint evaluation on CBVD-5.

### 5. Temporal Provenance
- **MmCows**: Exact timestamps (epoch and ISO), 15-second contiguous recording blocks, and indexed temporal sequences. Enables frame-sequence modeling (TCN, GRU/LSTM) on cropped bounding boxes.
- **CBVD-5**: Continuous 10-second MP4 video clips at 25 fps. However, bounding boxes are only annotated at 1.0-second intervals (1 fps). Dense frame-by-frame tracking requires running an external tracker. Furthermore, no global timestamps link video clips across the 887 files.

### 6. Camera & Viewpoint Structure
- **MmCows**: 4 calibrated CCTV cameras with overlapping fields of view; **87.15% of events are synchronized multi-camera captures**, enabling true multi-view representation learning.
- **CBVD-5**: Uncalibrated ceiling camera views without multi-view synchronization.

### 7. Leakage-Safe Split Feasibility
- **MmCows**: **Fully verified Cow-Disjoint Grouped Split** (11 Train / 2 Val / 3 Test cows) and 4-Fold GroupKFold suite. 0 cow overlap, 0 exact duplicate leakage.
- **CBVD-5**: Can only be split by `video_id`. However, because 107 cows in the same pen appear across multiple video clips, video-disjoint splitting guarantees massive biological cow leakage between train and test. Furthermore, the official AVA split contains severe flaws: **Val and Test share 100% of their 50 videos**, and 5 videos overlap with Train.

### 8. Temporal Modeling Suitability
- **MmCows**: Highly suited for temporal pooling, TCN, and GRU/LSTM over sequential bounding box feature vectors within 15s time blocks.
- **CBVD-5**: Highly suited for 3D-CNN / video action transformers (SlowFast, VideoMAE) on raw 10s video clips, but poorly suited for cattle-centric crop feature sequences due to 1 fps annotations.

### 9. Thesis Alignment (Cattle-Centered Visual Representations)
- Phase 3 evaluates whether cattle-centered priors (localization, masks, pose, viewpoint) prevent shortcut learning and improve robustness across tasks.
- In MmCows, persistent cow identities allow evaluating whether representations isolate behavior from cow identity (preventing the model from memorizing that Cow 1 likes to lie down).
- In CBVD-5, because cow identities are unknown, identity-vs-behavior shortcut analysis is impossible.

### 10. External Validation Consequences
- If CBVD-5 is promoted to Primary:
  1. **No External Behavior Benchmark Remains**: CBVD-5 is our only local large-scale external behavior dataset. Consuming it as primary destroys our external generalization test.
  2. **MmCows Cannot Function as an Effective External Test**: Evaluating a CBVD-trained model on MmCows would fail on Walking and Licking (which CBVD never learned) and could not evaluate Rumination (which MmCows never labeled).
  3. **Scientific Value Lost**: Keeping CBVD-5 as external validation allows us to test whether a model trained on MmCows (16 cows, Camera 1-4) generalizes to an unseen 107-cow commercial herd under completely different lighting and camera geometry.

---

## 3. Comparative Summary Table

| Evaluation Criterion | MmCows (Keep Primary) | CBVD-5 (Keep External) |
|---|---|---|
| **Scientific Role** | **PRIMARY IN-DOMAIN BENCHMARK** | **PRIMARY EXTERNAL BENCHMARK** |
| **Cow-Disjoint Splitting** | **YES (100% Enforceable)** | **NO (Impossible; No IDs)** |
| **Number of Labeled Crops** | **213,686** | **25,324** |
| **Median Crop Size** | **390 x 370 px** | **156 x 167 px** |
| **Behavior Classes** | **7 classes (including Walk & Lick)** | **5 classes (multi-label rumination)** |
| **Synchronized Multi-View** | **YES (4 Cameras)** | **NO (Single Views)** |
| **Shortcut Analysis** | **YES (Identity vs Behavior)** | **NO** |
| **External Generalization** | Evaluated on CBVD-5 | Serves as the independent test |

---

## 4. Final Recommendation

### **RECOMMENDATION: OPTION A — KEEP MMCOWS PRIMARY & CBVD-5 EXTERNAL**
1. **Scientific Integrity**: MmCows is the only dataset supporting leak-free cow-disjoint evaluation and shortcut analysis. Promoting CBVD-5 would introduce unquantifiable cow leakage and reduce labeled crop volume by 88%.
2. **Crop Resolution Reality**: While full CBVD-5 video frames are 1080p, individual cow bounding boxes are actually 2.5x smaller than MmCows crops.
3. **Preserving External Validation**: CBVD-5 is far more valuable as a rigorous, out-of-domain external stress test for the final Phase 3 model.

---

## 5. Artifacts & Deliverables
- **Visual Inspection Pack**: `docs/audits/phase3_behavior_dataset_manual_comparison.md`
- **Visual Asset Thumbnails**: `docs/audits/assets/behavior_dataset_comparison/` (28 review images)
- **Audit Script**: `scripts/audit_behavior_dataset_candidates.py`
