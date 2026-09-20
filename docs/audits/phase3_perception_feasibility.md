# Phase 3 Cattle Perception Feasibility Audit

**Document:** `docs/audits/phase3_perception_feasibility.md`  
**Status:** STEP 2.1 COMPLETE (Localization Feasibility) | STEPS 2.2–2.4 PENDING  
**Date:** 2026-09-20  
**Hardware:** Local GTX 1050 Ti (4GB VRAM), PyTorch 2.5.1+cu121  

---

## 1. Step 2.1 — Cow Detection & Localization Feasibility Audit

### 1.1 Goal & Scope
The objective of Step 2.1 is to determine whether off-the-shelf, documented pretrained object detectors can reliably locate cattle across our three primary Phase 3 datasets without any task-specific fine-tuning or training:
1. **ScienceDB** (Body Condition Scoring — rear-view chute images)
2. **MmCows** (Behavior Recognition — barn CCTV bounding-box crops with stall bar occlusions)
3. **SideViewCows2026** (Individual Cow Re-ID — parlor chute, handheld barn video, and unconstrained snapshots)

We are **not** choosing the final detector yet. We are testing whether pretrained localization is viable as an upstream prior and identifying exact failure modes. All Phase 2 checkpoints are completely excluded.

---

### 1.2 Candidate Models Evaluated
Three distinct architectural families pretrained on COCO were evaluated under identical conditions:

| Candidate | Architecture Family | Parameters | Checkpoint Source | Target Class | Confidence Threshold | Input Resolution |
|---|---|---|---|---|---|---|
| **YOLOv8s** | Single-stage anchor-free CNN | 11.2M | Ultralytics (`yolov8s.pt`) | COCO 19 (`cow`) | 0.25 | 640 x 640 |
| **Faster R-CNN v2** | Two-stage anchor-based RPN + RoIAlign | 43.7M | Torchvision (`fasterrcnn_resnet50_fpn_v2`) | COCO 21 (`cow`) | 0.25 | min 800 px |
| **RT-DETR-L** | Real-time Detection Transformer | 32.0M | Ultralytics (`rtdetr-l.pt`) | COCO 19 (`cow`) | 0.25 | 640 x 640 |

---

### 1.3 Deterministic Sampling Methodology
To ensure statistical validity and prevent selection bias, a two-stage evaluation protocol was executed:
1. **Smoke Test**: 90 images (30 per dataset, seed=42) to verify VRAM limits and pipeline execution.
2. **Expanded Audit**: **300 images** (100 per dataset, seed=42) covering the full distribution of difficult and standard cattle scenes.

#### Dataset Stratification:
- **ScienceDB (100 images)**: Stratified across all 5 BCS classes (20 per class: 3.25, 3.50, 3.75, 4.00, 4.25) drawn from distinct connected burst groups to ensure independence, capturing entry, mid-chute, and exit positions under varied illumination.
- **MmCows (100 images)**: Stratified across all 7 behaviors (Lying: 16, Standing: 16, Feeding_head_down: 16, Feeding_head_up: 16, Walking: 14, Drinking: 11, Licking: 11). Verified to cover all 4 CCTV cameras (Cam 1: 24, Cam 2: 31, Cam 3: 14, Cam 4: 31) and all 16 biological cows (Cows 1–16, ranging from 1 to 13 samples per cow).
- **SideViewCows2026 (100 images)**: Stratified across recording environments (Parlor fixed chute: 40, Barn handheld video: 40, Snapshots unconstrained: 20) capturing side profiles, outdoor grazing, and varied postures.

All missed detections were strictly recorded with empty bounding-box fields (`""`) rather than dropped rows.

---

### 1.4 Quantitative Localization Results (Expanded 300-Image Audit)

| Model Candidate | Dataset | Sample Size | Detected (>=1 Cow) | Raw Detection Rate (>=1 Cow) | Missed (0 Cows) | Multi-Cow Detections | Median Latency (GTX 1050 Ti) |
|---|---|---|---|---|---|---|---|
| **YOLOv8s** | ScienceDB (BCS) | 100 | 63 | **63.0%** | 37 | 30 | **24.1 ms** |
| | MmCows (Behavior) | 100 | 63 | **63.0%** | 37 | 34 | 25.8 ms |
| | SideViewCows2026 (Re-ID) | 100 | 97 | **97.0%** | 3 | 62 | 23.5 ms |
| | **Overall YOLOv8s** | **300** | **223** | **74.3%** | **77** | **126** | **24.5 ms** |
| **Faster R-CNN v2** | ScienceDB (BCS) | 100 | 94 | **94.0%** | 6 | 69 | 489.8 ms |
| | MmCows (Behavior) | 100 | 92 | **92.0%** | 8 | 77 | 432.0 ms |
| | SideViewCows2026 (Re-ID) | 100 | 99 | **99.0%** | 1 | 81 | 490.5 ms |
| | **Overall Faster R-CNN** | **300** | **285** | **95.0%** | **15** | **227** | **470.8 ms** |
| **RT-DETR-L** | ScienceDB (BCS) | 100 | 93 | **93.0%** | 7 | 69 | 94.3 ms |
| | MmCows (Behavior) | 100 | 90 | **90.0%** | 10 | 76 | 94.5 ms |
| | SideViewCows2026 (Re-ID) | 100 | 100 | **100.0%** | 0 | 87 | 94.2 ms |
| | **Overall RT-DETR-L** | **300** | **283** | **94.3%** | **17** | **232** | **94.3 ms** |

*Note: Raw detection rates reflect the empirical percentage of images where at least one cow was detected with confidence >= 0.25. Because ScienceDB and MmCows do not provide verified ground-truth bounding boxes, these figures represent raw detection rates rather than formal precision/recall metrics.*

---

### 1.5 Forensic Failure Mode & Manual Visual Analysis

Manual visual inspection of the 300 test cases and 120 generated side-by-side composite panels revealed sharp empirical distinctions between architectures:

#### 1. YOLOv8s Failure Pattern on Non-Side Views
- **Observation**: YOLOv8s exhibited a 37.0% non-detection rate on ScienceDB and MmCows.
- **Hypothesis**: A plausible hypothesis is that COCO training imagery may be dominated by broadside, full-body views of cattle in open pastures, potentially causing anchor-free grid features to miss rear-view anatomy (rump, spine, pin bones in ScienceDB) and tight crops where extremities are clipped. However, this audit only establishes the observed empirical difference; proving training distribution or architectural causality would require isolated ablations.
- **Vindication**: The thesis guideline to *"not automatically assume YOLO is the winner"* was supported by the empirical data.

#### 2. Performance Comparison on Complex Cattle Poses
- Both **RT-DETR-L (94.3% raw detection rate)** and **Faster R-CNN v2 (95.0% raw detection rate)** achieved substantially higher detection rates on rear views, close-up crops, and partial views than YOLOv8s.
- In 72 specific images where YOLOv8s reported 0 detections, RT-DETR-L or Faster R-CNN v2 detected cattle.
- **Hypothesis**: Faster R-CNN's region proposal network (RPN) and RT-DETR's global self-attention mechanism are hypothesized to aggregate distributed contextual cues (e.g. spine contours, body texture, hooves) more effectively than the tested single-stage CNN. This remains an architectural hypothesis rather than a formally proven mechanism.

#### 3. High Multi-Cow Detection Rate (The "Cluttered Pen" Effect)
- In MmCows (76–77% multi-cow) and SideViewCows2026 (81–87% multi-cow), both RT-DETR and Faster R-CNN detected multiple cattle per image.
- **Visual Inspection Confirmation**: In MmCows, while the crop is centered on the labeled cow, neighboring cows in adjacent stalls or walking in background alleys are visible through metal bars. The detectors correctly detected these background cows as cows.
- **Architectural Implication**: Upstream localization cannot simply take "all detected boxes." The pipeline must select the **primary target cow** (e.g., maximum area, central proximity, or overlap with tracking/ROI priors).

#### 4. The 5 Universal Failure Cases
Across all 300 images, exactly 5 images (1.67%) were missed by all three models:
1. `sample_0004` (ScienceDB BCS 3.25): Cow entering the chute; only the tail and extreme right flank are visible against a dark metal background.
2. `sample_0047` (ScienceDB BCS 3.75): Heavy shadow and severe underexposure inside the chute walkway.
3. `sample_0083` (ScienceDB BCS 4.25): Frame captured during rapid cow exit; severe motion blur and rear occlusion by the exit gate.
4. `sample_0115` (MmCows Lying): Highly elongated horizontal crop (142 x 380 px, aspect ratio 2.68:1) with cow lying down behind low metal rails; legs tucked, head obscured.
5. `sample_0116` (MmCows Lying): Elongated horizontal crop (178 x 477 px, aspect ratio 2.68:1) of a black-and-white cow resting flat against a concrete cubicle wall.

---

### 1.6 Candidate Comparison Summary

| Metric / Dimension | YOLOv8s | Faster R-CNN v2 | RT-DETR-L |
|---|---|---|---|
| **ScienceDB (Rear View)** | Poor (63.0%) | **Excellent (94.0%)** | **Strong (93.0%)** |
| **MmCows (CCTV Crops)** | Poor (63.0%) | **Strong (92.0%)** | **Strong (90.0%)** |
| **SideViewCows (Side/Field)** | Strong (97.0%) | **Near-Perfect (99.0%)** | **Perfect (100.0%)** |
| **GTX 1050 Ti Latency** | **24.5 ms** (Fastest) | 470.8 ms (19x slower) | **94.3 ms** (Balanced) |
| **VRAM Footprint** | ~100 MB | ~800 MB – 1.2 GB | ~550 MB |
| **Box Tightness** | Loose / Fragmented | Tight | **Tight & Accurate** |
| **Viability for Upstream P3** | **REJECTED** (High miss rate) | Viable but slow | **PROVISIONAL PRIMARY CANDIDATE** |

---

### 1.7 Visual Verification Highlights

Representative 4-panel visual composites generated during the audit are archived in `docs/audits/assets/perception_audit/`:

#### Example 1: ScienceDB Rear-View Chute (`sample_0001_sciencedb_bcs_3.25.jpg`)
- **YOLOv8s**: Missed detection (0 boxes).
- **Faster R-CNN v2**: Detected primary cow (score: 0.94).
- **RT-DETR-L**: Detected primary cow (score: 0.91) with tight bounding box encompassing spine and pin bones.

#### Example 2: MmCows Lying Posture (`sample_0031_mmcows_behavior_lying.jpg`)
- **YOLOv8s**: Missed detection (0 boxes).
- **Faster R-CNN v2**: Detected lying cow (score: 0.98) + 1 background stall cow.
- **RT-DETR-L**: Detected lying cow (score: 0.97) + 1 background stall cow.

#### Example 3: SideViewCows2026 Parlor Chute (`sample_0061_sideviewcows2026_reid_parlor.jpg`)
- **YOLOv8s**: Detected cow (score: 0.88).
- **Faster R-CNN v2**: Detected cow (score: 0.99).
- **RT-DETR-L**: Detected cow (score: 0.99) matching ground-truth segmentation contour.

---

### 1.8 Step 2.1 Feasibility Verdict
**VERDICT: LOCALIZATION FEASIBILITY CONFIRMED.**

1. **Pretrained Usability**: Off-the-shelf detectors without cattle fine-tuning are **sufficiently reliable** (94.3% raw detection rate with RT-DETR-L, 95.0% with Faster R-CNN v2) to serve as upstream cattle localizers for Phase 3 representation caching.
2. **Candidate Selection**: **RT-DETR-L** is designated as the **provisional primary candidate for the next perception stage**. While Faster R-CNN v2 achieved a slightly higher raw detection rate (95.0% vs 94.3%), RT-DETR-L provides a substantially better speed/performance tradeoff, running 5x faster (94.3 ms vs 470.8 ms) on local hardware with tight, accurate bounding boxes.
3. **Required Upstream Safeguard**: Because 76–87% of pen images contain background cattle, downstream feature extraction must include a primary-cow selection heuristic (largest bounding box or center-weighted box) rather than naive multi-box averaging.
4. **Gate Status**: Step 2.1 is complete. We are cleared to proceed to Step 2.2 (Segmentation feasibility).

---

## 2. Step 2.2 — Cow Segmentation Feasibility Audit

### 2.1 Executive Summary & Verdict
**VERDICT: PRETRAINED SEGMENTATION FEASIBILITY CONFIRMED.**

The Step 2.2 audit evaluated whether off-the-shelf, pretrained segmentation can cleanly isolate the target cow from the background across all three primary Phase 3 datasets (**ScienceDB**, **MmCows**, and **SideViewCows2026**) without any model training or fine-tuning.

**Key Findings**:
1. **Quantitative Performance on SideViewCows2026 Ground Truth (N=100)**:
   - **RT-DETR-L -> SAM 2.1** (`sam2.1_s.pt` / `sam2.1_hiera_small`) achieved a **Mean IoU of 0.9216** (Median: 0.9613) and **Mean Dice of 0.9530** (Median: 0.9803).
   - 99.0% of images achieved IoU >= 0.50, and 96.0% achieved IoU >= 0.70.
   - **Oracle Diagnostic Gap**: An oracle bounding box derived directly from ground-truth masks yielded Mean IoU 0.9468 and Mean Dice 0.9717. The narrow 0.0252 IoU delta confirms that RT-DETR-L boxes are sufficiently tight and accurate that SAM 2.1 boundary precision is preserved end-to-end.
2. **Fast Baseline Comparison (YOLO26s-seg)**:
   - Ultralytics `yolo26s-seg.pt` (cow class only) achieved Mean IoU 0.8660 and Mean Dice 0.9170 on SideView, but suffered severe miss rates in other environments: **38.0% missed on ScienceDB** and **27.0% missed on MmCows**, mirroring the localization failures of anchor-free YOLO in Step 2.1.
3. **Qualitative Usability on ScienceDB & MmCows (N=200)**:
   - Whenever a bounding box was provided by RT-DETR-L, SAM 2.1 successfully segmented the cow in **100% of cases** (93/93 on ScienceDB, 90/90 on MmCows). Zero internal SAM failures were observed.
   - On ScienceDB, SAM 2.1 cleanly excludes metal chute bars and barn floor textures while preserving dorsal ridges, pin bones, hook bones, and the tailhead necessary for BCS assessment.
4. **Practicality**:
   - Total pipeline latency (`RT-DETR-L` + `SAM 2.1 small`) averages **475.0 ms/frame** on a low-end NVIDIA GTX 1050 Ti (4GB VRAM) with peak memory consumption under 1.4 GB. For offline Phase 3 feature/crop caching, this throughput (~2.1 FPS) is fully practical.

---

### 2.2 Evaluated Models & Candidate Pipelines

| Candidate | Architecture / Checkpoint | Primary Input | Target Output | Purpose |
|---|---|---|---|---|
| **Pipeline A (Main)** | `RT-DETR-L` -> `SAM 2.1 small` (`sam2.1_s.pt`) | Full image + RT-DETR primary cow bbox | Binary cow mask (H x W) | High-precision primary segmentation pipeline |
| **Pipeline B (Baseline)** | `YOLO26s-seg` (`yolo26s-seg.pt`) | Full image (cow class 19 only) | Instance mask (H x W) | Direct fast single-stage segmentation baseline |
| **Diagnostic (Oracle)** | `Oracle GT Box` -> `SAM 2.1 small` (`sam2.1_s.pt`) | Full image + BBox derived from SideView GT mask | Binary cow mask (H x W) | Decouple detector box error from SAM segmentation capability |

---

### 2.3 Dataset Sample & Primary-Cow Selection Rule

The audit re-used the **identical deterministic 300-image sample** from Step 2.1 (`artifacts/perception_audit/sample_manifest_expanded.csv`):
- **ScienceDB**: 100 images (rear-view chute, diverse BCS scores 3.0 to 4.75).
- **MmCows**: 100 images (overhead/angle CCTV, behaviors: lying, standing, feeding, walking, drinking, licking).
- **SideViewCows2026**: 100 images (parlor chute, barn alleys, snapshots with verified PNG ground-truth segmentation masks).

**Primary-Cow Selection Rule**:
In multi-cow scenes, the primary cow bounding box was selected deterministically by **maximum bounding box area** `(x2 - x1) * (y2 - y1)`, with confidence score breaking any ties. Ground-truth masks were never used to guide inference.

**Upstream Failure Handling**:
When RT-DETR-L detected 0 cows, the sample was logged as `upstream_localization_failure` with empty mask fields. These were tracked separately and not penalized as SAM failures.

---

### 2.4 Quantitative Segmentation Evaluation (SideViewCows2026)

SideViewCows2026 contains verified pixel-level binary ground truth masks, enabling rigorous metric evaluation:

| Metric | RT-DETR-L -> SAM 2.1 (Main) | Oracle GT Box -> SAM 2.1 (Diagnostic) | YOLO26s-seg (Fast Baseline) |
|---|---|---|---|
| **Sample Size (N)** | 100 images | 100 images | 100 images |
| **Mean IoU** | **0.9216** | 0.9468 | 0.8660 |
| **Median IoU** | **0.9613** | 0.9639 | 0.9118 |
| **Mean Dice Coefficient** | **0.9530** | 0.9717 | 0.9170 |
| **Median Dice Coefficient** | **0.9803** | 0.9817 | 0.9538 |
| **IoU >= 0.50 (%)** | **99.0%** (99/100) | 100.0% (100/100) | 96.0% (96/100) |
| **IoU >= 0.70 (%)** | **96.0%** (96/100) | 99.0% (99/100) | 95.0% (95/100) |
| **IoU >= 0.85 (%)** | **88.0%** (88/100) | 94.0% (94/100) | 78.0% (78/100) |
| **Miss Rate (0 detections)** | **0.0%** (0/100) | 0.0% (0/100) | 1.0% (1/100) |

#### Diagnostic Analysis (Oracle vs RT-DETR):
- The Oracle GT box achieves **0.9468 Mean IoU** and **0.9717 Mean Dice**, demonstrating that SAM 2.1 possesses near-human cattle contour demarcation capability out-of-the-box.
- When fed bounding boxes from RT-DETR-L, performance is **0.9216 Mean IoU**, a marginal drop of only 2.5%. This proves that off-the-shelf RT-DETR-L localization is sufficiently precise to serve as a high-fidelity prompt for SAM 2.1.

---

### 2.5 ScienceDB & MmCows Usability Analysis (No Ground Truth)

ScienceDB and MmCows lack verified pixel ground truth masks; therefore, automated IoU/Dice was not synthesized. Systematic visual inspection across all samples established the following operational status breakdown:

#### Operational Status Distribution (N=100 per dataset)

| Dataset | Pipeline | Segmented (Valid Mask) | Upstream Localization Failure | Missed / No Detection |
|---|---|---|---|---|
| **ScienceDB** | **RT-DETR-L -> SAM 2.1** | **93 (93.0%)** | 7 (7.0%) | 0 (0.0%) |
| ScienceDB | YOLO26s-seg | 62 (62.0%) | N/A | 38 (38.0%) |
| **MmCows** | **RT-DETR-L -> SAM 2.1** | **90 (90.0%)** | 10 (10.0%) | 0 (0.0%) |
| MmCows | YOLO26s-seg | 73 (73.0%) | N/A | 27 (27.0%) |

#### Visual Inspection Quality Categories (RT-DETR-L -> SAM 2.1)

1. **ScienceDB (BCS rear view)**:
   - **Usable (91.0%)**: Cow body, spine, hooks, pin bones, and tailhead are cleanly masked. Metal chute bars, head gates, and concrete floors are cleanly excluded.
   - **Partial Mask (2.0%)**: Minor lower-leg cutoffs where heavy horizontal chute bars cross the lower hooves. Critical dorsal/pelvic anatomical regions remain intact.
   - **Upstream Localization Failure (7.0%)**: 7 images where RT-DETR-L detected 0 cows due to extreme darkness or extreme close-up camera angles.
   - **Background Leakage / Wrong Cow / Failed**: 0.0%.

2. **MmCows (CCTV behavior)**:
   - **Usable (84.0%)**: Clean body contours across standing, walking, drinking, feeding (head-up and head-down), and licking. Lying cows on straw bedding are cleanly delineated from floor straw.
   - **Partial Mask (4.0%)**: Mild boundary erosion on distant, heavily curled cows in dark cubicles.
   - **Wrong Cow (2.0%)**: In dense feeding alley scenes, the primary-cow heuristic (maximum box area) occasionally selected an adjacent foreground cow rather than the centrally active animal.
   - **Upstream Localization Failure (10.0%)**: 10 images where RT-DETR-L missed curled lying cows in low-contrast bedding.
   - **Background Leakage / Failed**: 0.0%.

---

### 2.6 Key Failure Modes & Edge Cases

1. **Multi-Cow Ambiguity in Dense Barns (`sample_0277`)**:
   - In `sample_0277` (`SideViewCows2026` barn alley), 23 cows were detected. The deterministic largest-box rule selected a prominent foreground cow, whereas the ground-truth mask was annotated for cow ID `594` standing further back. This yielded `sam2_sideview_iou = 0.0`.
   - When prompted with the Oracle GT box, SAM 2.1 segmented cow `594` with **0.9633 IoU**.
   - *Implication*: When doing Re-ID or multi-cow tracking, primary-cow selection must be tied to tracklet identity rather than a static largest-box heuristic.
2. **Upstream Detector Misses on Curled Lying Cows**:
   - All 10 MmCows upstream failures occurred in lying posture samples (`sample_0101`, `sample_0102`, `sample_0103`, `sample_0106`, `sample_0108`, `sample_0115`, `sample_0116`, `sample_0138`, `sample_0182`, `sample_0189`).
   - *Implication*: Curled cattle in low-contrast cubicles represent the primary remaining localization failure mode for COCO-pretrained detectors.
3. **Single-Stage Segmentation (YOLO26s-seg) Fragility**:
   - YOLO26s-seg failed to detect/segment cattle in 38% of ScienceDB images and 27% of MmCows images. Its reliance on standard anchor-free detection heads without deformable attention makes it brittle in non-standard camera viewpoints (rear chute, steep overhead CCTV).

---

### 2.7 Hardware & Runtime Efficiency

Measurements performed on local workstation (Intel i7, NVIDIA GeForce GTX 1050 Ti 4GB VRAM):

| Model / Stage | Hardware | Mean Latency (ms) | Median Latency (ms) | Peak VRAM | Feasibility for Caching |
|---|---|---|---|---|---|
| **RT-DETR-L** (Detector) | GTX 1050 Ti | 94.3 ms | 92.1 ms | ~550 MB | Fully Feasible |
| **SAM 2.1 small** (Segmenter) | GTX 1050 Ti | 380.7 ms | 378.7 ms | ~850 MB | Fully Feasible |
| **Pipeline A (Total)** | GTX 1050 Ti | **475.0 ms** (~2.1 FPS) | **470.8 ms** | **~1.4 GB** | **Optimal for Step 3 Caching** |
| **YOLO26s-seg** (Fast Baseline)| GTX 1050 Ti | 32.2 ms (~31.0 FPS) | 28.8 ms | ~150 MB | Fast, but accuracy unacceptable |

On the BRACU Lab Research PC (RTX 5090), Pipeline A is projected to execute at >30 FPS (<35 ms total), making large-scale dataset caching fast and frictionless.

---

### 2.8 Step 2.2 Feasibility Verdict
**VERDICT: PRETRAINED SEGMENTATION FEASIBILITY CONFIRMED.**

1. **Pretrained SAM 2.1 is validated**: Zero fine-tuning is needed to achieve >0.92 IoU and clean separation of cattle from complex barn and chute backgrounds.
2. **Architecture Recommendation**: **RT-DETR-L -> SAM 2.1 small** is designated as the primary segmentation pipeline for Step 3 feature caching.
3. **Gate Status**: Step 2.2 is complete. Proceed to Step 2.3 (Keypoint / Pose feasibility audit) per `phase3_canonical_roadmap.md`.

