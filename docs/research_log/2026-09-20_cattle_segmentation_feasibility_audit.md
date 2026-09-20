# Cattle Segmentation Feasibility Audit (Step 2.2)

**Date**: 2026-09-20  
**Auditors**: Hasin Ishrak & Antigravity  
**Status**: COMPLETE / VERIFIED  
**Investigation Trigger**: Phase 3 Step 2.2 milestone to determine whether off-the-shelf, pretrained segmentation models can cleanly separate target cattle from complex backgrounds across our three primary Phase 3 datasets without task-specific training or fine-tuning.

---

## 1. Executive Summary
A comprehensive segmentation feasibility audit was conducted across **ScienceDB** (BCS), **MmCows** (Behavior), and **SideViewCows2026** (Re-ID) comparing the proposed main pipeline (**RT-DETR-L primary box -> official pretrained SAM 2.1 small**) against the fast single-stage baseline (**YOLO26s-seg**, cow class only) and an **Oracle GT Box -> SAM 2.1** diagnostic. Testing the identical deterministic 300-image sample from Step 2.1 confirmed that pretrained segmentation is **highly feasible** for Phase 3:
1. **SideViewCows2026 Ground Truth (N=100)**: RT-DETR-L -> SAM 2.1 achieved **0.9216 Mean IoU** (Median: 0.9613) and **0.9530 Mean Dice** (Median: 0.9803), with 99.0% IoU >= 0.50 and 96.0% IoU >= 0.70.
2. **Oracle Diagnostic Gap**: Prompting SAM 2.1 with oracle boxes derived directly from ground-truth masks yielded 0.9468 Mean IoU. The small observed delta (0.0252 IoU) indicates that RT-DETR-L box prompts worked well with SAM 2.1 on this SideView sample.
3. **Qualitative Usability (ScienceDB & MmCows, N=200)**: Whenever a bounding box was supplied by RT-DETR-L, SAM 2.1 produced a mask in **100% of cases** (93/93 on ScienceDB, 90/90 on MmCows) with zero internal SAM failures. In reviewed composites, metal chute bars, head gates, and concrete/straw flooring were cleanly excluded while preserving dorsal ridges, pin bones, and silhouettes.
4. **Fast Baseline Comparison**: `YOLO26s-seg` had substantially higher non-detection rates in this audit (38.0% missed on ScienceDB, 27.0% missed on MmCows).

---

## 2. Context & Experimental Setup

### 2.1 Candidates Evaluated
- **Pipeline A (Main)**: `RT-DETR-L` cow box -> `SAM 2.1 small` (`sam2.1_s.pt` / `sam2.1_hiera_small`, 46M params).
- **Pipeline B (Fast Baseline)**: Ultralytics pretrained `YOLO26s-seg` (`yolo26s-seg.pt`, cow class 19 only).
- **Diagnostic Pipeline (Oracle)**: BBox derived from SideView GT mask -> `SAM 2.1 small` (SideView only).

### 2.2 Dataset Sample & Primary-Cow Rule
- **Sample Manifest**: Re-used the exact deterministic 300-image sample (`artifacts/perception_audit/sample_manifest_expanded.csv`): 100 ScienceDB (BCS 3.0–4.75), 100 MmCows (7 behaviors, 4 cameras, 16 cows), 100 SideViewCows2026 (parlor, barn, snapshots).
- **Primary-Cow Selection Rule**: In multi-cow scenes, the primary cow bounding box was selected deterministically by **maximum bounding box area** `(x2 - x1) * (y2 - y1)`, with confidence breaking ties. Ground-truth masks were never used during inference.
- **Upstream Localization Failure Handling**: If RT-DETR-L detected 0 cows, the sample was recorded as `upstream_localization_failure` with empty mask fields (not penalized as a SAM failure).

---

## 3. Quantitative & Qualitative Findings

### 3.1 SideViewCows2026 Ground-Truth Evaluation (N=100)

| Metric | RT-DETR-L -> SAM 2.1 (Main) | Oracle GT Box -> SAM 2.1 (Diagnostic) | YOLO26s-seg (Fast Baseline) |
|---|---|---|---|
| **Mean IoU** | **0.9216** | 0.9468 | 0.8660 |
| **Median IoU** | **0.9613** | 0.9639 | 0.9118 |
| **Mean Dice** | **0.9530** | 0.9717 | 0.9170 |
| **Median Dice** | **0.9803** | 0.9817 | 0.9538 |
| **IoU >= 0.50 (%)** | **99.0%** (99/100) | 100.0% (100/100) | 96.0% (96/100) |
| **IoU >= 0.70 (%)** | **96.0%** (96/100) | 99.0% (99/100) | 95.0% (95/100) |
| **Miss Rate (0 detections)** | **0.0%** (0/100) | 0.0% (0/100) | 1.0% (1/100) |

### 3.2 ScienceDB & MmCows Operational Breakdown (N=100 per dataset)

| Dataset | Pipeline | Segmented (Valid Mask) | Upstream Localization Failure | Missed / No Detection |
|---|---|---|---|---|
| **ScienceDB** | **RT-DETR-L -> SAM 2.1** | **93 (93.0%)** | 7 (7.0%) | 0 (0.0%) |
| ScienceDB | YOLO26s-seg | 62 (62.0%) | N/A | 38 (38.0%) |
| **MmCows** | **RT-DETR-L -> SAM 2.1** | **90 (90.0%)** | 10 (10.0%) | 0 (0.0%) |
| MmCows | YOLO26s-seg | 73 (73.0%) | N/A | 27 (27.0%) |

### 3.3 Visual Inspection Observations (RT-DETR-L -> SAM 2.1)
Based on manual visual review of the saved composite samples in `docs/audits/assets/perception_audit/`:
- **ScienceDB**:
  - **Usable**: In reviewed composites, cow body, dorsal ridge, pin bones, hooks, and tailhead are cleanly masked while metal chute bars and floors are excluded.
  - **Partial Mask**: Minor lower-leg cutoffs occur in some images where heavy bottom stanchions cross lower hooves.
  - **Upstream Fail**: 7 samples in the set had 0 cows detected by RT-DETR-L due to extreme darkness or extreme close-up angles.
  - **Background Leakage / Wrong Cow**: Not observed in reviewed composites.
- **MmCows**:
  - **Usable**: In reviewed composites, clean silhouettes are observed across standing, walking, drinking, feeding, and licking postures. Straw bedding is separated from lying cows.
  - **Partial Mask**: Slight boundary erosion observed on some distant curled cows.
  - **Wrong Cow**: In crowded feeding alleys, largest-box rule occasionally selected an adjacent foreground cow.
  - **Upstream Fail**: 10 samples in the set had 0 cows detected by RT-DETR-L (all lying cows in low-contrast cubicles).
  - **Background Leakage**: Not observed in reviewed composites.

---

## 4. Hardware & Efficiency Analysis
Measured on local workstation (Intel i7, NVIDIA GeForce GTX 1050 Ti 4GB VRAM):
- **RT-DETR-L Latency**: 94.3 ms mean
- **SAM 2.1 small Latency**: 380.7 ms mean (378.7 ms median)
- **Total Pipeline Latency**: **475.0 ms/frame** (~2.1 FPS)
- **Peak VRAM**: ~1.4 GB (zero memory leaks, clean PyTorch cache management)
- **Feasibility for Caching**: Because perception models will run offline to precompute and cache cattle crops/masks during Step 3, 2.1 FPS on a 1050 Ti is fully viable; RTX 5090 performance has not yet been measured and is expected to be faster.

---

## 5. Key Forensic Discoveries & Failure Modes
1. **Multi-Cow Selection Discrepancy (`sample_0277`)**:
   In `sample_0277` (`SideViewCows2026` barn alley), 23 cows were present. The deterministic largest-box heuristic selected a foreground cow rather than the ground-truth target cow `594`, resulting in `sam2_sideview_iou = 0.0`. When prompted with the Oracle GT box, SAM 2.1 achieved **0.9633 IoU**.
   *Implication*: In multi-cow video tracking, box prompts must be linked to persistent tracklet IDs rather than static area heuristics.
2. **Single-Stage Segmentation Non-Detection Rate**:
   `YOLO26s-seg` exhibited substantially higher non-detection rates (38% on ScienceDB, 27% on MmCows) in this audit. Architectural or data-distribution hypotheses remain unproven without controlled ablations.
3. **Curled Lying Cows as Remaining Localization Challenge**:
   All 10 MmCows upstream misses were lying cows curled against cubicle walls.

---

## 6. Architectural Decision
- **ADOPT**: **RT-DETR-L -> SAM 2.1 small** as the primary segmentation pipeline for Step 3 representation caching.
- **REJECT**: `YOLO26s-seg` as an upstream segmenter due to prohibitive miss rates on non-broadside cattle.
- **GATE STATUS**: Step 2.2 is complete and verified. Cleared to proceed to Step 2.3 (Keypoint / Pose feasibility audit).
