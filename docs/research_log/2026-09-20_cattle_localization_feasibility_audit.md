# Cattle Detection & Localization Feasibility Audit (Step 2.1)

**Date**: 2026-09-20  
**Auditors**: Hasin Ishrak & Antigravity  
**Status**: COMPLETE / VERIFIED  
**Investigation Trigger**: Phase 3 Step 2.1 milestone to determine whether off-the-shelf, documented pretrained object detectors can reliably locate cattle across our three primary Phase 3 datasets without task-specific training or fine-tuning.

---

## 1. Executive Summary
A comprehensive, two-stage localization feasibility audit was conducted across **ScienceDB** (BCS), **MmCows** (Behavior), and **SideViewCows2026** (Re-ID) comparing three pretrained object detectors: **YOLOv8s** (11.2M params), **Faster R-CNN ResNet-50 FPN v2** (43.7M params), and **RT-DETR-L** (32.0M params). Following an initial 90-image smoke test, an expanded **300-image evaluation** (100 images per dataset, seed=42) proved that pretrained localization is viable for Phase 3. Crucially, the investigation showed that **YOLOv8s suffered a 37.0% non-detection rate** on rear-view chute images and tight behavior crops, while **RT-DETR-L (94.3% raw detection rate)** and **Faster R-CNN v2 (95.0% raw detection rate)** achieved robust cattle localization. **RT-DETR-L is designated as the provisional primary candidate for the next perception stage**, providing a superior speed/performance tradeoff at **94.3 ms latency** (5x faster than Faster R-CNN v2) on local GTX 1050 Ti hardware.

---

## 2. Context & Motivation
Phase 3 investigates whether cattle-centered visual representations (localization, segmentation, anatomy/pose, viewpoint) can reduce background shortcut learning and improve multi-task performance across BCS, Behavior, and Re-ID. 

Before building the representation pipeline, Step 2 requires verifying that upstream perception tools reliably locate cows without having to train a detector from scratch or resurrect unverified Phase 2 checkpoints (which are strictly excluded). In particular, the audit aimed to test:
1. Whether COCO-pretrained detectors can detect cattle from atypical camera viewpoints (rear-view chute walkthroughs in ScienceDB).
2. How detectors handle CCTV motion blur, metal stall occlusions, and cropped bounding boxes in MmCows.
3. Whether detectors can handle unconstrained outdoor pasture scenes and multi-cow pens in SideViewCows2026.

---

## 3. Forensic Findings & Comparison Data

### 3.1 Quantitative Performance across 300 Representative Samples
The expanded audit evaluated 300 deterministically sampled images:
- **ScienceDB (100 images)**: Stratified across all 5 BCS classes (20 per class: 3.25 to 4.25) from distinct connected burst groups.
- **MmCows (100 images)**: Stratified across all 7 behaviors (Lying: 16, Standing: 16, Feeding_head_down: 16, Feeding_head_up: 16, Walking: 14, Drinking: 11, Licking: 11) and verified to cover all 4 CCTV cameras (Cam 1: 24, Cam 2: 31, Cam 3: 14, Cam 4: 31) and all 16 biological cows (Cows 1–16, ranging from 1 to 13 samples per cow).
- **SideViewCows2026 (100 images)**: Stratified across parlor (40), barn (40), and snapshots (20).

| Model | Architecture | Parameters | ScienceDB (100) | MmCows (100) | SideView (100) | Overall Raw Detection (300) | Median Latency |
|---|---|---|---|---|---|---|---|
| **YOLOv8s** | Anchor-free CNN | 11.2M | 63.0% (37 missed) | 63.0% (37 missed) | 97.0% (3 missed) | **74.3%** | **24.5 ms** |
| **Faster R-CNN v2** | Two-stage RPN | 43.7M | 94.0% (6 missed) | 92.0% (8 missed) | 99.0% (1 missed) | **95.0%** | 470.8 ms |
| **RT-DETR-L** | Transformer | 32.0M | 93.0% (7 missed) | 90.0% (10 missed) | 100.0% (0 missed) | **94.3%** | **94.3 ms** |

*Note: Raw detection rates represent the percentage of images with >=1 cow detected at confidence >= 0.25. Because ground-truth bounding boxes are unavailable for ScienceDB and MmCows, these represent empirical detection rates rather than formal precision/recall metrics. All missed detections were stored as valid rows with empty bbox fields.*

### 3.2 Key Forensic Discoveries
1. **YOLO Empirical Failure on Non-Side Views**:
   - In 72 specific images, RT-DETR-L or Faster R-CNN cleanly located the cow while YOLOv8s reported 0 detections (35 in MmCows, 34 in ScienceDB, 3 in SideView).
   - In ScienceDB, cows walk away from the camera through a chute (visible anatomy: spine, pin bones, rump, tail).
   - **Hypothesis**: A plausible hypothesis is that COCO pasture imagery predominantly features broadside profiles, which may explain why YOLOv8s's anchor-free grid features failed to activate on rear-view anatomy or tight crops. However, the audit establishes only the observed performance difference; verifying training distribution or architectural causality would require controlled ablation.
2. **Contextual Feature Aggregation**:
   - Both RT-DETR-L and Faster R-CNN v2 effectively recognized rear-view cattle and occluded cows behind metal stall bars.
   - **Hypothesis**: Faster R-CNN's region proposal network (RPN) and RT-DETR's global self-attention mechanism are hypothesized to aggregate distributed contextual cues (e.g. spine contours, body texture, hooves) more effectively than the tested single-stage CNN.
3. **Pen Clutter & Multi-Cow Contamination**:
   - In MmCows, 76% (RT-DETR) and 77% (Faster R-CNN) of images produced multiple cow detections. Visual inspection confirmed that background cows in adjacent cubicles visible through the bars are detected.
   - In SideViewCows2026, 81–87% produced multiple detections due to herd members visible in the background.
4. **Universal Misses (1.67% of cases)**:
   - Only 5 of 300 images were missed by all three models: 3 in ScienceDB (extreme chute entry/exit where only tail or partial rump was visible in dark shadow) and 2 in MmCows (`sample_0115` and `sample_0116`, which are extreme horizontal aspect ratio 2.68:1 crops of cows lying flat against concrete dividers).

---

## 4. Architectural Decisions & Action Plan

### Decision 1: Reject YOLOv8s for Upstream Cattle Localization
- **Rationale**: A 37% non-detection rate on ScienceDB and MmCows makes YOLOv8s unviable as an upstream visual prior. Using YOLOv8s would drop or corrupt more than a third of BCS and behavior samples.

### Decision 2: Designate RT-DETR-L as Provisional Primary Candidate for Step 2.2
- **Rationale**: While Faster R-CNN v2 achieved a slightly higher raw detection rate (95.0% vs 94.3%), RT-DETR-L runs **5x faster** (94.3 ms vs 470.8 ms) on local hardware with tight, accurate bounding boxes, making it the most balanced candidate for feeding upstream bounding boxes to SAM 2 / SAM 2.1.

### Decision 3: Implement Primary-Cow Filtering Heuristic
- **Rationale**: Because commercial pens contain multiple visible cows, downstream caching must not naively pool all detections. A primary-cow selection rule (e.g., maximum bounding-box area or center-proximity) must be enforced.

---

## 5. Artifacts & File Registry
- **Reproducible Audit Script**: `scripts/audit_localization_feasibility.py`
- **Audit Report**: `docs/audits/phase3_perception_feasibility.md`
- **Detections Manifests**:
  - `artifacts/perception_audit/sample_manifest_smoke.csv` (90 samples)
  - `artifacts/perception_audit/sample_manifest_expanded.csv` (300 samples)
  - `artifacts/perception_audit/localization_detections_smoke.csv`
  - `artifacts/perception_audit/localization_detections_expanded.csv`
  - `artifacts/perception_audit/localization_summary_smoke.csv`
  - `artifacts/perception_audit/localization_summary_expanded.csv`
- **Visual Inspection Composites**: `docs/audits/assets/perception_audit/` (120 4-panel review composites comparing Original, YOLOv8s, Faster R-CNN v2, and RT-DETR-L).

---

## 6. Next Steps
1. **Proceed to Step 2.2**: Cow segmentation feasibility audit (evaluating SAM 2 / SAM 2.1 mask generation prompted by RT-DETR bounding boxes).
2. Complete full Step 2 feasibility audit before caching upstream perception outputs in Step 3.
