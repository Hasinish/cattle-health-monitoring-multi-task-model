# Research Log: Cattle Pose / Keypoint Feasibility Audit (Step 2.3)

**Date**: 2026-09-20  
**Phase**: Phase 3 (Pretrained Perception Feasibility Audit)  
**Deliverable Document**: [docs/audits/phase3_perception_feasibility.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_perception_feasibility.md)  
**Execution Script**: [scripts/audit_pose_feasibility.py](file:///d:/cattle-health-monitoring-multi-task-model/scripts/audit_pose_feasibility.py)  

---

## 1. Objective & Scope

Determine whether existing pretrained animal-pose foundation models can extract usable anatomical keypoints across our three primary Phase 3 datasets without any training or fine-tuning:
1. **ScienceDB** (BCS rear-view chute)
2. **MmCows** (Overhead/angled CCTV behavior)
3. **SideViewCows2026** (Milking parlor and barn Re-ID)

This is an empirical feasibility audit only; no commitment is made yet to include pose in the final Phase 3 architecture.

---

## 2. Models & Setup

* **Framework**: DeepLabCut 3.0.1 (PyTorch engine)
* **Pose Foundation Model**: `SuperAnimal-Quadruped`
* **Evaluated Backbones**:
  1. `HRNet-W32` (`superanimal_quadruped_hrnet_w32.pt`)
  2. `ResNet-50` (`superanimal_quadruped_resnet_50.pt`)
* **Detector**: `fasterrcnn_resnet50_fpn_v2` (`superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt`)
* **Schema**: Official 39-keypoint quadruped schema dynamically extracted from model metadata.
* **Pipeline**:
  `Full Image` -> `Step 2.1 RT-DETR-L primary cow crop` -> `Official SuperAnimal pipeline (max_individuals=1)` -> `Map keypoints back to original-image coordinates`.
* **Cattle-Specific Check**:
  `No verified directly usable pretrained cattle-specific pose checkpoint was found for this feasibility audit.` (CattleEyeView has no public weights; BECA has no pose labels).
* **Deterministic Dataset**: Identical 300-image expanded sample from Step 2.1 and Step 2.2 (`artifacts/perception_audit/sample_manifest_expanded.csv`).

---

## 3. Quantitative Results Summary

### Operational Status Breakdown (N=300 per model)

| Model | Dataset | `pose_output_returned` | `pose_detector_failure` | `upstream_localization_failure` | `pose_inference_error` |
|---|---|---|---|---|---|
| **HRNet-W32** | ScienceDB | 85 (85.0%) | 8 (8.0%) | 7 (7.0%) | 0 (0.0%) |
| | MmCows | 68 (68.0%) | 22 (22.0%) | 10 (10.0%) | 0 (0.0%) |
| | SideViewCows2026 | 90 (90.0%) | 10 (10.0%) | 0 (0.0%) | 0 (0.0%) |
| | **Total** | **243 (81.0%)** | **40 (13.3%)** | **17 (5.7%)** | **0 (0.0%)** |
| **ResNet-50** | ScienceDB | 85 (85.0%) | 8 (8.0%) | 7 (7.0%) | 0 (0.0%) |
| | MmCows | 68 (68.0%) | 22 (22.0%) | 10 (10.0%) | 0 (0.0%) |
| | SideViewCows2026 | 90 (90.0%) | 10 (10.0%) | 0 (0.0%) | 0 (0.0%) |
| | **Total** | **243 (81.0%)** | **40 (13.3%)** | **17 (5.7%)** | **0 (0.0%)** |

### Confidence & Geometric Sanity Check (on `pose_output_returned`)

| Model | Dataset | N | Mean Raw Conf | Median Raw Conf | Pts >= 0.2 (/39) | Keypoints-Inside-Mask Rate |
|---|---|---|---|---|---|---|
| **HRNet-W32** | ScienceDB | 85 | 0.1324 | 0.1004 | 8.2 (21.0%) | N/A (no mask GT) |
| | MmCows | 68 | 0.2287 | 0.1911 | 15.7 (40.3%) | N/A (no mask GT) |
| | SideViewCows2026 | 90 | 0.4093 | 0.4113 | 25.1 (64.4%) | **72.74%** |
| **ResNet-50** | ScienceDB | 85 | 0.2878 | 0.2522 | 25.3 (64.9%) | N/A (no mask GT) |
| | MmCows | 68 | 0.3607 | 0.3311 | 28.3 (72.6%) | N/A (no mask GT) |
| | SideViewCows2026 | 90 | 0.4838 | 0.4734 | 32.2 (82.6%) | **77.21%** |

---

## 4. Key Scientific Findings & Visual Inspection

A persistent manual visual-validation record covering the contact-sheet review samples was recorded in `artifacts/perception_audit/pose_manual_review.csv` (N=60 reviews across 30 unique samples and 2 models; `review_source = ChatGPT-assisted visual review + human verified`).

### Ground-Truth & Metric Caveats:
* **No Keypoint Ground Truth**: ScienceDB, MmCows, and SideViewCows2026 do not provide verified keypoint annotations; therefore, no PCK, OKS, or pose mAP can be claimed.
* **Raw Confidence != Accuracy**: Higher confidence indicates internal model activation strength, not ground-truth anatomical accuracy.
* **Mask Containment is a Geometric Sanity Check**: The `keypoints-inside-mask rate` on SideView confirms keypoints land roughly on cow pixels, not that specific joints are correctly localized.

### Dataset-Specific Observations:
1. **ScienceDB (BCS rear view)**:
   * Visual inspection indicates outputs were frequently anatomically implausible or scattered across the rump and chute bars.
   * Rear-view pose appears weak for capturing cattle anatomy from this perspective. The model hallucinates cranial points (`lower_jaw`, `upper_jaw`) on cows whose heads are occluded behind their bodies.
   * The SuperAnimal schema completely lacks hip/pin/hook bone keypoints (*tuber coxae*, *tuber ischiadicum*) or pelvic depression markers, which are the primary anatomical landmarks needed for BCS.
   * **Visual Assessment**: Zero-shot pose is not recommended for downstream BCS based on visual inspection.

2. **MmCows (Behavior postures)**:
   * Visual inspection indicates mixed and fragile outputs.
   * While successful outputs sometimes capture coarse posture for standing or walking cattle, keypoints are frequently unreliable under stall bars and lying occlusions.
   * Curled lying cows and cows behind heavy stall bars trigger a 22% internal detector failure rate (`pose_detector_failure`).
   * **Visual Assessment**: Outputs are fragile with significant failure modes on occluded/lying postures.

3. **SideViewCows2026 (Re-ID)**:
   * SideView outputs appeared more anatomically plausible than ScienceDB or MmCows, aligning generally with visible body contours (though occasional drift and background limb errors remain).
   * Geometric sanity check confirms 72.7% (HRNet) and 77.2% (ResNet) of predicted points fall within the ground-truth cow mask.
   * **Visual Assessment**: SideView outputs appeared more anatomically plausible and are promising for downstream ablation; downstream utility must still be tested in Step 6.

4. **Model Comparison**:
   * **ResNet-50 vs HRNet-W32**: ResNet-50 produced higher raw confidence and slightly higher mask containment, but these do not establish higher pose accuracy. Both share identical 81.0% operational success rates. ResNet-50 is designated as the provisional candidate for Step 6 pose ablation.

---

## 5. Artifacts & Outputs

* Script: `scripts/audit_pose_feasibility.py`
* Manual Review Record: `artifacts/perception_audit/pose_manual_review.csv` (N=60 reviews across 30 samples)
* Contact Sheets: `docs/audits/assets/pose_visual_review/*.jpg`
* Visual Review Index: `docs/audits/phase3_pose_visual_review_index.md`
* Results CSVs:
  - `artifacts/perception_audit/pose_results_expanded_hrnet_w32.csv`
  - `artifacts/perception_audit/pose_results_expanded_resnet_50.csv`
* Keypoint CSVs:
  - `artifacts/perception_audit/pose_keypoints_expanded_hrnet_w32.csv`
  - `artifacts/perception_audit/pose_keypoints_expanded_resnet_50.csv`
* Schema JSON: `artifacts/perception_audit/superanimal_quadruped_schema.json`
* Composites: `docs/audits/assets/perception_audit/pose_*.jpg`

