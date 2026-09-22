# Phase 3 Real-Cattle Visual Quality Reassessment Audit

**Date**: 2026-09-22  
**Author / Auditor**: Antigravity Research Agent & Hasin Ishrak  
**Scope**: Step 2 Perception Quality Feasibility & Evidence Preservation  
**Source Manifest**: [`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv)  
**Contact Sheet Index**: [`docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md)  
**Visual Asset Directory**: [`docs/audits/assets/real_cattle_visual_quality_reassessment/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/real_cattle_visual_quality_reassessment/)  
**Status**: COMPLETE / SCIENTIFIC EVIDENCE PRESERVED (NO DATASET-ROLE OR ROADMAP CHANGES)  

---

## 1. Executive Summary & Purpose

This forensic audit preserves the empirical findings from the newly completed **1,000-image human-verified real-cattle annotation review** conducted across the primary Phase 3 datasets: **ScienceDB** (334 samples), **MmCows** (333 samples), and **SideViewCows2026** (333 samples).

The purpose of this audit is strictly **evidence preservation and scalable visual quality inspection**:
1. Recompute and verify all quantitative statistics directly from the source of truth manifest (`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`).
2. Provide a deterministic 42-sheet visual contact sheet pack covering 100% of the 1,000 reviewed images with explicit metadata annotations.
3. Formulate rigorous, defensible scientific conclusions regarding visual quality challenges (framing, occlusion, multi-cow interference, and viewpoint ambiguity).
4. **Strict Boundary**: This audit does **NOT** alter dataset roles, does **NOT** modify `phase3_canonical_roadmap.md`, does **NOT** train models, and does **NOT** claim alternative datasets (e.g. Ruchay 2026, Simmental, XGain) are superior without equivalent empirical task-specific audits.

---

## 2. Measurement Scope & Interpretation Guardrails

To prevent conflation of distinct visual and biological phenomena, the metrics in this audit are strictly delimited:

### What This 1,000-Image Review Schema Measures:
- **Viewpoint**: The coarse physical orientation of the target cow relative to the camera lens (`rear`, `rear-oblique`, `side`, `front-oblique`, `front`, or `unknown / ambiguous`).
- **Occlusion**: The degree to which foreground obstacles (stall divider pipes, stanchions, headlocks, curbs, or other cattle) obstruct the target animal's body (`none`, `partial`, `severe`).
- **Body Cutoff**: Truncation of the target cow by the camera sensor boundary or crop window (`none`, `partial`, `severe`).
- **Multiple Cows**: Presence of one or more non-target cattle in the image frame or crop (`yes`, `no`).

### What This Review Schema Does NOT Measure:
- **Blur / Sensor Sharpness**: High-frequency edge preservation, optical focus, or motion blur are not quantified here.
- **Expert BCS Scoreability**: Visibility of specific anatomical scoring landmarks (hook bones, pin bones, sacral ligament, thurl depression, tailhead fat pads) is not evaluated here.
- **Correctness of BCS Ground Truth Labels**: Whether the numerical 3.25–4.25 ratings match veterinary gold standards is not audited here.
- **Static Behavior-Label Correctness**: Whether the annotated behavior category matches the actual physical activity in static or temporal context is not re-evaluated here.
- **General Model Learnability**: Whether a deep neural network can overcome partial occlusion, multi-cow presence, or truncation via feature invariance is an empirical question for downstream ablations, not decided by manual inspection alone.

---

## 3. Quantitative Recomputation & Verification

All figures reported below were recomputed deterministically from [`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv) using `scripts/build_real_cattle_quality_contact_sheets.py`.

### 3.1 Strict-Clean Criterion Definition
A sample is formally designated **Strict-Clean** if and only if:
```text
occlusion == "none" AND
body_cutoff == "none" AND
multiple_cows == "no" AND
viewpoint != "unknown / ambiguous"
```

### 3.2 Overall 1,000-Image Dataset Distributions

- **Exact Row Count**: **1,000**
- **Provenance / Review Status**: **1,000 / 1,000 (100.0%) `human_verified`**
- **Dataset Counts**:
  - **ScienceDB**: 334 (33.40%)
  - **MmCows**: 333 (33.30%)
  - **SideViewCows2026**: 333 (33.30%)

| Metric Category | Value / Class | Total Count (N=1,000) | Overall Percentage |
| :--- | :--- | :---: | :---: |
| **Strict-Clean Status** | **Strict-Clean** | **342** | **34.20%** |
| | Non-Clean (Issues Present) | 658 | 65.80% |
| **Viewpoint** | Side | 453 | 45.30% |
| | Rear | 312 | 31.20% |
| | Unknown / Ambiguous | 149 | 14.90% |
| | Rear-Oblique | 43 | 4.30% |
| | Front-Oblique | 41 | 4.10% |
| | Front | 2 | 0.20% |
| **Occlusion** | None | 715 | 71.50% |
| | Partial | 163 | 16.30% |
| | Severe | 122 | 12.20% |
| **Body Cutoff** | None | 799 | 79.90% |
| | Partial | 136 | 13.60% |
| | Severe | 65 | 6.50% |
| **Multiple Cows** | No (Single Cow) | 761 | 76.10% |
| | Yes (Multiple Cows) | 239 | 23.90% |

---

### 3.3 Per-Dataset Detailed Distributions

#### A. ScienceDB (Primary BCS Benchmark, N=334)
- **Strict-Clean**: **171 / 334 (51.20%)**
- **Viewpoint Distribution**:
  - `rear`: 304 (91.02%)
  - `rear-oblique`: 15 (4.49%)
  - `unknown / ambiguous`: 15 (4.49%)
  - `side`: 0 (0.00%)
  - `front-oblique`: 0 (0.00%)
  - `front`: 0 (0.00%)
- **Occlusion Distribution**:
  - `none`: 333 (99.70%)
  - `partial`: 1 (0.30%)
  - `severe`: 0 (0.00%)
- **Body Cutoff Distribution**:
  - `none`: 291 (87.13%)
  - `partial`: 29 (8.68%)
  - `severe`: 14 (4.19%)
- **Multiple Cows Distribution**:
  - `no`: 206 (61.68%)
  - `yes`: **128 (38.32%)**

#### B. MmCows (Primary Behavior Benchmark, N=333)
- **Strict-Clean**: **43 / 333 (12.91%)**
- **Viewpoint Distribution**:
  - `unknown / ambiguous`: **133 (39.94%)**
  - `side`: 130 (39.04%)
  - `front-oblique`: 38 (11.41%)
  - `rear-oblique`: 22 (6.61%)
  - `rear`: 8 (2.40%)
  - `front`: 2 (0.60%)
- **Occlusion Distribution**:
  - `none`: 152 (45.65%)
  - `severe`: **93 (27.93%)**
  - `partial`: **88 (26.43%)**
  - *Combined Occlusion Rate (Partial + Severe)*: **181 / 333 (54.35%)**
- **Body Cutoff Distribution**:
  - `none`: 282 (84.68%)
  - `partial`: 32 (9.61%)
  - `severe`: 19 (5.71%)
- **Multiple Cows Distribution**:
  - `no`: 255 (76.58%)
  - `yes`: 78 (23.42%)

#### C. SideViewCows2026 (Primary Re-ID Benchmark, N=333)
- **Strict-Clean**: **128 / 333 (38.44%)**
- **Viewpoint Distribution**:
  - `side`: 323 (97.00%)
  - `rear-oblique`: 6 (1.80%)
  - `front-oblique`: 3 (0.90%)
  - `unknown / ambiguous`: 1 (0.30%)
  - `rear`: 0 (0.00%)
  - `front`: 0 (0.00%)
- **Occlusion Distribution**:
  - `none`: 230 (69.07%)
  - `partial`: 74 (22.22%)
  - `severe`: 29 (8.71%)
  - *Combined Occlusion Rate (Partial + Severe)*: **103 / 333 (30.93%)**
- **Body Cutoff Distribution**:
  - `none`: 226 (67.87%)
  - `partial`: 75 (22.52%)
  - `severe`: 32 (9.61%)
  - *Combined Cutoff Rate (Partial + Severe)*: **107 / 333 (32.13%)**
- **Multiple Cows Distribution**:
  - `no`: 300 (90.09%)
  - `yes`: 33 (9.91%)

---

## 4. Verification of Currently Discussed Claims

We explicitly benchmark the empirical findings against the claims previously circulating in pair-programming discussions:

| Claim Description | Chat / Hypothesized Claim | Recomputed Truth (CSV) | Status | Forensic Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **MmCows Sample Count** | 333 reviewed samples | **333** | **VERIFIED** | Exactly 333 rows in manifest. |
| **MmCows Ambiguous Viewpoint** | 133 / 333 ambiguous | **133 / 333 (39.94%)** | **VERIFIED** | Exactly 133 samples have `viewpoint == 'unknown / ambiguous'`. |
| **MmCows Partial Occlusion** | 88 partial occlusion | **88 / 333 (26.43%)** | **VERIFIED** | Exactly 88 samples have `occlusion == 'partial'`. |
| **MmCows Severe Occlusion** | 93 severe occlusion | **93 / 333 (27.93%)** | **VERIFIED** | Exactly 93 samples have `occlusion == 'severe'`. |
| **MmCows Strict-Clean** | ~43 / 333 strict-clean | **43 / 333 (12.91%)** | **VERIFIED** | Exactly 43 samples meet the strict-clean criteria. |
| **ScienceDB Sample Count** | 334 reviewed samples | **334** | **VERIFIED** | Exactly 334 rows in manifest. |
| **ScienceDB Multiple Cows** | 128 / 334 multiple cows | **128 / 334 (38.32%)** | **VERIFIED** | Exactly 128 samples have `multiple_cows == 'yes'`. |
| **SideView Sample Count** | 333 reviewed samples | **333** | **VERIFIED** | Exactly 333 rows in manifest. |

All 8 specific quantitative claims are **100% verified and mathematically exact**.

---

## 5. Visual Inspection of Contact Sheets: Dataset-by-Dataset Qualitative Summary

The visual contact-sheet pack comprises 42 sheets located at [`docs/audits/assets/real_cattle_visual_quality_reassessment/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/real_cattle_visual_quality_reassessment/) and indexed in [`docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md).

### 5.1 ScienceDB Qualitative Analysis (Sheets `sciencedb_sheet_01.jpg` to `sciencedb_sheet_14.jpg`)
- **Viewpoint Homogeneity**: Visual inspection confirms that ScienceDB is overwhelmingly rear-oriented (91.02% pure rear, 4.49% rear-oblique). The animals are walking through fixed squeeze chutes or weigh scales.
- **Occlusion Quality**: ScienceDB has virtually **zero foreground occlusion** (99.70% `none`). The chute sides do not block the pelvic anatomy.
- **The Multi-Cow Challenge**: 38.32% of images contain non-target cattle. Visually, these manifest in two distinct ways:
  1. Adjacent chute lines: In multi-lane facilities (e.g. `YM_Farm2`), cows in parallel chutes are visible in the background.
  2. Queue following: Cows following closely behind in the race are partially visible at the bottom or sides of the frame.
- **Framing & Cutoff**: 12.87% have partial or severe body cutoffs, typically where the top of the cow or the lower legs are clipped by the camera field of view.
- **Scientific Interpretation for BCS**: While multi-cow presence and framing cutoffs are present, they primarily affect the peripheral background or head/feet. Crucially, the rear pelvic anatomy (hooks, pins, tailhead) remains visible in most samples. This evidence documents framing issues but is **not sufficient by itself to demote ScienceDB from primary BCS**, because pelvic anatomical scoreability was not explicitly evaluated in this schema.

### 5.2 MmCows Qualitative Analysis (Sheets `mmcows_sheet_01.jpg` to `mmcows_sheet_14.jpg`)
- **Viewpoint Ambiguity**: 39.94% of MmCows samples are classified as `unknown / ambiguous` (concentrated on sheets `mmcows_sheet_01.jpg` through `mmcows_sheet_06.jpg`). Visually, this is driven by:
  1. Steep overhead CCTV angles where cows are curled in resting postures, making head vs tail indistinguishable from single frames.
  2. Extreme tight bounding-box crops where only a dorsal patch or torso section is captured.
- **Severe Occlusion**: 54.35% of samples have partial or severe occlusion. Galvanized steel cubicle divider pipes, headlocks, feed bunk bars, and stanchions cross directly across the cow's body, frequently bisecting limbs or torso.
- **Low Strict-Clean Rate**: Only 12.91% (43/333) of crops are strict-clean.
- **Reconciliation with Earlier 20-Sample Visual Inspection**:
  The earlier 20-sample visual inspection in [`docs/audits/phase3_behavior_agent_visual_inspection.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_behavior_agent_visual_inspection.md) rated 19/20 samples as "usable" (12 GOOD, 7 ACCEPTABLE, 1 QUESTIONABLE). This discrepancy is explained by:
  1. **Sample Size & Selection**: The earlier inspection was a small 20-sample audit designed to assess whether MmCows was completely corrupted relative to CBVD-5. The current 333-sample audit provides a much larger, statistically robust sample across all cameras, behaviors, and cows.
  2. **Evaluation Metric Difference**: The 20-sample inspection evaluated *gross behavior recognizability* (can an agent/model tell the cow is lying down or feeding behind bars?). The 1,000-image schema evaluated *strict photographic cleanliness* (viewpoint ambiguity, bar occlusion, cutoffs).
  3. **Historical Handling**: The earlier 20-sample conclusion is treated as **sample-size-limited**, not fraudulent. The new 333-sample evidence decisively demonstrates that MmCows has substantial ambiguity and occlusion that must be accounted for in representation learning.

### 5.3 SideViewCows2026 Qualitative Analysis (Sheets `sideview_sheet_01.jpg` to `sideview_sheet_14.jpg`)
- **Viewpoint Homogeneity**: 97.00% pure side views. Animals walk broadside past milking parlor cameras or through barn alleys.
- **Parlor Framing & Body Cutoff**: 32.13% of samples exhibit partial or severe body cutoff. Visually, cows entering or exiting the narrow parlor frame frequently have their heads, necks, or hindquarters clipped by the image edge.
- **Stanchion & Curb Occlusion**: 30.93% have partial or severe occlusion from milking parlor stall framework, curb rails, or pit edges crossing the lower limbs and udder.
- **Strict-Clean Rate**: 38.44% (128/333). While parlor framework causes cutoffs, the lateral flank coat patterns (the primary signal for Re-ID) remain extensively visible.

---

## 6. Scientific Decision Constraints & Conclusions

1. **NO Dataset-Role Change in This Task**:
   - ScienceDB remains the **Primary BCS Benchmark** (burst-group-disjoint).
   - MmCows remains the **Primary Behavior Benchmark** (cow-disjoint).
   - SideViewCows2026 remains the **Primary Re-ID Benchmark** (4 canonical protocols).
   No dataset roles are altered as a result of this audit.

2. **NO Roadmap Modification in This Task**:
   `phase3_canonical_roadmap.md` remains canonical and locked for execution.

3. **NO Premature Promotion of Alternatives**:
   We make no claim that Ruchay 2026, 2026 Simmental, XGain, or other alternatives are superior. Any alternative dataset must undergo identical rigorous leakage and visual quality audits before being considered for an operational role.

4. **Warranted Next Investigation**:
   This evidence-preservation audit proves that real-world agricultural vision datasets contain substantial domain-specific artifacts (38.3% multi-cow in ScienceDB, 54.4% occlusion in MmCows, 32.1% cutoff in SideView). Therefore, a **task-specific visual BCS quality audit of the candidate external benchmark (Ruchay et al. 2026 RGB-D BCS)** is warranted as the next candidate investigation before finalizing upstream perception caching and baseline training.

---

## 7. Artifact Registry

- **Source Manifest**: [`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv)
- **Contact Sheet Generator**: [`scripts/build_real_cattle_quality_contact_sheets.py`](file:///d:/cattle-health-monitoring-multi-task-model/scripts/build_real_cattle_quality_contact_sheets.py)
- **Contact Sheet Index**: [`docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md)
- **Visual Assets**: [`docs/audits/assets/real_cattle_visual_quality_reassessment/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/real_cattle_visual_quality_reassessment/) (42 JPEG sheets, ~25 MB total)
- **Research Log**: [`docs/research_log/2026-09-22_real_cattle_visual_quality_reassessment.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/research_log/2026-09-22_real_cattle_visual_quality_reassessment.md)
