# Research Log: Cattle Viewpoint Taxonomy & Manual Review Feasibility Audit (Step 2.4)

**Date**: 2026-09-20  
**Phase**: Phase 3 (Pretrained Perception Feasibility Audit — Step 2.4 Viewpoint / Orientation Feasibility)  
**Deliverable Document**: [docs/audits/phase3_perception_feasibility.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_perception_feasibility.md)  
**Visual Review Index**: [docs/audits/phase3_viewpoint_visual_review_index.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_viewpoint_visual_review_index.md)  
**Manifest**: [artifacts/perception_audit/viewpoint_manual_review_manifest.csv](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_manual_review_manifest.csv)  
**Script**: [scripts/build_viewpoint_contact_sheets.py](file:///d:/cattle-health-monitoring-multi-task-model/scripts/build_viewpoint_contact_sheets.py)  

---

## 1. Executive Summary

This study conducted the initial visual feasibility audit of cattle body viewpoint categories across the three primary Phase 3 datasets (ScienceDB, MmCows, SideViewCows2026). A deliberately diverse 60-image manual review pack (20 images per dataset) was curated across BCS classes, farm sources, behavior categories, surveillance cameras, and capture subsets to stress-test candidate viewpoint labels (`rear`, `rear-oblique`, `side`, `front-oblique`, `front`, and `unknown / ambiguous`). Human visual verification (following ChatGPT-assisted initial proposals) proved the coarse taxonomy is visually usable and consistently annotatable across all three domains, provided that an explicit `unknown / ambiguous` category is retained to avoid forced misclassification under occlusion, partial crops, or top-down CCTV angles.

---

## 2. Context & Motivation

Phase 3 proposes conditioning multi-task representations on cattle viewpoint or incorporating a shared orientation head to align representations across disparate tasks (rear-dominated BCS vs. side-dominated Re-ID vs. multi-angle Behavior). 

Before training any viewpoint classifier or committing architecture capacity, two prerequisite questions must be answered:
1. Are the proposed coarse viewpoint categories (`rear`, `rear-oblique`, `side`, `front-oblique`, `front`, `unknown / ambiguous`) visually definable and consistently distinguishable across our actual dataset imagery?
2. Can camera ID be used as a proxy for viewpoint, or is physical animal orientation independent of camera position?

To prevent the selection-bias flaw observed in Step 2.3 (where sequential sampling restricted review to single classes), this audit curated a balanced, deliberately diverse subset across available metadata dimensions.

---

## 3. Forensic Findings & Visual Review Data

### 3.1 Sampling Strategy & Diversity

60 samples were selected deterministically from `artifacts/perception_audit/sample_manifest_expanded.csv`:
- **ScienceDB (20)**: 4 samples across each of the 5 BCS classes (`3.25`, `3.50`, `3.75`, `4.00`, `4.25`), covering all 3 farm sources (`GS_Gansu`: 8, `YM_Farm2`: 8, `STEREO_Farm3`: 4).
- **MmCows (20)**: Covers all 7 behaviors (`Lying`: 3, `Standing`: 3, `Feeding head down`: 3, `Feeding head up`: 3, `Walking`: 3, `Drinking`: 3, `Licking`: 2), spanning 14 unique cows and all 4 surveillance camera angles (`Cam 1`, `Cam 2`, `Cam 3`, `Cam 4`).
- **SideViewCows2026 (20)**: Covers `Parlor` (8), `Barn` (8), and `Snapshots` (4) across multiple cows and sessions.

*Caveat*: This sample is a deliberately diverse manual feasibility subset; it is not a statistical census of the full datasets.

### 3.2 Verified Viewpoint Distribution (N=60)

| Dataset | Total Samples | `rear` | `rear-oblique` | `side` | `front-oblique` | `front` | `unknown / ambiguous` |
|---|---|---|---|---|---|---|---|
| **ScienceDB** | 20 | 10 (50.0%) | 8 (40.0%) | 0 (0.0%) | 1 (5.0%) | 0 (0.0%) | 1 (5.0%) |
| **MmCows** | 20 | 1 (5.0%) | 6 (30.0%) | 9 (45.0%) | 1 (5.0%) | 0 (0.0%) | 3 (15.0%) |
| **SideViewCows2026** | 20 | 0 (0.0%) | 0 (0.0%) | 18 (90.0%) | 1 (5.0%) | 0 (0.0%) | 1 (5.0%) |
| **Overall** | **60** | **11 (18.3%)** | **14 (23.3%)** | **27 (45.0%)** | **3 (5.0%)** | **0 (0.0%)** | **5 (8.3%)** |

### 3.3 Key Findings by Dataset

1. **ScienceDB (BCS)**:
   - Strongly rear/rear-oblique dominated (18/20, 90.0%). The standard passage chute setup captures cows walking away from the sensor.
   - Non-standard orientations exist: `sample_0084` is a front-oblique capture from an alternate farm camera.
   - Ambiguity arises: `sample_0022` was judged `unknown / ambiguous` due to severe chute-gate occlusion and a tight partial crop.

2. **MmCows (Behavior)**:
   - Broadest viewpoint distribution across all categories (side: 9, rear-oblique: 6, rear: 1, front-oblique: 1).
   - Highest ambiguity rate (3/20, 15.0%):
     - `sample_0107`: Lying cow heavily occluded by stall bars and rails.
     - `sample_0140`: Feeding cow obscured by headlocks / feeding stanchion.
     - `sample_0190`: Licking behavior in dark cubicle with steep top-down overhead perspective.
   - Confirms that overhead CCTV cameras yield arbitrary animal orientations as cows navigate loose housing.

3. **SideViewCows2026 (Re-ID)**:
   - Overwhelmingly side-view dominated (18/20, 90.0%) in milking parlors and barn walkways.
   - Non-side examples verified: `sample_0215` captured at parlor entry exhibits a front-oblique angle.
   - Ambiguity verified: `sample_0277` is ambiguous due to multiple overlapping cows in a barn alley.

4. **Camera ID vs. Physical Viewpoint**:
   - **Camera ID must NOT be treated as viewpoint**: While fixed camera placements bias orientations (ScienceDB towards rear, SideView towards lateral profile), individual animal movement introduces significant angular variance. In MmCows, any given camera captures all viewpoints over time.

---

## 4. Architectural Decisions & Action Plan

1. **Taxonomy Retention**: The 6-class taxonomy (`rear`, `rear-oblique`, `side`, `front-oblique`, `front`, `unknown / ambiguous`) is validated as visually workable.
2. **Explicit Ambiguity Handling**: Any automated viewpoint predictor or conditioning mechanism must support `unknown / ambiguous` or output continuous orientation confidence to handle occluded stall scenes.
3. **No Model Training in Step 2.4 (Initial Phase)**: In accordance with the canonical roadmap, no classifier was trained and no pre-trained weights were downloaded.
4. **Step 2.4 Status**: Step 2.4 manual visual taxonomy review is completed and persisted. Step 2.4 and overall Step 2 remain open pending decisions on automated viewpoint labeling strategies.

---

## 5. Artifacts & File Registry

- **Manifest**: [artifacts/perception_audit/viewpoint_manual_review_manifest.csv](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_manual_review_manifest.csv) (60 rows, verified labels, provenance notes)
- **Visual Review Index**: [docs/audits/phase3_viewpoint_visual_review_index.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_viewpoint_visual_review_index.md)
- **Contact Sheets**:
  - `docs/audits/assets/viewpoint_visual_review/sciencedb_viewpoint_page1.jpg`
  - `docs/audits/assets/viewpoint_visual_review/sciencedb_viewpoint_page2.jpg`
  - `docs/audits/assets/viewpoint_visual_review/mmcows_viewpoint_page1.jpg`
  - `docs/audits/assets/viewpoint_visual_review/mmcows_viewpoint_page2.jpg`
  - `docs/audits/assets/viewpoint_visual_review/sideviewcows2026_viewpoint_page1.jpg`
  - `docs/audits/assets/viewpoint_visual_review/sideviewcows2026_viewpoint_page2.jpg`
- **Generation Script**: [scripts/build_viewpoint_contact_sheets.py](file:///d:/cattle-health-monitoring-multi-task-model/scripts/build_viewpoint_contact_sheets.py)
- **Audit Section**: [docs/audits/phase3_perception_feasibility.md#4-step-24-cattle-viewpoint--orientation-feasibility-audit-manual-visual-taxonomy-review](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_perception_feasibility.md)

---

## 6. Next Steps

- [x] Curate deliberately diverse 60-image viewpoint review pack across ScienceDB, MmCows, SideViewCows2026.
- [x] Generate high-resolution contact sheets and index document.
- [x] Perform visual review with ChatGPT-assisted initial labeling followed by human verification and correction.
- [x] Persist verified labels and provenance metadata in `viewpoint_manual_review_manifest.csv`.
- [ ] Determine downstream viewpoint operational strategy (whether to train a lightweight classifier, use geometric bounding box aspect ratios, or rely on task-specific domain assumptions).
- [ ] Conclude Step 2.4 and decide Gate 2 feasibility status.
