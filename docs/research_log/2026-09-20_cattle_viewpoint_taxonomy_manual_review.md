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

## 4. Operational Strategy Assessment for Step 3 Viewpoint Generation

To determine the most scientifically defensible method for producing `coarse_viewpoint` labels during Step 3 caching across ScienceDB, MmCows, and SideViewCows2026, four candidate operational options were rigorously evaluated:

### 4.1 Option 1: Metadata / Domain Heuristics
- **Analysis**:
  - **ScienceDB & SideViewCows2026**: High-level capture geometry is constrained (chute walkaway vs milking parlor profile). In our 20-sample audits, ScienceDB was 90% rear/rear-oblique and SideView was 90% side.
  - **MmCows Failure**: MmCows metadata contains **zero** viewpoint information. Ceiling cameras view loose-housing pens where cows move, feed, lie, and lick in arbitrary 360-degree orientations.
  - **Limitation**: Even in ScienceDB and SideView, naive heuristics miss non-standard angles (e.g. ScienceDB `sample_0084` front-oblique, SideView `sample_0215` parlor entrance turn) and cannot identify ambiguous cases (`sample_0022`, `sample_0277`).
- **Verdict**: **Insufficient as a standalone solution**. Usable only as a strong domain prior for fixed-chute setups, but completely incapable of resolving MmCows.

### 4.2 Option 2: Geometric Rules (Bounding Box Aspect Ratio / Mask Moments)
- **Analysis**:
  - **Front vs. Rear Degeneracy**: Bounding box aspect ratio ($w/h$) and 2D silhouette area **cannot scientifically distinguish front vs. rear**. Both a cow facing directly toward the camera (`front`) and facing away (`rear`) present a narrow, tall/square cross-section ($w/h \approx 0.6 - 1.0$).
  - **Posture Corruption**: Lying/curled cows in MmCows completely break rigid aspect-ratio assumptions regardless of orientation.
  - **Stall Occlusions**: Vertical stall bars or feeding headlocks truncate bounding boxes, corrupting aspect ratios.
- **Verdict**: **REJECTED as a viewpoint classifier**. Aspect ratio can easily separate broadside profiles ($w/h \gg 1.0$) from axial views, but is mathematically degenerate between front and rear.

### 4.3 Option 3: Pretrained Cattle Viewpoint Resources (MOO — Multi-view Oriented Observations)
- **Forensic Verification of MOO (Grolleau et al., CVPR 2026 CV4Animals Workshop, arXiv:2603.04314)**:
  - **What MOO Actually Provides**: A large-scale **synthetic dataset** of 128,000 images across 1,000 synthetic cattle identities rendered in Blender from 128 uniformly sampled spherical viewpoints ($360^\circ$ azimuth, $-25^\circ$ to $90^\circ$ elevation) with depth maps.
  - **Availability**: Code repository at `https://github.com/TurtleSmoke/MOO`; dataset hosted as `MOO.zip` on CEA servers.
  - **Pretrained Predictor Checkpoint**: **DOES NOT EXIST**. MOO is a synthetic benchmark and pre-training resource for aerial-ground Re-ID under elevation shifts; it does **not** provide an off-the-shelf orientation predictor model or weights.
  - **Taxonomy Mapping**: Continuous azimuth $\phi$ and elevation $\theta$ could theoretically map to our taxonomy (rear: $\sim 180^\circ$, side: $\sim 90^\circ/270^\circ$, front: $\sim 0^\circ$, ambiguous: $\theta \ge 45^\circ$).
  - **Severe Domain Gap Risks**: MOO features clean, unoccluded synthetic Blender cattle in free space. Real-world target imagery contains severe domain shifts: narrow metal chutes with manure (ScienceDB), low-light CCTV with stall-bar occlusions and straw bedding (MmCows), and industrial milking parlors (SideView).
  - **Roadmap Violation**: Training an orientation regressor from scratch on 128k synthetic images violates the Phase 3 core constraint: *"Not core right now: Cattle foundation model training from scratch... Training a new segmentation model from scratch"*.
- **Verdict**: **REJECTED for Step 3 operational labeling**. MOO provides no off-the-shelf model, and synthetic-to-real transfer to cluttered farm environments is unproven and high-risk.

### 4.4 Option 4: Lightweight Viewpoint Classifier / Zero-Shot Vision-Language Foundation Model
- **Analysis**:
  - **Supervised Training Problem**: We currently possess 60 verified human labels. Training a supervised CNN/ViT classifier on 60 samples guarantees severe overfitting, memorization, and lack of generalizability.
  - **Zero-Shot Foundation Model Viability**: Modern frozen vision-language models (e.g. `open_clip`, `SigLIP`, `CLIP ViT-B/32`) possess broad semantic priors that recognize animal body orientation without task-specific fine-tuning (e.g. ranking prompts such as *"a rear view of a cow"*, *"a side profile of a cow"*, *"a front view of a cow"*, *"an occluded or ambiguous cow"*).
  - **Zero-Training Advantage**: Requires zero training, zero checkpoint downloads from untrusted sources, and introduces zero parameters.
- **Verdict**: **RECOMMENDED AS PRIMARY CANDIDATE FOR EVALUATION**.

### 4.5 Camera ID Shortcut Leakage Warning
- **Camera ID must NEVER be used as a viewpoint proxy**:
  - In MmCows, cows rotate dynamically across all 360 degrees within each camera's field of view.
  - In ScienceDB and SideView, associating camera ID with viewpoint causes downstream models to learn camera/farm/lighting shortcuts rather than anatomical orientation, directly defeating the thesis core hypothesis of eliminating background shortcuts.

---

## 5. Architectural Decisions & Action Plan

1. **Recommended Operational Strategy**:
   - **Chute Tasks (ScienceDB & SideView)**: Use domain-informed structural priors (ScienceDB = rear-dominated, SideView = side-dominated), verified by geometric aspect-ratio checks to flag off-axis anomalies.
   - **MmCows (Behavior)**: Deploy a lightweight frozen zero-shot vision-language model (e.g. CLIP / SigLIP) to classify dynamic animal orientations without training.
   - **Explicit Ambiguity Class**: Preserve `unknown / ambiguous` for heavily occluded, curled lying, or multi-cow instances.
2. **Smallest Experiment Needed Next**:
   - Evaluate zero-shot CLIP / SigLIP prompt classification on the human-verified 60-image benchmark (`artifacts/perception_audit/viewpoint_manual_review_manifest.csv`).
   - If zero-shot agreement is acceptable, adopt it for Step 3 offline caching.
   - If zero-shot foundation models fail, adopt domain priors for ScienceDB/SideView with simple heuristic clustering for MmCows.
3. **Step 2.4 & Step 2 Gate Status**:
   - Step 2.4 manual visual taxonomy review is complete and persisted.
   - The operational strategy assessment is complete.
   - Step 2.4 and overall Step 2 remain open until the small zero-shot validation experiment confirms the operational generator.

---

## 6. Artifacts & File Registry

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

## 7. Next Steps

- [x] Curate deliberately diverse 60-image viewpoint review pack across ScienceDB, MmCows, SideViewCows2026.
- [x] Generate high-resolution contact sheets and index document.
- [x] Perform visual review with ChatGPT-assisted initial labeling followed by human verification and correction.
- [x] Persist verified labels and provenance metadata in `viewpoint_manual_review_manifest.csv`.
- [x] Conduct operational strategy audit (heuristics vs geometry vs MOO vs zero-shot classifier).
- [ ] Run smallest experiment: benchmark zero-shot CLIP/SigLIP prompt classification on the 60 verified review samples.
- [ ] Conclude Step 2.4 and decide Gate 2 feasibility status.
