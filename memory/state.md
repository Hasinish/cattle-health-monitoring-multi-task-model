# state.md — Current Workspace State

## Phase 3 Roadmap Status
- **Status**: CANONICAL / LOCKED FOR EXECUTION (`phase3_canonical_roadmap.md`)
- **Core Question**: Can cattle-centered visual representations (localization, soft masks, anatomy/pose, viewpoint) reduce shortcut learning and improve robustness across BCS, Behavior, and Re-ID compared with generic RGB representations?
- **Single Source of Truth**: [phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md) (also mirrored at [docs/phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/phase3_canonical_roadmap.md))

## Active Goals & Todo (STEP 1: COMPLETE | GATE 1: CLEARED | STEP 2.1: COMPLETE | STEP 2.2: COMPLETE | STEP 2.3: COMPLETE | STEP 2.4 MANUAL REVIEW: COMPLETE)
- **Immediate next action:** STEP 2.4 / GATE 2 — Decide downstream viewpoint operational strategy and conclude Step 2 perception feasibility audit.
- [x] STEP 2.4 (Manual Taxonomy Review): Cattle viewpoint manual visual taxonomy review completed across ScienceDB, MmCows, and SideViewCows2026 (`artifacts/perception_audit/viewpoint_manual_review_manifest.csv`; `docs/audits/phase3_viewpoint_visual_review_index.md`; 60-image deliberately diverse review pack; user verified/corrected after ChatGPT initial labeling; ScienceDB strongly rear/rear-oblique [90%], MmCows broad mixture with highest ambiguity [15%], SideView overwhelmingly side-view [90%]; confirmed coarse taxonomy `rear, rear-oblique, side, front-oblique, front, unknown / ambiguous` is visually usable; camera ID is not viewpoint; no model trained; Step 2.4 / Step 2 remain open; `docs/research_log/2026-09-20_cattle_viewpoint_taxonomy_manual_review.md`)
- [x] STEP 2.3: Cattle pose / keypoint feasibility audit completed across ScienceDB, MmCows, and SideViewCows2026 (`docs/audits/phase3_perception_feasibility.md`; evaluated official DeepLabCut SuperAnimal-Quadruped HRNet-W32 and ResNet-50 on RT-DETR-L target crops; 81.0% operational success [243/300], 13.3% pose_detector_failure [40/300], 5.7% upstream_localization_failure [17/300], 0% crashes; human visual review by user with ChatGPT-assisted organization [N=60 across 30 samples in `artifacts/perception_audit/pose_manual_review.csv`]: documented selection bias [ScienceDB BCS 3.25 only, MmCows Lying only, SideView parlor only; not extrapolated to unreviewed classes/settings]; in reviewed ScienceDB subset, rear-view outputs judged visually bad / anatomically unreliable (`clearly_wrong`) and missing BCS landmarks, not recommended for downstream BCS; in reviewed MmCows subset, lying outputs judged visually bad / anatomically unreliable (`clearly_wrong`) with 50% localization misses; SideView parlor was the only group that looked genuinely plausible, promising for Step 6 ablation [77.2% mask containment sanity check] but not proven useful yet; ResNet-50 provisional candidate due to higher raw confidence and mask containment [does not establish higher pose accuracy]; `docs/research_log/2026-09-20_cattle_pose_feasibility_audit.md`)
- [x] STEP 2.2: Cattle segmentation feasibility audit completed across ScienceDB, MmCows, and SideViewCows2026 (`docs/audits/phase3_perception_feasibility.md`; evaluated RT-DETR-L -> SAM 2.1 small [Mean IoU 0.9216 / Dice 0.9530 on SideView GT; 0.0252 delta from Oracle GT box; 93/100 segmented on ScienceDB, 90/100 on MmCows] vs YOLO26s-seg [Mean IoU 0.8660, 38% missed on ScienceDB, 27% on MmCows]; confirmed pretrained segmentation feasible without fine-tuning; docs/research_log/2026-09-20_cattle_segmentation_feasibility_audit.md)
- [x] STEP 2.1: Cattle detection / localization feasibility audit completed across ScienceDB, MmCows, and SideViewCows2026 (`docs/audits/phase3_perception_feasibility.md`; evaluated YOLOv8s, Faster R-CNN v2, RT-DETR-L; proved YOLOv8s 37% non-detection rate on rear-view chute and tight crops; designated RT-DETR-L [94.3% raw detection rate, 94ms latency] as provisional primary candidate for Step 2.2; docs/research_log/2026-09-20_cattle_localization_feasibility_audit.md)
- [x] Download & restore 213,686 MmCows behavior images via Hugging Face (213,686 indexed crops valid in `behaviors/`; 427,390 total local JPGs; raw videos purged)
- [x] Download & index OpenCows2020 (4,736 images across 46 classes via Kagglehub - designated Legacy Baseline)
- [x] Purge 36+ GB raw behavior videos, zip archives, and cache to reclaim local disk space
- [x] Complete forensic CattleLameness dataset audit & leakage investigation (`docs/audits/cattle_lameness_audit_report.md`)
- [x] Establish research log hub (`docs/research_log/`) and automated persistence rules (`.agents/rules/research_logging.md`, `AGENTS.md`)
- [x] Finalize P3 task scope: BCS, Behavior, Cow ID/Re-ID; lameness removed from primary P3 MTL model
- [x] Formulate & adopt 13-step Phase 3 Canonical Roadmap (`phase3_canonical_roadmap.md`)
- [x] Conduct read-only physical filesystem inventory audit (`docs/research_log/2026-09-20_local_dataset_inventory_audit.md`)
- [x] Build canonical dataset registry (`datasets/dataset_registry.csv` generated; 13 datasets, 28 fields, local physical statuses cataloged)
- [x] Document MultiCamCows2024 upstream blocker and formally adopt SideViewCows2026 contingency (`docs/research_log/2026-09-20_multicam_contingency_assessment.md`)
- [x] Download & restore ScienceDB BCS raw images (53,566 images across 5 classes restored via 24-thread fast downloader & 7-Zip; index validated)
- [x] Validate ScienceDB identity parser and repair burst-group split (`datasets/bcs/sciencedb/`; 5,653 repaired burst groups, 0 exact duplicates, 0 cross-burst leakage; 100% verified)
- [x] Rebuild MmCows grouped evaluation protocol with time-block / multi-view protection (`datasets/behavior/mmcows/folds/`; 213,686 crops, 16 cows, canonical split + 4-fold GroupKFold suite, 0 leakage)
- [x] Audit and rebuild OpenCows2020 legacy Re-ID evaluation protocol (`datasets/id/opencow2020/`; 4,736 images, 46 cows; official test 496 preserved; train: 3,586, val: 654; contiguous frame-index heuristic + duplicate harmonization; 0 exact-duplicate overlap; true tracklet/temporal leakage unrecoverable from provenance)
- [x] Build and verify deterministic SideViewCows2026 primary Re-ID protocols (Protocol A: cross-setting parlor-to-barn/snapshots; Protocol B: longitudinal; Protocol C: open-set; Protocol D: closed-set) and run duplicate/near-duplicate audit (0 exact duplicates, min perceptual distance 7 bits; Gate 1 CLEARED)
- [x] Retrieve & index Ruchay 2026 metadata (Zenodo record 20290988 verified; 25,700 samples, 1,025 cows; manifest generated in datasets/bcs/external/ruchay2026/; 77.74 GB raw archives on Zenodo)
- [x] Audit Dryad BCS local count/class discrepancy (5,940 TIFFs verified across classes 2–7; older ~5,923 count omitted Class 7 [17 imgs from Cow_52]; 54 biological cows census; manifest generated; bcs_index.csv updated)
- [x] Download/index SideViewCows2026: 80,260 images + 80,260 binary segmentation masks across 110 cows (snapshots: 607, parlor: 54,393, barn: 25,260) 100% downloaded and extracted in `datasets/id/external/sideviewcows2026/`.
- [x] Download/index BECA-D / BECA-L: 29,061 images across 5,661 cows (BECA-D) and 103 longitudinally tracked cows (BECA-L) 100% downloaded and extracted in `datasets/id/external/beca/`.
- [x] Download & verify CBVD-5: 887 videos, 206,100 frames, 5,322 annotated label frames across 107 cows 100% downloaded and verified in `datasets/behavior/external/cbvd5/`.
- [x] Update canonical `datasets/dataset_registry.csv` to mark SideViewCows2026, BECA-D, BECA-L, and CBVD-5 as `AVAILABLE`.
- [x] Add multi-account Modal billing monitor and live dashboard generator (`scripts/billing_monitor.py`, `scripts/modal_billing.py`, `billing_monitor.py`, `BILLING.md`).
- [x] Run automated duplicate / near-duplicate audit across all primary datasets (ScienceDB: 0 exact, 88,944 near-duplicates, overlapping passage vulnerability identified and repaired via burst clustering; MmCows: 0 exact, 0 cow overlap, clean; OpenCows: 0 exact, 1,239 near-duplicates, legacy baseline only; docs/research_log/2026-09-20_phase3_duplicate_nearduplicate_audit.md)
- [x] Correct ScienceDB passage-disjoint split into connected burst blocks (`scripts/repair_sciencedb_splits.py`; 5,653 repaired burst groups; 0 exact duplicates, 0 cross-burst overlap; 100% verified)
- [x] Generate manual human visual-verification pack for primary Phase 3 datasets (ScienceDB 16, MmCows 16, SideView 16; 48 checks + 3-panel composites generated deterministically via `scripts/build_manual_dataset_visual_verification.py`; `docs/audits/phase3_manual_dataset_visual_verification.md`)
- [x] Audit MmCows vs. CBVD-5 for Primary Behavior role (`docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md`; 14-pair manual side-by-side pack at `docs/audits/phase3_behavior_dataset_manual_comparison.md`; recommended Option A: keep MmCows Primary, preserve CBVD-5 as External Validation)
- [x] Complete multimodal agent visual inspection of 20 MmCows and 20 CBVD-5 samples (`docs/audits/phase3_behavior_agent_visual_inspection.md`; confirmed 19/20 MmCows usable, exposed CBVD-5 wide-angle crop resolution deficit [median 156px vs 390px] and temporal label inconsistency; visually validated prior audit claims; recommended retaining Option A)
- [ ] STEP 2: Cattle-perception feasibility audit (`docs/audits/phase3_perception_feasibility.md` — Step 2.1 Localization COMPLETE; Step 2.2 Segmentation COMPLETE; Step 2.3 Pose COMPLETE; Step 2.4 Viewpoint manual visual taxonomy review COMPLETE, overall Step 2.4 / Step 2 pending)
- [ ] STEP 3: Cache upstream cattle information (bbox, soft_mask, pose_coords, viewpoint)
- [ ] STEP 4: Clean RGB single-task baselines
- [ ] STEP 5: Localization / segmentation ablation
- [ ] STEP 6: Anatomy / pose ablation
- [ ] STEP 7: Viewpoint ablation
- [ ] STEP 8: Temporal Behavior experiments (frame -> avg pool -> TCN -> GRU/LSTM -> pose seq -> fusion)
- [ ] STEP 9: Build consolidated task-conditioned cattle-centered P3 architecture
- [ ] STEP 10: Cross-domain / robustness evaluation
- [ ] STEP 11: Revisit sharing / MTL (Single vs Hard vs Partial vs Adapters vs PCGrad/GradNorm)
- [ ] STEP 12: Optional cattle-specific pretraining (stretch goal)
- [ ] STEP 13: Final repeated runs (3 seeds) + thesis tables
- [ ] Prepare P3 draft submission by September 26

## Canonical Task & Dataset Stack (Scientific Roles)
- **1. Body Condition Scoring (BCS)**:
  - **Primary (In-Domain)**: ScienceDB (53,566 RGB images across 5 classes: 3.25–4.25 from 5,653 repaired connected burst groups; burst-group disjoint with no leakage detected under the implemented checks; true cow IDs not released by publisher)
  - **Primary External Validation**: Ruchay et al. RGB-D BCS (2026) (multi-breed, wide BCS range, different camera geometry)
  - **Secondary External**: Dryad BCS (5,940 DGE images across discrete classes '2'-'7')
- **2. Behavior Recognition**:
  - **Primary (In-Domain)**: MmCows (213,686 bounding-box crops across 7 active classes from 16 cows, multi-camera CCTV)
  - **Primary External Validation**: CBVD-5 (887 videos, 206,100 frames across 107 cows; reserved for external validation)
- **3. Individual Cow Identification / Re-ID**:
  - **Primary (In-Domain)**: SideViewCows2026 (80,260 images + 80,260 binary masks, 110 cows; 4 canonical protocols: cross-setting, longitudinal, open-set, closed-set; Gate 1 CLEARED)
  - **Primary External Longitudinal Validation**: BECA-L (12,172 images across 103 cows, 134 dates over 7+ months)
  - **Secondary External Scale / Stress Validation**: BECA-D (16,889 images across 5,661 cows)
  - **Legacy Baseline (Historical / Reference Only)**: OpenCows2020 (4,736 images across 46 cows)
- **Lameness Status**: Excluded from primary Phase 3 MTL; CattleLameness retained for historical P2 audit only.

## Current Local Availability on this Machine (Physical Verification Baseline)
- **ScienceDB BCS**:
  - Scientific Role: Primary BCS dataset
  - Local Status: **AVAILABLE & LOCKED**. Fully restored, repaired, and verified locally (`datasets/bcs/sciencedb/`).
  - Physical Counts: 53,566 RGB images across 5 classes: 3.25 (7,536), 3.50 (13,256), 3.75 (14,255), 4.00 (12,556), 4.25 (5,963).
  - Repaired Protocol: `datasets/bcs/sciencedb/` (5,653 repaired burst groups, train: 37,045, val: 8,481, test: 8,040). Verified 0 exact cross-duplicates and 0 cross-burst overlap. Master index `datasets/bcs/sciencedb_bcs_index.csv` updated.
- **Dryad BCS**:
  - Scientific Role: Secondary external BCS validation
  - Local Status: **AVAILABLE & FULLY INDEXED** (`datasets/bcs/dryad_bcs/Total_sorted_DGE_images/`).
  - Physical Counts: 5,940 TIFF files verified across class folders 2 through 7 (100% valid 224x224 RGB DGE images; 54 biological cows, 148 sessions).
  - Index Status: `datasets/bcs/dryad/manifest.csv` (5,940 rows), `cow_audit.csv` (54 cows), and `datasets/bcs/bcs_index.csv` (5,940 rows) fully populated and verified.
- **MmCows**:
  - Scientific Role: Primary Behavior dataset
  - Local Status: **Raw cropped data present locally** (`datasets/behavior/mmcows/cropped_bboxes/`).
  - Physical Counts: 213,686 indexed behavior crops in `behaviors/` (100% valid/resolving). Total local JPG count is 427,390 due to additional `lying/` (83,620) and `standing/` (130,084) crop folders.
  - Manifest & Protocol: `datasets/behavior/mmcows/manifest.csv` (213,686 rows), `provenance_audit.csv` (16 cows), canonical splits (`train.csv`, `val.csv`, `test.csv`), and 4-Fold GroupKFold suite (`folds/fold_[0-3].csv`) verified with no cross-split leakage detected under the implemented checks.
- **SideViewCows2026**:
  - Scientific Role: Primary Re-ID dataset (Approved Contingency)
  - Local Status: **AVAILABLE** (`datasets/id/external/sideviewcows2026/`).
  - Physical Counts: 80,260 images + 80,260 binary segmentation masks across 110 biological cows (parlor: 54,393, barn: 25,260, snapshots: 607).
  - Protocol Status: **COMPLETE & VERIFIED** (`datasets/id/sideviewcows2026/`). 4 canonical protocols generated and audited: `protocol_cross_setting.csv`, `protocol_longitudinal.csv`, `protocol_open_set.csv`, `protocol_closed_set.csv`, `leakage_audit.csv`, and `split_report.md`. Exact duplicate, identity-disjointness, mask matching, and session separation verified with no leakage detected under the implemented checks; perceptual near-duplicate audit on 10,094 sampled session anchors found min distance of 7 bits.
- **BECA-L**:
  - Scientific Role: Primary external longitudinal Re-ID validation benchmark
  - Local Status: **AVAILABLE** (`datasets/id/external/beca/BECA-L/`).
  - Physical Counts: 12,172 images across 103 beef cattle tracked over 7+ months (134 dates) across 3 cowsheds.
- **BECA-D**:
  - Scientific Role: External large-scale / population stress benchmark
  - Local Status: **AVAILABLE** (`datasets/id/external/beca/BECA-D/`).
  - Physical Counts: 16,889 images across 5,661 beef cattle (16,083 train with 3 imgs/cow, 806 val).
- **OpenCows2020**:
  - Scientific Role: Legacy Re-ID baseline
  - Local Status: **AVAILABLE** (`datasets/id/opencows2020/`).
  - Physical Counts: 4,736 images across 46 cows (3,586 train, 654 val, 496 test).
- **CBVD-5**: AVAILABLE locally (`datasets/behavior/external/cbvd5/`; 887 videos, 206,100 frames, 5,322 annotated label frames across 107 cows).
- **CattleEyeView / SuperAnimal / MOO**: Not present locally.
- **CattleLameness**: Locally present (50 MP4s in `CattleLameness/Data/` + 9,950 frames in `frames/`). Historical Phase 2 material only; remains strictly excluded from core Phase 3 MTL.

> [!IMPORTANT]
> **Multi-Environment Awareness**: Physical dataset availability may differ across machines and execution environments (e.g. this local laptop vs. Modal cloud volumes vs. the BRACU Lab Research PC with RTX 5090). Future agents MUST inspect physical files on disk before assuming a dataset is available locally.

## Last Session (Convo e78aa1ac-ddc7-4c32-8bab-f25894ade0df)
- Completed Step 2.4 (Manual Visual Taxonomy Review): Built deliberately diverse 60-image viewpoint review pack across ScienceDB (20), MmCows (20), and SideViewCows2026 (20).
- Generated 6 high-resolution contact sheets (`docs/audits/assets/viewpoint_visual_review/`) and visual review index (`docs/audits/phase3_viewpoint_visual_review_index.md`).
- Completed visual review with ChatGPT-assisted initial proposals and user verification/corrections:
  - ScienceDB: 10 rear, 8 rear-oblique, 1 front-oblique, 1 unknown / ambiguous (strongly rear/rear-oblique dominated).
  - MmCows: 9 side, 6 rear-oblique, 1 rear, 1 front-oblique, 3 unknown / ambiguous (broadest mixture, highest ambiguity due to stall bars/rails/top-down CCTV).
  - SideViewCows2026: 18 side, 1 front-oblique, 1 unknown / ambiguous (overwhelmingly side-view dominated; 1 front-oblique parlor entry, 1 multi-cow barn alley ambiguity).
  - Overall: 27 side, 14 rear-oblique, 11 rear, 3 front-oblique, 5 unknown / ambiguous, 0 front.
- Persisted verified labels in `artifacts/perception_audit/viewpoint_manual_review_manifest.csv` with status `human_verified` and accurate provenance documentation.
- Updated `docs/audits/phase3_perception_feasibility.md` with Section 4, created research log `docs/research_log/2026-09-20_cattle_viewpoint_taxonomy_manual_review.md`, and indexed in `docs/research_log/README.md`.
- Maintained constraints: no model trained, no weights downloaded, Step 2.4 and overall Step 2 remain open.

## Current Blockers & Notes
- **STEP 1 IS 100% COMPLETE & LOCKED (Gate 1 Cleared)**.
- **STEP 2.1 (Localization Feasibility) IS 100% COMPLETE**.
- **STEP 2.2 (Segmentation Feasibility) IS 100% COMPLETE**.
- **STEP 2.3 (Pose Feasibility) IS 100% COMPLETE**.
- **STEP 2.4 (Manual Visual Taxonomy Review) IS COMPLETE; overall Step 2.4 / Step 2 remain open**.
- Deliverable updated: `docs/audits/phase3_perception_feasibility.md` (Localization, Segmentation, Pose, and Viewpoint Manual Review documented).
- Antigravity sync rule: changes mirrored to `D:\custom-antigravity`.
