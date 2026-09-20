# state.md — Current Workspace State

## Phase 3 Roadmap Status
- **Status**: CANONICAL / LOCKED FOR EXECUTION (`phase3_canonical_roadmap.md`)
- **Core Question**: Can cattle-centered visual representations (localization, soft masks, anatomy/pose, viewpoint) reduce shortcut learning and improve robustness across BCS, Behavior, and Re-ID compared with generic RGB representations?
- **Single Source of Truth**: [phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md) (also mirrored at [docs/phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/phase3_canonical_roadmap.md))

## Active Goals & Todo (STEP 1: COMPLETE | GATE 1: CLEARED)
- **Immediate next action:** STEP 2 — Cattle-perception feasibility audit (`docs/audits/phase3_perception_feasibility.md`).
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
- [x] Download & restore ScienceDB Cattle BCS raw images (53,566 images across 5 classes restored via 24-thread fast downloader & 7-Zip; index validated)
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
- [ ] STEP 2: Cattle-perception feasibility audit (`docs/audits/phase3_perception_feasibility.md`)
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
  - **Primary External Validation**: CBVD-5 (larger-herd external test)
  - **Optional External**: CVB, XGain, Simmental 2026 (for compatible label intersections only)
- **3. Individual Cow Identification / Re-Identification (Re-ID)**:
  - **Primary (Approved Contingency)**: SideViewCows2026 (110 cows, 80,260 images + 80,260 binary masks, parlor/barn/snapshots nested settings, >9 months span; replaces MultiCam under approved contingency; COMPLETE & VERIFIED)
  - **Primary External Validation (Longitudinal)**: BECA-L (103 beef cattle, 12,172 images, 134 dates over 7+ months, top-down dorsal view, 3 cowsheds)
  - **External Scale Stress**: BECA-D (5,661 beef cattle, 16,889 images, 3 shots/cow)
  - **Contingency-Excluded Intended Primary**: MultiCamCows2024 (90 cows, 101,329 images; blocked upstream by persistent server connection resets; preserved in project records)
  - **Legacy Baseline**: OpenCows2020 (retained strictly for backward comparability; contiguous frame-index heuristic + duplicate harmonization protocol verified)
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
  - Discrepancy Note: Discrepancy fully resolved (`docs/research_log/2026-09-20_dryad_bcs_discrepancy_audit.md`). Older ~5,923 count omitted Class 7 (17 images from Cow_52). Class 7 is 100% authentic Criollo beef data on 1–9 Wagner scale. Zero cross-cow duplicate leakage.
- **MmCows**:
  - Scientific Role: Primary Behavior dataset
  - Local Status: **Raw cropped data present locally** (`datasets/behavior/mmcows/cropped_bboxes/`).
  - Physical Counts: 213,686 indexed behavior crops in `behaviors/` (100% valid/resolving). Total local JPG count is 427,390 due to additional `lying/` (83,620) and `standing/` (130,084) crop folders. Raw source videos and archive zip were previously purged. Do not redefine the behavior dataset as 427,390 samples.
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
  - Scientific Role: Legacy Re-ID benchmark only
  - Local Status: **Locally present** (`datasets/id/opencow2020-DatasetNinja/`).
  - Physical Counts: 4,736 images across 46 identities (4,240 in `identification-train/img/`, 496 in `identification-test/img/`).
  - Protocol Status: `datasets/id/opencow2020/manifest.csv` (4,736 rows), `train.csv` (3,586), `val.csv` (654), `test.csv` (496 official preserved), `split_report.md`, and updated `id_index.csv` (4,736 rows). 0 exact-duplicate overlap; frame-index adjacency crossings reduced from 1,023 to 48. True tracklet/temporal leakage cannot be verified because provenance is unavailable.
- **MultiCamCows2024**:
  - Scientific Role: Intended primary Re-ID dataset (Contingency-excluded)
  - Local Status: **NOT present locally**. 0 files/archives.
  - Upstream Status: Official download attempt blocked upstream (`data.bris.ac.uk` connection reset verified locally and on Modal cloud). Formally replaced by SideViewCows2026 under approved contingency (2026-09-20).
- **Ruchay 2026**: Metadata and 25,700-sample manifest verified locally in `datasets/bcs/external/ruchay2026/`; raw 77.74 GB RGB-D zip archives remain on Zenodo (DOI: 10.5281/zenodo.20290988).
- **CBVD-5**: AVAILABLE locally (`datasets/behavior/external/cbvd5/`; 887 videos, 206,100 frames, 5,322 annotated label frames across 107 cows).
- **CattleEyeView / SuperAnimal / MOO**: Not present locally.
- **CattleLameness**: Locally present (50 MP4s in `CattleLameness/Data/` + 9,950 frames in `frames/`). Historical Phase 2 material only; remains strictly excluded from core Phase 3 MTL.

> [!IMPORTANT]
> **Multi-Environment Awareness**: Physical dataset availability may differ across machines and execution environments (e.g. this local laptop vs. Modal cloud volumes vs. the BRACU Lab Research PC with RTX 5090). Future agents MUST inspect physical files on disk before assuming a dataset is available locally.

## Last Session (Convo 27375138-e032-457f-a2a6-753e72f4a342)
- Built and executed `scripts/repair_sciencedb_splits.py` to repair the ScienceDB Cattle BCS dataset split. Clustered overlapping 1-frame-shifted video bursts (including GS_1818/GS_1823) into 5,653 connected burst groups via Disjoint Set Union (threshold: dHash/aHash <= 2, pixel MAE <= 5.0). Generated 70/15/15 stratified split (train: 37,045, val: 8,481, test: 8,040). Verified 0 exact cross-duplicates and 0 cross-burst overlap. Promoted to canonical `datasets/bcs/sciencedb/`.
- Conducted forensic suitability audit of accessible Re-ID datasets (`docs/research_log/2026-09-20_multicam_contingency_assessment.md`). Formulated Strategy 1 (SideViewCows2026 as primary Re-ID, BECA-L as external longitudinal, BECA-D as external scale stress).
- User formally APPROVED the MultiCam contingency on 2026-09-20. Updated canonical roadmap and dataset registry roles.
- Engineered `scripts/build_sideview_reid_protocols.py` with 7-stage clean progress UI. Recovered 3,604 discrete recording sessions (`dt <= 60s`). Generated and verified 4 canonical evaluation protocols (Cross-setting, Longitudinal, Open-set 77/11/22 cows, Closed-set 70/15/15 parlor + barn/snapshots). Verified exact duplicate uniqueness, identity-disjointness, and session-safe separation with no leakage detected under the implemented checks; perceptual near-duplicate audit on 10,094 sampled session anchors found min distance of 7 bits. Promoted deliverables to `datasets/id/sideviewcows2026/`.
- Cleared Gate 1! Completed Step 1 (Data Registry and Clean Splits).
- Built deterministic manual visual verification generator (`scripts/build_manual_dataset_visual_verification.py`). Generated 48 compact visual checks across ScienceDB (16), MmCows (16), and SideViewCows2026 (16 with 3-panel RGB|Mask|Overlay composites) at `docs/audits/phase3_manual_dataset_visual_verification.md` with 1.4 MB asset thumbnails in `docs/audits/assets/manual_dataset_verification/`. Tested and verified.
- Conducted exhaustive forensic comparison of MmCows vs. CBVD-5 for Primary Behavior role (`scripts/audit_behavior_dataset_candidates.py`). Proved CBVD-5 has ZERO biological cow IDs (actor ID = 1 dummy value), median crop size 156x167px (2.5x smaller than MmCows), lacks walking/licking, and official AVA splits leak 100% of val/test videos. Built 14-pair side-by-side visual comparison guide (`docs/audits/phase3_behavior_dataset_manual_comparison.md`) and published comprehensive research log (`docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md`). Decisively recommended Option A: Retain MmCows as Primary and CBVD-5 as External Validation. Step 1 remains in progress pending user sign-off.

## Current Blockers & Notes
- **STEP 1 IS 100% COMPLETE**.
- **GATE 1 IS CLEARED & LOCKED**.
- Current immediate position is **STEP 2 — Cattle-Perception Feasibility Audit**.
- Next deliverable: `docs/audits/phase3_perception_feasibility.md`.
- All primary splits locked:
  - ScienceDB BCS: Repaired and locked; no leakage detected under implemented checks (`datasets/bcs/sciencedb/`).
  - MmCows behavior: Grouped evaluation locked; no leakage detected under implemented checks (`datasets/behavior/mmcows/folds/`).
  - SideViewCows2026 Re-ID: Protocols locked; no leakage detected under implemented checks (`datasets/id/sideviewcows2026/`).
  - OpenCows2020: Contiguous heuristic rebuild locked as legacy benchmark.
- Antigravity sync rule: changes mirrored to `D:\custom-antigravity`.

