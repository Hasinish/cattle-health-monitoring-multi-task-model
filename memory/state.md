# state.md — Current Workspace State

## Phase 3 Roadmap Status
- **Status**: CANONICAL / LOCKED FOR EXECUTION (`phase3_canonical_roadmap.md`)
- **Core Question**: Can cattle-centered visual representations (localization, soft masks, anatomy/pose, viewpoint) reduce shortcut learning and improve robustness across BCS, Behavior, and Re-ID compared with generic RGB representations?
- **Single Source of Truth**: [phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md) (also mirrored at [docs/phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/phase3_canonical_roadmap.md))

## Active Goals & Todo (STEP 1: Data Registry & Clean Splits)
- [x] Download & restore 213,686 MmCows behavior images via Hugging Face (213,686 indexed crops valid in `behaviors/`; 427,390 total local JPGs; raw videos purged)
- [x] Download & index OpenCows2020 (4,736 images across 46 classes via Kagglehub - designated Legacy Baseline)
- [x] Purge 36+ GB raw behavior videos, zip archives, and cache to reclaim local disk space
- [x] Complete forensic CattleLameness dataset audit & leakage investigation (`docs/audits/cattle_lameness_audit_report.md`)
- [x] Establish research log hub (`docs/research_log/`) and automated persistence rules (`.agents/rules/research_logging.md`, `AGENTS.md`)
- [x] Finalize P3 task scope: BCS, Behavior, Cow ID/Re-ID; lameness removed from primary P3 MTL model
- [x] Formulate & adopt 13-step Phase 3 Canonical Roadmap (`phase3_canonical_roadmap.md`)
- [x] Conduct read-only physical filesystem inventory audit (`docs/research_log/2026-09-20_local_dataset_inventory_audit.md`)
- [x] Build canonical dataset registry (`datasets/dataset_registry.csv` generated; 13 datasets, 28 fields, local physical statuses cataloged)
- [ ] Download and index MultiCamCows2024 (BLOCKED — upstream download currently unavailable; intended primary Re-ID)
- [x] Download & restore ScienceDB Cattle BCS raw images (53,566 images across 5 classes restored via 24-thread fast downloader & 7-Zip; index validated)
- [ ] Validate ScienceDB identity parser and verify cow-disjoint split (`datasets/bcs/sciencedb/`) once raw data restored
- [ ] Rebuild MmCows grouped evaluation protocol with time-block / multi-view protection (`datasets/behavior/mmcows/folds/`)
- [ ] Create MultiCamCows protocols: tracklet-disjoint, cross-day, cross-camera, open-set
- [ ] Download/index Ruchay 2026 (primary external BCS benchmark)
- [ ] Audit Dryad BCS local count/class discrepancy (5,940 TIFFs across classes 2–7 vs ~5,923 expected across classes 2–6)
- [ ] Download/index SideViewCows2026 (external Re-ID benchmark)
- [ ] Download/index BECA-D / BECA-L (scale & long-term Re-ID stress tests)
- [ ] Verify CBVD-5 raw data and identity metadata (external behavior benchmark)
- [ ] Run automated duplicate / near-duplicate audit across all primary datasets
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

## Canonical Task & Dataset Stack (Scientific Roles — UNCHANGED)
- **1. Body Condition Scoring (BCS)**:
  - **Primary (In-Domain)**: ScienceDB (53,566 RGB images across 5 classes: 3.25–4.25 from 10,898 project-parsed cows)
  - **Primary External Validation**: Ruchay et al. RGB-D BCS (2026) (multi-breed, wide BCS range, different camera geometry)
  - **Secondary External**: Dryad BCS (5,923 DGE images across discrete classes '2'-'6')
- **2. Behavior Recognition**:
  - **Primary (In-Domain)**: MmCows (213,686 bounding-box crops across 7 active classes from 16 cows, multi-camera CCTV)
  - **Primary External Validation**: CBVD-5 (larger-herd external test)
  - **Optional External**: CVB, XGain, Simmental 2026 (for compatible label intersections only)
- **3. Individual Cow Identification / Re-Identification (Re-ID)**:
  - **Primary (New Benchmark)**: MultiCamCows2024 (90 cows, 101,329 images, 3 cameras, 7 days, sequence-safe tracklets; replaces OpenCows2020)
  - **Primary External Validation**: SideViewCows2026 (side-view re-identification with masks)
  - **Long-term / Scale Stress**: BECA-L (appearance change over time), BECA-D (large population)
  - **Legacy Baseline**: OpenCows2020 (retained for backward comparability only; random train/val split abandoned)
- **Lameness Status**: Excluded from primary Phase 3 MTL; CattleLameness retained for historical P2 audit only.

## Current Local Availability on this Machine (Physical Verification Baseline)
- **ScienceDB BCS**:
  - Scientific Role: Primary BCS dataset
  - Local Status: **AVAILABLE**. Fully restored and verified locally (`datasets/bcs/sciencedb_bcs/dataset/`).
  - Physical Counts: 53,566 RGB images (and 53,566 XML annotation files) across 5 classes: 3.25 (7,536), 3.50 (13,256), 3.75 (14,255), 4.00 (12,556), 4.25 (5,963).
  - Manifest: `datasets/bcs/sciencedb_bcs_index.csv` validated (53,566 rows; 100% of sample paths verified existing on disk).
- **Dryad BCS**:
  - Scientific Role: Secondary external BCS validation
  - Local Status: **Raw data present locally** (`datasets/bcs/dryad_bcs/Total_sorted_DGE_images/`).
  - Physical Counts: 5,940 TIFF files present across class folders 2 through 7 (verified readable RGB TIFF, 224x224).
  - Index Status: `datasets/bcs/bcs_index.csv` currently has 0 rows (unindexed).
  - Discrepancy Note: Flagged for a later dedicated audit because prior project notes expected ~5,923 active images across classes 2–6. Discrepancy is intentionally left unresolved for now.
- **MmCows**:
  - Scientific Role: Primary Behavior dataset
  - Local Status: **Raw cropped data present locally** (`datasets/behavior/mmcows/cropped_bboxes/`).
  - Physical Counts: 213,686 indexed behavior crops in `behaviors/` (100% valid/resolving). Total local JPG count is 427,390 due to additional `lying/` (83,620) and `standing/` (130,084) crop folders. Raw source videos and archive zip were previously purged. Do not redefine the behavior dataset as 427,390 samples.
  - Index Status: `datasets/behavior/behavior_index.csv` valid and active (213,686 rows).
- **OpenCows2020**:
  - Scientific Role: Legacy Re-ID benchmark only
  - Local Status: **Locally present** (`datasets/id/opencow2020-DatasetNinja/`).
  - Physical Counts: 4,736 images across 46 identities (4,240 train in `identification-train/img/`, 496 test in `identification-test/img/`).
  - Index Status: `datasets/id/id_index.csv` valid and active (4,736 rows).
- **MultiCamCows2024**:
  - Scientific Role: Intended primary Re-ID dataset
  - Local Status: **NOT present locally**. 0 files/archives.
  - Upstream Status: Current official download attempt blocked upstream (connection reset verified locally and via Modal cloud). Status is **BLOCKED**, not complete.
- **Ruchay 2026**: Not present locally.
- **CBVD-5**: Not present locally (only a single 21 MB demo clip exists in `workspaces/nusrat/`; does not count as dataset availability).
- **SideViewCows2026**: Not present locally.
- **BECA-D / BECA-L**: Not present locally.
- **CattleEyeView / SuperAnimal / MOO**: Not present locally.
- **CattleLameness**: Locally present (50 MP4s in `CattleLameness/Data/` + 9,950 frames in `frames/`). Historical Phase 2 material only; remains strictly excluded from core Phase 3 MTL.

> [!IMPORTANT]
> **Multi-Environment Awareness**: Physical dataset availability may differ across machines and execution environments (e.g. this local laptop vs. Modal cloud volumes vs. the BRACU Lab Research PC with RTX 5090). Future agents MUST inspect physical files on disk before assuming a dataset is available locally.

## Current Blockers & Notes
- Current immediate position is **STEP 1 — Data Registry and Clean Splits**. Model training is strictly blocked until Gate 1 is passed.
- MultiCamCows2024 official download is blocked upstream by server-side connection resets on `data.bris.ac.uk/datasets/`; remains intended primary Re-ID.
- ScienceDB raw images are absent on this laptop; required prior to Step 4 BCS baseline training.
- OpenCows2020 is superseded as primary Re-ID by MultiCamCows2024 to eliminate temporal near-duplicate frame leakage and support tracklet/cross-day/cross-camera protocols.
- PCGrad/GradNorm are deferred to Step 11 as experimental controls, not primary thesis novelty.
- Do not make changes to roadmap scope without recording justification in `docs/research_log/`.
- Global Antigravity sync rule instituted: any change to agent rules, internals, skills, or templates automatically mirrors and pushes to `D:\custom-antigravity`.
