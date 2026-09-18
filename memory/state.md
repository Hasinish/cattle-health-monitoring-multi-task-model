# state.md — Current Workspace State

## Phase 3 Roadmap Status
- **Status**: CANONICAL / LOCKED FOR EXECUTION (`phase3_canonical_roadmap.md`)
- **Core Question**: Can cattle-centered visual representations (localization, soft masks, anatomy/pose, viewpoint) reduce shortcut learning and improve robustness across BCS, Behavior, and Re-ID compared with generic RGB representations?
- **Single Source of Truth**: [phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md) (also mirrored at [docs/phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/phase3_canonical_roadmap.md))

## Active Goals & Todo (STEP 1: Data Registry & Clean Splits)
- [x] Download & restore ScienceDB Cattle BCS dataset (Primary, 53,566 RGB images, index CSV generated)
- [x] Download & restore 213,686 MmCows behavior images via Hugging Face (`cropped_bboxes.zip`)
- [x] Download & index OpenCows2020 (4,736 images across 46 classes via Kagglehub - designated Legacy Baseline)
- [x] Purge 36+ GB raw behavior videos, zip archives, and cache to reclaim local disk space
- [x] Complete forensic CattleLameness dataset audit & leakage investigation (`docs/audits/cattle_lameness_audit_report.md`)
- [x] Establish research log hub (`docs/research_log/`) and automated persistence rules (`.agents/rules/research_logging.md`, `AGENTS.md`)
- [x] Finalize P3 task scope: BCS, Behavior, Cow ID/Re-ID; lameness removed from primary P3 MTL model
- [x] Formulate & adopt 13-step Phase 3 Canonical Roadmap (`phase3_canonical_roadmap.md`)
- [ ] Build canonical dataset registry (`datasets/dataset_registry.csv`)
- [ ] Download and index MultiCamCows2024 (Primary Re-ID: 90 cows, 101,329 images, 3 cameras, 7 days)
- [ ] Validate ScienceDB identity parser and verify cow-disjoint split (`datasets/bcs/sciencedb/`)
- [ ] Rebuild MmCows grouped evaluation protocol with time-block / multi-view protection (`datasets/behavior/mmcows/folds/`)
- [ ] Create MultiCamCows protocols: tracklet-disjoint, cross-day, cross-camera, open-set
- [ ] Download/index Ruchay 2026 (primary external BCS benchmark)
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

## Canonical Task & Dataset Stack
- **1. Body Condition Scoring (BCS)**:
  - **Primary**: ScienceDB (53,566 RGB images across 5 classes: 3.25–4.25 from 10,898 project-parsed cows)
  - **Primary External Validation**: Ruchay et al. RGB-D BCS (2026) (multi-breed, wide BCS range, different camera geometry)
  - **Secondary External**: Dryad BCS (5,923 DGE images across discrete classes '2'-'6')
- **2. Behavior Recognition**:
  - **Primary**: MmCows (213,686 bounding-box crops across 7 active classes from 16 cows, multi-camera CCTV)
  - **Primary External Validation**: CBVD-5 (larger-herd external test)
  - **Optional External**: CVB, XGain, Simmental 2026 (for compatible label intersections only)
- **3. Individual Cow Identification / Re-Identification (Re-ID)**:
  - **Primary (New Benchmark)**: MultiCamCows2024 (90 cows, 101,329 images, 3 cameras, 7 days, sequence-safe tracklets; replaces OpenCows2020)
  - **Primary External Validation**: SideViewCows2026 (side-view re-identification with masks)
  - **Long-term / Scale Stress**: BECA-L (appearance change over time), BECA-D (large population)
  - **Legacy Baseline**: OpenCows2020 (retained for backward comparability only; random train/val split abandoned)
- **Lameness Status**: Excluded from primary Phase 3 MTL; CattleLameness retained for historical P2 audit only.

## Current Blockers & Notes
- Current immediate position is **STEP 1 — Data Registry and Clean Splits**. Model training is strictly blocked until Gate 1 is passed.
- OpenCows2020 is superseded as primary Re-ID by MultiCamCows2024 to eliminate temporal near-duplicate frame leakage and support tracklet/cross-day/cross-camera protocols.
- PCGrad/GradNorm are deferred to Step 11 as experimental controls, not primary thesis novelty.
- Do not make changes to roadmap scope without recording justification in `docs/research_log/`.


