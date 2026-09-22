# state.md — Current Workspace State

## Phase 3 Roadmap Status
- **Status**: CANONICAL / LOCKED FOR EXECUTION (`phase3_canonical_roadmap.md`)
- **Core Question**: Can cattle-centered visual representations (localization, soft masks, anatomy/pose, viewpoint) reduce shortcut learning and improve robustness across BCS, Behavior, and Re-ID compared with generic RGB representations?
- **Single Source of Truth**: [phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md) (also mirrored at [docs/phase3_canonical_roadmap.md](file:///d:/cattle-health-monitoring-multi-task-model/docs/phase3_canonical_roadmap.md))

## Active Goals & Todo (STEP 1: COMPLETE | GATE 1: CLEARED | STEP 2.1: COMPLETE | STEP 2.2: COMPLETE | STEP 2.3: COMPLETE | STEP 2.4: REOPENED | STEP 2 DATA QUALITY MILESTONE: COMPLETE | STEP 2: IN PROGRESS)
- **Immediate next action:** Resolve operational viewpoint strategy for Step 3 caching; synthesize upstream caching pipeline parameters.
- [x] STEP 2 Ruchay 2026 RGB Visual Quality Audit & Verdict (COMPLETE): Completed formal manual visual review (Hasin Ishrak with ChatGPT vision assistance) across all 10 contact sheets (100 samples, 10 Ferguson BCS classes, 94 unique cows). Verified dorsal spine, loin, hooks, pins, and tailhead morphology are sharp and observable under overhead nadir 1080p Kinect imaging. Stall divider pipes and adjacent cows necessitate upstream localization and masking. Comprehensive cross-tabulation of full 25,700-sample manifest (1,025 cows) exposed severe session confounding: BCS 2.75 is 100% confined to `06.12.2024` (4 cows), while classes 4.25–5.00 are 100% confined to `27.03.2025` (133 cows). Final verdict: **PASS FOR EXTERNAL BCS VALIDATION**. Ruchay 2026 is confirmed as the Primary External BCS Validation benchmark under frozen evaluation rules with explicit documentation of session confounding. Deliverables: `docs/audits/phase3_ruchay_bcs_visual_quality_audit.md`, `docs/audits/phase3_ruchay_bcs_visual_audit_index.md`, `docs/research_log/2026-09-22_ruchay_bcs_manual_visual_quality_verdict.md`, `docs/research_log/2026-09-22_ruchay_bcs_visual_audit_pack.md`.
- [x] STEP 2 MmCows Sequential Clips & Temporal Continuity Audit: Investigated temporal sampling characteristics of MmCows behavior crops. Proven that MmCows provides 4,768 discrete 15-second timestamps across 21.0 hours with crops sampled at exactly 15-second intervals (0.067 Hz; standard ethological scan sampling). Over 180,000 consecutive 15-second frame transitions exist across the dataset, with continuous runs reaching up to 829 consecutive frames (3.5 hours) for resting. Built 8 panoramic sequential filmstrip contact sheets (`scripts/build_mmcows_sequential_contact_sheets.py`, `docs/audits/assets/mmcows_sequential_verification/`) demonstrating 6-frame consecutive progressions (t=0s to t=+75s) and active behavior state transitions. Concluded that sequences are 100% useful for 2D spatial feature learning and macro-behavior temporal state modeling (LSTM/GRU), but not intended for 30 fps micro-kinematics / dense optical flow. Deliverables: `docs/audits/phase3_mmcows_sequential_clips_verification.md`, `docs/research_log/2026-09-22_mmcows_sequential_clips_verification.md`.
- [x] STEP 2 MmCows Behavior Visual Verification & Crop Quality Audit: Generated 7 high-resolution contact sheets (3x3 grid, 9 samples per class, 560x445 px per tile, 1720x1450 px per sheet) with complete provenance metadata banners (Cow ID, Camera ID, Split, Resolution, Filename) for all 7 active behavior categories (`Walking`, `Standing`, `Feeding_head_up`, `Feeding_head_down`, `Licking`, `Drinking`, `Lying`). Direct visual inspection confirmed MmCows crops are overwhelmingly GOOD and structurally sound for behavior classification (median resolution ~390x370 px, up to 931x719 px). Stall pipe occlusions are typical commercial loose-housing CCTV conditions; behaviors (feeding neck angles, walking gaits, recumbency) are visually distinct. Deliverables: `docs/audits/phase3_mmcows_behavior_visual_verification.md`, `docs/audits/assets/mmcows_behavior_verification/`, `docs/research_log/2026-09-22_mmcows_behavior_visual_verification.md`.
- [x] STEP 2 Data Quality Milestone (1,000-Image Human Visual Quality Reassessment & Contact Sheet Pack): Recomputed all statistics deterministically from `artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv` across ScienceDB (334), MmCows (333), and SideViewCows2026 (333). Verified only 34.20% (342/1000) meet strict-clean criteria. Key findings: MmCows has 39.94% (133/333) unknown/ambiguous viewpoints, 54.35% (181/333) combined occlusion (93 severe, 88 partial), and 12.91% (43/333) strict-clean; ScienceDB has 91.02% rear views and 99.70% occlusion-free, but suffers from 38.32% (128/334) multiple cows and 12.87% (43/334) body cutoff; SideViewCows2026 has 97.00% side views, 32.13% body cutoff, 30.93% occlusion, and 38.44% strict-clean. Built 42-sheet visual contact pack (1,000 tiles, 100% unique) with visible metadata banners. Reconciled 20-sample MmCows inspection as sample-size-limited. Decisive scientific boundaries enforced: NO dataset-role change, NO roadmap change, NO promotion of alternatives without empirical proof. Deliverables: `docs/audits/phase3_real_cattle_visual_quality_reassessment.md`, `docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md`, `docs/research_log/2026-09-22_real_cattle_visual_quality_reassessment.md`.
- [ ] STEP 2.4 (MOO Synthetic-to-Real Viewpoint Transfer Audit & Full Directional Fine-Tuning):
  - Stage 1 (Frozen Linear Probe on CPU, 2026-09-20): 93.2% synthetic val acc; 12.63% real diagnostic accuracy (40 false fronts; clean baseline). Post-hoc 63.16% mapping classified as an `exploratory post-hoc benchmark fit; invalid as final held-out evaluation evidence`. Independent visual check refuted 90-degree anatomical mismatch.
  - Stage 2 (Full Fine-Tuned ResNet-18 on L4 GPU, 2026-09-21): Trained full 8-direction ResNet-18 on canonical identity-disjoint MOO split (`scripts/modal_moo_pipeline.py`, checkpoint `artifacts/checkpoints/moo_resnet18_viewpoint8_full_l4.pth`, commit `d2c6578eaea060c4aeae9300d95ce5c77bbf25a7`). Best epoch: 14. Synthetic validation: 99.59% accuracy, Macro-F1 0.9959. Synthetic held-out test: 99.44% accuracy, Macro-F1 0.9944 on 9,600 images from 100 unseen synthetic cow identities.
  - Real Diagnostic Benchmark Evaluation (N=95 non-ambiguous samples from reused 100-sample diagnostic benchmark): Clean physical accuracy 27.37% (26/95), Macro-F1 0.1564. False-front predictions dropped from 40 to 1. Observed a strong rear/rear-oblique prediction bias (91/95 raw 8-direction predictions were back/back-left/back-right; 47 of 54 side-view cows misclassified as rear-oblique or rear; 58.82% on ScienceDB rear-facing crops). Benchmark contains zero true front samples; front recall cannot be assessed. Benchmark treated strictly as diagnostic/development evidence, NOT thesis-final held-out evidence.
  - Verdict: Full synthetic fine-tuning improved substantially over the frozen probe (27.37% vs. 12.63%), but a substantial synthetic-to-real domain gap is observed and current synthetic-only transfer remains insufficient as a standalone operational real-cattle viewpoint generator. Operational viewpoint generator: NOT YET SELECTED. STEP 2.4 remains REOPENED / IN PROGRESS; STEP 2 remains IN PROGRESS.
- [x] STEP 2.4 (Method A & Method B Zero-Shot VLM Benchmark, N=100): Evaluated frozen zero-shot vision-language models (`openai/clip-vit-base-patch32`, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, `google/siglip-base-patch16-224`) on the 100-sample cross-checked benchmark with RT-DETR-L target crops (`artifacts/perception_audit/viewpoint_zeroshot_expanded100/`).
  - Method A (6-class explicit with 'unknown / ambiguous' prompt, N=100): OpenAI CLIP: 30.0% accuracy, macro-F1 0.2468, 16 false front predictions; OpenCLIP: 6.0% accuracy; SigLIP: 5.0% accuracy. Under the tested explicit ambiguous-prompt setup, OpenCLIP and SigLIP predicted `unknown / ambiguous` for the great majority of samples. Do NOT generalize this to all VLMs.
  - Method B (Raw 5-class physical evaluation, N=95 non-ambiguous samples): OpenAI CLIP: 33.68% accuracy, macro-F1 0.2711, 19 false fronts; OpenCLIP: 15.79% accuracy, macro-F1 0.1149, 13 false fronts; SigLIP: 12.63% accuracy, macro-F1 0.0958, 66 false fronts.
  - Method B (Margin rejection at tau=0.10, N=100): OpenAI CLIP: 27.0% accuracy, 100% ambiguous recall (5/5), 50/95 false ambiguous; OpenCLIP: 8.0%; SigLIP: 7.0%.
  - Verdict: Frozen zero-shot CLIP/OpenCLIP/SigLIP is REJECTED as the operational viewpoint generator UNDER THE TESTED SETUP.
- [x] STEP 2.4 (100-Sample Expanded Cross-Check & User Adjudication): Constructed 100-sample non-overlapping review set (ScienceDB: 34, MmCows: 33, SideViewCows2026: 33; seed 2026). Generated 10 blind contact sheets; obtained independent ChatGPT vision cross-check (79/100 raw agreement). Audited 21 disagreements in `docs/audits/phase3_viewpoint_mismatch_user_review.md` and applied user adjudications (20 ChatGPT accepted, 1 user override for `vp2_0060` to `rear`). Manifest finalized with strict provenance: agent visual labeling, ChatGPT vision cross-check, user adjudication of 21 disagreements (79 were cross-check consensus only; NOT fully human-generated ground truth) at `artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv`; `docs/research_log/2026-09-20_cattle_viewpoint_expanded_crosscheck.md`. Ground truth contains zero true `front` samples; therefore front recall cannot currently be validated, but false-front predictions can still be counted.
- [x] STEP 2.4 (Manual Taxonomy Review & Operational Strategy Synthesis): Completed manual visual taxonomy review (60 samples verified) and synthesized viewpoint operational strategy in `docs/audits/phase3_perception_feasibility.md`:
  - Coarse viewpoint taxonomy (`rear`, `rear-oblique`, `side`, `front-oblique`, `front`, `unknown / ambiguous`): FEASIBLE / DEFINABLE.
  - Domain heuristics: INSUFFICIENT as a general solution (usable only as an auxiliary prior for fixed chutes; fails completely on MmCows 360-degree pen rotations).
  - Simple geometry / aspect ratio: REJECTED as a standalone classifier (mathematically degenerate between front and rear, w/h ≈ 0.6–1.0; broken by lying postures and stall bars).
  - Frozen zero-shot CLIP/OpenCLIP/SigLIP: REJECTED as the operational viewpoint generator UNDER THE TESTED SETUP.
  - MOO-supervised estimator: ROADMAP-APPROVED CANDIDATE (93.2% synthetic val clean; 12.63% clean raw real transfer; 63.16% retained only as exploratory post-hoc benchmark fit, invalid as final test evidence; independent visual verification refuted the 90° anatomical mismatch claim).
  - Supervised cattle-specific viewpoint classifier: OPEN AS ROADMAP CANDIDATE.
  - Operational viewpoint generator: NOT YET SELECTED.
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
- [x] Ingest & physically verify CVB behavior dataset in Modal volume `cvb-data`: Downloaded all 226,344 files (225,829 1080p JPEG frames + 503 COCO JSON annotations across 589 cuts; 14.29 GB) using turbo 64-connection `aria2c` on minimal container (`cpu=1.0, memory=2048`; cost ~$1.13; `scripts/modal_cvb_pipeline.py`). Verified 100% file integrity and sample images/annotations via `verify_cvb`; updated canonical `datasets/dataset_registry.csv`.
- [x] Add multi-account Modal billing monitor and live dashboard generator (`scripts/billing_monitor.py`, `scripts/modal_billing.py`, `billing_monitor.py`, `BILLING.md`).
- [x] Run automated duplicate / near-duplicate audit across all primary datasets (ScienceDB: 0 exact, 88,944 near-duplicates, overlapping passage vulnerability identified and repaired via burst clustering; MmCows: 0 exact, 0 cow overlap, clean; OpenCows: 0 exact, 1,239 near-duplicates, legacy baseline only; docs/research_log/2026-09-20_phase3_duplicate_nearduplicate_audit.md)
- [x] Correct ScienceDB passage-disjoint split into connected burst blocks (`scripts/repair_sciencedb_splits.py`; 5,653 repaired burst groups; 0 exact duplicates, 0 cross-burst overlap; 100% verified)
- [x] Generate manual human visual-verification pack for primary Phase 3 datasets (ScienceDB 16, MmCows 16, SideView 16; 48 checks + 3-panel composites generated deterministically via `scripts/build_manual_dataset_visual_verification.py`; `docs/audits/phase3_manual_dataset_visual_verification.md`)
- [x] Audit MmCows vs. CBVD-5 for Primary Behavior role (`docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md`; 14-pair manual side-by-side pack at `docs/audits/phase3_behavior_dataset_manual_comparison.md`; recommended Option A: keep MmCows Primary, preserve CBVD-5 as External Validation)
- [x] Complete multimodal agent visual inspection of 20 MmCows and 20 CBVD-5 samples (`docs/audits/phase3_behavior_agent_visual_inspection.md`; confirmed 19/20 MmCows usable, exposed CBVD-5 wide-angle crop resolution deficit [median 156px vs 390px] and temporal label inconsistency; visually validated prior audit claims; recommended retaining Option A)
- [ ] STEP 2: Cattle-perception feasibility audit (`docs/audits/phase3_perception_feasibility.md` — Step 2.1 Localization COMPLETE; Step 2.2 Segmentation COMPLETE; Step 2.3 Pose COMPLETE; Step 2.4 Viewpoint taxonomy FEASIBLE, frozen zero-shot REJECTED under tested setup, full MOO ResNet-18 fine-tuning 99.44% synthetic test acc, 27.37% real diagnostic acc with strong rear/rear-oblique bias, operational generator NOT YET SELECTED; STEP 2.4 REOPENED | STEP 2 IN PROGRESS)
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
- **ScienceDB**:
  - Scientific Role: Primary BCS dataset
  - Local Status: **Raw images present locally** (`datasets/bcs/sciencedb_bcs/dataset/`).
  - Physical Counts: 53,566 images across classes 3.25–4.25 (100% valid/resolving).
  - Split Status: 5,653 repaired burst groups; 0 exact duplicates, 0 cross-burst overlap; 100% verified.
- **Dryad BCS**:
  - Scientific Role: Secondary external BCS benchmark
  - Local Status: **Raw images present locally** (`datasets/bcs/dryad/`).
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

## Last Session (Convo 27258369-7cbf-4892-a73a-a5cd707dc4d5)
- Executed formal evidence-preservation audit and visual contact-sheet inspection for the newly completed 1,000-image human-verified real-cattle review (`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`) across ScienceDB (334), MmCows (333), and SideViewCows2026 (333).
- Recomputed all statistics directly from the source manifest:
  - Overall strict-clean rate: 34.20% (342/1000).
  - MmCows: 39.94% (133/333) unknown/ambiguous viewpoints, 54.35% (181/333) combined occlusion (93 severe, 88 partial), and 12.91% (43/333) strict-clean.
  - ScienceDB: 91.02% rear views, 99.70% occlusion-free, but 38.32% (128/334) multiple cows and 12.87% (43/334) body cutoff.
  - SideViewCows2026: 97.00% side views, 32.13% body cutoff, 30.93% occlusion, and 38.44% (128/333) strict-clean.
- Verified all 8 discussed claims (MmCows 333 samples, 133 ambiguous, 88 partial, 93 severe, 43 strict-clean; ScienceDB 334 samples, 128 multiple cows; SideView 333 samples).
- Generated 42 contact sheets (`docs/audits/assets/real_cattle_visual_quality_reassessment/`) covering 100% of the 1,000 images exactly once, with visible metadata banners and status tags.
- Authored full contact sheet index (`docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md`) and comprehensive main audit report (`docs/audits/phase3_real_cattle_visual_quality_reassessment.md`).
- Reconciled earlier 20-sample MmCows inspection as sample-size-limited without modifying historical reports.
- Enforced strict scientific decision boundaries: NO dataset roles changed, NO roadmap changes, NO promotion of alternatives without empirical proof. Warranted next investigation: task-specific visual BCS quality audit of Ruchay et al. 2026 RGB-D BCS.

## Current Blockers & Notes
- **STEP 1 IS 100% COMPLETE & LOCKED (Gate 1 Cleared)**.
- **STEP 2.1 (Localization Feasibility) IS 100% COMPLETE**.
- **STEP 2.2 (Segmentation Feasibility) IS 100% COMPLETE**.
- **STEP 2.3 (Pose Feasibility) IS 100% COMPLETE**.
- **STEP 2.4 REOPENED | STEP 2 IN PROGRESS**: Viewpoint taxonomy is FEASIBLE; frozen zero-shot REJECTED under tested setup; full MOO ResNet-18 fine-tuning completed (99.59% syn val, 99.44% syn test on 9,600 images from 100 cow IDs; 27.37% real diagnostic accuracy on N=95 non-ambiguous samples, 1 false front, strong rear/rear-oblique prediction bias). Operational viewpoint generator remains unselected.
- **DATASET QUALITY AUDITS COMPLETE**: 1,000-image multi-dataset review completed across ScienceDB, MmCows, and SideViewCows2026 (`docs/audits/phase3_real_cattle_visual_quality_reassessment.md`). Dedicated Ruchay 2026 BCS manual visual audit completed (`docs/audits/phase3_ruchay_bcs_visual_quality_audit.md`); verdict **PASS FOR EXTERNAL BCS VALIDATION** with documented session confounding limitation. ScienceDB retained as primary BCS; Ruchay 2026 retained as Primary External BCS Validation benchmark; MmCows retained as primary behavior; SideViewCows2026 retained as primary Re-ID.
- Immediate next action: Resolve operational viewpoint strategy for Step 3 caching.
- Antigravity sync rule: changes mirrored to `D:\custom-antigravity`.
