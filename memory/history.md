# Session Summary — 2026-09-20 (Phase 3 Step 2.4 Cattle Viewpoint Taxonomy & Operational Strategy Audit)

- Executed Step 2.4 initial manual visual feasibility audit of cattle viewpoint categories across ScienceDB, MmCows, and SideViewCows2026.
- Formulated candidate coarse viewpoint taxonomy: `rear`, `rear-oblique`, `side`, `front-oblique`, `front`, and `unknown / ambiguous`.
- Curated a deliberately diverse 60-image manual review pack (20 ScienceDB, 20 MmCows, 20 SideViewCows2026) across BCS classes, farm sources, behavior categories, surveillance cameras, and capture subsets to avoid selection-bias pitfalls.
- Generated 6 high-resolution 2-column contact sheets (10 images each) in `docs/audits/assets/viewpoint_visual_review/` and created index at `docs/audits/phase3_viewpoint_visual_review_index.md`.
- Completed visual review with ChatGPT-assisted initial proposals and final verification/corrections by the user across all 60 samples:
  - ScienceDB (20): 10 rear, 8 rear-oblique, 1 front-oblique (`sample_0084`), 1 unknown / ambiguous (`sample_0022` chute occlusion). Strongly rear/rear-oblique dominated (90.0%).
  - MmCows (20): 9 side, 6 rear-oblique, 1 rear, 1 front-oblique (`sample_0185`), 3 unknown / ambiguous (`sample_0107`, `0140`, `0190` due to stall bars, stanchions, dark top-down CCTV). Broadest mixture and highest ambiguity rate (15.0%).
  - SideViewCows2026 (20): 18 side, 1 front-oblique (`sample_0215` parlor entrance/turn), 1 unknown / ambiguous (`sample_0277` multi-cow barn alley). Overwhelmingly side-view dominated (90.0%).
  - Overall (60): 27 side, 14 rear-oblique, 11 rear, 3 front-oblique, 5 unknown / ambiguous, 0 front.
- Persisted verified labels and provenance metadata in `artifacts/perception_audit/viewpoint_manual_review_manifest.csv` (`review_status = human_verified`).
- Completed Step 2.4 Operational Strategy Audit evaluating 4 candidate options for Step 3 caching:
  - Option 1 (Metadata heuristics): Usable as strong domain prior for ScienceDB/SideView, but fails on MmCows (360-degree rotation in pens).
  - Option 2 (Geometric rules): REJECTED; aspect ratio is mathematically degenerate between front and rear ($w/h < 1.0$), and broken by lying postures.
  - Option 3 (MOO — Multi-view Oriented Observations, arXiv:2603.04314): REJECTED; synthetic Blender dataset (128k images) without any pretrained predictor model or weights; training from scratch violates roadmap and faces high synthetic-to-real domain gap.
  - Option 4 (Zero-shot foundation vision model): RECOMMENDED for evaluation; frozen CLIP/SigLIP requires zero training, zero parameter expansion, and leverages broad semantic priors.
  - Camera ID: Confirmed camera ID must NEVER be treated as viewpoint to prevent shortcut leakage.
- Updated `docs/audits/phase3_perception_feasibility.md` (Sections 4.5 & 4.6), updated `docs/research_log/2026-09-20_cattle_viewpoint_taxonomy_manual_review.md`, and synchronized workspace state.
- Maintained constraints: no model trained, no weights downloaded, Step 2.4 and overall Step 2 remain open. Recommended next experiment: `scripts/audit_viewpoint_zeroshot.py` testing frozen zero-shot CLIP/SigLIP against the 60 human-verified benchmark images.

# Session Summary — 2026-09-20 (Phase 3 Step 2.3 Cattle Pose / Keypoint Feasibility Audit)

- Executed Step 2.3 (Cattle Pose / Keypoint Feasibility Audit) across ScienceDB (BCS), MmCows (Behavior), and SideViewCows2026 (Re-ID).
- Evaluated official DeepLabCut 3.0+ SuperAnimal-Quadruped pose foundation models comparing two backbones: HRNet-W32 (`superanimal_quadruped_hrnet_w32.pt`) and ResNet-50 (`superanimal_quadruped_resnet_50.pt`) with official Faster R-CNN detector (`fasterrcnn_resnet50_fpn_v2`).
- Built reproducible audit script `scripts/audit_pose_feasibility.py` evaluating top-down pose inference on Step 2.1 RT-DETR-L target crops with `max_individuals=1`, 4-stage failure categorization (`upstream_localization_failure`, `pose_detector_failure`, `pose_output_returned`, `pose_inference_error`), raw confidence preservation, dynamic 39-keypoint schema extraction, SideView ground-truth mask sanity check, crash-safe `--resume`, and terminal progress display.
- Tested optional cattle-specific pose checkpoint check: `No verified directly usable pretrained cattle-specific pose checkpoint was found for this feasibility audit` (CattleEyeView has no public weights; BECA has no pose labels).
- Ran 30-image smoke test and 300-image expanded audit (100 ScienceDB, 100 MmCows, 100 SideViewCows2026) across both backbones on local GTX 1050 Ti.
- Operational Status Breakdown (N=300 per model):
  - `pose_output_returned`: 243 / 300 (81.0%) across both models (ScienceDB: 85, MmCows: 68, SideView: 90).
  - `pose_detector_failure`: 40 / 300 (13.3%) across both models (ScienceDB: 8, MmCows: 22, SideView: 10).
  - `upstream_localization_failure`: 17 / 300 (5.7%) across both models (ScienceDB: 7, MmCows: 10, SideView: 0).
  - `pose_inference_error`: 0 / 300 (0.0%) across both models — absolute zero technical crashes.
- Confidence & Geometric Sanity Check on `pose_output_returned`:
  - SideViewCows2026 (Re-ID): ResNet-50 mean raw confidence 0.4838 (HRNet 0.4093); 77.2% of keypoints fall inside ground-truth cow mask (HRNet 72.7%) as a geometric sanity check; visually more anatomically plausible and promising for Step 6 ablation, but not proven useful yet.
  - ScienceDB (BCS rear view): Confidence heavily depressed (HRNet 0.1324, ResNet 0.2878). Model hallucinates cranial points on cows facing away. SuperAnimal schema completely lacks hip/pin/hook bone keypoints (*tuber coxae*, *tuber ischiadicum*) or pelvic depression markers needed for BCS. Visual inspection indicates outputs frequently anatomically implausible; not recommended for downstream BCS.
  - MmCows (Behavior): In the reviewed 10-sample lying subset, returned pose outputs were judged visually bad / anatomically unreliable; the expanded audit had a 22% internal pose-detector failure rate on MmCows overall (ResNet mean conf 0.3607, HRNet 0.2287). Standing/walking examples were not manually reviewed here.
- Model Selection: ResNet-50 designated provisional candidate because of higher raw confidence and slightly higher mask containment; these do not establish higher pose accuracy.
- Human Visual Review & Deliverables: Persistent manual visual-validation record created at `artifacts/perception_audit/pose_manual_review.csv` (N=60 reviews across 30 samples, Human visual review [user] with ChatGPT-assisted organization). Documented review-set selection bias (ScienceDB BCS 3.25 only, MmCows Lying only, SideView parlor only; not extrapolated to unreviewed settings). ScienceDB and MmCows lying rated `clearly_wrong` (visually bad / anatomically unreliable); SideView parlor rated `plausible` / `partially_plausible` (the only genuinely plausible group). Updated `docs/audits/phase3_perception_feasibility.md` (Section 3), published `docs/research_log/2026-09-20_cattle_pose_feasibility_audit.md`, and indexed in `docs/research_log/README.md`. Step 2.3 COMPLETE!



# Session Summary — 2026-09-20 (Phase 3 Step 2.2 Cattle Segmentation Feasibility Audit)

- Executed Step 2.2 (Cattle Segmentation Feasibility Audit) across ScienceDB (BCS), MmCows (Behavior), and SideViewCows2026 (Re-ID).
- Built reproducible audit script `scripts/audit_segmentation_feasibility.py` evaluating Pipeline A (`RT-DETR-L` box -> pretrained `SAM 2.1 small` [`sam2.1_s.pt`]), Pipeline B (`YOLO26s-seg` [`yolo26s-seg.pt`], cow class only), and Diagnostic Pipeline (`Oracle GT box` -> `SAM 2.1 small`).
- Ran initial 30-image smoke test and expanded 300-image evaluation (100 images per primary dataset; seed=42) on local GTX 1050 Ti.
- SideViewCows2026 Ground Truth Results (N=100):
  - RT-DETR-L -> SAM 2.1: Mean IoU 0.9216 | Median IoU 0.9613 | Mean Dice 0.9530 | Median Dice 0.9803 | 99% IoU >= 0.50 | 96% IoU >= 0.70.
  - Oracle GT Box -> SAM 2.1: Mean IoU 0.9468 | Median IoU 0.9639 | Mean Dice 0.9717 | Median Dice 0.9817 | 100% IoU >= 0.50 | 99% IoU >= 0.70.
  - YOLO26s-seg: Mean IoU 0.8660 | Median IoU 0.9118 | Mean Dice 0.9170 | Median Dice 0.9538 | 96% IoU >= 0.50 | 95% IoU >= 0.70.
- Quantitative finding: The small observed delta (0.0252 IoU) between Oracle GT and RT-DETR-L indicates that RT-DETR box prompts worked well with SAM 2.1 on this SideView sample.
- ScienceDB & MmCows Usability Results (N=200):
  - SAM 2.1 segmented 100% of detected cattle (93/93 ScienceDB, 90/90 MmCows) with zero internal SAM failures.
  - In reviewed composites, clean exclusion of metal chute bars, head gates, and concrete/straw flooring was observed, with preservation of dorsal ridges, pin bones, and postures.
  - Fast baseline YOLO26s-seg had substantially higher non-detection rates (38% on ScienceDB, 27% on MmCows).
- Hardware efficiency: Total Pipeline A latency is 475.0 ms/frame (~2.1 FPS) on GTX 1050 Ti with ~1.4 GB peak VRAM. Fully viable for offline Step 3 caching.
- Failure case analysis: Discovered `sample_0277` multi-cow ambiguity (23 cows in barn; primary-cow area heuristic selected foreground non-target cow, yielding IoU 0.0 against target GT; Oracle GT box achieved 0.9633 IoU).
- Extended `docs/audits/phase3_perception_feasibility.md` with Section 2, generated 36 4-panel visual composites in `docs/audits/assets/perception_audit/`, and published `docs/research_log/2026-09-20_cattle_segmentation_feasibility_audit.md`. Step 2.2 COMPLETE!

# Session Summary — 2026-09-20 (Phase 3 Step 2.1 Cattle Localization Feasibility Audit)

- Executed Step 2.1 (Cattle Detection / Localization Feasibility Audit) across ScienceDB (BCS), MmCows (Behavior), and SideViewCows2026 (Re-ID).
- Built reproducible audit script `scripts/audit_localization_feasibility.py` evaluating three pretrained architectures: YOLOv8s (11.2M params), Faster R-CNN ResNet-50 FPN v2 (43.7M params), and RT-DETR-L (32.0M params).
- Ran initial 90-image smoke test and expanded 300-image audit (100 images per primary dataset; seed=42) on local GTX 1050 Ti.
- Forensic finding: YOLOv8s exhibited a 37.0% non-detection rate on ScienceDB rear-view chute images and MmCows behavior crops (hypothesized COCO broadside pasture bias / anchor-free feature grid limitations).
- Forensic finding: RT-DETR-L (94.3% raw detection rate, 94.3ms latency) and Faster R-CNN v2 (95.0% raw detection rate, 470.8ms latency) achieve robust localization without fine-tuning.
- Verified MmCows 100-sample representation across all 4 cameras (Cam 1: 24, Cam 2: 31, Cam 3: 14, Cam 4: 31) and all 16 biological cows (Cows 1–16).
- Identified multi-cow background clutter in pens (76–87% of images) requiring primary-cow selection heuristics in downstream representation caching.
- Identified that only 5 of 300 images (1.67%) were missed by all three models (extreme entrance/exit occlusions and 2.68:1 extreme horizontal lying crops).
- Designated RT-DETR-L as the provisional primary candidate for Step 2.2 due to its balanced speed/performance tradeoff.
- Generated 120 4-panel visual composites in `docs/audits/assets/perception_audit/`, authored `docs/audits/phase3_perception_feasibility.md` (Step 2.1 section), published research log `docs/research_log/2026-09-20_cattle_localization_feasibility_audit.md`. Step 2.1 COMPLETE!

# Session Summary — 2026-09-20 (Phase 3 Step 1 MmCows vs CBVD-5 Agent Visual Inspection)

- Executed direct multimodal agent visual inspection of 20 MmCows crops and 20 CBVD-5 crops + scenes using native vision model.
- Documented per-sample visual descriptions, resolutions, and verdicts in `docs/audits/phase3_behavior_agent_visual_inspection.md`.
- Evaluated MmCows: 12 GOOD, 7 ACCEPTABLE, 1 QUESTIONABLE (MM-03 narrow crop), 0 BAD. 95% of samples fully usable for deep learning.
- Evaluated CBVD-5: 11 GOOD, 6 ACCEPTABLE, 2 QUESTIONABLE (CBVD-02 distant, CBVD-04 partial stall bar), 1 BAD (CBVD-14 88x102 px pixelated smudge).
- Visually confirmed the "wide-angle sharpness illusion": CBVD-5 full scenes look crisp at 1080p, but individual cow bounding boxes are distant and 2.5x smaller in area than MmCows (median 156px vs 390px).
- Discovered temporal label inconsistency in CBVD-5 Video 621 (Frame 5 actively foraging across rail but labeled ONLY "stand").
- Confirmed rumination cannot be verified on static crops without temporal video modeling.
- Recommended keeping MmCows as Primary Behavior and CBVD-5 as External Validation (Option A).

# Session Summary — 2026-09-20 (Phase 3 Step 1 MmCows vs CBVD-5 Primary Behavior Assessment)

- Conducted exhaustive forensic comparison of MmCows vs. CBVD-5 across 10 dimensions to address user visual quality concerns regarding blurry MmCows crops.
- Proved that while CBVD-5 video frames are 1080p, its individual cow bounding boxes are 2.5x smaller in area (median 156x167 px) than MmCows crops (median 390x370 px).
- Proved CBVD-5 contains ZERO biological cow ID annotations (actor ID = 1 hardcoded everywhere), making cow-disjoint evaluation impossible and precluding identity-vs-behavior shortcut analysis.
- Identified that CBVD-5 lacks walking and licking behaviors, uses multi-label rumination/posture combinations, and exhibits 100% video overlap between official val and test splits.
- Generated 14-pair side-by-side visual comparison pack (`docs/audits/phase3_behavior_dataset_manual_comparison.md`) with 42 review images (1.0 MB) in `docs/audits/assets/behavior_dataset_comparison/`.
- Published formal research log (`docs/research_log/2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md`).
- Decisively recommended Option A: Retain MmCows as Primary and CBVD-5 as External Validation. Step 1 remains in progress pending user decision.

# Session Summary — 2026-09-20 (Phase 3 Manual Visual Verification Pack Generated)

- Engineered `scripts/build_manual_dataset_visual_verification.py` to create a deterministic manual human visual inspection pack across ScienceDB (BCS), MmCows (Behavior), and SideViewCows2026 (Re-ID) prior to beginning Step 2.
- Designed 48 curated, representative visual checks (16 ScienceDB rear-view checks across 5 classes and 3 farms, 16 MmCows behavior checks across all 7 classes + synchronized views, 16 SideView 3-panel composites `[RGB | Binary Mask | Contour Overlay]` across parlor/barn/snapshots).
- Built clean Windows terminal 4-stage `tqdm` progress UI (`ascii=True`, fixed width, no flicker).
- Executed generation run in 2.5 seconds, saving 48 Git-friendly compressed images (1.40 MB total) in `docs/audits/assets/manual_dataset_verification/`.
- Generated Markdown audit guide: `docs/audits/phase3_manual_dataset_visual_verification.md` with relative image links for VS Code and GitHub preview.
- Unignored `!docs/audits/assets/**` in `.gitignore` to allow Git tracking of audit assets while keeping raw GB data ignored.
- Updated `memory/state.md` and `memory/index.md`. Gate 1 remains cleared and locked.

# Session Summary — 2026-09-20 (Phase 3 Step 1 SideViewCows2026 Protocol Generation & Gate 1 Cleared)

- Engineered `scripts/build_sideview_reid_protocols.py` with 7-stage Windows-compatible ASCII progress UI and standalone `--verify-only` verification mode.
- Recovered 3,604 discrete recording sessions across 80,260 images and 110 cows using temporal delta clustering (`dt <= 60s`).
- Built and verified all 4 canonical Re-ID evaluation protocols under `datasets/id/sideviewcows2026/`:
  - **Protocol A (Cross-Setting)**: parlor gallery (36,811 imgs) vs barn query (25,260 imgs) vs snapshots query (607 imgs) vs parlor training representation (17,582 imgs).
  - **Protocol B (Longitudinal)**: early parlor gallery (35,433 imgs) vs late parlor query (18,960 imgs, strictly positive time gap) + long-range barn and snapshots queries (>200 days later).
  - **Protocol C (Open-Set)**: 77 Train / 11 Val / 22 Test cows (100% disjoint cow identities; balanced across subset types).
  - **Protocol D (Closed-Set)**: 110 cows with 70% train (40,745 imgs), 15% val (7,373 imgs), 15% test_parlor (6,275 imgs) + out-of-domain test sets (test_barn: 25,260 imgs, test_snapshots: 607 imgs).
- Executed exact duplicate and perceptual near-duplicate audits:
  - 0 duplicate SHA-256 hashes across all 80,260 images.
  - 0 adjacent-frame video burst crossings.
  - 0 mask mismatches (100% 1-to-1 stem and dimension match).
  - Minimum perceptual near-duplicate distance across partition boundaries is 7 bits (clean).
- Promoted deliverables to canonical `datasets/id/sideviewcows2026/`.
- Published research log: `docs/research_log/2026-09-20_sideviewcows2026_protocol_and_leakage_audit.md`.
- **GATE 1 STATUS: CLEARED & LOCKED**. Step 1 (Data Registry & Clean Splits) is 100% complete!

# Session Summary — 2026-09-20 (Phase 3 Step 1 MultiCamCows Contingency Adoption & Re-ID Roles Locked)

- User formally APPROVED the MultiCamCows2024 contingency proposed in `docs/research_log/2026-09-20_multicam_contingency_assessment.md`.
- Formally adopted approved Re-ID dataset roles across the canonical Phase 3 roadmap and registries:
  - **SideViewCows2026**: PRIMARY Re-ID dataset (replaces MultiCam under approved contingency; 80,260 images + 80,260 binary segmentation masks across 110 biological cows; protocol generation pending).
  - **BECA-L**: Primary external longitudinal Re-ID benchmark (103 beef cattle, 12,172 images, 134 dates over 7+ months, top-down dorsal view).
  - **BECA-D**: External large-scale / population stress benchmark (5,661 beef cattle, 16,889 images, 3 shots/cow).
  - **OpenCows2020**: LEGACY benchmark only (retained strictly for literature comparison; random split replaced by contiguous frame-index heuristic + duplicate harmonization).
  - **MultiCamCows2024**: BLOCKED / contingency-excluded for current Phase 3 execution due to persistent upstream connection resets (`data.bris.ac.uk`). Preserved in historical records.
- Updated canonical documents: `phase3_canonical_roadmap.md`, `docs/phase3_canonical_roadmap.md`, `datasets/dataset_registry.csv`, `memory/state.md`, `memory/index.md`, and `docs/research_log/README.md`.
- Cleaned all stale references to ScienceDB in `memory/state.md` (burst-group repair is 100% complete and locked).
- Maintained Gate 1 as OPEN pending implementation and verification of deterministic SideViewCows2026 protocols.

# Session Summary — 2026-09-20 (Phase 3 Step 1 ScienceDB Burst Split Repair Script)

- Engineered `scripts/repair_sciencedb_splits.py` to repair the ScienceDB Cattle BCS dataset split by clustering overlapping video burst passages into leak-free connected burst groups.
- Designed multi-signal conservative burst evidence:
  1. Exact SHA-256 content byte hashing.
  2. Multi-Index Hashing (MIH) on 64-bit dHash and aHash (Hamming distance <= 2) within the same farm source.
  3. In-memory normalized pixel Mean Absolute Error (MAE) on 64x64 grayscale with threshold <= 5.0 (out of 255).
  4. Disjoint Set Union (Union-Find) connected component graph clustering.
- Built rock-solid Windows PowerShell compatible `tqdm` progress UI (`ascii=True`, fixed width, multi-stage, eta, rate, zero flickering, zero spammy prints).
- Implemented multi-check verification suite (53,566 image count, group disjointness, 0 exact cross-duplicates, 0 confirmed cross-burst links).
- Executed manual run in terminal: 5,653 repaired burst groups formed from 5,662 initial passages (9 confirmed video burst overlaps merged into connected components); 70/15/15 stratified split generated (train: 37,045, val: 8,481, test: 8,040 across 3,958 train, 850 val, 845 test groups).
- Verified 0 exact duplicates, 0 cross-burst leakage; confirmed GS_1818 and GS_1823 are 100% unified in val.
- Promoted staging deliverables to canonical `datasets/bcs/sciencedb/` and updated `sciencedb_bcs_index.csv`.
- Documented in `docs/research_log/2026-09-20_sciencedb_burst_group_split_repair.md` and updated research log README.md index.

# Session Summary — 2026-09-20 (Phase 3 Duplicate Audit & Cold Turkey System Purge)

- Conducted exhaustive exact (SHA-256) and perceptual near-duplicate (Multi-Index Hashed dHash/aHash, $d \le 6$) leakage audit across all split-bearing Phase 3 datasets:
  - **ScienceDB Cattle BCS (53,566 images)**: 0 exact cross-duplicates; 88,944 near-duplicate cross-partition suspects flagged. Discovered critical vulnerability: `GS_1818` (val) and `GS_1823` (train) are consecutive frames of the same video passage shifted by 1 frame (MAE 0.25). Split **NEEDS CORRECTION** before Step 4 training by clustering passages into connected temporal burst blocks.
  - **MmCows Behavior (213,686 images)**: 0 exact cross-duplicates. 100% cow disjointness verified across canonical split (11 train, 2 val, 3 test cows) and all 4 GroupKFold splits (`overlap: set()`). Split is **CLEAN & LOCKED**.
  - **OpenCows2020 (4,736 images)**: 0 exact cross-duplicates; 1,239 near-duplicates (100% within same cow identity due to author randomization). Designated **LEGACY BASELINE ONLY**.
- Documented findings in `docs/research_log/2026-09-20_phase3_duplicate_nearduplicate_audit.md` and updated `docs/research_log/README.md`.
- Performed forensic audit and complete eradication of Cold Turkey Blocker from host system:
  - Verified background services, active processes, and main installation directories were already deleted.
  - Found and purged lingering browser extensions in Google Chrome (`pganeibhckoanndahmnfggfoeofncnii`) and Microsoft Edge (`jfphahkinplobmabmgjmjgflbhjjddeb`).
  - Purged user registry key `HKCU:\Software\Cold Turkey`.
  - Executed elevated registry cleanup removing orphaned `HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{6498E673-B9C2-4544-A722-1E854B5B573E}_is1` and `HKLM:\Software\Cold Turkey`. Verified 100% removal across all system hives.

# Session Summary — 2026-09-20 (Modal Billing Monitor & BILLING.md Integration)

- Added real-time multi-account Modal billing monitor and dashboard generator identical to `modal-qwen`.
- Created `scripts/billing_monitor.py` supporting profile discovery from `~/.modal.toml`, active profile detection, thread-safe JSON query of metered/billed cost across all 6 accounts, remaining credit calculation, and L40S/H100/L4 GPU runtime estimations.
- Created `scripts/modal_billing.py` for one-shot terminal summaries and root wrapper `billing_monitor.py` for direct command-line execution (`python billing_monitor.py` and `python billing_monitor.py --loop`).
- Successfully generated `BILLING.md` in workspace root and mirrored it to `D:\custom-antigravity\BILLING.md`.

# Session Summary — 2026-09-20 (External Benchmarks Hydration: SideViewCows, BECA, CBVD-5)

- Built high-speed multi-threaded resumable downloader `scripts/fast_download_sideviewcows.py` for SideViewCows2026 (Zenodo record 21605650, 80,260 images + masks, 23.33 GB) with browser header spoofing and custom `\r` `CleanProgressBar`. Fully downloaded and extracted `snapshots.zip` (607 imgs), `parlor.zip` (108,786 files), and `barn.zip` (15.05 GB).
- Created `scripts/fast_download_beca.py` for BECA dataset (Figshare file 63928608, Article 32070171: BECA-D + BECA-L, 18.91 GB, 29,061 images) with multi-threaded HTTP Range and automatic S3 redirect resolution. Fully downloaded and extracted `BECA.zip` (68,350 files).
- Created `scripts/download_cbvd5.py` for CBVD-5 behavior dataset from Kaggle via `kagglehub` API with automatic syncing into canonical `datasets/behavior/external/cbvd5/`. Fully downloaded and synced 887 MP4 videos, 206,100 mini frames, 5,322 annotated label frames (10.84 GB).
- Verified and updated canonical `datasets/dataset_registry.csv` via `scripts/build_dataset_registry.py` to transition SideViewCows2026, BECA-D, BECA-L, and CBVD-5 to `AVAILABLE`.
- Documented complete investigation in `docs/research_log/2026-09-20_external_benchmarks_hydration_audit.md` and updated `docs/research_log/README.md`.
- Updated hardware configuration in `personal_info.md` to document local GTX 1050 Ti laptop and remote BRACU lab RTX 5090 research rig. Mirrored changes to `D:\custom-antigravity`.

# Session Summary — 2026-09-20 (OpenCows2020 Legacy Re-ID Protocol Rebuild)

- Audited OpenCows2020 (4,736 images across 46 identities) and proved that sequence / tracklet structure cannot be recovered from provenance (frame numbers are unordered crops; consecutive MAE is identical to random pairs within cow).
- Quantified extensive random within-identity mixing in legacy `context/preprocess_id.py` random split: 1,023 frame-index adjacent pairs ($|f_1 - f_2| = 1$), 2,942 near frame-index pairs ($|f_1 - f_2| \le 5$), 1,760 visually similar pairs (MAE < 15), and 3 exact-duplicate pairs (identical SHA256) crossed train and val.
- Preserved the official benchmark `identification-test` set (496 images, 46 cows) 100% untouched.
- Rebuilt training-side train/val via contiguous frame-index heuristic + duplicate harmonization: Train=3,586, Val=654, Test=496 (total 4,736 images across 46 cows).
- Eliminated exact-duplicate leakage (0 duplicate hash overlap across any split) and reduced frame-index adjacency crossings from 1,023 down to 48 (single boundary transition per cow).
- Explicitly documented that true tracklet/temporal leakage cannot be verified because provenance is unavailable.
- Created `datasets/id/opencow2020/manifest.csv`, `train.csv`, `val.csv`, `test.csv`, `split_report.md`, and updated `datasets/id/id_index.csv`.
- Created `scripts/build_opencows_splits.py` and standalone verification `scripts/verify_opencows_splits.py` (100% pass).
- Updated `datasets/dataset_registry.csv` and documented findings in `docs/research_log/2026-09-20_opencows2020_legacy_reid_audit.md`.
- Maintained OpenCows2020 strictly as a **LEGACY BASELINE ONLY**; MultiCamCows2024 remains intended primary Re-ID.

# Session Summary — 2026-09-20

- Completed Dryad Cattle BCS Discrepancy Audit (`docs/research_log/2026-09-20_dryad_bcs_discrepancy_audit.md`).
- Fully reconciled the discrepancy between the older ~5,923 expectation and 5,940 physical files: proved that legacy `preprocess_bcs.py` hardcoded classes 2–6 (5,923 imgs) to fit a 5-class head and silently dropped class folder '7' (17 imgs from `Cow_52`).
- Proved Class 7 is 100% authentic Criollo beef cattle data on the 1–9 Wagner scale (Winkler & Boucheron, NMSU, Dryad DOI: 10.5061/dryad.tqjq2bw4s); `Cow_52` was also recorded at BCS 6 (`Cow_52_29`).
- Reconstructed 54 biological cows (`Cow_1`..`Cow_55`, `Cow_39` absent) across 148 session folders; exposed cross-session animal identity leakage if splitting by folder name instead of biological cow.
- Verified 100% image readability (all 5,940 are valid 224x224 RGB DGE TIFFs) and 0 cross-cow duplicate leakage.
- Exported master manifest `datasets/bcs/dryad/manifest.csv` (5,940 rows), `cow_audit.csv` (54 cows), `audit_report.md`, and updated legacy `datasets/bcs/bcs_index.csv` (5,940 rows).
- Completed MmCows Behavior Grouped Evaluation Protocol and Leakage Audit (`docs/research_log/2026-09-20_mmcows_grouped_protocol_and_leakage_audit.md`).
- Verified that 213,686 image crops stem from 16 genuine biological Holstein dairy cows recorded across 4 synchronized CCTV cameras over 21.0 hours (NeurIPS 2024 Spotlight, Purdue NEIS Lab).
- Protected 87.15% synchronized multi-camera events (64,830 / 74,388 events) and 15s contiguous periodic time blocks via strict cow-disjoint grouping.
- Identified Class 5 (Licking) rarity (2,009 crops, 41.7:1 imbalance) with 5 cows having zero licking crops; constructed balanced 4-Fold GroupKFold cross-validation suite evaluating 100% of cows with guaranteed positive support for all 7 classes.
- Exported master manifest `datasets/behavior/mmcows/manifest.csv` (213,686 rows), `provenance_audit.csv` (16 cows), canonical splits (`train.csv`, `val.csv`, `test.csv`), 4-fold suite (`folds/fold_[0-3].csv`), and `split_report.md`.
- Implemented and passed all leak-free assertions in `scripts/build_mmcows_splits.py` and `scripts/verify_mmcows_splits.py` (0 identity overlap, 0 multi-cam leak, 100/100 sampled path resolution).
- Conducted read-only physical filesystem inventory audit on GTX 1050 Ti machine (`docs/research_log/2026-09-20_local_dataset_inventory_audit.md`).
- Clarified physical presence vs scientific roles: ScienceDB BCS marked `METADATA_ONLY` locally (raw files absent; stale 53,566-row index); MultiCamCows2024 marked `BLOCKED` upstream (connection reset verified locally and on Modal cloud); Dryad BCS verified locally (5,940 TIFFs across classes 2-7, discrepancy flagged for future audit); MmCows verified locally (213,686 active behavior crops valid and indexed; 427,390 total JPGs include auxiliary folders; raw videos purged); OpenCows2020 verified locally (4,736 images across 46 classes).
- Created deterministic canonical dataset registry `datasets/dataset_registry.csv` via `scripts/build_dataset_registry.py` (13 datasets, 28 columns, machine-awareness fields `local_status_1050ti` and `local_verified_date`).

# Session Summary — 2026-09-19

- Adopted and locked the 13-step Phase 3 Canonical Roadmap (`phase3_canonical_roadmap.md` and `docs/phase3_canonical_roadmap.md`).
- Established primary tasks & datasets: BCS (Primary: ScienceDB, Ext: Ruchay 2026, Dryad), Behavior (Primary: MmCows, Ext: CBVD-5), Cow ID / Re-ID (New Primary: MultiCamCows2024, Ext: SideViewCows2026, Legacy: OpenCows2020).
- Locked immediate priority to STEP 1: Data Registry (`datasets/dataset_registry.csv`) and Clean Splits.
- Documented roadmap in research log (`docs/research_log/2026-09-19_phase3_canonical_roadmap.md`) and synchronized workspace memory.

# Session Summary — 2026-09-18

- Audited primary Phase 3 dataset split protocols across ScienceDB BCS (53,566 images), MmCows behavior (213,686 crops), and OpenCows2020 cow ID (4,736 images).
- Corrected status claims: verified that indexing does not equate to methodologically clean splits; verified MmCows has 7 active classes (not 5); verified Dryad BCS labels are folder classes '2'-'6' (not continuous 1-5).
- Re-aligned experiment order: locked 6-step sequence (Split Audit -> Single-Task Baselines -> Hard-Sharing MTL -> Partial-Sharing MTL -> Transfer Analysis -> optional PCGrad/GradNorm).
- Updated memory/state.md to reflect split validation status and prevent premature model training.

# Session Summary — 2026-09-17

- Conducted exhaustive forensic audit of Mendeley CattleLameness dataset (50 clips, 9,950 frames); exposed critical train/test leakage (`N (9).mp4` in Train vs `N (3).mp4` in Test from identical YouTube source).
- Built 42-animal/source grouping manifest (`cattle_lameness_manifest.csv`) with leak-free 5-fold StratifiedGroupKFold cross-validation (marked provisional/historical).
- Forensically audited 4 candidate lameness datasets: evaluated Russello 2026 (98 cows, 272 trajectories, pose-only), rejected Wu NWAFU (static pose only), whsu2s (missing repo data), and Duan 2025 (closed/private data).
- Established centralized Research Logging Hub under `docs/research_log/` and created `.agents/rules/research_logging.md` and workspace root `AGENTS.md` to permanently enforce documentation of all findings, manifests, and audits.
- Codified canonical Phase 3 scope: Body Condition Scoring (Primary: ScienceDB, Secondary: Dryad), Behavior Recognition (MmCows), and Individual Cow Identification (OpenCows2020); Lameness excluded from primary MTL training.

# Session Summary — 2026-09-16

- Fully restored and indexed 3 of 4 core datasets: Mendeley CattleLameness (9,950 frames), MmCows Behavior (213,686 images), and OpenCows2020 Cow ID (4,736 images across 46 classes).
- Built high-speed Kagglehub restore for OpenCows2020 and automated disk cleanup removing 36+ GB of raw video/zip clutter.
- Corrected Dryad BCS dataset DOI reference (doi:10.5061/dryad.tqjq2bw4s) and built automated browser launcher & real-time download watcher in `download_all.py` (commit `f543f16`).
- Packaged complete custom Antigravity environment into dedicated private GitHub repo `https://github.com/Hasinish/custom-antigravity.git` (commit `7391dc1`, `20d72e8`).
- Packaged all 4 Modal compute accounts into `custom-antigravity/credentials/modal.toml` with 1-click deployment in `setup.ps1`.
- Resurrected the lost custom Modal billing script from historical transcripts into `custom-antigravity/scripts/modal_billing.py` (commit `5fc1c76`), verified real-time balances ($104.97 total remaining credit).

# Session Summary — 2026-09-15

- Workspace synchronization check against GitHub remote origin/main. Local commit 6c3437f matches remote exactly. Untracked compiled PDFs detected. Workspace memory initialized.

# history.md — Conversation Log History

<!-- IMPORTANT FOR AGENTS: Always prepend new conversation log entries to the top of this list (most recent first). Do not append to the bottom. -->

- **[2026-09-20] Convo 27375138-e032-457f-a2a6-753e72f4a342**: Completed forensic local dataset inventory on GTX 1050 Ti machine; distinguished physical local availability from canonical scientific roles; marked MultiCamCows2024 download as BLOCKED (upstream issue); created canonical dataset registry `datasets/dataset_registry.csv` (13 datasets, 28 columns).
- **[2026-09-19] Convo 27375138-e032-457f-a2a6-753e72f4a342**: Synchronized workspace and custom-antigravity from origin/main (+287k lines, 12 config files). Deployed Antigravity customizations via setup.ps1. Added Modal accounts hasinishrak2015 and dryousufmozumder ($30 grants each), upgraded tigerwood697 ($30 grant), reaching 6 accounts and $107.01 total credit (~55 hrs L40S). Synced canonical roadmap clarifications to docs/ and pushed to GitHub. Clean start locked for STEP 1 execution.
- **[2026-09-19] Convo fa09b269-a73b-49f3-aecc-c870aef77dac**: Git synchronization status check across workspace and custom-antigravity repository. Verified clean working trees and all commits pushed to origin/main.
- **[2026-09-19] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Formally adopted Phase 3 Canonical Roadmap (13-step pipeline). Replaced OpenCows2020 with MultiCamCows2024 as primary Re-ID; designated Ruchay 2026, CBVD-5, and SideViewCows2026 as external validation; locked Step 1 (Data Registry & Clean Splits) as immediate next priority.
- **[2026-09-18] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Audited split integrity across ScienceDB, MmCows, and OpenCows2020; corrected 7-class MmCows and '2'-'6' Dryad label definitions; codified 6-step experiment sequence.
- **[2026-09-17] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Audited CattleLameness & 4 candidate datasets. Uncovered train/test leak, built 42-group leak-safe manifest, and established research log hub + agent persistence rules.
- **[2026-09-16] Convo 6522ab9b-43bd-4ef9-97e2-2228a4cfe879**: Started session to guide and execute dataset downloads on local laptop environment.
- **[2026-09-16] Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef**: Restored Lameness (9,950), Behavior (213,686), and ID (4,736) datasets. Purged 36+ GB raw behavior videos. Integrated Dryad browser stream automation. Pushed commits to GitHub. Initialized IDE customization transfer document.
- **[2026-09-15] Convo dba0f4b9-4e4d-469f-870d-b0601e3fb7ef**: Verified workspace is fully up to date with remote GitHub repo `Hasinish/cattle-health-monitoring-multi-task-model` (HEAD at 6c3437f). Flagged untracked PDF reports. Initialized memory files.
