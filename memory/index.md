# Workspace Index

## Directory Structure & Detailed File Context

### `memory/`
Workspace memory and context tracking system for AI agents.
- `personality.md`: Defines assistant persona, roasting rules, tone parameters, humor mechanics, and guardrails.
- `personal_info.md`: Detailed user profile for Hasin Ishrak (education, roles, technical stack, target companies).
- `purpose.md`: Workspace mission statement, thesis milestones, and research objectives.
- `state.md`: Active goals, session logs, blockers, and workspace sync status.
- `history.md`: Chronological log of past conversations and session achievements.
- `index.md`: Comprehensive workspace index detailing directories, code, models, and LaTeX thesis assets.

### `cattle_thesis_p1_latex/`
Phase 1 (P1) LaTeX thesis sources and artifacts for the CSE400 pre-thesis submission.
- `main.tex`: Master LaTeX document assembling P1 chapters, abstract, title page, and layout configurations.
- `chapters/`: Individual chapter source files detailing problem introduction, preliminary literature review, and proposed methodology.
- `bibliography/`: BibTeX references cited throughout Phase 1.
- `core/`: Template macros, styling packages, and university formatting definitions.
- `images/`: Architectural diagrams, baseline flowcharts, and sample figures for P1.
- `main.pdf`: Compiled Phase 1 thesis report.

### `cattle_thesis_p2_latex/`
Phase 2 (P2) LaTeX thesis sources, defense posters, and compiled final thesis reports.
- `main.tex`: Master LaTeX document orchestrating the complete final thesis manuscript.
- `chapters/`: Modular LaTeX files for Chapters 1 through 6:
  - Chapter 1: Introduction, problem statement, research motivation, objectives, and contributions.
  - Chapter 2: Literature review on computer vision in livestock, individual identification, BCS assessment, behavior classification, and lameness detection.
  - Chapter 3: Proposed Multi-Task Learning (MTL) methodology, backbone architectures (ResNet-18, EfficientNet), attention mechanisms (CBAM), loss balancing formulations (Total Loss = w_id * L_id + w_bcs * L_bcs + w_beh * L_beh + w_lame * L_lame), and temporal spatiotemporal lameness modeling.
  - Chapter 4: Experimental setup, dataset descriptions (Dryad, ScienceDB, CBVD, MMCows), augmentation strategies, training hyperparameters, and evaluation metrics.
  - Chapter 5: Results, performance metrics (Accuracy, Precision, Recall, F1, Loss curves), ablation studies comparing single-task vs multi-task architectures, and qualitative visualizations.
  - Chapter 6: Conclusion, limitations, ethical considerations, and future directions.
- `bibliography/`: Complete BibTeX references for peer-reviewed citations.
- `images/`: Model architecture diagrams (`mtl_architecture.png`), loss curves, confusion matrices, and crop samples.
- `T25301094_P2 Poster.pdf`: Compiled presentation poster for the P2 thesis defense exhibition.
- `T25301094_P2_Report.pdf`: Compiled final P2 thesis report document ready for department archiving.

### `context/`
Dataset preprocessing routines, member work splits, and original problem context files.
- `context1_master_plan.txt`: Research master plan and pipeline roadmap.
- `context2_bcs.txt` & `context2_behavior.txt`: Dataset specifications, label encodings, and split guidelines for BCS and behavior tasks.
- `Context3_[Member]_[Task].txt`: Individual member task configurations (Hasin, Nusrat, Shouvik, Bithi, Namira) detailing assigned model variants and datasets.
- `Pre-Thesis I Report.pdf`: Original Phase 1 evaluation report.
- `preprocess_bcs.py`: Image cropping, resizing, and normalization pipeline for Dryad BCS images.
- `preprocess_cbvd.py`: Processing script for cattle behavior video dataset frames.
- `preprocess_id.py`: Extraction and alignment pipeline for individual cattle identification datasets.
- `preprocess_lameness.py`: Frame sequence extraction and optical flow/spatiotemporal prep for lameness analysis.
- `preprocess_mmcows_behavior.py`: Data loader and bounding-box crop generator for MMCows behavior dataset.
- `preprocess_sciencedb_bcs.py`: Preprocessing script for ScienceDB thermal/RGB cattle dataset.

### `thesis_marking_rubrics.md`
Official BRAC University CSE400 Thesis Marking Rubrics markdown sheet detailing full mark distributions (100 total: Pre-thesis 1: 5, Pre-Thesis 2: 10, Defense Panel: 30, Supervisor: 55), Course Outcomes (CO1 to CO14), Program Outcomes (PO 2 to PO 12), chapter content mappings, and assessment strategies.

### `phase3_deadline_execution_2026-09-26.md`
Active deadline execution priority overlay for the 26 September 2026 thesis deadline. Enforces the guiding principle: "Before the 26 Sep deadline, execute only the minimum defensible thesis runs. All exhaustive ablations remain deferred roadmap work and can be completed later if needed." Preserves the full canonical roadmap without scientific cancellation. Codifies the 8-run sequence (3 single-task baselines, 3 perception-enhanced models, E1 basic hard sharing control, E3 main deadline MTL model with modular adapters/private pathways). Enforces strict viewpoint rules (tested/deferred, never camera ID) and scientific claim boundaries.

### `phase3_canonical_roadmap.md` & `docs/phase3_canonical_roadmap.md`
Master canonical 13-step roadmap locked for Phase 3 execution. Formulates the core thesis question: *"Which cattle-specific visual priors (localization, soft masks, anatomy/pose, viewpoint) are useful for which downstream task, and what information should each task preserve or suppress?"* Locks downstream scope to BCS (ScienceDB; Ruchay 2026 external), Behavior (CVB + Kaggle Beef as the primary dense-video training stack; MmCows external identity-aware validation; CBVD-5 secondary external), and Re-ID (SideViewCows2026 as primary under approved contingency; BECA-L longitudinal external; BECA-D scale stress external; OpenCows2020 legacy baseline; MultiCamCows2024 contingency-excluded). Outlines a strict 13-step progression from Step 1 data registry to final 3-seed benchmark tables.

### `docs/`
Structured project documentation, defense resources, forensic audits, and official deliverables.
- `README.md`: Master directory guide for the organized `docs/` workspace.
- `phase3_canonical_roadmap.md`: Canonical Phase 3 execution roadmap mirror.
- `audits/`: Detailed dataset forensic investigations, anti-leakage manifests, and cross-validation architectures.
  - `phase3_real_cattle_visual_quality_reassessment.md`: Formal evidence-preservation audit report for the 1,000-image human-verified real-cattle review across ScienceDB (334), MmCows (333), and SideViewCows2026 (333), detailing exact distributions (34.20% strict-clean, 39.94% MmCows viewpoint ambiguity, 54.35% MmCows occlusion, 38.32% ScienceDB multi-cow), verifying 8 discussed claims, and preserving dataset roles.
  - `phase3_mmcows_sequential_clips_verification.md`: Comprehensive audit report on MmCows temporal sampling rate (15s intervals) and sequential clip utility with 8 embedded panoramic sequential filmstrips (6 consecutive frames, +0s to +75s).
  - `assets/mmcows_sequential_verification/`: Directory containing 8 high-resolution sequential filmstrip contact sheets (`mmcows_seq_<behavior>.jpg` and `mmcows_seq_transition.jpg`) with temporal metadata banners (rel time, clock time, cow ID, camera ID, resolution).
  - `phase3_mmcows_behavior_visual_verification.md`: Comprehensive visual verification report and qualitative assessment for MmCows behavior crops across all 7 active categories with 7 embedded high-resolution contact sheets (3x3 grid, 9 samples per class, 560x445 px per tile).
  - `assets/mmcows_behavior_verification/`: Directory containing 7 high-resolution contact sheets (`mmcows_<behavior>_sheet.jpg`) with metadata banners (Cow ID, Camera ID, Split, Resolution, Filename) evaluating crop usability.
  - `phase3_real_cattle_visual_quality_contact_sheet_index.md`: Contact sheet index document for the 42-sheet visual audit pack covering all 1,000 reviewed images with per-sheet summaries and targeted issue navigation.
  - `assets/real_cattle_visual_quality_reassessment/`: Directory containing 42 high-resolution 5x5 grid contact sheets (14 ScienceDB, 14 MmCows, 14 SideViewCows2026) covering 100% of the 1,000 reviewed images exactly once with metadata banners and status tags.
  - `phase3_perception_feasibility.md`: Cattle perception feasibility audit document. Step 2.1 (Cow Detection / Localization; evaluated YOLOv8s, Faster R-CNN v2, RT-DETR-L), Step 2.2 (Cow Segmentation; evaluated RT-DETR-L -> SAM 2.1 small vs YOLO26s-seg vs Oracle GT -> SAM 2.1), Step 2.3 (Cattle Pose; evaluated SuperAnimal-Quadruped HRNet-W32 and ResNet-50), and Step 2.4 (Manual Visual Taxonomy Review; candidate 6-class viewpoint taxonomy verified across 60 samples).
  - `phase3_viewpoint_visual_review_index.md`: Step 2.4 viewpoint visual review index document embedding 6 contact sheets across ScienceDB, MmCows, and SideViewCows2026 with verified distribution tables and defensible findings.
  - `phase3_viewpoint_expanded_crosscheck_index.md`: Step 2.4 100-sample expanded viewpoint cross-check index embedding 10 blind contact sheets, documenting independent ChatGPT vision cross-check, 79% raw agreement, and user adjudication results.
  - `phase3_viewpoint_mismatch_user_review.md`: User review and adjudication document for the 21 disagreements between agent visual predictions and independent ChatGPT vision.
  - `assets/bcs_perception/`: Visual audit directory containing high-resolution 4-panel visual contact sheet (`bcs_perception_contact_sheet.jpg`, 1280x1250 px) across 5 BCS classes verifying RT-DETR-L detection bounding boxes, primary cow crops, SAM 2.1 binary masks, and 4-channel guided model inputs.
  - `assets/beef_rtdetr_failure_fallback/`: Staging directory containing 3 4-panel visual composites (`Original | Center Point | SAM Mask | Overlay`) and master contact sheet (`contact_sheet.jpg`) evaluating single positive center point fallback on the 3 RT-DETR-L Kaggle Beef detection misses.
  - `assets/beef_A4_A5_fresh40/`: Directory containing 40 3-panel visual composites and master contact sheet (`contact_sheet.jpg`) evaluating condition A4 vs A5 on 40 fresh Kaggle Beef canonical train frames.
  - `assets/beef_sam_prompt_rescue/`: Directory containing 120 3-panel visual composites and 6 master contact sheets evaluating 6 prompt conditions on 20 Kaggle Beef frames.
  - `assets/behavior_primary_segmentation/`: Directory containing 70 visual composites and 3 master contact sheets evaluating SAM 2.1 on CVB and Beef.
  - `assets/behavior_primary_localization/`: Directory containing 45 4-panel visual composites and 2 master contact sheets evaluating RT-DETR-L on CVB and Beef.
  - `assets/viewpoint_visual_review/`: Directory containing 6 high-resolution 2-column contact sheets (10 samples each, 60 total) for manual visual review of cattle viewpoints.
  - `assets/perception_audit/`: Directory containing 120 4-panel localization composites, 36 4-panel segmentation composites, and 90 pose overlay JPGs across ScienceDB, MmCows, and SideViewCows2026.
  - `candidate_lameness_datasets_audit.md`: Deep forensic audit of 4 potential alternative lameness datasets (Russello 2026, Wu/NWAFU, whsu2s, Duan 2025).
  - `cattle_lameness_audit_report.md`: Forensic audit of the CattleLameness dataset (50 clips, 42 cattle, ezgif container footprints, cross-split leakage identification).
  - `cattle_lameness_grouping_report.md`: Anti-leakage clustering architecture, multi-clip group resolution (8 groups, 16 clips), and balanced 5-fold StratifiedGroupKFold cross-validation specification.
  - `phase3_near_duplicate_suspects.csv`: Exhaustive export of cross-partition near-duplicate suspect pairs with 64x64 grayscale MAE scores across ScienceDB, MmCows, and OpenCows2020.
  - `phase3_manual_dataset_visual_verification.md`: Curated 48-check visual verification document embedding resized thumbnails and 3-panel composites for human visual sanity checking across ScienceDB, MmCows, and SideViewCows2026.
  - `assets/manual_dataset_verification/`: Directory of 48 compact, Git-friendly review images (1.4 MB total) including 16 ScienceDB rear-view thumbnails, 16 MmCows behavior crops, and 16 SideView 3-panel composites (`[RGB | Binary Mask | Contour Overlay]`).
  - `phase3_behavior_dataset_manual_comparison.md`: 14-pair side-by-side manual visual comparison document evaluating MmCows vs. CBVD-5 across matching/unique behaviors, crop resolutions, and scene context.
  - `assets/behavior_dataset_comparison/`: Directory of 42 review images (1.0 MB total) containing paired MmCows crops, CBVD-5 crops, and full-scene 1080p thumbnails with bounding box overlays.
  - `phase3_behavior_agent_visual_inspection.md`: Multimodal agent visual inspection report auditing 20 MmCows and 20 CBVD-5 samples, analyzing image quality, crop boundaries, resolution, and annotation validity.
  - `assets/agent_behavior_inspection/`: Staging directory containing 20 MmCows crops, 20 CBVD-5 crops, 20 CBVD-5 scenes, and `manifest.json`.
- `research_log/`: Centralized research log repository documenting experiments, dataset investigations, architectural decisions, and ablation studies.
  - `README.md`: Research log protocol, entry structure guidelines, and historical log index table.
  - `2026-09-24_sciencedb_bcs_perception_pipeline_preparation.md`: Architecture, cache pipeline, failure counting consistency, test isolation protocol, and matched-subset baseline comparison for Run 4 ScienceDB BCS Perception-Enhanced model.
  - `2026-09-23_sciencedb_bcs_modal_training_preparation.md`: Pre-flight readiness audit and cloud preparation for full 30-epoch ScienceDB RGB single-task BCS baseline on Modal (`tigerwood697`); verified CUDA (T4), 53,566 images across 5 classes at `/data/dataset`, canonical split hashes, 15/15 path resolutions via PIL, persistent checkpoint volume `/checkpoints`, and live tqdm streaming.
  - `2026-09-23_beef_rtdetr_failure_fallback.md`: Empirical verification of proposed fallback rule (IF RT-DETR detects no cow -> SAM 2.1 with ONE center point (112, 112)) on the 3 known detection misses from fresh-40 audit on Modal; achieved 100% recovery (3/3 non-empty masks; Drinking 6 comps, Feeding 146 comps, Lying 25 comps); status: PENDING HUMAN REVIEW.
  - `2026-09-23_beef_A4_A5_fresh40.md`: Comparative evaluation of detector-guided condition A4 (RT-DETR box -> SAM 2.1) vs A5 (RT-DETR box + center point -> SAM 2.1) across 40 fresh canonical Kaggle Beef training frames (Seed 2026, 29 sessions; Modal tigerwood693, GPU T4); both achieved 92.5% mask return (37/40) with 3 upstream RT-DETR failures (7.5%); A5 reduced connected components by 37.6% (19.6 vs 31.4); status: PENDING HUMAN REVIEW.
  - `2026-09-23_beef_sam_prompt_rescue.md`: Controlled prompt-rescue audit testing 6 prompt conditions against baseline A0 across 20 Kaggle Beef frames on Modal; point prompts achieved 100% mask return; detector prompts achieved 95% return; status: PENDING HUMAN REVIEW.
  - `2026-09-23_behavior_primary_segmentation_sanity.md`: Small zero-shot segmentation sanity check using pretrained SAM 2.1 Small across 45 frames (25 CVB, 20 Beef); verified CVB 100% return; exposed Beef 50% technical failure on full crop box; status: PENDING HUMAN REVIEW.
  - `2026-09-23_behavior_primary_localization_sanity.md`: Small empirical localization sanity check using pretrained RT-DETR-L on 45 frames (25 CVB, 20 Beef); verified CVB 96% hit rate and 100% overlap; verified Beef single-cow crop status.
  - `2026-09-23_cvb_beef_behavior_protocol_verification.md`: Complete audit and verification log for the canonical CVB + Kaggle Beef Behavior dataset protocol (`datasets/behavior/cvb_beef/`), detailing annotation-level CVB parsing on Modal (2,481 segments), Kaggle Beef clip alignment (2,793 clips), deterministic multi-objective group-stratified partitioning (Seed 2026), 100% group isolation across 267 groups, and Gate 1 clearance.
  - `2026-09-23_sciencedb_bcs_baseline_pipeline.md`: Implementation, auditing, and smoke verification of the Phase 3 Step 4 single-task ScienceDB RGB BCS baseline pipeline (`scripts/train_sciencedb_bcs_baseline.py`).
  - `2026-09-23_behavior_primary_stack_correction.md`: Approved evidence-based roadmap correction replacing MmCows as the sole primary Behavior dataset with a combined CVB + Kaggle Beef dense-video training stack; retains MmCows as external identity-aware validation and requires new source/session-grouped splits before training.
  - `2026-09-22_real_cattle_visual_quality_reassessment.md`: Step 2 1,000-image human-verified real-cattle visual quality reassessment and contact sheet audit across ScienceDB, MmCows, and SideViewCows2026.
  - `2026-09-21_moo_resnet18_full_viewpoint_training_and_real_diagnostic.md`: Step 2.4 full fine-tuning of ResNet-18 on 8-direction MOO synthetic split and real cattle diagnostic benchmark evaluation.
  - `2026-09-20_cattle_viewpoint_taxonomy_manual_review.md`: Step 2.4 viewpoint taxonomy manual review feasibility audit across ScienceDB, MmCows, and SideViewCows2026.
  - `2026-09-20_cattle_pose_feasibility_audit.md`: Step 2.3 pose feasibility audit evaluating DeepLabCut SuperAnimal-Quadruped HRNet-W32 and ResNet-50 across 300 samples with human visual validation.
  - `2026-09-20_cattle_segmentation_feasibility_audit.md`: Step 2.2 segmentation feasibility audit evaluating RT-DETR-L -> SAM 2.1 small (0.9216 Mean IoU, 0.9530 Mean Dice on SideView; 93/100 segmented on ScienceDB, 90/100 on MmCows) vs YOLO26s-seg (0.8660 IoU, 38% missed on ScienceDB, 27% on MmCows), confirming pretrained segmentation feasibility without fine-tuning.
  - `2026-09-20_cattle_localization_feasibility_audit.md`: Two-stage localization audit (90 smoke, 300 expanded) comparing YOLOv8s, Faster R-CNN v2, and RT-DETR-L across ScienceDB, MmCows, and SideViewCows2026, establishing RT-DETR-L (94.3% raw detection rate, 94ms latency) as primary upstream localizer.
  - `2026-09-20_mmcows_vs_cbvd5_primary_behavior_assessment.md`: Forensic assessment comparing MmCows and CBVD-5 for primary behavior role across 10 dimensions, proving CBVD has zero cow IDs, median 156x167px crops, lacks walking/licking, and recommending Option A (keep MmCows primary, preserve CBVD-5 as external).
  - `2026-09-20_sideviewcows2026_protocol_and_leakage_audit.md`: Protocol generation and leakage audit for SideViewCows2026, building 4 canonical protocols, recovering 3,604 recording sessions, verifying 0 duplicate and 0 adjacent-frame leakage, and clearing Gate 1.
  - `2026-09-20_multicam_contingency_assessment.md`: Forensic assessment of MultiCamCows2024 upstream block, evaluating SideViewCows2026, BECA-L, BECA-D, and OpenCows2020, and adopting SideViewCows2026 as primary Re-ID benchmark under approved contingency.
  - `2026-09-20_sciencedb_burst_group_split_repair.md`: Forensic repair of ScienceDB Cattle BCS split into 5,653 unified burst groups, eliminating 1-frame-shifted video burst leakage (e.g., `GS_1818` vs `GS_1823`) with verified 0 cross-burst overlap.
  - `2026-09-20_phase3_duplicate_nearduplicate_audit.md`: Automated exact (SHA-256) and perceptual near-duplicate (dHash/aHash, $d \le 6$) audit across all split-bearing Phase 3 datasets, exposing ScienceDB overlapping passage vulnerability.
  - `2026-09-20_external_benchmarks_hydration_audit.md`: Hydration and verification audit for SideViewCows2026, BECA, and CBVD-5.
  - `2026-09-20_opencows2020_legacy_reid_audit.md`: Forensic audit of OpenCows2020 legacy Re-ID protocol, proving lack of sequence recoverability from provenance, exposing 1,023 frame-index adjacency crossings in legacy random shuffle, and verifying contiguous heuristic rebuild.
  - `2026-09-20_local_dataset_inventory_audit.md`: Physical inventory audit on GTX 1050 Ti machine and canonical dataset registry creation.
  - `2026-09-19_phase3_canonical_roadmap.md`: Formal log adopting the 13-step Phase 3 Canonical Roadmap and MultiCamCows2024 Re-ID replacement.
  - `2026-09-18_cattle_centered_anatomy_aware_direction.md`: Proposed research direction on segmentation-guided, anatomy-aware, and viewpoint-aware representation learning.
  - `2026-09-18_dataset_split_integrity_audit.md`: Forensic audit of ScienceDB, MmCows, and OpenCows2020 split integrity and leakage risks.
  - `2026-09-17_lameness_investigation.md`: Complete forensic audit log covering CattleLameness leakage discovery, 42-group resolution, and candidate dataset audit.
- `defense (p2)/`: Oral examination preparation guides, presentation notes, and formatted crib sheets for Phase 2.
  - `presentation_key_concepts.md` (and `.html`, `.docx`): Defense presentation key concepts, talking points, and Q&A crib sheet.
  - `qa_study_guide.md`: Comprehensive defense Q&A preparation guide covering deep learning theory, MTL tradeoffs, and thesis defense questions.
  - `style_template.html`: CSS formatting template for HTML export styling.
- `thesis/`: Thesis manuscript reviews, chapter summaries, and deep technical documentation.
  - `cattle_thesis_p2_preview.md`: Markdown preview summary of P2 thesis content.
  - `detailed_p2_info.md`: Supplementary technical documentation for Phase 2.
  - `deep_analysis.md`: Detailed architectural and statistical deep dive into experimental results.
  - `thesis_review.md`: Peer-review feedback, Turnitin similarity audit, and revision checklist.
- `deliverables/`: Official final compiled presentation posters and thesis submission PDFs.
  - `T25301094_P2_Report.pdf`: Final Phase 2 thesis report document ready for department archiving.
  - `T25301094_P2 Poster.pdf`: Compiled presentation poster for the P2 thesis defense exhibition.
- `results/`: Raw model logs and extracted performance dumps.
  - `extracted_results.txt`: Tabulated benchmark metrics across all model runs and task heads.

### `.agents/` & Root Rules
- `AGENTS.md`: Root workspace rules governing continuous research logging, manifest preservation, and dual-device git synchronization.
  - `rules/research_logging.md`: Strict agent operational rule enforcing documentation of all audits and experiments into `docs/research_log/`.

### `datasets/` (Tracked manifests and registry; raw large data directories git-ignored)
Canonical benchmark data and task registries.
- `dataset_registry.csv`: Canonical Phase 3 dataset registry cataloging all 13 candidate and benchmark datasets across 28 schema fields, distinguishing scientific roles from local physical machine availability (`local_status_1050ti`).
- `bcs/`:
  - `sciencedb_bcs_index.csv`: 53,566-row master index for Primary BCS task aligned with leak-free passage split.
  - `sciencedb/`:
    - `train.csv` (37,045 rows), `val.csv` (8,481 rows), `test.csv` (8,040 rows): 100% burst-group-disjoint / sequence-safe split manifests.
    - `burst_group_audit.csv`: Complete audit cataloging all 5,653 connected burst groups, member passages, and label distributions.
    - `confirmed_burst_overlap_links.csv`: 9 confirmed overlapping burst pairs resolved via Disjoint Set Union.
    - `split_report.md`: Forensic audit report documenting the burst-group clustering methodology and zero-leakage verification.
  - `external/ruchay2026/`:
    - `ruchay2026_manifest.csv`: 25,700-sample deterministic manifest of Ruchay et al. 2026 RGB-D BCS benchmark (1,025 cows, 4 sessions, 10 ordinal classes 2.75–5.00).
    - `Dataset.xlsx`: Official metadata file from Zenodo record 20290988.
  - `bcs_index.csv`: 5,940-row legacy index mapping all Dryad DGE images to labels and cow IDs.
  - `dryad/`:
    - `manifest.csv`: 5,940-row master manifest mapping all DGE TIFFs across classes 2–7, biological cow numbers, session tags, and SHA256 hashes.
    - `cow_audit.csv`: 54-biological-cow census cataloging session counts, BCS trajectories, and image totals.
    - `audit_report.md`: Forensic report reconciling 5,923 vs 5,940 discrepancy and proving validity of Class 7.
- `behavior/`:
  - `behavior_index.csv`: 213,686-row master index mapping crops to 7 active classes and legacy split.
  - `cvb_beef/`:
    - `manifest.csv`: 5,274-row unified canonical Primary Behavior dataset manifest combining CVB and Kaggle Beef with complete tracklet, frame, camera, and session provenance.
    - `train.csv` (3,785 rows / 184 groups), `val.csv` (680 rows / 39 groups), `test.csv` (809 rows / 44 groups): Leakage-safe source/session-grouped split manifests.
    - `label_mapping.csv`: Complete 17-row mapping defining the canonical 5-class taxonomy (`Standing`, `Lying`, `Feeding`, `Drinking`, `Walking`) and documenting exclusions.
    - `split_report.md`: Forensic audit report and mathematical distribution proofs for the CVB + Kaggle Beef protocol.
  - `cvb/`:
    - `cvb_cuts_manifest.csv`: 502-row cut-level forensic manifest mapping cuts to camera IDs, dates, start/end frames, and primary tags.
    - `cvb_tracks_manifest.csv`: 3,693-row annotation-level tracklet segment manifest extracted directly from 502 CVB COCO JSONs on Modal volume `cvb-data` (2,481 canonical, 1,212 excluded).
  - `beef_cattle_behavior/`:
    - `manifest.csv`: 4,337-row clip-level master manifest of Kaggle Beef Behavior dataset cataloging video clips, sessions, duration, fps, and behavior tags.
  - `mmcows/`:
    - `manifest.csv`: 213,686-row master manifest mapping all crops to cow ID, camera ID, epoch, ISO timestamp, time block, synchronized event ID, canonical split, and folds 0–3.
    - `provenance_audit.csv`: Cow-level provenance audit table summarizing all 16 cows across 20 parameters.
    - `train.csv` (148,401 rows), `val.csv` (25,134 rows), `test.csv` (40,151 rows): Leakage-safe canonical split manifests.
    - `folds/`:
      - `fold_0.csv`, `fold_1.csv`, `fold_2.csv`, `fold_3.csv`: 4-Fold GroupKFold cross-validation suite (each 213,686 rows) evaluating 100% of cows with guaranteed 7-class positive coverage.
    - `split_report.md`: Comprehensive audit report verifying 16 biological cows, 4 CCTV cameras, and 0 multi-camera / time-block leakage.
- `id/`:
  - `id_index.csv`: 4,736-row backward-compatible index mapping OpenCows2020 images to 46 cow classes and contiguous frame-index splits.
  - `opencow2020/`:
    - `manifest.csv`: 4,736-row master manifest mapping all images to cow ID, frame ID, official split, new split, SHA256, width, and height.
    - `train.csv` (3,586 rows), `val.csv` (654 rows), `test.csv` (496 rows): Contiguous frame-index split manifests with duplicate harmonization (0 exact-duplicate overlap).
    - `split_report.md`: Forensic audit report detailing legacy random within-identity mixing (1,023 frame-index adjacency crossings) and contiguous heuristic rebuild.
  - `sideviewcows2026/`:
    - `manifest.csv`: 80,260-row master manifest mapping all images and masks across 110 cows, recording IDs, and subset types.
    - `protocol_cross_setting.csv`: 80,260-row Protocol A manifest (parlor gallery vs barn/snapshots queries).
    - `protocol_longitudinal.csv`: 80,260-row Protocol B manifest (early parlor gallery vs late parlor query vs long-range barn/snapshots).
    - `protocol_open_set.csv`: 80,260-row Protocol C manifest (77 Train / 11 Val / 22 Test cows, 100% disjoint).
    - `protocol_closed_set.csv`: 80,260-row Protocol D manifest (110 cows with 70/15/15 parlor session split + barn/snapshots test).
    - `leakage_audit.csv`: Verification scorecard confirming 0 exact duplicates, 0 adjacent-frame leakage, and min perceptual distance 7 bits.
    - `split_report.md`: Detailed audit report with mathematical distribution proofs.
  - `external/sideviewcows2026/`: 80,260 images + 80,260 binary segmentation masks across 110 biological cows in snapshots (607), parlor (54,393), and barn (25,260) subsets (Zenodo record 21605650). Designated Primary Re-ID dataset under approved contingency.
  - `external/beca/`: BECA-D (16,889 images across 5,661 beef cattle; external scale stress benchmark) and BECA-L (12,172 images across 103 beef cattle tracked over 7+ months across 134 dates; primary external longitudinal Re-ID benchmark).
- `viewpoint/`:
  - `self/`: Raw, untouched self-collected cattle viewpoint dataset (1,057 files, 1,050 images across 7 subdirectories with .txt/.csv provenance).
  - `self_clean_v1/`:
    - `front/` (392 clean images), `rear/` (266 clean images), `side/` (222 clean images): Canonical 3-class normalized viewpoint dataset (880 images total).
    - `metadata/`:
      - `manifest.csv`: 1,050-row master audit manifest with 18 columns (`clean_id`, `clean_path`, `normalized_class`, `original_path`, `original_label`, `source_url`, `source_domain`, `width`, `height`, `sha256`, `perceptual_hash`, `duplicate_group_id`, `duplicate_status`, `stock_flag`, `quality_status`, `inclusion_status`, `exclusion_reason`, `notes`).
      - `review_required.csv`: 33-row review manifest flagging commercial stock exclusions and edge-case candidates (100% resolved by Hasin Ishrak).
      - `cleaning_report.md`: Comprehensive audit and leakage report documenting all rules and counts.
  - `self_clean_v1_rtdetr_crop/`:
    - `front/` (392 crops), `side/` (222 crops), `rear/` (265 crops): Derived RT-DETR-L cattle-cropped real viewpoint dataset (879 crops total, JPEG Quality 95, 1,867.40 MB / 1.824 GB; -46.8% size reduction vs source).
    - `train.csv` (616 crops, 70.1%), `val.csv` (132 crops, 15.0%), `test.csv` (131 crops, 14.9%): Deterministic Seed-2026 group-stratified splits partitioned strictly by `duplicate_group_id` with 0 cross-split leakage (also mirrored in `splits/`).
    - `split_report.md`: Detailed split audit report documenting class balance, 100% group isolation, and SHA-256 cryptographic checksums.
    - `metadata/`:
      - `crop_manifest.csv`: Master 879-row manifest with 26 columns containing bounding box coordinates, 5% proportional margin parameters, dimensions, file sizes, and preserved `duplicate_group_id` provenance.
      - `detection_failures.csv`: 1-row quarantine record documenting the single detection failure (`rear_0003`, working oxen with farmer) without fabricating crops.
      - `crop_report.md`: Comprehensive audit report detailing counts, class distributions, and compression metrics.
- `lameness/`:
  - `cattle_lameness_manifest.csv`: 50-clip master manifest defining filename, class, source URLs, proposed group IDs, confidence scores, evidence, and 5-fold cross-validation assignments.
  - `lameness_index.csv`: Extracted frame index mapping 9,950 frames across 50 video clips to labels and splits.
  - `frames/`: Directory containing 9,950 resized 224x224 RGB video frames extracted from CattleLameness clips.
  - `CattleLameness/`: Cloned source repository containing raw 500x500 MP4 clips across `Data/Lame` and `Data/Normal`.

### `final_models/`
Pre-trained object detection and feature extraction weight checkpoints.
- `p2 archive/`: Archive folder for legacy Phase 2 model weights.
  - `yolov8n.pt`: Legacy YOLOv8-nano weights from Phase 2 cow detection / localization pipeline.

### `P2 Samples/`
Reference materials, prior sample defense posters, and official CSE400 formatting templates from BRAC University.

- `MODAL_PROFILES.md`: Cheatsheet and multi-account mapping across all 6 Modal profiles (`tigerwood693`, `tigerwood697`, `hasinishrak74001`, `dryousufmozumder`, `hasinishrak2015`, `mohtasimahmedsamii`) with role descriptions, command reference, and grant tracking.
### `scripts/`
Automation utilities for batch experiments, metric aggregation, dataset restoration, and environment setup.
- `build_behavior_perception_cache.py`: Real Behavior perception cache generator for Phase 3 Run 5. Implements exact CVB ground-truth target tracklet bounding-box lookups (`CVBExactAnnotationCache`) with full-frame SAM 2.1 Small segmentation; implements Kaggle Beef RT-DETR-L largest box + positive center point prompting (A5) with center point (112, 112) fallback. Extracts cattle RGB crops aligned with binary masks ({0, 255} PNG), asserts zero-dummy masks, and records `perception_manifest.csv` and `perception_summary.json`.
- `train_cvb_beef_behavior_tcn.py`: Phase 3 Run 5 Behavior Temporal Core and Perception Integration training engine. Supports dual modes: `input_mode="rgb"` (historical 3-channel baseline, 11,900,485 params) and `input_mode="rgb_mask"` (real Run 5 4-channel perception model, 11,903,621 params). Couples ImageNet-pretrained ResNet-18 (512-D per frame) with lightweight 1D TCN (2 Conv1d blocks, GELU, BatchNorm1d, Dropout=0.2, AdaptiveAvgPool1d, Linear head). Implements deterministic T=8 temporal sampling, authentic CVB target tracklet bbox preservation, Kaggle Beef single-cow clip sampling, and visual perception contact sheet generation.
- `modal_train_cvb_beef_behavior_tcn.py`: Modal cloud wrapper for Run 5 Behavior Perception Smoke Test on profile `tigerwood693` (NVIDIA T4 GPU, 2 CPUs, 8GB RAM), mounting `cvb-data`, `beef-behavior-data`, and `behavior-checkpoints`. Manages Ultralytics weights caching, perception cache generation, 4-channel training, checkpoint verification, and artifact serialization.
- `crop_self_viewpoint_rtdetr.py`: Standalone reproducible RT-DETR-L cattle localization and cropping pipeline with 5% margin expansion for `self_clean_v1/`, generating 879 crops in `self_clean_v1_rtdetr_crop/`.
- `build_viewpoint_crop_splits.py`: Deterministic Seed-2026 group-stratified train/val/test split generator partitioned strictly by `duplicate_group_id` for `self_clean_v1_rtdetr_crop/` with 0 cross-split leakage.
- `upload_viewpoint_crops_modal.py`: Resumable live-progress Modal volume uploader for `self_clean_v1_rtdetr_crop/` to volume `viewpoint-real-data` on profile `tigerwood693`, with automated zero-GPU verification container.
- `train_moo_real_viewpoint.py`: Core PyTorch fine-tuning engine transferring MOO ResNet-18 synthetic backbone weights, replacing 8-class head with 3-class linear head (`front`, `side`, `rear`), with label-preserving bilateral flips, CosineAnnealingLR, AdamW, and selection by `val_macro_f1`.
- `modal_train_moo_real_viewpoint.py`: Modal cloud wrapper mounting `viewpoint-real-data`, `moo-data`, and `viewpoint-checkpoints` on profile `tigerwood693` for Tesla T4 GPU training and zero-GPU readiness verification (`verify_readiness`).
- `extract_cvb_track_segments.py`: Remote Modal data extraction script (`cvb-data` volume on `tigerwood693`, minimal CPU/RAM) parsing 1,163,408 bounding boxes from 502 CVB `instances_default.json` files into contiguous single-behavior track segments.
- `probe_beef_clips_ffprobe.py`: Remote Modal ffprobe script (`beef-behavior-data` volume on `tigerwood693`, minimal CPU/RAM) extracting exact fps, n_frames, and duration_sec across all 4,337 Kaggle Beef clips.
- `build_cvb_beef_behavior_protocol.py`: Deterministic protocol generation and verification suite implementing multi-objective group-stratified search (Seed 2026), building `datasets/behavior/cvb_beef/` manifests, and running rigorous anti-leakage assertions.
- `audit_cvb_stage1.py` & `audit_cvb_stage2.py`: Diagnostic forensic scripts auditing CVB filesystem structure, JSON schemas, and bounding box behaviors on Modal.
- `audit_beef_behavior_stage4.py`: Forensic audit script evaluating Kaggle Beef video clips, frame counts, and session distributions.
- `build_real_cattle_quality_contact_sheets.py`: Deterministic generation script for the 1,000-image real-cattle visual quality contact-sheet pack. Generates 42 high-resolution 5x5 sheets with visible metadata bars across ScienceDB, MmCows, and SideViewCows2026, and writes `phase3_real_cattle_visual_quality_contact_sheet_index.md`.
- `modal_moo_pipeline.py`: Modal cloud pipeline for MOO (Multi-view Oriented Observations) dataset. Ingests 34.03 GB archive into Modal volume `moo-data`, extracts 55.24 GB `data.hdf5` and 136.36 MB `metadata.json`, inspects metadata/HDF5 structure, and trains a 5-class linear viewpoint head on frozen ImageNet ResNet-18 in 0.64s on CPU (`moo_resnet18_viewpoint.pth`).
- `modal_mmcows_pipeline.py`: Modal cloud pipeline for MmCows Behavior dataset. Downloads 12.7 GB `cropped_bboxes.zip` via Rust `hf_transfer` in ~90s into volume `mmcows-data` on `tigerwood697`, unzips all 213,686 behavior crops, and purges zip archive.
- `modal_sciencedb_pipeline.py`: Modal cloud pipeline for ScienceDB Cattle BCS dataset. Downloads 4.11 GB `dataset.rar` via 16-stream `aria2c` from `china.scidb.cn` into volume `sciencedb-data` on `tigerwood697`, unzips all 107,132 files across all 5 classes (`3.25`..`4.25`) via `unar` (handling RAR5 format), and purges `.rar`.
- `evaluate_moo_viewpoint.py`: Benchmarks MOO-trained ResNet-18 on the 100-sample real cattle benchmark using RT-DETR-L target crops. Computes physical accuracy, Macro-F1, confusion matrix, false-front count, and executes systematic 90-degree coordinate rotation and axis-swap search (proving 63.16% accuracy under corrected alignment).
- `build_viewpoint_contact_sheets.py`: Deterministic generation script for Step 2.4 viewpoint manual review pack. Extracts 60 diverse samples (20 per dataset) across ScienceDB, MmCows, and SideViewCows2026, generates `viewpoint_manual_review_manifest.csv`, stitches 6 high-resolution 2-column contact sheets (10 images each), and builds `phase3_viewpoint_visual_review_index.md`.
- `build_viewpoint_expanded_crosscheck_pack.py`: Deterministic generation script for Step 2.4 100-sample expanded viewpoint review pack. Extracts 100 non-overlapping samples (ScienceDB 34, MmCows 33, SideView 33; seed 2026), stitches 10 blind contact sheets, and creates `phase3_viewpoint_expanded_crosscheck_index.md`.
- `audit_beef_rtdetr_failure_fallback.py`: Kaggle Beef fallback perception audit script evaluating single positive center point (112, 112) -> SAM 2.1 Small on the 3 known RT-DETR-L upstream detection failures from the fresh-40 audit on Modal (`tigerwood693`, T4).
- `audit_beef_A4_A5_fresh40.py`: Kaggle Beef SAM 2.1 head-to-head comparison script between detector-guided condition A4 (RT-DETR-L largest box -> SAM 2.1) and condition A5 (RT-DETR box + center point -> SAM 2.1) across 40 fresh canonical training frames on Modal (`tigerwood693`, T4). Generates 40 3-panel composites and 4x10 master contact sheet.
- `audit_beef_sam_prompt_rescue.py`: Kaggle Beef SAM 2.1 prompt-rescue audit script evaluating 6 prompt conditions (A0 baseline full crop, A1 center point, A2 multi-positive 5 pts, A3 center pos + neg corners, A4 largest RT-DETR box, A5 RT-DETR box + center point) across 20 training frames on Modal (`tigerwood693`, T4). Generates 120 visual composites and 6 master contact sheets.
- `audit_behavior_primary_segmentation.py`: Small SAM 2.1 (`sam2.1_s.pt`) segmentation sanity check script evaluating Prompt A (official target GT bbox) and Prompt B (matched RT-DETR-L bbox) on 25 CVB frames and full-crop box prompt on 20 Kaggle Beef crops on Modal (`tigerwood693`, T4). Generates 70 visual composites and 3 master contact sheets.
- `audit_behavior_primary_localization.py`: Empirical RT-DETR-L (`rtdetr-l.pt`) localization sanity check script evaluating 45 midpoint frames (25 CVB, 20 Kaggle Beef) across 5 canonical behaviors on Modal (`tigerwood693`, T4).
- `audit_pose_feasibility.py`: Deterministic cattle pose / keypoint feasibility auditor (Step 2.3) evaluating official DeepLabCut SuperAnimal-Quadruped (HRNet-W32 and ResNet-50) top-down pose estimation on RT-DETR-L target-cow crops across ScienceDB, MmCows, and SideViewCows2026. Supports smoke (30 imgs) and expanded (300 imgs) modes, 4-stage failure categorization (`upstream_localization_failure`, `pose_detector_failure`, `pose_output_returned`, `pose_inference_error`), raw confidence preservation, dynamic keypoint schema extraction, SideView ground-truth mask sanity check, crash-safe `--resume`, and visible progress display.
- `repair_sciencedb_volume.py`: Modal cloud volume repair and exhaustive verification script (`tigerwood697`). Verifies pre-repair state for 1,753 affected zero-byte JPG files, applies minimal 84.75 MB patch archive (`sciencedb_patch_1753.zip`) overwriting 0-byte stubs, verifies 100% of patched files, performs an exhaustive 64-worker multi-threaded PIL readability audit across ALL 53,566 images (verifying 0 zero-byte files, 53,566 valid PIL opens, 0 decode failures, exact class counts: 7536, 13256, 14255, 12556, 5963), and validates canonical split hashes (`train`, `val`, `test`).
- `modal_train_sciencedb_bcs.py`: Modal cloud wrapper for Phase 3 ScienceDB RGB single-task BCS baseline training on Modal (`tigerwood697`, NVIDIA L4 GPU with 24GB VRAM; initial readiness verified on Tesla T4, upgraded & re-verified on NVIDIA L4), mounting `sciencedb-data` at `/data` and `sciencedb-checkpoints` at `/checkpoints`, featuring live batch-level tqdm streaming, automated per-epoch persistent checkpoint commits, hardened pre-flight volume verification checking `st_size > 0` on all 53,566 images, and full pre-flight readiness audit suite (`verify_readiness`).
- `build_sciencedb_perception_cache.py`: Deterministic ScienceDB RT-DETR-L cattle localization + SAM 2.1 foreground mask preprocessing cache generator with 5% margin, provenance tracking, and explicit failure recording without box fabrication.
- `train_sciencedb_bcs_perception.py`: Phase 3 Run 4 BCS perception-enhanced training and evaluation pipeline on canonical 5,653-burst-group split with 4-channel ResNet-18 (`[R, G, B, Mask]`, +3,136 parameters), ImageNet RGB weights, Frank & Hall ordinal BCE head, and synchronized spatial augmentations.
- `modal_train_sciencedb_bcs_perception.py`: Modal cloud wrapper for Run 4 cache generation (NVIDIA L4) and full training (NVIDIA L40S) on profile `tigerwood697`, mounting `sciencedb-data`, `sciencedb-perception-cache`, and `sciencedb-checkpoints` with live tqdm streaming.
- `build_bcs_perception_contact_sheet.py`: Visual audit utility generating 4-panel visual contact sheets (`[Original + BBox | Crop | SAM 2.1 Mask | 4-Ch Guided Input]`) across all 5 BCS classes.
- `train_sciencedb_bcs_baseline.py`: Complete Phase 3 ScienceDB RGB single-task BCS baseline training and evaluation pipeline on canonical 5,653-burst-group split with ImageNet ResNet-18, Frank & Hall (2001) `ordinal_bce` vs Cao et al. (2020) `coral` heads, physiological BCS MAE calculation (scale 3.25 to 4.25), runtime path remapping (`resolve_image_path`) for Windows/Linux cross-compatibility, live batch-level `tqdm` progress bars, bit-identical checkpoint determinism, strict test-set isolation in smoke mode, and full-run/dry-run CLI support.
- `audit_segmentation_feasibility.py`: Deterministic multi-model cattle segmentation feasibility auditor (Step 2.2) evaluating Pipeline A (RT-DETR-L box -> SAM 2.1 small), Pipeline B (YOLO26s-seg), and Diagnostic Pipeline (Oracle GT Box -> SAM 2.1) across ScienceDB, MmCows, and SideViewCows2026 with clean single-line progress UI, SideView IoU/Dice calculation, and 4-panel composite generator.
- `audit_localization_feasibility.py`: Deterministic multi-model cattle localization feasibility auditor (Step 2.1) evaluating YOLOv8s, Faster R-CNN v2, and RT-DETR-L in smoke (90 images) and expanded (300 images) modes with Windows-clean single-line progress UI and automated composite generator.
- `aggregate.py`: Collects training logs and performance metrics across member workspaces and generates consolidated summary tables.
- `build_dataset_registry.py`: Deterministic generation script for `datasets/dataset_registry.csv` compiling Phase 3 datasets, roles, and local physical status.
- `build_sciencedb_splits.py`: ScienceDB identity audit, de-leakage grouping (5,662 passage clusters), and stratified train/val/test split builder.
- `repair_sciencedb_splits.py`: ScienceDB burst-group clustering via DSU (dHash/aHash <= 2, pixel MAE <= 5.0), repaired 5,653 burst groups, stratified 70/15/15 train/val/test splits, and verified zero cross-burst overlap.
- `build_mmcows_splits.py`: MmCows provenance audit, synchronized multi-camera protection, canonical baseline split, and 4-Fold GroupKFold suite generator.
- `verify_mmcows_splits.py`: Standalone assertion and verification suite checking MmCows file existence, 100% cow disjointness, multi-camera event protection, and path resolution.
- `build_dryad_manifest.py`: Dryad BCS discrepancy audit, biological cow parser (54 cows), master manifest generator, and bcs_index.csv updater.
- `build_opencows_splits.py`: OpenCows2020 legacy Re-ID protocol builder, duplicate harmonizer, manifest generator, and split report author.
- `verify_opencows_splits.py`: Standalone verification script asserting OpenCows2020 46 cows across all splits, 0 duplicate leakage, official test set integrity, and path resolution.
- `clean_self_viewpoint.py`: Standalone reproducible Python cleaning, deduplication, and 3-class normalization pipeline for the raw self-collected cattle viewpoint dataset (`datasets/viewpoint/self`), outputting to `datasets/viewpoint/self_clean_v1/`.
- `finalize_human_review.py`: Standalone script applying human review adjudications to `review_required.csv`, `manifest.csv`, and `cleaning_report.md` with 0 pending items.
- `build_sideview_reid_protocols.py`: SideViewCows2026 4-protocol generator, session recovery, exact and perceptual near-duplicate auditor with 7-stage Windows progress UI and standalone --verify-only mode.
- `build_leakage_safe_manifest.py`: Audits CattleLameness clips, applies heuristic & perceptual clustering, generates StratifiedGroupKFold assignments, and exports `cattle_lameness_manifest.csv`.
- `generate_doc.py`: Generates formatted documentation and reports from raw markdown and text data.
- `run_all_training.py`: Orchestrates multi-gpu batch execution of all task training scripts.
- `download_all.py`: Master dataset pipeline orchestrating restoration and indexing across all active thesis tasks directly into `datasets/`.
- `download_sciencedb.py`: Direct downloader, resume-supported streamer, unrar extractor, and preprocessor for ScienceDB Cattle BCS dataset.
- `fast_download_sciencedb.py`: High-speed multi-threaded (16-stream parallel) resumable chunk downloader and extractor for ScienceDB Cattle BCS dataset to bypass GFW throttling.
- `repair_sciencedb_splits.py`: ScienceDB burst-group repair script merging 1-frame-shifted video passages into 5,653 connected burst groups via DSU and creating 70/15/15 stratified split.
- `build_sideview_reid_protocols.py`: SideViewCows2026 session recovery and 4-protocol generator with 7-stage progress bar and leakage audit.
- `build_manual_dataset_visual_verification.py`: Deterministic generator for manual human visual inspection pack across ScienceDB, MmCows, and SideViewCows2026 with 4-stage progress UI and 3-panel mask composites.
- `audit_behavior_dataset_candidates.py`: Forensic audit script comparing MmCows vs. CBVD-5 across 10 dimensions, scanning video properties, analyzing crop sizes, generating 14 paired comparison review assets, and producing evaluation guides.
- `build_mmcows_behavior_contact_sheets.py`: Generates 7 high-resolution contact sheets (3x3 grid, 9 samples per class, 560x445 px per tile, 1720x1450 px per sheet) with provenance metadata banners for MmCows behavior visual verification.
- `build_mmcows_sequential_contact_sheets.py`: Generates 8 panoramic sequential filmstrips (6 consecutive frames per sequence, +0s to +75s @ 15s intervals) and dynamic behavior transition sheets for MmCows temporal continuity verification.
- `fast_download_sideviewcows.py`: Multi-threaded (16-thread) resumable chunk downloader and 7-Zip extractor for SideViewCows2026.
- `fast_download_beca.py`: Multi-threaded (16-thread) resumable chunk downloader and extractor for BECA dataset.
- `download_cbvd5.py`: Automated Kagglehub downloader and normalizer for CBVD-5 dataset.
- `billing_monitor.py`: Real-time multi-account Modal billing monitor and dashboard updater. Discovers accounts, pulls metered/billed cost via Modal CLI, calculates remaining credit balances and GPU runtimes, and writes to `BILLING.md`.
- `modal_billing.py`: Instant one-shot terminal summary runner for Modal billing.
- `pcgrad.py`: NeurIPS 2020 Projecting Conflicting Gradients implementation for mitigating negative gradient interference across multitask heads.
- `setup_research_pc.ps1`: 1-click lab PC bootstrap script for Git pull, virtual environment setup, and dependency installation.

### `artifacts/`
Generated manifests, audit CSVs, and model evaluation outputs.
- `behavior_temporal_smoke/`: Phase 3 Run 5 Behavior Temporal Core smoke test artifacts on profile `tigerwood693` (NVIDIA T4), containing `behavior_tcn_metrics.json` (architecture parameters, balanced smoke counts by dataset and class, loss, validation metrics, and bit-identical checkpoint reload verification) and `temporal_samples_contact_sheet.jpg` (visual verification of 8-frame progression across CVB and Beef sequences).
- `bcs_baseline/`: Phase 3 single-task ScienceDB BCS baseline outputs, containing `bcs_baseline_metrics.json` (structured training curves, provenance metadata, split hashes, full 30-epoch history, per-class metrics, and confusion matrix), `bcs_baseline_30epoch_summary.md` (30-epoch training and test evaluation report), and `bcs_baseline_smoke_summary.md` (smoke test summary).
- `bcs_perception_smoke/`: Phase 3 Run 4 BCS perception pipeline smoke test artifacts, containing `cache_schema.json` (canonical 18-column manifest schema defining crop geometry, detection status, SAM return codes, and fallback semantics), smoke split CSVs, metrics, and checkpoint resumption logs.
- `perception_audit/`:
  - `beef_rtdetr_failure_fallback.csv`: 3-row evaluation results verifying the fallback rule (IF RT-DETR detects no cow -> SAM 2.1 with ONE center point (112, 112)) on the 3 known detection misses from fresh-40 audit, recording mask return flags, area ratios, connected components, and latencies.
  - `beef_A4_A5_fresh40.csv`: 80-row evaluation results comparing detector-guided conditions A4 vs A5 on 40 fresh Kaggle Beef canonical train frames (Seed 2026, 29 sessions) with prompt coordinates, return flags, area ratios, components, RT-DETR detection counts, and latencies.
  - `beef_sam_prompt_rescue.csv`: 120-row prompt-rescue audit results of SAM 2.1 Small (`sam2.1_s.pt`) across 6 prompt conditions (A0-A5) on 20 Kaggle Beef frames with coordinates, return flags, areas, components, and detector stats.
  - `behavior_primary_segmentation_sanity.csv`: 45-row audit results of SAM 2.1 Small (`sam2.1_s.pt`) across CVB Prompt A, CVB Prompt B, and Kaggle Beef full-crop prompt with area ratios, sanity metrics, and components.
  - `behavior_primary_localization_sanity.csv`: 45-row audit results of RT-DETR-L (`rtdetr-l.pt`) across 25 CVB frames and 20 Kaggle Beef clips with IoU metrics, detection counts, and area ratios.
  - `viewpoint_manual_review_manifest.csv`: 60-row verified manual review manifest mapping sample IDs across ScienceDB, MmCows, and SideViewCows2026 to verified viewpoint categories (`rear`, `rear-oblique`, `side`, `front-oblique`, `unknown / ambiguous`), provenance notes, and review status `human_verified`.
  - `pose_manual_review.csv`: 60-row verified manual review record for Step 2.3 pose feasibility across ScienceDB, MmCows, and SideViewCows2026 with ratings and review provenance.
  - `sample_manifest_smoke.csv` (90 samples) & `sample_manifest_expanded.csv` (300 samples): Deterministic stratified sample manifests across ScienceDB, MmCows, and SideViewCows2026.
  - `localization_detections_smoke.csv` & `localization_detections_expanded.csv`: Bounding box detection records from YOLOv8s, Faster R-CNN v2, and RT-DETR-L.
  - `localization_summary_smoke.csv` & `localization_summary_expanded.csv`: Aggregated detection counts, raw detection rates, and latencies.
  - `segmentation_results_smoke.csv` & `segmentation_results_expanded.csv`: Step 2.2 segmentation results comparing RT-DETR-L -> SAM 2.1 small, YOLO26s-seg, and Oracle GT -> SAM 2.1 with SideView IoU/Dice metrics.
  - `pose_results_expanded_hrnet_w32.csv` & `pose_results_expanded_resnet_50.csv`: Step 2.3 pose evaluation results across 300 samples per backbone.
  - `pose_keypoints_expanded_hrnet_w32.csv` & `pose_keypoints_expanded_resnet_50.csv`: Extracted 39-keypoint coordinates and confidence scores across 300 samples per backbone.
  - `superanimal_quadruped_schema.json`: Official DeepLabCut 39-keypoint quadruped schema and body part index mapping.

### `videos/`
Test video files and sample inference output clips demonstrating real-time bounding box detection, tracking, and multitask predictions.

### `workspaces/`
Member-specific experimental sandboxes containing model training scripts, loss curves, evaluation outputs, and PyTorch `.pth` model checkpoints.
- `workspaces/hasin/`: Hasin's experimental sandbox (ResNet-18 baseline and MTL implementations, loss curves, evaluation results for ID, BCS, behavior, and lameness).
- `workspaces/nusrat/`: Nusrat's sandbox focusing on temporal MTL architectures, CBAM attention mechanisms, and video inference scripts.
- `workspaces/shouvik/`: Shouvik's sandbox with standalone task training runs and comparative benchmarks.
- `workspaces/bithi/` & `workspaces/namira/`: Additional team member ablation and experimental directories.

### Root Workspace Files
- `BILLING.md`: Live auto-refreshed Modal billing and credit dashboard tracking all 6 accounts, credit grants ($122.00), metered spend, remaining balance, and L40S/H100 runtime estimates.
- `monitor.py`: Live auto-looping Modal billing daemon that displays the initial scoreboard and continuously refreshes `BILLING.md` every 10s until `Ctrl+C`.
- `billing_monitor.py`: Root entry-point wrapper to run billing monitor (`python billing_monitor.py` or `python billing_monitor.py --loop`).
- `phase3_canonical_roadmap.md`: Master canonical 13-step Phase 3 execution roadmap.
- `AGENTS.md`: Workspace rules, research log requirements, and memory synchronization guidelines.
- `README.md`: Workspace introduction and quick overview.
