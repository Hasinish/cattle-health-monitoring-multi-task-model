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

### `phase3_canonical_roadmap.md` & `docs/phase3_canonical_roadmap.md`
Master canonical 13-step roadmap locked for Phase 3 execution. Formulates the core thesis question: *"Which cattle-specific visual priors (localization, soft masks, anatomy/pose, viewpoint) are useful for which downstream task, and what information should each task preserve or suppress?"* Locks downstream scope to BCS (ScienceDB; Ruchay 2026 external), Behavior (MmCows; CBVD-5 external), and Re-ID (MultiCamCows2024 replacing OpenCows2020; SideViewCows2026 external). Outlines a strict 13-step progression from Step 1 data registry to final 3-seed benchmark tables.

### `docs/`
Structured project documentation, defense resources, forensic audits, and official deliverables.
- `README.md`: Master directory guide for the organized `docs/` workspace.
- `phase3_canonical_roadmap.md`: Canonical Phase 3 execution roadmap mirror.
- `audits/`: Detailed dataset forensic investigations, anti-leakage manifests, and cross-validation architectures.
  - `candidate_lameness_datasets_audit.md`: Deep forensic audit of 4 potential alternative lameness datasets (Russello 2026, Wu/NWAFU, whsu2s, Duan 2025).
  - `cattle_lameness_audit_report.md`: Forensic audit of the CattleLameness dataset (50 clips, 42 cattle, ezgif container footprints, cross-split leakage identification).
  - `cattle_lameness_grouping_report.md`: Anti-leakage clustering architecture, multi-clip group resolution (8 groups, 16 clips), and balanced 5-fold StratifiedGroupKFold cross-validation specification.
- `research_log/`: Centralized research log repository documenting experiments, dataset investigations, architectural decisions, and ablation studies.
  - `README.md`: Research log protocol, entry structure guidelines, and historical log index table.
  - `2026-09-20_sciencedb_identity_audit_and_leakage_free_split.md`: Forensic audit of ScienceDB identity parser, disproving 10,898 cows claim, exposing 94.6% stereo leakage in legacy split, and building leak-free 5,662-passage split.
  - `2026-09-20_sciencedb_roadmap_correction.md`: Formal roadmap correction replacing ScienceDB cow-disjoint wording with passage-disjoint / sequence-safe evaluation and setting MmCows grouped evaluation as the next Step 1 action.
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
    - `train.csv` (37,126 rows), `val.csv` (8,099 rows), `test.csv` (8,341 rows): 100% passage-disjoint split manifests.
    - `identity_audit.csv`: Complete audit cataloging all 5,662 passage clusters and label distributions.
    - `split_report.md`: Forensic audit report disproving 10,898 cows claim and documenting de-leakage methodology.
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
  - `mmcows/`:
    - `manifest.csv`: 213,686-row master manifest mapping all crops to cow ID, camera ID, epoch, ISO timestamp, time block, synchronized event ID, canonical split, and folds 0–3.
    - `provenance_audit.csv`: Cow-level provenance audit table summarizing all 16 cows across 20 parameters.
    - `train.csv` (148,401 rows), `val.csv` (25,134 rows), `test.csv` (40,151 rows): Leakage-safe canonical split manifests.
    - `folds/`:
      - `fold_0.csv`, `fold_1.csv`, `fold_2.csv`, `fold_3.csv`: 4-Fold GroupKFold cross-validation suite (each 213,686 rows) evaluating 100% of cows with guaranteed 7-class positive coverage.
    - `split_report.md`: Comprehensive audit report verifying 16 biological cows, 4 CCTV cameras, and 0 multi-camera / time-block leakage.
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

### `scripts/`
Automation utilities for batch experiments, metric aggregation, dataset restoration, and environment setup.
- `aggregate.py`: Collects training logs and performance metrics across member workspaces and generates consolidated summary tables.
- `build_dataset_registry.py`: Deterministic generation script for `datasets/dataset_registry.csv` compiling Phase 3 datasets, roles, and local physical status.
- `build_sciencedb_splits.py`: ScienceDB identity audit, de-leakage grouping (5,662 passage clusters), and stratified train/val/test split builder.
- `build_mmcows_splits.py`: MmCows provenance audit, synchronized multi-camera protection, canonical baseline split, and 4-Fold GroupKFold suite generator.
- `verify_mmcows_splits.py`: Standalone assertion and verification suite checking MmCows file existence, 100% cow disjointness, multi-camera event protection, and path resolution.
- `build_dryad_manifest.py`: Dryad BCS discrepancy audit, biological cow parser (54 cows), master manifest generator, and bcs_index.csv updater.
- `build_leakage_safe_manifest.py`: Audits CattleLameness clips, applies heuristic & perceptual clustering, generates StratifiedGroupKFold assignments, and exports `cattle_lameness_manifest.csv`.
- `generate_doc.py`: Generates formatted documentation and reports from raw markdown and text data.
- `run_all_training.py`: Orchestrates multi-gpu batch execution of all task training scripts.
- `download_all.py`: Master dataset pipeline orchestrating restoration and indexing across all active thesis tasks directly into `datasets/`.
- `download_sciencedb.py`: Direct downloader, resume-supported streamer, unrar extractor, and preprocessor for ScienceDB Cattle BCS dataset.
- `fast_download_sciencedb.py`: High-speed multi-threaded (16-stream parallel) resumable chunk downloader and extractor for ScienceDB Cattle BCS dataset to bypass GFW throttling.
- `download_mmcows.py`: High-speed Hugging Face automated downloader and extractor for MmCows behavior bounding boxes with automatic zip cleanup.
- `download_opencows.py`: Fast Kagglehub downloader and normalizer for OpenCows2020 identification images.
- `pcgrad.py`: NeurIPS 2020 Projecting Conflicting Gradients implementation for mitigating negative gradient interference across multitask heads.
- `setup_research_pc.ps1`: 1-click lab PC bootstrap script for Git pull, virtual environment setup, and dependency installation.

### `thesis template/`
Clean, official CSE400 LaTeX template skeleton used as the foundation for the thesis formatting.

### `videos/`
Test video files and sample inference output clips demonstrating real-time bounding box detection, tracking, and multitask predictions.

### `workspaces/`
Member-specific experimental sandboxes containing model training scripts, loss curves, evaluation outputs, and PyTorch `.pth` model checkpoints.
- `workspaces/hasin/`: Hasin's experimental sandbox (ResNet-18 baseline and MTL implementations, loss curves, evaluation results for ID, BCS, behavior, and lameness).
- `workspaces/nusrat/`: Nusrat's sandbox focusing on temporal MTL architectures, CBAM attention mechanisms, and video inference scripts.
- `workspaces/shouvik/`: Shouvik's sandbox with standalone task training runs and comparative benchmarks.
- `workspaces/bithi/` & `workspaces/namira/`: Additional team member ablation and experimental directories.
