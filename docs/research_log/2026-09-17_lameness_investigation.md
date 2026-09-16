# Research Log: Lameness Dataset Forensic Audit, Leakage Resolution, and Candidate Repositories

**Date**: 2026-09-17  
**Author**: Hasin Ishrak  
**Supervision**: Dr. Md. Khalilur Rahman  
**Project**: Cattle Health Monitoring Multi-Task Deep Learning Model (BRAC University)

---

## 1. Executive Summary
Conducted an exhaustive forensic audit on the current Mendeley CattleLameness dataset (50 clips) and 4 candidate academic repositories. Uncovered severe train/test data leakage in the existing `context/preprocess_lameness.py` pipeline, where clips `N (9).mp4` (Train) and `N (3).mp4` (Test) originated from the exact same YouTube clip (`ntAGV4SQ0Fw`). The automated grouping produced a provisional 42-group manifest matching the dataset authors' reported count of 42 cattle, but subsequent manual inspection showed that some visual identity judgments were unreliable. Several normal-class clips were also identified as CGI/Blender footage. Therefore, the grouping manifest is preserved as a historical forensic artifact and is NOT considered validated ground truth for final Phase 3 training. Furthermore, investigated 4 candidate lameness datasets: Russello 2026 was identified as the strongest publicly available candidate, while rejecting Wu NWAFU (static pose only), whsu2s (missing repository data), and Duan 2025 (closed/private data). Subsequently, lameness was removed from the primary Phase 3 multi-task experiment to focus on RGB-native tasks (BCS, Behavior, Cow ID).

---

## 2. Part I: Mendeley CattleLameness Forensic Audit & Data Leakage Resolution

### 2.1 Dataset Inventory & Characteristics
- **Total Videos**: 50 MP4 files (25 Lame, 25 Normal).
- **Source**: Web-scraped clips from YouTube and TikTok, standardized using `ezgif.com` into 500x500 square crops.
- **Framerates (FPS)**: Variable framerates ranging from 25.00 FPS up to 60.00 FPS (e.g., `L (2).mp4`, `L (5).mp4`, and `L (7).mp4` are 60.00 FPS).
- **Frame Count**: 9,950 total extracted frames:
  - **Lame**: 5,900 frames (59.30%)
  - **Normal**: 4,050 frames (40.70%)
- **Visual Artifacts**: Heavy watermarks (ezgif.com, TikTok handles), variable aspect ratio stretching, synthetic Blender animations, and inconsistent camera angles.

### 2.2 Leakage Discovery in Existing Splitting Code
Inspection of `context/preprocess_lameness.py` revealed that video assignment was conducted via random index modulo arithmetic without cow identity tracking:
- **Critical Leak**: `datasets/lameness/raw/Normal/N (9).mp4` and `datasets/lameness/raw/Normal/N (3).mp4` are temporal sub-segments of the identical YouTube short (`https://www.youtube.com/watch?v=ntAGV4SQ0Fw`). Under the old split, `N (9)` was placed in Train and `N (3)` in Test. The model was evaluating on frames of the exact same cow walking across the exact same barn background.
- Additional suspect clusters were identified based on identical camera viewpoints, coat patterns, and upload batches:
  - `L (1).mp4` and `L (2).mp4` (same cow/pen)
  - `L (10).mp4` and `L (11).mp4` (same recording)
  - `L (15).mp4` and `L (16).mp4` (same TikTok source)
  - `L (20).mp4`, `L (21).mp4`, and `L (22).mp4` (consecutive TikTok clips)
  - `N (4).mp4` and `N (5).mp4` (same cow/milking parlor walkway)
  - `N (12).mp4` and `N (13).mp4` (same pasture recording)

### 2.3 Provisional Grouping & Evaluation Protocol
An automated grouping script (`scripts/build_leakage_safe_manifest.py`) grouped the 50 clips into 42 provisional clusters matching the authors' reported 42-cow count:
- **Lame Class**: 21 provisional groups (17 single-video, 4 multi-video groups).
- **Normal Class**: 21 provisional groups (18 single-video, 3 multi-video groups).
- **Proposed Evaluation Protocol**: A 5-fold `StratifiedGroupKFold` split:
  - The proposed grouped folds reduce known source-level leakage, but because exact animal identities are not reliably available, zero animal-level leakage cannot be guaranteed.
  - Exactly 5 Lame clips and 5 Normal clips in every fold (total 10 clips per test fold).
- **Manifest Location**: `datasets/lameness/cattle_lameness_manifest.csv`.

> [!IMPORTANT]
> **STATUS: PROVISIONAL / HISTORICAL ONLY.**  
> This manifest is not approved for final Phase 3 training. The exact clip-to-animal grouping has not been independently verified. The manifest may still be useful for documenting known duplicate/source leakage, but it must not be treated as validated cow identity ground truth.

### 2.4 Manual Synthetic / CGI Findings
Manual visual inspection of the video sequences identified multiple synthetic clips:
- `N (7).mp4` — CGI
- `N (21).mp4` — same/related CGI from a different angle
- `N (6).mp4` — CGI with artificial grass/sky-style background
- `N (8).mp4` — CGI / synthetic footage
- `N (9).mp4` — 3D Blender footage

These findings further reduce confidence in CattleLameness as a valid real-world gait benchmark.

---

## 3. Part II: Forensic Audit of 4 Candidate Lameness Datasets

To evaluate alternatives and address synthetic/watermarked web-scraped footage, four candidate repositories were forensically investigated:

### 3.1 Candidate Evaluation Matrix

| Candidate | Status | Samples / Sequence Count | Unique Cows | Modality | Ground Truth Lameness | Verdict & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Russello 2026** (`hrussel/lstm-lameness-detection`) | **STRONGEST CANDIDATE (Pose-Only)** | 272 trajectories | **98 cows** | 2D Keypoints (17 T-LEAP points) | Scores 1-4 (143 Normal, 129 Lame) | **Strongest public candidate**. Documented cow IDs and animal-disjoint evaluation. Pose trajectories only (no RGB video). |
| **Wu Dairy Cow** (`wusaisa/Dairy-cow-dataset`) | **REJECTED** | 1,941 JPG images | Unknown | Static RGB + 16-point pose TXT | None (0) | **UNUSABLE**: Static pose estimation dataset for cow detection. Zero gait/lameness labels. |
| **whsu2s** (`whsu2s/Lameness-Detection`) | **REJECTED** | 0 videos / 0 JSONs in repo | 64 cows claimed | None (Data not pushed) | Mobility scores 0-3 | **UNUSABLE**: Repository only contains thesis text and label CSV. Video/skeleton data never pushed to GitHub. |
| **Duan et al. 2025** (Frontiers in Vet Sci) | **REJECTED** | 1,280 samples | Unknown | Overhead RGB-D | Multi-feature classification | **UNAVAILABLE**: Closed-source / private dataset requiring author inquiry. |

### 3.2 Findings on Russello 2026 (`hrussel/lstm-lameness-detection`)
- **Repository Size**: 23 MB.
- **Evaluation Protocol**: Russello 2026 was the strongest publicly available lameness candidate identified in this audit. Its released evaluation code uses animal-disjoint `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` based on cow IDs, which substantially reduces animal-level leakage risk.
- **Data Modality**: Contains 272 2D coordinate trajectory CSVs (`data/videos_keypoints/*.csv`) rather than raw RGB video clips.

---

## 4. Current Phase 3 Decision

Lameness has been removed from the primary Phase 3 multi-task experiment.

The main P3 tasks are now:
1. **Body Condition Scoring (BCS)**:
   - **Primary**: ScienceDB (53,566 RGB rear-view images across 5 classes)
   - **Secondary / Cross-Domain Comparison**: Dryad (5,923 Depth Grayscale Edge maps)
2. **Behavior Recognition** (Primary: MmCows — 213,686 bounding-box crops)
3. **Individual Cow Identification (Cow ID)** (Primary: OpenCows2020 — 4,736 images across 46 classes)

CattleLameness is retained only as:
- a historical Phase 2 dataset
- evidence of dataset-quality problems
- evidence of evaluation leakage risk
- an optional future stress-test if needed

Russello 2026 remains the strongest public lameness candidate identified during the audit. However, the released Russello data consist of pose/keypoint trajectories rather than raw RGB video. Because the main P3 multi-task architecture is currently focused on RGB-based tasks, Russello is not being integrated into the primary P3 model.

Possible future uses:
- supplementary lameness experiment
- future multimodal extension
- future work section

### Status Update / Superseding Decision
The initial grouping/CV plan for CattleLameness and the prospective integration of Russello trajectories were superseded by the decision to remove lameness from the primary Phase 3 multi-task experiment. This narrows the scope of Phase 3 to RGB-native vision tasks with validated ground truth.

---

## 5. Artifacts and Generated Deliverables
1. **Provisional Manifest File**:
   - [cattle_lameness_manifest.csv](../../datasets/lameness/cattle_lameness_manifest.csv): 50-video provisional manifest with 42 groups and 5 stratified folds (Historical / Non-validated).
2. **Manifest Generation Script**:
   - [build_leakage_safe_manifest.py](../../scripts/build_leakage_safe_manifest.py): Script executing metadata extraction, heuristic grouping, and StratifiedGroupKFold.
3. **Comprehensive Audit Reports**:
   - [cattle_lameness_audit_report.md](../audits/cattle_lameness_audit_report.md): Detailed audit of CattleLameness dataset and leakage diagnosis.
   - [cattle_lameness_grouping_report.md](../audits/cattle_lameness_grouping_report.md): Analysis of the provisional 42 groups, visual evidence, and fold distributions.
   - [candidate_lameness_datasets_audit.md](../audits/candidate_lameness_datasets_audit.md): Full comparative audit of the 4 candidate repositories.

---

## 6. Immediate Next Steps
- [x] Document forensic audit and candidate repository evaluations.
- [x] Supersede lameness integration; finalize P3 multi-task scope (BCS, Behavior, Cow ID).
- [ ] Download & restore ScienceDB BCS dataset (Primary, https://scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c) to `datasets/bcs/sciencedb_bcs/` and run `preprocess_sciencedb_bcs.py`.
- [ ] Place Dryad BCS archive (Secondary / Comparison, `Total_sorted_DGE_images.zip` from doi:10.5061/dryad.tqjq2bw4s) and run `preprocess_bcs.py`.
- [ ] Finalize data loaders and PCGrad multi-task training script for the 3 core RGB tasks.
