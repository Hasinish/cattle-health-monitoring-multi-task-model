# Research Log: Lameness Dataset Forensic Audit, Leakage Resolution, and Candidate Repositories

**Date**: 2026-09-17  
**Author**: Hasin Ishrak  
**Supervision**: Dr. Md. Khalilur Rahman  
**Project**: Cattle Health Monitoring Multi-Task Deep Learning Model (BRAC University)

---

## 1. Executive Summary
Conducted an exhaustive forensic audit on the current Mendeley CattleLameness dataset (50 clips) and 4 candidate academic repositories. Uncovered severe train/test data leakage in the existing `context/preprocess_lameness.py` pipeline, where clips `N (9).mp4` (Train) and `N (3).mp4` (Test) originated from the exact same YouTube clip (`ntAGV4SQ0Fw`). Resolved all 50 clips into 42 distinct animal/source groups and generated a leak-proof 5-fold StratifiedGroupKFold cross-validation manifest. Furthermore, investigated 4 candidate lameness datasets: selected Russello 2026 (98 unique cows, 272 trajectories) as our primary candidate, while rejecting Wu NWAFU (static pose only), whsu2s (missing repository data), and Duan 2025 (closed/private data).

---

## 2. Part I: Mendeley CattleLameness Forensic Audit & Data Leakage Resolution

### 2.1 Dataset Inventory & Characteristics
- **Total Videos**: 50 MP4 files (25 Lame, 25 Normal).
- **Source**: Web-scraped clips from YouTube and TikTok, standardized using `ezgif.com` into 500x500 square crops at 25-30 FPS.
- **Frame Count**: 9,950 total extracted frames (5,050 Lame, 4,900 Normal).
- **Visual Artifacts**: Heavy watermarks (ezgif.com, TikTok handles), variable aspect ratio stretching, synthetic Blender animations (e.g., `L (19).mp4`), and inconsistent camera angles.

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

### 2.3 Master Grouping & Leakage-Safe Manifest
A dedicated grouping algorithm (`scripts/build_leakage_safe_manifest.py`) resolved the 50 clips into **42 distinct animal/source groups**:
- **Lame Class**: 21 unique groups (17 single-video, 4 multi-video groups).
- **Normal Class**: 21 unique groups (18 single-video, 3 multi-video groups).
- **Cross-Validation Split**: Configured a 5-fold `StratifiedGroupKFold` split ensuring:
  - Zero animal/source leakage across folds (all clips from a group reside exclusively in one fold).
  - Exactly 5 Lame clips and 5 Normal clips in every fold (total 10 clips per test fold).
- **Master Manifest**: Exported to `datasets/lameness/cattle_lameness_manifest.csv`.

---

## 3. Part II: Forensic Audit of 4 Candidate Lameness Datasets

To establish a gold-standard benchmark and eliminate synthetic/watermarked web-scraped clips, four candidate repositories were forensically evaluated:

### 3.1 Candidate Evaluation Matrix

| Candidate | Status | Samples / Sequence Count | Unique Cows | Modality | Ground Truth Lameness | Verdict & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Russello 2026** (`hrussel/lstm-lameness-detection`) | **ACCEPTED (Primary)** | 272 trajectories | **98 cows** | 2D Keypoints (17 T-LEAP points) | Scores 1-4 (143 Normal, 129 Lame) | **RECOMMENDED**: Clean, public, documented cow IDs, leak-free StratifiedGroupKFold. Ready to use. |
| **Wu Dairy Cow** (`wusaisa/Dairy-cow-dataset`) | **REJECTED** | 1,941 JPG images | Unknown | Static RGB + 16-point pose TXT | None (0) | **UNUSABLE**: Static pose estimation dataset for cow detection. Zero gait/lameness labels. |
| **whsu2s** (`whsu2s/Lameness-Detection`) | **REJECTED** | 0 videos / 0 JSONs in repo | 64 cows claimed | None (Data not pushed) | Mobility scores 0-3 | **UNUSABLE**: Repository only contains thesis text and label CSV. Video/skeleton data never pushed to GitHub. |
| **Duan et al. 2025** (Frontiers in Vet Sci) | **REJECTED** | 1,280 samples | Unknown | Overhead RGB-D | Multi-feature classification | **UNAVAILABLE**: Closed-source / private dataset requiring author inquiry. |

### 3.2 Deep Dive: Russello 2026 (`hrussel/lstm-lameness-detection`)
- **Repository Size**: 23 MB.
- **Data Files**:
  - `data/videos_keypoints/*.csv`: 272 coordinate trajectory CSVs.
  - `data/videos_lameness_scores.csv`: Video-level mapping with Cow ID, locomotion score (1=Sound, 2-4=Lame), and binary label.
  - `data/video_information.csv`: Frame counts (avg 100-300 frames per pass) and capture metadata.
- **Evaluation Standard**: Utilizes `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `cow_id`.
- **Note on Modality**: Contains extracted coordinate sequences rather than raw RGB MP4s. Can be modeled directly using 1D-CNN / BiLSTM or transformed into spatial coordinate heatmaps for ResNet-18.

---

## 4. Architectural Decisions & Strategic Roadmap

### Option A: Russello 2026 Keypoint Trajectory Pipeline (Gold Standard)
- Add a lightweight trajectory encoder (BiLSTM or 1D-CNN) alongside the multi-task ResNet-18 backbone.
- Alternatively, render 2D keypoint trajectory skeleton frames to feed standard visual backbones.
- **Pros**: Perfectly leak-proof, peer-reviewed, zero watermarks/CGI.

### Option B: CattleLameness RGB 5-Fold Cross-Validation (Robust Baseline)
- Update `context/preprocess_lameness.py` and PyTorch DataLoader to read fold assignments directly from `datasets/lameness/cattle_lameness_manifest.csv`.
- Discard random modulo splits permanently.
- **Pros**: Direct raw image features compatible with existing ResNet-18 multi-task architecture.

---

## 5. Artifacts and Generated Deliverables
1. **Manifest File**:
   - `datasets/lameness/cattle_lameness_manifest.csv`: 50-video leakage-safe master manifest with 42 groups and 5 stratified folds.
2. **Manifest Generation Script**:
   - `scripts/build_leakage_safe_manifest.py`: Script executing video metadata extraction, grouping logic, and StratifiedGroupKFold.
3. **Comprehensive Audit Reports**:
   - `docs/audits/cattle_lameness_audit_report.md`: Detailed audit of CattleLameness dataset and leakage diagnosis.
   - `docs/audits/cattle_lameness_grouping_report.md`: Deep dive into the 42 groups, visual evidence, and fold distributions.
   - `docs/audits/candidate_lameness_datasets_audit.md`: Full comparative audit of the 4 candidate repositories.

---

## 6. Immediate Next Steps
- [ ] Review decision between Option A (Russello keypoint trajectories) vs Option B (CattleLameness grouped RGB 5-fold CV).
- [ ] Update `context/preprocess_lameness.py` to enforce `cattle_lameness_manifest.csv`.
- [ ] Commit research log, audit docs, and manifest scripts to Git repository.
