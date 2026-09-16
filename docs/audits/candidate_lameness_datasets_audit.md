# Forensic Audit: Candidate Cattle Lameness Datasets & Repositories

**Date**: September 17, 2026  
**Auditor**: AntiGravity Coding Engine  
**Objective**: Evaluate four potential alternative lameness benchmark datasets to replace or augment the problematic CattleLameness YouTube dataset.

---

## Executive Summary

| Candidate | Verified Cattle Count | Verified Samples | Lameness Labels Present? | Modality | Download Status | Usability Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Russello 2026** (`hrussel`) | **98 cows** | **272 trajectories** | **YES** (143 sound, 129 lame) | 2D Keypoint Trajectories | **23 MB (Public Git)** | ✅ **PRIMARY RECOMMENDATION** |
| **2. Wu / NWAFU** (`wusaisa`) | 63 cows | 1,941 images | ❌ **NONE** (0 labels) | Static RGB + 16 KPs | 1.73 GB (Public GDrive) | ❌ **UNUSABLE** (Zero lameness data) |
| **3. whsu2s** (`whsu2s`) | 64 cows | Claimed 500 clips | In CSV only | Skeletons / MP4 | 48 MB (Clips omitted) | ❌ **UNUSABLE** (Ghost files not committed) |
| **4. Duan et al. 2025** | ~100+ cows | Claimed 741 seqs | In Paper only | RGB-D + 8 Dorsal KPs | Restricted / Private | ❌ **UNUSABLE** (Author permission required) |

---

## 1. Candidate 1: Russello 2026 (`hrussel/lstm-lameness-detection`)

* **Repository**: `https://github.com/hrussel/lstm-lameness-detection`
* **Paper Reference**: Russello, Helena, Rik van der Tol, Eldert J. van Henten, and Gert Kootstra. *"Lameness detection in dairy cows using pose estimation and bidirectional LSTMs."* *Smart Agricultural Technology* 8 (2026): 101831. ([DOI: 10.1016/j.atech.2026.101831](https://doi.org/10.1016/j.atech.2026.101831))
* **Inspection Methodology**: Cloned repository into scratch workspace, verified files, ran data loaders, and inspected cross-validation architecture.

### Detailed Findings
1. **Sample & Animal Counts**:
   * **Total Samples**: Exactly **272 keypoint trajectories** (`data/videos_keypoints/001.csv` to `272.csv`).
   * **Unique Cattle**: Exactly **98 cows** (column `ID` in `data/videos_lameness_scores.csv`). Verified with zero missing values.
2. **Class Distribution**:
   * Locomotion Score 1 (Normal / Sound): **143 trajectories** (52.6%)
   * Locomotion Score 2 (Mildly lame): **96 trajectories**
   * Locomotion Score 3 (Moderately lame): **20 trajectories**
   * Locomotion Score 4 (Severely lame): **13 trajectories**
   * **Binary Classification Split** (Score 1 vs Scores >= 2): **143 Normal / 129 Lame** (nearly balanced: 52.6% vs 47.4%).
3. **Modality & Keypoint Topology**:
   * **Format**: Temporal sequence of 2D keypoints `(x, y)` plus detection likelihoods over walking time frames.
   * **Raw Keypoints**: **17 keypoints** extracted by T-LEAP (LFHoof, LFAnkle, LFKnee, RFHoof, RFAnkle, RFKnee, LHHoof, LHAnkle, LHKnee, RHHoof, RHAnkle, RHKnee, Nose, HeadTop, Spine1, Spine2, Spine3).
   * **Model Keypoints**: The author's BiLSTM architecture (`models/KPLSTM.py`) subsamples **9 keypoints** (`input_dim = 9 * 2 = 18 coordinates`).
4. **Acquisition Protocol**:
   * Side-view camera mounted perpendicular to a controlled concrete exit walkway from an automated milking system at a dairy research farm.
   * Full body and all four limbs are consistently visible in straight-line walking locomotion.
5. **Raw Video Availability**:
   * **Raw RGB videos are NOT included** in the public repository. Only the extracted keypoint trajectory CSVs are provided.
6. **Train/Validation Split & Anti-Leakage Protocol**:
   * The repository uses animal-disjoint **StratifiedGroupKFold**:
     ```python
     # main.py, line 310
     sgkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
     for i, (train_idx, val_idx) in enumerate(sgkf.split(sk_dataset.data, sk_dataset.labels, sk_dataset.ids)):
     ```
   * **Leakage Risk**: **ZERO animal-level leakage**. Multiple clips from the same cow ID are strictly confined to the same fold.
7. **Usability Verdict**: **USABLE IMMEDIATELY** for temporal and skeleton-based architectures (LSTM, 1D-CNN, ST-GCN, Transformer).

---

## 2. Candidate 2: Wu Dairy Cow Dataset (`wusaisa/Dairy-cow-dataset`)

* **Repository**: `https://github.com/wusaisa/Dairy-cow-dataset`
* **Author**: Saisai Wu et al. (Northwest A&F University — NWAFU).
* **README Claim**: *"Three experimental data are included: object detection, pose estimation, lameness detection"*.

### Detailed Findings
1. **Google Drive Link Audit**:
   * Current README link (`1konCZYCqC7Od09nQC3jI4ri7rUAIqXTB`) returns **HTTP 401 Unauthorized** (private permission wall).
   * Earlier commit `7e366599` contained `1bLQFHd9rqllmEaYvbcAqEBJtlyfuKPdP`, which remains **100% publicly accessible**.
2. **Archive Content Analysis (`NWAFU_CattleDataset.zip`, 1.73 GB)**:
   * Inspected archive central directory via HTTP range requests without unzipping.
   * `NWAFU_CattleDataset/images/`: **1,941 JPG images**
   * `NWAFU_CattleDataset/annotations/`: **2,083 TXT annotation files**
3. **The Fatal Finding**:
   * **ZERO LAMENESS DATA**: Despite the README claim, there are no gait scores, no mobility labels, no sound/lame tags, and no temporal walking video sequences.
   * The archive is exclusively a **static 2D pose estimation dataset** (63 cattle, annotated with 16 body keypoints in YOLO format for single-frame keypoint localization).
4. **Usability Verdict**: **COMPLETELY UNUSABLE for lameness detection** due to total lack of lameness ground-truth labels.

---

## 3. Candidate 3: Lameness-Detection Skeleton Dataset (`whsu2s/Lameness-Detection`)

* **Repository**: `https://github.com/whsu2s/Lameness-Detection`
* **Author**: Wei-Chan Hsu (Master's Thesis).
* **README Claim**: 500 trimmed video clips (680x420, 20 fps) with DeepLabCut 25-keypoint JSON skeleton sequences.

### Detailed Findings
1. **Repository Audit**:
   * Cloned repository size is only **48.2 MB**.
   * Inspected complete directory tree: **Zero video clips, Zero JSON skeleton sequence files**.
2. **Source of the Missing Data**:
   * Inspected `src/hrnn.ipynb` and `src/data_split_st-gcn.ipynb`:
     ```python
     datadir = '/home/wei-chan.hsu/Dokumente/Thesis/src/annotation/data/all2/data_json/'
     ```
   * The actual video clips and skeleton JSON files were kept on the author's local workstation and were never committed to GitHub or uploaded to any cloud storage.
3. **What is in the Repo**:
   * `utils/data_labels.csv`: Mobility score annotations for 64 cows across multiple days (`tag0` to `tag14`).
   * Model implementations for HRNN and ST-GCN.
   * Thesis proposal and protocol documentation PDFs.
4. **Usability Verdict**: **COMPLETELY UNUSABLE** without directly contacting the author to obtain the missing 500 clips/skeletons.

---

## 4. Candidate 4: Duan et al. 2025 (Overhead RGB-D Lameness Dataset)

* **Paper Reference**: Duan, Weijun, Fang Wang, Honghui Li, Na Liu, and Xueliang Fu. *"Lameness detection in dairy cows from overhead view: high-precision keypoint localization and multi-feature fusion classification."* *Frontiers in Veterinary Science* (Published September 9, 2025). ([DOI: 10.3389/fvets.2025.1472535](https://doi.org/10.3389/fvets.2025.1472535))
* **Claimed Dataset**: 741 RGB-D walking sequences, 2,520 keypoint images (8 dorsal keypoints: poll, withers, left/right scapula, lumbar, left/right tuber coxae, sacral tuber).

### Detailed Findings
1. **Data Availability Statement**:
   > *"The original contributions presented in the study are included in the article/supplementary material, further inquiries can be directed to the corresponding author."*
2. **Public Availability**:
   * No public download links (Zenodo, Dryad, Figshare, GitHub, or Kaggle) are provided.
   * The data is closed and held privately at Inner Mongolia Agricultural University.
3. **Usability Verdict**: **UNAVAILABLE for current thesis execution**.

---

## Architectural Comparison & Final Recommendations

### 1. PRIMARY Lameness Dataset: **Russello 2026 (`hrussel/lstm-lameness-detection`)**
* **Why**: The only audited candidate that is verified, complete, public, and scientifically sound.
* **Key Advantages**:
  * 272 trajectories across 98 distinct cows.
  * Balanced classes (143 sound, 129 lame).
  * True cow IDs enable leak-proof 5-fold StratifiedGroupKFold evaluation.
  * Controlled farm walkway environment eliminating scene and lighting confounds.
* **Integration Strategy**:
  * Utilize a temporal sequence head (BiLSTM or 1D-CNN) or graph convolutional network (ST-GCN).
  * Alternatively, render 2D skeleton stick-figures / coordinate heatmaps to feed into a shared visual backbone.

### 2. BACKUP Lameness Dataset: **CattleLameness (Governed by `cattle_lameness_manifest.csv`)**
* **Why**: If raw RGB pixels are strictly required for a pure image-based multi-task backbone (e.g. ResNet-18), our newly created [cattle_lameness_manifest.csv](file:///d:/cattle-health-monitoring-multi-task-model/datasets/lameness/cattle_lameness_manifest.csv) enforces a 42-group leak-proof 5-fold CV split that prevents test contamination.

### 3. Datasets Rejected from Thesis Usage:
* **Wu (`wusaisa/Dairy-cow-dataset`)**: Rejected. Contains static pose estimation only; zero lameness data.
* **whsu2s (`whsu2s/Lameness-Detection`)**: Rejected. 500 clips and skeleton files were never published to GitHub.
* **Duan et al. 2025**: Rejected. Private data requiring author email authorization.
