# Forensic Audit Report: CattleLameness Dataset

**Dataset**: CattleLameness (Derived from YouTube Videos)  
**Authors**: Fahim Sohan et al.  
**Repository**: `https://github.com/fahimsohan/CattleLameness`  
**Paper Reference**: *"Direct Video-Based Spatiotemporal Deep Learning for Cattle Lameness Detection"* ([PMC12764510](https://pmc.ncbi.nlm.nih.gov/articles/PMC12764510/))  
**Audit Date**: September 17, 2026  
**Status**: Read-Only Inspection Complete. No source code or dataset files modified.

---

## 1. Executive Summary & Core Metrics

* **Total Videos**: **50 video clips**
  * `Data/Lame/`: 25 videos (`L (1).mp4` to `L (25).mp4`)
  * `Data/Normal/`: 25 videos (`N (1).mp4` to `N (25).mp4`)
* **Total Extracted Frames**: **9,950 frames**
  * Lame frames: 5,900 (59.30%)
  * Normal frames: 4,050 (40.70%)
* **Total Video Runtime**: **282.15 seconds** (~4.70 minutes across the entire benchmark)
* **Raw Video Resolution**: **500x500 square** (100% of the 50 clips)
* **Framerates (FPS)**: Variable framerates between 25.00 FPS and 60.00 FPS
* **Ground-Truth Cattle IDs**: **None provided** in filenames, metadata, or repository.
* **Documented Animal Count**: **42 individual cattle** across 50 video clips (per official README).
* **Proven Data Leakage**: **Confirmed.** Due to multi-clip representation of the same cattle/source, the project's naive video-level splitting assigns segments of the exact same animal/source to both the **Train** and **Test** sets.

---

## 2. Dataset Distribution & Summary Metrics

| Metric | Lame (`L`) | Normal (`N`) | Combined / Benchmark Total |
| :--- | :--- | :--- | :--- |
| **Video Count** | 25 clips (50.0%) | 25 clips (50.0%) | 50 clips (100.0%) |
| **Total Frames** | 5,900 frames (59.3%) | 4,050 frames (40.7%) | 9,950 frames |
| **Total Duration** | 148.38 seconds | 133.77 seconds | 282.15 seconds (4.70 min) |
| **Average Duration** | 5.94 seconds | 5.35 seconds | 5.64 seconds |
| **Min / Max Duration** | 3.61s / 10.13s | 1.35s / 12.28s | 1.35s / 12.28s |
| **Average Frame Count** | 236.0 frames | 162.0 frames | 199.0 frames |
| **Min / Max Frames** | 108 / 479 frames | 81 / 368 frames | 81 / 479 frames |
| **Resolution** | 500x500 (100%) | 500x500 (100%) | 500x500 (100%) |
| **FPS Distribution** | 29.5 to 60.0 FPS | 25.0 to 60.0 FPS | 25.0 to 60.0 FPS |

---

## 3. Comprehensive Video Inventory (All 50 Files)

### 3.1. Lame Videos (`Data/Lame/`)

| Filename | Resolution | FPS | Total Frames | Duration (s) | File Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `L (1).mp4` | 500x500 | 29.97 | 120 | 4.00s | 0.40 MB |
| `L (2).mp4` | 500x500 | 60.00 | 390 | 6.50s | 1.09 MB |
| `L (3).mp4` | 500x500 | 30.00 | 138 | 4.60s | 0.90 MB |
| `L (4).mp4` | 500x500 | 29.50 | 167 | 5.66s | 0.84 MB |
| `L (5).mp4` | 500x500 | 60.00 | 366 | 6.10s | 1.41 MB |
| `L (6).mp4` | 500x500 | 30.00 | 213 | 7.10s | 0.68 MB |
| `L (7).mp4` | 500x500 | 60.00 | 341 | 5.68s | 1.63 MB |
| `L (8).mp4` | 500x500 | 29.97 | 180 | 6.01s | 0.65 MB |
| `L (9).mp4` | 500x500 | 30.00 | 228 | 7.60s | 1.05 MB |
| `L (10).mp4` | 500x500 | 29.97 | 125 | 4.17s | 0.65 MB |
| `L (11).mp4` | 500x500 | 29.92 | 303 | 10.13s | 1.62 MB |
| `L (12).mp4` | 500x500 | 60.00 | 479 | 7.98s | 1.46 MB |
| `L (13).mp4` | 500x500 | 60.00 | 308 | 5.13s | 0.83 MB |
| `L (14).mp4` | 500x500 | 29.92 | 223 | 7.45s | 1.47 MB |
| `L (15).mp4` | 500x500 | 29.97 | 134 | 4.47s | 0.63 MB |
| `L (16).mp4` | 500x500 | 59.94 | 395 | 6.59s | 1.90 MB |
| `L (17).mp4` | 500x500 | 30.00 | 125 | 4.17s | 1.02 MB |
| `L (18).mp4` | 500x500 | 29.97 | 150 | 5.00s | 0.48 MB |
| `L (19).mp4` | 500x500 | 29.92 | 108 | 3.61s | 0.63 MB |
| `L (20).mp4` | 500x500 | 30.00 | 177 | 5.90s | 0.65 MB |
| `L (21).mp4` | 500x500 | 60.00 | 302 | 5.03s | 1.23 MB |
| `L (22).mp4` | 500x500 | 30.00 | 260 | 8.67s | 2.78 MB |
| `L (23).mp4` | 500x500 | 30.00 | 141 | 4.70s | 0.50 MB |
| `L (24).mp4` | 500x500 | 29.50 | 193 | 6.54s | 1.32 MB |
| `L (25).mp4` | 500x500 | 59.94 | 334 | 5.57s | 1.34 MB |

### 3.2. Normal Videos (`Data/Normal/`)

| Filename | Resolution | FPS | Total Frames | Duration (s) | File Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `N (1).mp4` | 500x500 | 60.00 | 146 | 2.43s | 0.93 MB |
| `N (2).mp4` | 500x500 | 29.58 | 121 | 4.09s | 0.49 MB |
| `N (3).mp4` | 500x500 | 30.00 | 156 | 5.20s | 0.74 MB |
| `N (4).mp4` | 500x500 | 29.00 | 116 | 4.00s | 0.54 MB |
| `N (5).mp4` | 500x500 | 28.70 | 204 | 7.11s | 0.71 MB |
| `N (6).mp4` | 500x500 | 25.00 | 205 | 8.20s | 0.37 MB |
| `N (7).mp4` | 500x500 | 30.00 | 155 | 5.17s | 0.22 MB |
| `N (8).mp4` | 500x500 | 25.00 | 138 | 5.52s | 0.16 MB |
| `N (9).mp4` | 500x500 | 30.00 | 161 | 5.37s | 0.11 MB |
| `N (10).mp4` | 500x500 | 30.00 | 191 | 6.37s | 0.92 MB |
| `N (11).mp4` | 500x500 | 25.00 | 127 | 5.08s | 0.68 MB |
| `N (12).mp4` | 500x500 | 29.17 | 148 | 5.07s | 1.00 MB |
| `N (13).mp4` | 500x500 | 29.58 | 132 | 4.46s | 0.64 MB |
| `N (14).mp4` | 500x500 | 30.00 | 114 | 3.80s | 0.48 MB |
| `N (15).mp4` | 500x500 | 29.92 | 157 | 5.25s | 1.18 MB |
| `N (16).mp4` | 500x500 | 29.97 | 205 | 6.84s | 1.20 MB |
| `N (17).mp4` | 500x500 | 30.00 | 152 | 5.07s | 1.14 MB |
| `N (18).mp4` | 500x500 | 60.00 | 111 | 1.85s | 0.37 MB |
| `N (19).mp4` | 500x500 | 30.00 | 141 | 4.70s | 1.23 MB |
| `N (20).mp4` | 500x500 | 29.97 | 247 | 8.24s | 0.25 MB |
| `N (21).mp4` | 500x500 | 30.00 | 149 | 4.97s | 0.16 MB |
| `N (22).mp4` | 500x500 | 30.00 | 246 | 8.20s | 0.78 MB |
| `N (23).mp4` | 500x500 | 60.00 | 81 | 1.35s | 0.24 MB |
| `N (24).mp4` | 500x500 | 29.97 | 368 | 12.28s | 0.38 MB |
| `N (25).mp4` | 500x500 | 25.00 | 79 | 3.16s | 0.47 MB |

---

## 4. Metadata, Container Header & Provenance Audit

### 4.1. Embedded MP4 Container Tags
Parsing container atoms (`ftyp`, `moov`, `trak`, `udta`) revealed:
* **Toolchain**: `Lavf60.16.100` (FFmpeg libavformat).
* **Embedded Processing Comment**:
  ```text
  MP4 resized with https://ezgif.com/resize-video
  ```
  Found across all 50 video files. The original source clips were uploaded to ezgif.com to crop and downscale to 500x500 square video.

### 4.2. Source URL Records (`Data/video_sources.txt`)
* Lists **only 20 YouTube links** for 50 total clips (10 Lame, 10 Normal).
* The authors explicitly note: *"This is a partial list; additional video sources may be added later as they are recovered."*
* **Direct Duplicate Link Found**:
  * Normal Link #3: `https://youtube.com/shorts/ntAGV4SQ0Fw?si=4IiZFH8c1jlbhxKW`
  * Normal Link #9: `https://youtube.com/shorts/ntAGV4SQ0Fw?si=1KqB9frbi08wIrYC`
  These two links represent the **exact same video URL**. Two distinct clips (`N (3)` and `N (9)`) were derived from this single YouTube short.

---

## 5. Animal Identity & Multi-Clip Presence Analysis

### 5.1. The 42 Cattle vs 50 Clips Discrepancy
In the primary documentation (`README.md`), the authors state:
> *"We curate and publicly release a balanced set of 50 online video clips featuring 42 individual cattle, recorded from multiple viewpoints in both indoor and outdoor environments."*

This mathematically confirms:
* **Unique animals = 42**
* **Total clips = 50**
* **Multi-clip appearances = at least 8 clips** (minimum of 8 cattle appear in 2 or more separate video clips).

### 5.2. Visual Similarity & Color Histogram Correlation
Pairwise 3D color histogram comparisons between video mid-frames reveal high correlations between separate video files, indicating shared animals, identical pens, and identical camera setups:

| Pair | Correlation | Notes |
| :--- | :---: | :--- |
| `N (20).mp4` <--> `N (24).mp4` | **0.985** | Identical environment, identical cow, distinct time segments |
| `N (18).mp4` <--> `N (23).mp4` | **0.960** | Identical environment and cow |
| `L (8).mp4` <--> `L (18).mp4` | **0.973** | Identical background pen, identical cow markings |
| `L (14).mp4` <--> `L (19).mp4` | **0.941** | Identical scene and cow |
| `N (5).mp4` <--> `N (11).mp4` | **0.891** | Shared background setup |
| `N (4).mp4` <--> `N (23).mp4` | **0.876** | Shared background setup |

---

## 6. Project Splitting Implementation & Complete Split Mapping

### 6.1. Current Splitting Implementation (`context/preprocess_lameness.py`)

```python
# Configuration
TARGET_SIZE = (224, 224)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def process_video_folder(directory, label, class_prefix):
    video_files = sorted([f for f in os.listdir(directory) if f.endswith('.mp4')])
    
    # Shuffle videos to perform video-wise (cow-wise) split
    random.shuffle(video_files)
    
    num_videos = len(video_files)
    train_end = int(num_videos * 0.70)
    val_end = train_end + int(num_videos * 0.15)
    
    video_data = []
    
    for idx, video_file in enumerate(video_files):
        video_path = directory / video_file
        video_id = f"{class_prefix}_{idx + 1}"
        
        # Determine split
        if idx < train_end:
            split = 'train'
        elif idx < val_end:
            split = 'val'
        else:
            split = 'test'
            
        # Frame extraction follows...
```

### 6.2. Exact Assignment of Videos under `random.seed(42)`

| Class Prefix | Index ID | Source Video File | Split Assignment | Frame Count |
| :--- | :--- | :--- | :---: | :---: |
| **Lame** | `Lame_1` | `L (24).mp4` | **train** | 193 |
| **Lame** | `Lame_2` | `L (20).mp4` | **train** | 177 |
| **Lame** | `Lame_3` | `L (18).mp4` | **train** | 150 |
| **Lame** | `Lame_4` | `L (4).mp4` | **train** | 167 |
| **Lame** | `Lame_5` | `L (3).mp4` | **train** | 138 |
| **Lame** | `Lame_6` | `L (15).mp4` | **train** | 134 |
| **Lame** | `Lame_7` | `L (14).mp4` | **train** | 223 |
| **Lame** | `Lame_8` | `L (19).mp4` | **train** | 108 |
| **Lame** | `Lame_9` | `L (23).mp4` | **train** | 141 |
| **Lame** | `Lame_10` | `L (6).mp4` | **train** | 213 |
| **Lame** | `Lame_11` | `L (2).mp4` | **train** | 390 |
| **Lame** | `Lame_12` | `L (25).mp4` | **train** | 334 |
| **Lame** | `Lame_13` | `L (10).mp4` | **train** | 125 |
| **Lame** | `Lame_14` | `L (22).mp4` | **train** | 260 |
| **Lame** | `Lame_15` | `L (7).mp4` | **train** | 341 |
| **Lame** | `Lame_16` | `L (21).mp4` | **train** | 302 |
| **Lame** | `Lame_17` | `L (11).mp4` | **train** | 303 |
| **Lame** | `Lame_18` | `L (8).mp4` | **val** | 180 |
| **Lame** | `Lame_19` | `L (13).mp4` | **val** | 308 |
| **Lame** | `Lame_20` | `L (9).mp4` | **val** | 228 |
| **Lame** | `Lame_21` | `L (16).mp4` | **test** | 395 |
| **Lame** | `Lame_22` | `L (17).mp4` | **test** | 125 |
| **Lame** | `Lame_23` | `L (1).mp4` | **test** | 120 |
| **Lame** | `Lame_24` | `L (12).mp4` | **test** | 479 |
| **Lame** | `Lame_25` | `L (5).mp4` | **test** | 366 |
| **Normal** | `Normal_1` | `N (11).mp4` | **train** | 127 |
| **Normal** | `Normal_2` | `N (20).mp4` | **train** | 247 |
| **Normal** | `Normal_3` | `N (2).mp4` | **train** | 121 |
| **Normal** | `Normal_4` | `N (16).mp4` | **train** | 205 |
| **Normal** | `Normal_5` | `N (9).mp4` | **train** | 161 |
| **Normal** | `Normal_6` | `N (12).mp4` | **train** | 148 |
| **Normal** | `Normal_7` | `N (18).mp4` | **train** | 111 |
| **Normal** | `Normal_8` | `N (7).mp4` | **train** | 155 |
| **Normal** | `Normal_9` | `N (25).mp4` | **train** | 79 |
| **Normal** | `Normal_10` | `N (23).mp4` | **train** | 81 |
| **Normal** | `Normal_11` | `N (6).mp4` | **train** | 205 |
| **Normal** | `Normal_12` | `N (5).mp4` | **train** | 204 |
| **Normal** | `Normal_13` | `N (24).mp4` | **train** | 368 |
| **Normal** | `Normal_14` | `N (22).mp4` | **train** | 246 |
| **Normal** | `Normal_15` | `N (10).mp4` | **train** | 191 |
| **Normal** | `Normal_16` | `N (4).mp4` | **train** | 116 |
| **Normal** | `Normal_17` | `N (15).mp4` | **train** | 157 |
| **Normal** | `Normal_18` | `N (13).mp4` | **val** | 132 |
| **Normal** | `Normal_19` | `N (8).mp4` | **val** | 138 |
| **Normal** | `Normal_20` | `N (19).mp4` | **val** | 141 |
| **Normal** | `Normal_21` | `N (21).mp4` | **test** | 149 |
| **Normal** | `Normal_22` | `N (14).mp4` | **test** | 114 |
| **Normal** | `Normal_23` | `N (1).mp4` | **test** | 146 |
| **Normal** | `Normal_24` | `N (17).mp4` | **test** | 152 |
| **Normal** | `Normal_25` | `N (3).mp4` | **test** | 156 |

### 6.3. Aggregate Frame Counts by Split

| Split | Lame Frames | Normal Frames | Combined Frames |
| :--- | :---: | :---: | :---: |
| **Train (70%)** | 3,894 | 2,914 | **6,808 frames** (68.4%) |
| **Val (15%)** | 716 | 511 | **1,227 frames** (12.3%) |
| **Test (15%)** | 1,490 | 717 | **1,915 frames** (19.2%) |
| **Total** | **5,900** | **4,050** | **9,950 frames** (100.0%) |

---

## 7. Forensic Identification of Data Leakage

### 7.1. Proven Animal-Level & Source-Level Cross-Split Leakage
* **Flawed Core Assumption**: `preprocess_lameness.py` synthesizes artificial animal IDs via `f"{class_prefix}_{idx + 1}"`, assuming that each video represents an independent, distinct animal.
* **The Smoking Gun**:
  * As documented in `video_sources.txt`, Normal entry #3 and entry #9 are identical: `https://youtube.com/shorts/ntAGV4SQ0Fw`.
  * Under `random.seed(42)`:
    * `N (9).mp4` is assigned to `Normal_5` in the **TRAIN** split.
    * `N (3).mp4` is assigned to `Normal_25` in the **TEST** split.
  * **Result**: Frames of the exact same cow and identical video recording exist simultaneously in both training and test sets. The test evaluation measures clip memorization rather than generalized out-of-sample gait assessment.

### 7.2. High-Correlation Cross-Split Leakage (Train vs Val)
* `L (18).mp4` is mapped to `Lame_3` in **TRAIN**.
* `L (8).mp4` is mapped to `Lame_18` in **VAL**.
* These two videos possess a pairwise color histogram correlation of **0.973** and visually portray the same animal in the exact same pen. Validation loss and early stopping based on this validation set are artificially optimistic due to feature leakage.

### 7.3. Unfiltered Temporal Redundancy
* Frame extraction processes every single frame at 30–60 FPS.
* Consecutive frames within an individual clip have near-zero semantic variance. When combined with source leakage between splits (such as `N (9)` in Train and `N (3)` in Test), near-duplicate frames are evaluated in test inference.

---

## 8. Conclusion & Recommendations

1. **Splitting Integrity**: The current splitting method in `preprocess_lameness.py` cannot be considered an animal-independent or identity-disjoint partition.
2. **Immediate Corrective Action Needed**:
   * Explicitly cluster the 50 video clips into their true 42 animal identities (merging known duplicate sources like `N (3)` and `N (9)`, and visually verified pairs).
   * Perform Group-K-Fold or animal-level disjoint splitting where all clips of a given animal/source remain strictly within one partition.
   * Apply frame decimation or sampling (e.g. uniform sampling of 20–30 frames per video clip) to eliminate extreme temporal redundancy.
