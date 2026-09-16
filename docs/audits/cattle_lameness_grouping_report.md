# CattleLameness Leakage-Safe Grouping & Evaluation Report

**Dataset**: CattleLameness (Derived from YouTube Videos)  
**Authors**: Fahim Sohan et al.  
**Repository**: `https://github.com/fahimsohan/CattleLameness`  
**Manifest File**: [cattle_lameness_manifest.csv](file:///d:/cattle-health-monitoring-multi-task-model/cattle_lameness_manifest.csv)  
**Report Date**: September 17, 2026  
**Status**: Grouping & Cross-Validation Architecture Complete. No model trained.

---

## 1. Executive Summary

A forensic audit of the 50 CattleLameness clips revealed that the project's original train/val/test split suffered from direct animal-level and source-level data leakage between training and testing splits (specifically `N (9)` in Train and `N (3)` in Test originating from the exact same YouTube short).

To eliminate this leakage without modifying the underlying raw dataset files, a **leakage-safe grouping manifest** was constructed. All 50 clips were audited, cross-referenced with source metadata, evaluated using perceptual hashing, color histogram correlation, temporal boundary matching, and coat/environment visual inspection.

### Key Milestones
1. **True Group Resolution**: Identified **42 unique animal/source groups** (21 Lame, 21 Normal), exactly corroborating the authors' documented count of 42 individual cattle across 50 clips.
2. **Multi-Clip Clustered Groups**: 8 distinct groups contain multiple video clips (4 Lame groups, 4 Normal groups).
3. **Evaluation Protocol**: Verified that **5-fold StratifiedGroupKFold** is **100% mathematically feasible and perfectly class-balanced**.
4. **Zero Cross-Split Leakage**: In the proposed 5-fold architecture, every single validation fold contains exactly **5 Lame clips and 5 Normal clips (10 clips per fold)**, with 0 group overlap between training and validation.

---

## 2. Manifest Schema (`cattle_lameness_manifest.csv`)

The generated manifest contains 10 structured fields for every clip:

| Field | Type | Description |
| :--- | :---: | :--- |
| `filename` | string | Exact video filename in dataset (`L (1).mp4` to `L (25).mp4`, `N (1).mp4` to `N (25).mp4`) |
| `class` | string | Target binary classification label: `lame` or `normal` |
| `source_url` | string | Original YouTube source URL if confirmed/available from `video_sources.txt` |
| `proposed_group_id` | string | Unique animal/source group identifier (`GRP_LAME_01` to `GRP_LAME_24`, `GRP_NORM_01` to `GRP_NORM_25`) |
| `confidence` | string | Degree of grouping certainty: `confirmed`, `high`, `medium`, or `low` |
| `evidence` | string | Detailed forensic justification (URL match, histogram correlation, boundary continuity, coat markings) |
| `proposed_fold` | integer | Stratified group fold assignment (1 to 5) |
| `duration` | float | Video duration in seconds (1.35s to 12.28s) |
| `fps` | float | Encoded framerate in frames per second (25.0 to 60.0 FPS) |
| `resolution` | string | Video frame dimensions (`500x500` for all 50 clips) |

---

## 3. Grouping Analysis & Multi-Clip Clusters

Out of 50 video clips, **16 clips belong to 8 multi-clip groups**, while the remaining **34 clips represent independent single-clip animals/sources**.

### 3.1. Confirmed Multi-Clip Groups (Strongest Ground-Truth & Forensic Evidence)

#### 1. `GRP_NORM_03` — [N (3).mp4, N (9).mp4]
* **Confidence**: `confirmed`
* **Evidence**: Direct URL duplication in `video_sources.txt`. Item #3 (`https://youtube.com/shorts/ntAGV4SQ0Fw?si=4IiZFH8c1jlbhxKW`) and Item #9 (`https://youtube.com/shorts/ntAGV4SQ0Fw?si=1KqB9frbi08wIrYC`) are identical YouTube short uploads by channel *"Cow Life"*. The two clips represent segments of the exact same video recording.
* **Leakage Impact**: Previously, `N (9)` was assigned to Train (`Normal_5`) and `N (3)` to Test (`Normal_25`). In this grouping, both are permanently locked into **Fold 3**.

#### 2. `GRP_LAME_08` — [L (8).mp4, L (18).mp4]
* **Confidence**: `confirmed`
* **Evidence**: Temporal boundary continuity and identical cow coat markings. The last frame of `L (8).mp4` and first frame of `L (18).mp4` exhibit a mean pixel delta of only **14.2** (continuous video segment). Pairwise color histogram correlation is **0.912** with identical outdoor pasture background and rear-right leg lameness. Both are locked into **Fold 1**.

---

### 3.2. High-Confidence Multi-Clip Groups (Extensive Visual & Perceptual Match)

#### 3. `GRP_NORM_20` — [N (20).mp4, N (24).mp4]
* **Confidence**: `high`
* **Evidence**: 3D color histogram correlation of **0.984**; mean BGR distance of **9.8**; identical Holstein black-and-white patch distribution along the left flank; identical dirt roadway background. Both are locked into **Fold 1**.

#### 4. `GRP_NORM_18` — [N (18).mp4, N (23).mp4]
* **Confidence**: `high`
* **Evidence**: Both recorded at **60.0 FPS**; mean BGR distance of **11.5**; boundary frame difference of **30.0**; pairwise color correlation of **0.826**; identical brown Jersey heifer walking through an identical concrete holding lane. Both are locked into **Fold 2**.

#### 5. `GRP_LAME_13` — [L (13).mp4, L (25).mp4]
* **Confidence**: `high`
* **Evidence**: Both recorded at high framerate (**60.0 FPS** and **59.94 FPS**); pairwise color histogram correlation of **0.937**; identical Holstein markings and concrete parlor exit chute. Both are locked into **Fold 2**.

---

### 3.3. Medium-Confidence Multi-Clip Groups

#### 6. `GRP_LAME_06` — [L (6).mp4, L (19).mp4]
* **Confidence**: `medium`
* **Evidence**: Solid dark black cow in low-light concrete barn; boundary frame difference of **34.9**; perceptual hash distance of **24**; pairwise color histogram correlation of **0.756**. Both are locked into **Fold 4**.

#### 7. `GRP_NORM_13` — [N (13).mp4, N (22).mp4]
* **Confidence**: `medium`
* **Evidence**: Pairwise color histogram correlation of **0.925**; mean BGR distance of **21.8**; identical indoor dairy parlor lighting and highly consistent Holstein patch geometry. Both are locked into **Fold 5**.

---

### 3.4. Low-Confidence / Uncertain Multi-Clip Groups

#### 8. `GRP_LAME_16` — [L (16).mp4, L (17).mp4]
* **Confidence**: `low`
* **Evidence**: Pairwise color histogram correlation of **0.892**; mean BGR distance of **15.9**; similar outdoor yard background. However, framerates differ (59.94 FPS vs 30.00 FPS) and walking orientation is flipped. Grouped together as a conservative anti-leakage safeguard, but marked with `low` confidence. Both are locked into **Fold 3**.

---

## 4. Feasibility of 5-Fold StratifiedGroupKFold

### Feasibility Verdict: **100% FEASIBLE & PERFECTLY BALANCED**

Because the dataset condenses into **21 Lame groups** and **21 Normal groups** (42 groups total):
1. **Clip Balance**: Every fold contains **exactly 10 clips (5 Lame clips and 5 Normal clips)**. Class balance across every single validation fold is exactly 50.0% / 50.0%.
2. **Group Integrity**: No group is partitioned across folds. All clips from any animal or source are evaluated together in one fold and trained on in the other four folds.
3. **Zero Contamination**: Train size per fold is 40 clips (20 Lame, 20 Normal); Validation size per fold is 10 clips (5 Lame, 5 Normal).

### Fold Distribution Summary Table

| Fold ID | Lame Clips | Normal Clips | Total Clips | Lame Groups | Normal Groups | Total Groups | Total Frames | Total Duration (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 5 | 5 | **10** | 4 | 4 | **8** | 1,939 | 57.77s |
| **Fold 2** | 5 | 5 | **10** | 4 | 4 | **8** | 1,515 | 40.66s |
| **Fold 3** | 5 | 5 | **10** | 4 | 4 | **8** | 2,130 | 58.74s |
| **Fold 4** | 5 | 5 | **10** | 4 | 5 | **9** | 2,137 | 64.91s |
| **Fold 5** | 5 | 5 | **10** | 5 | 4 | **9** | 2,229 | 60.07s |
| **Total** | **25** | **25** | **50** | **21** | **21** | **42** | **9,950** | **282.15s** |

---

## 5. Complete Fold Breakdown

### Fold 1 (Validation Partition 1)
* **Lame Clips (5)**:
  * `L (2).mp4` (`GRP_LAME_02`, confirmed source `CXi6rtvqvEU`)
  * `L (5).mp4` (`GRP_LAME_05`, confirmed source `EuQ06yYNh70`)
  * `L (8).mp4` (`GRP_LAME_08`, confirmed source `fo8bdQteCGM`, paired with L18)
  * `L (18).mp4` (`GRP_LAME_08`, paired with L8)
  * `L (22).mp4` (`GRP_LAME_22`, independent high-confidence animal)
* **Normal Clips (5)**:
  * `N (7).mp4` (`GRP_NORM_07`, confirmed source `4pm4UMxkm90`)
  * `N (16).mp4` (`GRP_NORM_16`, independent high-confidence animal)
  * `N (20).mp4` (`GRP_NORM_20`, paired with N24)
  * `N (24).mp4` (`GRP_NORM_20`, paired with N20)
  * `N (25).mp4` (`GRP_NORM_25`, independent high-confidence animal)

### Fold 2 (Validation Partition 2)
* **Lame Clips (5)**:
  * `L (1).mp4` (`GRP_LAME_01`, confirmed source `NMfoeqsbSa0`)
  * `L (4).mp4` (`GRP_LAME_04`, confirmed source `7TSo4g9yeS8`)
  * `L (13).mp4` (`GRP_LAME_13`, paired with L25)
  * `L (23).mp4` (`GRP_LAME_23`, independent high-confidence animal)
  * `L (25).mp4` (`GRP_LAME_13`, paired with L13)
* **Normal Clips (5)**:
  * `N (8).mp4` (`GRP_NORM_08`, confirmed source `MK09R0CK3sE`)
  * `N (14).mp4` (`GRP_NORM_14`, independent high-confidence animal)
  * `N (18).mp4` (`GRP_NORM_18`, paired with N23)
  * `N (21).mp4` (`GRP_NORM_21`, independent high-confidence animal)
  * `N (23).mp4` (`GRP_NORM_18`, paired with N18)

### Fold 3 (Validation Partition 3)
* **Lame Clips (5)**:
  * `L (12).mp4` (`GRP_LAME_12`, independent high-confidence animal)
  * `L (15).mp4` (`GRP_LAME_15`, independent high-confidence animal)
  * `L (16).mp4` (`GRP_LAME_16`, low-confidence pair with L17)
  * `L (17).mp4` (`GRP_LAME_16`, low-confidence pair with L16)
  * `L (21).mp4` (`GRP_LAME_21`, independent high-confidence animal)
* **Normal Clips (5)**:
  * `N (2).mp4` (`GRP_NORM_02`, confirmed source `icEm_s9Ku_Q`)
  * `N (3).mp4` (`GRP_NORM_03`, confirmed source `ntAGV4SQ0Fw`, paired with N9)
  * `N (9).mp4` (`GRP_NORM_03`, confirmed source `ntAGV4SQ0Fw`, paired with N3)
  * `N (10).mp4` (`GRP_NORM_10`, confirmed source `izzguC1uoWI`)
  * `N (12).mp4` (`GRP_NORM_12`, independent high-confidence animal)

### Fold 4 (Validation Partition 4)
* **Lame Clips (5)**:
  * `L (3).mp4` (`GRP_LAME_03`, confirmed source `-pFmPSLWQO4`)
  * `L (6).mp4` (`GRP_LAME_06`, confirmed source `pLa-thOWK5E`, paired with L19)
  * `L (10).mp4` (`GRP_LAME_10`, confirmed source `c2xvLyejH2A`)
  * `L (19).mp4` (`GRP_LAME_06`, paired with L6)
  * `L (24).mp4` (`GRP_LAME_24`, independent high-confidence animal)
* **Normal Clips (5)**:
  * `N (4).mp4` (`GRP_NORM_04`, confirmed source `T408rP1Nlck`)
  * `N (5).mp4` (`GRP_NORM_05`, confirmed source `4PF_-QHZ5rM` [Blender 3D CGI])
  * `N (6).mp4` (`GRP_NORM_06`, confirmed source `sVViydUNkdE` [Green Screen])
  * `N (11).mp4` (`GRP_NORM_11`, independent high-confidence animal)
  * `N (15).mp4` (`GRP_NORM_15`, independent high-confidence animal)

### Fold 5 (Validation Partition 5)
* **Lame Clips (5)**:
  * `L (7).mp4` (`GRP_LAME_07`, confirmed source `RXkYdPIVyrg`)
  * `L (9).mp4` (`GRP_LAME_09`, confirmed source `Qdl5ZvGOBp4`)
  * `L (11).mp4` (`GRP_LAME_11`, independent high-confidence animal)
  * `L (14).mp4` (`GRP_LAME_14`, independent high-confidence animal)
  * `L (20).mp4` (`GRP_LAME_20`, independent high-confidence animal)
* **Normal Clips (5)**:
  * `N (1).mp4` (`GRP_NORM_01`, confirmed source `ntSw904m8rY`)
  * `N (13).mp4` (`GRP_NORM_13`, paired with N22)
  * `N (17).mp4` (`GRP_NORM_17`, independent high-confidence animal)
  * `N (19).mp4` (`GRP_NORM_19`, independent high-confidence animal)
  * `N (22).mp4` (`GRP_NORM_13`, paired with N13)

---

## 6. Implementation Roadmap for Multi-Task Pipeline

When training models on CattleLameness:
1. **Drop Random Shuffling**: Deprecate `random.shuffle(video_files)` in `preprocess_lameness.py`.
2. **Index via Manifest**: Point the dataset loader directly to [cattle_lameness_manifest.csv](file:///d:/cattle-health-monitoring-multi-task-model/cattle_lameness_manifest.csv).
3. **Cross-Validation Training**: Train 5 models using the `proposed_fold` column:
   * **Fold 1**: Validate on Fold 1, Train on Folds [2, 3, 4, 5]
   * **Fold 2**: Validate on Fold 2, Train on Folds [1, 3, 4, 5]
   * **Fold 3**: Validate on Fold 3, Train on Folds [1, 2, 4, 5]
   * **Fold 4**: Validate on Fold 4, Train on Folds [1, 2, 3, 5]
   * **Fold 5**: Validate on Fold 5, Train on Folds [1, 2, 3, 4]
4. **Report Mean & Std Metrics**: Compute out-of-fold cross-validated Macro-F1, Accuracy, and AUC across all 5 folds to ensure robust, leak-free performance reporting for Thesis Phase 3.
