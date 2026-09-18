# Research Log: Primary Task Dataset Split Integrity & Leakage Audit

**Date**: 2026-09-18  
**Author**: Hasin Ishrak  
**Supervision**: Dr. Md. Khalilur Rahman  
**Project**: Cattle Health Monitoring Multi-Task Deep Learning Model (BRAC University)

---

## 1. Executive Summary
Audited the dataset split protocols across the three active primary Phase 3 tasks: Body Condition Scoring (ScienceDB, 53,566 images), Behavior Recognition (MmCows, 213,686 crops), and Cow Identification (OpenCows2020, 4,736 images). While all datasets are downloaded and indexed, indexing does not prove methodological purity. ScienceDB demonstrates clean cow-disjoint partitioning (10,898 unique cows, 0 cross-split leakage). MmCows enforces cow-disjoint splits, but because it contains only 16 cattle total, validation (2 cows) and test (3 cows) are vulnerable to subject-specific evaluation bias. OpenCows2020 features a critical methodological flaw: although the official test set is separate (496 images across 46 cows), the internal train/val split was created by random shuffling of sequential tracking crops, inducing frame-to-frame near-duplicate leakage between train and val. The experimental roadmap is formally locked to a 6-step sequence starting with split fixes and single-task baselines; PCGrad/GradNorm are deferred to Step 6.

---

## 2. Context & Motivation
Following the hostile review critique (§4.2, §5.4.1, §5.5.4 of `P2_hostile_review.md`) and subsequent corrections to project status:
- Overconfident claims that datasets were "clean" or "leak-free" simply because they were indexed needed to be dialed back.
- PCGrad was prematurely scheduled before establishing clean single-task baselines and hard-sharing multi-task models.
- Confusion existed regarding task labels: MmCows was mistakenly recorded as having 5 behaviors instead of 7, and Dryad BCS was mistakenly described as continuous 1–5 instead of discrete folder classes `'2'` through `'6'`.

---

## 3. Forensic Findings & Data

### 3.1 ScienceDB (Body Condition Scoring — Primary)
- **Code**: `context/preprocess_sciencedb_bcs.py`
- **Total Images**: 53,566 images across 5 classes (`3.25`, `3.5`, `3.75`, `4.0`, `4.25`).
- **Grouping Field**: Filename prefixes parsed via regex/stem logic (`GS_xxxx`: 3,347 cows, `YM_xxxx`: 2,056 cows, `ixxxx`: 5,495 cows).
- **Cow IDs Present**: Yes, 10,898 unique cow IDs parsed from stems (100% parse coverage, zero unhandled fallbacks).
- **Cross-Split Cow Overlap**: **0 cows cross splits.**
  - Train: 7,628 cows (37,688 images, ~70.4%)
  - Val: 1,634 cows (7,580 images, ~14.2%)
  - Test: 1,636 cows (8,298 images, ~15.5%)
- **Duplicate / Near-Duplicate Risk**: 744 individual cows possess images scored at different points in time (e.g., cow `GS_1012` scored at 3.50 in Month 1 and 3.75 in Month 2). Because splitting is strictly grouped by cow ID, all images of any given cow remain grouped in the same partition. Zero cow-level identity leakage exists between splits.
- **Class Distribution**:
  - `3.25`: 7,163 (13.4%) | Train: 5,096 | Val: 1,009 | Test: 1,058
  - `3.50`: 13,830 (25.8%) | Train: 9,726 | Val: 1,939 | Test: 2,165
  - `3.75`: 15,116 (28.2%) | Train: 10,689 | Val: 2,143 | Test: 2,284
  - `4.00`: 11,884 (22.2%) | Train: 8,300 | Val: 1,732 | Test: 1,852
  - `4.25`: 5,573 (10.4%) | Train: 3,877 | Val: 757 | Test: 939
  - *Assessment*: Class balance across splits is stable (~13–14%, ~24–26%, ~26–29%, ~22–24%, ~10–11%).

### 3.2 MmCows (Behavior Recognition — Primary)
- **Code**: `context/preprocess_mmcows_behavior.py`
- **Total Images**: 213,686 bounding box crops across **7 active classes** (`1, 2, 3, 4, 5, 6, 7`).
- **Grouping Field**: Cow ID extracted from filename token index 2 (`<timestamp>_<time>_<cow_id>_<crop_id>.jpg`).
- **Cow IDs Present**: Only **16 unique cattle** in the entire dataset (`cow_id` 1 to 16).
- **Split Breakdown**:
  - Train: 11 cows (`1, 3, 4, 6, 8, 9, 10, 11, 12, 14, 15`) -> 148,401 images (69.4%)
  - Val: 2 cows (`7, 13`) -> 25,134 images (11.8%)
  - Test: 3 cows (`2, 5, 16`) -> 40,151 images (18.8%)
- **Cross-Split Cow Overlap**: **0 cows cross splits.**
- **Methodological Vulnerability**:
  1. *Sample Size Vulnerability*: Evaluating on only 2 validation cows and 3 test cows creates high variance and vulnerability to individual animal idiosyncrasies (posture quirks, coat color, specific pen location).
  2. *Severe Class Imbalance*:
     - Class 1 (Walking): 4,118 crops
     - Class 2 (Standing): 70,107 crops
     - Class 3 (Feeding head up): 19,080 crops
     - Class 4 (Feeding head down): 31,255 crops
     - Class 5 (Licking): 2,009 crops
     - Class 6 (Drinking): 3,311 crops
     - Class 7 (Lying): 83,806 crops
     Ratio of majority (Class 7) to minority (Class 5) is ~42:1.
  3. *Frame Redundancy*: Consecutive crops taken seconds apart of a lying cow represent highly correlated redundant data, necessitating per-epoch sample capping.

### 3.3 OpenCows2020 (Individual Cow Identification — Primary)
- **Code**: `context/preprocess_id.py`
- **Total Images**: 4,736 images across 46 closed-set cow classes (`cow_id` 1 to 46).
- **Split Code Logic**:
  - `identification-test` (496 images) is assigned entirely to `test` (46 cows represented).
  - `identification-train` (4,240 images across 46 cows) is split per cow using `random.shuffle()`: 85% train (3,583 images), 15% val (657 images).
- **Leakage Risk (CRITICAL)**:
  - The images in `identification-train` are sequential tracking/video crops (`<cow>_000001.jpg`, `<cow>_000002.jpg`, ...).
  - Shuffling these crops randomly per cow causes adjacent frames from the same video track/recording pass to fall across both `train` and `val`.
  - While closed-set identification requires all 46 cows in train, val, and test, splitting within a sequence leaks temporal and background correlation.
  - The test set is from the official benchmark partition, but the validation set is corrupted by near-duplicate frame correlation.

### 3.4 Dryad BCS (Body Condition Scoring — Secondary / Comparison)
- **Code**: `context/preprocess_bcs.py`
- **Total Images**: 5,923 Depth Grayscale Edge (DGE) images in `Total_sorted_DGE_images.zip`.
- **True Label Values**:
  - The archive folders are named `1` through `9`.
  - Folders `1`, `8`, and `9` are completely empty.
  - Folder `7` contains only 17 images.
  - Folders `2`, `3`, `4`, `5`, `6` contain valid data:
    - `2`: 546 images
    - `3`: 536 images
    - `4`: 2,212 images
    - `5`: 1,962 images
    - `6`: 667 images
  - *Assessment*: Real labels are discrete folder classes `'2', '3', '4', '5', '6'`, mapped ordinally to `0-4`. They are NOT continuous floating values or a simple "1–5" scale.

---

## 4. Synthesis & Audit Summary Table

| Dataset | Current Split | Leakage Risk | Clean Enough? | Required Fix |
| :--- | :--- | :--- | :--- | :--- |
| **ScienceDB** (BCS) | Cow-disjoint 70/14/16 (7,628 train, 1,634 val, 1,636 test cows) | Low / None (0 cows cross splits; all temporal instances of multi-scored cows grouped together) | **YES** | Maintain current cow-grouped split. Add stratified grouping seed if fine-grained class rebalancing is desired. |
| **MmCows** (Behavior) | Cow-disjoint 70/12/19 (11 train, 2 val, 3 test cows) | Low on cow identity (0 cross-split cows); High on subject bias & extreme class imbalance (42:1) | **CONDITIONAL** | Zero cow leak, but tiny evaluation pool (2 val, 3 test cows). Require k-fold cow-grouped CV or repeated seeds; enforce capped sampling during training. |
| **OpenCows2020** (Cow ID) | Official test (496 imgs); Random 85/15 shuffle on train images per cow for val | **HIGH between train and val** (consecutive video crops randomly shuffled into train and val) | **NO (Val split contaminated)** | Keep official test split, but restructure train/val split to be sequence/tracklet-disjoint or block-temporal rather than random frame shuffle. |

---

## 5. Architectural Decisions & Experiment Sequence
The experimental sequence is strictly ordered:
1. **STEP 1**: Audit and verify dataset splits (resolve OpenCows2020 train/val sequence leakage).
2. **STEP 2**: Run clean Single-Task baselines (ScienceDB BCS, MmCows Behavior, OpenCows2020 ID).
3. **STEP 3**: Run clean 3-task Hard-Sharing MTL.
4. **STEP 4**: Run Partial-Sharing MTL.
5. **STEP 5**: Compare Single-Task vs Hard-Sharing vs Partial-Sharing and quantify negative transfer.
6. **STEP 6**: Optionally test PCGrad / GradNorm only if gradient conflict or loss balancing questions require it.

---

## 6. Next Steps
1. Refactor `context/preprocess_id.py` to prevent sequential frame leakage between train and val splits.
2. Present findings and proposed fix to Hasin for review before re-generating manifests.
3. Once splits are validated, establish clean single-task baseline scripts.
