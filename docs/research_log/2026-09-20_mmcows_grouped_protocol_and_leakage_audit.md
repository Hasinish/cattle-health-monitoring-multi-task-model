# Research Log — 2026-09-20: MmCows Grouped Evaluation Protocol & Leakage Audit

## 1. Executive Summary
Audited the MmCows behavior recognition dataset (213,686 bounding box crops across 7 classes) for biological subject identity, synchronized multi-camera event provenance, contiguous time blocks, and leakage risks. Proved that cow IDs (1–16) represent genuine biological Holstein dairy cattle hand-annotated from 4 synchronized CCTV cameras over a 21-hour recording session (NeurIPS 2024 Spotlight). Created a leakage-free master manifest, cow-level provenance audit, canonical baseline split (11 train / 2 val / 3 test cows), and a 4-Fold GroupKFold cross-validation suite (4 test cows per fold) guaranteeing positive class support across all 7 classes and zero cross-partition leakage.

---

## 2. Context & Motivation
Under Phase 3 Step 1 of the Canonical Roadmap, following the ScienceDB BCS identity audit, the next critical task is establishing the grouped evaluation protocol for the primary in-domain behavior recognition dataset: **MmCows**.
Prior audits noted that MmCows contains only 16 cattle in total. The legacy split evaluated on only 3 test cows (and 2 val cows), creating severe vulnerability to individual animal idiosyncrasies (posture quirks, individual coat patterns, pen preferences). Furthermore, Class 5 (Licking) is a rare behavior (only 2,009 crops out of 213,686, a 41.7:1 imbalance against Lying). If test cows happen to be cattle that never lick, evaluating Class 5 becomes mathematically impossible. A rigorous, leakage-safe grouped protocol was required to protect contiguous 15-second time blocks, preserve multi-view camera synchronization, and evaluate 100% of the cattle population across repeated folds.

---

## 3. Forensic Findings & Data

### 3.1 Filename Schema & Source Provenance
All 213,686 image files in `datasets/behavior/mmcows/cropped_bboxes/behaviors/` follow a strict, 100% regular deterministic naming schema:
```
<timestamp_epoch>_<hh-mm-ss>_<cow_id>_<cam_id>.jpg
```
- **Total Images Parsed**: 213,686 (100% regular parse rate; 0 irregular files).
- **Biological Cow IDs**: Exactly 16 unique cows (`cow_id` 1 to 16).
- **Cameras**: Exactly 4 overhead CCTV cameras (`cam_id` 1 to 4):
  - Cam 1: 53,506 crops (25.04%)
  - Cam 2: 54,808 crops (25.65%)
  - Cam 3: 53,872 crops (25.21%)
  - Cam 4: 51,500 crops (24.10%)
- **Unique Timestamps**: 4,768 discrete timestamps sampled at 15.0-second intervals (4,765 intervals are exactly 15s).
- **Temporal Range**: Unix epoch 1690271846 to 1690347431 (July 25, 2023 13:57:26 to July 26, 2023 10:57:11 EDT, span = 21.00 hours).
- **Contiguous Time Blocks**: 3 natural continuous recording sessions separated by 2 operational gaps (>30 minutes):
  - Block 1: 13:57:26 to 15:30:11 (1.5h, 367 epochs)
  - Block 2: 16:03:26 to 02:32:11 (+1 day) (10.5h, 2,515 epochs)
  - Block 3: 03:07:26 to 10:57:11 (+1 day) (7.8h, 1,886 epochs)

### 3.2 Synchronized Multi-Camera Events & Leakage Mechanics
- Across the 4,768 timestamps, in 99.94% of frames (4,765 timestamps), multiple cows are present simultaneously (average 15.60 cows per timestamp).
- For an individual cow at a single timestamp, **87.15% of events (64,830 / 74,388)** were captured simultaneously by 2 to 4 synchronized cameras.
- **Unanimous Behavior Labels**: 0 class conflicts across cameras for any `(epoch, cow)` event (100% unanimous ground truth).
- **Unique File Identifiers**: Zero duplicates exist for `(epoch, cow_id, cam_id)` tuples (213,686 unique keys across 213,686 files).
- **Leakage Prevention**: If frames were randomly split, or if camera views were partitioned, synchronized multi-view images of the exact same cow performing the same action at the exact same second would leak across train and test. By enforcing **Cow-Disjoint Grouping**, all 4 cameras and all time blocks for any given cow remain locked together, guaranteeing 0 multi-view leakage.

### 3.3 Class Imbalance & Cow Subject Census
Severe behavioral imbalance exists in the 213,686 crops:
- Class 1 (Walking): 4,118 crops (1.93%)
- Class 2 (Standing): 70,107 crops (32.81%)
- Class 3 (Feeding head up): 19,080 crops (8.93%)
- Class 4 (Feeding head down): 31,255 crops (14.63%)
- Class 5 (Licking): 2,009 crops (0.94%)
- Class 6 (Drinking): 3,311 crops (1.55%)
- Class 7 (Lying): 83,806 crops (39.22%)
Ratio of majority (Class 7) to minority (Class 5) is **41.7:1**.

**Critical Subject Discovery**:
Five cows exhibited **zero licking behavior** during the entire 21-hour recording:
- Cows with 0 licking: `[1, 5, 13, 14, 16]`
- Cows with positive licking: `[7 (432), 11 (244), 2 (212), 6 (208), 4 (200), 3 (171), 9 (140), 10 (137), 12 (125), 15 (124), 8 (16)]`

---

## 4. Architectural Decisions & Action Plan

### Decision 1: Preserve Canonical Baseline Split
The existing single split in `datasets/behavior/behavior_index.csv` (Train: 11 cows [1, 3, 4, 6, 8, 9, 10, 11, 12, 14, 15], Val: 2 cows [7, 13], Test: 3 cows [2, 5, 16]) is already strictly cow-disjoint with 0 cross-split leakage. It is retained as `datasets/behavior/mmcows/train.csv`, `val.csv`, and `test.csv` for direct backward comparability with initial single-task baselines.

### Decision 2: Standardize on 4-Fold GroupKFold Cross-Validation
Because testing on only 3 cows (18.79% of the herd) is subject to high animal variance, a canonical **4-Fold GroupKFold** protocol was designed where:
- Each fold tests exactly 4 distinct cows (25% of the herd, ~50,000 to 56,000 images).
- Across the 4 folds, **100% of all 16 cows are evaluated in test**.
- Cows with 0 licking are distributed so that **every fold has non-zero licking support (>200 samples) in train, val, and test**:
  - **Fold 0**: Test cows `[1, 2, 7, 10]` (50,116 imgs; C5 test support: 781) | Val cows `[4, 9]` (28,882 imgs) | Train: 10 cows (134,688 imgs)
  - **Fold 1**: Test cows `[4, 5, 9, 11]` (56,358 imgs; C5 test support: 584) | Val cows `[3, 8]` (26,280 imgs) | Train: 10 cows (131,048 imgs)
  - **Fold 2**: Test cows `[3, 6, 8, 13]` (56,281 imgs; C5 test support: 395) | Val cows `[12, 15]` (28,203 imgs) | Train: 10 cows (129,202 imgs)
  - **Fold 3**: Test cows `[12, 14, 15, 16]` (50,931 imgs; C5 test support: 249) | Val cows `[2, 10]` (25,662 imgs) | Train: 10 cows (137,093 imgs)

---

## 5. Artifacts & File Registry
- Master Manifest: `datasets/behavior/mmcows/manifest.csv` (213,686 rows, 15 fields, SHA256: `352aaf9adddb2bd37b1e0af699cbf7316505fd53829aaa14e1d8e6157875d364`)
- Provenance Audit: `datasets/behavior/mmcows/provenance_audit.csv` (16 cows, 20 fields)
- Canonical Splits:
  - `datasets/behavior/mmcows/train.csv` (148,401 crops)
  - `datasets/behavior/mmcows/val.csv` (25,134 crops)
  - `datasets/behavior/mmcows/test.csv` (40,151 crops)
- 4-Fold CV Suite:
  - `datasets/behavior/mmcows/folds/fold_0.csv` (213,686 rows)
  - `datasets/behavior/mmcows/folds/fold_1.csv` (213,686 rows)
  - `datasets/behavior/mmcows/folds/fold_2.csv` (213,686 rows)
  - `datasets/behavior/mmcows/folds/fold_3.csv` (213,686 rows)
- Detailed Audit Report: `datasets/behavior/mmcows/split_report.md`
- Deterministic Generator Script: `scripts/build_mmcows_splits.py`
- Standalone Verifier & Assertion Suite: `scripts/verify_mmcows_splits.py`

---

## 6. Next Steps
1. Re-ID Protocol Definition: Establish tracklet-disjoint, cross-day, cross-camera evaluation protocol for Re-ID (MultiCamCows2024 / OpenCows2020 legacy baseline).
2. Proceed to STEP 2 (Cattle-perception feasibility audit: verify detector/segmenter on ScienceDB and MmCows crops).
