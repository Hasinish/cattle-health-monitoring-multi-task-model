# MmCows Behavior Dataset: Grouped Evaluation Protocol & Leakage Audit Report

**Date**: 2026-09-20  
**Task**: Phase 3 Step 1 — Primary Behavior Recognition In-Domain Benchmark  
**Dataset**: MmCows (`neis-lab/mmcows`, NeurIPS 2024 Spotlight)  
**Status**: VERIFIED / LEAKAGE-SAFE / READY FOR PHASE 3  

---

## 1. Executive Summary

This audit establishes the rigorous, leakage-safe evaluation protocol for the **MmCows** behavior recognition dataset. MmCows contains **213,686 bounding-box crops** across **7 active behavioral classes** collected from **16 Holstein dairy cows** using **4 synchronized overhead CCTV cameras** over an uninterrupted 21-hour deployment (July 25–26, 2023) at Purdue University.

### Key Findings:
1. **Cow IDs are 100% Genuine Biological Subjects**: Ground truth cow IDs (1–16) are backed by wearable UWB location tags, neck IMUs, ankle accelerometers, and ear tags in an active dairy research barn. Zero synthetic, random, or ambiguous identifiers exist.
2. **Full Provenance & Temporal Reconstruction**: 100% of the 213,686 filenames follow the strict deterministic schema `<epoch>_<time>_<cow_id>_<cam_id>.jpg`. Epoch timestamps (1690271846 to 1690347431) map with 100% mathematical consistency to 15-second periodic sampling across three contiguous time blocks.
3. **Synchronized Multi-Camera Protection**: 87.15% of behavioral events (64,830 of 74,388 events) were captured simultaneously by 2 to 4 cameras. By enforcing strict **Cow-Disjoint Grouping**, all synchronized views of any cow at any instant are guaranteed to reside in the exact same partition, completely eliminating multi-view cross-contamination.
4. **Class Imbalance & Rare Behavior Mitigation**: Class 7 (Lying, 83,806 crops) outnumbers Class 5 (Licking, 2,009 crops) by 41.7:1. Crucially, 5 cows (1, 5, 13, 14, 16) exhibited **zero licking behavior**. The legacy split tested only 3 cows, making evaluation fragile. We provide both the backward-compatible canonical split AND a balanced **4-Fold GroupKFold Cross-Validation Suite** that guarantees positive sample support for all 7 classes in every partition while evaluating 100% of the 16 cows.
5. **Zero Leakage**: Strict assertions confirm zero cross-split cow identity overlap, zero synchronized multi-camera event leakage, and zero time-block fragmentation.

---

## 2. Dataset Provenance & Physical Parameters

| Metric | Value | Verification Source |
| :--- | :--- | :--- |
| **Original Paper** | *MmCows: A Large-scale Multimodal Dataset for Dairy Cattle Behavior Monitoring and Health Management* | NeurIPS 2024 Spotlight (NEIS Lab, Purdue University) |
| **Dataset DOI / HF** | `https://huggingface.co/datasets/cair/MmCows` | Verified Hugging Face snapshot |
| **Total Verified Crops** | **213,686** | 100% indexed and physically verified |
| **Total Biological Cows** | **16** (Holstein dairy cattle) | Physical RFID/UWB tagged subjects |
| **Cameras** | **4** synchronized CCTV cameras | 1 (53,506), 2 (54,808), 3 (53,872), 4 (51,500) |
| **Recording Duration** | 21.00 hours (July 25, 2023 13:57:26 to July 26, 2023 10:57:11 EDT) | Unix epoch range: 1690271846 to 1690347431 |
| **Sampling Frequency** | 15 seconds (4,765 of 4,767 intervals = 15.0s) | Periodic multi-view frame extraction |
| **Synchronized Events** | 74,388 unique `(epoch, cow)` events | 0 label conflicts across cameras (100% unanimous) |
| **Contiguous Time Blocks** | 3 blocks separated by 2 natural operational gaps (>30m) | Block 1 (1.5h), Block 2 (10.5h), Block 3 (7.8h) |

---

## 3. Behavioral Class Distribution & Imbalance

| Class ID | Behavior Name | Total Crops | Percent | Canonical Train | Canonical Val | Canonical Test | Imbalance Ratio (vs C5) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Walking | 4,118 | 1.93% | 2,759 | 444 | 915 | 2.0:1 |
| **2** | Standing | 70,107 | 32.81% | 49,848 | 5,059 | 15,200 | 34.9:1 |
| **3** | Feeding_head_up | 19,080 | 8.93% | 12,178 | 3,299 | 3,603 | 9.5:1 |
| **4** | Feeding_head_down | 31,255 | 14.63% | 20,846 | 3,156 | 7,253 | 15.6:1 |
| **5** | Licking | 2,009 | 0.94% | 1,365 | 432 | 212 | 1.0:1 |
| **6** | Drinking | 3,311 | 1.55% | 2,306 | 275 | 730 | 1.6:1 |
| **7** | Lying | 83,806 | 39.22% | 59,099 | 12,469 | 12,238 | 41.7:1 |
| **Total** | **All 7 Classes** | **213,686** | **100.0%** | **148,401** | **25,134** | **40,151** | — |

---

## 4. Cow Subject Census & Behavioral Support

| Cow ID | Total Crops | Unique Epochs | Cam 1 | Cam 2 | Cam 3 | Cam 4 | C1 (Walk) | C2 (Stand) | C3 (FeedUp) | C4 (FeedDn) | C5 (Lick) | C6 (Drink) | C7 (Lie) | Dominant | Canonical | Test Fold |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **1** | 14,029 | 4,714 | 2,742 | 3,509 | 3,973 | 3,805 | 275 | 3800 | 749 | 1565 | **0** | 242 | 7398 | Lying | train | Fold_0 |
| **2** | 13,294 | 4,709 | 4,238 | 3,441 | 2,696 | 2,919 | 174 | 4686 | 1060 | 2543 | **212** | 144 | 4475 | Standing | test | Fold_0 |
| **3** | 13,494 | 4,718 | 2,772 | 3,747 | 3,652 | 3,323 | 175 | 3169 | 2053 | 1469 | **171** | 305 | 6152 | Lying | train | Fold_2 |
| **4** | 15,550 | 4,732 | 3,793 | 4,139 | 4,311 | 3,307 | 193 | 5845 | 1926 | 1563 | **200** | 184 | 5639 | Standing | train | Fold_1 |
| **5** | 14,755 | 4,676 | 4,186 | 4,013 | 3,565 | 2,991 | 314 | 6305 | 1520 | 1990 | **0** | 259 | 4367 | Standing | test | Fold_1 |
| **6** | 15,292 | 4,729 | 3,855 | 4,288 | 3,485 | 3,664 | 333 | 7427 | 1438 | 2457 | **208** | 253 | 3176 | Standing | train | Fold_2 |
| **7** | 10,425 | 4,449 | 2,466 | 2,504 | 2,761 | 2,694 | 192 | 1887 | 1442 | 1597 | **432** | 83 | 4792 | Lying | val | Fold_0 |
| **8** | 12,786 | 4,679 | 2,438 | 2,856 | 4,013 | 3,479 | 392 | 4386 | 807 | 2101 | **16** | 268 | 4816 | Lying | train | Fold_2 |
| **9** | 13,332 | 4,701 | 2,957 | 3,154 | 3,447 | 3,774 | 255 | 4823 | 483 | 2526 | **140** | 137 | 4968 | Lying | train | Fold_1 |
| **10** | 12,368 | 4,675 | 3,720 | 3,608 | 1,745 | 3,295 | 295 | 4613 | 765 | 1377 | **137** | 170 | 5011 | Lying | train | Fold_0 |
| **11** | 12,721 | 4,630 | 3,507 | 3,807 | 3,178 | 2,229 | 117 | 2766 | 1120 | 838 | **244** | 97 | 7539 | Lying | train | Fold_1 |
| **12** | 15,186 | 4,747 | 3,572 | 4,226 | 3,711 | 3,677 | 260 | 6909 | 1958 | 1792 | **125** | 279 | 3863 | Standing | train | Fold_3 |
| **13** | 14,709 | 4,724 | 3,907 | 4,242 | 3,596 | 2,964 | 252 | 3172 | 1857 | 1559 | **0** | 192 | 7677 | Lying | val | Fold_2 |
| **14** | 10,626 | 4,447 | 3,081 | 2,142 | 2,777 | 2,626 | 181 | 2281 | 224 | 2500 | **0** | 123 | 5317 | Lying | train | Fold_3 |
| **15** | 13,017 | 4,627 | 3,447 | 2,712 | 3,497 | 3,361 | 283 | 3829 | 655 | 2658 | **124** | 248 | 5220 | Lying | train | Fold_3 |
| **16** | 12,102 | 4,431 | 2,825 | 2,420 | 3,465 | 3,392 | 427 | 4209 | 1023 | 2720 | **0** | 327 | 3396 | Standing | test | Fold_3 |

> [!IMPORTANT]
> **Class 5 (Licking) Rarity Warning**:
> Notice that Cows 1, 5, 13, 14, and 16 have **0 licking crops**. An unstratified split that assigns only zero-licking cows to a test or validation partition renders evaluation of Class 5 mathematically impossible (support = 0). The 4-Fold GroupKFold suite explicitly solves this by distributing licking-active cows to guarantee positive support in every partition.

---

## 5. Grouped Protocols

### 5.1 Canonical Single Split (Backward-Compatible Baseline)
- **Train (11 cows)**: `[1, 3, 4, 6, 8, 9, 10, 11, 12, 14, 15]` -> 148,401 crops (69.45%)
- **Val (2 cows)**: `[7, 13]` -> 25,134 crops (11.76%)
- **Test (3 cows)**: `[2, 5, 16]` -> 40,151 crops (18.79%)
- **Cross-Split Cow Overlap**: **0 cows** (Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0)
- **Cross-Split Multi-Camera Leakage**: **0 events**
- **Cross-Split Time-Block Leakage**: **0 time blocks**

### 5.2 4-Fold GroupKFold Cross-Validation Protocol (Primary Recommendation)
Because testing on only 3 cattle induces high variance due to animal idiosyncrasies, the 4-Fold GroupKFold protocol partitions all 16 cattle so that **100% of cows are tested**:

| Fold | Partition | Cow IDs | Image Count | Percentage | Class 5 (Lick) Support | Class 1 (Walk) Support |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 0** | Train | `[3, 5, 6, 8, 11, 12, 13, 14, 15, 16]` | 134,688 | 63.0% | 888 | 2,734 |
| **Fold 0** | Val | `[4, 9]` | 28,882 | 13.5% | 340 | 448 |
| **Fold 0** | Test | `[1, 2, 7, 10]` | 50,116 | 23.5% | 781 | 936 |
| **Fold 1** | Train | `[1, 2, 6, 7, 10, 12, 13, 14, 15, 16]` | 131,048 | 61.3% | 1,238 | 2,672 |
| **Fold 1** | Val | `[3, 8]` | 26,280 | 12.3% | 187 | 567 |
| **Fold 1** | Test | `[4, 5, 9, 11]` | 56,358 | 26.4% | 584 | 879 |
| **Fold 2** | Train | `[1, 2, 4, 5, 7, 9, 10, 11, 14, 16]` | 129,202 | 60.5% | 1,365 | 2,423 |
| **Fold 2** | Val | `[12, 15]` | 28,203 | 13.2% | 249 | 543 |
| **Fold 2** | Test | `[3, 6, 8, 13]` | 56,281 | 26.3% | 395 | 1,152 |
| **Fold 3** | Train | `[1, 3, 4, 5, 6, 7, 8, 9, 11, 13]` | 137,093 | 64.2% | 1,411 | 2,498 |
| **Fold 3** | Val | `[2, 10]` | 25,662 | 12.0% | 349 | 469 |
| **Fold 3** | Test | `[12, 14, 15, 16]` | 50,931 | 23.8% | 249 | 1,151 |

---

## 6. Leakage Audit & Verification Assertions

1. **Identity Overlap**:  
   - Canonical Split: `train_cows ∩ val_cows = ∅`, `train_cows ∩ test_cows = ∅`, `val_cows ∩ test_cows = ∅` (VERIFIED).  
   - All 4 Folds: `train_cows ∩ val_cows = ∅`, `train_cows ∩ test_cows = ∅`, `val_cows ∩ test_cows = ∅` (VERIFIED).  
2. **Synchronized Multi-Camera Views**:  
   - Zero events have crops assigned to different splits. If Cow 5 is in Test, all 4 camera angles at timestamp T are in Test. (VERIFIED across all 74,388 events).
3. **Contiguous Time Blocks**:  
   - Periodic 15s frame sequences never cross partitions because grouping is strictly by cow. (VERIFIED).
4. **Exact Content Duplicates**:  
   - All 213,686 records possess a unique `(epoch, cow_id, camera_id)` key. Zero duplicate keys exist.

---

## 7. Artifact Registry

- Master Manifest: `datasets/behavior/mmcows/manifest.csv` (213,686 rows, 15 fields)
- Provenance Audit: `datasets/behavior/mmcows/provenance_audit.csv` (16 cows, 20 fields)
- Canonical Splits: `train.csv` (148,401), `val.csv` (25,134), `test.csv` (40,151)
- 4-Fold Suite: `datasets/behavior/mmcows/folds/fold_[0-3].csv` (each 213,686 rows with split column)
- Reusable Builder & Assertion Script: `scripts/build_mmcows_splits.py`

**Conclusion**: MmCows Behavior is 100% verified, leak-free, and ready for Phase 3 Step 1 baseline training.
