# ScienceDB Cattle BCS: Identity Audit & Leakage-Free Split Report

**Date:** 2026-09-20  
**Dataset Version:** ScienceDB DOI `10.57760/sciencedb.16704`  
**Audit Status:** COMPLETED & VERIFIED LEAK-FREE  

## 1. Executive Summary

- **Total Verified Images:** 53,566 RGB images (and 53,566 corresponding Pascal VOC XML annotations).
- **Total Independent Passage Clusters:** 5,662 (replacing the flawed 10,898 parsed IDs).
- **Split Protocol:** Stratified Group Split by `(farm_source, primary_bcs)` with Random Seed 42.
- **Disjointness:** 100% passage-disjoint across Train, Val, and Test.
  - `train_clusters ∩ val_clusters = 0`
  - `train_clusters ∩ test_clusters = 0`
  - `val_clusters ∩ test_clusters = 0`
- **Sequence Leakage Prevention:** 0 video frames from consecutive camera passages cross splits.
- **Exact Duplicate Leakage:** 0 cross-split duplicates (all 19 duplicate pairs contained within their respective groups).

## 2. Forensic Investigation of the '10,898 Cows' Parser Flaw

### What the Legacy Parser Did:
The legacy script (`context/preprocess_sciencedb_bcs.py`) parsed filenames using naive string splitting:
1. `GS_XXXX_YY` -> parsed as cow `GS_XXXX` (3,347 groups)
2. `YM_XXXX_YY` -> parsed as cow `YM_XXXX` (2,056 groups)
3. `L-iXXXX` / `R-iXXXX` -> parsed as cow `iXXXX` (5,495 groups)
4. **3,347 + 2,056 + 5,495 = 10,898 claimed 'cows'**.

### The Severe Leakage Discovered:
- For `L-i` and `R-i`, `iXXXX` was **NOT a cow ID**; it was the **video frame index** of a continuous 30 FPS stereo camera recording.
- Consecutive frames (e.g. `L-i1035`, `L-i1036`, `L-i1037` spaced ~33 ms apart as the cow walks past the camera) were treated as **completely different cows**.
- When randomly shuffled into train, val, and test, **94.64% of the 261 stereo cow passages had frames scattered across train, val, and test**.
- A model trained on the legacy split was evaluating on nearly identical 30 ms video frames seen in training!

### What the Ground Truth IDs Actually Represent:
- The original authors (Huang et al., 2024) **did not record or release true individual cow ear tag / RFID numbers**.
- The filenames represent **individual cow passage / visit events** through the chute/lane.
- The true observational units are **5,662 passage sequence events**, not 10,898 biological cows.

## 3. De-Leakage Grouping Methodology

To guarantee 100% leakage-free evaluation, all images were consolidated into **5,662 atomic passage clusters**:

1. **GS Subset (29,833 images):** 3,347 burst sequences (`GS_1` through `GS_3347`). Each burst is atomic.
2. **YM Subset (14,761 images):** 2,054 sequences. Identical duplicate clips (`YM_1249` & `YM_1264`, and `YM_1794` & `YM_1905`) were merged via connected components into unified clusters (`YM_1249_1264`, `YM_1794_1905`).
3. **Stereo L/R Subset (8,972 images):** 261 temporal blocks (`STEREO_blk_001` through `STEREO_blk_261`). Consecutive frames within <= 5 frame intervals (~0.2s) and both Left/Right camera views are locked together.

## 4. Exact & Near-Duplicate Analysis

- Total exact duplicate byte-level pairs found: **19 pairs (38 images)**.
- All duplicates were located within the `YM` subset (inter-sequence duplicate clips and intra-sequence paused frames).
- **Cross-Split Duplicate Leakage:** **0 images** (all duplicates locked into same splits).

## 5. Farm & Location Metadata Recovery

- **Recoverable Coarse Sources:**
  - `GS_Gansu`: 29,833 images (3,347 passages) from Wuwei, Gansu.
  - `YM_Farm2`: 14,761 images (2,054 passages) from Yimin Dairy Farm.
  - `STEREO_Farm3`: 8,972 images (261 passages) from third stereo camera facility.
- **Unrecoverable Metadata:** Specific barn IDs, feed rations, cow birth dates, and GPS coordinates are **UNAVAILABLE** in the published dataset.

## 6. Final Split Statistics

### Passage Clusters & Image Totals:

| Split | Passage Clusters | % Clusters | Total Images | % Images |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 3963 | 70.0% | 37126 | 69.3% |
| **Val** | 849 | 15.0% | 8099 | 15.1% |
| **Test** | 850 | 15.0% | 8341 | 15.6% |
| **Total** | **5662** | **100.0%** | **53,566** | **100.0%** |

### BCS Class Distribution Across Splits:

| BCS Class | Overall Count (%) | Train Count (%) | Val Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| **3.25** | 7536 (14.1%) | 5454 (14.7%) | 1032 (12.7%) | 1050 (12.6%) |
| **3.5** | 13256 (24.7%) | 9251 (24.9%) | 2010 (24.8%) | 1995 (23.9%) |
| **3.75** | 14255 (26.6%) | 9908 (26.7%) | 2207 (27.3%) | 2140 (25.7%) |
| **4.0** | 12556 (23.4%) | 8676 (23.4%) | 2003 (24.7%) | 1877 (22.5%) |
| **4.25** | 5963 (11.1%) | 3837 (10.3%) | 847 (10.5%) | 1279 (15.3%) |

### Farm Source Distribution Across Splits:

| Farm Source | Total Images | Train Images | Val Images | Test Images |
| :--- | :--- | :--- | :--- | :--- |
| **GS_Gansu** | 29833 | 20902 | 4455 | 4476 |
| **STEREO_Farm3** | 8972 | 5915 | 1449 | 1608 |
| **YM_Farm2** | 14761 | 10309 | 2195 | 2257 |

## 7. Reusable Verification Code
Run `python scripts/build_sciencedb_splits.py --verify-only` at any time to automatically re-verify that all assertions pass.
