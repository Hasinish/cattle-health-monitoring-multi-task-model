# OpenCows2020 Legacy Re-ID Evaluation Protocol & Forensic Audit

**Date**: 2026-09-20  
**Author**: Hasin Ishrak  
**Status**: COMPLETED (LEGACY RE-ID BENCHMARK ONLY)  
**Primary Intended Re-ID**: MultiCamCows2024 (Blocked upstream; OpenCows2020 MUST NOT be promoted to primary)  

---

## 1. Executive Summary

1. **Legacy Status Retained**: OpenCows2020 remains strictly classified as a **LEGACY BASELINE ONLY**. It is not promoted back to primary Re-ID; MultiCamCows2024 remains the intended primary benchmark once upstream access is restored.
2. **Official Test Partition 100% Preserved**: The official `identification-test` set provided by Andrew et al. (496 images across all 46 cows, ~10.47% of the dataset) is preserved completely intact and untouched.
3. **Flaw in Legacy Preprocessor (Random Within-Identity Mixing)**: `context/preprocess_id.py` previously executed `random.shuffle()` on image frames within each cow to create train and validation splits. This caused extensive random within-identity mixing:
   - **1,023 frame-index adjacent pairs** ($|f_1 - f_2| = 1$) crossed between train and validation.
   - **2,942 near frame-index pairs** ($|f_1 - f_2| \le 5$) crossed between train and validation.
   - **3 exact-duplicate image pairs** (identical SHA256 hashes) crossed between train and validation.
4. **Rebuilt Protocol**: Rebuilt using a **Contiguous Frame-Index Heuristic** (first ~85% of sorted frames to Train, remaining ~15% to Val) combined with **Exact-Duplicate Harmonization**.
   - Frame-index adjacency crossings reduced from 1,023 down to 48 (single boundary transition per cow).
   - Exact-duplicate leakage between train and val eliminated to **0**.
   - Exact-duplicate leakage between train/val and test is **0**.
   - All 46 identities present across train, val, and test.
   - **Provenance Limitation**: True tracklet/temporal leakage cannot be verified because provenance is unavailable.

---

## 2. Context & Motivation

Phase 3 Step 1 of the canonical roadmap requires establishing defensible, reproducible evaluation protocols for all active datasets before any model training begins.

OpenCows2020 was introduced by William Andrew et al. (University of Bristol, 2020 / 2021) as a visual identification benchmark of 46 Holstein-Friesian cattle using dorsal coat patterns. The local export from DatasetNinja contains two directories: `identification-train` (4,240 images) and `identification-test` (496 images), totaling 4,736 images.

The legacy project preprocessor (`context/preprocess_id.py`) randomly shuffled the 4,240 training images within each cow to produce an 85/15 train/val split. However, random within-identity mixing placed near-duplicate and exact-duplicate frames across the split, artificially inflating validation accuracy. This audit investigated the source structure to determine whether true sequence/tracklet grouping could be reconstructed, quantified the crossings and duplicates, and established a defensible evaluation protocol.

---

## 3. Forensic Findings & Data

### 3.1 Dataset Census & Class Distribution
- **Total Images**: 4,736
- **Unique Cow Identities**: 46 (labeled `1` through `46`)
- **Partitions**:
  - `identification-train`: 4,240 images across all 46 cows (min 18, max 316, mean 92.2)
  - `identification-test`: 496 images across all 46 cows (min 2, max 36, mean 10.8)
  - Ratio: Exactly ~89.5% train / 10.5% test across each individual cow.

### 3.2 Duplicate Analysis (SHA256)
A complete cryptographic audit across all 4,736 images revealed:
- **Total Unique Hashes**: 4,728 out of 4,736.
- **Duplicate Hash Groups**: Exactly 8 duplicate pairs (16 images total), all located within `identification-train`:
  1. `13_000026.jpg` == `13_000172.jpg` (Cow 13)
  2. `13_000089.jpg` == `13_000115.jpg` (Cow 13)
  3. `15_000066.jpg` == `15_000150.jpg` (Cow 15)
  4. `16_000094.jpg` == `16_000099.jpg` (Cow 16)
  5. `2_000071.jpg` == `2_000094.jpg` (Cow 2)
  6. `6_000007.jpg` == `6_000075.jpg` (Cow 6)
  7. `7_000048.jpg` == `7_000123.jpg` (Cow 7)
  8. `8_000051.jpg` == `8_000140.jpg` (Cow 8)
- **Test Set Cleanliness**: `identification-test` has **0 duplicate hashes** internally and **0 duplicate hashes** crossing with `identification-train`.

### 3.3 Can Sequence / Tracklet Structure Be Reconstructed?
**NO.** Forensic analysis proved that true sequence and tracklet boundaries cannot be reconstructed from provenance:
1. **Missing Metadata**: The DatasetNinja distribution contains only pre-cropped JPEGs named `<cow_id>_<frame_id:06d>.jpg`. No timestamps, flight logs, camera IDs, or raw video streams are provided.
2. **Dimension Instability**: In over 75% of consecutive frame transitions ($f_i \to f_{i+1}$), image dimensions jump by >30% (e.g. `(96, 44)` to `(311, 191)`).
3. **Visual Discontinuity**: Mean Absolute Error (MAE) between consecutive frame numbers is statistically indistinguishable from randomly sampled pairs of the same cow:
   - Cow 1: Consecutive MAE = 69.32 vs Random MAE = 71.25
   - Cow 2: Consecutive MAE = 54.48 vs Random MAE = 54.77
   - Cow 3: Consecutive MAE = 68.73 vs Random MAE = 69.41
4. **Duplicate Dispersion**: Identical duplicate frames occur at widely separated indices (e.g. frame 26 vs frame 172 in Cow 13), indicating frames were pooled or concatenated across multiple flights/passages.

### 3.4 Flaw in Legacy Split (`preprocess_id.py`): Random Within-Identity Mixing
Because `context/preprocess_id.py` shuffled frames randomly:
- **1,023 frame-index adjacent pairs** ($|f_1 - f_2| = 1$) crossed between train and validation.
- **2,942 near frame-index pairs** ($|f_1 - f_2| \le 5$) crossed between train and validation.
- **1,760 visually similar pairs** (MAE < 15) crossed between train and validation (2.07% of all train-val pairs).
- **3 exact-duplicate image pairs** (Cow 13 frames 26/172, Cow 2 frames 71/94, Cow 7 frames 48/123) crossed between train and validation.

---

## 4. Architectural Decisions & Action Plan

1. **Preserve Official Benchmark Test Set**:
   - `identification-test` (496 images) is maintained 100% untouched as the official evaluation test set.
2. **Contiguous Frame-Index Heuristic**:
   - For each cow in `identification-train`, images are sorted by frame ID.
   - The first ~85% of sorted frames form the training split; the remaining ~15% form the validation split.
   - Frame-index adjacency crossings are reduced from 1,023 down to 48 (single boundary transition per cow).
3. **Exact-Duplicate Harmonization**:
   - Sibling pairs of identical SHA256 hashes that straddled the cutoff boundary are assigned together to the training partition.
   - Result: Exactly 0 duplicate hashes cross between train and val, and 0 cross with test.
4. **Final Partition Counts**:
   - **Train**: 3,586 images (46 cows)
   - **Validation**: 654 images (46 cows)
   - **Test**: 496 images (46 cows, official benchmark)
   - **Total**: 4,736 images
5. **Defensible Heuristic, Not True Sequence Split**:
   - Because true tracklets cannot be proven without inventing fake provenance, we explicitly acknowledge that contiguous frame-index block splitting is a defensible heuristic to eliminate random within-identity frame hopping, but true tracklet/temporal leakage cannot be verified because provenance is unavailable.
   - It does NOT guarantee camera- or flight-disjoint evaluation.
   - This limitation reinforces why OpenCows2020 must remain a **LEGACY BASELINE ONLY**.

---

## 5. Artifacts & File Registry

| File | Purpose | Rows / Size |
| :--- | :--- | :--- |
| `datasets/id/opencow2020/manifest.csv` | Master manifest with SHA256, dimensions, official split, and new split | 4,736 rows |
| `datasets/id/opencow2020/train.csv` | Contiguous frame-index training split (0 exact-duplicate overlap) | 3,586 rows |
| `datasets/id/opencow2020/val.csv` | Contiguous frame-index validation split (0 exact-duplicate overlap) | 654 rows |
| `datasets/id/opencow2020/test.csv` | Official test benchmark partition | 496 rows |
| `datasets/id/opencow2020/split_report.md` | Comprehensive audit and protocol report | Markdown |
| `datasets/id/id_index.csv` | Updated pipeline-compatible index file | 4,736 rows |
| `scripts/build_opencows_splits.py` | Deterministic split builder and report generator | Python |
| `scripts/verify_opencows_splits.py` | Standalone verification script asserting counts, IDs, and 0 duplicate overlap | Python |
| `datasets/dataset_registry.csv` | Updated canonical dataset registry | 13 datasets |

---

## 6. Next Steps

1. Commit and push all OpenCows2020 protocol artifacts, manifests, scripts, and logs.
2. Maintain OpenCows2020 strictly as a legacy baseline; do not promote to primary Re-ID.
3. Continue Phase 3 Step 1 roadmap tasks (external dataset manifests or model setup when Step 1 data audit phase concludes).

