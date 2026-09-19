# OpenCows2020 Legacy Re-ID Evaluation Protocol & Forensic Audit Report

**Date**: 2026-09-20  
**Status**: COMPLETED (LEGACY RE-ID BENCHMARK ONLY)  
**Primary Intended Re-ID**: MultiCamCows2024 (Blocked upstream; OpenCows2020 MUST NOT be promoted to primary)  

---

## Executive Summary

1. **Official Test Set Preservation**: The official `identification-test` benchmark partition (496 images across all 46 cows) is preserved **100% intact and untouched**.
2. **Old Split Flaw (Random Within-Identity Mixing)**: The legacy `context/preprocess_id.py` script applied `random.shuffle()` to frames within each cow, causing extensive within-identity mixing:
   - **1,023 frame-index adjacent pairs** ($|f_1 - f_2| = 1$) crossed between train and validation.
   - **2,942 near frame-index pairs** ($|f_1 - f_2| \le 5$) crossed between train and validation.
   - **3 exact-duplicate image pairs** (identical SHA256 hashes) crossed between train and validation.
3. **Rebuilt Protocol**: Rebuilt using a **Contiguous Frame-Index Heuristic** (first ~85% of sorted frames to Train, remaining ~15% to Val) combined with **Exact-Duplicate Harmonization**.
   - Frame-index adjacency crossings reduced from 1,023 down to 48 (single boundary transition per cow).
   - Exact-duplicate leakage between train and val eliminated to **0**.
   - Exact-duplicate leakage between train/val and test is **0**.
   - **Provenance Limitation**: True tracklet/temporal leakage cannot be verified because provenance is unavailable.

---

## Dataset Overview & Image Counts

| Split | Images | % of Total | Unique Cows | Min Imgs/Cow | Max Imgs/Cow | Mean Imgs/Cow |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 3586 | 75.72% | 46 | 15 | 268 | 78.0 |
| **Validation** | 654 | 13.81% | 46 | 3 | 48 | 14.2 |
| **Test (Official)** | 496 | 10.47% | 46 | 2 | 36 | 10.8 |
| **Total** | 4,736 | 100.0% | 46 | 20 | 352 | 102.96 |

---

## Exact Duplicate Analysis (SHA256)

A full cryptographic audit of all 4,736 images identified **8 duplicate hash groups** (16 images total), all residing inside `identification-train`:

1. `13_000026.jpg` == `13_000172.jpg` (Cow 13)
2. `13_000089.jpg` == `13_000115.jpg` (Cow 13)
3. `15_000066.jpg` == `15_000150.jpg` (Cow 15)
4. `16_000094.jpg` == `16_000099.jpg` (Cow 16)
5. `2_000071.jpg` == `2_000094.jpg` (Cow 2)
6. `6_000007.jpg` == `6_000075.jpg` (Cow 6)
7. `7_000048.jpg` == `7_000123.jpg` (Cow 7)
8. `8_000051.jpg` == `8_000140.jpg` (Cow 8)

**Harmonization**: In the rebuilt protocol, whenever duplicate frames straddled the contiguous cutoff (specifically pairs 1, 5, and 7), the validation image was assigned to train. As a result:
- **Train ∩ Val duplicate hashes: 0**
- **Train ∩ Test duplicate hashes: 0**
- **Val ∩ Test duplicate hashes: 0**

---

## Sequence & Tracklet Reconstruction Findings

### Can sequence / tracklet structure be reconstructed?
**NO.** Strict forensic analysis reveals:
1. **No Temporal Provenance**: The OpenCows2020 export provided by DatasetNinja contains only cropped JPEGs named `<cow_id>_<frame_id:06d>.jpg`. No timestamps, flight logs, camera IDs, or video sources are provided.
2. **Dimension Instability**: In over 75% of consecutive frame transitions ($f_i \to f_{i+1}$), image dimensions jump by >30% (e.g. `(96, 44)` to `(311, 191)`).
3. **Visual Discontinuity**: The Mean Absolute Error (MAE) between consecutive frame numbers is statistically indistinguishable from randomly selected pairs within the same cow (e.g. Cow 1: Consecutive MAE = 69.32 vs Random MAE = 71.25; Cow 2: 54.48 vs 54.77; Cow 3: 68.73 vs 69.41).
4. **Duplicate Dispersion**: Identical duplicate frames occur at widely separated indices (e.g. frame 26 vs frame 172 in Cow 13), indicating frames were pooled or stitched from separate passes.

### Limitation & Scientific Honesty
Because true tracklet boundaries cannot be recovered without inventing fake provenance, we explicitly document this limitation:
- True tracklet/temporal leakage cannot be verified because provenance is unavailable.
- OpenCows2020 cannot provide a verifiable camera-disjoint or tracklet-disjoint evaluation.
- The contiguous frame-index split is a **heuristic** to eliminate random within-identity mixing, not a proven sequence-safe or leakage-free partition.
- **OpenCows2020 remains strictly a LEGACY BASELINE**. MultiCamCows2024 remains the intended primary Re-ID benchmark once upstream server access is restored.

---

## Verification Assertions

- [x] All 46 cow identities present in Train, Validation, and Test partitions.
- [x] Official identification-test benchmark partition (496 images) is 100% preserved.
- [x] Zero exact-duplicate overlap between any partition.
- [x] Reusable manifests generated (`manifest.csv`, `train.csv`, `val.csv`, `test.csv`).
- [x] Legacy `datasets/id/id_index.csv` updated for pipeline compatibility.
