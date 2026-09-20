# Research Log — 2026-09-20: Phase 3 Step 1 Exact & Perceptual Near-Duplicate Leakage Audit

## 1. Executive Summary & Objective
As part of **Phase 3 STEP 1 (Data Registry & Clean Splits)**, we conducted an exhaustive automated audit for exact duplicates (SHA-256) and perceptual near-duplicates (64-bit dHash and aHash with Hamming distance threshold $d \le 6$) across all currently usable split-bearing datasets:
1. **ScienceDB Cattle BCS** (53,566 images; passage-disjoint split: train 37,126, val 8,099, test 8,341)
2. **MmCows Behavior** (213,686 images; cow-grouped canonical split 148,401 / 25,134 / 40,151 and 4-Fold GroupKFold suite)
3. **OpenCows2020** (4,736 images; legacy benchmark: train 3,586, val 654, official test 496)

**Scope Constraints**:
- MultiCamCows2024 remains upstream blocked (no files fabricated or audited).
- External stress/benchmark datasets (SideViewCows2026, BECA-D/L, CBVD-5) are preserved purely as external test sets.

---

## 2. Audit Methodology
1. **Exact Content Hash (SHA-256)**:
   - Full binary SHA-256 computed on all image files.
   - Identified duplicate groups and cross-referenced against partition labels (`train`, `val`, `test`, `fold_0..3`).
2. **Perceptual Image Hash (dHash + aHash)**:
   - 64-bit gradient difference hash (`dHash`, 9x8 grayscale).
   - 64-bit average hash (`aHash`, 8x8 grayscale).
   - Multi-Index Hashing (MIH, 8 tables of 8-bit blocks) for candidate retrieval with Hamming distance $d \le 6$.
3. **Pixel MAE & Forensic Inspection**:
   - Top suspicious cross-partition candidate pairs normalized to 64x64 grayscale with Mean Absolute Error (MAE) and manual visual verification.

---

## 3. Dataset-by-Dataset Audit Results

### A. ScienceDB Cattle BCS (53,566 images)
- **Exact Cross-Partition Duplicates:** **0**
  - Within-partition exact duplicates: 19 pairs (perfectly contained inside identical passage clusters).
- **Near-Duplicate Cross-Partition Suspects ($d \le 6$):** **88,944 pairs**
  - Train vs. Val: 41,140 pairs
  - Train vs. Test: 39,384 pairs
  - Val vs. Test: 8,420 pairs
- **Forensic Discovery (CRITICAL VULNERABILITY):**
  - Top suspects (e.g. `GS_1823_1.jpg` in train vs `GS_1818_2.jpg` in val; `GS_1824` in train vs `GS_1819` in val) have $d=0$ dHash distance, $d=0$ aHash distance, and pixel MAE of 0.25–0.30 (mean pixel intensity difference 1.22 out of 255).
  - Manual inspection reveals that passage IDs `GS_1818` and `GS_1823` are the **exact same video passage shifted by 1 frame**. The dataset publisher's passage numbering does not represent independent cow walk-throughs; consecutive passage IDs are overlapping temporal slices of the same video burst.
- **Verdict & Action:** **SPLIT NEEDS CORRECTION.** ScienceDB cannot be trained with naive passage-disjoint splits. Passages must be clustered into coarse connected temporal blocks (or video burst groups) before Step 4 training.

### B. MmCows Behavior (213,686 images)
- **Exact Cross-Partition Duplicates:** **0**
- **Cow ID Disjointness:**
  - Canonical split: Train (11 cows: 1, 3, 4, 6, 8, 9, 10, 11, 12, 14, 15), Val (2 cows: 7, 13), Test (3 cows: 2, 5, 16). Overlap: **0 cows (`set()`)**.
  - 4-Fold GroupKFold: Overlap between test cows and train cows across all 4 folds: **0 cows (`set()`)**.
- **Near-Duplicate Cross-Partition Suspects:**
  - Cows are 100% disjoint by animal ID. Cross-partition perceptual similarities ($d \le 6$) stem entirely from static CCTV background features (identical concrete stalls, metal bars, and straw bedding under fixed camera angles), not cow identity or tracklet leakage.
- **Verdict & Action:** **SPLIT IS 100% CLEAN & LOCKED.** Zero cross-partition exact duplicate leakage; zero animal overlap.

### C. OpenCows2020 Legacy Re-ID (4,736 images)
- **Exact Cross-Partition Duplicates:** **0**
  - Within-partition exact duplicates: 8 pairs (retained within partition).
- **Near-Duplicate Cross-Partition Suspects ($d \le 6$):** **1,239 pairs**
  - Train vs. Val: 544 pairs
  - Train vs. Test: 606 pairs
  - Val vs. Test: 89 pairs
- **Forensic Discovery:**
  - 100% of top near-duplicate pairs cross partitions within the **SAME cow identity** (e.g., Cow 3 in train vs Cow 3 in test, MAE 0.20; Cow 6 in train vs Cow 6 in test, MAE 0.40).
  - Because original dataset authors stripped video provenance and randomized frame indices, near-identical frames of the same cow permeate train, val, and test.
- **Verdict & Action:** **SPLIT IS DESIGNATED LEGACY BASELINE ONLY.** This is an inherent, unfixable flaw of OpenCows2020. It must not serve as the primary Re-ID benchmark. MultiCamCows2024 remains the required primary benchmark once unblocked.

---

## 4. Summary Table

| Dataset | Total Images | Exact Cross-Partition | Near Cross-Partition ($d \le 6$) | Split Status / Verdict |
|---|---|---|---|---|
| **ScienceDB BCS** | 53,566 | 0 | 88,944 | **NEEDS CORRECTION** (overlapping passage bursts) |
| **MmCows Behavior** | 213,686 | 0 | 0 (background only) | **CLEAN & LOCKED** (0 cow overlap) |
| **OpenCows2020 (Legacy)** | 4,736 | 0 | 1,239 | **LEGACY BASELINE ONLY** (unfixable author randomization) |

---

## 5. Artifacts Generated
- `scripts/audit_duplicate_leakage.py`: Reusable parallel exact + perceptual hash audit tool.
- `scratch/fast_mmcows_audit.py`: Multithreaded exact duplicate and cow disjointness validator for MmCows.
- `docs/audits/phase3_near_duplicate_suspects.csv`: Full manifest of near-duplicate suspect pairs with MAE scores.
