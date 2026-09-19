# Dryad Cattle BCS Dataset: Discrepancy Investigation & Forensic Audit Report

**Date**: 2026-09-20  
**Task**: Phase 3 Step 1 — Secondary External BCS Benchmark Discrepancy Resolution  
**Dataset**: Dryad Cattle BCS (`Total_sorted_DGE_images.zip`, DOI: `10.5061/dryad.tqjq2bw4s`)  
**Authors**: Zachary Winkler & Laura Boucheron (New Mexico State University)  
**Status**: DISCREPANCY FULLY RESOLVED / VERIFIED / READY FOR EXTERNAL BENCHMARKING  

---

## 1. Executive Summary

An investigation into the discrepancy between the older project expectation (~5,923 images across classes 2–6) and the physically observed local filesystem (5,940 TIFF files across classes 2–7) was conducted. The discrepancy was proven to be a **purely artificial omission in legacy project preprocessing**, not an upstream or filesystem corruption.

### Key Findings:
1. **Discrepancy Cause**: The older project script `context/preprocess_bcs.py` hardcoded `VALID_CLASSES = ['2', '3', '4', '5', '6']`, deliberately omitting class folder `'7'` to fit a 5-class classification setup (`DRYAD_LABEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}`). Classes 2 through 6 contain exactly **5,923 images** (`546 + 536 + 2212 + 1962 + 667 = 5,923`). Class folder `'7'` contains exactly **17 images**. `5,923 + 17 = 5,940`.
2. **Authenticity of Class 7**: Class 7 is **100% genuine original source data** from the Winkler & Boucheron repository. The dataset scores Criollo beef cattle on the standard **9-point Wagner Beef BCS scale** (which explains the 9 folder directories `1` through `9`). In the NMSU experimental herd, cows scored from 2 to 7. Folders 1, 8, and 9 were provided in the archive but remained empty because no animal was emaciated (BCS 1) or obese (BCS 8–9).
3. **Identity of Cow_52**: The 17 images in Class 7 belong to `Cow_52`. Crucially, `Cow_52` is also present in Class 6 (`Cow_52_29`, 5 images), representing a biological animal recorded at BCS 6 in one session and at BCS 7 in another session.
4. **Biological Subject Census (54 Cows vs 148 Folders)**: The 148 subdirectories across classes 2–7 represent multiple recording sessions of **54 unique biological cows** (`Cow_1` through `Cow_55`, with `Cow_39` absent). Treating folders as independent cows (the legacy approach) induces cross-session biological identity leakage.
5. **Data Quality**: 100% of all 5,940 images are valid, uncorrupted, 224x224 3-channel RGB TIFFs (DGE: Depth, Grayscale, Edge). Zero cross-cow or cross-class duplicate leakage exists.

---

## 2. Forensic Class & Image Distribution

| Class Folder | BCS Score (1–9 Beef Scale) | Unique Cow Subdirs | Image Count | Percent | Legacy Status | Audit Verdict |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `1` | 1.0 (Emaciated) | 0 | 0 | 0.00% | Empty | Normal (no emaciated cattle in herd) |
| `2` | 2.0 (Extremely Thin) | 5 | 546 | 9.19% | Included in legacy (546) | **Valid Source Data** |
| `3` | 3.0 (Thin) | 18 | 536 | 9.02% | Included in legacy (536) | **Valid Source Data** |
| `4` | 4.0 (Borderline) | 48 | 2,212 | 37.24% | Included in legacy (2,212) | **Valid Source Data** |
| `5` | 5.0 (Moderate) | 52 | 1,962 | 33.03% | Included in legacy (1,962) | **Valid Source Data** |
| `6` | 6.0 (Good) | 24 | 667 | 11.23% | Included in legacy (667) | **Valid Source Data** |
| `7` | 7.0 (Very Good / Fleshy) | 1 | 17 | 0.29% | **Omitted in legacy** | **Valid Source Data (Cow_52)** |
| `8` | 8.0 (Fat) | 0 | 0 | 0.00% | Empty | Normal (no obese cattle in herd) |
| `9` | 9.0 (Extremely Obese) | 0 | 0 | 0.00% | Empty | Normal (no extremely obese cattle) |
| **Total** | **Active Classes 2–7** | **148** | **5,940** | **100.0%** | Legacy total was 5,923 | **Complete & Verified** |

---

## 3. Discrepancy Deconstruction: 5,923 vs 5,940

In `context/preprocess_bcs.py`, lines 11–14:
```python
VALID_CLASSES = ['2', '3', '4', '5', '6']  # Class 7 was excluded!
```
And in `context/context2_bcs.txt`, line 84:
```python
DRYAD_LABEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
```
Because the prior developer designed a 5-class head matching classes 2 through 6, they silently dropped folder `7`. The physical archive `Total_sorted_DGE_images.zip` extracted on disk contains all 5,940 images.

**Mathematical Reconciliation**:
- Classes 2–6 image sum = 546 + 536 + 2,212 + 1,962 + 667 = **5,923 images**.
- Class 7 image count = **17 images**.
- Complete dataset on disk = 5,923 + 17 = **5,940 images**.

---

## 4. Biological Identity: 54 True Cows vs 148 Session Folders

The folder names inside class directories follow patterns such as `Cow_1`, `Cow_1_27`, `Cow_1_6006`, `Cow_11_2`, `Cow_11_0018`. Parsing these tokens reveals:
- **54 unique biological cattle** (`Cow_1` to `Cow_55`, with `Cow_39` absent).
- **47 cattle** appear across multiple recording sessions over time.
- **Example**: `Cow_11` appears in 5 sessions across 3 different BCS scores:
  - `2/Cow_11` (BCS 2) and `2/Cow_11_2` (BCS 2)
  - `4/Cow_11_0018` (BCS 4) and `4/Cow_11_0075` (BCS 4)
  - `6/Cow_11_07` (BCS 6)

> [!IMPORTANT]
> **Grouping Rule for Dryad Benchmarking**:
> When using Dryad for external evaluation or cross-dataset transfer, partitioning MUST be grouped by base animal ID (`Cow_X`), not by folder name. Partitioning by folder name leaks the same animal across train and test.

---

## 5. Image Integrity & Duplication Checks

1. **Readability**: All 5,940 images opened successfully with PIL. All files are 224x224, 3-channel RGB-encoded TIFF files. 0 corrupt or unreadable files.
2. **Exact Content Duplicates**: 348 duplicate SHA256 hash groups were found. All 348 groups consist of adjacent video frames within the exact same recording pass (e.g. frames 116 and 117 of `Cow_11_2`).
3. **Cross-Cow Duplicate Leakage**: **0 duplicate hashes cross between different cows.**
4. **Cross-Class Duplicate Leakage**: **0 duplicate hashes cross between different classes.**

---

## 6. Artifact Registry

- Master Manifest: `datasets/bcs/dryad/manifest.csv` (5,940 rows, 11 fields)
- Biological Cow Census: `datasets/bcs/dryad/cow_audit.csv` (54 cows, 14 fields)
- Updated Index: `datasets/bcs/bcs_index.csv` (5,940 rows, replaces empty 31-byte file)
- Discrepancy Audit Report: `datasets/bcs/dryad/audit_report.md`
- Generator Script: `scripts/build_dryad_manifest.py`

**Conclusion**: The Dryad BCS dataset is 100% verified at 5,940 images across classes 2–7. Class 7 is authentic. The dataset is clean, indexed, and ready as a secondary external validation benchmark.
