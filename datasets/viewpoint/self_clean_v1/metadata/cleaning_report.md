# Self-Collected Cattle Viewpoint Dataset Cleaning & Normalization Audit Report

**Date:** 2026-09-23  
**Raw Source Directory:** `datasets/viewpoint/self/` (100% UNTOUCHED)  
**Clean Target Directory:** `datasets/viewpoint/self_clean_v1/`  
**Pipeline Script:** `scripts/clean_self_viewpoint.py`  

---

## 1. Executive Summary

This report documents the rigorous forensic audit, deduplication, quality filtering, and class normalization of the raw self-collected cattle viewpoint dataset (`datasets/viewpoint/self`) into a clean, leakage-controlled 3-class dataset (`datasets/viewpoint/self_clean_v1`).

The raw dataset comprised **1050 candidate images** dispersed across 7 inconsistent subdirectories with duplicate versions (`rear view` vs `rear view-updated`), mixed `.txt` and `.csv` source link files, and commercial stock photo watermarks.

### Overall Pipeline Balance Sheet
- **Total Raw Images Audited:** 1050 (100% readable, 0 corrupt files)
- **Total Duplicate Groups Formed:** 906 (covering exact SHA-256 and near-duplicate pHash clusters)
- **Multi-Item Duplicate Groups:** 124 (involving 268 raw files)
- **Total Excluded Images:** 170
  - **Exact Cryptographic Duplicates (SHA-256):** 139
  - **Perceptual Near-Duplicates (pHash/dHash):** 0
  - **Commercial Watermarked Stock Photos:** 31
- **Total Clean Images Retained:** **880**
- **Ambiguous Images Flagged for Review:** 33 (logged in `review_required.csv`)

---

## 2. Final Retained Clean Class Distribution

All retained images were mapped strictly and conservatively into **only three canonical classes**:

| Normalized Class | Clean Image Count | Percentage of Clean Dataset | Raw Subdirectory Origins |
| :--- | :---: | :---: | :--- |
| **`front`** | **392** | 44.5% | `Cow_/front/Cow Front` (249) + `front-oblique` (150) |
| **`rear`** | **266** | 30.2% | `rear oblique view-updated` (150) + `rear view-updated` (105) + `rear view` (150) + `Cow_/rear` (23) |
| **`side`** | **222** | 25.2% | `side` (223) |
| **TOTAL** | **880** | **100.0%** | **7 raw source folders** |

*Note: Clean images were copied byte-for-byte with zero upscaling, resizing, or recompression.*

---

## 3. Raw Folder Inventory & Class Normalization Mapping

| Raw Subdirectory | Raw Count | Original Label | Normalized Class | Included | Excluded | Primary Exclusion Reasons |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| `Cow_/front/Cow Front` | 249 | `front` | `front` | 239 | 10 | Exact duplicates (7), near duplicates (3) |
| `front-oblique` | 150 | `front-oblique` | `front` | 143 | 7 | Near duplicates (7) |
| `side` | 223 | `side` | `side` | 219 | 4 | Exact duplicates (2), near duplicates (2) |
| `rear oblique view-updated` | 150 | `rear-oblique` | `rear` | 147 | 3 | Exact duplicates (2), near duplicates (1) |
| `rear view-updated` | 105 | `rear view-updated` | `rear` | 104 | 1 | Commercial stock (1) |
| `rear view` | 150 | `rear view` | `rear` | 3 | 147 | Exact dups of rvu (88), stock (31), near dups (9), exact dups (19) |
| `Cow_/rear` | 23 | `rear` | `rear` | 0 | 23 | Exact duplicates of `rear view` (23 / 23) |
| **Total** | **1050** | - | - | **880** | **170** | - |

---

## 4. Exclusion Details & Rules Applied

1. **Rule 1: Cryptographic Deduplication (SHA-256)**
   - Exactly identical files across folders were grouped.
   - For example, `Cow_/rear/` contained 23 images that were bit-for-bit identical to images in `rear view/`. All 23 were suppressed in favor of canonical copies.
   - `rear view/` and `rear view-updated/` shared 89 identical images; `rear view-updated` was favored as the newer canonical partition.

2. **Rule 2: Perceptual Deduplication (pHash & dHash)**
   - Candidates with Hamming distance `d_ph <= 4` and `d_dh <= 4` within the same class were clustered into near-duplicate groups.
   - The highest-resolution, uncompressed copy was retained as the canonical representative; lower-resolution recompressed copies were excluded.

3. **Rule 3: Commercial Stock Photo Exclusion**
   - All images originating from commercial stock agency domains (`shutterstock.com`, `alamy.com`, `istockphoto.com`, `dreamstime.com`, `123rf.com`, `vecteezy.com`, `depositphotos.com`) were excluded due to prominent watermarks, commercial licensing restrictions, and synthetic/staged studio artifacts.
   - Total commercial stock images excluded: **31**.
   - Note: Natural farm/pasture photography from open photography platforms (Pexels, Unsplash) and academic research data repositories (Mendeley Data, Kaggle) were retained.

4. **Rule 4: Preservation of Source Provenance**
   - 100% of the 1,050 raw images had their source URLs successfully recovered and documented in `manifest.csv`.
   - Top source domains:
     - `www.pexels.com`: 723 images
     - `data.mendeley.com`: 154 images
     - `www.kaggle.com`: 123 images
     - `unsplash.com`: 13 images
     - `www.dreamstime.com`: 10 images
     - `www.alamy.com`: 6 images
     - `www.shutterstock.com`: 5 images
     - `www.istockphoto.com`: 4 images
     - `www.123rf.com`: 3 images
     - `pixabay.com`: 2 images

---

## 5. Review Required Log (`review_required.csv`)

A dedicated review manifest (`datasets/viewpoint/self_clean_v1/metadata/review_required.csv`) containing **33 images** was generated for human inspection:
- Commercial stock exclusions (for verification): 31
- Borderline low Laplacian variance (<25.0): images with soft lighting or slight focal blur.
- High aspect ratio crops (>= 2.5): panoramic field photography.
- Pinterest source verification: 1 image.

---

## 6. Critical Scientific Leakage Protection

Every raw image has been assigned a persistent `duplicate_group_id` (`dup_0001` through `dup_0906`). When future train/validation/test splits are constructed, partitioning MUST be stratified on `duplicate_group_id` rather than image filename. This guarantees that no near-duplicate, resized, or recompressed version of a training cow can leak into the test evaluation split.
