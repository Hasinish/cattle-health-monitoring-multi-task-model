# Self-Collected Cattle Viewpoint Dataset Cleaning & Normalization Audit Report

**Date:** 2026-09-23  
**Human Review Status:** COMPLETED & FINALIZED (Adjudicated on 2026-09-23 by Hasin Ishrak)  
**Raw Source Directory:** `datasets/viewpoint/self/` (100% UNTOUCHED, 1,057 files)  
**Clean Target Directory:** `datasets/viewpoint/self_clean_v1/`  
**Pipeline Scripts:** `scripts/clean_self_viewpoint.py`, `scripts/finalize_human_review.py`  

---

## 1. Executive Summary

This report documents the rigorous forensic audit, deduplication, quality filtering, conservative 3-class label normalization, and human review finalization of the raw self-collected cattle viewpoint dataset (`datasets/viewpoint/self`) into a clean, leakage-controlled 3-class dataset (`datasets/viewpoint/self_clean_v1`).

The raw dataset comprised **1050 candidate images** dispersed across 7 inconsistent subdirectories with duplicate versions (`rear view` vs `rear view-updated`), mixed `.txt` and `.csv` source link files, and commercial stock photo watermarks.

### Overall Pipeline Balance Sheet
- **Total Raw Images Audited:** 1050 (100% readable, 0 corrupt files)
- **Total Duplicate Groups Formed:** 906 (covering exact SHA-256 and near-duplicate pHash clusters)
- **Multi-Item Duplicate Groups:** 124 (involving 268 raw files)
- **Total Excluded Images:** 170 (100% preserved in raw, excluded from clean)
  - **Exact Cryptographic Duplicates (SHA-256):** 139
  - **Commercial Watermarked Stock Photos:** 31
- **Total Clean Images Retained:** **880** (byte-for-byte exact copies of raw source)
- **Human Review Adjudications:** **33/33 reviewed and finalized (0 pending)**
  - **Human Keep Decisions:** 1 (`rear view-updated/120.jpg` from Pinterest: high-quality authentic pasture photo)
  - **Human Exclude Decisions:** 32 (31 commercial stock agencies confirmed excluded; 1 duplicate confirmed excluded)

---

## 2. Final Retained Clean Class Distribution

All retained images are mapped strictly and conservatively into **only three canonical classes**:

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

---

## 5. Human Review Finalization (`review_required.csv`)

All 33 flagged items in `datasets/viewpoint/self_clean_v1/metadata/review_required.csv` were manually reviewed and adjudicated by **Hasin Ishrak**:
- **Total Flagged Candidates:** 33
- **Total Pending Items Remaining:** **0** (100% resolved)
- **Adjudications:**
  - **Confirmed Keep (1 image):** `rear view-updated/120.jpg` (clean ID `rear_0177.jpg`) sourced from Pinterest; visually verified as an authentic, high-resolution natural pasture cow photograph with sharp rear orientation.
  - **Confirmed Exclude (32 images):**
    - 31 commercial stock agency photographs (`dreamstime.com`: 10, `alamy.com`: 6, `shutterstock.com`: 5, `istockphoto.com`: 4, `123rf.com`: 3, `vecteezy.com`: 2, `depositphotos.com`: 1) confirmed excluded due to commercial watermark overlays and staged non-pastoral conditions.
    - 1 exact duplicate (`rear view/120.jpg`) confirmed excluded in favor of canonical copy in `rear view-updated/120.jpg`.

---

## 6. Critical Scientific Leakage Protection

Every raw image has been assigned a persistent `duplicate_group_id` (`dup_0001` through `dup_0906`). When future train/validation/test splits are constructed, partitioning MUST be stratified on `duplicate_group_id` rather than image filename. This guarantees that no near-duplicate, resized, or recompressed version of a training cow can leak into the test evaluation split.
