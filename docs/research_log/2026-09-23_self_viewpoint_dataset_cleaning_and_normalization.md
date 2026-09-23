# Self-Collected Cattle Viewpoint Dataset Cleaning & Normalization Audit

**Date:** 2026-09-23  
**Status:** COMPLETE & 100% CERTIFIED  
**Topic:** Clean, Deduplicate, and Normalize Raw Self-Collected Viewpoint Dataset (`datasets/viewpoint/self`) into a Canonical 3-Class Clean Dataset (`datasets/viewpoint/self_clean_v1`)  
**Raw Source Directory:** `datasets/viewpoint/self/` (100% UNTOUCHED, 1,057 files)  
**Clean Target Directory:** `datasets/viewpoint/self_clean_v1/`  
**Pipeline Script:** `scripts/clean_self_viewpoint.py`  

---

## 1. Executive Summary

This investigation and pipeline implementation performed an exhaustive forensic audit, cryptographic/perceptual deduplication, commercial stock filtering, and conservative 3-class label normalization on the raw self-collected cattle viewpoint dataset (`datasets/viewpoint/self`). The raw dataset comprised 1,050 candidate images across 7 inconsistent subdirectories with duplicate partitions (`rear view` vs `rear view-updated`), unstandardized URLs, and commercial stock watermarks.

The pipeline successfully created the clean dataset `datasets/viewpoint/self_clean_v1/` containing exactly 3 classes (`front`: 392, `rear`: 266, `side`: 222; total **880 clean images**) alongside complete forensic metadata (`manifest.csv`, `review_required.csv`, and `cleaning_report.md`). Exactly 170 candidate images were excluded (139 exact SHA-256 duplicates and 31 commercial stock photos). Zero images were resized, re-encoded, or upscaled; all clean images are byte-for-byte exact copies of their highest-quality canonical raw files. The raw folder remained 100% untouched (1,057 files). Every image was assigned a persistent `duplicate_group_id` (`dup_0001` to `dup_0906`) to prevent train/val/test leakage in future partition design.

---

## 2. Context & Motivation

Phase 3 viewpoint modeling requires authentic, high-quality cattle images categorized into distinct directional perspectives. The user's raw collection (`datasets/viewpoint/self`) was gathered from diverse web searches and academic repositories, but suffered from severe structural and quality challenges:
1. Inconsistent folder hierarchies (`Cow_/front/Cow Front`, `front-oblique`, `side`, `rear oblique view-updated`, `rear view-updated`, `rear view`, `Cow_/rear`).
2. Redundant downloads and overlapping folders (`rear view` contained 150 images, 88 of which were duplicated in `rear view-updated`, and 23 of which were duplicated in `Cow_/rear`).
3. Commercial stock images with watermarks and staged/studio conditions (`shutterstock.com`, `alamy.com`, `istockphoto.com`, `dreamstime.com`, etc.).
4. Lack of uniform metadata linking images to their origin URLs.

The user mandated a strict, non-destructive cleaning pipeline that extracts a pristine 3-class subset (`front`, `side`, `rear`) into `datasets/viewpoint/self_clean_v1/` without modifying the raw folder, without training any classifier, and without constructing train/val/test splits.

---

## 3. Forensic Findings & Data

### 3.1 Raw Directory Inventory (1,057 Total Files)
- **Image Files:** 1,050
- **Provenance Files:** 7 (.txt and .csv)
  - `front-oblique/urls.txt`: 150 URLs
  - `side/urls.txt`: 223 URLs
  - `rear oblique view-updated/Rear oblique view-updated - Sheet1.csv`: 150 URLs
  - `rear view/Rear view links - Sheet1.csv`: 150 URLs
  - `rear view-updated/Rear view links - updated - Rear view links - updateed.csv.csv`: 105 URLs
  - `Cow_/front/Cow Front/Cow Front.txt`: 249 URLs
  - `Cow_/rear/cow Rear.txt`: 23 URLs
- **Provenance Recovery:** 1,050 / 1,050 images (100.0%) successfully matched to authentic URLs and source domains.

### 3.2 Label Normalization Mapping
Labels were normalized conservatively:
- `Cow_/front/Cow Front` (249) -> `front`
- `front-oblique` (150) -> `front`
- `side` (223) -> `side`
- `rear oblique view-updated` (150) -> `rear`
- `rear view-updated` (105) -> `rear`
- `rear view` (150) -> `rear`
- `Cow_/rear` (23) -> `rear`

No generic `oblique` folders existed in the raw dataset. All subdirectories clearly indicated front-oblique or rear-oblique orientation.

### 3.3 Deduplication & Clustering
- Total duplicate groups formed: **906**.
- Multi-item duplicate groups: **124** (encompassing 268 raw files).
- Exact SHA-256 duplicate exclusions: **139**.
  - All 23 images in `Cow_/rear` were byte-for-byte identical to images in `rear view/`.
  - 88 images in `rear view/` were byte-for-byte identical to images in `rear view-updated/`.
- Perceptual near-duplicate cross-class conflicts: **0** (verified zero leakage across front, side, and rear classes).

### 3.4 Quality & Commercial Stock Exclusions
- Commercial stock photo exclusions: **31 images** from commercial stock agencies (`dreamstime.com`: 10, `alamy.com`: 6, `shutterstock.com`: 5, `istockphoto.com`: 4, `123rf.com`: 3, `vecteezy.com`: 2, `depositphotos.com`: 1).
- Open pasture photography platforms (`www.pexels.com`: 723 retained; `unsplash.com`: 13 retained) and academic repositories (`data.mendeley.com`: 154 retained; `www.kaggle.com`: 123 retained) were preserved.
- Corrupted/unreadable images: **0**.
- Low-resolution exclusions (<200px or <40,000 px area): All 2 detected low-resolution thumbnails were already excluded as commercial stock images (`127.jpg` and `131.webp` from Shutterstock).

---

## 4. Final Clean Dataset Deliverables

### 4.1 Class Distribution (`datasets/viewpoint/self_clean_v1/`)
| Normalized Class | Clean Image Count | Share | Raw Directory Origins |
| :--- | :---: | :---: | :--- |
| **`front`** | **392** | 44.5% | `Cow Front` (249) + `front-oblique` (150) |
| **`rear`** | **266** | 30.2% | `rear oblique view-updated` (150) + `rear view-updated` (105) + `rear view` (150) + `Cow_/rear` (23) |
| **`side`** | **222** | 25.2% | `side` (223) |
| **TOTAL** | **880** | **100.0%** | **7 raw source folders** |

### 4.2 Metadata Artifacts
1. `datasets/viewpoint/self_clean_v1/metadata/manifest.csv`:
   - 1,050 rows (880 included, 170 excluded).
   - Contains all 18 required columns: `clean_id`, `clean_path`, `normalized_class`, `original_path`, `original_label`, `source_url`, `source_domain`, `width`, `height`, `sha256`, `perceptual_hash`, `duplicate_group_id`, `duplicate_status`, `stock_flag`, `quality_status`, `inclusion_status`, `exclusion_reason`, `notes`.
   - Updated with explicit human review notes for all 33 reviewed candidates.
2. `datasets/viewpoint/self_clean_v1/metadata/review_required.csv`:
   - 33 reviewed candidates (18 columns including `review_status`, `pending_status`, `human_decision`, `adjudicated_by`, `review_date`, `adjudication_notes`).
   - Status: **100% RESOLVED / 0 PENDING ITEMS**.
3. `datasets/viewpoint/self_clean_v1/metadata/cleaning_report.md`:
   - Comprehensive audit report detailing counts, rules, leakage prevention, and finalized human review adjudications.

### 4.3 Integrity & Non-Destructive Invariance
- Raw folder verification: `len(list(raw_root.rglob('*.*'))) == 1057` PASSED. Not a single file modified or deleted.
- Bit-for-bit identity check: 880 / 880 clean images verified bit-identical to their canonical raw source files with matching SHA-256 hashes.

---

## 5. Human Review Finalization (`review_required.csv`)

All 33 flagged items in `datasets/viewpoint/self_clean_v1/metadata/review_required.csv` were manually inspected and adjudicated by **Hasin Ishrak**:
- **Total Flagged Candidates:** 33
- **Total Pending Items Remaining:** **0** (100% resolved)
- **Adjudications:**
  - **Confirmed Keep (1 image):** `rear view-updated/120.jpg` (clean ID `rear_0177.jpg`) sourced from Pinterest; visually verified as an authentic, high-resolution natural pasture cow photograph with sharp rear orientation.
  - **Confirmed Exclude (32 images):**
    - 31 commercial stock agency photographs (`dreamstime.com`: 10, `alamy.com`: 6, `shutterstock.com`: 5, `istockphoto.com`: 4, `123rf.com`: 3, `vecteezy.com`: 2, `depositphotos.com`: 1) confirmed excluded due to commercial watermark overlays and staged non-pastoral conditions.
    - 1 exact duplicate (`rear view/120.jpg`) confirmed excluded in favor of canonical copy in `rear view-updated/120.jpg`.

---

## 6. Critical Scientific Leakage Protection

Every raw and clean image has been tagged with `duplicate_group_id` (`dup_0001` through `dup_0906`). When train/val/test splits are constructed in future work, split partitioning MUST be grouped on `duplicate_group_id`. This prevents near-duplicate, cropped, or recompressed versions of the same cow image from leaking across partitions.

---

## 7. Artifacts & File Registry

| File | Description |
| :--- | :--- |
| `scripts/clean_self_viewpoint.py` | Standalone reproducible Python cleaning and deduplication pipeline script. |
| `scripts/finalize_human_review.py` | Standalone script recording human review adjudications and certifying metadata synchronization. |
| `datasets/viewpoint/self_clean_v1/front/` | 392 clean front/front-oblique cattle images (`front_0001.jpg` to `front_0392.jpg`). |
| `datasets/viewpoint/self_clean_v1/rear/` | 266 clean rear/rear-oblique cattle images (`rear_0001.jpg` to `rear_0266.jpg`). |
| `datasets/viewpoint/self_clean_v1/side/` | 222 clean side cattle images (`side_0001.jpg` to `side_0222.jpg`). |
| `datasets/viewpoint/self_clean_v1/metadata/manifest.csv` | Master 1,050-row audit manifest with 18 columns, updated with human adjudication notes. |
| `datasets/viewpoint/self_clean_v1/metadata/review_required.csv` | Finalized 33-row review manifest with 0 pending items. |
| `datasets/viewpoint/self_clean_v1/metadata/cleaning_report.md` | Formal audit and cleaning summary report documenting finalized human review. |

---

## 8. Next Steps

- As commanded by the user, **STOP after finalizing and recording the human review.**
- Do NOT create train/val/test splits.
- Do NOT train any viewpoint models.
- When viewpoint training is scheduled on the roadmap, use `duplicate_group_id` stratification.
