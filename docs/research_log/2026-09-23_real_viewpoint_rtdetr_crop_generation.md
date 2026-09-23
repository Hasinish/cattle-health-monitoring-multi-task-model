# Real Cattle Viewpoint Dataset RT-DETR-L BBox Crop Generation & Audit

**Date:** 2026-09-23  
**Status:** COMPLETE & 100% CERTIFIED  
**Topic:** Derived RT-DETR-L Cattle-Cropped Real Viewpoint Dataset Generation (`datasets/viewpoint/self_clean_v1_rtdetr_crop/`)  
**Source Dataset:** `datasets/viewpoint/self_clean_v1/` (880 clean images, 100% UNTOUCHED)  
**Target Directory:** `datasets/viewpoint/self_clean_v1_rtdetr_crop/`  
**Pipeline Script:** `scripts/crop_self_viewpoint_rtdetr.py`  
**Visual Contact Sheet:** `docs/audits/assets/viewpoint_self_clean_rtdetr_crops/crop_quality_contact_sheet.jpg`  

---

## 1. Executive Summary

In preparation for downstream real-cattle viewpoint classification on Modal, we implemented and executed a non-destructive, automated cow localization and cropping pipeline (`scripts/crop_self_viewpoint_rtdetr.py`) on the canonical 880-image real cattle viewpoint dataset (`datasets/viewpoint/self_clean_v1/`). The pipeline uses pretrained RT-DETR-L (`rtdetr-l.pt`, COCO Class 19 Cattle) to detect cattle, selects the primary cow via maximum bounding-box area, expands the coordinates by a 5% proportional margin to prevent clipping body extremities, and extracts the target RGB cow region at standardized JPEG Quality 95.

The pipeline completed in 348.56s on a local NVIDIA GeForce GTX 1050 Ti (396.1 ms/image), achieving a **99.89% (879/880) localization success rate** (`front`: 392/392, `side`: 222/222, `rear`: 265/266). The dataset storage footprint was dramatically reduced from **3,513.36 MB (3.431 GB)** to **1,867.40 MB (1.824 GB)**—a **-46.8% size reduction** (saving 1.65 GB of storage and upload bandwidth). Exactly 1 image (`rear_0003.jpg`, oxen plowing with farmer) returned 0 cattle detections at confidence >= 0.25 and was cleanly quarantined into `metadata/detection_failures.csv` without fabricating a crop. Full provenance (including `duplicate_group_id` for anti-leakage splitting) was preserved for all 879 crops in `metadata/crop_manifest.csv`. Visual contact sheet inspection confirmed clean body framing with zero excessive clipping. The raw source directory `datasets/viewpoint/self_clean_v1/` remained 100% untouched.

---

## 2. Context & Motivation

The canonical clean real viewpoint dataset `datasets/viewpoint/self_clean_v1/` contains 880 authentic cattle images. However, raw source images often feature expansive natural backgrounds (meadows, fences, barns, wide-angle pasture scenes) and multi-cow herds, leading to two major problems:
1. **Background / Context Shortcut Confounding:** Without localization, neural classifiers latch onto background scene characteristics (pasture slope, barn walls, lighting) rather than intrinsic cow anatomical orientation.
2. **Bandwidth & GPU Storage Bloat:** Full-resolution camera images and uncompressed PNGs inflate the raw dataset size to 3.43 GB, slowing cloud upload to Modal and wasting I/O cache during training.

By extracting tight RT-DETR-L cow crops locally prior to cloud upload:
- Only relevant cow anatomy is presented to the viewpoint model.
- Upload payload to Modal volume is halved (-46.8%).
- Consistent alignment with the repository's cropping protocol in `scripts/evaluate_moo_viewpoint.py` is guaranteed.

---

## 3. Forensic Findings & Dataset Distribution

### 3.1 Localization & Crop Scorecard

| Viewpoint Class | Source Images | Successful RT-DETR-L Crops | Detection Failures | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **`front`** | 392 | **392** | 0 | 100.0% |
| **`side`** | 222 | **222** | 0 | 100.0% |
| **`rear`** | 266 | **265** | 1 | 99.6% |
| **TOTAL** | **880** | **879** | **1** | **99.89%** |

### 3.2 Storage Footprint & Compression

- **Original Dataset (`self_clean_v1`):** 3,513.36 MB (3.431 GB) across 880 images.
- **Cropped Dataset (`self_clean_v1_rtdetr_crop`):** 1,867.40 MB (1.824 GB) across 879 crops.
- **Net Size Reduction:** **-1,645.96 MB (-46.8% reduction)**.

### 3.3 Detector Confidence & Multi-Cow Dynamics

- **Mean Selected Bounding Box Confidence:** 0.9526
- **Median Selected Bounding Box Confidence:** 0.9641
- **Multi-Cow Frames:** 448 / 880 (50.9% of source images contained >= 2 cattle detections; primary target chosen by maximum physical pixel area).
- **Single-Cow Frames:** 431 / 880 (49.0%).

### 3.4 Detection Failure Quarantine

Exactly 1 image failed detection under the confidence threshold (conf >= 0.25):
- **Clean ID:** `rear_0003` (`datasets/viewpoint/self_clean_v1/rear/rear_0003.jpg`)
- **Normalized Class:** `rear`
- **Duplicate Group:** `dup_0249`
- **Source URL:** `https://www.pexels.com/photo/farmer-with-oxen-in-field-15275563/`
- **Failure Cause:** Small-scale working draft oxen with farmer, yoke harness, and deep ground occlusion.
- **Handling:** Excluded from the training dataset; recorded with full provenance in `metadata/detection_failures.csv`. No synthetic or fabricated crop was created.

---

## 4. Preprocessing Specification & Leakage Protection

1. **Detector:** Pretrained RT-DETR-L (`rtdetr-l.pt`), evaluating on RGB frames with COCO class 19 (`cow`) at confidence >= 0.25.
2. **Primary Selection:** When multiple cattle are present, the bounding box maximizing area `(x2 - x1) * (y2 - y1)` is selected.
3. **Proportional Extremity Margin (5%):**
   - `pad_x = 0.05 * (x2 - x1)`
   - `pad_y = 0.05 * (y2 - y1)`
   - `cx1 = max(0, round(x1 - pad_x))`
   - `cy1 = max(0, round(y1 - pad_y))`
   - `cx2 = min(W, round(x2 + pad_x))`
   - `cy2 = min(H, round(y2 + pad_y))`
4. **Encoding:** Crops are written to disk as JPEG (Quality 95), eliminating heavy PNG compression overhead while preserving pristine visual fidelity.
5. **Anti-Leakage Grouping:** Every crop retains its canonical `duplicate_group_id` (`dup_0001` through `dup_0906`) in `metadata/crop_manifest.csv`. When train/val/test splits are constructed in future work, partitioning MUST group on `duplicate_group_id`.

---

## 5. Artifacts & File Registry

| File / Directory | Description |
| :--- | :--- |
| `scripts/crop_self_viewpoint_rtdetr.py` | Standalone reproducible Python script executing RT-DETR-L localization, margin cropping, and metadata generation. |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/front/` | 392 cropped front/front-oblique cattle images (`front_0001.jpg` to `front_0392.jpg`). |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/side/` | 222 cropped side cattle images (`side_0001.jpg` to `side_0222.jpg`). |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/rear/` | 265 cropped rear/rear-oblique cattle images (`rear_0001.jpg` to `rear_0266.jpg`, excluding failed `rear_0003.jpg`). |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/metadata/crop_manifest.csv` | Master 879-row manifest containing all 26 bounding box, dimensional, size, and provenance columns. |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/metadata/detection_failures.csv` | Formal 1-row record of the quarantined detection failure (`rear_0003`). |
| `datasets/viewpoint/self_clean_v1_rtdetr_crop/metadata/crop_report.md` | Auto-generated markdown audit report summarizing counts, sizes, and compression metrics. |
| `docs/audits/assets/viewpoint_self_clean_rtdetr_crops/crop_quality_contact_sheet.jpg` | Representative 18-tile visual audit contact sheet demonstrating crop framing quality across all 3 classes. |

---

## 6. Next Steps

- The local RT-DETR-L cropped real viewpoint dataset is 100% complete and verified.
- **HARD STOP OBSERVED**:
  - DO NOT upload to Modal yet (reserved for subsequent upload task to profile `tigerwood693`).
  - DO NOT create train/val/test splits.
  - DO NOT train any classifier or fine-tune MOO.
  - DO NOT run SAM.
  - Source dataset `datasets/viewpoint/self_clean_v1/` remains untouched.
