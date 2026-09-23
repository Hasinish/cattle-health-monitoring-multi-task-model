# RT-DETR-L Cattle Viewpoint Crop Report

**Date:** 2026-09-23 21:53:31  
**Mode:** FULL RUN  
**Source Dataset:** `datasets/viewpoint/self_clean_v1/`  
**Derived Target:** `datasets/viewpoint/self_clean_v1_rtdetr_crop/`  
**Localization Model:** Pretrained RT-DETR-L (`rtdetr-l.pt`, COCO Class 19 Cattle)  
**Confidence Threshold:** 0.25  
**Margin Rule:** Proportional 5% on all bounding box boundaries  
**Provenance File:** `datasets/viewpoint/self_clean_v1/metadata/manifest.csv`

---

## 1. Executive Summary & Size Reduction

| Metric | Source (`self_clean_v1`) | Derived (`self_clean_v1_rtdetr_crop`) | Delta / Reduction |
| :--- | :---: | :---: | :---: |
| **Total Images** | 880 | 879 successful crops | 1 failed detections |
| **Total Size (MB)** | 3513.36 MB | 1867.40 MB | **-46.8% reduction** |
| **Total Size (GB)** | 3.431 GB | 1.824 GB | **-46.8% reduction** |

---

## 2. Crop Counts by Viewpoint Class

| Class | Source Included | Successful Crops | Failed Detections | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **`front`** | 392 | 392 | 0 | 100.0% |
| **`side`** | 222 | 222 | 0 | 100.0% |
| **`rear`** | 266 | 265 | 1 | 99.6% |
| **TOTAL** | **880** | **879** | **1** | **99.9%** |

---

## 3. Localization Diagnostics

- **Multi-Cow Images:** 448 (50.9% of samples contained >1 cattle detection; primary cow selected by maximum bbox area).
- **Mean Detector Confidence:** 0.9526
- **Median Detector Confidence:** 0.9641
- **Detection Failures:** 1 (recorded in `metadata/detection_failures.csv`; excluded from training dataset).

---

## 4. Preprocessing & Crop Rule Specification

1. **Detection:** Pretrained RT-DETR-L (`rtdetr-l.pt`) applied to full RGB images with confidence threshold >= 0.25 filtering COCO class 19 (`cow`).
2. **Target Selection:** For images with multiple cattle detections, the candidate bounding box maximizing physical area (x2 - x1) * (y2 - y1) is designated as the primary animal.
3. **Margin Expansion:** Bounding box coordinates [x1, y1, x2, y2] are expanded by a proportional margin of 5%:
   - pad_x = 0.05 * (x2 - x1)
   - pad_y = 0.05 * (y2 - y1)
   - cx1 = max(0, round(x1 - pad_x))
   - cy1 = max(0, round(y1 - pad_y))
   - cx2 = min(W, round(x2 + pad_x))
   - cy2 = min(H, round(y2 + pad_y))
4. **Encoding:** Cropped BGR arrays are encoded to JPEG format at Quality 95, preventing high-frequency compression artifacts while dramatically reducing storage footprint.
5. **Leakage Protection:** `duplicate_group_id` from the source manifest is strictly preserved in `crop_manifest.csv` for downstream split partition design.

---

## 5. Artifact Registry

- `metadata/crop_manifest.csv`: Master crop manifest with bounding box and dimensional provenance.
- `metadata/detection_failures.csv`: Complete record of images with zero RT-DETR cattle detections.
- `metadata/crop_report.md`: This comprehensive audit report.
