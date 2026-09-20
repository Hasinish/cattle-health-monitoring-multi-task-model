# Research Log: External Validation Benchmarks Hydration & Verification Audit

**Date:** 2026-09-20  
**Author:** Hasin Ishrak  
**Focus:** SideViewCows2026, BECA (BECA-D / BECA-L), and CBVD-5 Dataset Acquisition & Local Verification  
**Phase 3 Status:** STEP 1 (Data Registry & Clean Splits) — External Validation Benchmarks Hydrated  

---

## 1. Executive Summary
Successfully acquired, verified, and extracted all three external validation benchmarks required by the Phase 3 Canonical Roadmap: **SideViewCows2026** (80,260 images + 80,260 binary segmentation masks across 110 cows), **BECA** (29,061 dorsal images across 5,661 cows in BECA-D and 103 longitudinally tracked cows in BECA-L), and **CBVD-5** (887 videos, 206,100 frames, 5,322 annotated label frames across 107 cows). Built high-speed multi-threaded resumable downloaders (`scripts/fast_download_sideviewcows.py`, `scripts/fast_download_beca.py`) with thread-safe carriage-return (`\r`) progress bars, and an automated Kaggle organizer (`scripts/download_cbvd5.py`). Updated `datasets/dataset_registry.csv` to mark all three external benchmarks as `AVAILABLE` on disk.

---

## 2. Context & Motivation
In Phase 3 Step 1 of our thesis roadmap (`phase3_canonical_roadmap.md`), in addition to establishing leak-free protocols for primary in-domain datasets (ScienceDB for BCS, MmCows for Behavior, and MultiCamCows2024 / OpenCows2020 for Re-ID), rigorous external validation benchmarks are mandatory to defend against reviewer skepticism regarding domain shift and shortcut learning:
1. **Re-ID External Validation:** SideViewCows2026 (multi-view side/parlor/barn geometry with explicit ground-truth segmentation masks).
2. **Re-ID Scale & Long-Term Stress:** BECA-D (extreme identity diversity across 5,661 cattle) and BECA-L (continuous 5-month appearance tracking across 103 cattle).
3. **Behavior External Validation:** CBVD-5 (larger-herd multi-angle CCTV video benchmark evaluating cross-domain behavior generalization).

Previously, these datasets were missing locally (`NOT_FOUND` in registry). This session automated their retrieval, verified their archive integrity, and extracted all files onto local storage.

---

## 3. Forensic Findings & Verification Data

### A. SideViewCows2026 (Zenodo Record 21605650)
- **Total Download Size:** 23.33 GB across 7 files.
- **Local Directory:** `datasets/id/external/sideviewcows2026/`
- **Subsets & Composition:**
  - `snapshots/`: 607 RGB images + 607 binary PNG masks across 63 cows (field/cubicle snapshots, variable camera angles).
  - `parlor/`: 54,393 RGB images + 54,393 binary PNG masks across 110 cows (milking parlor fixed camera gallery, 108,786 files total).
  - `barn/`: 25,260 RGB images + 25,260 binary PNG masks across 69 cows (handheld video recorded inside barn).
  - `manifest.csv` (12.00 MB), `SHA256SUMS` (14.68 MB), `README.md`, `preview.jpg`.
- **Integrity:** `snapshots.zip` (375,219,777 B), `parlor.zip` (9,598,064,961 B), and `barn.zip` (15,051,008,416 B) matched exact remote byte counts and extracted with 0 errors.

### B. BECA Dataset (Figshare Article 32070171 / File 63928608)
- **Total Download Size:** 18.91 GB (`BECA.zip`, 20,309,712,996 bytes).
- **Local Directory:** `datasets/id/external/beca/`
- **Subsets & Composition:**
  - `BECA-D/`: 16,889 images across 5,661 beef cattle (diversity benchmark testing representation scale).
  - `BECA-L/`: 12,172 images across 103 beef cattle tracked over 5 continuous months (long-term appearance benchmark testing robustness to seasonal coat and weight changes).
- **Integrity:** Downloaded via direct S3 pre-signed Range requests, assembled, and verified with 68,350 files extracted.

### C. CBVD-5 Dataset (Kaggle: `fandaoerji/cbvd-5cow-behavior-video-dataset`)
- **Total Download Size:** 10.84 GB (`11.archive`, 11,102.76 MB).
- **Local Directory:** `datasets/behavior/external/cbvd5/`
- **Subsets & Composition:**
  - `videos/`: 687 MP4 behavior video segments.
  - `videos_add/`: 200 supplementary MP4 behavior video segments.
  - `rawframes_mini/`: 205,800 extracted mini video frames.
  - `labelframes/`: 4,122 annotated label frames.
  - `labelframes_add/`: 1,200 supplementary annotated label frames.
  - `minilabelframes/`: 4,122 mini label frames.
  - `annotations/` & `miniannotations/`: 11 annotation files.
  - `CBVD-5.csv`: Master metadata index (2.35 MB).
- **Integrity:** Synced and verified via `kagglehub` API with 0 file loss.

---

## 4. Architectural Decisions & Action Plan
1. **External Benchmarking Independence:**
   - These three datasets remain strictly **EXTERNAL VALIDATION** sets.
   - Zero samples from SideViewCows2026, BECA, or CBVD-5 will be mixed into the primary training splits.
   - They will serve as frozen, out-of-domain evaluation targets in Phase 3 Step 10 to evaluate domain transfer and evaluate whether cattle-specific visual priors (localization, segmentation, anatomy) outperform baseline RGB models under severe domain shift.
2. **Registry Synchronization:**
   - Updated `datasets/dataset_registry.csv` via `scripts/build_dataset_registry.py` to transition `SideViewCows2026`, `BECA-D`, `BECA-L`, and `CBVD-5` from `PLANNED` / `NOT_FOUND` to `DOWNLOADED` / `AVAILABLE`.

---

## 5. Artifacts & File Registry
| Artifact Path | Description |
| :--- | :--- |
| `scripts/fast_download_sideviewcows.py` | Multi-threaded resumable HTTP Range downloader for SideViewCows2026 (Zenodo 21605650). |
| `scripts/fast_download_beca.py` | Multi-threaded resumable HTTP Range downloader for BECA (Figshare 63928608). |
| `scripts/download_cbvd5.py` | Automated downloader and directory organizer for CBVD-5 from Kaggle via `kagglehub`. |
| `datasets/id/external/sideviewcows2026/` | Extracted SideViewCows2026 images, masks, and manifests (80,260 samples). |
| `datasets/id/external/beca/` | Extracted BECA-D (16,889 imgs) and BECA-L (12,172 imgs) directories. |
| `datasets/behavior/external/cbvd5/` | Extracted CBVD-5 video segments, label frames, and master CSV (887 videos, 206,100 frames). |
| `datasets/dataset_registry.csv` | Canonical dataset registry updated with verified availability for all external benchmarks. |

---

## 6. Next Steps
1. Synchronize `memory/state.md` and `memory/history.md`.
2. Push memory changes to `D:\custom-antigravity`.
3. Proceed to **Phase 3 Step 2: Cattle-Perception Feasibility Audit** (`docs/audits/phase3_perception_feasibility.md`) to evaluate YOLOv8 cattle localization and segmentation backbones.
