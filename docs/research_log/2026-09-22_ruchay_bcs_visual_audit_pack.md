# Research Log: Ruchay 2026 RGB Visual Quality Audit Pack

**Date**: 2026-09-22  
**Author**: Hasin Ishrak & Antigravity Research Agent  
**Context**: Generation of representative 100-sample high-resolution contact sheet pack and provenance index for Ruchay et al. (2026) candidate external BCS validation benchmark.

---

## 1. Executive Summary
Constructed a representative, deterministic 100-sample visual contact-sheet pack for the Ruchay et al. (2026) candidate Primary External BCS validation dataset (`10.5281/zenodo.20290988`). Using deterministic seed `2026`, exactly 10 RGB images were sampled for each of the 10 Ferguson 5-point BCS classes (2.75 to 5.00), maximizing biological cow diversity to 94 unique cows (the absolute theoretical maximum given that BCS 2.75 contains only 4 cows in the entire 25,700-sample dataset). Instead of downloading the full 77.74 GB raw archives, a high-throughput buffered HTTP Range streaming pipeline extracted only the required 100 full-resolution 1080p (1920x1080) RGB PNG files (~340 MB) from Zenodo. Generated 10 high-resolution contact sheets (2x5 grid, 640x360 px per image tile, 1320x2240 px per sheet) with legible provenance metadata banners, alongside a full manifest and comprehensive markdown index. Initial visual inspection reveals an overhead nadir camera view over milking stalls where cows stand side-by-side in parallel stanchions, exhibiting prominent dorsal anatomical landmarks (spinous processes, tuber coxae, tuber ischii).

---

## 2. Context & Motivation
Following the 1,000-image visual quality audit of ScienceDB, MmCows, and SideViewCows2026 (which revealed that ScienceDB exhibits 38.3% multi-cow frames and 12.9% body cutoff, though preserving rear pelvic features), the canonical roadmap (`phase3_canonical_roadmap.md`) warranted a dedicated visual quality audit of candidate external benchmarks. Ruchay 2026 is designated as the Primary External BCS Validation benchmark, but its 77.74 GB raw data had never been visually inspected locally. This task provides the concrete visual evidence pack so Hasin and ChatGPT vision can manually inspect whether the RGB imagery is suitable for BCS evaluation without altering roadmap roles or fabricating performance comparisons.

---

## 3. Forensic Sampling & Dataset Demographics

### Class & Session Cross-Tabulation (N=100)
| BCS Class | 06.12.2024 | 20.02.2025 | 20.03.2025 | 27.03.2025 | Total Samples | Unique Cows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.75** | 10 | 0 | 0 | 0 | **10** | **4** |
| **3.00** | 3 | 3 | 2 | 2 | **10** | **10** |
| **3.25** | 3 | 3 | 2 | 2 | **10** | **10** |
| **3.50** | 3 | 3 | 2 | 2 | **10** | **10** |
| **3.75** | 3 | 3 | 2 | 2 | **10** | **10** |
| **4.00** | 2 | 0 | 5 | 3 | **10** | **10** |
| **4.25** | 0 | 0 | 0 | 10 | **10** | **10** |
| **4.50** | 0 | 0 | 0 | 10 | **10** | **10** |
| **4.75** | 0 | 0 | 0 | 10 | **10** | **10** |
| **5.00** | 0 | 0 | 0 | 10 | **10** | **10** |
| **TOTAL** | **24** | **12** | **13** | **51** | **100** | **94** |

### Key Forensic Insights:
1. **Severe Session Imbalance in Extreme Classes**:
   - Class `2.75` exists exclusively in session `06.12.2024` across only 4 cows (80 total frames). All 4 cows were sampled with non-adjacent frames (relative passage frame indices [5, 10, 15] or [7, 13]) to prevent video burst redundancy.
   - Classes `4.25`, `4.50`, `4.75`, and `5.00` exist exclusively in session `27.03.2025`.
2. **Maximum Unique Biological Cow Diversity**:
   - 94 unique biological cow IDs across 100 samples (4 cows from class 2.75, plus 10 unique cows for each of the other 9 classes).
3. **Overhead Nadir Stanchion Setup**:
   - Unlike ScienceDB (ground-level rear-chute RGB), Ruchay 2026 imagery is acquired by a Microsoft Kinect sensor mounted at 3.05m nadir looking down at cows in parallel stalls. Flanking cows are visible in adjacent stanchions, making cow localization and bounding box segmentation critical.

---

## 4. Architectural Decisions & Action Plan
- **Zero-Waste Selective Hydration**: Utilized `CachedHTTPRangeFile` with 4 MB read-ahead caching and multi-threaded parallel extraction across Zenodo archives (`06.12.2024.zip`, `20.02.2025.zip`, `20.03.2025.zip`, `27.03.2025.1.zip`, `27.03.2025.2.zip`). Avoided downloading 77.74 GB of raw archives by downloading only ~340 MB of exact target PNGs in under 2 minutes.
- **Visual Presentation Standards**: Contact sheets rendered at 1320x2240 px (2 columns x 5 rows, 640x360 px per image tile, long side 640 px >= 500 px) with legible metadata banners (Sample ID, BCS, Cow ID, Session, Passage, Filename).
- **Strict Role Boundaries Preserved**: In strict compliance with thesis governance, Ruchay 2026 remains the candidate Primary External BCS Validation Benchmark. No roadmap changes, no model training, and no performance claims are asserted.

---

## 5. Artifacts & File Registry
- Generator Script: [`scripts/build_ruchay_bcs_visual_audit_pack.py`](file:///d:/cattle-health-monitoring-multi-task-model/scripts/build_ruchay_bcs_visual_audit_pack.py)
- Audit Manifest: [`artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv)
- High-Resolution Contact Sheets (10 files, ~9.2 MB total): [`docs/audits/assets/ruchay_bcs_visual_audit/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/ruchay_bcs_visual_audit/)
  - `ruchay_bcs_sheet_01_bcs2.75.jpg`
  - `ruchay_bcs_sheet_02_bcs3.00.jpg`
  - `ruchay_bcs_sheet_03_bcs3.25.jpg`
  - `ruchay_bcs_sheet_04_bcs3.50.jpg`
  - `ruchay_bcs_sheet_05_bcs3.75.jpg`
  - `ruchay_bcs_sheet_06_bcs4.00.jpg`
  - `ruchay_bcs_sheet_07_bcs4.25.jpg`
  - `ruchay_bcs_sheet_08_bcs4.50.jpg`
  - `ruchay_bcs_sheet_09_bcs4.75.jpg`
  - `ruchay_bcs_sheet_10_bcs5.00.jpg`
- Comprehensive Audit Index & Gallery: [`docs/audits/phase3_ruchay_bcs_visual_audit_index.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_ruchay_bcs_visual_audit_index.md)

---

## 6. Next Steps
1. Conduct manual human visual inspection of all 10 contact sheets with Hasin and ChatGPT vision to assess BCS anatomical clarity and flanking cow stall occlusion.
2. Synchronize memory files (`state.md`, `history.md`).
3. Commit audit deliverables to Git repository.
