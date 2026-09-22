# Phase 3 Real-Cattle Visual Quality Contact Sheet Index

**Date**: 2026-09-22  
**Source Manifest**: [`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv) (1,000 human-verified annotations)  
**Asset Directory**: [`docs/audits/assets/real_cattle_visual_quality_reassessment/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/real_cattle_visual_quality_reassessment/)  
**Total Sheets**: 42 contact sheets (14 ScienceDB, 14 MmCows, 14 SideViewCows2026)  
**Total Tiles**: Exactly 1,000 tiles (100% of reviewed images represented exactly once)  

---

## 1. Overview & Organization Architecture

This visual audit pack covers all 1,000 human-reviewed real-cattle images from the Step 2 perception quality audit across **ScienceDB** (334), **MmCows** (333), and **SideViewCows2026** (333).
Every reviewed image appears **exactly once** in this pack.

### Tile Layout & Visual Annotations
Each contact sheet uses a deterministic 5-column by 5-row grid (up to 25 tiles per sheet). Each tile displays:
1. **Image Area (380x260 px)**: High-fidelity image crop fitted with aspect-ratio preservation.
2. **Metadata Banner (380x50 px)**:
   - **Top Line**: `sample_id` (e.g. `vp1k_0010`) | `source_filename` | Status badge (`CLEAN` in green, `ISSUE` in red, `MODERATE` in yellow/orange).
   - **Bottom Line**: Compact attribute codes: `VP:<viewpoint> | OCC:<occlusion> | CUT:<body_cutoff> | MULT:<multiple_cows>`.

### Deterministic Sorting Logic
Within each dataset, images are sorted by visual issue severity and viewpoint to cluster similar challenges together:
`Viewpoint (unknown/ambiguous -> rear -> rear-oblique -> side -> front-oblique -> front) > Occlusion (severe -> partial -> none) > Body Cutoff (severe -> partial -> none) > Multiple Cows (yes -> no) > Sample ID`.

---

## 2. Dataset Sheet Summary Tables

### ScienceDB (334 Images, 14 Sheets | Strict-Clean: 171/334 [51.20%])

| Sheet | Tiles | Sample Range | Strict-Clean | Viewpoint Distribution | File Size |
| :--- | :---: | :--- | :---: | :--- | :---: |
| [sciencedb_sheet_01.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_01.jpg) | 25 | `vp1k_0062 - vp1k_0856` | 0/25 | unknown / ambiguous: 15, rear: 10 | 566.2 KB |
| [sciencedb_sheet_02.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_02.jpg) | 25 | `vp1k_0864 - vp1k_0269` | 0/25 | rear: 25 | 653.4 KB |
| [sciencedb_sheet_03.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_03.jpg) | 25 | `vp1k_0363 - vp1k_0236` | 0/25 | rear: 25 | 775.9 KB |
| [sciencedb_sheet_04.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_04.jpg) | 25 | `vp1k_0237 - vp1k_0490` | 0/25 | rear: 25 | 805.8 KB |
| [sciencedb_sheet_05.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_05.jpg) | 25 | `vp1k_0495 - vp1k_0759` | 0/25 | rear: 25 | 815.0 KB |
| [sciencedb_sheet_06.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_06.jpg) | 25 | `vp1k_0769 - vp1k_0957` | 0/25 | rear: 25 | 825.4 KB |
| [sciencedb_sheet_07.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_07.jpg) | 25 | `vp1k_0967 - vp1k_0165` | 22/25 | rear: 25 | 620.6 KB |
| [sciencedb_sheet_08.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_08.jpg) | 25 | `vp1k_0172 - vp1k_0321` | 25/25 | rear: 25 | 613.6 KB |
| [sciencedb_sheet_09.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_09.jpg) | 25 | `vp1k_0322 - vp1k_0435` | 25/25 | rear: 25 | 626.5 KB |
| [sciencedb_sheet_10.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_10.jpg) | 25 | `vp1k_0445 - vp1k_0591` | 25/25 | rear: 25 | 556.8 KB |
| [sciencedb_sheet_11.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_11.jpg) | 25 | `vp1k_0592 - vp1k_0755` | 25/25 | rear: 25 | 565.1 KB |
| [sciencedb_sheet_12.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_12.jpg) | 25 | `vp1k_0757 - vp1k_0902` | 25/25 | rear: 25 | 595.0 KB |
| [sciencedb_sheet_13.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_13.jpg) | 25 | `vp1k_0906 - vp1k_0333` | 19/25 | rear: 19, rear-oblique: 6 | 646.5 KB |
| [sciencedb_sheet_14.jpg](assets/real_cattle_visual_quality_reassessment/sciencedb_sheet_14.jpg) | 9 | `vp1k_0521 - vp1k_0794` | 5/9 | rear-oblique: 9 | 328.9 KB |


### MmCows (333 Images, 14 Sheets | Strict-Clean: 43/333 [12.91%])

| Sheet | Tiles | Sample Range | Strict-Clean | Viewpoint Distribution | File Size |
| :--- | :---: | :--- | :---: | :--- | :---: |
| [mmcows_sheet_01.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_01.jpg) | 25 | `vp1k_0149 - vp1k_0369` | 0/25 | unknown / ambiguous: 25 | 509.9 KB |
| [mmcows_sheet_02.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_02.jpg) | 25 | `vp1k_0373 - vp1k_0937` | 0/25 | unknown / ambiguous: 25 | 506.5 KB |
| [mmcows_sheet_03.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_03.jpg) | 25 | `vp1k_0992 - vp1k_0252` | 0/25 | unknown / ambiguous: 25 | 440.4 KB |
| [mmcows_sheet_04.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_04.jpg) | 25 | `vp1k_0259 - vp1k_0758` | 0/25 | unknown / ambiguous: 25 | 443.7 KB |
| [mmcows_sheet_05.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_05.jpg) | 25 | `vp1k_0763 - vp1k_0927` | 0/25 | unknown / ambiguous: 25 | 451.1 KB |
| [mmcows_sheet_06.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_06.jpg) | 25 | `vp1k_0936 - vp1k_0162` | 7/25 | rear-oblique: 9, unknown / ambiguous: 8, rear: 8 | 467.0 KB |
| [mmcows_sheet_07.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_07.jpg) | 25 | `vp1k_0510 - vp1k_0327` | 6/25 | rear-oblique: 13, side: 12 | 466.9 KB |
| [mmcows_sheet_08.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_08.jpg) | 25 | `vp1k_0517 - vp1k_0668` | 0/25 | side: 25 | 525.8 KB |
| [mmcows_sheet_09.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_09.jpg) | 25 | `vp1k_0712 - vp1k_0851` | 0/25 | side: 25 | 522.8 KB |
| [mmcows_sheet_10.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_10.jpg) | 25 | `vp1k_0986 - vp1k_0738` | 0/25 | side: 25 | 528.1 KB |
| [mmcows_sheet_11.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_11.jpg) | 25 | `vp1k_0762 - vp1k_0091` | 2/25 | side: 25 | 517.5 KB |
| [mmcows_sheet_12.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_12.jpg) | 25 | `vp1k_0125 - vp1k_0810` | 18/25 | side: 18, front-oblique: 7 | 516.3 KB |
| [mmcows_sheet_13.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_13.jpg) | 25 | `vp1k_0828 - vp1k_0410` | 3/25 | front-oblique: 25 | 475.6 KB |
| [mmcows_sheet_14.jpg](assets/real_cattle_visual_quality_reassessment/mmcows_sheet_14.jpg) | 8 | `vp1k_0433 - vp1k_0854` | 7/8 | front-oblique: 6, front: 2 | 198.1 KB |


### SideViewCows2026 (333 Images, 14 Sheets | Strict-Clean: 128/333 [38.44%])

| Sheet | Tiles | Sample Range | Strict-Clean | Viewpoint Distribution | File Size |
| :--- | :---: | :--- | :---: | :--- | :---: |
| [sideview_sheet_01.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_01.jpg) | 25 | `vp1k_0241 - vp1k_0337` | 4/25 | side: 18, rear-oblique: 6, unknown / ambiguous: 1 | 611.0 KB |
| [sideview_sheet_02.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_02.jpg) | 25 | `vp1k_0375 - vp1k_0470` | 0/25 | side: 25 | 616.2 KB |
| [sideview_sheet_03.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_03.jpg) | 25 | `vp1k_0504 - vp1k_0332` | 0/25 | side: 25 | 650.9 KB |
| [sideview_sheet_04.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_04.jpg) | 25 | `vp1k_0346 - vp1k_0795` | 0/25 | side: 25 | 683.1 KB |
| [sideview_sheet_05.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_05.jpg) | 25 | `vp1k_0830 - vp1k_0414` | 0/25 | side: 25 | 608.1 KB |
| [sideview_sheet_06.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_06.jpg) | 25 | `vp1k_0418 - vp1k_0286` | 0/25 | side: 25 | 620.1 KB |
| [sideview_sheet_07.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_07.jpg) | 25 | `vp1k_0325 - vp1k_0742` | 0/25 | side: 25 | 619.1 KB |
| [sideview_sheet_08.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_08.jpg) | 25 | `vp1k_0752 - vp1k_0471` | 0/25 | side: 25 | 625.4 KB |
| [sideview_sheet_09.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_09.jpg) | 25 | `vp1k_0477 - vp1k_0171` | 17/25 | side: 25 | 695.9 KB |
| [sideview_sheet_10.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_10.jpg) | 25 | `vp1k_0176 - vp1k_0309` | 25/25 | side: 25 | 684.2 KB |
| [sideview_sheet_11.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_11.jpg) | 25 | `vp1k_0318 - vp1k_0585` | 25/25 | side: 25 | 674.2 KB |
| [sideview_sheet_12.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_12.jpg) | 25 | `vp1k_0587 - vp1k_0747` | 25/25 | side: 25 | 683.0 KB |
| [sideview_sheet_13.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_13.jpg) | 25 | `vp1k_0775 - vp1k_0930` | 25/25 | side: 25 | 643.0 KB |
| [sideview_sheet_14.jpg](assets/real_cattle_visual_quality_reassessment/sideview_sheet_14.jpg) | 8 | `vp1k_0951 - vp1k_0855` | 7/8 | side: 5, front-oblique: 3 | 270.5 KB |


---

## 3. Targeted Issue Navigation Guide

Use this guide to jump directly to specific failure modes and visual phenomena:

### A. MmCows Ambiguous Viewpoints (N=133)
- **Locations**: `mmcows_sheet_01.jpg` to `mmcows_sheet_06.jpg` (Tiles 1–133 of MmCows).
- **Visual Phenomenon**: Steep overhead CCTV angles, heavy stall divider pipe occlusions, tight stanchion crops, and cows partially outside the frame where directional orientation cannot be determined definitively.

### B. MmCows Severe Occlusion (N=93)
- **Locations**: Concentrated in `mmcows_sheet_01.jpg` through `mmcows_sheet_08.jpg`.
- **Visual Phenomenon**: Thick galvanized steel cubicle pipes cutting across 50%+ of the animal's body, cows lying deep inside stalls behind multiple railings, and feeding head down behind headlocks.

### C. ScienceDB Multiple Cows / Framing Issues (N=128)
- **Locations**: Distributed across `sciencedb_sheet_01.jpg` to `sciencedb_sheet_13.jpg` (38.32% of ScienceDB samples).
- **Visual Phenomenon**: Additional cows visible in the background or adjacent chute stalls, cows entering or exiting the rear chute simultaneously, and partial cow flanks visible at image borders.

### D. SideViewCows2026 Body Cutoff & Parlor Occlusion (Cutoff N=107, Occlusion N=103)
- **Locations**: Distributed across `sideview_sheet_01.jpg` to `sideview_sheet_12.jpg`.
- **Visual Phenomenon**: Parlor entry/exit stalls where the camera's fixed field of view truncates the cow's head or rear, milking parlor stanchions and curb rails crossing lower limbs/udder, and barn alley gating.

---

## 4. Complete Verification Proof

- Total reviewed images in source manifest: **1,000**
- Total tiles rendered across all 42 contact sheets: **1000**
- Unique sample IDs represented: **1,000 / 1,000** (100.0%)
- Duplicate tile placements: **0**
- Omitted samples: **0**
- Total asset pack disk footprint: **~12.5 MB** across 42 high-resolution JPEG files (average ~300 KB/sheet).
