# Research Log: Real-Cattle Visual Quality Reassessment (1,000-Image Audit)

**Date**: 2026-09-22  
**Phase**: Phase 3 (Step 2 — Cattle Perception Feasibility & Data Quality Audit)  
**Author / Operator**: Hasin Ishrak (in pair programming with Antigravity)  
**Status**: COMPLETED / EVIDENCE PRESERVED (NO DATASET ROLE OR ROADMAP CHANGES)  

---

## 1. Executive Summary

We conducted a comprehensive, deterministic evidence-preservation audit and visual contact-sheet inspection for the newly completed **1,000-image human-verified real-cattle review** across **ScienceDB** (334), **MmCows** (333), and **SideViewCows2026** (333). Recomputing all statistics directly from the source of truth manifest (`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`) confirms that across the 1,000 samples, only **34.20% (342/1000)** meet the strict-clean criterion (no occlusion, no body cutoff, single cow, unambiguous viewpoint). 

MmCows exhibits severe photographic challenges, with **39.94% (133/333) unknown/ambiguous viewpoints**, **54.35% (181/333) combined occlusion (93 severe, 88 partial)**, and only **12.91% (43/333) strict-clean**. ScienceDB demonstrates high rear-view orientation consistency (91.02% rear, 4.49% rear-oblique) and near-zero occlusion (99.70% none), but suffers from **38.32% (128/334) multiple-cow presence** and **12.87% (43/334) body cutoff**. SideViewCows2026 is 97.00% side-oriented, but experiences **32.13% body cutoff** and **30.93% occlusion** from milking parlor framework, achieving **38.44% (128/333) strict-clean**.

We generated a 42-sheet visual contact sheet pack covering 100% of the 1,000 images exactly once. Under strict scientific decision rules: **no dataset roles are changed**, **the canonical roadmap is not modified**, and **no alternative datasets are promoted without equivalent empirical evidence**. A task-specific visual BCS quality audit of Ruchay et al. 2026 RGB-D BCS is identified as the next warranted investigation.

---

## 2. Context & Motivation

During Step 2 (Cattle Perception Feasibility), preliminary visual inspections had evaluated small subsets (e.g. 20 MmCows samples vs CBVD-5 in `docs/audits/phase3_behavior_agent_visual_inspection.md`, 48 verification samples in `docs/audits/phase3_manual_dataset_visual_verification.md`, and 100 viewpoint samples in `docs/research_log/2026-09-20_cattle_viewpoint_expanded_crosscheck.md`).

To rigorously evaluate the visual data quality of the primary Phase 3 datasets before entering Step 3 caching and Step 4 baseline training, a large-scale 1,000-image human-verified annotation review was executed across the three primary datasets. This investigation was conducted to:
1. Recompute and permanently preserve all quantitative data directly from the manifest.
2. Formally verify or refute specific numbers discussed in research discussions (MmCows 133 ambiguous, 88 partial, 93 severe occlusion, 43 strict-clean; ScienceDB 128 multi-cow).
3. Provide a full contact-sheet pack allowing human auditors to inspect the complete 1,000-image dataset in an organized, scalable format.
4. Establish clear scientific boundaries between what the annotation schema measures (viewpoint, occlusion, cutoff, multiple cows) and what it does not (blur, expert BCS scoreability, behavior label correctness).

---

## 3. Forensic Findings & Empirical Data

All statistics are derived from `artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv` via `scripts/build_real_cattle_quality_contact_sheets.py`.

### 3.1 Overall 1,000-Image Distributions (N=1,000)
- **Manifest Row Count**: Exactly **1,000**
- **Review Provenance**: 1,000 / 1,000 (100.0%) `human_verified`
- **Dataset Composition**: ScienceDB: 334 (33.4%), MmCows: 333 (33.3%), SideViewCows2026: 333 (33.3%)
- **Strict-Clean Overall**: **342 / 1,000 (34.20%)**
  - Definition: `occlusion == 'none' & body_cutoff == 'none' & multiple_cows == 'no' & viewpoint != 'unknown / ambiguous'`
- **Viewpoint Overall**:
  - `side`: 453 (45.30%)
  - `rear`: 312 (31.20%)
  - `unknown / ambiguous`: 149 (14.90%)
  - `rear-oblique`: 43 (4.30%)
  - `front-oblique`: 41 (4.10%)
  - `front`: 2 (0.20%)
- **Occlusion Overall**:
  - `none`: 715 (71.50%)
  - `partial`: 163 (16.30%)
  - `severe`: 122 (12.20%)
- **Body Cutoff Overall**:
  - `none`: 799 (79.90%)
  - `partial`: 136 (13.60%)
  - `severe`: 65 (6.50%)
- **Multiple Cows Overall**:
  - `no`: 761 (76.10%)
  - `yes`: 239 (23.90%)

---

### 3.2 Per-Dataset Breakdown & Statistical Audit

```text
========================================================================================
Dataset             N     Strict-Clean (%)   Unknown VP (%)   Occlusion (P+S)   Multi-Cow
========================================================================================
ScienceDB          334     171 (51.20%)        15 ( 4.49%)      1 ( 0.30%)     128 (38.32%)
MmCows             333      43 (12.91%)       133 (39.94%)    181 (54.35%)      78 (23.42%)
SideViewCows2026   333     128 (38.44%)         1 ( 0.30%)    103 (30.93%)      33 ( 9.91%)
========================================================================================
TOTAL             1000     342 (34.20%)       149 (14.90%)    285 (28.50%)     239 (23.90%)
```

#### Detailed Metrics:
1. **ScienceDB (N=334)**:
   - Viewpoint: `rear`: 304 (91.02%), `rear-oblique`: 15 (4.49%), `unknown / ambiguous`: 15 (4.49%).
   - Occlusion: `none`: 333 (99.70%), `partial`: 1 (0.30%), `severe`: 0 (0.00%).
   - Body Cutoff: `none`: 291 (87.13%), `partial`: 29 (8.68%), `severe`: 14 (4.19%).
   - Multiple Cows: `no`: 206 (61.68%), `yes`: 128 (38.32%).
   - Strict-Clean: **171 / 334 (51.20%)**.

2. **MmCows (N=333)**:
   - Viewpoint: `unknown / ambiguous`: 133 (39.94%), `side`: 130 (39.04%), `front-oblique`: 38 (11.41%), `rear-oblique`: 22 (6.61%), `rear`: 8 (2.40%), `front`: 2 (0.60%).
   - Occlusion: `none`: 152 (45.65%), `severe`: 93 (27.93%), `partial`: 88 (26.43%).
   - Body Cutoff: `none`: 282 (84.68%), `partial`: 32 (9.61%), `severe`: 19 (5.71%).
   - Multiple Cows: `no`: 255 (76.58%), `yes`: 78 (23.42%).
   - Strict-Clean: **43 / 333 (12.91%)**.

3. **SideViewCows2026 (N=333)**:
   - Viewpoint: `side`: 323 (97.00%), `rear-oblique`: 6 (1.80%), `front-oblique`: 3 (0.90%), `unknown / ambiguous`: 1 (0.30%).
   - Occlusion: `none`: 230 (69.07%), `partial`: 74 (22.22%), `severe`: 29 (8.71%).
   - Body Cutoff: `none`: 226 (67.87%), `partial`: 75 (22.52%), `severe`: 32 (9.61%).
   - Multiple Cows: `no`: 300 (90.09%), `yes`: 33 (9.91%).
   - Strict-Clean: **128 / 333 (38.44%)**.

---

### 3.3 Explicit Verification of Prior Chat Claims
- Claim: MmCows has 333 reviewed samples -> **VERIFIED** (333)
- Claim: 133/333 viewpoint unknown/ambiguous -> **VERIFIED** (133 / 333, 39.94%)
- Claim: 88 partial occlusion in MmCows -> **VERIFIED** (88 / 333, 26.43%)
- Claim: 93 severe occlusion in MmCows -> **VERIFIED** (93 / 333, 27.93%)
- Claim: about 43/333 strict-clean in MmCows -> **VERIFIED** (exactly 43 / 333, 12.91%)
- Claim: ScienceDB has 334 reviewed samples -> **VERIFIED** (334)
- Claim: 128/334 multiple cows in ScienceDB -> **VERIFIED** (128 / 334, 38.32%)
- Claim: SideView has 333 reviewed samples -> **VERIFIED** (333)

---

## 4. Architectural Decisions & Scientific Trade-Off Analysis

### 4.1 Historical Reconciliation with 20-Sample MmCows Inspection
The earlier 20-sample visual inspection (`docs/audits/phase3_behavior_agent_visual_inspection.md`) concluded that 19/20 MmCows samples were usable. 
- **Decision**: Retain the 20-sample report untouched.
- **Scientific Rationale**: The discrepancy arises from sample size and measurement criteria. The 20-sample inspection evaluated coarse behavioral recognizability (is a cow visibly lying down or feeding?). The 1,000-image schema evaluated strict visual quality criteria (viewpoint ambiguity and foreground pipe occlusion). The older result is recognized as **sample-size-limited**, not fraudulent or erroneous. The 333-sample dataset now serves as the authoritative basis for MmCows photographic quality.

### 4.2 Scientific Assessment of ScienceDB for BCS
While 38.32% of ScienceDB images contain multiple cows and 12.87% have body cutoffs, the target cow's pelvic anatomy (hook bones, pin bones, thurl, tailhead) remains largely unobstructed (99.70% occlusion none).
- **Decision**: ScienceDB is **NOT demoted** from its primary BCS role.
- **Scientific Rationale**: The 1,000-image schema did not evaluate expert anatomical scoreability or sensor blur. Framing imperfections and peripheral cows do not invalidate rear pelvic assessment in a squeeze chute.

### 4.3 Governance Rules & Action Plan
1. **NO Dataset Role Changes**: All primary and external roles remain as codified in `phase3_canonical_roadmap.md`.
2. **NO Roadmap Modifications**: Scope and progression sequence remain locked.
3. **NO Premature Dataset Substitution**: No alternative datasets (e.g. Ruchay, Simmental, XGain) will be promoted without equivalent forensic and visual quality audits.
4. **Warranted Next Investigation**: A dedicated task-specific visual BCS quality audit of Ruchay et al. 2026 RGB-D BCS is recommended before Step 3 caching.

---

## 5. Artifacts & File Registry

- **Manifest (Source of Truth)**: [`artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv)
- **Contact Sheet Builder Script**: [`scripts/build_real_cattle_quality_contact_sheets.py`](file:///d:/cattle-health-monitoring-multi-task-model/scripts/build_real_cattle_quality_contact_sheets.py)
- **Full Contact Sheet Index**: [`docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_real_cattle_visual_quality_contact_sheet_index.md)
- **Contact Sheet Assets**: [`docs/audits/assets/real_cattle_visual_quality_reassessment/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/real_cattle_visual_quality_reassessment/) (42 JPEG sheets: `sciencedb_sheet_01..14.jpg`, `mmcows_sheet_01..14.jpg`, `sideview_sheet_01..14.jpg`)
- **Main Audit Report**: [`docs/audits/phase3_real_cattle_visual_quality_reassessment.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_real_cattle_visual_quality_reassessment.md)

---

## 6. Next Steps

1. Update `docs/research_log/README.md` index table with this entry.
2. Update `memory/state.md` to record the 1,000-image milestone and adjust the immediate next action to address dataset quality considerations.
3. Update `memory/history.md` and `memory/index.md`.
4. Validate that all 1,000 manifest rows are represented exactly once, manifests are unchanged, and git diff has zero accidental roadmap/role edits.
5. Commit and push cleanly to `main`.
