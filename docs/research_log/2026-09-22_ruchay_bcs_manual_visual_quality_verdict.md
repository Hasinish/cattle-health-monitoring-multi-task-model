# Research Log: Ruchay 2026 BCS Manual Visual Quality Audit & Verdict

**Date**: 2026-09-22  
**Author**: Hasin Ishrak & Antigravity Research Agent  
**Context**: Finalization of the human visual-quality audit and benchmark verdict for candidate Primary External BCS Validation benchmark Ruchay et al. (2026) (`10.5281/zenodo.20290988`).

---

## 1. Executive Summary
Completed the formal manual visual-quality inspection of the 100-sample high-resolution contact-sheet review pack for candidate Primary External BCS validation benchmark Ruchay et al. (2026), evaluated across all 10 Ferguson 5-point BCS classes (2.75 to 5.00; 10 samples/class; 94 unique biological cow IDs). Manual inspection by Hasin Ishrak assisted by ChatGPT vision confirmed that dorsal spine, loin, hooks, pins, and tailhead morphology are sharp and observable under overhead nadir 1080p Kinect imaging, despite the presence of parallel stall pipes and flanking neighboring cattle. Exhaustive verification of the complete 25,700-sample master manifest exposed a severe structural confound: BCS 2.75 exists exclusively in session `06.12.2024` (4 cows, 80 frames), while high BCS classes 4.25–5.00 exist exclusively in session `27.03.2025` (133 cows, 2,660 frames). Final scientific verdict is **PASS FOR EXTERNAL BCS VALIDATION**; Ruchay 2026 is formally retained as the Primary External BCS Validation benchmark under frozen evaluation rules with explicit documentation of session confounding.

---

## 2. Context & Motivation
In Phase 3 of the thesis, Body Condition Scoring (BCS) requires both in-domain validation (ScienceDB, 53,566 images across 5 classes from ground-level rear race chutes) and rigorous external validation under domain shift. Ruchay et al. (2026) was provisionally designated as the Primary External BCS Validation benchmark. Following the creation of the 100-sample deterministic visual audit pack (`scripts/build_ruchay_bcs_visual_audit_pack.py`), direct visual review was conducted to determine whether the overhead nadir camera framing, stall hardware, and loose-housing cows are suitable for veterinary BCS scoring, and to audit dataset metadata for hidden distribution shortcuts.

---

## 3. Forensic Findings & Data

### Visual Inspection Findings (10 Contact Sheets, N=100)
1. **Anatomical Visibility**: Dorsal spinal processes, lumbar shelves, tuber coxae (hooks), tuber ischii (pins), and caudal fat folds are visible and distinct. Extreme thinness (2.75) presents pronounced angular cavities; obesity (5.00) presents thick, rounded fat blankets around the tailhead.
2. **Scene Framing & Flanking Occlusion**: Overhead nadir perspective at 3.05m captures cows standing side-by-side in parallel milking stalls. Metal divider bars and adjacent cows frequently flank the subject, establishing that target-cow localization (RT-DETR-L) and segmentation masking (SAM 2.1) are essential prerequisites for downstream inference.
3. **Head/Anterior Truncation**: Anterior cervical spine and head are frequently truncated by the overhead framing, but this does not hinder dorsal/pelvic BCS assessment.

### Verified Full-Manifest Distribution (datasets/bcs/external/ruchay2026/ruchay2026_manifest.csv)
Comprehensive cross-tabulation across the entire 25,700 samples (1,025 unique cows, 4 recording dates) revealed:

| BCS Class | 06.12.2024 (Samples / Cows) | 20.02.2025 (Samples / Cows) | 20.03.2025 (Samples / Cows) | 27.03.2025 (Samples / Cows) | Total Samples | Total Unique Cows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.75** | 80 / 4 | 0 / 0 | 0 / 0 | 0 / 0 | **80** | **4** |
| **3.00** | 940 / 46 | 340 / 17 | 380 / 19 | 400 / 20 | **2,060** | **96** |
| **3.25** | 1,360 / 66 | 2,940 / 145 | 2,720 / 135 | 2,500 / 124 | **9,520** | **410** |
| **3.50** | 360 / 18 | 1,120 / 56 | 1,520 / 76 | 4,220 / 209 | **7,220** | **347** |
| **3.75** | 60 / 3 | 160 / 8 | 280 / 14 | 2,040 / 102 | **2,540** | **125** |
| **4.00** | 40 / 2 | 0 / 0 | 120 / 6 | 1,460 / 73 | **1,620** | **81** |
| **4.25** | 0 / 0 | 0 / 0 | 0 / 0 | 380 / 19 | **380** | **19** |
| **4.50** | 0 / 0 | 0 / 0 | 0 / 0 | 1,080 / 54 | **1,080** | **54** |
| **4.75** | 0 / 0 | 0 / 0 | 0 / 0 | 360 / 18 | **360** | **18** |
| **5.00** | 0 / 0 | 0 / 0 | 0 / 0 | 840 / 42 | **840** | **42** |
| **TOTAL** | **2,840 / 139** | **4,560 / 226** | **5,020 / 250** | **13,280 / 608** | **25,700** | **1,025** |

### Critical Session Confounding Limitation:
- BCS 2.75 is 100% confined to `06.12.2024` (4 cows).
- High BCS classes 4.25, 4.50, 4.75, and 5.00 are 100% confined to `27.03.2025` (133 cows).
- No cow above 4.00 was recorded in February or earlier March sessions.
- Unmasked global image models could exploit date-specific barn illumination, stall dust, or camera sensor gains as shortcut predictors for extreme BCS scores.

---

## 4. Architectural Decisions & Action Plan

1. **Audit Verdict**: **PASS FOR EXTERNAL BCS VALIDATION**.
2. **Benchmark Role Preserved**: Ruchay et al. (2026) is confirmed as the **Primary External BCS Validation Benchmark**.
3. **Operational Evaluation Rules**:
   - Ruchay remains strictly a frozen external test set; no tuning or model selection may be performed using Ruchay metrics.
   - External validation in Step 10 must utilize localized cow crops and foreground masks to suppress background session shortcuts.
   - The session confounding limitation must be explicitly disclosed in the thesis.
4. **Scope Boundaries**: The visual audit establishes *visual suitability*, NOT proven generalization accuracy. ScienceDB remains the primary in-domain training dataset.

---

## 5. Artifacts & File Registry
- Main Audit Report: [`docs/audits/phase3_ruchay_bcs_visual_quality_audit.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_ruchay_bcs_visual_quality_audit.md)
- Audit Index & Contact Gallery: [`docs/audits/phase3_ruchay_bcs_visual_audit_index.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_ruchay_bcs_visual_audit_index.md)
- High-Resolution Contact Sheets (10 sheets): [`docs/audits/assets/ruchay_bcs_visual_audit/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/ruchay_bcs_visual_audit/)
- 100-Sample Manifest: [`artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv)
- Master Manifest: [`datasets/bcs/external/ruchay2026/ruchay2026_manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/bcs/external/ruchay2026/ruchay2026_manifest.csv)

---

## 6. Next Steps
1. Resolve Step 2.4 viewpoint generator operational strategy for Step 3 caching.
2. Update `memory/state.md` and `docs/research_log/README.md`.
3. Commit deliverables to Git and push to `origin/main`.
