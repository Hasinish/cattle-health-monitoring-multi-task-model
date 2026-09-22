# Ruchay et al. (2026) RGB-D BCS Visual Quality Audit Report

**Date**: 2026-09-22  
**Auditors**: Hasin Ishrak & Antigravity Research Agent (with ChatGPT Vision Assistance)  
**Dataset**: Ruchay et al. (2026) (*RGB-D dataset of dairy cows for body condition scoring*, Zenodo record `10.5281/zenodo.20290988`)  
**Status**: COMPLETE / AUDITED  
**Verdict**: **PASS FOR EXTERNAL BCS VALIDATION** (Frozen Benchmark Role Preserved)

---

## 1. Review Basis

This visual quality audit provides the formal empirical basis for evaluating the visual suitability of Ruchay et al. (2026) as the **Primary External BCS Validation Benchmark** in Phase 3.

The audit was conducted via direct manual visual inspection of the 100-sample high-resolution contact-sheet pack generated under deterministic seed `2026`:
- **Image Count**: Exactly 100 RGB images sampled across all 10 Ferguson 5-point BCS classes (2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00).
- **Class Balance**: Exactly 10 images per BCS class.
- **Biological Diversity**: 94 unique biological cow IDs represented (the theoretical maximum attainable diversity in the dataset, given that class 2.75 contains only 4 cows in the entire 25,700-image dataset).
- **Image Fidelity**: Full-resolution 1080p (1920x1080) RGB imagery acquired from the raw Zenodo ZIP archives via lossless extraction.
- **Sensor Setup**: Overhead nadir Microsoft Kinect sensor mounted at 3.05m height above a milking parlor / chute passage.
- **Review Methodology**: Independent manual visual review performed by Hasin Ishrak with assistive visual cross-check from ChatGPT vision across all 10 contact sheets (`docs/audits/assets/ruchay_bcs_visual_audit/ruchay_bcs_sheet_01_bcs2.75.jpg` through `ruchay_bcs_sheet_10_bcs5.00.jpg`).

---

## 2. Main Visual Findings

Detailed visual inspection of the 10 contact sheets yielded the following consistent observational characteristics:

1. **Dorsal and Rump Morphology Is Generally Clearly Visible**:
   - The primary anatomical landmarks required for veterinary Body Condition Scoring on the Ferguson 5-point scale—specifically the thoracic and lumbar spinous processes (chine and loin), tuber coxae (hooks), tuber ischii (pins), thurl depression, sacral ligament, and tailhead fat pocket—are observable in the vast majority of frames.
   - Lean cattle (BCS 2.75–3.00) display prominent angular bony protrusions and deep triangular depressions between hooks and pins.
   - Obese cattle (BCS 4.50–5.00) display rounded contours, smooth tissue coverage over the transverse processes, and thick fat deposits surrounding the tailhead.

2. **Image Sharpness and Spatial Resolution**:
   - Full 1080p (1920x1080) sensor frames provide sufficient spatial resolution to distinguish subtle contour gradients across the lumbar shelf and pelvis.
   - Motion blur is minimal, consistent with dairy cattle standing or walking at low speed through milking stalls.
   - Diffuse indoor barn illumination provides adequate contrast without severe specular blowouts, though shadow fall-off is present along the lateral flanks.

3. **Multi-Cow Crowding and Stall Architecture**:
   - Unlike ScienceDB (where cows move through a narrow single-file chute photographed from ground level rear), Ruchay imagery is captured from an overhead nadir perspective over parallel milking/holding stalls.
   - Multiple neighboring cows standing in adjacent stalls are visible along the lateral margins in a substantial fraction of frames.
   - Metallic stall divider pipes, stanchions, and gates are regularly present in the field of view. However, these structures flank the cow laterally and usually do **not** occlude or obstruct the central dorsal spine or sacral/tailhead anatomy.

4. **Necessity of Upstream Cattle Localization**:
   - Because flanking cows and metal stall hardware appear frequently in the scene, feeding uncropped, raw 1080p frames directly into a global image classifier would invite severe spatial shortcut learning and multi-cow interference.
   - **Conclusion on Pipeline Design**: Upstream target-cow localization (e.g. RT-DETR-L bounding-box detection) and soft segmentation masking (e.g. SAM 2.1) are essential prerequisites before feeding Ruchay crops to downstream BCS scoring heads.

5. **Head and Front-Body Truncation**:
   - Consistent with overhead nadir cameras focused on the rear/dorsal region, the head and anterior cervical spine are frequently truncated or omitted from the top of the frame.
   - This truncation is non-critical for BCS scoring, as veterinary scoring guidelines rely almost exclusively on the dorsal loin, pelvic triangle, and caudal tailhead morphology.

---

## 3. Important Dataset Caveat: Session & Environmental Confounding

Before asserting any conclusions regarding cross-domain model evaluation, an exhaustive cross-tabulation of the complete master manifest (`datasets/bcs/external/ruchay2026/ruchay2026_manifest.csv`, N=25,700 samples, 1,025 unique cows across 4 recording dates) was executed.

### Verified Full-Manifest Distribution (Sample Counts)

| BCS Class | 06.12.2024 | 20.02.2025 | 20.03.2025 | 27.03.2025 | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2.75** | 80 | 0 | 0 | 0 | **80** |
| **3.00** | 940 | 340 | 380 | 400 | **2,060** |
| **3.25** | 1,360 | 2,940 | 2,720 | 2,500 | **9,520** |
| **3.50** | 360 | 1,120 | 1,520 | 4,220 | **7,220** |
| **3.75** | 60 | 160 | 280 | 2,040 | **2,540** |
| **4.00** | 40 | 0 | 120 | 1,460 | **1,620** |
| **4.25** | 0 | 0 | 0 | 380 | **380** |
| **4.50** | 0 | 0 | 0 | 1,080 | **1,080** |
| **4.75** | 0 | 0 | 0 | 360 | **360** |
| **5.00** | 0 | 0 | 0 | 840 | **840** |
| **TOTAL** | **2,840** | **4,560** | **5,020** | **13,280** | **25,700** |

### Verified Full-Manifest Distribution (Unique Cow IDs)

| BCS Class | 06.12.2024 | 20.02.2025 | 20.03.2025 | 27.03.2025 | Total Unique Cows |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2.75** | 4 | 0 | 0 | 0 | **4** |
| **3.00** | 46 | 17 | 19 | 20 | **96** |
| **3.25** | 66 | 145 | 135 | 124 | **410** |
| **3.50** | 18 | 56 | 76 | 209 | **347** |
| **3.75** | 3 | 8 | 14 | 102 | **125** |
| **4.00** | 2 | 0 | 6 | 73 | **81** |
| **4.25** | 0 | 0 | 0 | 19 | **19** |
| **4.50** | 0 | 0 | 0 | 54 | **54** |
| **4.75** | 0 | 0 | 0 | 18 | **18** |
| **5.00** | 0 | 0 | 0 | 42 | **42** |
| **TOTAL (Unique)** | **139** | **226** | **250** | **608** | **1,025** |

### Critical Confounding Risk Analysis
The physical data proves an undeniable, structural association between BCS classes and recording dates:
1. **BCS 2.75 Is Exclusively Confined to Session 06.12.2024**:
   - In the entire dataset, only 4 biological cows have BCS 2.75, and all 80 frames were recorded on December 6, 2024. No other session contains any samples below 3.00.
2. **High BCS Classes (4.25 to 5.00) Are Exclusively Confined to Session 27.03.2025**:
   - All 2,660 images across classes 4.25 (380), 4.50 (1,080), 4.75 (360), and 5.00 (840)—representing 133 unique cows—were recorded on March 27, 2025.
   - Sessions `20.02.2025` and `20.03.2025` contain zero cows rated above 4.00 (and `20.02.2025` has none above 3.75).
3. **Implications for External Evaluation**:
   - Any model that relies on global scene characteristics (e.g. ambient lighting temperature, background stall cleanliness, seasonal barn dust, or sensor gain shifts between December and March) could achieve spuriously high accuracy on extreme BCS classes by identifying the recording session rather than the cow's anatomy.
   - **Audit Directive**: When evaluating trained models on Ruchay 2026 in Step 10, researchers must not interpret high classification performance naively. Isolating the cow via bounding-box cropping and background segmentation masking is scientifically mandatory to prevent session-shortcut exploitation.

---

## 4. Final Audit Verdict

### **PASS FOR EXTERNAL BCS VALIDATION**

Ruchay et al. (2026) **SHALL REMAIN the Primary External BCS Validation Benchmark** for Phase 3.

### Rationale:
The visual audit conclusively demonstrates that:
1. The relevant dorsal, lumbar, pelvic, and tailhead morphology required for visual Body Condition Scoring is clearly observable and sharp in full-resolution RGB frames.
2. The imagery presents a legitimate, substantial domain shift from the in-domain training set (ScienceDB), featuring an overhead nadir 3.05m perspective, parallel stall fixtures, and loose-housing cows.
3. Metal stall hardware does not significantly obstruct the primary dorsal scoring areas.

### Scientific Governance & Execution Constraints:
- **Frozen Benchmark Status**: Ruchay 2026 must remain strictly held-out external evaluation data. Under no circumstances may model architectures, loss functions, or hyperparameters be tuned or selected against Ruchay test metrics.
- **Mandatory Confounding Acknowledgment**: When reporting external validation performance in the thesis, the documented BCS-by-session imbalance must be formally acknowledged as an inherent data limitation.
- **Visual Suitability vs. Empirical Generalization**: This audit establishes that the imagery is *visually and anatomically suitable* for BCS scoring. It does **not** prove, predict, or guarantee that any deep learning model will achieve high generalization accuracy on this benchmark prior to running the actual experiments.

---

## 5. Thesis Relevance & Role in Phase 3

The retention of Ruchay 2026 serves a pivotal role in answering the core thesis question:
> *Can cattle-centered visual representations reduce shortcut learning and improve robustness compared with generic RGB representations?*

### Domain-Shift Characteristics:
- **ScienceDB (Primary In-Domain)**: Ground-level, rear-facing horizontal camera, narrow single-file race chute, tight rear pelvic framing, zero overhead views.
- **Ruchay 2026 (Primary External Validation)**: Ceiling-mounted nadir camera at 3.05m, overhead dorsal perspective, parallel milking stall framing with flanking neighboring cows and metal stall pipes.

Because the visual environment and camera geometry differ radically from ScienceDB, Ruchay 2026 provides a stringent, highly diagnostic test of whether cattle-centered priors (bounding box localization and foreground segmentation masks) successfully strip away irrelevant background chute features and enable genuine anatomical transfer.

> [!NOTE]
> In accordance with scientific integrity rules, this audit makes no claim that cattle-centered models *will* improve transfer performance on Ruchay 2026; that hypothesis remains to be tested empirically during Step 10 robustness evaluation.

---

## 6. Associated Audit Deliverables

- **Visual Contact Sheets Gallery**: [`docs/audits/assets/ruchay_bcs_visual_audit/`](assets/ruchay_bcs_visual_audit/) (10 sheets, BCS 2.75–5.00)
- **Visual Audit Index**: [`docs/audits/phase3_ruchay_bcs_visual_audit_index.md`](phase3_ruchay_bcs_visual_audit_index.md)
- **100-Sample Provenance Manifest**: [`artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv`](../../artifacts/perception_audit/ruchay_bcs_visual_audit_manifest.csv)
- **Contact Sheet Builder Script**: [`scripts/build_ruchay_bcs_visual_audit_pack.py`](../../scripts/build_ruchay_bcs_visual_audit_pack.py)
- **Master Dataset Manifest**: [`datasets/bcs/external/ruchay2026/ruchay2026_manifest.csv`](../../datasets/bcs/external/ruchay2026/ruchay2026_manifest.csv)
