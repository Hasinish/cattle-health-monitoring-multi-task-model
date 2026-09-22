# Research Log: Kaggle Beef SAM 2.1 Prompt-Rescue Audit

**Date:** 2026-09-23  
**Status:** COMPLETED / EMPIRICALLY AUDITED  
**Qualitative Visual Verdict:** **PENDING HUMAN REVIEW**  
**Task:** Controlled prompt-rescue audit testing whether alternative prompting strategies can rescue SAM 2.1 segmentation on the SAME 20 Kaggle Beef training frames evaluated during the baseline localization and segmentation audits.  
**Execution Environment:** Modal cloud (profile `tigerwood693`, GPU Tier `T4`, CPU 2.0, RAM 4096MB).  
**Git Base Commit:** `828b62c6e95c9c049709e365372005f6c2747e03`  
**Sample Provenance:** Exactly 20 midpoint frames from `datasets/behavior/cvb_beef/train.csv` inherited without alteration from `artifacts/perception_audit/behavior_primary_localization_sanity.csv` (Seed 2026).  
**Models & Checkpoints:**
- SAM 2.1 Small: `sam2.1_s.pt` (`sam2.1_hiera_small.pt`, Ultralytics / Meta FAIR)
- RT-DETR-L: `rtdetr-l.pt` (COCO class 19 `cow`, confidence >= 0.25)  
**Artifacts Generated:**
- `scripts/audit_beef_sam_prompt_rescue.py`
- `artifacts/perception_audit/beef_sam_prompt_rescue.csv` (120 evaluation rows)
- `docs/audits/assets/beef_sam_prompt_rescue/beef_A0_contact_sheet.jpg`
- `docs/audits/assets/beef_sam_prompt_rescue/beef_A1_contact_sheet.jpg`
- `docs/audits/assets/beef_sam_prompt_rescue/beef_A2_contact_sheet.jpg`
- `docs/audits/assets/beef_sam_prompt_rescue/beef_A3_contact_sheet.jpg`
- `docs/audits/assets/beef_sam_prompt_rescue/beef_A4_contact_sheet.jpg`
- `docs/audits/assets/beef_sam_prompt_rescue/beef_A5_contact_sheet.jpg`
- 120 individual 4-panel visual composites in `docs/audits/assets/beef_sam_prompt_rescue/`

---

## 1. Executive Summary

In the initial segmentation sanity audit, prompting SAM 2.1 with the full crop boundary (`[0, 0, 223, 223]`) failed critically on Kaggle Beef (50.0% zero-mask return rate; mean 118.3 fragmented connected components through stall bars). We executed a controlled prompt-rescue audit evaluating five alternative prompt strategies against baseline A0 across the identical 20 frames:
1. **Point-Based Prompts (A1, A2, A3) Rescued the Technical Zero-Mask Failure:**
   - **Condition A1 (Center Positive Point `(cx, cy)`):** Achieved **100.0% mask return rate (20/20)** across all behaviors, reducing mean connected components from **118.3 down to 16.6** (median 11.5).
   - **Condition A2 (Multi-Positive Body Points, 5 Pts):** Achieved **100.0% mask return rate (20/20)**, with mean area ratio 0.2618 and mean connected components 34.5 (median 20.5).
   - **Condition A3 (Positive Center + 4 Negative Corners):** Achieved **100.0% mask return rate (20/20)**, producing the tightest masks (mean area ratio 0.1549, mean connected components 26.6).
2. **Detector-Guided Prompts (A4, A5) Restored High Area Coverage but Depend on Upstream Localization:**
   - **Condition A4 (Largest RT-DETR-L Cow Box):** Achieved **95.0% mask return rate (19/20)**, mean area ratio 0.3048, mean connected components 25.8 (median 12.0). 1 technical failure occurred on recumbent cow `beef_00000000580000000_2_clip_3` where RT-DETR detected 0 cows behind dense stanchion pipes (`no_rtdetr_prompt`).
   - **Condition A5 (Largest RT-DETR Box + Center Point):** Achieved **95.0% mask return rate (19/20)**, mean area ratio 0.3143, mean connected components 22.7 (median 14.0). Failed on the same single recumbent sample.
3. **Qualitative Visual Verdict:** While all rescue conditions dramatically eliminated the 50% zero-mask collapse and substantially reduced fragmentation, **NO qualitative winner is selected and Kaggle Beef segmentation is NOT declared solved.** The visual quality of all masks remains **PENDING HUMAN REVIEW**. Comprehensive contact sheets have been generated for direct human visual inspection.

---

## 2. Ultralytics SAM 2.1 API Verification & Supported Prompt Syntax

Prior to execution, we forensically audited the installed Ultralytics (v8.4.51) SAM API source code (`ultralytics.models.sam.predict.Predictor`):
- `_prepare_prompts` and `_inference_features` natively support bounding boxes, sparse point coordinates, point labels (1=positive foreground, 0=negative background), and dense masks.
- **Dimensionality Discovery:** Passing a 2D point list `[[x, y], ...]` causes SAM to interpret each point as a separate prompt for $N$ separate objects, returning $N$ distinct masks. To guide a **single object** using multiple points (or positive + negative points), points must be structured as a 3D tensor of shape `(1, num_points, 2)` with matching label tensor of shape `(1, num_points)`.
- **Combined Box + Point Prompting (Condition A5):** Fully and genuinely supported. In `_inference_features`, `self.model.prompt_encoder(points=points, boxes=bboxes, masks=masks)` concurrently feeds sparse box corner embeddings and sparse point prompt embeddings into the attention sequence of the SAM 2.1 mask decoder.

---

## 3. Experimental Setup & Prompt Conditions

All 20 midpoint frames ($224 \times 224$ px) were drawn from `datasets/behavior/cvb_beef/train.csv` (Seed 2026, 5 samples each for `Drinking`, `Feeding`, `Lying`, `Standing` across 19 unique recording sessions).

| Condition ID | Strategy Name | Prompt Definition | Exact Coordinates / Logic |
| :--- | :--- | :--- | :--- |
| **A0** | Baseline Full Crop | Full crop bounding box | `[0, 0, 223, 223]` (reused baseline result) |
| **A1** | Center Positive Point | 1 foreground point at geometric crop center | `(112, 112)`, label `1` |
| **A2** | Multi-Positive Body Points | 5 deterministic points covering central cow torso | `(112, 112)`, `(78, 112)`, `(146, 112)`, `(112, 78)`, `(112, 146)`, all label `1` |
| **A3** | Positive Center + Neg Corners | Center positive point + 4 negative corner points (8% border margin) | Pos: `(112, 112)` [lbl 1]; Neg: `(18, 18)`, `(206, 18)`, `(18, 206)`, `(206, 206)` [lbl 0] |
| **A4** | Largest RT-DETR-L Cow Box | Bounding box of largest area among class-19 cow detections | Largest `[x1, y1, x2, y2]`; if 0 detections -> `no_rtdetr_prompt` |
| **A5** | RT-DETR Box + Center Point | Largest RT-DETR box + positive point at box center | Bbox + `( (x1+x2)/2, (y1+y2)/2 )` [lbl 1]; if 0 detections -> `no_rtdetr_prompt` |

---

## 4. Quantitative Results & Comparative Analysis

### 4.1 Cross-Condition Performance Summary (N=20 frames per condition)

| Metric | A0: Baseline Full Crop | A1: Center Positive Point | A2: Multi-Positive Body Points | A3: Center Pos + Neg Corners | A4: Largest RT-DETR Box | A5: RT-DETR Box + Center Point |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mask Return Rate** | 10 / 20 (50.0%) | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** | 19 / 20 (95.0%) | 19 / 20 (95.0%) |
| - *Drinking (N=5)* | 1 / 5 (20.0%) | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** |
| - *Feeding (N=5)* | 1 / 5 (20.0%) | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** |
| - *Lying (N=5)* | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | 4 / 5 (80.0%) | 4 / 5 (80.0%) |
| - *Standing (N=5)* | 3 / 5 (60.0%) | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** | **5 / 5 (100.0%)** |
| **Technical Failures** | 10 (`sam_no_mask`) | **0** | **0** | **0** | 1 (`no_rtdetr_prompt`) | 1 (`no_rtdetr_prompt`) |
| **Area Ratio (Mean)** | 0.3776 | 0.1827 | 0.2618 | 0.1549 | 0.3048 | 0.3143 |
| **Area Ratio (Median)** | 0.3725 | 0.2022 | 0.2701 | 0.1789 | 0.3164 | 0.3071 |
| **Area Ratio (Min / Max)**| 0.1071 / 0.8004 | 0.0134 / 0.3758 | 0.0195 / 0.4308 | 0.0140 / 0.2987 | 0.0559 / 0.7056 | 0.0542 / 0.6211 |
| **Connected Comps (Mean)**| 118.3 | **16.6** | 34.5 | 26.6 | 25.8 | 22.7 |
| **Connected Comps (Median)**| 111.5 | **11.5** | 20.5 | 20.0 | **12.0** | 14.0 |
| **Connected Comps (Min / Max)**| 12 / 233 | **1 / 83** | 5 / 120 | 2 / 105 | 1 / 147 | 1 / 116 |
| **Mean Latency (ms)** | 155.2 | 145.9 | 146.2 | 144.8 | **138.0** | 138.8 |

*(Note: Area ratio and connected component statistics are calculated over non-empty returned masks).*

---

## 5. Technical Failure Modes & Edge Case Analysis

### 1. Elimination of Zero-Mask Failures via Point Prompting (A1, A2, A3)
- In Condition A0, 10 of 20 samples completely failed to return any mask because whole-image box prompting provided zero foreground/background contrast.
- In Conditions A1, A2, and A3, specifying explicit point prompts inside the cow body completely eliminated this failure mode (**0 zero-masks across all 60 prompt evaluations**).

### 2. Upstream Detector Failure in A4 and A5 (`beef_00000000580000000_2_clip_3`)
- Sample `beef_00000000580000000_2_clip_3` (Lying) depicts a cow in sternal recumbency viewed through heavy, intersecting horizontal and vertical metal stall bars.
- Zero-shot RT-DETR-L detected 0 cattle (`number_of_cow_detections = 0`). Consequently, Conditions A4 and A5 could not construct a prompt and recorded `no_rtdetr_prompt`.
- In contrast, pure point prompting (A1, A2, A3) successfully produced masks for this sample without relying on an upstream bounding-box detector.

### 3. Fragmentation Reduction Across Prompt Types
- Under baseline A0, returned masks suffered extreme fragmentation (**mean 118.3 components**, maximum 233 on standing cow `beef_191_2_clip_0`).
- Point prompting reduced fragmentation by up to **86.0%**:
  - Condition A1 reduced mean components to **16.6** (median 11.5).
  - Condition A5 reduced mean components to **22.7** (median 14.0).
  - Condition A4 reduced mean components to **25.8** (median 12.0).

---

## 6. Qualitative Visual Review & Master Contact Sheets

Per mandatory scientific policy:
> **QUALITATIVE VISUAL STATUS: PENDING HUMAN REVIEW**  
> We do NOT automatically declare any condition a winner. Higher return rate or lower component counts do not prove that a mask captures the correct anatomical boundary or suppresses stall bars. Hasin and research advisors must inspect the contact sheets directly.

Master contact sheets ($2400 \times 920$ px, 4 columns $\times$ 5 rows) displaying `Original | Prompt Vis | SAM Mask | Overlay` for all 20 frames per condition are generated and available at:

1. **Condition A0 (Baseline Full Crop Box):**  
   [beef_A0_contact_sheet.jpg](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/beef_sam_prompt_rescue/beef_A0_contact_sheet.jpg)
2. **Condition A1 (Center Positive Point `(cx, cy)`):**  
   [beef_A1_contact_sheet.jpg](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/beef_sam_prompt_rescue/beef_A1_contact_sheet.jpg)
3. **Condition A2 (Multi-Positive Body Points, 5 Pts):**  
   [beef_A2_contact_sheet.jpg](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/beef_sam_prompt_rescue/beef_A2_contact_sheet.jpg)
4. **Condition A3 (Positive Center + 4 Negative Corners):**  
   [beef_A3_contact_sheet.jpg](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/beef_sam_prompt_rescue/beef_A3_contact_sheet.jpg)
5. **Condition A4 (Largest RT-DETR-L Cow Box):**  
   [beef_A4_contact_sheet.jpg](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/beef_sam_prompt_rescue/beef_A4_contact_sheet.jpg)
6. **Condition A5 (Largest RT-DETR Box + Center Point):**  
   [beef_A5_contact_sheet.jpg](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/beef_sam_prompt_rescue/beef_A5_contact_sheet.jpg)

Individual 4-panel composites for all 120 conditions are archived in:  
`docs/audits/assets/beef_sam_prompt_rescue/`

---

## 7. Deliverables & File Registry

| File Path | Description |
| :--- | :--- |
| `scripts/audit_beef_sam_prompt_rescue.py` | Standalone Modal audit script evaluating A0 through A5 across all 20 frames on Modal T4 GPU. |
| `artifacts/perception_audit/beef_sam_prompt_rescue.csv` | Full 120-row deterministic audit dataset containing prompt coordinates, mask return flags, statuses, areas, and component counts. |
| `docs/audits/assets/beef_sam_prompt_rescue/` | Asset directory containing 6 master contact sheets and 120 individual 4-panel composites. |
| `docs/research_log/2026-09-23_beef_sam_prompt_rescue.md` | This research log. |
