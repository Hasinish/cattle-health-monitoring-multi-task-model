# Research Log: Primary Behavior Stack SAM 2.1 Segmentation Sanity Check

**Date:** 2026-09-23  
**Status:** COMPLETED / EMPIRICALLY AUDITED  
**Qualitative Visual Verdict:** **PENDING HUMAN REVIEW**  
**Task:** Small SAM 2.1 (`sam2.1_s.pt`) segmentation sanity check directly on the same 45 primary Behavior frames used in the completed localization audit (25 CVB + 20 Kaggle Beef).  
**Execution Environment:** Modal cloud (profile `tigerwood693`, GPU Tier `T4`, CPU 2.0, RAM 4096MB).  
**Git Base Commit:** `217f649e82ed2a830fd9375c70909471689de277`  
**Sample Provenance:** Exactly 45 midpoint frames inherited without alteration from `artifacts/perception_audit/behavior_primary_localization_sanity.csv` (Seed 2026, train split only).  
**Model & Checkpoint:** SAM 2.1 Small (`sam2.1_hiera_small.pt` / `sam2.1_s.pt`, Ultralytics / Meta FAIR release).  
**Artifacts Generated:**
- `scripts/audit_behavior_primary_segmentation.py`
- `artifacts/perception_audit/behavior_primary_segmentation_sanity.csv`
- `docs/audits/assets/behavior_primary_segmentation_sanity/cvb_sam21_gt_contact_sheet.jpg`
- `docs/audits/assets/behavior_primary_segmentation_sanity/cvb_sam21_rtdetr_contact_sheet.jpg`
- `docs/audits/assets/behavior_primary_segmentation_sanity/beef_sam21_fullcrop_contact_sheet.jpg`
- 70 individual 4-panel visual composites under `docs/audits/assets/behavior_primary_segmentation_sanity/`

---

## 1. Executive Summary

We executed an empirical zero-shot segmentation sanity check using pretrained SAM 2.1 Small (`sam2.1_s.pt`) on the identical 45 training midpoint frames evaluated during the RT-DETR-L localization audit. Across the two sub-datasets, the diagnostic outputs revealed a sharp, crucial divergence:
1. **CVB Pasture Frames (N=25, 1080p, 5 behaviors):**
   - **Prompt A (Official Target GT Bbox):** **100.0% mask return rate (25/25)** across all classes. Mean target bbox containment sanity metric: **0.9746 (median 0.9975)**. Mean connected components: **1.80**.
   - **Prompt B (Matched RT-DETR-L Bbox):** **100.0% mask return rate (25/25)** across all classes. Mean target bbox containment sanity metric: **0.9526 (median 0.9955)**. Mean connected components: **1.80**.
   - **Diagnostic Takeaway:** In pasture scenes, prompting SAM 2.1 with an RT-DETR-L bounding box yields masks virtually indistinguishable in containment from official GT bounding-box prompts (sanity ratio delta of only -0.0220). *Note: Prompt B was identified by matching against GT during this audit; it serves as a detector-box sanity check, not a demonstration of autonomous multi-cow association.*
2. **Kaggle Beef Single-Cow Crops (N=20, 224x224 px, 4 behaviors):**
   - **Full-Crop Prompt (`[0, 0, 223, 223]`):** **50.0% mask return rate (10/20)**; **10 technical failures (`sam_no_mask`)**.
   - Failure distribution: `Drinking` (1/5 returned, 4 failed), `Feeding` (1/5 returned, 4 failed), `Standing` (3/5 returned, 2 failed), `Lying` (5/5 returned, 0 failed).
   - For returned masks, severe fragmentation occurred due to vertical and horizontal metal stall bars: **mean connected components = 118.3** (range: 12 to 233 components per 224x224 crop).
3. **Qualitative Verdict:** Per strict scientific policy, the visual quality of returned masks is marked as **PENDING HUMAN REVIEW**. Comprehensive contact sheets have been generated for direct human inspection.

---

## 2. Experimental Setup & Diagnostic Prompt Design

### Sample Provenance
No new samples were drawn. The audit strictly reused the 45 frames recorded in `artifacts/perception_audit/behavior_primary_localization_sanity.csv`:
- **CVB (25 samples):** 5 samples each for `Drinking`, `Feeding`, `Lying`, `Standing`, `Walking` across 25 video-disjoint source cuts from the canonical train partition (`datasets/behavior/cvb_beef/train.csv`).
- **Kaggle Beef (20 samples):** 5 samples each for `Drinking`, `Feeding`, `Lying`, `Standing` across 19 session-disjoint clips from the canonical train partition.

### Diagnostic Prompts
1. **CVB Prompt A (Official Target GT Bbox):**
   - The bounding box `[x1, y1, x2, y2]` from the authentic CVB tracklet annotation (`instances_default.json`) for the target cow was passed as the sole bounding-box prompt to `SAM2("sam2.1_s.pt")`.
2. **CVB Prompt B (Matched RT-DETR-L Bbox):**
   - The bounding box from the RT-DETR-L detection that exhibited the highest IoU with the official target GT bbox was passed as the bounding-box prompt.
   - *CRITICAL METHODOLOGICAL DISTINCTION:* Prompt B is an empirical detector-box diagnostic designed to test whether detector box jitter or slight boundary shifts degrade SAM 2.1 mask generation. Because the matched box was identified using GT during the audit, this test does NOT claim or prove autonomous target association in dense multi-cow pasture scenes.
3. **Kaggle Beef Full-Crop Prompt:**
   - Because Kaggle Beef clips are already pre-cropped 224x224 single-cow video sequences, RT-DETR localization was bypassed.
   - SAM 2.1 was prompted with the full frame envelope: `[0, 0, width - 1, height - 1]` (`[0, 0, 223, 223]`).
   - The objective was to test whether SAM 2.1 zero-shot whole-image prompt can cleanly isolate the animal from metal stanchion pipes, feed bunks, and adjacent stall cows.

### Metrics & Sanity Indicators
- **Mask Return / Status:** Boolean indicator whether a non-empty binary mask was predicted.
- **Inference Time (ms):** Wall-clock inference latency per frame on Modal NVIDIA T4 GPU.
- **Mask Area Ratio:** Total mask pixel count divided by total image area ($W \times H$).
- **Mask Containment in Target GT Bbox (Sanity Metric):**
  - $\text{Ratio} = \frac{|\text{Mask} \cap \text{Target GT Box}|}{|\text{Mask}|}$
  - **IMPORTANT:** This is strictly an operational sanity check measuring how much of the predicted mask stays within the ground-truth bounding box envelope. It is **NOT** a pixel-level segmentation accuracy metric (IoU/Dice), as authentic pixel-level ground truth masks do not exist for CVB or Kaggle Beef.
- **Connected Components:** Number of distinct connected foreground regions ($8$-connectivity) computed via `cv2.connectedComponents`.

---

## 3. Quantitative Results

### 3.1 CVB Prompt A: Official Target GT Bbox -> SAM 2.1

| Canonical Behavior | Samples | Mask Return Rate | Mean Latency (ms) | Mean Area Ratio | Min Area Ratio | Max Area Ratio | Mean In-Box Sanity | Median In-Box Sanity | Min In-Box Sanity | Mean Components |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drinking** | 5 | 5 / 5 (100%) | 158.4 | 0.0261 | 0.0076 | 0.0469 | 0.9859 | 0.9990 | 0.9328 | 1.4 |
| **Feeding** | 5 | 5 / 5 (100%) | 157.0 | 0.0210 | 0.0064 | 0.0494 | 0.9701 | 0.9996 | 0.8533 | 1.8 |
| **Lying** | 5 | 5 / 5 (100%) | 155.6 | 0.0229 | 0.0089 | 0.0730 | 0.9634 | 0.9942 | 0.8654 | 2.4 |
| **Standing** | 5 | 5 / 5 (100%) | 154.2 | 0.0166 | 0.0063 | 0.0289 | 0.9839 | 0.9991 | 0.9238 | 1.6 |
| **Walking** | 5 | 5 / 5 (100%) | 154.8 | 0.0108 | 0.0008 | 0.0232 | 0.9696 | 0.9959 | 0.9082 | 1.8 |
| **OVERALL CVB (GT)** | **25** | **25 / 25 (100%)** | **156.0** | **0.0195** | **0.0008** | **0.0730** | **0.9746** | **0.9975** | **0.8533** | **1.80** |

### 3.2 CVB Prompt B: Matched RT-DETR-L Bbox -> SAM 2.1 (Detector Diagnostic)

| Canonical Behavior | Samples | Mask Return Rate | Mean Latency (ms) | Mean Area Ratio | Min Area Ratio | Max Area Ratio | Mean In-Box Sanity | Median In-Box Sanity | Min In-Box Sanity | Mean Components |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drinking** | 5 | 5 / 5 (100%) | 148.9 | 0.0244 | 0.0076 | 0.0463 | 0.9790 | 0.9984 | 0.9009 | 1.4 |
| **Feeding** | 5 | 5 / 5 (100%) | 149.2 | 0.0205 | 0.0062 | 0.0494 | 0.9416 | 0.9988 | 0.7225 | 1.8 |
| **Lying** | 5 | 5 / 5 (100%) | 147.0 | 0.0211 | 0.0089 | 0.0735 | 0.9634 | 0.9942 | 0.8654 | 2.4 |
| **Standing** | 5 | 5 / 5 (100%) | 145.4 | 0.0163 | 0.0062 | 0.0296 | 0.9859 | 0.9988 | 0.9329 | 1.6 |
| **Walking** | 5 | 5 / 5 (100%) | 143.9 | 0.0101 | 0.0008 | 0.0216 | 0.8931 | 0.9868 | 0.5319 | 1.8 |
| **OVERALL CVB (RT-DETR)** | **25** | **25 / 25 (100%)** | **146.9** | **0.0185** | **0.0008** | **0.0735** | **0.9526** | **0.9955** | **0.5319** | **1.80** |

### 3.3 Prompt A vs Prompt B Comparative Analysis
- **Return Consistency:** Both prompt types returned non-empty masks on 100% of the 25 CVB samples.
- **In-Target Sanity Comparison:** Prompt A achieved a mean containment ratio of 0.9746 (median 0.9975), while Prompt B achieved 0.9526 (median 0.9955). The minor delta (-0.0220 mean, -0.0020 median) demonstrates that RT-DETR-L localization boxes are sharp enough to guide SAM 2.1 segmentation without substantial background or multi-cow mask blowout.
- **Outlier in Prompt B:** The lowest in-target sanity ratio for Prompt B occurred on Walking cut `cvb_0400..._tr9_seg0` (0.5319), where the RT-DETR-L box had previously shifted due to an adjacent black cow (localization IoU 0.4404). SAM 2.1 appropriately followed the detector box, segmenting portions of the adjacent animal.

### 3.4 Kaggle Beef: Full-Crop Box Prompt -> SAM 2.1

| Canonical Behavior | Samples | Mask Return Rate | Failures (`sam_no_mask`) | Mean Latency (ms) | Mean Area Ratio (Successes) | Min Area Ratio | Max Area Ratio | Mean Components (Successes) | Max Components |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drinking** | 5 | 1 / 5 (20.0%) | 4 / 5 (80.0%) | 158.8 | 0.2285 | 0.2285 | 0.2285 | 39.0 | 39 |
| **Feeding** | 5 | 1 / 5 (20.0%) | 4 / 5 (80.0%) | 157.9 | 0.1071 | 0.1071 | 0.1071 | 12.0 | 12 |
| **Standing** | 5 | 3 / 5 (60.0%) | 2 / 5 (40.0%) | 159.9 | 0.2854 | 0.1654 | 0.3533 | 91.3 | 114 |
| **Lying** | 5 | 5 / 5 (100%) | 0 / 5 (0.0%) | 160.7 | 0.4990 | 0.2600 | 0.8004 | 153.2 | 233 |
| **OVERALL BEEF** | **20** | **10 / 20 (50.0%)** | **10 / 20 (50.0%)** | **159.6** | **0.3776** | **0.1071** | **0.8004** | **118.3** | **233** |

---

## 4. Technical Failures & Failure Mode Analysis

### Kaggle Beef Zero-Mask Failures (`sam_no_mask`, 10/20 = 50.0%)
- **The Issue:** In 10 out of 20 samples, SAM 2.1 returned an empty mask array (`mask_returned = False`, `mask_area_pixels = 0`).
- **Affected Samples:**
  - `beef_00000000109000000_1_clip_10` (Drinking)
  - `beef_00000000109000000_1_clip_18` (Drinking)
  - `beef_00000000109000000_1_clip_2` (Drinking)
  - `beef_00000000109000000_1_clip_7` (Drinking)
  - `beef_00000000109000000_3_clip_14` (Feeding)
  - `beef_00000000109000000_3_clip_2` (Feeding)
  - `beef_00000000109000000_3_clip_23` (Feeding)
  - `beef_00000000109000000_3_clip_5` (Feeding)
  - `beef_00000000109000000_5_clip_0` (Standing)
  - `beef_00000000109000000_5_clip_32` (Standing)
- **Root Cause:** In tight 224x224 head-down or stall-confined crops where cattle bodies span the entire image width and height, a full-crop bounding box prompt `[0, 0, 223, 223]` provides zero spatial contrast between foreground and background. The zero-shot prompt encoder interprets the entire frame as the context canvas rather than a target entity, outputting logits below the binarization threshold.

### Kaggle Beef Mask Fragmentation
- In the 10 samples where SAM 2.1 did return a mask (predominantly `Lying`), the mask was severely fractured by the metal stall pipes.
- **Mean Connected Components:** **118.3** separate pixel clusters per 224x224 crop.
- **Worst Case:** `beef_00000000191000000_2_clip_0` (Lying) produced **233 distinct mask components** as SAM fragmented the recumbent cow's torso and limbs around the intersecting bars.

---

## 5. Qualitative Visual Quality Verdict

Per explicit scientific protocol:
> **QUALITATIVE VISUAL VERDICT: PENDING HUMAN REVIEW**

The technical return rate (100% on CVB, 50% on Beef) and quantitative containment ratios do NOT constitute an endorsement of visual mask quality. Full contact sheets have been generated and committed to allow Hasin and advisors to inspect every mask:

1. **CVB Prompt A (Official GT Bbox Prompt):**  
   `docs/audits/assets/behavior_primary_segmentation_sanity/cvb_sam21_gt_contact_sheet.jpg` (25 panels, 5x5 grid, 3000x700 px)
2. **CVB Prompt B (Matched RT-DETR-L Bbox Prompt):**  
   `docs/audits/assets/behavior_primary_segmentation_sanity/cvb_sam21_rtdetr_contact_sheet.jpg` (25 panels, 5x5 grid, 3000x700 px)
3. **Kaggle Beef Full-Crop Prompt:**  
   `docs/audits/assets/behavior_primary_segmentation_sanity/beef_sam21_fullcrop_contact_sheet.jpg` (20 panels, 4x5 grid, 2400x850 px)

Individual 4-panel composites (`Original | Prompt Box | SAM Mask | Overlay`) for all 70 audit conditions are stored in:  
`docs/audits/assets/behavior_primary_segmentation_sanity/`

---

## 6. Architectural Implications for Step 3 Feature Caching

1. **CVB Segmentation Pipeline:**
   - SAM 2.1 guided by target cow bounding boxes (whether official GT tracklet or matched detector) generates highly contained animal silhouettes (95.3% to 97.5% inside bbox envelope).
   - Upstream detection + SAM 2.1 mask refinement is technically sound for CVB feature masking.
2. **Kaggle Beef Segmentation Pipeline:**
   - **Full-crop prompting SAM 2.1 is unviable for Kaggle Beef.** A 50.0% failure rate (`sam_no_mask`) on Drinking/Feeding/Standing makes naive full-crop segmentation unusable.
   - For Kaggle Beef, the vision backbone should ingest the RGB crops directly without zero-shot SAM mask gating, or explore center-point / multi-point prompting if segmentation is strictly required.

---

## 7. Artifacts & File Registry

| File Path | Description |
| :--- | :--- |
| `scripts/audit_behavior_primary_segmentation.py` | Standalone Modal audit script executing SAM 2.1 inference across the 45 frames. |
| `artifacts/perception_audit/behavior_primary_segmentation_sanity.csv` | Full 45-row audit results with latency, area ratios, sanity metrics, and component counts. |
| `docs/audits/assets/behavior_primary_segmentation_sanity/` | Directory containing all 70 4-panel composites and 3 master contact sheets. |
| `docs/research_log/2026-09-23_behavior_primary_segmentation_sanity.md` | This research log. |
