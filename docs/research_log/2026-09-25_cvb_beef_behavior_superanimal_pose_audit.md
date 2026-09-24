# CVB + Kaggle Beef Behavior SuperAnimal-Quadruped Pose Feasibility Audit

## 1. Executive Summary
To evaluate whether pretrained cattle/quadruped pose priors are technically feasible for direct integration into the primary Phase 3 Run 5 Behavior stack (CVB + Kaggle Beef, T=8 sequences, cattle-centered RGB crops + binary masks), we conducted a forensic feasibility audit of frozen DeepLabCut SuperAnimal-Quadruped ResNet-50 on Modal profile `hasinishrak2015` (NVIDIA Tesla T4). Using a deterministic, stratified representative sample (seed `2026`) covering all 9 available source x class cells (54 sequences total x 8 frames = 432 frames from authentic `retained_train.csv` and `retained_val.csv`), the audit revealed a frame-level pose return rate of **54.40%** (235/432 frames) and a detector failure rate of **45.60%** (197/432 frames). Only **31.48%** of sequences (17/54) returned pose on all 8 frames. In particular, the detector exhibited high missingness on **Lying** cattle: a **27.08%** frame return rate, a **72.92%** detector failure rate, **0.0%** sequence-level complete return (0/12 sequences), and a mean keypoint confidence of **0.3148**. Temporal consistency analysis recorded **880 individual keypoint-transition displacement threshold exceedances** (>0.25 normalized displacement across consecutive valid frames), with **1 of 54 sequences (1.85%)** exhibiting stable tracking across all 8 frames. Predicted-mask geometric containment sanity averaged **73.77%** overall (92.57% on CVB vs 54.16% on Beef).

**Scientific Conclusion**:
> The current frozen SuperAnimal-Quadruped ResNet-50 pipeline shows poor feasibility for direct integration into Run 5 because of high pose missingness, especially for Lying cattle, and weak temporal consistency. Therefore Run 5 + Pose is **deferred/not prioritized under the deadline**. This audit does NOT prove that pose would reduce downstream Behavior accuracy because no Run 5 + Pose model was trained.

---

## 2. Context & Scientific Motivation
In Phase 3 Run 5, cattle behavior recognition is modeled using a 4-channel ResNet-18 spatial trunk coupled to a 1D Temporal Convolutional Network (TCN) processing T=8 chronological frames per sequence. Run 5 achieved:
- Balanced Accuracy: 74.43% (+3.13% over Run 2 baseline)
- Test Loss: 0.4430 (-19.5% reduction)
- Walking minority class F1: 0.2456 (+13.0% relative gain)

While 2D keypoint priors theoretically offer explicit geometric posture landmarks, prior literature and project notes flagged potential extraction challenges on recumbent postures. The user directed an empirical audit of the frozen SuperAnimal pipeline on the **current authentic CVB + Kaggle Beef representation** on Modal profile `hasinishrak2015` before deciding whether to build a Behavior+Pose model. Historical MmCows pose results were explicitly excluded as evidence for the CVB+Beef representation.

### Audit Design & Boundary Conditions
1. **Target Profile**: `hasinishrak2015`, volume `mtl-data` mounted at `/mtl-data`.
2. **Data Scope**: Staged authentic Run 5 sequences (`/mtl-data/behavior/{sample_id}/frame_{t:02d}.jpg`, `mask_{t:02d}.png`). Train and validation splits only (`retained_train.csv`, `retained_val.csv`). Held-out test set strictly untouched.
3. **Stratified Sampling**: Seed `2026`. Exactly 6 sequences (4 train, 2 val) sampled per available source x class cell. Walking exists only in CVB (no synthetic Beef Walking cell was fabricated). Total sample: 9 cells x 6 sequences = 54 sequences (432 frames).
4. **Frozen Model**: Official DeepLabCut SuperAnimal-Quadruped ResNet-50 (`fasterrcnn_resnet50_fpn_v2` detector) with official 39-keypoint schema. Zero fine-tuning.
5. **Metric Boundaries**: Evaluated geometric containment against predicted Run 5 masks strictly labeled as **predicted-mask geometric containment sanity check** (a geometric sanity metric, NOT ground-truth pose accuracy; no keypoint annotations exist on CVB/Beef). Frame-to-frame displacement evaluated strictly as a **temporal consistency sanity indicator** (NOT ground-truth motion tracking).

---

## 3. Quantitative Audit Findings

### 3.1 Overall Performance Summary
| Metric | Value | Detail / Interpretation |
| :--- | :--- | :--- |
| **Total Sequences Evaluated** | **54** | Deterministically stratified (seed 2026, 6 per cell) |
| **Total Frames Evaluated** | **432** | T=8 frames per sequence |
| **Frame-Level Pose Return Rate** | **54.40%** | 235 / 432 frames returned valid keypoint sets |
| **Frame-Level Detector Failure Rate** | **45.60%** | 197 / 432 frames dropped by Faster R-CNN detector |
| **Sequence-Level All-8-Frames Rate** | **31.48%** | 17 of 54 sequences had pose on all 8 frames |
| **Sequence-Level >=1-Frame Rate** | **87.04%** | 47 of 54 sequences had at least 1 detected frame |
| **Mean Keypoint Confidence** | **0.4658** | Median: 0.4381 across all returned keypoints |
| **Predicted-Mask Containment Sanity (All)** | **73.77%** | Geometric sanity metric: predicted keypoints landing inside Run-5 mask |
| **Predicted-Mask Containment Sanity (Conf>=0.2)** | **73.88%** | Geometric sanity metric for keypoints with conf >= 0.2 |
| **Mean Temporal Displacement (Norm)** | **0.1116** | Normalized to 224px crop (~25.0 pixels average displacement) |
| **Keypoint-Transition Jumps (>0.25 Norm)** | **880** | Individual keypoint-transition threshold exceedances |
| **Sequences Flagged STABLE** | **1 / 54 (1.85%)** | Returned >=6 frames with 0 keypoint jumps >0.25 norm |
| **Sequences Flagged PARTIAL_DETECTION_FAILURE** | **32 / 54 (59.26%)** | Returned < 6 detected frames out of 8 |
| **Sequences Flagged SEVERE_INSTABILITY** | **17 / 54 (31.48%)** | Exhibited > 5 keypoint-transition jumps >0.25 norm |
| **Sequences Flagged MODERATE_JUMPS** | **4 / 54 (7.41%)** | Exhibited 1–5 keypoint-transition jumps >0.25 norm |

---

### 3.2 Results by Source: CVB vs. Kaggle Beef
| Metric | CVB (Indoor CCTV Barn) | Kaggle Beef (Pasture / Feedlot) | Delta / Comparison |
| :--- | :--- | :--- | :--- |
| **Total Sequences / Frames** | 30 seqs / 240 frames | 24 seqs / 192 frames | 5 classes vs 4 classes |
| **Pose Return Rate** | **50.00%** (120/240) | **59.90%** (115/192) | +9.90% higher in Beef |
| **Detector Failure Rate** | **50.00%** (120/240) | **40.10%** (77/192) | Higher failure in CVB |
| **All-8-Frames Return Rate** | **33.33%** (10/30) | **29.17%** (7/24) | ~30% complete sequence return in both |
| **Mean Keypoint Confidence** | **0.4673** | **0.4643** | Similar mean confidence |
| **Predicted-Mask Containment (All)** | **92.57%** | **54.16%** | Higher containment on CVB crops |
| **Predicted-Mask Containment (Conf>=0.2)** | **92.84%** | **54.10%** | Higher containment on CVB crops |
| **Mean Temporal Displacement (Norm)** | **0.0786** | **0.1528** | Higher displacement on Beef |
| **Keypoint Jumps (>0.25 Norm)** | **364** | **516** | Individual keypoint-transition exceedances |

---

### 3.3 Results by Canonical Behavior Class
| Canonical Class | Total Seqs / Frames | Pose Return Rate | All-8 Return Rate | Mean Conf | Mask Containment Sanity | Mean Norm Disp | Jumps >0.25 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing** | 12 / 96 | **55.21%** | 25.00% (3/12) | 0.4730 | 76.20% | 0.1121 | 197 |
| **Lying** | 12 / 96 | **27.08%** | **0.00% (0/12)** | **0.3148** | 76.23% | 0.0789 | 45 |
| **Feeding** | 12 / 96 | **58.33%** | 41.67% (5/12) | 0.4603 | 65.06% | 0.1481 | 299 |
| **Drinking** | 12 / 96 | **76.04%** | 50.00% (6/12) | 0.4807 | 71.16% | 0.1503 | 327 |
| **Walking** (CVB only) | 6 / 48 | **56.25%** | 50.00% (3/6) | **0.5686** | **91.74%** | **0.0255** | **12** |

---

### 3.4 Results by Source x Class Cells (All 9 Cells)
| Cell Identifier | Source | Class | Seqs | Return Rate | All-8 Rate | Mean Conf | Mask Containment | Mean Disp | Jumps >0.25 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cvb_Standing` | CVB | Standing | 6 | 41.67% | 16.67% | 0.4078 | 91.15% | 0.0713 | 71 |
| `cvb_Lying` | CVB | Lying | 6 | **27.08%** | **0.00%** | **0.2945** | 94.67% | 0.0609 | 19 |
| `cvb_Feeding` | CVB | Feeding | 6 | 41.67% | 16.67% | 0.4500 | 94.62% | 0.1252 | 100 |
| `cvb_Drinking` | CVB | Drinking | 6 | 83.33% | 83.33% | 0.4935 | 92.12% | 0.1103 | 162 |
| `cvb_Walking` | CVB | Walking | 6 | 56.25% | 50.00% | 0.5686 | 91.74% | 0.0255 | 12 |
| `beef_Standing` | Beef | Standing | 6 | 68.75% | 33.33% | 0.5126 | 67.13% | 0.1529 | 126 |
| `beef_Lying` | Beef | Lying | 6 | **27.08%** | **0.00%** | **0.3352** | 57.79% | 0.0968 | 26 |
| `beef_Feeding` | Beef | Feeding | 6 | 75.00% | 66.67% | 0.4660 | 48.65% | 0.1710 | 199 |
| `beef_Drinking` | Beef | Drinking | 6 | 68.75% | 16.67% | 0.4651 | 45.77% | 0.1904 | 165 |

---

### 3.5 Keypoint Confidence & Reliability Distribution
From the official 39-keypoint SuperAnimal-Quadruped schema, individual keypoints exhibited varying levels of confidence:

#### Top 10 Highest Confidence Keypoints
1. `front_right_paw`: Mean Conf 0.6289 (Median 0.6741, 97.9% >= 0.2, 70.6% >= 0.5, MaskIn 63.4%)
2. `tail_base`: Mean Conf 0.6160 (Median 0.6599, 94.0% >= 0.2, 64.3% >= 0.5, MaskIn 88.1%)
3. `left_eye`: Mean Conf 0.6062 (Median 0.6755, 88.9% >= 0.2, 63.8% >= 0.5, MaskIn 74.0%)
4. `front_left_paw`: Mean Conf 0.5977 (Median 0.6172, 90.6% >= 0.2, 64.3% >= 0.5, MaskIn 64.7%)
5. `back_left_thai`: Mean Conf 0.5881 (Median 0.5608, 97.5% >= 0.2, 56.2% >= 0.5, MaskIn 82.1%)
6. `front_right_thai`: Mean Conf 0.5857 (Median 0.6286, 94.5% >= 0.2, 65.5% >= 0.5, MaskIn 85.5%)
7. `right_eye`: Mean Conf 0.5856 (Median 0.6274, 90.6% >= 0.2, 63.0% >= 0.5, MaskIn 80.9%)
8. `back_right_thai`: Mean Conf 0.5769 (Median 0.5980, 91.5% >= 0.2, 59.2% >= 0.5, MaskIn 72.8%)
9. `back_left_knee`: Mean Conf 0.5719 (Median 0.5880, 89.4% >= 0.2, 57.9% >= 0.5, MaskIn 68.9%)
10. `nose`: Mean Conf 0.5709 (Median 0.5664, 86.8% >= 0.2, 54.9% >= 0.5, MaskIn 74.0%)

#### Bottom 10 Lowest Confidence Keypoints
30. `left_antler_end`: Mean Conf 0.3399 (13.6% >= 0.5)
31. `right_antler_end`: Mean Conf 0.3383 (12.8% >= 0.5)
32. `throat_end`: Mean Conf 0.3358 (14.9% >= 0.5)
33. `right_earend`: Mean Conf 0.3246 (11.9% >= 0.5)
34. `throat_base`: Mean Conf 0.3216 (20.0% >= 0.5)
35. `body_middle_right`: Mean Conf 0.3146 (10.2% >= 0.5)
36. `body_middle_left`: Mean Conf 0.3086 (11.9% >= 0.5)
37. `neck_end`: Mean Conf 0.3022 (12.8% >= 0.5)
38. `left_earend`: Mean Conf 0.2932 (12.3% >= 0.5)
39. `tail_end`: Mean Conf 0.2049 (Median 0.1572, 5.96% >= 0.5, 38.7% >= 0.2)

---

## 4. Observed Technical Limitations & Patterns

### 4.1 High Detector Missingness on Lying Postures
- **Observation**: Out of 96 frames across 12 Lying sequences (across both CVB and Beef), the detector returned bounding boxes on only 26 frames (**27.08% return rate**), failing on **70 frames (72.92%)**. Exactly **0 out of 12 sequences** returned all 8 frames.
- **Characteristics**: When keypoints were returned on Lying cattle, the mean confidence was **0.3148**, the lowest among all five canonical classes.

### 4.2 Incomplete Sequence Coverage Across Consecutive Frames
- **Observation**: Overall, 45.60% of frames experienced detector misses. Across sequences, detection was frequently intermittent rather than continuous across all 8 frames.
- **Coverage**: Only 31.48% of sampled sequences (17/54) had complete 8-frame pose streams, while 68.52% had one or more frames with missing pose outputs.

### 4.3 Keypoint-Transition Displacement Exceedances
- **Observation**: Across consecutive valid frames, **880 individual keypoint transitions exceeded 0.25 normalized displacement (>56 pixels)**.
- **Sequence Stability Distribution**: Using the tracking criteria (STABLE: >=6 detected frames and 0 transitions >0.25 norm disp):
  - 1 sequence (1.85%) met the STABLE criteria.
  - 4 sequences (7.41%) showed moderate jumps (1–5 transitions >0.25).
  - 17 sequences (31.48%) showed >5 transition exceedances.
  - 32 sequences (59.26%) had partial detection (<6 frames returned).

---

## 5. Architectural Decision & Action Plan

### Core Question
> *Does the evidence support proceeding to a controlled Run 5 + Pose ablation on the CVB + Kaggle Beef Behavior representation?*

### Definitive Scientific Decision: **RUN 5 + POSE IS DEFERRED / NOT PRIORITIZED UNDER THE DEADLINE.**

### Defensible Thesis Rationale
1. **High Pose Missingness**: With an overall detector failure rate of 45.60% and 72.92% on Lying cattle, integrating raw pose features directly into the TCN would require handling significant missing data across sequence frames.
2. **Weak Temporal Consistency**: With 880 individual keypoint transitions exceeding the 0.25 displacement threshold and only 1.85% of sequences meeting stable tracking criteria, the current frozen model output exhibits high frame-to-frame variability.
3. **Class-Disproportionate Missingness**: Lying cattle exhibit a 27.08% return rate compared to 55–76% for Standing, Feeding, and Drinking, resulting in uneven modality availability across behavior classes.
4. **Preserving Deadline Priority**: Aligns with the active Phase 3 Deadline Priority Plan (`phase3_deadline_execution_2026-09-26.md`), which focuses on executing the core defensible runs (Runs 1–6 and MTL Runs 7–8) while keeping auxiliary ablations deferred.

> **Important Scientific Scope**: This audit evaluates the feasibility and quality of frozen SuperAnimal-Quadruped keypoint extraction on the CVB+Beef dataset representation. It does **NOT** prove that pose would reduce downstream Behavior classification accuracy, because no Run 5 + Pose model was trained.

---

## 6. Artifact Registry
| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `scripts/audit_cvb_beef_pose_feasibility.py` | Python | Standalone CVB + Beef Behavior SuperAnimal pose feasibility audit engine |
| `scripts/modal_audit_cvb_beef_pose.py` | Python | Modal cloud runner targeting `hasinishrak2015` on Tesla T4 |
| `artifacts/behavior_pose_audit/behavior_pose_frame_results.csv` | CSV (432 rows) | Frame-level audit results: status, confidences, 39 keypoints, mask containment sanity |
| `artifacts/behavior_pose_audit/behavior_pose_sequence_summary.csv` | CSV (54 rows) | Sequence-level summary: return rates, temporal displacements, jumps, stability flags |
| `artifacts/behavior_pose_audit/behavior_pose_aggregate_metrics.json` | JSON | Full aggregate metrics, breakdowns by source, class, cell, and keypoint rankings |
| `artifacts/behavior_pose_audit/superanimal_39_keypoints_schema.json` | JSON | Official 39-keypoint schema reference, bodyparts, and 37 skeleton connections |
| `artifacts/behavior_pose_audit/behavior_pose_all_classes_contact_sheet.jpg` | Image (2064x2408) | 9-row visual contact sheet displaying all 9 cells across all T=8 chronological frames |
| `docs/audits/assets/behavior_pose_audit/behavior_pose_all_classes_contact_sheet.jpg` | Image (2064x2408) | Tracked visual contact sheet asset for thesis documentation |

---

## 7. Next Steps
1. Maintain Run 5's validated 4-channel ResNet-18 + TCN perception model as the canonical single-task Behavior benchmark.
2. Keep the canonical Phase 3 roadmap preserved without modification.
3. User manually launches full 30-epoch training for Run 7 (E1 Hard-Shared MTL Control) on `hasinishrak2015`.
