# CVB + Kaggle Beef Behavior SuperAnimal-Quadruped Pose Feasibility Audit

## 1. Executive Summary
To rigorously determine whether pretrained cattle/quadruped pose priors can technically enhance the primary Phase 3 Run 5 Behavior stack (CVB + Kaggle Beef, T=8 sequences, cattle-centered RGB crops + binary masks), we conducted a forensic feasibility audit of frozen DeepLabCut SuperAnimal-Quadruped ResNet-50 on Modal profile `hasinishrak2015` (NVIDIA Tesla T4). Using a deterministic, stratified representative sample (seed `2026`) covering all 9 available source x class cells (54 sequences total x 8 frames = 432 frames from authentic `retained_train.csv` and `retained_val.csv`), the audit revealed a frame-level pose return rate of only **54.40%** (235/432 frames) with a detector failure rate of **45.60%** (197/432 frames). Only **31.48%** of sequences (17/54) returned pose on all 8 frames. Most critically, the quadruped detector experienced a catastrophic collapse on **Lying** cattle: only **27.08%** frame return rate, a **72.92%** detector failure rate, **0.0%** sequence-level complete return (0/12 sequences), and a depressed mean keypoint confidence of **0.3148**. Temporal consistency analysis detected **880 skeleton jumps** (>0.25 normalized displacement), with only **1 of 54 sequences (1.9%)** exhibiting stable tracking across all 8 frames. Predicted-mask geometric containment sanity averaged **73.77%** overall (92.57% on CVB vs 54.16% on Beef). Based on this empirical evidence, adding frozen SuperAnimal pose to Run 5 is **DEFENSIVELY REJECTED**: injecting 45-73% missing pose signals and violent temporal jitter into the 1D TCN would severely destabilize temporal behavior recognition.

---

## 2. Context & Scientific Motivation
In Phase 3 Run 5, cattle behavior recognition is powered by a 4-channel ResNet-18 spatial trunk coupled to a 1D Temporal Convolutional Network (TCN) processing T=8 chronological frames per sequence. Run 5 achieved:
- Balanced Accuracy: 74.43% (+3.13% over Run 2 baseline)
- Test Loss: 0.4430 (-19.5% reduction)
- Walking minority class F1: 0.2456 (+13.0% relative gain)

While 2D keypoint priors theoretically offer posture cues that disentangle spatial limb arrangements from visual texture, prior research logs noted potential unreliability on recumbent postures. The user explicitly directed an empirical audit on the **current authentic CVB + Kaggle Beef representation** on Modal profile `hasinishrak2015` before attempting any Behavior+Pose classifier training. Old MmCows pose results were explicitly excluded as evidence for CVB+Beef.

### Audit Design & Boundary Conditions
1. **Target Profile**: `hasinishrak2015`, volume `mtl-data` mounted at `/mtl-data`.
2. **Data Scope**: Staged authentic Run 5 sequences (`/mtl-data/behavior/{sample_id}/frame_{t:02d}.jpg`, `mask_{t:02d}.png`). Train and validation splits only (`retained_train.csv`, `retained_val.csv`). Held-out test set strictly untouched.
3. **Stratified Sampling**: Seed `2026`. Exactly 6 sequences (4 train, 2 val) sampled per available source x class cell. Walking exists only in CVB (no synthetic Beef Walking cell was fabricated). Total sample: 9 cells x 6 sequences = 54 sequences (432 frames).
4. **Frozen Model**: Official DeepLabCut SuperAnimal-Quadruped ResNet-50 (`fasterrcnn_resnet50_fpn_v2` detector) with official 39-keypoint schema. Zero fine-tuning.
5. **Metric Boundaries**: Evaluated geometric containment against predicted Run 5 masks strictly labeled as **predicted-mask geometric containment sanity check** (NOT pose accuracy; no keypoint annotations exist). Frame-to-frame displacement evaluated strictly as **temporal consistency sanity indicator** (NOT ground-truth motion tracking).

---

## 3. Quantitative Audit Findings

### 3.1 Overall Performance Summary
| Metric | Value | Detail / Interpretation |
| :--- | :--- | :--- |
| **Total Sequences Evaluated** | **54** | Deterministically stratified (seed 2026, 6 per cell) |
| **Total Frames Evaluated** | **432** | T=8 frames per sequence |
| **Frame-Level Pose Return Rate** | **54.40%** | 235 / 432 frames returned valid keypoint sets |
| **Frame-Level Detector Failure Rate** | **45.60%** | 197 / 432 frames dropped by Faster R-CNN detector |
| **Sequence-Level All-8-Frames Rate** | **31.48%** | Only 17 of 54 sequences had pose on all 8 frames |
| **Sequence-Level >=1-Frame Rate** | **87.04%** | 47 of 54 sequences had at least 1 detected frame |
| **Mean Keypoint Confidence** | **0.4658** | Median: 0.4381 (low overall confidence) |
| **Predicted-Mask Containment Sanity (All)** | **73.77%** | Percentage of predicted keypoints landing inside Run-5 mask |
| **Predicted-Mask Containment Sanity (Conf>=0.2)** | **73.88%** | Percentage of confident keypoints landing inside Run-5 mask |
| **Mean Temporal Displacement (Norm)** | **0.1116** | Normalized to 224px crop (~25.0 pixels average jump) |
| **Total Skeleton Jumps (>0.25 Norm)** | **880** | Severe frame-to-frame anatomical relocations |
| **Sequences Flagged STABLE** | **1 / 54 (1.85%)** | Exactly one sequence exhibited stable temporal tracking |
| **Sequences Flagged PARTIAL_DETECTION_FAILURE** | **32 / 54 (59.26%)** | Returned < 6 detected frames out of 8 |
| **Sequences Flagged SEVERE_INSTABILITY** | **17 / 54 (31.48%)** | Exhibited > 5 severe skeleton jumps |
| **Sequences Flagged MODERATE_JUMPS** | **4 / 54 (7.41%)** | Exhibited 1–5 skeleton jumps |

---

### 3.2 Results by Source: CVB vs. Kaggle Beef
| Metric | CVB (Indoor CCTV Barn) | Kaggle Beef (Pasture / Feedlot) | Delta / Disparity |
| :--- | :--- | :--- | :--- |
| **Total Sequences / Frames** | 30 seqs / 240 frames | 24 seqs / 192 frames | 5 classes vs 4 classes |
| **Pose Return Rate** | **50.00%** (120/240) | **59.90%** (115/192) | +9.90% higher in Beef |
| **Detector Failure Rate** | **50.00%** (120/240) | **40.10%** (77/192) | Higher failure in CVB |
| **All-8-Frames Return Rate** | **33.33%** (10/30) | **29.17%** (7/24) | Both severely depressed |
| **Mean Keypoint Confidence** | **0.4673** | **0.4643** | Virtually identical |
| **Predicted-Mask Containment (All)** | **92.57%** | **54.16%** | **+38.41% higher in CVB** |
| **Predicted-Mask Containment (Conf>=0.2)** | **92.84%** | **54.10%** | **+38.74% higher in CVB** |
| **Mean Temporal Displacement (Norm)** | **0.0786** | **0.1528** | Beef displacement is ~2x higher |
| **Skeleton Jumps (>0.25 Norm)** | **364** | **516** | 1.42x more jumps in Beef |

**Source Disparity Insight**:
- On CVB, when the detector succeeds, keypoints tightly overlap the GT-prompted SAM mask (92.57% containment). However, the detector fails on exactly half of the frames (50.0%) due to barn lighting, pen bars, and camera angle.
- On Kaggle Beef, detection return is slightly higher (59.9%), but mask containment collapses to 54.16%, and temporal displacement doubles (0.1528 vs 0.0786), showing severe spatial jitter and points projecting far outside the cow contour.

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
From the official 39-keypoint SuperAnimal-Quadruped schema, individual keypoints exhibited stark divides in reliability:

#### Top 10 Most Reliable Keypoints
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

#### Bottom 10 Noisiest / Weakest Keypoints
30. `left_antler_end`: Mean Conf 0.3399 (13.6% >= 0.5)
31. `right_antler_end`: Mean Conf 0.3383 (12.8% >= 0.5)
32. `throat_end`: Mean Conf 0.3358 (14.9% >= 0.5)
33. `right_earend`: Mean Conf 0.3246 (11.9% >= 0.5)
34. `throat_base`: Mean Conf 0.3216 (20.0% >= 0.5)
35. `body_middle_right`: Mean Conf 0.3146 (10.2% >= 0.5)
36. `body_middle_left`: Mean Conf 0.3086 (11.9% >= 0.5)
37. `neck_end`: Mean Conf 0.3022 (12.8% >= 0.5)
38. `left_earend`: Mean Conf 0.2932 (12.3% >= 0.5)
39. `tail_end`: Mean Conf 0.2049 (Median 0.1572, only 5.96% >= 0.5, 38.7% >= 0.2)

---

## 4. Forensic Failure Mode Analysis

### Failure Mode 1: Catastrophic Detector Collapse on Recumbent Postures (Lying Cattle)
- **Manifestation**: Out of 96 frames across 12 Lying sequences (both CVB and Beef), the detector failed on **70 frames (72.92%)**. Exactly **0 out of 12 sequences** returned all 8 frames.
- **Root Cause**: Pretrained SuperAnimal-Quadruped's Faster R-CNN detector was trained overwhelmingly on upright quadrupeds (dogs, standing horses, deer, walking cattle). When a cow lies down with folded limbs, its aspect ratio flattens horizontally into an uncharacteristic blob, causing bounding-box proposal confidence to plunge below detection threshold.
- **Impact on Run 5**: Lying is one of the most critical health indicators in dairy monitoring (ruminating vs rest). In a temporal model, feeding a sequence with 73% missing frames would force zero-padding or imputation, destroying the learned temporal representation for this class.

### Failure Mode 2: Intermittent Frame Dropping & Temporal Flickering
- **Manifestation**: Overall, 45.60% of frames suffered detector failure. Instead of failing an entire sequence cleanly, the detector routinely succeeded on frames 0, 2, and 5, while dropping frames 1, 3, 4, 6, and 7.
- **Consequence**: Only 31.48% of sequences have continuous 8-frame pose streams. A 1D TCN operating on temporal features requires fixed-step temporal consistency; intermittent frame loss would inject artificial discontinuous transients into the convolutions.

### Failure Mode 3: Extreme Frame-to-Frame Skeleton Instability (880 Jumps)
- **Manifestation**: In consecutive frames where both frames were detected, keypoints underwent massive instantaneous displacements. Across the 54 sequences, **880 keypoint transitions exceeded 0.25 normalized crop size (>56 pixels)**.
- **Root Cause**: Left/right limb ambiguity during grazing/feeding. In feeding cows, the head down posture causes ears, eyes, and jaws to swap or collapse onto the forelimbs. Furthermore, pasture grass and barn stanchions intermittently occlude lower hooves, causing the model to hallucinate limb positions across the body.
- **Stability Rating**: Exactly **1 sequence out of 54 (1.85%)** was rated STABLE. 59.26% suffered partial detection dropouts, and 31.48% exhibited severe instability.

---

## 5. Architectural Decision & Action Plan

### Core Question
> *Does the evidence support proceeding to a controlled Run 5 + Pose ablation on the CVB + Kaggle Beef Behavior representation?*

### Definitive Scientific Decision: **NO. DO NOT PROCEED TO RUN 5 + POSE.**

### Defensible Thesis Rationale
1. **Severe Missing Data**: A 45.6% overall detector failure rate and a 72.9% failure rate on Lying cattle renders the pose vector unusable as an automated auxiliary modality. Zero-padding 45% of feature frames would actively degrade the 1D TCN's existing representations.
2. **Severe Temporal Instability**: With 880 skeleton jumps and only 1.85% stable sequences, frozen SuperAnimal keypoints do not provide coherent temporal dynamics. Instead of helping the TCN discern Walking vs Feeding or Standing vs Lying, noisy keypoints would add high-variance noise.
3. **Class-Asymmetric Bias**: Lying would be penalized with 73% missing pose signals, while Drinking/Feeding retain 60–76% pose returns, introducing an artificial, uncalibrated modality drop that penalizes recumbent posture recognition.
4. **Roadmap Alignment**: The Phase 3 Deadline Priority Plan (`phase3_deadline_execution_2026-09-26.md`) explicitly excluded SuperAnimal pose from Run 5 due to anticipated unreliability on recumbent cattle. This audit provides bulletproof, empirical proof on the authentic Run 5 dataset confirming that theoretical intuition.

---

## 6. Artifact Registry
| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `scripts/audit_cvb_beef_pose_feasibility.py` | Python | Standalone CVB + Beef Behavior SuperAnimal pose feasibility audit engine |
| `scripts/modal_audit_cvb_beef_pose.py` | Python | Modal cloud runner targeting `hasinishrak2015` on Tesla T4 |
| `artifacts/behavior_pose_audit/behavior_pose_frame_results.csv` | CSV (432 rows) | Frame-level audit results: status, confidences, 39 keypoints, mask containment |
| `artifacts/behavior_pose_audit/behavior_pose_sequence_summary.csv` | CSV (54 rows) | Sequence-level summary: return rates, temporal displacements, jumps, stability flags |
| `artifacts/behavior_pose_audit/behavior_pose_aggregate_metrics.json` | JSON | Full aggregate metrics, breakdowns by source, class, cell, and keypoint rankings |
| `artifacts/behavior_pose_audit/superanimal_39_keypoints_schema.json` | JSON | Official 39-keypoint schema reference, bodyparts, and 37 skeleton connections |
| `artifacts/behavior_pose_audit/behavior_pose_all_classes_contact_sheet.jpg` | Image (2064x2408) | 9-row visual contact sheet displaying all 9 cells across all T=8 chronological frames |

---

## 7. Next Steps
1. Update `docs/research_log/README.md` index table with this audit entry.
2. Synchronize `memory/state.md` with verified empirical findings.
3. Maintain Run 5's clean 4-channel ResNet-18 + TCN perception representation without pose pollution.
4. Prepare for user manual launch of Run 7 (E1 Hard-Shared MTL full training) on `hasinishrak2015`.
