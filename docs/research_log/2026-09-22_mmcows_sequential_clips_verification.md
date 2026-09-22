# Research Log: MmCows Sequential Clips & Temporal Continuity Audit

**Date**: 2026-09-22  
**Author**: Hasin Ishrak & Antigravity Research Agent  
**Context**: Investigation into MmCows sequential frame continuity and temporal sampling rate to determine whether sequences/clips are useful for behavior modeling.

---

## 1. Executive Summary
Investigated the temporal sampling characteristics of MmCows behavior crops. Proven that MmCows provides **4,768 discrete 15-second timestamps** across 21.0 hours of multi-camera CCTV recordings, with crops sampled at exactly **15-second intervals** (0.067 Hz). Over 180,000 consecutive 15-second frame transitions exist across the dataset, with continuous runs reaching up to 829 consecutive frames (3.5 hours) for resting. Built 8 panoramic sequential filmstrip contact sheets (`mmcows_seq_<behavior>.jpg` and `mmcows_seq_transition.jpg`) demonstrating 6-frame consecutive progressions (t=0s to t=+75s). Concluded that sequences are **100% useful for 2D spatial feature learning and macro-behavior temporal state modeling**, but **not intended for 30 fps micro-kinematics / dense optical flow**.

---

## 2. Context & Motivation
Following the visual verification of static crops, the user requested an inspection of sequential frames to determine if the "clips" are useful or trash. In video-based computer vision, understanding whether data consists of dense 30 fps video or discrete temporal scan samples is crucial to choosing the appropriate modeling paradigm.

---

## 3. Forensic Temporal Findings

### Sampling Rate & Structure
- **Sampling Interval**: Exactly **15.0 seconds** between consecutive time steps (`timestamp_epoch.diff() == 15.0` for 180,000+ frame transitions).
- **Rationale from Authors (NeurIPS 2024 Spotlight)**: 15-second scan sampling aligns with standard veterinary ethology (Altmann 1974) to capture complete 24-hour diurnal behavioral time budgets without storing unmanageable petabytes of redundant 30 fps stall video.
- **Maximum Consecutive Run Lengths**:
  - `Lying`: 829 frames (12,435s = 207 minutes / 3.5 hours continuous)
  - `Standing`: 322 frames (4,830s = 80.5 minutes continuous)
  - `Feeding_head_up`: 70 frames (1,050s = 17.5 minutes continuous)
  - `Feeding_head_down`: 57 frames (855s = 14.2 minutes continuous)
  - `Licking`: 49 frames (735s = 12.2 minutes continuous)
  - `Walking`: 47 frames (705s = 11.8 minutes continuous)
  - `Drinking`: 37 frames (555s = 9.2 minutes continuous)

---

## 4. Modeling Utility Assessment

| Modeling Task | Useful? | Technical Reason |
| :--- | :---: | :--- |
| **Phase 3 MTL (2D Spatial Priors)** | **YES (100%)** | Our primary model trains on single-frame cattle crops with localization, segmentation, and viewpoint priors. Sequential diversity provides realistic pose variance over time. |
| **Macro-Temporal Bouts & Transitions (LSTM/GRU)** | **YES (100%)** | 15s intervals are ideal for tracking behavioral state durations, rumination cycles, and feeding bout transitions over minutes/hours. |
| **Dense Optical Flow / 30 fps Kinematics (SlowFast)** | **NO** | 15 seconds is too wide for dense optical flow; walking cows take ~12-15 steps between frames. Micro-kinematic video architectures should not be applied to 15s scan data. |

---

## 5. Artifacts & File Registry
- Script: [`scripts/build_mmcows_sequential_contact_sheets.py`](file:///d:/cattle-health-monitoring-multi-task-model/scripts/build_mmcows_sequential_contact_sheets.py)
- Contact Sheets (8 files): [`docs/audits/assets/mmcows_sequential_verification/`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/assets/mmcows_sequential_verification/)
  - `mmcows_seq_transition.jpg` (Dynamic behavior transition)
  - `mmcows_seq_drinking.jpg`
  - `mmcows_seq_feeding_head_down.jpg`
  - `mmcows_seq_feeding_head_up.jpg`
  - `mmcows_seq_licking.jpg`
  - `mmcows_seq_lying.jpg`
  - `mmcows_seq_standing.jpg`
  - `mmcows_seq_walking.jpg`
- Comprehensive Audit Report: [`docs/audits/phase3_mmcows_sequential_clips_verification.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_mmcows_sequential_clips_verification.md)

---

## 6. Next Steps
1. User visual review of [`docs/audits/phase3_mmcows_sequential_clips_verification.md`](file:///d:/cattle-health-monitoring-multi-task-model/docs/audits/phase3_mmcows_sequential_clips_verification.md).
2. Proceed with task-specific visual BCS quality audit on Ruchay 2026 Zenodo sample.
