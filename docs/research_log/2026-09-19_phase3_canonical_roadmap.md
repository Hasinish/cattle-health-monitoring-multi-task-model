# Research Log: Adoption of the Phase 3 Canonical Roadmap

**Date**: 2026-09-19  
**Author**: Hasin Ishrak  
**Supervision**: Dr. Md. Khalilur Rahman  
**Project**: Cattle Health Monitoring Multi-Task Deep Learning Model (BRAC University)  
**Status**: CANONICAL / LOCKED FOR EXECUTION  

---

## 1. Executive Summary
Formally adopted and locked the comprehensive **Phase 3 Canonical Roadmap** (`phase3_canonical_roadmap.md` and `docs/phase3_canonical_roadmap.md`) as the single source of truth for all future research and engineering turns. The thesis pivots from generic RGB multi-task parameter sharing to investigating: *"Which cattle-specific visual priors (localization, soft segmentation masks, keypoints/pose, and camera viewpoint) are useful for which downstream task, and what information should each task preserve or suppress?"* Downstream tasks are locked to Body Condition Scoring (BCS), Behavior Recognition, and Cow Identification / Re-Identification. Crucially, **MultiCamCows2024** (90 cows, 101,329 images, 3 cameras, 7 days, sequence-safe tracklets) replaces OpenCows2020 as the primary Re-ID benchmark, while Ruchay et al. (RGB-D BCS 2026), CBVD-5, and SideViewCows2026 are designated as frozen external validation benchmarks.

---

## 2. Context & Motivation
In Phase 2, feeding raw RGB frames directly to a hard-shared CNN backbone caused negative transfer (BCS and Behavior degraded compared to single-task baselines) and allowed models to exploit background/environmental shortcuts (pen floors, fences, lighting, camera geometry). Furthermore:
- CattleLameness was compromised by data leakage and synthetic Blender footage, leading to lameness removal from primary Phase 3 MTL.
- OpenCows2020 suffered from sequential video-frame shuffling leakage between train and val.
- The defense committee criticized the simple merging of existing task models without an original, cattle-centered research thesis.

The Phase 3 Canonical Roadmap addresses these flaws through a rigorous 13-step progression grounded in explicit cattle priors and leakage-resistant protocols.

---

## 3. Canonical Task & Dataset Stack

### 3.1 Body Condition Scoring (BCS)
- **Primary (In-Domain)**: **ScienceDB** (53,566 RGB images, 10,898 project-parsed cows, cow-disjoint split).
- **Primary External Validation**: **Ruchay et al. RGB-D BCS (2026)** (multi-breed, wide BCS range, different camera geometry, optional RGB vs RGB-D test).
- **Secondary External**: **Dryad BCS** (DGE images across discrete classes '2'-'6').

### 3.2 Behavior Recognition
- **Primary (In-Domain)**: **MmCows** (213,686 bounding-box crops across 7 active classes, 16 cows, multi-camera CCTV).
- **Primary External Validation**: **CBVD-5** (large-herd external behavior benchmark).
- **Optional External**: CVB, XGain, Simmental 2026 (for compatible label intersections only).

### 3.3 Individual Cow Identification / Re-Identification (Re-ID)
- **Primary (New Benchmark)**: **MultiCamCows2024** (90 cows, 101,329 images, 3 cameras, 7 days, tracklets; enables tracklet-disjoint, cross-day, cross-camera, and open-set evaluation). Replaces OpenCows2020 as primary.
- **Primary External Validation**: **SideViewCows2026** (side-view re-identification with masks).
- **Long-term / Scale Stress**: BECA-L (appearance change over time), BECA-D (large population).
- **Legacy Baseline**: **OpenCows2020** (retained for backward comparability only; random train/val split abandoned).

---

## 4. 13-Step Canonical Execution Pipeline
```
STEP 1: Data registry (dataset_registry.csv) + clean splits
   ↓
STEP 2: Cattle-perception feasibility audit (small sample: bbox, mask, pose, viewpoint)
   ↓
STEP 3: Cache upstream cattle information (bbox, soft_mask, pose_coords, viewpoint)
   ↓
STEP 4: Clean RGB single-task baselines (ScienceDB, MmCows, MultiCamCows)
   ↓
STEP 5: Localization / segmentation ablation (A0 Raw RGB, A1 Crop, A2 Mask, A3 FG-only, A4 Context stream)
   ↓
STEP 6: Anatomy / pose ablation (B0 Visual, B1 Visual+Pose, B2 Shuffled Pose control, B3 Confidence-aware)
   ↓
STEP 7: Viewpoint ablation (C0 Aug only, C1 View token, C2 Shuffled view control, C3 FiLM/gating)
   ↓
STEP 8: Temporal Behavior ladder (D0 Frame, D1 Avg pool, D2 TCN, D3 GRU/LSTM, D4 Pose seq, D5 Fusion)
   ↓
STEP 9: Build consolidated task-conditioned cattle-centered P3 architecture (only components surviving ablation)
   ↓
STEP 10: Cross-domain / robustness evaluation (Ruchay 2026, CBVD-5, SideViewCows2026)
   ↓
STEP 11: Revisit sharing / MTL (E0 Single, E1 Hard, E2 Partial, E3 Adapters, E4 PCGrad, E5 GradNorm)
   ↓
STEP 12: Optional cattle-specific pretraining (background swapping/invariance - stretch goal)
   ↓
STEP 13: Final repeated runs (3 seeds: 42, 123, 2026) + thesis tables
```

---

## 5. Hard Scientific Rules & Go/No-Go Gates
- **Never**: Tune on test sets; randomly split adjacent video frames; call camera ID viewpoint; call PCGrad/GradNorm primary novelty; claim zero leakage without evidence; claim masks/pose help before ablations.
- **Gate 1 (Data Ready)**: Proceed to perception/baselines only when ScienceDB, MmCows, and MultiCamCows are clean and leak-checked.
- **Gate 2 (Perception Feasible)**: Proceed to pose/view experiments only if SuperAnimal/CattleEyeView pass visual audit on 100-300 samples. If pose fails, do NOT force it into the thesis.
- **Gate 3 (Component Value)**: Components enter the final model ONLY if gains survive control baselines (e.g. shuffled pose/view controls).
- **Gate 5 (MTL Deferred)**: Do not begin MTL until clean single-task final models exist and negative transfer can be measured fairly.

---

## 6. Immediate Action Items (STEP 1)
1. Download and index MultiCamCows2024.
2. Build canonical dataset registry (`datasets/dataset_registry.csv`).
3. Validate ScienceDB identity parser and verify existing cow-disjoint split.
4. Rebuild MmCows grouped evaluation protocol (Group K-Fold / time-block protection).
5. Create MultiCamCows protocols (tracklet-disjoint, cross-day, cross-camera, open-set).
6. Download/index Ruchay 2026, SideViewCows2026, and BECA-D/L.
7. Verify CBVD-5 raw data and identity metadata.
8. Run automatic duplicate / near-duplicate audits.
