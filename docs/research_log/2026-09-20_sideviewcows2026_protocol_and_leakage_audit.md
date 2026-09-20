# SideViewCows2026 Canonical Re-ID Protocols & Leakage Audit

**Date**: 2026-09-20  
**Status**: COMPLETE / VERIFIED  
**Auditors**: Hasin Ishrak & Antigravity  
**Dataset**: SideViewCows2026 (Zenodo Record: [10.5281/zenodo.21605650](https://doi.org/10.5281/zenodo.21605650))  
**Scientific Role**: **PRIMARY Re-ID Benchmark** (Adopted under approved MultiCamCows2024 contingency)

---

## 1. Executive Summary
Following the formal approval of the MultiCamCows2024 contingency proposal, deterministic evaluation protocols and comprehensive leakage audits were implemented and verified for SideViewCows2026 (80,260 images, 80,260 binary segmentation masks, 110 biological dairy cows). Using `scripts/build_sideview_reid_protocols.py`, discrete video recording passages were recovered across subsets using temporal delta clustering (`dt <= 60s`), yielding 3,604 discrete recording sessions. Four canonical evaluation protocols were built and verified: Protocol A (Cross-Setting Domain Shift), Protocol B (Longitudinal / Cross-Temporal), Protocol C (Open-Set / Identity-Disjoint with 77 Train / 11 Val / 22 Test cows), and Protocol D (Closed-Set Identification with sequence-safe recording partition). All 80,260 images exhibit 100% unique SHA-256 checksums (zero exact duplicates), 100% 1-to-1 image-to-mask correspondence, 0 identity overlap in open-set, and zero adjacent-frame leakage in closed-set.

---

## 2. Context & Motivation
1. **MultiCam Contingency Execution**: MultiCamCows2024 was excluded from active execution due to persistent upstream connection resets from `data.bris.ac.uk`. SideViewCows2026 was promoted to primary Re-ID benchmark to preserve the core Phase 3 research hypothesis: evaluating whether cattle-specific segmentation masks suppress background shortcut learning in metric Re-ID.
2. **Author Warning on Sequence Correlation**: The dataset author (Möller et al., 2026) explicitly warned that frames from the same recording are strongly correlated and random splitting leads to inflated/optimistic performance.
3. **Leakage Prevention**: To guarantee rigorous, publication-grade benchmarks, every protocol must enforce physical recording session separation, preventing adjacent frames from crossing split boundaries.

---

## 3. Dataset Characteristics & Session Recovery
- **Physical Verification**: 80,260 RGB JPEG images and 80,260 PNG binary masks verified locally in `datasets/id/external/sideviewcows2026/`.
- **Subsets**:
  - `parlor`: 54,393 images across all 110 cows (fixed camera at milking parlor entrance, days 0.0 to 156.0).
  - `barn`: 25,260 images across 69 cows (handheld video in barn, days 355.4 to 592.4; recorded >200 days after parlor).
  - `snapshots`: 607 images across 63 cows (unconstrained photos indoor/outdoor/lying down, days 355.3 to 634.4).
- **Session Recovery**:
  - Contiguous frames with `dt <= 60s` were clustered into discrete recording sessions (`recording_id`).
  - Total recovered sessions: **3,604** (parlor: 3,356 milking passages, barn: 71 video recordings, snapshots: 177 burst sessions).
  - Every cow in parlor has between 9 and 103 separate milking passages (mean 30.5 passages/cow), providing natural boundaries for sequence-safe partitioning.

---

## 4. Canonical Protocols Specifications

### Protocol A: Cross-Setting Domain Shift (`protocol_cross_setting.csv`)
- **Objective**: Evaluates representation robustness and retrieval under camera and domain shift.
- **Gallery**: Fixed-camera `parlor` entrance frames for the 69 multi-setting cows (**36,811 images**).
- **Query 1**: Handheld video in `barn` (**25,260 images** across 69 cows).
- **Query 2**: Unconstrained `snapshots` (**607 images** across 63 cows).
- **Train Representation**: Parlor frames from the 41 `parlor_only` cows (**17,582 images**) reserved for in-domain training with zero test identity contamination.

### Protocol B: Longitudinal / Cross-Temporal (`protocol_longitudinal.csv`)
- **Objective**: Evaluates coat pattern persistence and identification stability over time.
- **Gallery / Early Train**: First 60% of chronological parlor recording sessions per cow (**35,433 images**).
- **Query Late (In-Domain)**: Subsequent 40% of parlor recording sessions per cow (**18,960 images**). Strictly positive time delta (`min_query_time > max_gallery_time`).
- **Query Long-Range (Cross-Domain)**: Barn video (**25,260 images**) and Snapshots (**607 images**) recorded >200 days later.

### Protocol C: Open-Set / Identity-Disjoint (`protocol_open_set.csv`)
- **Objective**: Evaluates open-set feature generalization to unseen individual cattle.
- **Stratification**: Balanced across cow subset types (`multi`: 63, `parlor_barn`: 6, `parlor_only`: 41).
- **Split Distribution**:
  - `train`: **77 cows** (57,605 images) [44 multi, 4 parlor_barn, 29 parlor_only]
  - `val`: **11 cows** (7,225 images) [6 multi, 1 parlor_barn, 4 parlor_only]
  - `test`: **22 cows** (15,430 images) [13 multi, 1 parlor_barn, 8 parlor_only]
- **Disjointness**: 100% disjoint cow identities (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).

### Protocol D: Closed-Set Identification (`protocol_closed_set.csv`)
- **Objective**: 110-class metric identification under sequence-safe recording protection.
- **Method**: Chronological recording session partition per cow (70% train, 15% val, 15% test).
- **Split Distribution**:
  - `train`: Early parlor sessions (**40,745 images**)
  - `val`: Intermediate parlor sessions (**7,373 images**)
  - `test_parlor`: Late parlor sessions (**6,275 images**)
  - `test_barn`: Barn handheld video (**25,260 images**)
  - `test_snapshots`: Snapshots (**607 images**)
- **Sequence Protection**: Zero adjacent-frame leakage across video bursts.

---

## 5. Leakage Audit Results
1. **Exact Byte Duplicates**: 0 duplicate SHA-256 hashes across all 80,260 images.
2. **Identity Disjointness**: Verified 0 cow overlap across Open-Set splits (77/11/22 cows).
3. **Temporal Monotonicity**: Verified positive elapsed time delta between gallery and query in Protocol B and Protocol D.
4. **Perceptual Near-Duplicate Audit**: Sampled 10,094 session anchor frames and computed 64-bit dHash. Minimum cross-partition Hamming distance is 7 bits (well above the near-duplicate threshold of <= 2 bits and <= 6 bits).
5. **Mask Integrity**: 100% of images have an identically named mask with matching dimensions.

---

## 6. Artifacts & Deliverables
- Protocol Builder Script: [`scripts/build_sideview_reid_protocols.py`](file:///d:/cattle-health-monitoring-multi-task-model/scripts/build_sideview_reid_protocols.py)
- Canonical Manifest: [`datasets/id/sideviewcows2026/manifest.csv`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/id/sideviewcows2026/manifest.csv)
- Protocol A CSV: [`datasets/id/sideviewcows2026/protocol_cross_setting.csv`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/id/sideviewcows2026/protocol_cross_setting.csv)
- Protocol B CSV: [`datasets/id/sideviewcows2026/protocol_longitudinal.csv`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/id/sideviewcows2026/protocol_longitudinal.csv)
- Protocol C CSV: [`datasets/id/sideviewcows2026/protocol_open_set.csv`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/id/sideviewcows2026/protocol_open_set.csv)
- Protocol D CSV: [`datasets/id/sideviewcows2026/protocol_closed_set.csv`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/id/sideviewcows2026/protocol_closed_set.csv)
- Leakage Audit Scorecard: [`datasets/id/sideviewcows2026/leakage_audit.csv`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/id/sideviewcows2026/leakage_audit.csv)
- Detailed Split Report: [`datasets/id/sideviewcows2026/split_report.md`](file:///d:/cattle-health-monitoring-multi-task-model/datasets/id/sideviewcows2026/split_report.md)

---

## 7. Gate 1 Resolution
With the completion and verification of SideViewCows2026 protocols:
- [x] ScienceDB burst-group-disjoint split verified (`datasets/bcs/sciencedb/`)
- [x] MmCows grouped evaluation defined (`datasets/behavior/mmcows/folds/`)
- [x] SideViewCows2026 primary Re-ID protocols generated and verified (`datasets/id/sideviewcows2026/`)
- [x] Automated duplicate / near-duplicate audits passed across all primary task datasets
- **GATE 1 STATUS: CLEARED / READY FOR STEP 2 (CATTLE-PERCEPTION AUDIT)**
