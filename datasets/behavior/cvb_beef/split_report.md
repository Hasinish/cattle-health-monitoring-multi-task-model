# Canonical CVB + Kaggle Beef Behavior Split Report

**Date:** 2026-09-23  
**Protocol Version:** Phase 3 Canonical Primary Behavior Stack (Revision 1.0)  
**Evaluation Protocol:** Source-Video / Session-Disjoint Grouped Partitioning  
**Audit & Verification Status:** VERIFIED LEAK-FREE & PROVENANCE-ISOLATED  
**Split Seed:** `2026`  

---

## 1. Executive Summary & Forensic Provenance

This report documents the canonical primary Behavior Recognition training partition for the Phase 3 Multi-Task Cattle Model. The partition unifies two continuous dense-video benchmarks:
1. **CVB (Cattle Visual Behaviors)**: Open-pasture 30 FPS Full HD recordings of Angus cattle from CSIRO Armidale station.
2. **Kaggle Beef Cattle Behavior**: Feedlot/barn 25 FPS 224x224 CCTV recordings of captive beef cattle.

### Critical Scientific Governance Declarations:
- **NOT Cow-Disjoint**: Neither CVB nor Kaggle Beef provides reliable biological cow identities (CVB has 0 cow IDs; Kaggle Beef has fragmented ByteTrack IDs on 6 captive cows). The partition is scientifically designated as **`source-video / session-disjoint`**.
- **No Random Frame/Clip Splitting**: Every temporal segment and cut from a source surveillance recording is locked atomically into exactly one split.
- **Walking Confounding Explicitly Acknowledged**: Kaggle Beef contains **0 Walking clips**. In the combined primary stack, 100% of Walking samples originate from CVB. Cross-domain Walking generalization must be evaluated on the external cow-disjoint **MmCows** validation benchmark.
- **MmCows Protocol Frozen**: The existing cow-disjoint MmCows protocol (`datasets/behavior/mmcows/`) is completely preserved and serves as the primary external identity-aware stress test.

---

## 2. File Hashes & Checksums

| File | Relative Path | Samples | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Combined Manifest** | `datasets/behavior/cvb_beef/manifest.csv` | 5,274 | `19b82484d126507d085d66afc5a0f7b49a1a098f498d2c80052a13a5a279e919` |
| **Train Split** | `datasets/behavior/cvb_beef/train.csv` | 3,785 | `fa8126c987179152ac677f8bf4cf5ae608cca0245be4302f50a5485507444750` |
| **Validation Split** | `datasets/behavior/cvb_beef/val.csv` | 680 | `7cd1cdfaa4cc203cfedf8ee192ff8e5d045c41b8ec1fb13785a53032cd1fdf68` |
| **Test Split** | `datasets/behavior/cvb_beef/test.csv` | 809 | `57709e2aa2916684ea7ca6329b3e755d373ab773fbd91ed3037c3446ab0bda87` |
| **Label Mapping** | `datasets/behavior/cvb_beef/label_mapping.csv` | 17 | `08f1482f6ee1014885ee3dbb8bacc671d178d0570c56aa8726c008f5005a482f` |

---

## 3. Disjointness & Anti-Leakage Verification

| Verification Check | Partition Comparison | Overlap Count | Status | Evidence / Metric |
| :--- | :--- | :--- | :--- | :--- |
| **CVB Source Video Disjointness** | Train ∩ Val | 0 videos | **PASS** | 44 Train vs 10 Val |
| **CVB Source Video Disjointness** | Train ∩ Test | 0 videos | **PASS** | 44 Train vs 12 Test |
| **CVB Source Video Disjointness** | Val ∩ Test | 0 videos | **PASS** | 10 Val vs 12 Test |
| **Beef Session Disjointness** | Train ∩ Val | 0 sessions | **PASS** | 140 Train vs 29 Val |
| **Beef Session Disjointness** | Train ∩ Test | 0 sessions | **PASS** | 140 Train vs 32 Test |
| **Beef Session Disjointness** | Val ∩ Test | 0 sessions | **PASS** | 29 Val vs 32 Test |
| **Sample ID Uniqueness** | Across All Splits | 0 collisions | **PASS** | Exactly 5,274 unique keys |
| **Walking Confounding Check** | Beef Walking Count | 0 samples | **PASS** | Beef Walking = 0; CVB Walking = 171 |
| **Label Purity Check** | Excluded Label Count | 0 samples | **PASS** | Zero rumination/hidden/noise |

---

## 4. Split Statistics & Class Distributions

### 4.1 Combined Dataset Summary

| Partition | Total Samples | % Samples | CVB Samples | Beef Samples | Total Groups |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 3,785 | 71.8% | 1,747 | 2,038 | 184 (44 CVB + 140 Beef) |
| **Val** | 680 | 12.9% | 312 | 368 | 39 (10 CVB + 29 Beef) |
| **Test** | 809 | 15.3% | 422 | 387 | 44 (12 CVB + 32 Beef) |
| **Total** | **5,274** | **100.0%** | **2,481** | **2,793** | **267 (66 CVB + 201 Beef)** |

### 4.2 Combined 5-Class Distribution

| Canonical Class | Train Count (%) | Val Count (%) | Test Count (%) | Total Count (%) | Primary Origin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing** | 821 (21.7%) | 103 (15.1%) | 113 (14.0%) | 1,037 (19.7%) | Both (CVB: 399, Beef: 638) |
| **Lying** | 1,337 (35.3%) | 253 (37.2%) | 258 (31.9%) | 1,848 (35.0%) | Both (CVB: 486, Beef: 1,362) |
| **Feeding** | 1,239 (32.7%) | 253 (37.2%) | 349 (43.1%) | 1,841 (34.9%) | Both (CVB: 1,295, Beef: 546) |
| **Drinking** | 269 (7.1%) | 45 (6.6%) | 63 (7.8%) | 377 (7.1%) | Both (CVB: 130, Beef: 247) |
| **Walking** | 119 (3.1%) | 26 (3.8%) | 26 (3.2%) | 171 (3.2%) | **CVB ONLY (CVB: 171, Beef: 0)** |
| **Total** | **3,785** | **680** | **809** | **5,274** | |

### 4.3 CVB Sub-Dataset Breakdown

| Canonical Class | Train Count | Val Count | Test Count | CVB Total |
| :--- | :--- | :--- | :--- | :--- |
| **Standing** | 312 | 50 | 37 | 399 |
| **Lying** | 363 | 57 | 66 | 486 |
| **Feeding** | 851 | 169 | 275 | 1,295 |
| **Drinking** | 102 | 10 | 18 | 130 |
| **Walking** | 119 | 26 | 26 | 171 |
| **CVB Total** | **1,747** | **312** | **422** | **2,481** |

### 4.4 Kaggle Beef Cattle Sub-Dataset Breakdown

| Canonical Class | Train Count | Val Count | Test Count | Beef Total |
| :--- | :--- | :--- | :--- | :--- |
| **Standing** | 509 | 53 | 76 | 638 |
| **Lying** | 974 | 196 | 192 | 1,362 |
| **Feeding** | 388 | 84 | 74 | 546 |
| **Drinking** | 167 | 35 | 45 | 247 |
| **Walking** | **0** | **0** | **0** | **0 (Absent in Beef)** |
| **Beef Total** | **2,038** | **368** | **387** | **2,793** |

---

## 5. Verification Commands

To independently re-verify this protocol at any time, run:

```powershell
python scripts/build_cvb_beef_behavior_protocol.py --verify-only
```
