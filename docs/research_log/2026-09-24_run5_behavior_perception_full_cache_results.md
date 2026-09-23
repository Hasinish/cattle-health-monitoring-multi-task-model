# Research Log: Phase 3 Run 5 Behavior Perception Full Cache Execution & Verification

**Date:** 2026-09-24  
**Author:** Hasin Ishrak  
**Target Modal Profile:** `tigerwood693`  
**GPU Tier:** NVIDIA L40S (24GB)  
**Volumes:** `cvb-data`, `beef-behavior-data`, `behavior-checkpoints`  
**App ID:** `ap-4zbuvcBQn60WhlOltpInYn`  
**Total Runtime:** 3,492.3s (58.2 mins; Train: 49.1m, Val: 9.0m)  
**Total Candidates:** 4,465 sequences (3,785 Train, 680 Val)  

---

## 1. Executive Summary

Executed the complete full production perception caching pass for Phase 3 Run 5 (Behavior Perception-Enhanced 4-Channel TCN) across all 4,465 Train and Val candidate sequences on Modal (`tigerwood693`, NVIDIA L40S).

The pipeline generated **34,168 authentic binary masks** with zero synthetic empty/all-ones fabrication. Exactly 194 occluded sequences (4.3%) were cleanly isolated and excluded due to severe stall stanchion occlusions (`sam_returned_empty_mask`). Exactly **4,271 sequences (3,641 Train, 630 Val)** were retained and permanently committed to `/cache/production` on `behavior-checkpoints`. Critical minority class `Walking` achieved 100% retention (119/119 in Train, 26/26 in Val). The canonical test set (`test.csv`, 809 samples) was strictly isolated and untouched.

---

## 2. Partition & Provenance Statistics

| Partition | Candidates Requested | Sequences Retained | Sequences Excluded | Exclusion Rate | Total Real Masks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 3,785 | **3,641** | 144 | 3.8% | 29,128 |
| **Val** | 680 | **630** | 50 | 7.3% | 5,040 |
| **Total** | **4,465** | **4,271** | **194** | **4.3%** | **34,168** |

### Frame Mask Breakdown
- **CVB Exact GT-BBox Prompt**: 16,464 frames
- **Kaggle Beef Detector A5 Prompt (Box + Center Point)**: 14,688 frames
- **Kaggle Beef Fallback (Center Point Prompt)**: 3,016 frames
- **Total Real Masks Generated**: **34,168**

### Class Distribution (Before vs After Exclusions)
- **Lying**: Train 1,337 -> 1,231; Val 253 -> 217
- **Feeding**: Train 1,239 -> 1,216; Val 253 -> 251
- **Standing**: Train 821 -> 816; Val 103 -> 94
- **Drinking**: Train 269 -> 259; Val 45 -> 42
- **Walking (CVB-Only)**: Train 119 -> **119 (100% preserved)**; Val 26 -> **26 (100% preserved)**

---

## 3. Storage & Integrity Audit
- **Master Manifest**: `/cache/production/perception_manifest.csv` (34,168 rows)
- **Split Manifests**: `perception_manifest_train.csv`, `perception_manifest_val.csv`
- **Retained IDs**: `retained_train.csv` (3,641 rows), `retained_val.csv` (630 rows)
- **Excluded IDs**: `failed_train.csv` (144 rows), `failed_val.csv` (50 rows)
- **Summary**: `perception_summary.json`
- **Strict Test Isolation**: `test.csv` (809 samples) was NEVER opened, parsed, or evaluated.

---

## 4. Status & Next Actions
- **Perception Cache Status**: **100% CERTIFIED & LOCKED**.
- **Immediate Next Action**: Launch full 30-epoch Run 5 TCN training (`train_full_run5_l40s_remote`).
